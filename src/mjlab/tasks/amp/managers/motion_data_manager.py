"""Motion data manager for loading and interpolating reference motion clips."""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field

import joblib
import torch

from mjlab.utils.lab_api.math import quat_apply_inverse


# ---------------------------------------------------------------------------
# Math helpers (not available in mjlab.utils.lab_api.math)
# ---------------------------------------------------------------------------


def vel_forward_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
  """Compute velocity via forward finite differences. Last frame copies prev."""
  vel = torch.zeros_like(x)
  vel[:-1] = (x[1:] - x[:-1]) / dt
  vel[-1] = vel[-2]
  return vel


def ang_vel_from_quat_diff(
  q: torch.Tensor, dt: float, in_frame: str = "world"
) -> torch.Tensor:
  """Compute angular velocity from quaternion sequence via finite diff.

  Args:
    q: (T, 4) quaternion sequence in (w, x, y, z).
    dt: Timestep between frames.
    in_frame: 'world' or 'body'.
  """
  T = q.shape[0]
  ang_vel = torch.zeros(T, 3, device=q.device, dtype=q.dtype)
  # q_diff = q_{t+1} * q_t^{-1}  (world frame)
  # For body frame: q_diff = q_t^{-1} * q_{t+1}
  q_next = q[1:]
  q_curr = q[:-1]

  # Quaternion conjugate for unit quats
  q_curr_inv = q_curr.clone()
  q_curr_inv[:, 1:] = -q_curr_inv[:, 1:]

  if in_frame == "world":
    # q_diff = q_next * q_curr_inv
    q_diff = _quat_mul(q_next, q_curr_inv)
  else:
    # q_diff = q_curr_inv * q_next
    q_diff = _quat_mul(q_curr_inv, q_next)

  # Ensure shortest path
  mask = q_diff[:, 0] < 0
  q_diff[mask] = -q_diff[mask]

  # axis-angle from quat: angle = 2 * acos(w), axis = xyz / sin(angle/2)
  half_angle = torch.acos(torch.clamp(q_diff[:, 0], -1.0, 1.0))
  sin_half = torch.sin(half_angle).unsqueeze(-1).clamp(min=1e-8)
  axis = q_diff[:, 1:] / sin_half
  angle = 2.0 * half_angle

  ang_vel[:-1] = axis * angle.unsqueeze(-1) / dt
  ang_vel[-1] = ang_vel[-2]
  return ang_vel


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Multiply two quaternions (w, x, y, z)."""
  w1, x1, y1, z1 = q1.unbind(-1)
  w2, x2, y2, z2 = q2.unbind(-1)
  return torch.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dim=-1,
  )


def quat_slerp_batched(
  q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor
) -> torch.Tensor:
  """Spherical linear interpolation with per-sample blend factor.

  Args:
    q0: (N, 4) start quaternions (w, x, y, z).
    q1: (N, 4) end quaternions (w, x, y, z).
    blend: (N,) interpolation factors in [0, 1].
  """
  dot = (q0 * q1).sum(dim=-1, keepdim=True)
  # Ensure shortest path
  q1 = torch.where(dot < 0, -q1, q1)
  dot = dot.abs().clamp(max=1.0)

  theta = torch.acos(dot)
  sin_theta = torch.sin(theta).clamp(min=1e-8)

  t = blend.unsqueeze(-1)
  s0 = torch.sin((1.0 - t) * theta) / sin_theta
  s1 = torch.sin(t * theta) / sin_theta

  # Fallback to linear for very small angles
  small = (dot > 0.9995).squeeze(-1)
  result = s0 * q0 + s1 * q1
  if small.any():
    lin = (1.0 - t) * q0 + t * q1
    lin = lin / lin.norm(dim=-1, keepdim=True)
    result[small] = lin[small]
  return result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class LoopMode(enum.IntEnum):
  CLAMP = 0
  WRAP = 1


@dataclass
class MotionDataTermCfg:
  """Configuration for a motion data term."""

  motion_data_dir: str = ""
  """Directory containing .pkl motion data files."""
  motion_data_weights: dict[str, float] = field(default_factory=dict)
  """Mapping from motion clip name (without .pkl) to sampling weight."""


# ---------------------------------------------------------------------------
# MotionDataTerm
# ---------------------------------------------------------------------------


class MotionDataTerm:
  """Loads and serves motion reference data for a single dataset."""

  def __init__(self, cfg: MotionDataTermCfg, num_envs: int, device: str) -> None:
    self.cfg = cfg
    self.num_envs = num_envs
    self.device = device

    if not os.path.isdir(cfg.motion_data_dir):
      raise FileNotFoundError(f"Motion data directory not found: {cfg.motion_data_dir}")
    self._load_motion_data()

  # -- loading --

  def _load_motion_data(self) -> None:
    available = [f for f in os.listdir(self.cfg.motion_data_dir) if f.endswith(".pkl")]
    if not available:
      raise ValueError(f"No .pkl files in {self.cfg.motion_data_dir}")

    durations: list[float] = []
    fps_list: list[float] = []
    dt_list: list[float] = []
    num_frames_list: list[int] = []
    loop_modes_list: list[int] = []
    weights_list: list[float] = []

    root_pos_w_parts: list[torch.Tensor] = []
    root_quat_parts: list[torch.Tensor] = []
    root_vel_w_parts: list[torch.Tensor] = []
    root_ang_vel_w_parts: list[torch.Tensor] = []
    dof_pos_parts: list[torch.Tensor] = []
    dof_vel_parts: list[torch.Tensor] = []
    key_body_pos_w_parts: list[torch.Tensor] = []

    for name, weight in self.cfg.motion_data_weights.items():
      fname = f"{name}.pkl"
      if fname not in available:
        raise ValueError(
          f"Motion '{name}' not found in {self.cfg.motion_data_dir}. "
          f"Available: {available}"
        )
      path = os.path.join(self.cfg.motion_data_dir, fname)
      raw = joblib.load(path)

      fps = float(raw["fps"])
      dt = 1.0 / fps
      nf = len(raw["root_pos"])
      if nf < 2:
        raise ValueError(f"Motion '{name}' has <2 frames.")
      duration = dt * (nf - 1)
      loop_mode = int(raw.get("loop_mode", LoopMode.WRAP))

      durations.append(duration)
      fps_list.append(fps)
      dt_list.append(dt)
      num_frames_list.append(nf)
      loop_modes_list.append(loop_mode)
      weights_list.append(weight)

      root_pos = torch.as_tensor(
        raw["root_pos"], dtype=torch.float32, device=self.device
      )
      root_quat = torch.as_tensor(
        raw["root_rot"], dtype=torch.float32, device=self.device
      )
      root_pos_w_parts.append(root_pos)
      root_quat_parts.append(root_quat)
      root_vel_w_parts.append(vel_forward_diff(root_pos, dt))
      root_ang_vel_w_parts.append(
        ang_vel_from_quat_diff(root_quat, dt, in_frame="world")
      )
      dof_pos = torch.as_tensor(raw["dof_pos"], dtype=torch.float32, device=self.device)
      dof_pos_parts.append(dof_pos)
      dof_vel_parts.append(vel_forward_diff(dof_pos, dt))
      key_body_pos_w_parts.append(
        torch.as_tensor(raw["key_body_pos"], dtype=torch.float32, device=self.device)
      )

    # Tensors
    self.motion_durations = torch.tensor(
      durations, dtype=torch.float32, device=self.device
    )
    self.motion_dt = torch.tensor(dt_list, dtype=torch.float32, device=self.device)
    self.motion_num_frames = torch.tensor(
      num_frames_list, dtype=torch.int32, device=self.device
    )
    self.motion_loop_modes = torch.tensor(
      loop_modes_list, dtype=torch.int32, device=self.device
    )
    weights_t = torch.tensor(weights_list, dtype=torch.float32, device=self.device)
    self.motion_weights = weights_t / weights_t.sum()

    self.num_dofs = dof_pos_parts[0].shape[1]
    self.num_key_bodies = key_body_pos_w_parts[0].shape[1]

    self.root_pos_w = torch.cat(root_pos_w_parts, dim=0)
    self.root_quat = torch.cat(root_quat_parts, dim=0)
    self.root_vel_w = torch.cat(root_vel_w_parts, dim=0)
    self.root_ang_vel_w = torch.cat(root_ang_vel_w_parts, dim=0)
    self.dof_pos = torch.cat(dof_pos_parts, dim=0)
    self.dof_vel = torch.cat(dof_vel_parts, dim=0)
    self.key_body_pos_w = torch.cat(key_body_pos_w_parts, dim=0)

    lengths_shifted = self.motion_num_frames.roll(1)
    lengths_shifted[0] = 0
    self.motion_start_indices = torch.cumsum(lengths_shifted, dim=0)

  # -- public API --

  def get_num_motions(self) -> int:
    return self.motion_num_frames.shape[0]

  def get_total_duration(self) -> float:
    return self.motion_durations.sum().item()

  def get_motion_durations(self, motion_ids: torch.Tensor) -> torch.Tensor:
    return self.motion_durations[motion_ids]

  def sample_motions(self, n: int) -> torch.Tensor:
    return torch.multinomial(self.motion_weights, num_samples=n, replacement=True)

  def sample_times(
    self,
    motion_ids: torch.Tensor,
    truncate_time_start: float | None = None,
    truncate_time_end: float | None = None,
  ) -> torch.Tensor:
    dur = self.motion_durations[motion_ids]
    t_start = torch.zeros_like(dur)
    t_end = dur.clone()
    if truncate_time_start is not None:
      t_start = torch.clamp(t_start + truncate_time_start, max=dur)
    if truncate_time_end is not None:
      t_end = torch.clamp(t_end - truncate_time_end, min=0.0)
    valid = torch.clamp(t_end - t_start, min=1e-6)
    phase = torch.rand(motion_ids.shape, device=self.device)
    return t_start + phase * valid

  def calc_motion_phase(
    self, motion_ids: torch.Tensor, times: torch.Tensor
  ) -> torch.Tensor:
    dur = self.motion_durations[motion_ids]
    loop = self.motion_loop_modes[motion_ids]
    phase = times / dur
    wrap_mask = loop == int(LoopMode.WRAP)
    phase[wrap_mask] = phase[wrap_mask] - phase[wrap_mask].floor()
    return phase.clamp(0.0, 1.0)

  def get_motion_state(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    f0, f1, blend = self._calc_frame_blend(motion_ids, motion_times)

    rp0, rp1 = self.root_pos_w[f0], self.root_pos_w[f1]
    rq0, rq1 = self.root_quat[f0], self.root_quat[f1]
    rv0, rv1 = self.root_vel_w[f0], self.root_vel_w[f1]
    ra0, ra1 = self.root_ang_vel_w[f0], self.root_ang_vel_w[f1]
    dp0, dp1 = self.dof_pos[f0], self.dof_pos[f1]
    dv0, dv1 = self.dof_vel[f0], self.dof_vel[f1]
    kb0, kb1 = self.key_body_pos_w[f0], self.key_body_pos_w[f1]

    root_quat = quat_slerp_batched(rq0, rq1, blend)

    b = blend.unsqueeze(-1)
    root_pos_w = torch.lerp(rp0, rp1, b)
    root_vel_w = torch.lerp(rv0, rv1, b)
    root_vel_b = quat_apply_inverse(root_quat, root_vel_w)
    root_ang_vel_w = torch.lerp(ra0, ra1, b)
    root_ang_vel_b = quat_apply_inverse(root_quat, root_ang_vel_w)
    dof_pos = torch.lerp(dp0, dp1, b)
    dof_vel = torch.lerp(dv0, dv1, b)
    key_body_pos_w = torch.lerp(kb0, kb1, b.unsqueeze(1))
    num_key_bodies = key_body_pos_w.shape[1]
    root_quat_exp = root_quat.unsqueeze(1).expand(-1, num_key_bodies, 4)
    key_body_pos_b = quat_apply_inverse(
      root_quat_exp,
      key_body_pos_w - root_pos_w.unsqueeze(1),
    )

    return {
      "root_pos_w": root_pos_w,
      "root_quat": root_quat,
      "root_vel_w": root_vel_w,
      "root_vel_b": root_vel_b,
      "root_ang_vel_w": root_ang_vel_w,
      "root_ang_vel_b": root_ang_vel_b,
      "dof_pos": dof_pos,
      "dof_vel": dof_vel,
      "key_body_pos_b": key_body_pos_b,
    }

  # -- internals --

  def _calc_frame_blend(
    self, motion_ids: torch.Tensor, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nf = self.motion_num_frames[motion_ids]
    start = self.motion_start_indices[motion_ids]
    phase = self.calc_motion_phase(motion_ids, times)

    f0 = (phase * (nf - 1).float()).long()
    f1 = torch.minimum(f0 + 1, nf - 1)
    blend = phase * (nf - 1).float() - f0.float()

    f0 = f0 + start
    f1 = f1 + start
    return f0, f1, blend


# ---------------------------------------------------------------------------
# MotionDataManager (thin wrapper over multiple MotionDataTerm instances)
# ---------------------------------------------------------------------------


class MotionDataManager:
  """Manages multiple :class:`MotionDataTerm` datasets."""

  def __init__(
    self,
    cfg: dict[str, MotionDataTermCfg],
    num_envs: int,
    device: str,
  ) -> None:
    self._terms: dict[str, MotionDataTerm] = {}
    for name, term_cfg in cfg.items():
      self._terms[name] = MotionDataTerm(term_cfg, num_envs, device)

  def get_term(self, name: str) -> MotionDataTerm:
    if name not in self._terms:
      raise KeyError(
        f"Motion data term '{name}' not found. Available: {list(self._terms.keys())}"
      )
    return self._terms[name]

  @property
  def active_terms(self) -> list[str]:
    return list(self._terms.keys())
