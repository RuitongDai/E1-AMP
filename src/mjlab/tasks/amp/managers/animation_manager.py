"""Animation manager: fetches and buffers reference motion states."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.amp.managers.motion_data_manager import MotionDataTerm

if TYPE_CHECKING:
  from mjlab.tasks.amp.amp_env import AmpEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AnimationTermCfg:
  """Configuration for an animation term."""

  motion_data_term: str = ""
  """Name of the motion data term to reference."""
  motion_data_components: list[str] = field(default_factory=list)
  """Which components to buffer (root_pos_w, root_quat, dof_pos, …)."""
  num_steps_to_use: int = 1
  """Positive → current + future; negative → past + current; 0 invalid."""
  random_initialize: bool = False
  """Randomly sample start time on reset."""
  random_fetch: bool = False
  """Randomly sample time each step (demonstration mode)."""


# ---------------------------------------------------------------------------
# AnimationTerm
# ---------------------------------------------------------------------------

_VEC3_COMPONENTS = frozenset(
  ["root_pos_w", "root_vel_w", "root_vel_b", "root_ang_vel_w", "root_ang_vel_b"]
)


class AnimationTerm:
  """Buffers multi-step reference states from a :class:`MotionDataTerm`."""

  def __init__(
    self, cfg: AnimationTermCfg, env: AmpEnv, motion_data_term: MotionDataTerm
  ) -> None:
    self.cfg = cfg
    self._env = env
    self.motion_data_term = motion_data_term
    self.num_envs = env.num_envs
    self.device = env.device

    if cfg.num_steps_to_use == 0:
      raise ValueError("num_steps_to_use cannot be zero.")

    if cfg.num_steps_to_use > 0:
      self.step_indices = torch.arange(
        0, cfg.num_steps_to_use, dtype=torch.long, device=self.device
      )
    else:
      self.step_indices = torch.arange(
        cfg.num_steps_to_use + 1, 1, dtype=torch.long, device=self.device
      )
    self.num_steps = len(self.step_indices)

    # Allocate buffers
    for comp in cfg.motion_data_components:
      shape: tuple[int, ...] = (self.num_envs, self.num_steps)
      if comp in _VEC3_COMPONENTS:
        shape += (3,)
      elif comp == "root_quat":
        shape += (4,)
      elif comp in ("dof_pos", "dof_vel"):
        shape += (motion_data_term.num_dofs,)
      elif comp == "key_body_pos_b":
        shape += (motion_data_term.num_key_bodies, 3)
      else:
        raise ValueError(f"Unknown motion component: {comp}")
      setattr(
        self,
        f"{comp}_buffer",
        torch.zeros(shape, device=self.device, dtype=torch.float32),
      )

    self.motion_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.motion_fetch_time = torch.zeros(
      (self.num_envs, self.num_steps), device=self.device
    )
    self.motion_durations = torch.zeros(self.num_envs, device=self.device)

    self.reset(torch.arange(self.num_envs, device=self.device))
    self._fetch_motion_data()

  # -- lifecycle --

  def reset(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.motion_ids[env_ids] = self.motion_data_term.sample_motions(n)
    self.motion_durations[env_ids] = self.motion_data_term.get_motion_durations(
      self.motion_ids[env_ids]
    )

    truncate = self.num_steps * self._env.step_dt
    if self.cfg.random_initialize:
      if self.cfg.num_steps_to_use > 0:
        t0 = self.motion_data_term.sample_times(
          self.motion_ids[env_ids], truncate_time_end=truncate
        )
      else:
        t0 = self.motion_data_term.sample_times(
          self.motion_ids[env_ids], truncate_time_start=truncate
        )
    else:
      t0 = torch.zeros(n, device=self.device)

    self.motion_fetch_time[env_ids, 0] = t0
    if self.num_steps > 1:
      offsets = self.step_indices[1:].float() * self._env.step_dt
      self.motion_fetch_time[env_ids, 1:] = t0.unsqueeze(-1) + offsets

    self._fetch_motion_data(env_ids)

  def update(self, dt: float) -> None:
    if self.cfg.random_fetch:
      truncate = self.num_steps * dt
      if self.cfg.num_steps_to_use > 0:
        t0 = self.motion_data_term.sample_times(
          self.motion_ids, truncate_time_end=truncate
        )
      else:
        t0 = self.motion_data_term.sample_times(
          self.motion_ids, truncate_time_start=truncate
        )
      self.motion_fetch_time[:, 0] = t0
      if self.num_steps > 1:
        offsets = self.step_indices[1:].float() * dt
        self.motion_fetch_time[:, 1:] = t0.unsqueeze(-1) + offsets

    self._fetch_motion_data()

    if not self.cfg.random_fetch:
      self.motion_fetch_time += dt

  # -- data access --

  def get_root_pos_w(self) -> torch.Tensor:
    return self.root_pos_w_buffer

  def get_root_quat(self) -> torch.Tensor:
    return self.root_quat_buffer

  def get_root_vel_w(self) -> torch.Tensor:
    return self.root_vel_w_buffer

  def get_root_ang_vel_w(self) -> torch.Tensor:
    return self.root_ang_vel_w_buffer

  def get_dof_pos(self) -> torch.Tensor:
    return self.dof_pos_buffer

  def get_dof_vel(self) -> torch.Tensor:
    return self.dof_vel_buffer

  def get_key_body_pos_b(self) -> torch.Tensor:
    return self.key_body_pos_b_buffer

  # -- internal --

  def _fetch_motion_data(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)

    n = len(env_ids)
    times_flat = self.motion_fetch_time[env_ids].reshape(-1)
    ids_flat = self.motion_ids[env_ids].repeat_interleave(self.num_steps)

    state = self.motion_data_term.get_motion_state(ids_flat, times_flat)

    for comp in self.cfg.motion_data_components:
      if comp in state:
        buf = getattr(self, f"{comp}_buffer")
        data = state[comp].view(n, self.num_steps, *state[comp].shape[1:])
        buf[env_ids] = data


# ---------------------------------------------------------------------------
# AnimationManager (thin wrapper)
# ---------------------------------------------------------------------------


class AnimationManager:
  """Manages multiple :class:`AnimationTerm` instances."""

  def __init__(self, cfg: dict[str, AnimationTermCfg], env: AmpEnv) -> None:
    self._terms: dict[str, AnimationTerm] = {}
    self._env = env
    for name, term_cfg in cfg.items():
      md_term = env.motion_data_manager.get_term(term_cfg.motion_data_term)
      self._terms[name] = AnimationTerm(term_cfg, env, md_term)

  def update(self, dt: float) -> None:
    for term in self._terms.values():
      term.update(dt)

  def reset(self, env_ids: torch.Tensor) -> None:
    for term in self._terms.values():
      term.reset(env_ids)

  def get_term(self, name: str) -> AnimationTerm:
    return self._terms[name]

  @property
  def active_terms(self) -> list[str]:
    return list(self._terms.keys())
