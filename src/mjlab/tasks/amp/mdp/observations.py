"""AMP-specific observation terms.

All functions follow the mjlab observation term signature:
  func(env, **params) -> torch.Tensor
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.amp.amp_env import AmpEnv


# ---------------------------------------------------------------------------
# Robot (policy) observations
# ---------------------------------------------------------------------------


def root_rot_tan_norm(
  env: ManagerBasedRlEnv, entity_name: str = "robot"
) -> torch.Tensor:
  """Rotation matrix tangent + normal vectors (columns 0 and 2)."""
  entity = env.scene[entity_name]
  root_quat = entity.data.root_quat_w
  rotm = matrix_from_quat(root_quat)
  tan_vec = rotm[:, :, 0]
  norm_vec = rotm[:, :, 2]
  return torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)


def root_local_rot_tan_norm(
  env: ManagerBasedRlEnv, entity_name: str = "robot"
) -> torch.Tensor:
  """Yaw-removed rotation tangent + normal."""
  entity = env.scene[entity_name]
  root_quat = entity.data.root_quat_w
  yaw_q = yaw_quat(root_quat)
  local_q = quat_mul(quat_conjugate(yaw_q), root_quat)
  rotm = matrix_from_quat(local_q)
  return torch.cat([rotm[:, :, 0], rotm[:, :, 2]], dim=-1)  # (N, 6)


def key_body_pos_b(
  env: ManagerBasedRlEnv,
  entity_name: str = "robot",
  body_names: tuple[str, ...] = (),
) -> torch.Tensor:
  """Key body positions in the base frame, flattened."""
  entity = env.scene[entity_name]
  body_ids = [entity.body_id(n) for n in body_names]
  kb_pos_w = entity.data.body_pos_w[:, body_ids, :]
  root_pos = entity.data.root_pos_w
  root_quat = entity.data.root_quat_w
  n_bodies = len(body_ids)
  kb_pos_b = quat_apply_inverse(
    root_quat.unsqueeze(1).expand(-1, n_bodies, -1),
    kb_pos_w - root_pos.unsqueeze(1),
  )
  return kb_pos_b.reshape(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Reference motion observations (from animation manager)
# ---------------------------------------------------------------------------


def ref_root_rot_tan_norm(env: AmpEnv, animation: str) -> torch.Tensor:
  """Reference motion root rotation tangent + normal, per step."""
  term = env.animation_manager.get_term(animation)
  rq = term.get_root_quat()  # (N, S, 4)
  rotm = matrix_from_quat(rq)
  tan = rotm[..., 0]  # (N, S, 3)
  norm = rotm[..., 2]
  obs = torch.cat([tan, norm], dim=-1)  # (N, S, 6)
  return obs


def ref_root_local_rot_tan_norm(env: AmpEnv, animation: str) -> torch.Tensor:
  """Reference yaw-removed rotation tangent + normal, per step."""
  term = env.animation_manager.get_term(animation)
  rq = term.get_root_quat()
  yaw_q = yaw_quat(rq)
  local_q = quat_mul(quat_conjugate(yaw_q), rq)
  rotm = matrix_from_quat(local_q)
  return torch.cat([rotm[..., 0], rotm[..., 2]], dim=-1)  # (N, S, 6)


def ref_root_ang_vel_b(env: AmpEnv, animation: str) -> torch.Tensor:
  """Reference root angular velocity in body frame, per step."""
  term = env.animation_manager.get_term(animation)
  ang_vel_w = term.get_root_ang_vel_w()
  rq = term.get_root_quat()
  return quat_apply_inverse(rq, ang_vel_w)  # (N, S, 3)


def ref_root_lin_vel_b(env: AmpEnv, animation: str) -> torch.Tensor:
  """Reference root linear velocity in body frame, per step."""
  term = env.animation_manager.get_term(animation)
  vel_w = term.get_root_vel_w()
  rq = term.get_root_quat()
  return quat_apply_inverse(rq, vel_w)  # (N, S, 3)


def ref_joint_pos(env: AmpEnv, animation: str) -> torch.Tensor:
  """Reference joint positions, per step."""
  term = env.animation_manager.get_term(animation)
  return term.get_dof_pos()  # (N, S, D)


def ref_joint_vel(env: AmpEnv, animation: str) -> torch.Tensor:
  """Reference joint velocities, per step."""
  term = env.animation_manager.get_term(animation)
  return term.get_dof_vel()  # (N, S, D)


def ref_key_body_pos_b(env: AmpEnv, animation: str) -> torch.Tensor:
  """Reference key body positions in base frame, per step.

  Returns shape (N, S, K*3) with the last two dims flattened.
  """
  term = env.animation_manager.get_term(animation)
  kb = term.get_key_body_pos_b()  # (N, S, K, 3)
  N, S = kb.shape[:2]
  return kb.reshape(N, S, -1)


def ref_root_projected_gravity(env: AmpEnv, animation: str) -> torch.Tensor:
  """Projected gravity in reference root frame, per step."""
  term = env.animation_manager.get_term(animation)
  rq = term.get_root_quat()  # (N, S, 4)
  grav = torch.tensor([0.0, 0.0, -1.0], device=rq.device).expand_as(rq[..., :3])
  return quat_apply_inverse(rq, grav)  # (N, S, 3)
