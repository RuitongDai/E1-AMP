"""Symmetry functions for TFBOT (31 DOF) in MuJoCo joint ordering.

MuJoCo (DFS) joint ordering:
#0:  r_hip_roll_joint         (X axis) negate
#1:  r_hip_pitch_joint        (Y axis) keep
#2:  r_hip_yaw_joint          (Z axis) negate
#3:  r_knee_pitch_joint       (Y axis) keep
#4:  r_ankle_roll_joint       (X axis) negate
#5:  r_ankle_pitch_joint      (Y axis) keep
#6:  l_hip_roll_joint         (X axis) negate
#7:  l_hip_pitch_joint        (Y axis) keep
#8:  l_hip_yaw_joint          (Z axis) negate
#9:  l_knee_pitch_joint       (Y axis) keep
#10: l_ankle_roll_joint       (X axis) negate
#11: l_ankle_pitch_joint      (Y axis) keep
#12: waist_roll_joint         (X axis) negate  [midline]
#13: waist_pitch_joint        (Y axis) keep    [midline]
#14: waist_yaw_joint          (Z axis) negate  [midline]
#15: r_shoulder_shrug_joint   (X axis) negate
#16: r_shoulder_pitch_joint   (Y axis) keep
#17: r_shoulder_roll_joint    (X axis) negate
#18: r_shoulder_yaw_joint     (Z axis) negate
#19: r_elbow_pitch_joint      (Y axis) keep
#20: r_radius_roll_joint      (Z axis) negate
#21: r_wrist_yaw_joint        (X axis) negate
#22: r_wrist_pitch_joint      (X axis) negate
#23: l_shoulder_shrug_joint   (X axis) negate
#24: l_shoulder_pitch_joint   (Y axis) keep
#25: l_shoulder_roll_joint    (X axis) negate
#26: l_shoulder_yaw_joint     (Z axis) negate
#27: l_elbow_pitch_joint      (Y axis) keep
#28: l_radius_roll_joint      (Z axis) negate
#29: l_wrist_yaw_joint        (X axis) negate
#30: l_wrist_pitch_joint      (X axis) negate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv

__all__ = ["compute_symmetric_states"]

NUM_JOINTS = 31

# Left-right swap pairs (mjlab MuJoCo DFS ordering)
_RIGHT_INDICES = [0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22]
_LEFT_INDICES = [6, 7, 8, 9, 10, 11, 23, 24, 25, 26, 27, 28, 29, 30]

# Indices to negate after swap (X and Z axis joints)
_NEGATE_INDICES = [
  0,
  2,
  4,
  6,
  8,
  10,
  12,
  14,
  15,
  17,
  18,
  20,
  21,
  22,
  23,
  25,
  26,
  28,
  29,
  30,
]


@torch.no_grad()
def compute_symmetric_states(
  env: VecEnv,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Augment observations and actions with left-right symmetry."""
  if obs is not None:
    batch_size = obs.batch_size[0]
    obs_aug = obs.repeat(2)
    # mjlab uses 'actor'/'critic' as observation group keys
    actor_key = "actor" if "actor" in obs.keys() else "policy"
    obs_aug[actor_key][:batch_size] = obs[actor_key][:]
    obs_aug[actor_key][batch_size:] = _transform_policy_obs(obs[actor_key])
    obs_aug["critic"][:batch_size] = obs["critic"][:]
    obs_aug["critic"][batch_size:] = _transform_critic_obs(obs["critic"])
  else:
    obs_aug = None

  if actions is not None:
    batch_size = actions.shape[0]
    actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
    actions_aug[:batch_size] = actions[:]
    actions_aug[batch_size:] = _transform_actions(actions)
  else:
    actions_aug = None

  return obs_aug, actions_aug


def _transform_policy_obs(obs: torch.Tensor) -> torch.Tensor:
  """Apply left-right symmetry to policy observation.

  Layout: ang_vel(3) | proj_gravity(3) | cmd(3) |
          joint_pos(31) | joint_vel(31) | actions(31)
  Total: 102 dims
  """
  obs = obs.clone()
  d = obs.device
  obs[:, 0:3] *= torch.tensor([-1, 1, -1], device=d)
  obs[:, 3:6] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 6:9] *= torch.tensor([1, -1, -1], device=d)
  obs[:, 9:40] = _switch_joints(obs[:, 9:40])
  obs[:, 40:71] = _switch_joints(obs[:, 40:71])
  obs[:, 71:102] = _switch_joints(obs[:, 71:102])
  return obs


def _transform_critic_obs(obs: torch.Tensor) -> torch.Tensor:
  """Apply left-right symmetry to critic observation.

  Layout: lin_vel(3) | ang_vel(3) | proj_gravity(3) | cmd(3) |
          joint_pos(31) | joint_vel(31) | actions(31)
  Total: 105 dims
  """
  obs = obs.clone()
  d = obs.device
  obs[:, 0:3] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 3:6] *= torch.tensor([-1, 1, -1], device=d)
  obs[:, 6:9] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 9:12] *= torch.tensor([1, -1, -1], device=d)
  obs[:, 12:43] = _switch_joints(obs[:, 12:43])
  obs[:, 43:74] = _switch_joints(obs[:, 43:74])
  obs[:, 74:105] = _switch_joints(obs[:, 74:105])
  return obs


def _transform_actions(actions: torch.Tensor) -> torch.Tensor:
  """Apply left-right symmetry to actions (31 dims)."""
  return _switch_joints(actions.clone())


def _switch_joints(joint_data: torch.Tensor) -> torch.Tensor:
  """Swap left-right joints and negate X/Z axis joints."""
  out = joint_data.clone()
  out[..., _LEFT_INDICES] = joint_data[..., _RIGHT_INDICES]
  out[..., _RIGHT_INDICES] = joint_data[..., _LEFT_INDICES]
  out[..., _NEGATE_INDICES] = -out[..., _NEGATE_INDICES]
  return out
