"""Symmetry functions for Booster T1 (23 DOF) in MuJoCo joint ordering.

MuJoCo (DFS) joint ordering:
#0:  AAHead_yaw               (Z axis) negate  [midline]
#1:  Head_pitch               (Y axis) keep    [midline]
#2:  Left_Shoulder_Pitch      (Y axis) keep
#3:  Left_Shoulder_Roll       (X axis) negate
#4:  Left_Elbow_Pitch         (Y axis) keep
#5:  Left_Elbow_Yaw           (Z axis) negate
#6:  Right_Shoulder_Pitch     (Y axis) keep
#7:  Right_Shoulder_Roll      (X axis) negate
#8:  Right_Elbow_Pitch        (Y axis) keep
#9:  Right_Elbow_Yaw          (Z axis) negate
#10: Waist                    (Z axis) negate  [midline]
#11: Left_Hip_Pitch           (Y axis) keep
#12: Left_Hip_Roll            (X axis) negate
#13: Left_Hip_Yaw             (Z axis) negate
#14: Left_Knee_Pitch          (Y axis) keep
#15: Left_Ankle_Pitch         (Y axis) keep
#16: Left_Ankle_Roll          (X axis) negate
#17: Right_Hip_Pitch          (Y axis) keep
#18: Right_Hip_Roll           (X axis) negate
#19: Right_Hip_Yaw            (Z axis) negate
#20: Right_Knee_Pitch         (Y axis) keep
#21: Right_Ankle_Pitch        (Y axis) keep
#22: Right_Ankle_Roll         (X axis) negate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv

__all__ = ["compute_symmetric_states"]

NUM_JOINTS = 23

# Left-right swap pairs (MuJoCo DFS ordering)
_LEFT_INDICES = [2, 3, 4, 5, 11, 12, 13, 14, 15, 16]
_RIGHT_INDICES = [6, 7, 8, 9, 17, 18, 19, 20, 21, 22]

# Indices to negate after swap (X and Z axis joints)
_NEGATE_INDICES = [
  0,   # AAHead_yaw (Z)
  3,   # Left_Shoulder_Roll (X)
  5,   # Left_Elbow_Yaw (Z)
  7,   # Right_Shoulder_Roll (X)
  9,   # Right_Elbow_Yaw (Z)
  10,  # Waist (Z)
  12,  # Left_Hip_Roll (X)
  13,  # Left_Hip_Yaw (Z)
  16,  # Left_Ankle_Roll (X)
  18,  # Right_Hip_Roll (X)
  19,  # Right_Hip_Yaw (Z)
  22,  # Right_Ankle_Roll (X)
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
    actor_key = "actor" if "actor" in obs.keys() else "policy"
    obs_aug[actor_key][:batch_size] = obs[actor_key][:]
    obs_aug[actor_key][batch_size:] = _transform_policy_obs(obs[actor_key])
    obs_aug["critic"][:batch_size] = obs["critic"][:]
    obs_aug["critic"][batch_size:] = _transform_critic_obs(obs["critic"])
  else:
    obs_aug = None

  if actions is not None:
    batch_size = actions.shape[0]
    actions_aug = torch.zeros(
      batch_size * 2, actions.shape[1], device=actions.device
    )
    actions_aug[:batch_size] = actions[:]
    actions_aug[batch_size:] = _transform_actions(actions)
  else:
    actions_aug = None

  return obs_aug, actions_aug


def _transform_policy_obs(obs: torch.Tensor) -> torch.Tensor:
  """Apply left-right symmetry to policy observation.

  Layout: ang_vel(3) | proj_gravity(3) | cmd(3) |
          joint_pos(23) | joint_vel(23) | actions(23)
  Total: 78 dims
  """
  obs = obs.clone()
  d = obs.device
  obs[:, 0:3] *= torch.tensor([-1, 1, -1], device=d)
  obs[:, 3:6] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 6:9] *= torch.tensor([1, -1, -1], device=d)
  obs[:, 9:32] = _switch_joints(obs[:, 9:32])
  obs[:, 32:55] = _switch_joints(obs[:, 32:55])
  obs[:, 55:78] = _switch_joints(obs[:, 55:78])
  return obs


def _transform_critic_obs(obs: torch.Tensor) -> torch.Tensor:
  """Apply left-right symmetry to critic observation.

  Layout: lin_vel(3) | ang_vel(3) | proj_gravity(3) | cmd(3) |
          joint_pos(23) | joint_vel(23) | actions(23)
  Total: 81 dims
  """
  obs = obs.clone()
  d = obs.device
  obs[:, 0:3] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 3:6] *= torch.tensor([-1, 1, -1], device=d)
  obs[:, 6:9] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 9:12] *= torch.tensor([1, -1, -1], device=d)
  obs[:, 12:35] = _switch_joints(obs[:, 12:35])
  obs[:, 35:58] = _switch_joints(obs[:, 35:58])
  obs[:, 58:81] = _switch_joints(obs[:, 58:81])
  return obs


def _transform_actions(actions: torch.Tensor) -> torch.Tensor:
  """Apply left-right symmetry to actions (23 dims)."""
  return _switch_joints(actions.clone())


def _switch_joints(joint_data: torch.Tensor) -> torch.Tensor:
  """Swap left-right joints and negate X/Z axis joints."""
  out = joint_data.clone()
  out[..., _LEFT_INDICES] = joint_data[..., _RIGHT_INDICES]
  out[..., _RIGHT_INDICES] = joint_data[..., _LEFT_INDICES]
  out[..., _NEGATE_INDICES] = -out[..., _NEGATE_INDICES]
  return out
