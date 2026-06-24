"""Booster T1 (23自由度)在MuJoCo关节顺序中的对称性函数。

MuJoCo (DFS深度优先搜索) 关节顺序:
#0:  AAHead_yaw               (Z轴) 取反  [中线]
#1:  Head_pitch               (Y轴) 保持  [中线]
#2:  Left_Shoulder_Pitch      (Y轴) 保持
#3:  Left_Shoulder_Roll       (X轴) 取反
#4:  Left_Elbow_Pitch         (Y轴) 保持
#5:  Left_Elbow_Yaw           (Z轴) 取反
#6:  Right_Shoulder_Pitch     (Y轴) 保持
#7:  Right_Shoulder_Roll      (X轴) 取反
#8:  Right_Elbow_Pitch        (Y轴) 保持
#9:  Right_Elbow_Yaw          (Z轴) 取反
#10: Waist                    (Z轴) 取反  [中线]
#11: Left_Hip_Pitch           (Y轴) 保持
#12: Left_Hip_Roll            (X轴) 取反
#13: Left_Hip_Yaw             (Z轴) 取反
#14: Left_Knee_Pitch          (Y轴) 保持
#15: Left_Ankle_Pitch         (Y轴) 保持
#16: Left_Ankle_Roll          (X轴) 取反
#17: Right_Hip_Pitch          (Y轴) 保持
#18: Right_Hip_Roll           (X轴) 取反
#19: Right_Hip_Yaw            (Z轴) 取反
#20: Right_Knee_Pitch         (Y轴) 保持
#21: Right_Ankle_Pitch        (Y轴) 保持
#22: Right_Ankle_Roll         (X轴) 取反
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv

__all__ = ["compute_symmetric_states"]

NUM_JOINTS = 23

# 左右关节交换对 (MuJoCo DFS顺序)
_LEFT_INDICES = [2, 3, 4, 5, 11, 12, 13, 14, 15, 16]
_RIGHT_INDICES = [6, 7, 8, 9, 17, 18, 19, 20, 21, 22]

# 交换后需要取反的索引 (X轴和Z轴关节)
_NEGATE_INDICES = [
  0,   # AAHead_yaw (Z轴)
  3,   # Left_Shoulder_Roll (X轴)
  5,   # Left_Elbow_Yaw (Z轴)
  7,   # Right_Shoulder_Roll (X轴)
  9,   # Right_Elbow_Yaw (Z轴)
  10,  # Waist (Z轴)
  12,  # Left_Hip_Roll (X轴)
  13,  # Left_Hip_Yaw (Z轴)
  16,  # Left_Ankle_Roll (X轴)
  18,  # Right_Hip_Roll (X轴)
  19,  # Right_Hip_Yaw (Z轴)
  22,  # Right_Ankle_Roll (X轴)
]


@torch.no_grad()
def compute_symmetric_states(
  env: VecEnv,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """通过左右对称性扩增观测和动作数据。"""
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
  """对策略网络观测应用左右对称变换。

  数据布局: 角速度(3) | 重力投影(3) | 命令(3) |
          关节位置(23) | 关节速度(23) | 动作(23)
  总维度: 78维
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
  """对评论网络观测应用左右对称变换。

  数据布局: 线速度(3) | 角速度(3) | 重力投影(3) | 命令(3) |
          关节位置(23) | 关节速度(23) | 动作(23)
  总维度: 81维
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
  """对动作(23维)应用左右对称变换。"""
  return _switch_joints(actions.clone())


def _switch_joints(joint_data: torch.Tensor) -> torch.Tensor:
  """交换左右关节并对X/Z轴关节取反。"""
  out = joint_data.clone()
  out[..., _LEFT_INDICES] = joint_data[..., _RIGHT_INDICES]
  out[..., _RIGHT_INDICES] = joint_data[..., _LEFT_INDICES]
  out[..., _NEGATE_INDICES] = -out[..., _NEGATE_INDICES]
  return out
