"""Booster E1 (25自由度)在MuJoCo关节顺序中的对称性函数。

根据 E1 的 XML 结构，MuJoCo (DFS深度优先搜索) 解析的有效自由度(排除了固定的头部)顺序为:
#0:  left_hip_pitch        (Y轴) 保持
#1:  left_hip_roll         (X轴) 取反
#2:  left_hip_yaw          (Z轴) 取反
#3:  left_knee             (Y轴) 保持
#4:  left_ankle_pitch      (Y轴) 保持
#5:  left_ankle_roll       (X轴) 取反
#6:  right_hip_pitch       (Y轴) 保持
#7:  right_hip_roll        (X轴) 取反
#8:  right_hip_yaw         (Z轴) 取反
#9:  right_knee            (Y轴) 保持
#10: right_ankle_pitch     (Y轴) 保持
#11: right_ankle_roll      (X轴) 取反
#12: waist_yaw             (Z轴) 取反  [中线]
#13: left_shoulder_pitch   (Y轴) 保持
#14: left_shoulder_roll    (X轴) 取反
#15: left_shoulder_yaw     (Z轴) 取反
#16: left_elbow            (Y轴) 保持
#17: left_wrist_roll       (X轴) 取反
#18: left_wrist_pitch      (Y轴) 保持
#19: right_shoulder_pitch  (Y轴) 保持
#20: right_shoulder_roll   (X轴) 取反
#21: right_shoulder_yaw    (Z轴) 取反
#22: right_elbow           (Y轴) 保持
#23: right_wrist_roll      (X轴) 取反
#24: right_wrist_pitch     (Y轴) 保持
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv

__all__ = ["compute_symmetric_states"]

NUM_JOINTS = 25

# 左右关节交换对
_LEFT_INDICES = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17, 18]
_RIGHT_INDICES = [6, 7, 8, 9, 10, 11, 19, 20, 21, 22, 23, 24]

# 交换后需要取反的索引 (所有绕 X 轴或 Z 轴旋转的关节)
_NEGATE_INDICES = [
  1, 2, 5,       # 左腿 roll, yaw, ankle_roll
  7, 8, 11,      # 右腿 roll, yaw, ankle_roll
  12,            # 腰部 yaw
  14, 15, 17,    # 左臂 roll, yaw, wrist_roll
  20, 21, 23     # 右臂 roll, yaw, wrist_roll
]

@torch.no_grad()
def compute_symmetric_states(
  env: VecEnv,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
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
  """
  数据布局: 角速度(3) | 重力投影(3) | 命令(3) | 关节位置(25) | 关节速度(25) | 动作(25)
  总维度: 9 + 25*3 = 84维
  """
  obs = obs.clone()
  d = obs.device
  obs[:, 0:3] *= torch.tensor([-1, 1, -1], device=d)
  obs[:, 3:6] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 6:9] *= torch.tensor([1, -1, -1], device=d)

  # ======= 数组切片针对 25 个关节进行移位计算 =======
  obs[:, 9:34] = _switch_joints(obs[:, 9:34])
  obs[:, 34:59] = _switch_joints(obs[:, 34:59])
  obs[:, 59:84] = _switch_joints(obs[:, 59:84])
  return obs

def _transform_critic_obs(obs: torch.Tensor) -> torch.Tensor:
  """
  数据布局: 线速度(3) | 角速度(3) | 重力投影(3) | 命令(3) | 关节位置(25) | 关节速度(25) | 动作(25)
  总维度: 12 + 25*3 = 87维
  """
  obs = obs.clone()
  d = obs.device
  obs[:, 0:3] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 3:6] *= torch.tensor([-1, 1, -1], device=d)
  obs[:, 6:9] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 9:12] *= torch.tensor([1, -1, -1], device=d)

  # ======= 数组切片针对 25 个关节进行移位计算 =======
  obs[:, 12:37] = _switch_joints(obs[:, 12:37])
  obs[:, 37:62] = _switch_joints(obs[:, 37:62])
  obs[:, 62:87] = _switch_joints(obs[:, 62:87])
  return obs

def _transform_actions(actions: torch.Tensor) -> torch.Tensor:
  return _switch_joints(actions.clone())

def _switch_joints(joint_data: torch.Tensor) -> torch.Tensor:
  out = joint_data.clone()
  out[..., _LEFT_INDICES] = joint_data[..., _RIGHT_INDICES]
  out[..., _RIGHT_INDICES] = joint_data[..., _LEFT_INDICES]
  out[..., _NEGATE_INDICES] = -out[..., _NEGATE_INDICES]
  return out