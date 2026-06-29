"""Moya01 V2 (X3, 28自由度)在MuJoCo关节顺序中的对称性函数。

MuJoCo (DFS深度优先搜索) 关节顺序:
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
#12: waist_roll            (X轴) 取反 [中线]
#13: waist_yaw             (Z轴) 取反 [中线]
#14: left_shoulder_pitch   (Y轴) 保持
#15: left_shoulder_roll    (X轴) 取反
#16: left_shoulder_yaw     (Z轴) 取反
#17: left_elbow            (Y轴) 保持
#18: left_wrist_roll       (X轴) 取反
#19: left_wrist_pitch      (Y轴) 保持
#20: left_wrist_yaw        (Z轴) 取反
#21: right_shoulder_pitch  (Y轴) 保持
#22: right_shoulder_roll   (X轴) 取反
#23: right_shoulder_yaw    (Z轴) 取反
#24: right_elbow           (Y轴) 保持
#25: right_wrist_roll      (X轴) 取反
#26: right_wrist_pitch     (Y轴) 保持
#27: right_wrist_yaw       (Z轴) 取反
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv

__all__ = ["compute_symmetric_states"]

NUM_JOINTS = 28

# 左右关节交换对
_LEFT_INDICES = [0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20]
_RIGHT_INDICES = [6, 7, 8, 9, 10, 11, 21, 22, 23, 24, 25, 26, 27]

# 交换后需要取反的索引 (X轴和Z轴关节)
_NEGATE_INDICES = [
  1, 2, 5,          # 左腿 Roll, Yaw
  7, 8, 11,         # 右腿 Roll, Yaw
  12, 13,           # 腰部 Roll, Yaw
  15, 16, 18, 20,   # 左臂 Roll, Yaw
  22, 23, 25, 27,   # 右臂 Roll, Yaw
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
  数据布局: 角速度(3) | 重力投影(3) | 命令(3) | 关节位置(28) | 关节速度(28) | 动作(28)
  总维度: 93维
  """
  obs = obs.clone()
  d = obs.device
  obs[:, 0:3] *= torch.tensor([-1, 1, -1], device=d)
  obs[:, 3:6] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 6:9] *= torch.tensor([1, -1, -1], device=d)
  obs[:, 9:37] = _switch_joints(obs[:, 9:37])
  obs[:, 37:65] = _switch_joints(obs[:, 37:65])
  obs[:, 65:93] = _switch_joints(obs[:, 65:93])
  return obs

def _transform_critic_obs(obs: torch.Tensor) -> torch.Tensor:
  """对评论网络观测应用左右对称变换。
  数据布局: 线速度(3) | 角速度(3) | 重力投影(3) | 命令(3) | 关节位置(28) | 关节速度(28) | 动作(28)
  总维度: 96维
  """
  obs = obs.clone()
  d = obs.device
  obs[:, 0:3] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 3:6] *= torch.tensor([-1, 1, -1], device=d)
  obs[:, 6:9] *= torch.tensor([1, -1, 1], device=d)
  obs[:, 9:12] *= torch.tensor([1, -1, -1], device=d)
  obs[:, 12:40] = _switch_joints(obs[:, 12:40])
  obs[:, 40:68] = _switch_joints(obs[:, 40:68])
  obs[:, 68:96] = _switch_joints(obs[:, 68:96])
  return obs

def _transform_actions(actions: torch.Tensor) -> torch.Tensor:
  """对动作(28维)应用左右对称变换。"""
  return _switch_joints(actions.clone())

def _switch_joints(joint_data: torch.Tensor) -> torch.Tensor:
  """交换左右关节并对X/Z轴关节取反。"""
  out = joint_data.clone()
  out[..., _LEFT_INDICES] = joint_data[..., _RIGHT_INDICES]
  out[..., _RIGHT_INDICES] = joint_data[..., _LEFT_INDICES]
  out[..., _NEGATE_INDICES] = -out[..., _NEGATE_INDICES]
  return out