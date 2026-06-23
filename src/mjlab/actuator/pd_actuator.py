"""理想PD控制执行器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
  from mjlab.entity import Entity

IdealPdCfgT = TypeVar("IdealPdCfgT", bound="IdealPdActuatorCfg")


@dataclass(kw_only=True)
class IdealPdActuatorCfg(ActuatorCfg):
  """理想PD执行器配置。"""

  stiffness: float
  """PD刚度（比例增益）。"""
  damping: float
  """PD阻尼（微分增益）。"""
  effort_limit: float = float("inf")
  """最大力/转矩限制。"""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> IdealPdActuator:
    return IdealPdActuator(self, entity, target_ids, target_names)


class IdealPdActuator(Actuator, Generic[IdealPdCfgT]):
  """理想PD控制执行器。"""

  def __init__(
    self,
    cfg: IdealPdCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.stiffness: torch.Tensor | None = None
    self.damping: torch.Tensor | None = None
    self.force_limit: torch.Tensor | None = None
    self.default_stiffness: torch.Tensor | None = None
    self.default_damping: torch.Tensor | None = None
    self.default_force_limit: torch.Tensor | None = None

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    # 为每个目标添加一个<motor>执行器。
    for target_name in target_names:
      actuator = create_motor_actuator(
        spec,
        target_name,
        effort_limit=self.cfg.effort_limit,
        armature=self.cfg.armature,
        frictionloss=self.cfg.frictionloss,
        transmission_type=self.cfg.transmission_type,
      )
      self._mjs_actuators.append(actuator)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    num_envs = data.nworld
    num_joints = len(self._target_names)
    self.stiffness = torch.full(
      (num_envs, num_joints), self.cfg.stiffness, dtype=torch.float, device=device
    )
    self.damping = torch.full(
      (num_envs, num_joints), self.cfg.damping, dtype=torch.float, device=device
    )
    self.force_limit = torch.full(
      (num_envs, num_joints), self.cfg.effort_limit, dtype=torch.float, device=device
    )

    # 保存默认值以便后续恢复。
    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()
    self.default_force_limit = self.force_limit.clone()

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.stiffness is not None
    assert self.damping is not None

    # 计算位置和速度误差。
    pos_error = cmd.position_target - cmd.pos
    vel_error = cmd.velocity_target - cmd.vel

    # PD控制：刚度 * 位置误差 + 阻尼 * 速度误差 + 目标力矩。
    computed_torques = self.stiffness * pos_error
    computed_torques += self.damping * vel_error
    computed_torques += cmd.effort_target

    return self._clip_effort(computed_torques)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    # 将转矩限制在允许范围内。
    assert self.force_limit is not None
    return torch.clamp(effort, -self.force_limit, self.force_limit)

  def set_gains(
    self,
    env_ids: torch.Tensor | slice,
    kp: torch.Tensor | None = None,
    kd: torch.Tensor | None = None,
  ) -> None:
    """为指定的环境设置PD增益。

    参数：
      env_ids: 要更新的环境索引。
      kp: 新的比例增益。形状：(num_envs, num_actuators) 或 (num_envs,)。
      kd: 新的微分增益。形状：(num_envs, num_actuators) 或 (num_envs,)。
    """
    assert self.stiffness is not None
    assert self.damping is not None

    if kp is not None:
      if kp.ndim == 1:
        kp = kp.unsqueeze(-1)
      self.stiffness[env_ids] = kp

    if kd is not None:
      if kd.ndim == 1:
        kd = kd.unsqueeze(-1)
      self.damping[env_ids] = kd

  def set_effort_limit(
    self, env_ids: torch.Tensor | slice, effort_limit: torch.Tensor
  ) -> None:
    """为指定的环境设置力/转矩限制。

    参数：
      env_ids: 要更新的环境索引。
      effort_limit: 新的力/转矩限制。形状：(num_envs, num_actuators) 或 (num_envs,)。
    """
    assert self.force_limit is not None

    if effort_limit.ndim == 1:
      effort_limit = effort_limit.unsqueeze(-1)
    self.force_limit[env_ids] = effort_limit
