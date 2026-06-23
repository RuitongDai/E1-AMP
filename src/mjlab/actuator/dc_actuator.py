"""直流电机执行器，具有基于速度的饱和模型。

该模块提供了一个直流电机执行器，实现了现实的转矩-速度曲线，用于更准确的电机行为模拟。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity

DcMotorCfgT = TypeVar("DcMotorCfgT", bound="DcMotorActuatorCfg")


@dataclass(kw_only=True)
class DcMotorActuatorCfg(IdealPdActuatorCfg):
  """基于速度的饱和直流电机执行器配置。

  该执行器实现了直流电机转矩-速度曲线，提供更逼真的执行器行为。
  电机在零速度时产生最大转矩（saturation_effort），并随着速度增加线性降低，
  在最大速度时转矩降为零。

  注意：effort_limit应显式设置为现实值以进行适当的电机建模。
  使用默认值（inf）将触发警告。如果需要无限转矩，请使用 IdealPdActuator。
  """

  saturation_effort: float
  """零速度时的峰值电机转矩（堵转转矩）。"""

  velocity_limit: float
  """最大电机速度（空载速度）。"""

  def __post_init__(self) -> None:
    """验证直流电机参数。"""
    import warnings

    if self.effort_limit == float("inf"):
      warnings.warn(
        "DcMotorActuator 的 effort_limit 设置为 inf，这会创建一个不现实的、"
        "具有无限连续转矩的电机。考虑将 effort_limit 设置为电机的连续额定值"
        "(<= saturation_effort)。如果真的需要无限转矩，请使用 IdealPdActuator。",
        UserWarning,
        stacklevel=2,
      )

    if self.effort_limit > self.saturation_effort:
      warnings.warn(
        f"effort_limit ({self.effort_limit}) 超过了 saturation_effort "
        f"({self.saturation_effort})。对于现实电机，连续转矩应 <= 峰值转矩。",
        UserWarning,
        stacklevel=2,
      )

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> DcMotorActuator:
    return DcMotorActuator(self, entity, target_ids, target_names)


class DcMotorActuator(IdealPdActuator[DcMotorCfgT], Generic[DcMotorCfgT]):
  """基于速度的饱和模型的直流电机执行器。

  该执行器扩展了 IdealPdActuator，实现了现实的直流电机模型，
  根据当前关节速度限制转矩。该模型实现了线性转矩-速度曲线：
  - 零速度时：能产生最大饱和转矩（stall torque）
  - 最大速度时：转矩为零
  - 中间：转矩限制线性变化

  连续转矩限制（effort_limit）进一步约束输出。
  """

  def __init__(
    self,
    cfg: DcMotorCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.saturation_effort: torch.Tensor | None = None
    self.velocity_limit_motor: torch.Tensor | None = None
    self._vel_at_effort_lim: torch.Tensor | None = None
    self._joint_vel_clipped: torch.Tensor | None = None

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

    self.saturation_effort = torch.full(
      (num_envs, num_joints),
      self.cfg.saturation_effort,
      dtype=torch.float,
      device=device,
    )
    self.velocity_limit_motor = torch.full(
      (num_envs, num_joints),
      self.cfg.velocity_limit,
      dtype=torch.float,
      device=device,
    )

    # 计算转矩-速度曲线与 effort_limit 的交点速度。
    assert self.force_limit is not None
    self._vel_at_effort_lim = self.velocity_limit_motor * (
      1 + self.force_limit / self.saturation_effort
    )
    self._joint_vel_clipped = torch.zeros(num_envs, num_joints, device=device)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    # 记录当前关节速度用于转矩限制计算。
    assert self._joint_vel_clipped is not None
    self._joint_vel_clipped[:] = cmd.vel
    return super().compute(cmd)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    assert self.saturation_effort is not None
    assert self.velocity_limit_motor is not None
    assert self.force_limit is not None
    assert self._vel_at_effort_lim is not None
    assert self._joint_vel_clipped is not None

    # 将速度裁剪到转矩限制角速度范围内。
    vel_clipped = torch.clamp(
      self._joint_vel_clipped,
      min=-self._vel_at_effort_lim,
      max=self._vel_at_effort_lim,
    )

    # 计算转矩-速度曲线限制。
    torque_speed_top = self.saturation_effort * (
      1.0 - vel_clipped / self.velocity_limit_motor
    )
    torque_speed_bottom = self.saturation_effort * (
      -1.0 - vel_clipped / self.velocity_limit_motor
    )

    # 应用连续转矩约束。
    max_effort = torch.clamp(torque_speed_top, max=self.force_limit)
    min_effort = torch.clamp(torque_speed_bottom, min=-self.force_limit)

    return torch.clamp(effort, min=min_effort, max=max_effort)
