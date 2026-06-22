"""扩展ManagerBasedRlEnv的AMP(动作匹配运动)环境。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.amp.managers import (
  AnimationManager,
  AnimationTermCfg,
  MotionDataManager,
  MotionDataTermCfg,
)
from mjlab.utils.logging import print_info


@dataclass(kw_only=True)
class AmpEnvCfg(ManagerBasedRlEnvCfg):
  """AMP环境的配置。"""

  motion_data: dict[str, MotionDataTermCfg] = field(default_factory=dict)
  """运动数据项配置 (数据集名称 → 配置)。"""

  animation: dict[str, AnimationTermCfg] = field(default_factory=dict)
  """动画项配置 (项名称 → 配置)。"""


class AmpEnv(ManagerBasedRlEnv):
  """具有运动数据和动画管理器的ManagerBasedRlEnv，用于AMP任务。"""

  cfg: AmpEnvCfg

  def __init__(
    self,
    cfg: AmpEnvCfg,
    device: str,
    render_mode: str | None = None,
    **kwargs,
  ) -> None:
    super().__init__(cfg, device, render_mode, **kwargs)

  def load_managers(self) -> None:
    # 运动数据和动画管理器必须在观测之前准备好
    # (观测项可能引用动画数据)。
    self.motion_data_manager = MotionDataManager(
      self.cfg.motion_data, self.num_envs, self.device
    )
    print_info(f"[INFO] MotionDataManager: {self.motion_data_manager.active_terms}")
    self.animation_manager = AnimationManager(self.cfg.animation, self)
    print_info(f"[INFO] AnimationManager: {self.animation_manager.active_terms}")

    # 现在加载标准管理器 (事件、命令、动作、观测等)。
    super().load_managers()

  def step(self, action: torch.Tensor):
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    # 更新新时间步的动画参考。
    self.animation_manager.update(dt=self.step_dt)

    # 环境计数器。
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # 终止条件和奖励。
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    # 重置已终止的环境。
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    # 所有环境的正向运动学。
    self.sim.forward()

    self.command_manager.compute(dt=self.step_dt)

    if "step" in self.event_manager.available_modes:
      self.event_manager.apply(mode="step", dt=self.step_dt)
    if "interval" in self.event_manager.available_modes:
      self.event_manager.apply(mode="interval", dt=self.step_dt)

    self.sim.sense()
    self.obs_buf = self.observation_manager.compute(update_history=True)

    return (
      self.obs_buf,
      self.reward_buf,
      self.reset_terminated,
      self.reset_time_outs,
      self.extras,
    )

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    # 在标准重置之前重置动画管理器，这样当观测管理器重置其历史时，
    # 它会拾取新的参考数据。
    if env_ids is not None:
      self.animation_manager.reset(env_ids)
    super()._reset_idx(env_ids)
