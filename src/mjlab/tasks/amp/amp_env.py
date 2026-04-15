"""AMP environment extending ManagerBasedRlEnv."""

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
  """Configuration for an AMP environment."""

  motion_data: dict[str, MotionDataTermCfg] = field(default_factory=dict)
  """Motion data term configurations (dataset name → cfg)."""

  animation: dict[str, AnimationTermCfg] = field(default_factory=dict)
  """Animation term configurations (term name → cfg)."""


class AmpEnv(ManagerBasedRlEnv):
  """ManagerBasedRlEnv with motion data and animation managers for AMP."""

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
    # Motion data and animation managers must be ready before observations
    # (observation terms may reference animation data).
    self.motion_data_manager = MotionDataManager(
      self.cfg.motion_data, self.num_envs, self.device
    )
    print_info(f"[INFO] MotionDataManager: {self.motion_data_manager.active_terms}")
    self.animation_manager = AnimationManager(self.cfg.animation, self)
    print_info(f"[INFO] AnimationManager: {self.animation_manager.active_terms}")

    # Now load standard managers (events, commands, actions, obs, …).
    super().load_managers()

  def step(self, action: torch.Tensor):
    self.action_manager.process_action(action.to(self.device))

    for _ in range(self.cfg.decimation):
      self._sim_step_counter += 1
      self.action_manager.apply_action()
      self.scene.write_data_to_sim()
      self.sim.step()
      self.scene.update(dt=self.physics_dt)

    # Update animation references for the new timestep.
    self.animation_manager.update(dt=self.step_dt)

    # Env counters.
    self.episode_length_buf += 1
    self.common_step_counter += 1

    # Terminations and rewards.
    self.reset_buf = self.termination_manager.compute()
    self.reset_terminated = self.termination_manager.terminated
    self.reset_time_outs = self.termination_manager.time_outs
    self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
    self.metrics_manager.compute()

    # Reset terminated envs.
    reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
      self._reset_idx(reset_env_ids)
      self.scene.write_data_to_sim()

    # Forward kinematics for all envs.
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
    # Reset animation manager before standard reset so that when observation
    # manager resets its history, it picks up fresh reference data.
    if env_ids is not None:
      self.animation_manager.reset(env_ids)
    super()._reset_idx(env_ids)
