"""AMP-specific reward terms.

These follow the mjlab reward-term signature:
  func(env, **params) -> torch.Tensor  (shape: (num_envs,))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking commanded xy linear velocity (exponential kernel)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  lin_vel_error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
    dim=1,
  )
  return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking commanded yaw angular velocity (exponential kernel)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  ang_vel_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-ang_vel_error / std**2)


def track_lin_vel_xy_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared xy velocity tracking error (use with negative weight)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  return torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]),
    dim=1,
  )


def is_alive(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for being alive (not terminated)."""
  return (~env.termination_manager.terminated).float()


def lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize z-axis base linear velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize xy-axis base angular velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation via projected gravity xy."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_vel_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint velocities (L2 squared)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def joint_acc_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint accelerations (L2 squared)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize rate of change of actions."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action),
    dim=1,
  )


def joint_torques_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize applied joint torques (L2 squared)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.square(asset.data.actuator_force[:, asset_cfg.actuator_ids]), dim=1
  )


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from default joint positions."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.abs(asset.data.joint_pos - asset.data.default_joint_pos),
    dim=1,
  )


def joint_energy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint energy consumption (|torque| * |velocity|)."""
  asset: Entity = env.scene[asset_cfg.name]
  qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  qfrc = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot lateral velocity while in contact with ground."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # [B, N, 3]
  root_vel_w = asset.data.root_link_lin_vel_w.unsqueeze(1)  # [B, 1, 3]
  rel_vel = foot_vel_w - root_vel_w  # [B, N, 3]
  # Transform to body frame and take xy (lateral) components.
  body_ids = asset_cfg.body_ids
  assert isinstance(body_ids, list)
  n_feet = len(body_ids)
  root_quat = asset.data.root_link_quat_w  # [B, 4]
  vel_body = torch.zeros(
    env.num_envs, n_feet, 3, device=env.device, dtype=rel_vel.dtype
  )
  for i in range(n_feet):
    vel_body[:, i, :] = quat_apply_inverse(root_quat, rel_vel[:, i, :])
  lateral_speed = torch.norm(vel_body[:, :, :2], dim=-1)  # [B, N]
  return torch.sum(lateral_speed * in_contact, dim=1)


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize high impact forces at first contact (sound suppression)."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.force is not None
  forces = contact_sensor.data.force  # [B, N, 3]
  force_mag = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = torch.square(force_mag) * first_contact.float()  # [B, N]
  return torch.sum(landing_impact, dim=1)


def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize contacts above a force threshold."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)
