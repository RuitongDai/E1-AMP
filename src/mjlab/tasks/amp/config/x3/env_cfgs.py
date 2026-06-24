"""Booster T1 AMP 环境配置文件。

该模块定义了T1人形机器人的AMP（Adversarial Motion Priors）环境配置，
包括观测、动作、命令、奖励、终止条件等。
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.asset_zoo.robots import T1_ACTION_SCALE, get_t1_robot_cfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.rewards import joint_pos_limits
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.amp import mdp
from mjlab.tasks.amp.amp_env import AmpEnvCfg
from mjlab.tasks.amp.managers import AnimationTermCfg, MotionDataTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

# 运动数据目录路径
MOTION_DATA_DIR = str(
  Path(__file__).resolve().parent.parent.parent / "data" / "t1_motions"
)

# AMP 模型使用的历史步数
AMP_NUM_STEPS = 3
# 动画项的名称
ANIMATION_TERM_NAME = "t1_anim"
# 运动数据项的名称
MOTION_DATA_TERM_NAME = "t1_motion"


def t1_amp_env_cfg(play: bool = False) -> AmpEnvCfg:
  """为 Booster T1 创建 AMP 环境配置。
  Args:
    play: 是否为播放模式（true则运行时间无限长）
  Returns:
    AmpEnvCfg: T1机器人的完整环境配置
  """
  # 脚部碰撞体几何体名称
  foot_geoms = ("l_foot_collision", "r_foot_collision")

  # 脚部与地面接触传感器配置
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # 非脚部与地面接触传感器配置
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision$",
      exclude=foot_geoms,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # 演员网络观测项（用于策略网络）
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  # 评论家网络观测项（用于价值网络）
  critic_terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  # 判别器网络观测项（用于动作判别）
  disc_terms = {
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel),
  }

  # 判别器演示观测项（参考动作数据）
  disc_demo_terms = {
    "ref_root_ang_vel_b": ObservationTermCfg(
      func=mdp.ref_root_ang_vel_b,
      params={"animation": ANIMATION_TERM_NAME},
    ),
    "ref_joint_pos": ObservationTermCfg(
      func=mdp.ref_joint_pos,
      params={"animation": ANIMATION_TERM_NAME},
    ),
    "ref_joint_vel": ObservationTermCfg(
      func=mdp.ref_joint_vel,
      params={"animation": ANIMATION_TERM_NAME},
    ),
  }

  # 观测配置分组
  observations = {
    # 演员网络（策略网络）观测
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    # 评论家网络（价值网络）观测
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    # 判别器网络观测
    "disc": ObservationGroupCfg(
      terms=disc_terms,
      concatenate_terms=True,
      concatenate_dim=-1,
      enable_corruption=False,
      history_length=AMP_NUM_STEPS,
      flatten_history_dim=False,
    ),
    # 判别器演示观测
    "disc_demo": ObservationGroupCfg(
      terms=disc_demo_terms,
      concatenate_terms=True,
      concatenate_dim=-1,
      enable_corruption=False,
    ),
  }

  # 动作配置
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=T1_ACTION_SCALE,
      use_default_offset=True,
    ),
  }

  # 命令配置
  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.0,
      heading_command=False,
      heading_control_stiffness=0.5,
      debug_vis=False,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.7),
        lin_vel_y=(-0.2, 0.2),
        ang_vel_z=(-0.8, 0.8),
      ),
    ),
  }

  # 事件配置
  events = {
    # 重置机器人基座位置和姿态
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    # 重置机器人关节位置和速度
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  # 脚部身体名称
  foot_bodies = ("left_foot_link", "right_foot_link")

  # 奖励配置
  rewards = {
    # ============ 任务相关奖励 ============
    # 线速度跟踪奖励
    "track_lin_vel_xy": RewardTermCfg(
      func=mdp.track_lin_vel_xy_exp,
      weight=1.25,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    # 角速度跟踪奖励
    "track_ang_vel_z": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp,
      weight=1.25,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    # 机器人活跃奖励
    "is_alive": RewardTermCfg(func=mdp.is_alive, weight=0.15),

    # ============ 基座惩罚 ============
    # 基座 XY 平面角速度惩罚
    "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.1),
    # 平坦身体姿态惩罚
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=-1.0,
    ),

    # ============ 关节惩罚 ============
    # 关节速度惩罚
    "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-2e-4),
    # 关节加速度惩罚
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    # 动作变化率惩罚
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    # 关节位置限制惩罚
    "joint_pos_limits": RewardTermCfg(
      func=joint_pos_limits,
      weight=-1.0,
    ),
    # 关节能量惩罚
    "joint_energy": RewardTermCfg(func=mdp.joint_energy, weight=-1e-4),
    # 关节扭矩惩罚
    "joint_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2, weight=-1e-5
    ),

    # ============ 脚部惩罚 ============
    # 脚部滑动惩罚
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide,
      weight=-0.1,
      params={
        "sensor_name": feet_ground_cfg.name,
        "asset_cfg": SceneEntityCfg("robot", body_names=foot_bodies),
      },
    ),
    # 柔和着陆奖励
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-5e-5,
      params={"sensor_name": feet_ground_cfg.name},
    ),

    # ============ 接触惩罚 ============
    # 不期望的接触惩罚
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-10.0,
      params={
        "sensor_name": nonfoot_ground_cfg.name,
        "force_threshold": 1.0,
      },
    ),
  }

  # 终止条件配置
  terminations = {
    # 超时终止
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # 摔倒终止
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    # 非法接触终止
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={
        "sensor_name": nonfoot_ground_cfg.name,
        "force_threshold": 10.0,
      },
    ),
  }

  # 运动数据配置
  motion_data = {
    MOTION_DATA_TERM_NAME: MotionDataTermCfg(
      motion_data_dir=MOTION_DATA_DIR,
      motion_data_weights={
        "female_walk1": 1.0,
        "female_stand_to_walk": 1.0,
        "female_walk_to_stand": 1.0,
        "female_walk_backwards": 1.0,
        "female_walk_turn_left_45": 1.0,
        "female_walk_turn_left_90": 1.0,
        "female_walk_turn_right_45": 1.0,
        "female_walk_turn_right_90": 1.0,
      },
    )
  }

  # 动画配置
  animation = {
    ANIMATION_TERM_NAME: AnimationTermCfg(
      motion_data_term=MOTION_DATA_TERM_NAME,
      motion_data_components=[
        "root_pos_w",
        "root_quat",
        "root_vel_w",
        "root_ang_vel_w",
        "dof_pos",
        "dof_vel",
        "key_body_pos_b",
      ],
      num_steps_to_use=AMP_NUM_STEPS,
      random_initialize=True,
      random_fetch=True,
    )
  }

  # 创建完整的环境配置
  cfg = AmpEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      sensors=(feet_ground_cfg, nonfoot_ground_cfg),
      num_envs=4096,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics={},
    motion_data=motion_data,
    animation=animation,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="Trunk",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=500,
      nconmax=None,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
      contact_sensor_maxmatch=64,
    ),
    decimation=4,
    episode_length_s=20.0,
  )

  # 为机器人设置实体配置
  cfg.scene.entities = {"robot": get_t1_robot_cfg(play=play)}

  # 播放模式设置
  if play:
    # 设置无限长的运行时间
    cfg.episode_length_s = int(1e9)
    # 禁用观测噪声
    cfg.observations["actor"].enable_corruption = False
    # 移除推动机器人事件
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    # 调整速度命令范围为固定前向速度
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges = UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.3, 0.3),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(0.0, 0.0),
    )
    twist_cmd.rel_standing_envs = 0.0
    twist_cmd.resampling_time_range = (1e9, 1e9)

  return cfg
