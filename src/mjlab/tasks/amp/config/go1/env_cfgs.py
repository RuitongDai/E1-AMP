"""Unitree Go1 AMP environment configuration."""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.asset_zoo.robots import (
  GO1_ACTION_SCALE,
  get_go1_robot_cfg,
)
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
from mjlab.terrains import TerrainEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViewerConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOTION_DATA_DIR = str(
  Path(__file__).resolve().parent.parent.parent / "data" / "go1_motions"
)

AMP_NUM_STEPS = 3  # Number of reference steps kept in the animation buffer.
ANIMATION_TERM_NAME = "go1_anim"
MOTION_DATA_TERM_NAME = "go1_motion"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def unitree_go1_amp_env_cfg(play: bool = False) -> AmpEnvCfg:
  """Create an AMP environment configuration for Go1."""

  foot_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  # -- Sensors --------------------------------------------------------

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # -- Observations ---------------------------------------------------

  # Policy (actor) observations
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

  # Critic observations (superset of actor)
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

  # Discriminator observations (from the robot — used with history).
  # Use absolute joint_pos/joint_vel to match the reference motion data,
  # which stores absolute values.
  disc_terms = {
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel),
  }

  # Discriminator demo observations (from the reference motion)
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

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "disc": ObservationGroupCfg(
      terms=disc_terms,
      concatenate_terms=True,
      concatenate_dim=-1,
      enable_corruption=False,
      history_length=AMP_NUM_STEPS,
      flatten_history_dim=False,
    ),
    "disc_demo": ObservationGroupCfg(
      terms=disc_demo_terms,
      concatenate_terms=True,
      concatenate_dim=-1,
      enable_corruption=False,
      # No history for demo — the animation manager already
      # returns (N, S, D) with AMP_NUM_STEPS as S.
    ),
  }

  # -- Actions --------------------------------------------------------

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=GO1_ACTION_SCALE,
      use_default_offset=True,
    ),
  }

  # -- Commands -------------------------------------------------------

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
        lin_vel_x=(-1.0, 1.5),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-0.5, 0.5),
      ),
    ),
  }

  # -- Events ---------------------------------------------------------

  events = {
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

  # -- Rewards --------------------------------------------------------

  rewards = {
    "track_lin_vel_xy": RewardTermCfg(
      func=mdp.track_lin_vel_xy_exp,
      weight=1.25,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_ang_vel_z": RewardTermCfg(
      func=mdp.track_ang_vel_z_exp,
      weight=0.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "is_alive": RewardTermCfg(
      func=mdp.is_alive,
      weight=0.5,
    ),
    "lin_vel_z_l2": RewardTermCfg(
      func=mdp.lin_vel_z_l2,
      weight=-2.0,
    ),
    "ang_vel_xy_l2": RewardTermCfg(
      func=mdp.ang_vel_xy_l2,
      weight=-0.05,
    ),
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.flat_orientation_l2,
      weight=-0.2,
    ),
    "joint_vel_l2": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-5e-4,
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
    ),
    "joint_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=-5e-5,
    ),
  }

  # -- Terminations ---------------------------------------------------

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={
        "sensor_name": nonfoot_ground_cfg.name,
        "force_threshold": 10.0,
      },
    ),
  }

  # -- Motion data & animation ----------------------------------------

  motion_data = {
    MOTION_DATA_TERM_NAME: MotionDataTermCfg(
      motion_data_dir=MOTION_DATA_DIR,
      motion_data_weights={"go1_standing": 1.0},
    ),
  }

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
    ),
  }

  # -- Assemble -------------------------------------------------------

  cfg = AmpEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      sensors=(feet_ground_cfg, nonfoot_ground_cfg),
      num_envs=1,
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
      body_name="trunk",
      distance=1.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=300,
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

  cfg.scene.entities = {"robot": get_go1_robot_cfg()}

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  return cfg
