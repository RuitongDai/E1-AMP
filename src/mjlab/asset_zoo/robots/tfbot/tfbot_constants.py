"""TFBOT humanoid constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

TFBOT_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "tfbot" / "xmls" / "tfbot.xml"
)
assert TFBOT_XML.exists()


# TFBOT meshes are under ../urdf/meshes relative to xml.
def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, TFBOT_XML.parent.parent / "urdf" / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TFBOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


# Actuator specs are from the reference TFBOT config used in TienKung-Lab.
# Using DcMotorActuatorCfg for torque-velocity curve saturation model.
# saturation_effort = effort_limit (stall torque = continuous torque for these motors).
# velocity_limit values match tfbot-s2d URDF.
TFBOT_ACT_LEG_HIP_ROLL = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=100.0,
  damping=3.0,
  effort_limit=82.0,
  saturation_effort=82.0,
  velocity_limit=6.5,
  armature=0.01,
)
TFBOT_ACT_LEG_HIP_PITCH = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=120.0,
  damping=4.0,
  effort_limit=90.0,
  saturation_effort=90.0,
  velocity_limit=9.2,
  armature=0.01,
)
TFBOT_ACT_LEG_HIP_YAW = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint",),
  stiffness=90.0,
  damping=3.0,
  effort_limit=38.0,
  saturation_effort=38.0,
  velocity_limit=5.75,
  armature=0.01,
)
TFBOT_ACT_LEG_KNEE = DcMotorActuatorCfg(
  target_names_expr=(".*_knee_pitch_joint",),
  stiffness=140.0,
  damping=4.0,
  effort_limit=148.0,
  saturation_effort=148.0,
  velocity_limit=11.5,
  armature=0.01,
)
TFBOT_ACT_FOOT_ANKLE_PITCH = DcMotorActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=38.0,
  saturation_effort=38.0,
  velocity_limit=5.75,
  armature=0.01,
)
TFBOT_ACT_FOOT_ANKLE_ROLL = DcMotorActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=38.0,
  saturation_effort=38.0,
  velocity_limit=5.75,
  armature=0.01,
)
TFBOT_ACT_WAIST_ROLL = DcMotorActuatorCfg(
  target_names_expr=("waist_roll_joint",),
  stiffness=150.0,
  damping=4.0,
  effort_limit=76.0,
  saturation_effort=76.0,
  velocity_limit=5.75,
  armature=0.01,
)
TFBOT_ACT_WAIST_PITCH = DcMotorActuatorCfg(
  target_names_expr=("waist_pitch_joint",),
  stiffness=150.0,
  damping=4.0,
  effort_limit=76.0,
  saturation_effort=76.0,
  velocity_limit=5.75,
  armature=0.01,
)
TFBOT_ACT_WAIST_YAW = DcMotorActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=80.0,
  damping=4.0,
  effort_limit=38.0,
  saturation_effort=38.0,
  velocity_limit=5.75,
  armature=0.01,
)
TFBOT_ACT_SHOULDER_SHRUG = DcMotorActuatorCfg(
  target_names_expr=(".*_shoulder_shrug_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=12.0,
  saturation_effort=12.0,
  velocity_limit=10.0,
  armature=0.01,
)
TFBOT_ACT_SHOULDER_PITCH = DcMotorActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_joint",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=38.0,
  saturation_effort=38.0,
  velocity_limit=5.75,
  armature=0.01,
)
TFBOT_ACT_SHOULDER_ROLL = DcMotorActuatorCfg(
  target_names_expr=(".*_shoulder_roll_joint",),
  stiffness=20.0,
  damping=1.5,
  effort_limit=38.0,
  saturation_effort=38.0,
  velocity_limit=5.75,
  armature=0.01,
)
TFBOT_ACT_SHOULDER_YAW = DcMotorActuatorCfg(
  target_names_expr=(".*_shoulder_yaw_joint",),
  stiffness=10.0,
  damping=1.0,
  effort_limit=25.0,
  saturation_effort=25.0,
  velocity_limit=10.0,
  armature=0.01,
)
TFBOT_ACT_ELBOW = DcMotorActuatorCfg(
  target_names_expr=(".*_elbow_pitch_joint",),
  stiffness=10.0,
  damping=1.0,
  effort_limit=12.0,
  saturation_effort=12.0,
  velocity_limit=10.0,
  armature=0.01,
)
TFBOT_ACT_RADIUS = DcMotorActuatorCfg(
  target_names_expr=(".*_radius_roll_joint",),
  stiffness=5.0,
  damping=0.5,
  effort_limit=12.0,
  saturation_effort=12.0,
  velocity_limit=10.0,
  armature=0.01,
)
TFBOT_ACT_WRIST_YAW = DcMotorActuatorCfg(
  target_names_expr=(".*_wrist_yaw_joint",),
  stiffness=3.0,
  damping=0.3,
  effort_limit=5.0,
  saturation_effort=5.0,
  velocity_limit=5.0,
  armature=0.01,
)
TFBOT_ACT_WRIST_PITCH = DcMotorActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint",),
  stiffness=3.0,
  damping=0.3,
  effort_limit=5.0,
  saturation_effort=5.0,
  velocity_limit=5.0,
  armature=0.01,
)

TFBOT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.93),
  joint_pos={
    "r_hip_roll_joint": 0.0,
    "r_hip_pitch_joint": -0.5,
    "r_hip_yaw_joint": 0.0,
    "r_knee_pitch_joint": 1.0,
    "r_ankle_roll_joint": 0.0,
    "r_ankle_pitch_joint": -0.5,
    "l_hip_roll_joint": 0.0,
    "l_hip_pitch_joint": -0.5,
    "l_hip_yaw_joint": 0.0,
    "l_knee_pitch_joint": 1.0,
    "l_ankle_roll_joint": 0.0,
    "l_ankle_pitch_joint": -0.5,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "r_shoulder_shrug_joint": 0.0,
    "r_shoulder_pitch_joint": 0.0,
    "r_shoulder_roll_joint": -0.1,
    "r_shoulder_yaw_joint": 0.0,
    "r_elbow_pitch_joint": -0.3,
    "r_radius_roll_joint": 0.0,
    "r_wrist_yaw_joint": 0.0,
    "r_wrist_pitch_joint": 0.0,
    "l_shoulder_shrug_joint": 0.0,
    "l_shoulder_pitch_joint": 0.0,
    "l_shoulder_roll_joint": 0.1,
    "l_shoulder_yaw_joint": 0.0,
    "l_elbow_pitch_joint": -0.3,
    "l_radius_roll_joint": 0.0,
    "l_wrist_yaw_joint": 0.0,
    "l_wrist_pitch_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^[lr]_foot_collision$": 3, ".*_collision": 1},
  priority={r"^[lr]_foot_collision$": 1},
  friction={r"^[lr]_foot_collision$": (0.6,)},
)

TFBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    TFBOT_ACT_LEG_HIP_ROLL,
    TFBOT_ACT_LEG_HIP_PITCH,
    TFBOT_ACT_LEG_HIP_YAW,
    TFBOT_ACT_LEG_KNEE,
    TFBOT_ACT_FOOT_ANKLE_PITCH,
    TFBOT_ACT_FOOT_ANKLE_ROLL,
    TFBOT_ACT_WAIST_ROLL,
    TFBOT_ACT_WAIST_PITCH,
    TFBOT_ACT_WAIST_YAW,
    TFBOT_ACT_SHOULDER_SHRUG,
    TFBOT_ACT_SHOULDER_PITCH,
    TFBOT_ACT_SHOULDER_ROLL,
    TFBOT_ACT_SHOULDER_YAW,
    TFBOT_ACT_ELBOW,
    TFBOT_ACT_RADIUS,
    TFBOT_ACT_WRIST_YAW,
    TFBOT_ACT_WRIST_PITCH,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_tfbot_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=TFBOT_INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=TFBOT_ARTICULATION,
  )


TFBOT_ACTION_SCALE: dict[str, float] = {}
for a in TFBOT_ARTICULATION.actuators:
  assert isinstance(a, DcMotorActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    TFBOT_ACTION_SCALE[n] = e / s
