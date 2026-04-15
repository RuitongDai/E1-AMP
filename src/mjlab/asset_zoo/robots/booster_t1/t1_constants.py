"""Booster T1 humanoid constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

_T1_DIR: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "booster_t1"
)

T1_XML: Path = _T1_DIR / "xmls" / "booster_t1.xml"
T1_TRAIN_XML: Path = _T1_DIR / "xmls" / "booster_t1_train.xml"
assert T1_XML.exists(), f"T1 XML not found: {T1_XML}"
assert T1_TRAIN_XML.exists(), f"T1 train XML not found: {T1_TRAIN_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, _T1_DIR / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Full mesh XML for play/visualization."""
  spec = mujoco.MjSpec.from_file(str(T1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_spec_train() -> mujoco.MjSpec:
  """Lightweight XML for headless training (no meshes)."""
  return mujoco.MjSpec.from_file(str(T1_TRAIN_XML))


# ---------------------------------------------------------------------------
# Actuator configurations
# Stiffness/damping values tuned for PD control; effort limits from T1 URDF.
# ---------------------------------------------------------------------------
T1_ACT_HEAD_YAW = DcMotorActuatorCfg(
  target_names_expr=("AAHead_yaw",),
  stiffness=20.0,
  damping=1.5,
  effort_limit=7.0,
  saturation_effort=7.0,
  velocity_limit=5.0,
  armature=0.01,
)
T1_ACT_HEAD_PITCH = DcMotorActuatorCfg(
  target_names_expr=("Head_pitch",),
  stiffness=20.0,
  damping=1.5,
  effort_limit=7.0,
  saturation_effort=7.0,
  velocity_limit=5.0,
  armature=0.01,
)
T1_ACT_SHOULDER_PITCH = DcMotorActuatorCfg(
  target_names_expr=(r".*_Shoulder_Pitch",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=18.0,
  saturation_effort=18.0,
  velocity_limit=6.5,
  armature=0.05,
)
T1_ACT_SHOULDER_ROLL = DcMotorActuatorCfg(
  target_names_expr=(r".*_Shoulder_Roll",),
  stiffness=30.0,
  damping=1.5,
  effort_limit=18.0,
  saturation_effort=18.0,
  velocity_limit=6.5,
  armature=0.01,
)
T1_ACT_ELBOW_PITCH = DcMotorActuatorCfg(
  target_names_expr=(r".*_Elbow_Pitch",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=18.0,
  saturation_effort=18.0,
  velocity_limit=8.0,
  armature=0.05,
)
T1_ACT_ELBOW_YAW = DcMotorActuatorCfg(
  target_names_expr=(r".*_Elbow_Yaw",),
  stiffness=15.0,
  damping=1.0,
  effort_limit=18.0,
  saturation_effort=18.0,
  velocity_limit=8.0,
  armature=0.01,
)
T1_ACT_WAIST = DcMotorActuatorCfg(
  target_names_expr=("Waist",),
  stiffness=80.0,
  damping=4.0,
  effort_limit=25.0,
  saturation_effort=25.0,
  velocity_limit=5.75,
  armature=0.01,
)
T1_ACT_HIP_PITCH = DcMotorActuatorCfg(
  target_names_expr=(r".*_Hip_Pitch",),
  stiffness=120.0,
  damping=4.0,
  effort_limit=45.0,
  saturation_effort=45.0,
  velocity_limit=9.2,
  armature=0.01,
)
T1_ACT_HIP_ROLL = DcMotorActuatorCfg(
  target_names_expr=(r".*_Hip_Roll",),
  stiffness=80.0,
  damping=3.0,
  effort_limit=25.0,
  saturation_effort=25.0,
  velocity_limit=6.5,
  armature=0.01,
)
T1_ACT_HIP_YAW = DcMotorActuatorCfg(
  target_names_expr=(r".*_Hip_Yaw",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=25.0,
  saturation_effort=25.0,
  velocity_limit=5.75,
  armature=0.01,
)
T1_ACT_KNEE = DcMotorActuatorCfg(
  target_names_expr=(r".*_Knee_Pitch",),
  stiffness=140.0,
  damping=4.0,
  effort_limit=60.0,
  saturation_effort=60.0,
  velocity_limit=11.5,
  armature=0.01,
)
T1_ACT_ANKLE_PITCH = DcMotorActuatorCfg(
  target_names_expr=(r".*_Ankle_Pitch",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=24.0,
  saturation_effort=24.0,
  velocity_limit=5.75,
  armature=0.05,
)
T1_ACT_ANKLE_ROLL = DcMotorActuatorCfg(
  target_names_expr=(r".*_Ankle_Roll",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=15.0,
  saturation_effort=15.0,
  velocity_limit=5.75,
  armature=0.05,
)

# ---- Initial standing pose ----
T1_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.7),
  joint_pos={
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "Left_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": 0.3,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": 0.0,
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": -0.3,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.0,
    "Waist": 0.0,
    "Left_Hip_Pitch": -0.4,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.8,
    "Left_Ankle_Pitch": -0.4,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.4,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.8,
    "Right_Ankle_Pitch": -0.4,
    "Right_Ankle_Roll": 0.0,
  },
  joint_vel={".*": 0.0},
)

# ---- Collision config ----
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^[lr]_foot_collision$": 3, ".*_collision": 1},
  priority={r"^[lr]_foot_collision$": 1},
  friction={r"^[lr]_foot_collision$": (0.6,)},
)

# ---- Articulation config ----
T1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    T1_ACT_HEAD_YAW,
    T1_ACT_HEAD_PITCH,
    T1_ACT_SHOULDER_PITCH,
    T1_ACT_SHOULDER_ROLL,
    T1_ACT_ELBOW_PITCH,
    T1_ACT_ELBOW_YAW,
    T1_ACT_WAIST,
    T1_ACT_HIP_PITCH,
    T1_ACT_HIP_ROLL,
    T1_ACT_HIP_YAW,
    T1_ACT_KNEE,
    T1_ACT_ANKLE_PITCH,
    T1_ACT_ANKLE_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_t1_robot_cfg(play: bool = False) -> EntityCfg:
  return EntityCfg(
    init_state=T1_INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec if play else get_spec_train,
    articulation=T1_ARTICULATION,
  )


T1_ACTION_SCALE: dict[str, float] = {}
for _a in T1_ARTICULATION.actuators:
  assert isinstance(_a, DcMotorActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    T1_ACTION_SCALE[_n] = _e / _s
