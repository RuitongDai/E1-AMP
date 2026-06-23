"""E1 人形机器人常数定义。

包含 E1 机器人的模型文件、执行器配置、初始姿态、碰撞配置等常数。
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

_E1_DIR: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "e1"
)

E1_XML: Path = _E1_DIR / "xmls" / "E1_25dof.xml"
E1_TRAIN_XML: Path = _E1_DIR / "xmls" / "E1_25dof_train.xml"
assert E1_XML.exists(), f"E1 XML not found: {E1_XML}"
assert E1_TRAIN_XML.exists(), f"E1 train XML not found: {E1_TRAIN_XML}"

def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, _E1_DIR / "meshes", meshdir)
  return assets

def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(E1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

def get_spec_train() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(E1_TRAIN_XML))

# ============================================================================
# 执行器配置 (Actuator Configs) - 使用 BuiltinPositionActuatorCfg
# ============================================================================
E1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=200.0,
  damping=5.0,
  effort_limit=60.0,
  armature=0.01,
)

E1_ACTUATOR_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_pitch_joint",
  ),
  stiffness=150.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.01,
)

E1_ACTUATOR_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_roll_joint",
  ),
  stiffness=100.0,
  damping=3.0,
  effort_limit=60.0,
  armature=0.01,
)

E1_ACTUATOR_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
  ),
  stiffness=100.0,
  damping=3.0,
  effort_limit=36.0,
  armature=0.01,
)

E1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_knee_joint",
  ),
  stiffness=150.0,
  damping=5.0,
  effort_limit=120.0,
  armature=0.01,
)

E1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
  ),
  stiffness=30.0,
  damping=2.0,
  effort_limit=30.0,
  armature=0.01,
)

E1_ACTUATOR_SHOULDER_PITCH_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_elbow_joint",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=60.0,
  armature=0.01,
)

E1_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_roll_joint",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=36.0,
  armature=0.01,
)

E1_ACTUATOR_SHOULDER_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_yaw_joint",
  ),
  stiffness=30.0,
  damping=2.0,
  effort_limit=15.0,
  armature=0.01,
)

E1_ACTUATOR_WRIST_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_roll_joint",
  ),
  stiffness=20.0,
  damping=0.5,
  effort_limit=15.0,
  armature=0.01,
)

E1_ACTUATOR_WRIST_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_pitch_joint",
  ),
  stiffness=10.0,
  damping=0.5,
  effort_limit=6.0,
  armature=0.01,
)

# ============================================================================
# 初始站立姿态配置
# ============================================================================
E1_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.67),
  joint_pos={
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.3,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,

    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": -0.3,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,

    "waist_yaw_joint": 0.0,

    "left_hip_pitch_joint": -0.4,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.8,
    "left_ankle_pitch_joint": -0.4,
    "left_ankle_roll_joint": 0.0,

    "right_hip_pitch_joint": -0.4,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.8,
    "right_ankle_pitch_joint": -0.4,
    "right_ankle_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# ============================================================================
# 碰撞配置
# ============================================================================
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^[lr]_foot_collision$": 3, ".*_collision": 1},
  priority={r"^[lr]_foot_collision$": 1},
  friction={r"^[lr]_foot_collision$": (0.6,)},
)

# 机器人关节组装
E1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    E1_ACTUATOR_WAIST,
    E1_ACTUATOR_PITCH,
    E1_ACTUATOR_ROLL,
    E1_ACTUATOR_YAW,
    E1_ACTUATOR_KNEE,
    E1_ACTUATOR_ANKLE,
    E1_ACTUATOR_SHOULDER_PITCH_ELBOW,
    E1_ACTUATOR_SHOULDER_ROLL,
    E1_ACTUATOR_SHOULDER_YAW,
    E1_ACTUATOR_WRIST_ROLL,
    E1_ACTUATOR_WRIST_PITCH,
  ),
  soft_joint_pos_limit_factor=0.9,
)

def get_e1_robot_cfg(play: bool = False) -> EntityCfg:
  return EntityCfg(
    init_state=E1_INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec if play else get_spec_train,
    articulation=E1_ARTICULATION,
  )

# 计算动作缩放因子
E1_ACTION_SCALE: dict[str, float] = {}
for _a in E1_ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    E1_ACTION_SCALE[_n] = _e / _s