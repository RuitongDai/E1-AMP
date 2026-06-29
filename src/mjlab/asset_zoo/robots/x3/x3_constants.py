"""Moya01 V2 (X3) 人形机器人常数定义。
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

_X3_DIR: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "x3"

# X3 机器人模型 XML 文件
X3_XML: Path = _X3_DIR / "xmls" / "Moya01_V2.xml"
# X3 无网格训练 XML 文件
X3_TRAIN_XML: Path = _X3_DIR / "xmls" / "Moya01_V2.xml"

assert X3_XML.exists(), f"X3 XML not found: {X3_XML}"

def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, _X3_DIR / "assets", meshdir)
  return assets

def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(X3_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

def get_spec_train() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(X3_TRAIN_XML))

# ============================================================================
# 执行器配置
# ============================================================================

X3_ACT_LEGS = DcMotorActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_knee_joint"
  ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=75.0,
  saturation_effort=75.0,
  velocity_limit=15.0,
  armature=0.01,
)

X3_ACT_HIPS_PITCH = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=75.0,
  saturation_effort=75.0,
  velocity_limit=15.0,
  armature=0.01,
)

X3_ACT_FEET = DcMotorActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=75.0,
  saturation_effort=75.0,
  velocity_limit=10.0,
  armature=0.01,
)

X3_ACT_WAIST = DcMotorActuatorCfg(
  target_names_expr=("waist_roll_joint", "waist_yaw_joint"),
  stiffness=100.0,
  damping=2.0,
  effort_limit=50.0,
  saturation_effort=50.0,
  velocity_limit=10.0,
  armature=0.01,
)

X3_ACT_SHOULDER = DcMotorActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
  ),
  stiffness=30.0,
  damping=2.0,
  effort_limit=25.0,
  saturation_effort=25.0,
  velocity_limit=10.0,
  armature=0.008,
)

X3_ACT_FORE_ARM = DcMotorActuatorCfg(
  target_names_expr=(".*_elbow_joint", ".*_wrist_roll_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=25.0,
  saturation_effort=25.0,
  velocity_limit=10.0,
  armature=0.005,
)

X3_ACT_HAND = DcMotorActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=20.0,
  damping=2.0,
  effort_limit=5.0,
  saturation_effort=5.0,
  velocity_limit=10.0,
  armature=0.005,
)

# ============================================================================
# 初始站立姿态配置
# ============================================================================
X3_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.80),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.2,
    ".*_ankle_pitch_joint": -0.1,
    ".*_elbow_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# ============================================================================
# 碰撞配置
# ============================================================================
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  # 左右脚底面接触维度为 3，其他部分为 1
  condim={r"^[lr]_foot_collision$": 3, ".*_collision": 1},
  # 脚底接触优先级设置为 1
  priority={r"^[lr]_foot_collision$": 1},
  # 脚底摩擦系数
  friction={r"^[lr]_foot_collision$": (0.6,)},
)

# ============================================================================
# 机器人关节配置
# ============================================================================
X3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    X3_ACT_LEGS,
    X3_ACT_HIPS_PITCH,
    X3_ACT_FEET,
    X3_ACT_WAIST,
    X3_ACT_SHOULDER,
    X3_ACT_FORE_ARM,
    X3_ACT_HAND,
  ),
  soft_joint_pos_limit_factor=0.9,
)

def get_x3_robot_cfg(play: bool = False) -> EntityCfg:
  """获取 X3 机器人配置"""
  return EntityCfg(
    init_state=X3_INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec if play else get_spec_train,
    articulation=X3_ARTICULATION,
  )

# ============================================================================
# 计算动作缩放因子
# ============================================================================
X3_ACTION_SCALE: dict[str, float] = {}
for _a in X3_ARTICULATION.actuators:
  assert isinstance(_a, DcMotorActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    X3_ACTION_SCALE[_n] = _e / _s