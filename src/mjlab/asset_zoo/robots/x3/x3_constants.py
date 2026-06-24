"""Booster T1 人形机器人常数定义。

包含 T1 机器人的模型文件、执行器配置、初始姿态、碰撞配置等常数。
"""

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

# T1 机器人完整模型 XML 文件（包含网格）
T1_XML: Path = _T1_DIR / "xmls" / "booster_t1.xml"
# T1 机器人训练模型 XML 文件（轻量级，无网格）
T1_TRAIN_XML: Path = _T1_DIR / "xmls" / "booster_t1_train.xml"
assert T1_XML.exists(), f"T1 XML not found: {T1_XML}"
assert T1_TRAIN_XML.exists(), f"T1 train XML not found: {T1_TRAIN_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  """获取 T1 机器人的网格资源。

  Args:
    meshdir: 网格目录路径

  Returns:
    网格资源字典，键为网格文件名，值为字节数据
  """
  assets: dict[str, bytes] = {}
  update_assets(assets, _T1_DIR / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """获取完整模型规范，用于播放和可视化。

  包含完整网格数据，适合渲染和交互。

  Returns:
    包含网格的 MuJoCo 模型规范
  """
  spec = mujoco.MjSpec.from_file(str(T1_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_spec_train() -> mujoco.MjSpec:
  """获取轻量级训练模型规范，无网格渲染。

  用于无头训练，加载速度快，内存占用小。

  Returns:
    轻量级的 MuJoCo 模型规范（无网格）
  """
  return mujoco.MjSpec.from_file(str(T1_TRAIN_XML))


# ============================================================================
# 执行器配置
# 刚度/阻尼值针对 PD 控制优化，力矩限制来自 T1 URDF 文件
# ============================================================================

# 头部偏航执行器配置
T1_ACT_HEAD_YAW = DcMotorActuatorCfg(
  target_names_expr=("AAHead_yaw",),
  stiffness=20.0,
  damping=1.5,
  effort_limit=7.0,
  saturation_effort=7.0,
  velocity_limit=5.0,
  armature=0.01,
)

# 头部俯仰执行器配置
T1_ACT_HEAD_PITCH = DcMotorActuatorCfg(
  target_names_expr=("Head_pitch",),
  stiffness=20.0,
  damping=1.5,
  effort_limit=7.0,
  saturation_effort=7.0,
  velocity_limit=5.0,
  armature=0.01,
)

# 肩膀俯仰执行器配置
T1_ACT_SHOULDER_PITCH = DcMotorActuatorCfg(
  target_names_expr=(r".*_Shoulder_Pitch",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=18.0,
  saturation_effort=18.0,
  velocity_limit=6.5,
  armature=0.05,
)

# 肩膀外展执行器配置
T1_ACT_SHOULDER_ROLL = DcMotorActuatorCfg(
  target_names_expr=(r".*_Shoulder_Roll",),
  stiffness=30.0,
  damping=1.5,
  effort_limit=18.0,
  saturation_effort=18.0,
  velocity_limit=6.5,
  armature=0.01,
)

# 肘部俯仰执行器配置
T1_ACT_ELBOW_PITCH = DcMotorActuatorCfg(
  target_names_expr=(r".*_Elbow_Pitch",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=18.0,
  saturation_effort=18.0,
  velocity_limit=8.0,
  armature=0.05,
)

# 肘部偏航执行器配置
T1_ACT_ELBOW_YAW = DcMotorActuatorCfg(
  target_names_expr=(r".*_Elbow_Yaw",),
  stiffness=15.0,
  damping=1.0,
  effort_limit=18.0,
  saturation_effort=18.0,
  velocity_limit=8.0,
  armature=0.01,
)

# 腰部执行器配置
T1_ACT_WAIST = DcMotorActuatorCfg(
  target_names_expr=("Waist",),
  stiffness=80.0,
  damping=4.0,
  effort_limit=25.0,
  saturation_effort=25.0,
  velocity_limit=5.75,
  armature=0.01,
)

# 髋关节俯仰执行器配置
T1_ACT_HIP_PITCH = DcMotorActuatorCfg(
  target_names_expr=(r".*_Hip_Pitch",),
  stiffness=120.0,
  damping=4.0,
  effort_limit=45.0,
  saturation_effort=45.0,
  velocity_limit=9.2,
  armature=0.01,
)

# 髋关节外展执行器配置
T1_ACT_HIP_ROLL = DcMotorActuatorCfg(
  target_names_expr=(r".*_Hip_Roll",),
  stiffness=80.0,
  damping=3.0,
  effort_limit=25.0,
  saturation_effort=25.0,
  velocity_limit=6.5,
  armature=0.01,
)

# 髋关节偏航执行器配置
T1_ACT_HIP_YAW = DcMotorActuatorCfg(
  target_names_expr=(r".*_Hip_Yaw",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=25.0,
  saturation_effort=25.0,
  velocity_limit=5.75,
  armature=0.01,
)

# 膝关节执行器配置
T1_ACT_KNEE = DcMotorActuatorCfg(
  target_names_expr=(r".*_Knee_Pitch",),
  stiffness=140.0,
  damping=4.0,
  effort_limit=60.0,
  saturation_effort=60.0,
  velocity_limit=11.5,
  armature=0.01,
)

# 踝关节俯仰执行器配置
T1_ACT_ANKLE_PITCH = DcMotorActuatorCfg(
  target_names_expr=(r".*_Ankle_Pitch",),
  stiffness=60.0,
  damping=3.0,
  effort_limit=24.0,
  saturation_effort=24.0,
  velocity_limit=5.75,
  armature=0.05,
)

# 踝关节外展执行器配置
T1_ACT_ANKLE_ROLL = DcMotorActuatorCfg(
  target_names_expr=(r".*_Ankle_Roll",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=15.0,
  saturation_effort=15.0,
  velocity_limit=5.75,
  armature=0.05,
)

# 初始站立姿态配置
T1_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.7),
  joint_pos={
    # 头部关节
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    # 左臂关节
    "Left_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": 0.3,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": 0.0,
    # 右臂关节
    "Right_Shoulder_Pitch": 0.0,
    "Right_Shoulder_Roll": -0.3,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.0,
    # 腰部关节
    "Waist": 0.0,
    # 左腿关节
    "Left_Hip_Pitch": -0.4,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.8,
    "Left_Ankle_Pitch": -0.4,
    "Left_Ankle_Roll": 0.0,
    # 右腿关节
    "Right_Hip_Pitch": -0.4,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.8,
    "Right_Ankle_Pitch": -0.4,
    "Right_Ankle_Roll": 0.0,
  },
  joint_vel={".*": 0.0},
)

# 碰撞配置
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  # 左右脚底面接触维度为 3，其他部分为 1
  condim={r"^[lr]_foot_collision$": 3, ".*_collision": 1},
  # 脚底接触优先级设置为 1
  priority={r"^[lr]_foot_collision$": 1},
  # 脚底摩擦系数
  friction={r"^[lr]_foot_collision$": (0.6,)},
)

# 机器人关节配置
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
  # 软限制因子，防止关节超出范围
  soft_joint_pos_limit_factor=0.9,
)


def get_t1_robot_cfg(play: bool = False) -> EntityCfg:
  """获取 T1 机器人配置。
  Args:
    play: 是否为播放模式。True 使用完整模型（含网格），False 使用轻量级训练模型
  Returns:
    T1 机器人的完整实体配置
  """
  return EntityCfg(
    init_state=T1_INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec if play else get_spec_train,
    articulation=T1_ARTICULATION,
  )


# 计算动作缩放因子（力矩限制 / 刚度）
T1_ACTION_SCALE: dict[str, float] = {}
for _a in T1_ARTICULATION.actuators:
  assert isinstance(_a, DcMotorActuatorCfg)
  _e = _a.effort_limit
  _s = _a.stiffness
  _names = _a.target_names_expr
  assert _e is not None
  for _n in _names:
    # 动作缩放 = 力矩限制 / 刚度，用于归一化动作输入
    T1_ACTION_SCALE[_n] = _e / _s
