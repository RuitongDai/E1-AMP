"""
python scripts/sim2sim_x3_mjlab.py \
  --checkpoint logs/rsl_rl/x3_amp/2026-xxxx/model_22500.pt \
  --duration 120.0
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.x3.x3_constants import (
  X3_ACTION_SCALE,
  X3_ARTICULATION,
  X3_INIT_STATE,
  X3_XML,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

# =============================================================================
# 1. 从 X3 constants 中提取电机参数
# =============================================================================
def _build_motor_params() -> tuple[
  list[str],
  list[str],
  np.ndarray,
  np.ndarray,
  np.ndarray,
  np.ndarray,
  np.ndarray,
  dict[str, float],
]:
  """
  从 XML 和 X3_ARTICULATION 中构造电机参数。
  返回：
    xml_joints:
      XML 中的关节顺序。
    act_joints:
      actuator 创建顺序。
    kp, kd，effort_limit, saturation_effort, velocity_limit，arm_map
  """
  spec = mujoco.MjSpec.from_file(str(X3_XML))

  # ---------------------------------------------------------------------------
  # 1.1 提取 XML 拓扑顺序下的 hinge joints
  # policy obs/action 一般是按照 XML DFS 关节顺序来的，而不是 actuator 定义顺序。
  # ---------------------------------------------------------------------------
  xml_joints: list[str] = []
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_HINGE:
      xml_joints.append(jnt.name)

  # ---------------------------------------------------------------------------
  # 1.2 按 constants 里的 actuator 配置创建 act_joints
  # ---------------------------------------------------------------------------
  act_joints: list[str] = []
  kp_list: list[float] = []
  kd_list: list[float] = []
  eff_list: list[float] = []
  sat_list: list[float] = []
  vel_list: list[float] = []
  arm_map: dict[str, float] = {}

  seen: set[str] = set()

  for act_cfg in X3_ARTICULATION.actuators:
    assert isinstance(act_cfg, DcMotorActuatorCfg)

    for pattern in act_cfg.target_names_expr:
      for jname in xml_joints:
        if not re.fullmatch(pattern, jname):
          continue

        if jname in seen:
          raise RuntimeError(
            f"关节 {jname} 被多个 actuator pattern 匹配到了，请检查 X3_ARTICULATION。"
          )

        seen.add(jname)
        act_joints.append(jname)

        kp_list.append(float(act_cfg.stiffness))
        kd_list.append(float(act_cfg.damping))
        eff_list.append(float(act_cfg.effort_limit))
        sat_list.append(float(act_cfg.saturation_effort))
        vel_list.append(float(act_cfg.velocity_limit))

        if act_cfg.armature is not None:
          arm_map[jname] = float(act_cfg.armature)

  if len(act_joints) != len(xml_joints):
    missing = [n for n in xml_joints if n not in seen]
    raise RuntimeError(
      "actuator 没有覆盖所有 XML hinge joints。\n"
      f"XML joints 数量: {len(xml_joints)}\n"
      f"Actuated joints 数量: {len(act_joints)}\n"
      f"缺失关节: {missing}"
    )

  return (
    xml_joints,
    act_joints,
    np.asarray(kp_list, dtype=np.float64),
    np.asarray(kd_list, dtype=np.float64),
    np.asarray(eff_list, dtype=np.float64),
    np.asarray(sat_list, dtype=np.float64),
    np.asarray(vel_list, dtype=np.float64),
    arm_map,
  )


XML_JOINTS, ACT_JOINTS, KP, KD, EFFORT_LIMIT, SATURATION_EFFORT, VELOCITY_LIMIT, ARMATURE_MAP = _build_motor_params()

NUM_DOF = len(XML_JOINTS)

# DC 电机模型中，用于 torque-speed 曲线的速度截断范围。
VEL_AT_EFFORT_LIM = VELOCITY_LIMIT * (1.0 + EFFORT_LIMIT / SATURATION_EFFORT)

# =============================================================================
# 2. DC 电机力矩限幅
# =============================================================================
def dc_motor_clip_effort(effort: np.ndarray, vel: np.ndarray) -> np.ndarray:
  """
  复刻 mjlab DcMotorActuator 的 torque-speed 饱和曲线。

  直流电机的特点：
    - 低速时可以输出较大力矩。
    - 转速越高，能输出的最大力矩越小。
    - 正转/反转方向的力矩上下限都和当前速度有关。

  Args:
    effort:
      PD 算出来的原始力矩。

    vel:
      当前关节速度。

  Returns:
    被 DC motor 曲线和 effort_limit 限幅后的力矩。
  """
  vel_clipped = np.clip(vel, -VEL_AT_EFFORT_LIM, VEL_AT_EFFORT_LIM)

  torque_top = SATURATION_EFFORT * (1.0 - vel_clipped / VELOCITY_LIMIT)
  torque_bottom = SATURATION_EFFORT * (-1.0 - vel_clipped / VELOCITY_LIMIT)

  max_effort = np.minimum(torque_top, EFFORT_LIMIT)
  min_effort = np.maximum(torque_bottom, -EFFORT_LIMIT)

  return np.clip(effort, min_effort, max_effort)


# =============================================================================
# 3. 四元数工具
# =============================================================================
def quat_rotate_inverse_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
  """
  用逆四元数把世界系向量旋转到 base/body 系。

  在 observation 里 projected_gravity 需要表达为机器人 base 坐标系下的重力方向。

  Args:
    q_wxyz:
      MuJoCo root quaternion，格式为 wxyz。

    v:
      世界坐标系向量。

  Returns:
    base 坐标系下的向量。
  """
  w = float(q_wxyz[0])
  q_vec = q_wxyz[1:4]

  a = v * (2.0 * w * w - 1.0)
  b = np.cross(q_vec, v) * (2.0 * w)
  c = q_vec * (2.0 * np.dot(q_vec, v))

  return a - b + c

# =============================================================================
# 4. 策略网络加载
# =============================================================================
class CheckpointActor(torch.nn.Module):
  """
  用纯 PyTorch 复刻训练时 actor 的 MLP。

  你的 X3 runner 配置里 actor 是：
    512 -> 256 -> 128
    activation = elu
  """

  def __init__(self, input_dim: int, output_dim: int):
    super().__init__()

    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(input_dim, 512),
      torch.nn.ELU(),
      torch.nn.Linear(512, 256),
      torch.nn.ELU(),
      torch.nn.Linear(256, 128),
      torch.nn.ELU(),
      torch.nn.Linear(128, output_dim),
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(obs)


def load_policy(checkpoint: Path) -> tuple[CheckpointActor, int, int]:
  """
  从 rsl_rl / mjlab 保存的 checkpoint 中加载 actor。

  Args:
    checkpoint:
      model_xxxxx.pt 路径。

  Returns:
    actor:
      可直接 forward 的 PyTorch 网络。

    obs_dim:
      observation 维度，X3 正常应该是 93。

    act_dim:
      action 维度，X3 正常应该是 28。
  """
  data = torch.load(checkpoint, map_location="cpu", weights_only=False)

  actor_sd = data.get("actor_state_dict")
  if actor_sd is None:
    raise KeyError(
      "checkpoint 中找不到 actor_state_dict。"
      "请确认这是 rsl_rl / mjlab 保存的 policy checkpoint。"
    )

  obs_dim = int(actor_sd["mlp.0.weight"].shape[1])
  act_dim = int(actor_sd["mlp.6.weight"].shape[0])

  actor = CheckpointActor(input_dim=obs_dim, output_dim=act_dim)
  actor.load_state_dict({k: v for k, v in actor_sd.items() if k.startswith("mlp.")})
  actor.eval()

  return actor, obs_dim, act_dim

# =============================================================================
# 5. 手柄控制模块
# =============================================================================
class GamepadCommandInput:
  def __init__(
    self,
    *,
    max_vx_forward: float = 0.7,
    max_vx_backward: float = 0.5,
    max_vy: float = 0.2,
    max_wyaw: float = 0.8,
    deadzone: float = 0.15,
  ):
    self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    self.max_vx_forward = float(max_vx_forward)
    self.max_vx_backward = float(max_vx_backward)
    self.max_vy = float(max_vy)
    self.max_wyaw = float(max_wyaw)
    self.deadzone = float(deadzone)

    self.joystick = None

    try:
      import pygame

      pygame.init()
      pygame.joystick.init()

      if pygame.joystick.get_count() > 0:
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        print(f"\n[INFO] 🎮 成功连接手柄: {self.joystick.get_name()}")
        print("[INFO] 操作说明:")
        print("       左摇杆 上/下 = 前进 / 后退 vx")
        print("       左摇杆 左/右 = 左右侧移 vy")
        print("       右摇杆 左/右 = 左右转向 wyaw")
        print("       松开摇杆 = 回到原地站立命令\n")
      else:
        print("\n[WARN] ⚠️ 未检测到手柄，机器人会保持原地站立。\n")

    except ImportError:
      print("\n[WARN] 未安装 pygame，无法使用手柄。安装方式：pip install pygame")
      print("[WARN] 当前机器人会保持原地站立。\n")

  def _deadzone(self, value: float) -> float:
    """摇杆死区，防止摇杆没有完全回中导致机器人缓慢漂移。"""
    if abs(value) < self.deadzone:
      return 0.0
    return value

  def poll(self) -> None:
    """
    每个 policy step 调用一次，刷新手柄命令。
    """
    if self.joystick is None:
      return

    import pygame

    pygame.event.pump()

    # 常见 Xbox / PS 手柄轴映射：
    # axis 0: 左摇杆左右
    # axis 1: 左摇杆上下
    # axis 3: 右摇杆左右
    lx = self._deadzone(float(self.joystick.get_axis(0)))
    ly = self._deadzone(float(self.joystick.get_axis(1)))
    rx = self._deadzone(float(self.joystick.get_axis(2)))

    # 左摇杆向上通常是 ly=-1，表示前进。
    if ly < 0.0:
      vx = -ly * self.max_vx_forward
    else:
      vx = -ly * self.max_vx_backward

    # 左摇杆左右控制侧移。
    # 这里取负号是为了常见手柄手感：
    # 左推 -> vy 正/负 的方向如果相反，就把符号改掉。
    vy = -lx * self.max_vy

    # 右摇杆左右控制 yaw 角速度。
    wyaw = -rx * self.max_wyaw

    self.cmd[0] = vx
    self.cmd[1] = vy
    self.cmd[2] = wyaw

    print(
      f"\r[手柄命令] vx={self.cmd[0]:+5.2f}  "
      f"vy={self.cmd[1]:+5.2f}  "
      f"wyaw={self.cmd[2]:+5.2f}    ",
      end="",
      flush=True,
    )

  def stop(self) -> None:
    """退出时关闭 pygame。"""
    if self.joystick is not None:
      import pygame
      pygame.quit()


# =============================================================================
# 6. 构建原生 MuJoCo X3 模型
# =============================================================================
def build_x3_model(model_path: Path) -> mujoco.MjModel:
  """
  构建 Sim2Sim 用的 X3 MuJoCo 模型。

  这里会做几件事：
    1. 从 XML 读取机器人。
    2. 添加一个 terrain 平面，方便单独 sim2sim。
    3. 设置碰撞：
       - terrain 可接触
       - robot collision geom 可接触 terrain
       - robot visual mesh 不接触
       - robot collision geom 使用 conaffinity=0，避免机器人自碰撞
    4. 添加纯力矩 actuator。
    5. 设置 MuJoCo solver 参数，尽量对齐训练环境。
  """
  spec = mujoco.MjSpec.from_file(str(model_path))

  # ---------------------------------------------------------------------------
  # 6.1 添加可视化天空盒和地面材质
  # ---------------------------------------------------------------------------
  sky = spec.add_texture()
  sky.name = "skybox"
  sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
  sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
  sky.rgb1 = (0.3, 0.5, 0.7)
  sky.rgb2 = (0.0, 0.0, 0.0)
  sky.width = 512
  sky.height = 512

  tex = spec.add_texture()
  tex.name = "texplane"
  tex.type = mujoco.mjtTexture.mjTEXTURE_2D
  tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
  tex.rgb1 = (0.2, 0.3, 0.4)
  tex.rgb2 = (0.1, 0.15, 0.2)
  tex.width = 512
  tex.height = 512
  tex.mark = mujoco.mjtMark.mjMARK_CROSS
  tex.markrgb = (0.8, 0.8, 0.8)

  mat = spec.add_material()
  mat.name = "matplane"
  mat.reflectance = 0.3
  mat.texrepeat = (4, 4)
  mat.texuniform = True
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "texplane"

  # ---------------------------------------------------------------------------
  # 6.2 添加 terrain 平面
  # ---------------------------------------------------------------------------
  terrain_body = spec.worldbody.add_body(name="terrain")
  terrain_geom = terrain_body.add_geom(
    name="terrain",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0.0, 0.0, 0.01),
  )
  terrain_geom.material = "matplane"
  terrain_geom.contype = 1
  terrain_geom.conaffinity = 1
  terrain_geom.condim = 3
  terrain_geom.friction = (0.6, 0.005, 0.0001)
  terrain_geom.priority = 1

  # ---------------------------------------------------------------------------
  # 6.3 设置碰撞
  # ---------------------------------------------------------------------------
  for geom in spec.geoms:
    if geom.name == "terrain":
      continue

    # 如果 XML 里还保留了 floor，为了避免和 terrain 重叠接触，这里关闭它的物理接触。
    if geom.name == "floor":
      geom.contype = 0
      geom.conaffinity = 0
      continue

    if geom.name and geom.name.endswith("_collision"):
      geom.contype = 1
      geom.conaffinity = 0
      geom.group = 3

      if geom.name in ("l_foot_collision", "r_foot_collision"):
        geom.condim = 3
        geom.friction = (0.6, 0.005, 0.0001)
        geom.priority = 1
      else:
        geom.condim = 1
    else:
      # visual mesh 不参与碰撞。
      geom.contype = 0
      geom.conaffinity = 0

  # ---------------------------------------------------------------------------
  # 6.4 添加纯力矩 actuator
  # 手工 PD + DcMotorActuator。
  # ---------------------------------------------------------------------------
  for i, joint_name in enumerate(ACT_JOINTS):
    effort = float(EFFORT_LIMIT[i])

    actuator = spec.add_actuator(name=f"motor_{joint_name}", target=joint_name)

    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_NONE

    actuator.gainprm[0] = 1.0
    actuator.biasprm[:] = 0.0

    actuator.forcelimited = True
    actuator.forcerange = (-effort, effort)

    actuator.ctrllimited = True
    actuator.ctrlrange = (-effort, effort)

    if joint_name in ARMATURE_MAP:
      spec.joint(joint_name).armature = ARMATURE_MAP[joint_name]

  # ---------------------------------------------------------------------------
  # 6.5 添加光照
  # ---------------------------------------------------------------------------
  light = spec.worldbody.add_light(name="main_light")
  light.pos = (0.0, 0.0, 3.0)
  light.dir = (0.0, 0.0, -1.0)
  light.ambient = (0.6, 0.6, 0.6)
  light.diffuse = (0.8, 0.8, 0.8)
  light.specular = (0.5, 0.5, 0.5)
  light.castshadow = True

  # ---------------------------------------------------------------------------
  # 6.6 编译模型并设置仿真参数
  # ---------------------------------------------------------------------------
  model = spec.compile()

  model.opt.timestep = 0.005
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
  model.opt.iterations = 10
  model.opt.ls_iterations = 20

  return model


# =============================================================================
# 7. 默认关节角和 action scale 解析
# =============================================================================
def build_default_joint_pos(xml_joints: list[str]) -> np.ndarray:

  defaults = np.zeros(len(xml_joints), dtype=np.float64)

  matched_indices, _, matched_values = resolve_matching_names_values(
    X3_INIT_STATE.joint_pos,
    list(xml_joints),
  )

  defaults[np.asarray(matched_indices, dtype=np.int64)] = np.asarray(
    matched_values,
    dtype=np.float64,
  )

  return defaults


def build_action_scale(xml_joints: list[str]) -> np.ndarray:
  """
  把 X3_ACTION_SCALE 解析成 XML 关节顺序数组。
  """
  action_scale = np.zeros(len(xml_joints), dtype=np.float64)

  matched_indices, _, matched_values = resolve_matching_names_values(
    X3_ACTION_SCALE,
    list(xml_joints),
  )

  action_scale[np.asarray(matched_indices, dtype=np.int64)] = np.asarray(
    matched_values,
    dtype=np.float64,
  )

  if np.any(action_scale == 0.0):
    zero_joints = [xml_joints[i] for i, v in enumerate(action_scale) if v == 0.0]
    print("[WARN] 以下关节 action_scale 为 0，请确认 X3_ACTION_SCALE 是否覆盖它们：")
    for name in zero_joints:
      print(f"       {name}")

  return action_scale


# =============================================================================
# 8. 主仿真循环
# =============================================================================

def run_sim(
  actor: CheckpointActor,
  obs_dim: int,
  act_dim: int,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  command: GamepadCommandInput,
  *,
  duration: float = 120.0,
  decimation: int = 4,
  realtime: bool = True,
  headless: bool = False,
  stand_time: float = 1.0,
) -> None:

  if obs_dim != 93:
    print(f"[WARN] checkpoint obs_dim={obs_dim}，但 X3 actor obs 正常应为 93。请确认训练配置。")

  if act_dim != NUM_DOF:
    raise RuntimeError(
      f"checkpoint action dim={act_dim}，但 X3 NUM_DOF={NUM_DOF}。"
      "这通常说明 checkpoint 不是这个 X3 任务训练出来的。"
    )

  # ---------------------------------------------------------------------------
  # 8.1 XML 顺序索引
  # observation 使用 XML 关节顺序。
  # ---------------------------------------------------------------------------
  xml_joint_ids = np.array(
    [
      int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
      for name in XML_JOINTS
    ],
    dtype=np.int64,
  )

  xml_qpos_idx = np.array(
    [int(model.jnt_qposadr[jid]) for jid in xml_joint_ids],
    dtype=np.int64,
  )

  xml_qvel_idx = np.array(
    [int(model.jnt_dofadr[jid]) for jid in xml_joint_ids],
    dtype=np.int64,
  )

  xml_defaults = build_default_joint_pos(XML_JOINTS)
  action_scale = build_action_scale(XML_JOINTS)

  # ---------------------------------------------------------------------------
  # 8.2 actuator 顺序索引
  # PD 力矩计算和 data.ctrl 写入使用 actuator 顺序。
  # ---------------------------------------------------------------------------
  act_joint_ids = np.array(
    [
      int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
      for name in ACT_JOINTS
    ],
    dtype=np.int64,
  )

  act_qpos_idx = np.array(
    [int(model.jnt_qposadr[jid]) for jid in act_joint_ids],
    dtype=np.int64,
  )

  act_qvel_idx = np.array(
    [int(model.jnt_dofadr[jid]) for jid in act_joint_ids],
    dtype=np.int64,
  )

  motor_ids = np.array(
    [
      int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}"))
      for name in ACT_JOINTS
    ],
    dtype=np.int64,
  )

  # 从 XML 顺序映射到 actuator 顺序。
  xml_name_to_idx = {name: i for i, name in enumerate(XML_JOINTS)}
  act_from_xml = np.array(
    [xml_name_to_idx[name] for name in ACT_JOINTS],
    dtype=np.int64,
  )

  # ---------------------------------------------------------------------------
  # 8.3 初始化机器人状态
  #
  # 出生时：
  #   root pos = X3_INIT_STATE.pos
  #   root quat = [1, 0, 0, 0]
  #   joints = X3_INIT_STATE.joint_pos 解析后的默认角
  #   qvel = 0
  # ---------------------------------------------------------------------------
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0

  data.qpos[0:3] = np.asarray(X3_INIT_STATE.pos, dtype=np.float64)
  data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
  data.qpos[xml_qpos_idx] = xml_defaults

  data.ctrl[:] = 0.0

  mujoco.mj_forward(model, data)

  last_action = np.zeros(NUM_DOF, dtype=np.float32)

  # ---------------------------------------------------------------------------
  # 8.4 viewer
  # ---------------------------------------------------------------------------
  viewer = None

  if not headless:
    viewer = mujoco.viewer.launch_passive(
      model,
      data,
      show_left_ui=False,
      show_right_ui=False,
    )

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body("pelvis").id
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -15.0

  print("\n[INFO] X3 Sim2Sim started.")
  print(f"[INFO] NUM_DOF={NUM_DOF}, obs_dim={obs_dim}, act_dim={act_dim}")
  print(f"[INFO] 前 {stand_time:.2f} 秒 command 强制为 0，让机器人先原地站稳。\n")

  step_count = 0

  # ---------------------------------------------------------------------------
  # 8.5 主循环
  # ---------------------------------------------------------------------------
  while data.time < duration and (viewer is None or viewer.is_running()):
    loop_start = time.perf_counter()

    # 出生后一段时间强制站立命令为 0。
    if data.time < stand_time:
      command.cmd[:] = 0.0
    else:
      command.poll()

    # -------------------------------------------------------------------------
    # 8.5.1 组装 actor observation
    #
    # X3 actor obs:
    #   base_ang_vel       3
    #   projected_gravity  3
    #   command            3
    #   joint_pos_rel      28
    #   joint_vel_rel      28
    #   last_action        28
    #
    # 总共 93 维。
    # -------------------------------------------------------------------------
    quat_wxyz = data.qpos[3:7].copy()
    base_ang_vel = data.qvel[3:6].copy()

    projected_gravity = quat_rotate_inverse_wxyz(
      quat_wxyz,
      np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
    )

    joint_pos_rel = data.qpos[xml_qpos_idx] - xml_defaults
    joint_vel = data.qvel[xml_qvel_idx]

    obs = np.concatenate(
      [
        base_ang_vel,
        projected_gravity,
        command.cmd,
        joint_pos_rel,
        joint_vel,
        last_action,
      ],
      axis=0,
    ).astype(np.float32)

    if obs.shape[0] != obs_dim:
      raise RuntimeError(
        f"组装出来的 obs 维度是 {obs.shape[0]}，但 checkpoint 需要 {obs_dim}。"
      )

    # -------------------------------------------------------------------------
    # 8.5.2 策略推理
    # actor 输出 action，action 的顺序按 XML_JOINTS。
    # -------------------------------------------------------------------------
    with torch.no_grad():
      action = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).cpu().numpy()

    # 防止极端输出炸仿真。
    action = np.clip(action, -100.0, 100.0).astype(np.float64)

    last_action[:] = action.astype(np.float32)

    # -------------------------------------------------------------------------
    # 8.5.3 action -> 目标关节角
    #
    # 和训练时 JointPositionActionCfg(use_default_offset=True) 对齐：
    #   target_q = default_q + action * action_scale
    # -------------------------------------------------------------------------
    target_q_xml = xml_defaults + action * action_scale
    target_q_act = target_q_xml[act_from_xml]

    # -------------------------------------------------------------------------
    # 8.5.4 物理 step
    #
    # 每次 actor 推理后，底层 MuJoCo step decimation 次。
    # 每个 step 都重新计算 PD 力矩并施加 DC motor 限幅。
    # -------------------------------------------------------------------------
    for _ in range(decimation):
      q = data.qpos[act_qpos_idx]
      dq = data.qvel[act_qvel_idx]

      # 手工 PD：
      #   torque = Kp * (target_q - q) - Kd * dq
      raw_torque = KP * (target_q_act - q) + KD * (0.0 - dq)

      # DC motor torque-speed 限幅。
      torque = dc_motor_clip_effort(raw_torque, dq)

      data.ctrl[motor_ids] = torque

      mujoco.mj_step(model, data)

      if viewer is not None:
        viewer.sync()

      if realtime:
        sleep_time = model.opt.timestep - (time.perf_counter() - loop_start)
        if sleep_time > 0.0:
          time.sleep(sleep_time)
        loop_start = time.perf_counter()

    step_count += 1

    if step_count % 50 == 0:
      lin_vel_b = quat_rotate_inverse_wxyz(data.qpos[3:7], data.qvel[0:3])
      print(
        f"\r[Time={data.time:6.2f}s] "
        f"cmd=({command.cmd[0]:+.2f}, {command.cmd[1]:+.2f}, {command.cmd[2]:+.2f}) "
        f"vel_b=({lin_vel_b[0]:+.2f}, {lin_vel_b[1]:+.2f}) "
        f"z={data.qpos[2]:.3f}     ",
        end="",
        flush=True,
      )

  if viewer is not None:
    viewer.close()


# =============================================================================
# 9. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="X3 mjlab AMP Sim2Sim with gamepad control")

  parser.add_argument(
    "--checkpoint",
    type=Path,
    required=True,
    help="训练好的 .pt checkpoint，例如 logs/rsl_rl/x3_amp/.../model_22500.pt",
  )

  parser.add_argument(
    "--model",
    type=Path,
    default=Path(X3_XML),
    help="X3 XML 路径，默认使用 X3_XML",
  )

  parser.add_argument(
    "--duration",
    type=float,
    default=120.0,
    help="仿真时长，单位秒",
  )

  parser.add_argument(
    "--decimation",
    type=int,
    default=4,
    help="每次 policy 推理后执行多少个 MuJoCo step。X3 训练默认是 4。",
  )

  parser.add_argument(
    "--stand-time",
    type=float,
    default=1.0,
    help="出生后前多少秒强制命令为 0，让机器人先站稳。",
  )

  parser.add_argument(
    "--headless",
    action="store_true",
    help="不打开 viewer",
  )

  parser.add_argument(
    "--no-realtime",
    action="store_true",
    help="不按实时速度 sleep，尽可能快地跑仿真",
  )

  parser.add_argument(
    "--max-vx-forward",
    type=float,
    default=0.7,
    help="手柄前进最大速度，和训练 command range 对齐",
  )

  parser.add_argument(
    "--max-vx-backward",
    type=float,
    default=0.5,
    help="手柄后退最大速度，和训练 command range 对齐",
  )

  parser.add_argument(
    "--max-vy",
    type=float,
    default=0.2,
    help="手柄左右侧移最大速度，和训练 command range 对齐",
  )

  parser.add_argument(
    "--max-wyaw",
    type=float,
    default=0.8,
    help="手柄转向最大角速度，和训练 command range 对齐",
  )

  return parser.parse_args()


def main() -> None:
  args = parse_args()

  actor, obs_dim, act_dim = load_policy(args.checkpoint)

  model = build_x3_model(args.model)
  data = mujoco.MjData(model)

  command = GamepadCommandInput(
    max_vx_forward=args.max_vx_forward,
    max_vx_backward=args.max_vx_backward,
    max_vy=args.max_vy,
    max_wyaw=args.max_wyaw,
  )

  try:
    run_sim(
      actor=actor,
      obs_dim=obs_dim,
      act_dim=act_dim,
      model=model,
      data=data,
      command=command,
      duration=args.duration,
      decimation=args.decimation,
      realtime=not args.no_realtime,
      headless=args.headless,
      stand_time=args.stand_time,
    )
  finally:
    command.stop()


if __name__ == "__main__":
  main()