"""
运行训练好的 mjlab E1 AMP 策略，在纯正的 MuJoCo (mj_step) 环境中进行仿真。

这是一个真实的 Sim-to-Sim (仿真到仿真) 验证路径：
- 策略是在 mjlab (基于 MJWarp 批处理) 中训练出来的，使用了 DcMotorActuator (电机执行器 + 手动 PD 控制)。
- 现在的验证是在原生的 MuJoCo 动力学 (单个 mj_step) 中执行，并严格匹配训练时的执行器模型。

核心要点：
mjlab 训练时创建的是“纯力矩电机”，并在 Python 中手动计算 PD 力矩并施加 DC 电机限幅。
我们必须在这里完美复刻这个计算过程，而绝对不能使用 MuJoCo 自带的“位置伺服(position servos)”，因为动力学表现完全不同。

使用示例:
  python scripts/sim2sim_e1_mjlab.py \
    --checkpoint logs/rsl_rl/e1_amp/2026-xxxx/model_50000.pt --vx 0.4
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
from mjlab.asset_zoo.robots.e1.e1_constants import (
  E1_ACTION_SCALE,
  E1_ARTICULATION,
  E1_INIT_STATE,
  E1_XML,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

# ---------------------------------------------------------------------------
# 提取每个关节的电机参数
# ---------------------------------------------------------------------------
def _build_motor_params() -> (
  tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]
):
  """
  返回:
    xml_joints: 按照 XML 拓扑顺序排列的关节名 (用于拼接神经网络的观测状态 Observation)
    act_joints: 按照驱动器定义顺序排列的关节名 (用于神经网络输出后计算 PD 力矩)
    kp, kd, effort_limit, saturation_effort, velocity_limit: 对应的物理限制数组
  """
  spec = mujoco.MjSpec.from_file(str(E1_XML))

  # 1. XML 关节顺序 (Kinematic Tree 顺序)
  xml_joints: list[str] = []
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_HINGE:
      xml_joints.append(jnt.name)

  # 2. 驱动器创建顺序
  act_joints: list[str] = []
  kp_list, kd_list, eff_list, sat_list, vel_list = [], [], [], [], []
  arm_map: dict[str, float] = {}

  for act_cfg in E1_ARTICULATION.actuators:
    assert isinstance(act_cfg, DcMotorActuatorCfg)
    for pattern in act_cfg.target_names_expr:
      for jname in xml_joints:
        if re.fullmatch(pattern, jname):
          act_joints.append(jname)
          kp_list.append(act_cfg.stiffness)
          kd_list.append(act_cfg.damping)
          eff_list.append(act_cfg.effort_limit)
          sat_list.append(act_cfg.saturation_effort)
          vel_list.append(act_cfg.velocity_limit)
          if act_cfg.armature is not None:
            arm_map[jname] = act_cfg.armature

  return (
    xml_joints, act_joints,
    np.array(kp_list, dtype=np.float64), np.array(kd_list, dtype=np.float64),
    np.array(eff_list, dtype=np.float64), np.array(sat_list, dtype=np.float64),
    np.array(vel_list, dtype=np.float64), arm_map,
  )

XML_JOINTS, ACT_JOINTS, KP, KD, EFFORT_LIMIT, SATURATION_EFFORT, VELOCITY_LIMIT, ARMATURE_MAP = _build_motor_params()
NUM_DOF = len(ACT_JOINTS)
VEL_AT_EFFORT_LIM = VELOCITY_LIMIT * (1.0 + EFFORT_LIMIT / SATURATION_EFFORT)

# ---------------------------------------------------------------------------
# DC 电机力矩限幅
# ---------------------------------------------------------------------------
def dc_motor_clip_effort(effort: np.ndarray, vel: np.ndarray) -> np.ndarray:
  """应用直流电机的 力矩-转速(Torque-Speed) 饱和曲线。转速越快，能输出的最大力矩越小。"""
  vel_clipped = np.clip(vel, -VEL_AT_EFFORT_LIM, VEL_AT_EFFORT_LIM)
  torque_top = SATURATION_EFFORT * (1.0 - vel_clipped / VELOCITY_LIMIT)
  torque_bottom = SATURATION_EFFORT * (-1.0 - vel_clipped / VELOCITY_LIMIT)
  max_eff = np.minimum(torque_top, EFFORT_LIMIT)
  min_eff = np.maximum(torque_bottom, -EFFORT_LIMIT)
  return np.clip(effort, min_eff, max_eff)

# ---------------------------------------------------------------------------
# 四元数数学工具
# ---------------------------------------------------------------------------
def quat_rotate_inverse_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
  """使用逆四元数旋转向量 (用于将世界坐标系的重力/速度转换到局部坐标系)"""
  w = float(q_wxyz[0])
  q_vec = q_wxyz[1:4]
  a = v * (2.0 * w * w - 1.0)
  b = np.cross(q_vec, v) * (2.0 * w)
  c = q_vec * (2.0 * np.dot(q_vec, v))
  return a - b + c


# ---------------------------------------------------------------------------
# 策略网络 (Actor) 加载器
# ---------------------------------------------------------------------------
class CheckpointActor(torch.nn.Module):
  """用纯 PyTorch 复刻 rl_games/rsl_rl 里的 MLP 架构"""
  def __init__(self, input_dim: int, output_dim: int):
    super().__init__()
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(input_dim, 512), torch.nn.ELU(),
      torch.nn.Linear(512, 256), torch.nn.ELU(),
      torch.nn.Linear(256, 128), torch.nn.ELU(),
      torch.nn.Linear(128, output_dim),
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(obs)

def load_policy(checkpoint: Path) -> tuple[CheckpointActor, int, int]:
  """从 .pt 文件中提取权重并注入到 Actor 中"""
  data = torch.load(checkpoint, map_location="cpu", weights_only=False)
  actor_sd = data.get("actor_state_dict")

  obs_dim = int(actor_sd["mlp.0.weight"].shape[1])
  act_dim = int(actor_sd["mlp.6.weight"].shape[0])

  actor = CheckpointActor(input_dim=obs_dim, output_dim=act_dim)
  actor.load_state_dict({k: v for k, v in actor_sd.items() if k.startswith("mlp.")})
  actor.eval()
  return actor, obs_dim, act_dim


# ---------------------------------------------------------------------------
# 手柄控制模块 (Gamepad Control)
# ---------------------------------------------------------------------------
class GamepadCommandInput:
  def __init__(self, vx: float, vy: float, wyaw: float):
    # 强制初始速度为 0，保证一出生是原地站立
    self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float64)
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
        print("       左摇杆 上/下 = 前进/后退")
        print("       左摇杆 左/右 = 左右侧移")
        print("       右摇杆 左/右 = 左右转向\n")
      else:
        print("\n[WARN] ⚠️ 未检测到手柄，机器人将保持原地站立！\n")
    except ImportError:
      print("\n[WARN] 未安装 pygame，无法使用手柄。请运行 `pip install pygame`\n")

  def poll(self) -> None:
    if self.joystick is None:
      return

    import pygame
    pygame.event.pump()  # 刷新手柄事件

    # 提取摇杆轴数据 (范围是 -1 到 1)
    # 注意：不同手柄的轴映射可能略有不同。通常 Xbox/PS 手柄：
    lx = self.joystick.get_axis(0)
    ly = self.joystick.get_axis(1)
    rx = self.joystick.get_axis(3)

    # 设置死区 (Deadzone)：防止物理摇杆没有完全回中导致机器人自动漂移
    deadzone = 0.15
    lx = 0.0 if abs(lx) < deadzone else lx
    ly = 0.0 if abs(ly) < deadzone else ly
    rx = 0.0 if abs(rx) < deadzone else rx

    # 映射摇杆输入到机器人的速度范围
    # 物理摇杆向上推，ly通常是负数(-1)；向下推是正数(1)
    if ly < 0:
      vx = -ly * 0.7  # 往前推，最大速度 0.7
    else:
      vx = -ly * 0.5  # 往后拉，最大后退速度 -0.5

    vy = -lx * 0.2  # 左摇杆左右控制侧移，最大 0.2
    wyaw = -rx * 0.8  # 右摇杆控制转向，最大 0.8

    # 平滑赋值
    self.cmd[0] = vx
    self.cmd[1] = vy
    self.cmd[2] = wyaw

    print(f"\r[手柄指令] vx={self.cmd[0]:+5.2f}  vy={self.cmd[1]:+5.2f}  wyaw={self.cmd[2]:+5.2f}    ", end="",
          flush=True)

  def stop(self) -> None:
    if self.joystick is not None:
      import pygame
      pygame.quit()

# ---------------------------------------------------------------------------
# 构建底层物理模型
# ---------------------------------------------------------------------------
def build_e1_model(model_path: Path) -> mujoco.MjModel:
  """构建 E1 模型，注入与训练环境完全一致的地形、摩擦力和电机执行器"""
  spec = mujoco.MjSpec.from_file(str(model_path))

  # ── 天空盒与地面材质 (美化) ──
  sky = spec.add_texture()
  sky.name, sky.type, sky.builtin = "skybox", mujoco.mjtTexture.mjTEXTURE_SKYBOX, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
  sky.rgb1, sky.rgb2, sky.width, sky.height = (0.3, 0.5, 0.7), (0.0, 0.0, 0.0), 512, 512

  tex = spec.add_texture()
  tex.name, tex.type, tex.builtin = "texplane", mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
  tex.rgb1, tex.rgb2, tex.width, tex.height, tex.mark, tex.markrgb = (0.2, 0.3, 0.4), (0.1, 0.15, 0.2), 512, 512, mujoco.mjtMark.mjMARK_CROSS, (0.8, 0.8, 0.8)

  mat = spec.add_material()
  mat.name, mat.reflectance, mat.texrepeat, mat.texuniform = "matplane", 0.3, (4, 4), True
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "texplane"

  # ── 添加地面 ──
  terrain_body = spec.worldbody.add_body(name="terrain")
  terrain_geom = terrain_body.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01))
  terrain_geom.material, terrain_geom.contype, terrain_geom.conaffinity = "matplane", 1, 1

  # ── 碰撞配置──
  for g in spec.geoms:
    if g.name == "terrain": continue
    if g.name and g.name.endswith("_collision"):
      g.contype = 1
      g.conaffinity = 1
      # 匹配e1_constants.py 里设置的脚底高摩擦
      if g.name in ("l_foot_collision", "r_foot_collision"):
        g.condim = 3
        g.friction = (0.6, 0.005, 0.0001)
        g.priority = 1
      else:
        g.condim = 1
    else:
      g.contype = 0
      g.conaffinity = 0

  # ── 将位置伺服替换为纯力矩电机 ──
  for i, jname in enumerate(ACT_JOINTS):
    eff = float(EFFORT_LIMIT[i])
    a = spec.add_actuator(name=f"motor_{jname}", target=jname)
    a.trntype, a.dyntype, a.gaintype, a.biastype = mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
    a.gainprm[0], a.biasprm[:] = 1.0, 0.0
    a.forcelimited, a.forcerange, a.ctrllimited, a.ctrlrange = True, (-eff, eff), True, (-eff, eff)
    if jname in ARMATURE_MAP:
      spec.joint(jname).armature = ARMATURE_MAP[jname]

  # ── 增强光照 ──
  light = spec.worldbody.add_light(name="main_light")
  light.pos = (0, 0, 3)
  light.dir = (0, 0, -1)
  light.ambient = (0.6, 0.6, 0.6)
  light.diffuse = (0.8, 0.8, 0.8)
  light.specular = (0.5, 0.5, 0.5)
  light.castshadow = True

  model = spec.compile()
  model.opt.timestep = 0.005
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
  model.opt.iterations, model.opt.ls_iterations = 10, 20
  return model


# ---------------------------------------------------------------------------
# 主仿真循环
# ---------------------------------------------------------------------------
def run_sim(
  actor: CheckpointActor, obs_dim: int, model: mujoco.MjModel, data: mujoco.MjData, command: CommandInput,
  *, duration: float = 60.0, decimation: int = 4, realtime: bool = True, headless: bool = False,
) -> None:

  # === 1. 建立 XML 顺序的索引 (供观察状态提取) ===
  xml_qpos_idx = np.array([int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in XML_JOINTS], dtype=np.int64)
  xml_qvel_idx = np.array([int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in XML_JOINTS], dtype=np.int64)
  xml_defaults = np.array([E1_INIT_STATE.joint_pos[n] for n in XML_JOINTS], dtype=np.float64)

  scale_idx, _, scale_vals = resolve_matching_names_values(E1_ACTION_SCALE, list(XML_JOINTS))
  action_scale = np.zeros(NUM_DOF, dtype=np.float64)
  action_scale[np.array(scale_idx, dtype=np.int64)] = np.array(scale_vals, dtype=np.float64)

  # === 2. 建立 驱动器顺序 的索引 (供施加控制力矩) ===
  act_joint_ids = model.actuator_trnid[:, 0].astype(np.int64).tolist()
  act_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in act_joint_ids]
  act_qpos_idx = np.array([int(model.jnt_qposadr[j]) for j in act_joint_ids], dtype=np.int64)
  act_qvel_idx = np.array([int(model.jnt_dofadr[j]) for j in act_joint_ids], dtype=np.int64)

  # 映射器：当我们需要给某个电机指令时，它在网络输出(XML顺序)里的位置
  xml_name_to_idx = {n: i for i, n in enumerate(XML_JOINTS)}
  act_from_xml = np.array([xml_name_to_idx[n] for n in act_joint_names], dtype=np.int64)

  # === 3. 初始化机器人姿态 ===
  data.qpos[:], data.qvel[:] = 0.0, 0.0
  data.qpos[0:3] = np.array(E1_INIT_STATE.pos, dtype=np.float64)
  data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
  data.qpos[xml_qpos_idx] = xml_defaults
  data.ctrl[:] = 0.0
  mujoco.mj_forward(model, data)
  last_action = np.zeros(NUM_DOF, dtype=np.float32)

  viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) if not headless else None

  if viewer is not None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    # 视角追踪骨盆
    viewer.cam.trackbodyid = model.body("pelvis").id
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.0, 135.0, -15.0

  step_count = 0
  while data.time < duration and (viewer is None or viewer.is_running()):
    loop_start = time.perf_counter()
    command.poll()

    # ── 1. 组装神经网络的观测值 ──
    quat_wxyz = data.qpos[3:7]
    ang_vel = data.qvel[3:6]
    projected_gravity = quat_rotate_inverse_wxyz(quat_wxyz, np.array([0.0, 0.0, -1.0]))

    obs = np.concatenate([
      ang_vel, projected_gravity, command.cmd,
      data.qpos[xml_qpos_idx] - xml_defaults,  # 关节相对位置
      data.qvel[xml_qvel_idx],                 # 关节速度
      last_action,                             # 上一步的动作
    ]).astype(np.float32)

    # ── 2. 神经网络推理 (输出为目标关节角度偏移量) ──
    with torch.no_grad():
      action = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    action = np.clip(action, -100.0, 100.0)
    last_action[:] = action

    # ── 3. 将目标角度转换为 驱动器顺序 ──
    target_q_xml = xml_defaults + action * action_scale
    target_q_act = target_q_xml[act_from_xml]

    # ── 4. 执行物理仿真步 (默认 decimation=4，即网络推理1次，物理走4步) ──
    for _ in range(decimation):
      q = data.qpos[act_qpos_idx]
      dq = data.qvel[act_qvel_idx]
      # 纯手工 PD 控制公式: Torque = Kp * (Target - Current) - Kd * Current_Vel
      raw_torque = KP * (target_q_act - q) + KD * (0.0 - dq)
      # 施加电机饱和限制
      torque = dc_motor_clip_effort(raw_torque, dq)

      data.ctrl[:NUM_DOF] = torque
      mujoco.mj_step(model, data)

      if viewer is not None: viewer.sync()

      if realtime:
        sleep = model.opt.timestep - (time.perf_counter() - loop_start)
        if sleep > 0: time.sleep(sleep)
        loop_start = time.perf_counter()

    step_count += 1
    if step_count % 50 == 0:
      lin_vel_b = quat_rotate_inverse_wxyz(data.qpos[3:7], data.qvel[0:3])
      print(f"\r[Time={data.time:.1f}s] E1局部速度 vx={lin_vel_b[0]:.3f} vy={lin_vel_b[1]:.3f} 高度z={data.qpos[2]:.3f}", end="", flush=True)

  if viewer is not None: viewer.close()

# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="E1 真实 Sim2Sim 部署验证脚本")
  parser.add_argument("--checkpoint", type=Path, required=True, help="PT 模型路径")
  parser.add_argument("--model", type=Path, default=Path(E1_XML))
  parser.add_argument("--duration", type=float, default=120.0)
  parser.add_argument("--decimation", type=int, default=4)
  parser.add_argument("--vx", type=float, default=0.4) # E1 默认速度设为 0.4
  parser.add_argument("--vy", type=float, default=0.0)
  parser.add_argument("--wyaw", type=float, default=0.0)
  parser.add_argument("--keyboard", action="store_true")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--no-realtime", action="store_true")
  return parser.parse_args()

def main() -> None:
  args = parse_args()
  actor, obs_dim, act_dim = load_policy(args.checkpoint)
  model = build_e1_model(args.model)
  data_obj = mujoco.MjData(model)

  cmd = GamepadCommandInput(vx=0.0, vy=0.0, wyaw=0.0)

  try:
    run_sim(actor, obs_dim, model, data_obj, cmd, duration=args.duration, decimation=args.decimation, realtime=not args.no_realtime, headless=args.headless)
  finally:
    cmd.stop()

if __name__ == "__main__":
  main()