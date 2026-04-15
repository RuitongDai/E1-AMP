"""Run a trained mjlab T1 AMP policy in pure MuJoCo (mj_step).

This is a true sim2sim path:
- policy learned in mjlab (MJWarp) with DcMotorActuator (motor actuators + manual PD)
- rollout executed in native MuJoCo dynamics (mj_step) with matching actuator model

The key insight: mjlab training creates *motor* actuators (force passthrough) and
computes PD torques + DC motor saturation in Python. We must replicate that here,
NOT use MuJoCo's built-in position servos which have different dynamics.

Example:
  uv run python scripts/sim2sim_t1_mjlab.py \\
    --checkpoint logs/rsl_rl/t1_amp/<run>/model_15000.pt --vx 0.3
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
from mjlab.asset_zoo.robots.booster_t1.t1_constants import (
  T1_ACTION_SCALE,
  T1_ARTICULATION,
  T1_INIT_STATE,
  T1_XML,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values


# ---------------------------------------------------------------------------
# Per-joint motor parameters (built from T1_ARTICULATION at import time)
# ---------------------------------------------------------------------------

def _build_motor_params() -> (
  tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]
):
  """Extract per-joint motor parameters and joint orderings.

  Returns:
    xml_joints: hinge joint names in XML kinematic tree order (for observations)
    act_joints: joint names in actuator creation order (for actions/ctrl)
    kp, kd, effort_limit, saturation_effort, velocity_limit: per-actuator arrays
    armature_map: joint_name -> armature value
  """
  spec = mujoco.MjSpec.from_file(str(T1_XML))

  # XML joint order (kinematic tree traversal) — used for observations.
  xml_joints: list[str] = []
  for jnt in spec.joints:
    if jnt.type == mujoco.mjtJoint.mjJNT_HINGE:
      xml_joints.append(jnt.name)

  # Actuator creation order — used for actions/ctrl.
  act_joints: list[str] = []
  kp_list: list[float] = []
  kd_list: list[float] = []
  eff_list: list[float] = []
  sat_list: list[float] = []
  vel_list: list[float] = []
  arm_map: dict[str, float] = {}

  for act_cfg in T1_ARTICULATION.actuators:
    assert isinstance(act_cfg, DcMotorActuatorCfg)
    for pattern in act_cfg.target_names_expr:
      for jname in xml_joints:  # resolve in XML order within each group
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
    xml_joints,
    act_joints,
    np.array(kp_list, dtype=np.float64),
    np.array(kd_list, dtype=np.float64),
    np.array(eff_list, dtype=np.float64),
    np.array(sat_list, dtype=np.float64),
    np.array(vel_list, dtype=np.float64),
    arm_map,
  )


XML_JOINTS, ACT_JOINTS, KP, KD, EFFORT_LIMIT, SATURATION_EFFORT, VELOCITY_LIMIT, ARMATURE_MAP = (
  _build_motor_params()
)
NUM_DOF = len(ACT_JOINTS)
VEL_AT_EFFORT_LIM = VELOCITY_LIMIT * (1.0 + EFFORT_LIMIT / SATURATION_EFFORT)


# ---------------------------------------------------------------------------
# DC motor torque saturation (matches mjlab DcMotorActuator._clip_effort)
# ---------------------------------------------------------------------------

def dc_motor_clip_effort(effort: np.ndarray, vel: np.ndarray) -> np.ndarray:
  """Apply DC motor torque-speed saturation curve."""
  vel_clipped = np.clip(vel, -VEL_AT_EFFORT_LIM, VEL_AT_EFFORT_LIM)
  torque_top = SATURATION_EFFORT * (1.0 - vel_clipped / VELOCITY_LIMIT)
  torque_bottom = SATURATION_EFFORT * (-1.0 - vel_clipped / VELOCITY_LIMIT)
  max_eff = np.minimum(torque_top, EFFORT_LIMIT)
  min_eff = np.maximum(torque_bottom, -EFFORT_LIMIT)
  return np.clip(effort, min_eff, max_eff)


# ---------------------------------------------------------------------------
# Quaternion math
# ---------------------------------------------------------------------------

def quat_rotate_inverse_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vector v by the inverse of quaternion q (wxyz convention)."""
  w = float(q_wxyz[0])
  q_vec = q_wxyz[1:4]
  a = v * (2.0 * w * w - 1.0)
  b = np.cross(q_vec, v) * (2.0 * w)
  c = q_vec * (2.0 * np.dot(q_vec, v))
  return a - b + c


# ---------------------------------------------------------------------------
# Actor network
# ---------------------------------------------------------------------------

class CheckpointActor(torch.nn.Module):
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
  """Load actor from checkpoint. Returns (actor, obs_dim, act_dim)."""
  data = torch.load(checkpoint, map_location="cpu", weights_only=False)
  actor_sd = data.get("actor_state_dict")
  if not isinstance(actor_sd, dict):
    raise ValueError(f"Invalid checkpoint format: {checkpoint}")

  first = actor_sd["mlp.0.weight"]
  last = actor_sd["mlp.6.weight"]
  obs_dim = int(first.shape[1])
  act_dim = int(last.shape[0])

  actor = CheckpointActor(input_dim=obs_dim, output_dim=act_dim)
  actor.load_state_dict({k: v for k, v in actor_sd.items() if k.startswith("mlp.")})
  actor.eval()
  print(f"[INFO] Loaded actor (obs={obs_dim}, act={act_dim}, iter={data.get('iter', '?')})")
  return actor, obs_dim, act_dim


# ---------------------------------------------------------------------------
# Keyboard command input
# ---------------------------------------------------------------------------

class CommandInput:
  """Keyboard-controlled velocity commands via terminal raw input.

  Reads keypresses directly from the terminal using termios raw mode +
  select for non-blocking reads. This avoids MuJoCo viewer shortcut
  conflicts — press keys in the **terminal** window, not the MuJoCo window.
  """

  _VX_STEP, _VY_STEP, _WYAW_STEP = 0.1, 0.05, 0.1
  _VX_RANGE = (-0.5, 0.7)
  _VY_RANGE = (-0.2, 0.2)
  _WYAW_RANGE = (-0.8, 0.8)

  def __init__(self, vx: float, vy: float, wyaw: float, enabled: bool):
    self.cmd = np.array([vx, vy, wyaw], dtype=np.float64)
    self.enabled = enabled
    self._old_settings = None
    if enabled:
      import sys
      if not sys.stdin.isatty():
        print("[WARN] stdin is not a TTY, keyboard control disabled.")
        self.enabled = False
        return
      import atexit, termios, tty
      self._old_settings = termios.tcgetattr(sys.stdin)
      atexit.register(self.stop)  # safety: restore terminal even on crash
      tty.setcbreak(sys.stdin.fileno())  # cbreak: char-at-a-time, no echo
      print("[INFO] Keyboard control enabled (press keys in THIS terminal):")
      print("       W/↑ = forward   S/↓ = backward")
      print("       A/← = left      D/→ = right")
      print("       Q = turn left   E = turn right")
      print("       Space = stop    Ctrl+C = quit")

  def poll(self) -> None:
    """Non-blocking poll for keypresses. Call once per sim step."""
    if not self.enabled:
      return
    import sys, select
    while select.select([sys.stdin], [], [], 0)[0]:
      ch = sys.stdin.read(1)
      if ch == "\x1b":  # escape sequence (arrow keys)
        seq = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0)[0] else ""
        if seq == "[A":    # Up
          ch = "w"
        elif seq == "[B":  # Down
          ch = "s"
        elif seq == "[D":  # Left
          ch = "a"
        elif seq == "[C":  # Right
          ch = "d"
        else:
          continue
      self._handle_key(ch)

  def _handle_key(self, ch: str) -> None:
    if ch in ("w", "W"):
      self.cmd[0] = np.clip(self.cmd[0] + self._VX_STEP, *self._VX_RANGE)
    elif ch in ("s", "S"):
      self.cmd[0] = np.clip(self.cmd[0] - self._VX_STEP, *self._VX_RANGE)
    elif ch in ("a", "A"):
      self.cmd[1] = np.clip(self.cmd[1] + self._VY_STEP, *self._VY_RANGE)
    elif ch in ("d", "D"):
      self.cmd[1] = np.clip(self.cmd[1] - self._VY_STEP, *self._VY_RANGE)
    elif ch in ("q", "Q"):
      self.cmd[2] = np.clip(self.cmd[2] + self._WYAW_STEP, *self._WYAW_RANGE)
    elif ch in ("e", "E"):
      self.cmd[2] = np.clip(self.cmd[2] - self._WYAW_STEP, *self._WYAW_RANGE)
    elif ch == " ":
      self.cmd[:] = 0.0
    else:
      return
    print(
      f"\r[CMD] vx={self.cmd[0]:+.2f}  vy={self.cmd[1]:+.2f}  "
      f"wyaw={self.cmd[2]:+.2f}    ",
      end="", flush=True,
    )

  def stop(self) -> None:
    if self._old_settings is not None:
      import sys, termios
      termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_t1_model(model_path: Path) -> mujoco.MjModel:
  """Build T1 model with ground plane, collision config, and motor actuators.

  Matches mjlab training: motor actuators (force passthrough) with manual PD.
  """
  spec = mujoco.MjSpec.from_file(str(model_path))

  # ── Skybox (blue gradient, matches TFBOT XML) ──
  sky = spec.add_texture()
  sky.name = "skybox"
  sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
  sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
  sky.rgb1 = (0.3, 0.5, 0.7)
  sky.rgb2 = (0.0, 0.0, 0.0)
  sky.width = 512
  sky.height = 512

  # ── Ground plane checker texture (matches TFBOT XML) ──
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

  # ── Ground material (texrepeat=4 matches training terrain_entity default) ──
  mat = spec.add_material()
  mat.name = "matplane"
  mat.reflectance = 0.3
  mat.texrepeat = (4, 4)
  mat.texuniform = True
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "texplane"

  # ── Ground plane geom ──
  terrain_body = spec.worldbody.add_body(name="terrain")
  terrain_geom = terrain_body.add_geom(
    name="terrain",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.01),
  )
  terrain_geom.material = "matplane"
  terrain_geom.contype = 1
  terrain_geom.conaffinity = 1

  # ── Collision config (match training CollisionCfg) ──
  for g in spec.geoms:
    if g.name == "terrain":
      continue
    if g.name and g.name.endswith("_collision"):
      g.contype = 1
      g.conaffinity = 1
      if g.name in ("l_foot_collision", "r_foot_collision"):
        g.condim = 3
        g.friction = (0.6, 0.005, 0.0001)
        g.priority = 1
      else:
        g.condim = 1
    else:
      g.contype = 0
      g.conaffinity = 0

  # ── Motor actuators (matches mjlab create_motor_actuator) ──
  # Training uses motor actuators where ctrl = torque (computed externally via PD).
  for i, jname in enumerate(ACT_JOINTS):
    eff = float(EFFORT_LIMIT[i])
    a = spec.add_actuator(name=f"motor_{jname}", target=jname)
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.dyntype = mujoco.mjtDyn.mjDYN_NONE
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.biastype = mujoco.mjtBias.mjBIAS_NONE
    a.gainprm[0] = 1.0
    a.biasprm[:] = 0.0
    a.forcelimited = True
    a.forcerange = (-eff, eff)
    a.ctrllimited = True
    a.ctrlrange = (-eff, eff)

    # Set armature.
    if jname in ARMATURE_MAP:
      spec.joint(jname).armature = ARMATURE_MAP[jname]

  model = spec.compile()

  # ── Physics options (match training MujocoCfg defaults) ──
  model.opt.timestep = 0.005
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
  model.opt.iterations = 10
  model.opt.ls_iterations = 20

  return model


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_sim(
  actor: CheckpointActor,
  obs_dim: int,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  command: CommandInput,
  *,
  duration: float = 60.0,
  decimation: int = 4,
  realtime: bool = True,
  headless: bool = False,
) -> None:
  """Run pure MuJoCo rollout with policy-in-the-loop.

  IMPORTANT: In training, both observations AND policy actions use XML joint
  order (entity's natural joint ordering from spec.joints). The actuator
  creation order (from T1_ARTICULATION) only matters for mapping PD torques
  to data.ctrl. We build a permutation to convert XML-order targets to
  actuator-order for the PD computation.
  """
  # ── XML joint order (used for obs, action, last_action, scale, offset) ──
  xml_qpos_idx = np.array(
    [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
     for n in XML_JOINTS],
    dtype=np.int64,
  )
  xml_qvel_idx = np.array(
    [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
     for n in XML_JOINTS],
    dtype=np.int64,
  )
  xml_defaults = np.array(
    [T1_INIT_STATE.joint_pos[n] for n in XML_JOINTS], dtype=np.float64
  )

  # Action scale in XML order (matches training's JointPositionAction resolution).
  scale_idx, _, scale_vals = resolve_matching_names_values(T1_ACTION_SCALE, list(XML_JOINTS))
  action_scale = np.zeros(NUM_DOF, dtype=np.float64)
  action_scale[np.array(scale_idx, dtype=np.int64)] = np.array(scale_vals, dtype=np.float64)

  # ── Actuator order (used for PD torques → data.ctrl) ──
  act_joint_ids = model.actuator_trnid[:, 0].astype(np.int64).tolist()
  act_joint_names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    for jid in act_joint_ids
  ]
  act_qpos_idx = np.array(
    [int(model.jnt_qposadr[j]) for j in act_joint_ids], dtype=np.int64
  )
  act_qvel_idx = np.array(
    [int(model.jnt_dofadr[j]) for j in act_joint_ids], dtype=np.int64
  )

  # Permutation: for each actuator, which XML index does it correspond to?
  xml_name_to_idx = {n: i for i, n in enumerate(XML_JOINTS)}
  act_from_xml = np.array([xml_name_to_idx[n] for n in act_joint_names], dtype=np.int64)

  print(f"[INFO] XML joint order:  {list(XML_JOINTS)}")
  print(f"[INFO] Act joint order:  {act_joint_names}")

  # ── Reset to initial pose ──
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  data.qpos[0:3] = np.array(T1_INIT_STATE.pos, dtype=np.float64)
  data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
  data.qpos[xml_qpos_idx] = xml_defaults
  data.ctrl[:] = 0.0
  mujoco.mj_forward(model, data)

  last_action = np.zeros(NUM_DOF, dtype=np.float32)

  print(f"[INFO] T1 sim2sim started (motor actuators + manual PD + DC motor saturation)")
  print(f"[INFO] obs_dim={obs_dim} act_dim={NUM_DOF} nu={model.nu}")
  print(f"[INFO] KP range=[{KP.min():.0f}, {KP.max():.0f}]  KD range=[{KD.min():.1f}, {KD.max():.1f}]")

  viewer = (
    mujoco.viewer.launch_passive(
      model, data, show_left_ui=False, show_right_ui=False,
    )
    if not headless
    else None
  )

  # Camera follows the robot: set initial trackbody + nice angle
  if viewer is not None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body("Trunk").id
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0

  step_count = 0
  while data.time < duration and (viewer is None or viewer.is_running()):
    loop_start = time.perf_counter()
    command.poll()  # non-blocking keyboard read from terminal

    # ── Build observation (XML joint order) ──
    quat_wxyz = data.qpos[3:7]
    ang_vel = data.qvel[3:6]  # body-frame angular velocity (freejoint)
    projected_gravity = quat_rotate_inverse_wxyz(quat_wxyz, np.array([0.0, 0.0, -1.0]))

    obs = np.concatenate([
      ang_vel,
      projected_gravity,
      command.cmd,
      data.qpos[xml_qpos_idx] - xml_defaults,   # joint_pos_rel in XML order
      data.qvel[xml_qvel_idx],                   # joint_vel_rel in XML order
      last_action,                                # last raw action (XML order)
    ]).astype(np.float32)

    # ── Policy inference (output is XML order) ──
    with torch.no_grad():
      action = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    action = np.clip(action, -100.0, 100.0)
    last_action[:] = action

    # ── Compute position target (XML order → actuator order) ──
    target_q_xml = xml_defaults + action * action_scale   # XML order
    target_q_act = target_q_xml[act_from_xml]             # permute to actuator order

    # ── Step with manual PD + DC motor saturation (actuator order) ──
    for _ in range(decimation):
      q = data.qpos[act_qpos_idx]
      dq = data.qvel[act_qvel_idx]
      raw_torque = KP * (target_q_act - q) + KD * (0.0 - dq)
      torque = dc_motor_clip_effort(raw_torque, dq)
      data.ctrl[:NUM_DOF] = torque
      mujoco.mj_step(model, data)

      if viewer is not None:
        viewer.sync()

      if realtime:
        elapsed = time.perf_counter() - loop_start
        sleep = model.opt.timestep - elapsed
        if sleep > 0:
          time.sleep(sleep)
        loop_start = time.perf_counter()

    step_count += 1
    if step_count % 50 == 0:
      lin_vel_w = data.qvel[0:3]
      lin_vel_b = quat_rotate_inverse_wxyz(data.qpos[3:7], lin_vel_w)
      print(
        f"\r[t={data.time:.1f}s] vx={lin_vel_b[0]:.3f} vy={lin_vel_b[1]:.3f} "
        f"z={data.qpos[2]:.3f}",
        end="",
        flush=True,
      )

  print("\n[INFO] T1 sim2sim finished")
  if viewer is not None:
    viewer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="mjlab T1 true sim2sim (MuJoCo mj_step)")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--model", type=Path, default=Path(T1_XML))
  parser.add_argument("--duration", type=float, default=60.0)
  parser.add_argument("--decimation", type=int, default=4)
  parser.add_argument("--vx", type=float, default=0.3)
  parser.add_argument("--vy", type=float, default=0.0)
  parser.add_argument("--wyaw", type=float, default=0.0)
  parser.add_argument("--keyboard", action="store_true")
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--no-realtime", action="store_true")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if not args.checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
  if not args.model.exists():
    raise FileNotFoundError(f"MuJoCo model not found: {args.model}")

  actor, obs_dim, act_dim = load_policy(args.checkpoint)

  model = build_t1_model(args.model)
  data_obj = mujoco.MjData(model)

  # When keyboard control is on, start standing still.
  init_vx = 0.0 if args.keyboard else args.vx
  init_vy = 0.0 if args.keyboard else args.vy
  init_wyaw = 0.0 if args.keyboard else args.wyaw
  cmd = CommandInput(vx=init_vx, vy=init_vy, wyaw=init_wyaw, enabled=args.keyboard)
  try:
    run_sim(
      actor, obs_dim, model, data_obj, cmd,
      duration=args.duration,
      decimation=args.decimation,
      realtime=not args.no_realtime,
      headless=args.headless,
    )
  finally:
    cmd.stop()


if __name__ == "__main__":
  main()
