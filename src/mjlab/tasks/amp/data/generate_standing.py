"""Generate a synthetic Go1 standing motion .pkl for AMP training."""

import numpy as np
import joblib

NUM_FRAMES = 50
FPS = 30.0
NUM_DOFS = 12
NUM_KEY_BODIES = 4  # Four feet

# Go1 standing pose
DEFAULT_JOINT_POS = np.array(
  [
    # FR: hip, thigh, calf
    0.1,
    0.9,
    -1.8,
    # FL: hip, thigh, calf
    -0.1,
    0.9,
    -1.8,
    # RR: hip, thigh, calf
    0.1,
    0.9,
    -1.8,
    # RL: hip, thigh, calf
    -0.1,
    0.9,
    -1.8,
  ],
  dtype=np.float32,
)

# Root position stays at slightly above ground
root_pos = np.zeros((NUM_FRAMES, 3), dtype=np.float32)
root_pos[:, 2] = 0.278  # Go1 standing height

# Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
root_rot = np.zeros((NUM_FRAMES, 4), dtype=np.float32)
root_rot[:, 0] = 1.0

# Joint positions are constant (standing)
dof_pos = np.tile(DEFAULT_JOINT_POS, (NUM_FRAMES, 1))

# Key body positions (approximate foot positions in world frame for standing Go1)
key_body_pos = np.zeros((NUM_FRAMES, NUM_KEY_BODIES, 3), dtype=np.float32)
# FR foot
key_body_pos[:, 0, :] = [0.183, -0.132, 0.0]
# FL foot
key_body_pos[:, 1, :] = [0.183, 0.132, 0.0]
# RR foot
key_body_pos[:, 2, :] = [-0.183, -0.132, 0.0]
# RL foot
key_body_pos[:, 3, :] = [-0.183, 0.132, 0.0]

data = {
  "fps": FPS,
  "root_pos": root_pos,
  "root_rot": root_rot,
  "dof_pos": dof_pos,
  "key_body_pos": key_body_pos,
  "loop_mode": 1,  # WRAP
}

out = "/home/zy0046-hlw/mjlab/src/mjlab/tasks/amp/data/go1_motions/go1_standing.pkl"
joblib.dump(data, out)
print(f"Saved to {out}")
print(f"  root_pos: {root_pos.shape}")
print(f"  root_rot: {root_rot.shape}")
print(f"  dof_pos: {dof_pos.shape}")
print(f"  key_body_pos: {key_body_pos.shape}")
