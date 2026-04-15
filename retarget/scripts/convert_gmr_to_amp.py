"""Convert GMR retargeted T1 motion .pkl files to mjlab AMP format.

GMR output format:
  - fps, root_pos(T,3), root_rot(T,4)[xyzw], dof_pos(T,23), local_body_pos=None, link_body_list=None

AMP expected format:
  - fps, root_pos(T,3), root_rot(T,4)[wxyz], dof_pos(T,23), key_body_pos(T,K,3), loop_mode
"""

import argparse
import os
import pickle

import numpy as np
import torch

# fmt: off
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from general_motion_retargeting.kinematics_model import KinematicsModel
# fmt: on

# Key bodies whose world positions are stored for AMP discriminator
KEY_BODY_NAMES = [
    "left_hand_link",
    "right_hand_link",
    "left_foot_link",
    "right_foot_link",
]

T1_XML = os.path.join(os.path.dirname(__file__), "..", "assets", "booster_t1", "T1_serial.xml")

# Motions that are cyclic (should wrap) vs one-shot (should clamp)
WRAP_MOTIONS = {
    "female_walk1",
    "female_walk_backwards",
}


def convert(src_path: str, dst_path: str, device: str = "cpu") -> None:
    with open(src_path, "rb") as f:
        raw = pickle.load(f)

    fps = float(raw["fps"])
    root_pos = np.array(raw["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.array(raw["root_rot"], dtype=np.float64)  # (T, 4) xyzw
    dof_pos = np.array(raw["dof_pos"], dtype=np.float64)

    # --- Quaternion reorder: xyzw -> wxyz ---
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

    # --- Compute key body positions via FK ---
    km = KinematicsModel(T1_XML, device)
    key_body_ids = [km.get_body_idx(n) for n in KEY_BODY_NAMES]

    root_pos_t = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
    # KinematicsModel expects wxyz quaternions
    root_rot_t = torch.as_tensor(root_rot_wxyz, dtype=torch.float32, device=device)
    dof_pos_t = torch.as_tensor(dof_pos, dtype=torch.float32, device=device)

    body_pos, _ = km.forward_kinematics(root_pos_t, root_rot_t, dof_pos_t)
    # body_pos: (T, num_bodies, 3)

    key_body_pos = body_pos[:, key_body_ids, :].detach().cpu().numpy()  # (T, K, 3)

    # --- Height adjustment: ensure lowest body part touches ground ---
    lowest_z = body_pos[..., 2].min().item()
    root_pos[:, 2] -= lowest_z

    # Recompute key_body_pos after height adjustment
    root_pos_t = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
    body_pos, _ = km.forward_kinematics(root_pos_t, root_rot_t, dof_pos_t)
    key_body_pos = body_pos[:, key_body_ids, :].detach().cpu().numpy()

    # --- XY origin offset: start at (0, 0) ---
    root_pos[:, :2] -= root_pos[0, :2]

    # Recompute key_body_pos after XY offset
    root_pos_t = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
    body_pos, _ = km.forward_kinematics(root_pos_t, root_rot_t, dof_pos_t)
    key_body_pos = body_pos[:, key_body_ids, :].detach().cpu().numpy()

    # --- Determine loop mode ---
    motion_name = os.path.splitext(os.path.basename(src_path))[0]
    loop_mode = 1 if motion_name in WRAP_MOTIONS else 0

    # --- Save in AMP format ---
    amp_data = {
        "fps": fps,
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot_wxyz.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "key_body_pos": key_body_pos.astype(np.float32),
        "loop_mode": loop_mode,
    }

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(dst_path, "wb") as f:
        pickle.dump(amp_data, f)

    nf = root_pos.shape[0]
    print(
        f"  Converted: {motion_name} | frames={nf} | dof={dof_pos.shape[1]} "
        f"| key_bodies={key_body_pos.shape[1]} | loop={loop_mode} | -> {dst_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Convert GMR T1 motion to AMP format")
    parser.add_argument("--src_dir", default="t1_retarget_output", help="Directory with GMR .pkl files")
    parser.add_argument("--dst_dir", default=None, help="Output directory for AMP .pkl files")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.dst_dir is None:
        args.dst_dir = os.path.join(args.src_dir, "amp_format")

    pkl_files = sorted(f for f in os.listdir(args.src_dir) if f.endswith(".pkl"))
    if not pkl_files:
        print(f"No .pkl files found in {args.src_dir}")
        return

    print(f"Converting {len(pkl_files)} motion files...")
    for fname in pkl_files:
        src = os.path.join(args.src_dir, fname)
        dst = os.path.join(args.dst_dir, fname)
        convert(src, dst, args.device)

    print(f"\nAll done. AMP motion files in: {args.dst_dir}")


if __name__ == "__main__":
    main()
