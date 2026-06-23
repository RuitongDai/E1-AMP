"""Convert GMR retargeted E1 motion .pkl files to mjlab AMP format.

GMR output format:
  - fps, root_pos(T,3), root_rot(T,4)[xyzw], dof_pos(T,25), local_body_pos=None, link_body_list=None

AMP expected format:
  - fps, root_pos(T,3), root_rot(T,4)[wxyz], dof_pos(T,25), key_body_pos(T,K,3), loop_mode
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

# ================= 手脚关键连杆=================
KEY_BODY_NAMES = [
    "left_wrist_pitch_link",   # 左手
    "right_wrist_pitch_link",  # 右手
    "left_ankle_roll_link",    # 左脚
    "right_ankle_roll_link",   # 右脚
]

# 循环动作关键词匹配
WRAP_MOTIONS = {
    "female_walk1",
    "female_walk_backwards",
}

def convert(src_path: str, dst_path: str, xml_path: str, device: str = "cpu") -> None:
    with open(src_path, "rb") as f:
        raw = pickle.load(f)

    fps = float(raw["fps"])
    root_pos = np.array(raw["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.array(raw["root_rot"], dtype=np.float64)  # (T, 4) xyzw
    dof_pos = np.array(raw["dof_pos"], dtype=np.float64)

    # --- 四元数格式重排: xyzw -> wxyz ---
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

    # --- 载入 E1 模型并计算前向运动学 ---
    km = KinematicsModel(xml_path, device)
    key_body_ids = [km.get_body_idx(n) for n in KEY_BODY_NAMES]

    root_pos_t = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
    root_rot_t = torch.as_tensor(root_rot_wxyz, dtype=torch.float32, device=device)
    dof_pos_t = torch.as_tensor(dof_pos, dtype=torch.float32, device=device)

    body_pos, _ = km.forward_kinematics(root_pos_t, root_rot_t, dof_pos_t)
    key_body_pos = body_pos[:, key_body_ids, :].detach().cpu().numpy()  # (T, K, 3)

    # --- 高度对齐：让身体最低点（通常是脚底）刚好贴地 ---
    lowest_z = body_pos[..., 2].min().item()
    root_pos[:, 2] -= lowest_z

    # 高度调整后重新计算 key_body_pos
    root_pos_t = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
    body_pos, _ = km.forward_kinematics(root_pos_t, root_rot_t, dof_pos_t)
    key_body_pos = body_pos[:, key_body_ids, :].detach().cpu().numpy()

    # --- XY 原点对齐：让第一帧从 (0,0) 开始 ---
    root_pos[:, :2] -= root_pos[0, :2]

    # XY 平移后再次重新计算 key_body_pos
    root_pos_t = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
    body_pos, _ = km.forward_kinematics(root_pos_t, root_rot_t, dof_pos_t)
    key_body_pos = body_pos[:, key_body_ids, :].detach().cpu().numpy()

    # --- 判断是否为循环动作 ---
    motion_name = os.path.splitext(os.path.basename(src_path))[0]
    # 如果文件名中包含 WRAP_MOTIONS 里的关键词，就标记为循环(1)
    loop_mode = 1 if any(w in motion_name.lower() for w in WRAP_MOTIONS) else 0

    # --- 保存为 AMP 最终格式 ---
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
        f"  ✅ 转换成功: {motion_name} | 帧数={nf} | DoF={dof_pos.shape[1]} "
        f"| 关键点={key_body_pos.shape[1]} | 循环模式={loop_mode} \n  -> {dst_path}"
    )

def main():
    parser = argparse.ArgumentParser(description="Convert GMR E1 motion to AMP format")
    parser.add_argument("--src_dir", default="e1_retarget_output", help="Directory with GMR .pkl files")
    parser.add_argument("--dst_dir", default=None, help="Output directory for AMP .pkl files")
    parser.add_argument("--xml_path", required=True, help="Path to your E1 XML file (e.g. assets/e1/e1.xml)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.dst_dir is None:
        args.dst_dir = os.path.join(args.src_dir, "amp_format")

    pkl_files = sorted(f for f in os.listdir(args.src_dir) if f.endswith(".pkl"))
    if not pkl_files:
        print(f"❌ 在 {args.src_dir} 中没有找到任何 .pkl 文件！")
        return

    print(f"🔄 正在转换 {len(pkl_files)} 个动作文件...")
    for fname in pkl_files:
        src = os.path.join(args.src_dir, fname)
        dst = os.path.join(args.dst_dir, fname)
        convert(src, dst, args.xml_path, args.device)

    print(f"\n🎉 全部完成！AMP 训练专用数据已保存在: {args.dst_dir}")

if __name__ == "__main__":
    main()