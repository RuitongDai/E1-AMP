"""Convert GMR retargeted E1 motion .pkl files to mjlab AMP format.
使用纯 MuJoCo 引擎进行绝对精准的高度对齐和关键点提取。
"""

import argparse
import os
import pickle
import numpy as np
import mujoco

# ================= 手脚关键连杆 (Body)=================
KEY_BODY_NAMES = [
    "left_ankle_roll_link",    # 左脚
    "right_ankle_roll_link",   # 右脚
    "right_shoulder_yaw_link",
    "left_shoulder_yaw_link",
    "pelvis",
]

# 循环动作关键词匹配
WRAP_MOTIONS = {
    "e1_walk",
}

def convert(src_path: str, dst_path: str, xml_path: str) -> None:
    # 1. 读取 GMR 原始数据
    with open(src_path, "rb") as f:
        raw = pickle.load(f)

    fps = float(raw["fps"])
    root_pos = np.array(raw["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.array(raw["root_rot"], dtype=np.float64)
    dof_pos = np.array(raw["dof_pos"], dtype=np.float64)

    # --- 四元数格式重排: xyzw -> wxyz ---
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    num_frames = root_pos.shape[0]

    # 2. 载入 MuJoCo 模型 (绝对精准的物理环境)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    l_foot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_foot_collision")
    r_foot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_foot_collision")

    if l_foot_geom_id == -1 or r_foot_geom_id == -1:
        raise ValueError("❌ XML 中找不到 l_foot_collision 或 r_foot_collision，请确保已经添加了 Box 碰撞体！")

    # 根据 XML 设定，Box 的 Z 轴 size
    foot_half_thickness = 0.00

    # 3. 第一遍扫描：寻找整个动作中脚底板到达的绝对最低点
    lowest_z = float('inf')
    for i in range(num_frames):
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_rot_wxyz[i]
        data.qpos[7:] = dof_pos[i]
        mujoco.mj_kinematics(model, data)  # 仅更新运动学，不进行物理仿真

        # 获取左右脚碰撞体中心的 Z 坐标，减去半厚度得到真正的脚底板底部
        l_z = data.geom_xpos[l_foot_geom_id][2] - foot_half_thickness
        r_z = data.geom_xpos[r_foot_geom_id][2] - foot_half_thickness
        lowest_z = min(lowest_z, l_z, r_z)

    # 4. 高度对齐 & XY 归零
    root_pos[:, 2] -= lowest_z
    root_pos[:, :2] -= root_pos[0, :2]

    # 5. 第二遍扫描：提取高度修正后的 Key Body 位置
    key_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in KEY_BODY_NAMES]
    key_body_pos = np.zeros((num_frames, len(KEY_BODY_NAMES), 3), dtype=np.float32)

    for i in range(num_frames):
        data.qpos[:3] = root_pos[i]
        data.qpos[3:7] = root_rot_wxyz[i]
        data.qpos[7:] = dof_pos[i]
        mujoco.mj_kinematics(model, data)

        for j, b_id in enumerate(key_body_ids):
            key_body_pos[i, j] = data.xpos[b_id]

    # 6. 判断循环模式并保存
    motion_name = os.path.splitext(os.path.basename(src_path))[0]
    loop_mode = 1 if any(w in motion_name.lower() for w in WRAP_MOTIONS) else 0

    amp_data = {
        "fps": fps,
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot_wxyz.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "key_body_pos": key_body_pos,
        "loop_mode": loop_mode,
    }

    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(dst_path, "wb") as f:
        pickle.dump(amp_data, f)

    print(
        f"  ✅ 转换成功: {motion_name} | 帧数={num_frames} | DoF={dof_pos.shape[1]} "
        f"| 关键点={key_body_pos.shape[1]} | 循环模式={loop_mode} \n  -> {dst_path}"
    )

def main():
    parser = argparse.ArgumentParser(description="Convert GMR E1 motion to AMP format using MuJoCo")
    parser.add_argument("--src_dir", default="e1_retarget_output", help="Directory with GMR .pkl files")
    parser.add_argument("--dst_dir", default=None, help="Output directory for AMP .pkl files")
    parser.add_argument("--xml_path", required=True, help="Path to your E1 XML file (e.g. assets/e1/E1_25dof.xml)")
    args = parser.parse_args()

    if args.dst_dir is None:
        args.dst_dir = os.path.join(args.src_dir, "amp_format")

    pkl_files = sorted(f for f in os.listdir(args.src_dir) if f.endswith(".pkl"))
    if not pkl_files:
        print(f"❌ 在 {args.src_dir} 中没有找到任何 .pkl 文件！")
        return

    print(f"🔄 正在使用 MuJoCo 引擎进行高精度转换，共 {len(pkl_files)} 个文件...")
    for fname in pkl_files:
        src = os.path.join(args.src_dir, fname)
        dst = os.path.join(args.dst_dir, fname)
        convert(src, dst, args.xml_path)

    print(f"\n🎉 全部完成！AMP 训练专用数据已保存在: {args.dst_dir}")

if __name__ == "__main__":
    main()