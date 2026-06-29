"""Convert GMR retargeted X3 motion .pkl files to mjlab AMP format.
使用纯 MuJoCo 引擎进行绝对精准的高度对齐和关键点提取。
python retarget/scripts/convert_gmr_to_amp_x3.py --src_dir /home/dai/datasets_retargeted/x3_amp/ --dst_dir src/mjlab/tasks/amp/data/x3_motions --xml_path src/mjlab/asset_zoo/robots/x3/xmls/Moya01_V2.xml
"""

import argparse
import os
import pickle
import numpy as np
import mujoco

# ================= 手脚关键连杆 (Body)=================
KEY_BODY_NAMES = [
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

# 循环动作关键词匹配
WRAP_MOTIONS = {
    "x3_walk",
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

    # 2. 载入 MuJoCo 物理引擎
    model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(model)

    # 找到关键连杆的 ID
    key_body_ids = []
    for name in KEY_BODY_NAMES:
        b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if b_id == -1:
            raise ValueError(f"❌ XML 中找不到连杆: {name}")
        key_body_ids.append(b_id)

    num_key_bodies = len(key_body_ids)
    key_body_pos = np.zeros((num_frames, num_key_bodies, 3), dtype=np.float64)

    # 3. 物理推演并提取关键点坐标
    for i in range(num_frames):
        mj_data.qpos[:3] = root_pos[i]
        mj_data.qpos[3:7] = root_rot_wxyz[i]
        mj_data.qpos[7:7 + dof_pos.shape[1]] = dof_pos[i]
        mujoco.mj_kinematics(model, mj_data)

        for j, b_id in enumerate(key_body_ids):
            # 记录此时关键连杆在世界坐标系下的绝对位置
            key_body_pos[i, j, :] = mj_data.xpos[b_id]

    # 4. 判断是否为循环动作
    motion_name = os.path.splitext(os.path.basename(src_path))[0]
    loop_mode = 1 if motion_name in WRAP_MOTIONS else 0

    # 5. 打包为 AMP 格式
    amp_data = {
        "fps": fps,
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_rot_wxyz.astype(np.float32),  # AMP 框架通常使用 wxyz
        "dof_pos": dof_pos.astype(np.float32),
        "key_body_pos": key_body_pos.astype(np.float32),
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
    parser = argparse.ArgumentParser(description="Convert GMR X3 motion to AMP format using MuJoCo")
    parser.add_argument("--src_dir", default="x3_retarget_output", help="Directory with GMR .pkl files")
    parser.add_argument("--dst_dir", default=None, help="Output directory for AMP .pkl files")
    parser.add_argument("--xml_path", required=True, help="Path to your X3 XML file (e.g. Moya01_V2.xml)")
    args = parser.parse_args()

    if args.dst_dir is None:
        args.dst_dir = os.path.join(args.src_dir, "amp_format")

    pkl_files = sorted(f for f in os.listdir(args.src_dir) if f.endswith(".pkl"))
    if not pkl_files:
        print(f"⚠️ 在 {args.src_dir} 中没有找到任何 .pkl 文件！")
        return

    print(f"🔄 正在使用 MuJoCo 引擎进行 X3 动作的高精度转换，共 {len(pkl_files)} 个文件...")
    for pkl_file in pkl_files:
        src_path = os.path.join(args.src_dir, pkl_file)
        dst_path = os.path.join(args.dst_dir, pkl_file)
        convert(src_path, dst_path, args.xml_path)

    print(f"\n🎉 全部完成！X3 AMP 训练专用数据已保存在: {args.dst_dir}")

if __name__ == "__main__":
    main()