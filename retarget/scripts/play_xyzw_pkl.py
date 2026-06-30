import argparse
import pickle
import time
import os
import sys
import numpy as np
import mujoco
import mujoco.viewer


class PlayerState:
    def __init__(self, num_frames, fps):
        self.num_frames = num_frames
        self.current_frame = 0
        self.is_paused = False
        self.fps = fps


def print_status(state):
    """在终端实时覆盖打印状态"""
    status = "PAUSED " if state.is_paused else "PLAYING"
    sys.stdout.write(
        f"\r[{status}] Frame: {state.current_frame:04d} / {state.num_frames:04d} | "
        f"FPS: {state.fps:.1f}  "
    )
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMP 处理后数据的专用播放器")
    parser.add_argument("--pkl_file", type=str, required=True, help="AMP 格式的 .pkl 文件路径")
    parser.add_argument("--xml_file", type=str, default="src/mjlab/asset_zoo/robots/x3/xmls/Moya01_V2.xml",
                        help="机器人的 XML 模型路径")
    args = parser.parse_args()

    # 1. 检查文件是否存在
    if not os.path.exists(args.xml_file):
        print(f"❌ 找不到模型文件: {args.xml_file}")
        sys.exit(1)
    if not os.path.exists(args.pkl_file):
        print(f"❌ 找不到动作文件: {args.pkl_file}")
        sys.exit(1)

    # 2. 加载模型与数据
    model = mujoco.MjModel.from_xml_path(args.xml_file)
    data = mujoco.MjData(model)

    with open(args.pkl_file, "rb") as f:
        amp_data = pickle.load(f)

    # AMP 格式提取，此时root_rot是wxyz格式
    fps = amp_data.get("fps", 30.0)
    root_pos = amp_data["root_pos"]
    root_rot_wxyz = amp_data["root_rot"]
    dof_pos = amp_data["dof_pos"]

    num_frames = root_pos.shape[0]
    state = PlayerState(num_frames, fps)

    print("=" * 60)
    print(f"🎬 AMP 动作播放器启动 | {os.path.basename(args.pkl_file)}")
    print(f"总帧数: {num_frames} | 循环模式: {amp_data.get('loop_mode', 0)}")
    print("------------------------------------------------------------")
    print("快捷键说明：")
    print("  [Space]  : 播放 / 暂停")
    print("  [Left]   : 上一帧 (暂停状态下微调查看)")
    print("  [Right]  : 下一帧 (暂停状态下微调查看)")
    print("============================================================")


    # 3. 键盘回调函数
    def key_callback(keycode):
        if keycode == ord(' '):  # 空格
            state.is_paused = not state.is_paused
        elif keycode == 262:  # 右方向键
            if state.is_paused:
                state.current_frame = min(state.current_frame + 1, state.num_frames - 1)
        elif keycode == 263:  # 左方向键
            if state.is_paused:
                state.current_frame = max(state.current_frame - 1, 0)
        print_status(state)


    # 4. 启动可视化窗口
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # 调整一下视角，方便看清脚底板和地面的接触情况
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -10
        viewer.cam.azimuth = 120
        viewer.cam.lookat[:] = [0, 0, 0.3]

        while viewer.is_running():
            step_start = time.time()

            # 获取当前帧数据
            idx = state.current_frame
            pos = root_pos[idx]
            rot_wxyz = root_rot_wxyz[idx]
            joints = dof_pos[idx]

            # 拼装完整的 qpos
            qpos = np.concatenate([pos, rot_wxyz, joints])
            data.qpos[:] = qpos

            # 使用 forward 计算运动学姿态
            mujoco.mj_forward(model, data)
            viewer.sync()
            print_status(state)

            # 帧数步进逻辑
            if not state.is_paused:
                state.current_frame += 1
                if state.current_frame >= state.num_frames:
                    state.current_frame = 0  # 自动循环播放

            # 维持设定的 FPS 帧率
            time_until_next_step = (1.0 / state.fps) - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)