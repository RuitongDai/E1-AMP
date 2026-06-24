#!/usr/bin/env python3
"""Analyze base height and velocities from a retargeted AMP .pkl motion file.

这个脚本用于读取 AMP 格式的 .pkl 运动数据，通过数值微分计算基座的线速度和角速度，
并将它们从世界坐标系转换到机体局部坐标系，以便评估训练任务中的速度期望值。
"""

import argparse
import pickle
from pathlib import Path
import numpy as np


# ==========================================
# 四元数与坐标系转换工具函数
# ==========================================

def quat_normalize(quat: np.ndarray) -> np.ndarray:
    """归一化四元数"""
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """求四元数的共轭 (w, -x, -y, -z)"""
    out = quat.copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """哈密顿乘积"""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([w, x, y, z], axis=-1)


def rotate_vector_world_to_body(q_bw: np.ndarray, v_w: np.ndarray) -> np.ndarray:
    """将向量从世界坐标系转换到机体坐标系 v_b = q_inv * v_w * q"""
    q_bw = quat_normalize(q_bw)
    q_wb = quat_conjugate(q_bw)
    zeros = np.zeros((v_w.shape[0], 1), dtype=v_w.dtype)
    v_quat = np.concatenate([zeros, v_w], axis=-1)
    v_b_quat = quat_multiply(quat_multiply(q_wb, v_quat), q_bw)
    return v_b_quat[..., 1:]


def print_stats(name: str, values: np.ndarray) -> None:
    """打印统计信息"""
    print(f"\n{name}")
    if values.ndim == 1:
        print(
            "  min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                float(np.min(values)), float(np.max(values)),
                float(np.mean(values)), float(np.std(values)),
            )
        )
    elif values.ndim == 2:
        labels = ["x", "y", "z"]
        for i in range(values.shape[1]):
            label = labels[i] if i < len(labels) else f"dim{i}"
            c = values[:, i]
            print(
                "  {}: min={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                    label, float(np.min(c)), float(np.max(c)),
                    float(np.mean(c)), float(np.std(c)),
                )
            )


# ==========================================
# 核心分析流程 (带速度差分计算)
# ==========================================

def analyze_pkl_motion(pkl_file: Path, preview: int) -> None:
    """读取并分析 AMP PKL 文件，自动计算线速度和角速度。"""
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # AMP 转换脚本输出的键名：fps, root_pos, root_rot, dof_pos, key_body_pos, loop_mode
    fps = float(data["fps"])
    dt = 1.0 / fps

    # 提取基座的位置和姿态 (已经是 wxyz 格式)
    base_pos_w = data["root_pos"]  # [T, 3]
    base_quat_w = data["root_rot"]  # [T, 4] wxyz
    num_frames = base_pos_w.shape[0]

    # 1. 计算世界坐标系下的线速度 (差分法)
    # 使用 np.gradient 可以在边界处进行合理估算，保持数组长度不变
    base_vel_w = np.gradient(base_pos_w, dt, axis=0)

    # 2. 计算世界坐标系下的角速度 (四元数差分法)
    # 公式： omega_w = 2 * q_dot * q_inv
    q_dot = np.gradient(base_quat_w, dt, axis=0)
    q_inv = quat_conjugate(base_quat_w)
    omega_quat = 2.0 * quat_multiply(q_dot, q_inv)
    base_ang_vel_w = omega_quat[..., 1:]  # 取出 x, y, z

    # 3. 提取高度
    base_height = base_pos_w[:, 2]

    # 4. 转换到机体局部坐标系
    base_vel_b = rotate_vector_world_to_body(base_quat_w, base_vel_w)
    base_ang_vel_b = rotate_vector_world_to_body(base_quat_w, base_ang_vel_w)

    print("=" * 80)
    print(f"分析文件: {pkl_file}")
    print(f"总帧数={num_frames}, FPS={fps:.1f}, 动作总时长={num_frames / fps:.3f}秒")
    print(f"是否为循环动作: {'是' if data.get('loop_mode', 0) == 1 else '否'}")
    print("=" * 80)

    # 打印各项指标的统计信息
    print_stats("Base Height (world z) [基座高度]", base_height)
    print_stats("Base Linear Velocity (world frame) [世界坐标系线速度]", base_vel_w)
    print_stats("Base Linear Velocity (base frame) [局部坐标系线速度]", base_vel_b)
    print_stats("Base Angular Velocity (world frame) [世界坐标系角速度]", base_ang_vel_w)
    print_stats("Base Angular Velocity (base frame) [局部坐标系角速度]", base_ang_vel_b)

    preview = max(0, int(preview))
    if preview > 0:
        show_n = min(preview, num_frames)
        print("\nPreview (前 {} 帧预览):".format(show_n))
        print(
            "frame | height  | v_w(x,y,z) 世界线速度 | v_b(x,y,z) 局部线速度 | w_w(x,y,z) 世界角速度 | w_b(x,y,z) 局部角速度")
        for i in range(show_n):
            vw = base_vel_w[i]
            vb = base_vel_b[i]
            ww = base_ang_vel_w[i]
            wb = base_ang_vel_b[i]
            print(
                f"{i:5d} | {base_height[i]: 7.4f} | "
                f"({vw[0]: 7.3f},{vw[1]: 7.3f},{vw[2]: 7.3f}) | "
                f"({vb[0]: 7.3f},{vb[1]: 7.3f},{vb[2]: 7.3f}) | "
                f"({ww[0]: 7.3f},{ww[1]: 7.3f},{ww[2]: 7.3f}) | "
                f"({wb[0]: 7.3f},{wb[1]: 7.3f},{wb[2]: 7.3f})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 AMP PKL 文件中的速度")
    parser.add_argument(
        "--pkl_file",
        type=Path,
        help="经过 convert_gmr_to_amp_e1.py 处理后的 AMP .pkl 文件路径"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="打印前 N 帧的详细数据",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_pkl_motion(args.pkl_file, args.preview)