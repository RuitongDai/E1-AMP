#!/usr/bin/env bash
# ============================================================================
# build_package.sh — 将 mjlab 源码 + GMR 源码打包进 T1-AMP-Walking 项目
#
# 用法：
#   cd /home/zy0046-hlw/T1-AMP-Walking
#   bash build_package.sh
#
# 前置条件：
#   - mjlab 源码位于 /home/zy0046-hlw/mjlab/
#   - GMR 源码位于  /home/zy0046-hlw/GMR/
# ============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
MJLAB_DIR="/home/zy0046-hlw/mjlab"
GMR_DIR="/home/zy0046-hlw/GMR"

echo "=========================================="
echo " T1-AMP-Walking Package Builder"
echo "=========================================="
echo "Project dir : $PROJ_DIR"
echo "mjlab source: $MJLAB_DIR"
echo "GMR source  : $GMR_DIR"
echo ""

# ------------------------------------------------------------------
# Step 1: 复制 mjlab 完整源码包 (训练框架 + 物理引擎接口 + AMP + T1机器人)
# ------------------------------------------------------------------
echo "[1/5] Copying mjlab source package..."
rm -rf "$PROJ_DIR/src"
mkdir -p "$PROJ_DIR/src"
cp -r "$MJLAB_DIR/src/mjlab" "$PROJ_DIR/src/mjlab"

# 删除 __pycache__ 减小体积
find "$PROJ_DIR/src/mjlab" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

MJLAB_PY_COUNT=$(find "$PROJ_DIR/src/mjlab" -name "*.py" | wc -l)
MJLAB_STL_COUNT=$(find "$PROJ_DIR/src/mjlab" -name "*.stl" -o -name "*.STL" | wc -l)
MJLAB_PKL_COUNT=$(find "$PROJ_DIR/src/mjlab" -name "*.pkl" | wc -l)
echo "  -> $MJLAB_PY_COUNT Python files, $MJLAB_STL_COUNT mesh files, $MJLAB_PKL_COUNT motion files"

# ------------------------------------------------------------------
# Step 2: 复制 GMR 完整源码包 (动作重定向框架)
# ------------------------------------------------------------------
echo "[2/5] Copying GMR source package..."
rm -rf "$PROJ_DIR/general_motion_retargeting"
cp -r "$GMR_DIR/general_motion_retargeting" "$PROJ_DIR/general_motion_retargeting"

# 删除 __pycache__
find "$PROJ_DIR/general_motion_retargeting" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

GMR_PY_COUNT=$(find "$PROJ_DIR/general_motion_retargeting" -name "*.py" | wc -l)
GMR_JSON_COUNT=$(find "$PROJ_DIR/general_motion_retargeting" -name "*.json" | wc -l)
echo "  -> $GMR_PY_COUNT Python files, $GMR_JSON_COUNT IK config files"

# ------------------------------------------------------------------
# Step 3: 复制 GMR 机器人资产 (T1 的 URDF/XML + mesh 用于重定向)
# ------------------------------------------------------------------
echo "[3/5] Copying GMR robot assets for T1..."
rm -rf "$PROJ_DIR/gmr_assets"
mkdir -p "$PROJ_DIR/gmr_assets"
if [ -d "$GMR_DIR/assets/booster_t1" ]; then
  cp -r "$GMR_DIR/assets/booster_t1" "$PROJ_DIR/gmr_assets/booster_t1"
  echo "  -> $(find "$PROJ_DIR/gmr_assets/booster_t1" -type f | wc -l) files"
else
  echo "  -> WARNING: $GMR_DIR/assets/booster_t1 not found, skipping"
fi

# 复制 SMPL-X body models (retarget 需要, 约4.7GB, 默认跳过)
# 如需重定向功能，取消下面的注释：
# if [ -d "$GMR_DIR/assets/body_models" ]; then
#   echo "  -> Copying SMPL-X body models (~4.7GB)..."
#   cp -r "$GMR_DIR/assets/body_models" "$PROJ_DIR/gmr_assets/body_models"
# fi
echo "  -> NOTE: SMPL-X body models (~4.7GB) skipped. Retarget motion data already included."

# ------------------------------------------------------------------
# Step 4: 复制/更新 sim2sim 脚本和重定向脚本
# ------------------------------------------------------------------
echo "[4/5] Copying scripts..."
mkdir -p "$PROJ_DIR/scripts"
cp "$MJLAB_DIR/scripts/sim2sim_t1_mjlab.py" "$PROJ_DIR/scripts/sim2sim_t1_mjlab.py"

mkdir -p "$PROJ_DIR/retarget/scripts" "$PROJ_DIR/retarget/ik_configs"
cp "$GMR_DIR/scripts/retarget_t1_walking.sh" "$PROJ_DIR/retarget/scripts/"
cp "$GMR_DIR/scripts/convert_gmr_to_amp.py" "$PROJ_DIR/retarget/scripts/"
cp "$GMR_DIR/general_motion_retargeting/ik_configs/smplx_to_t1.json" "$PROJ_DIR/retarget/ik_configs/"

echo "  -> sim2sim, retarget scripts copied"

# ------------------------------------------------------------------
# Step 5: 复制最新训练检查点
# ------------------------------------------------------------------
echo "[5/5] Copying latest checkpoint..."
mkdir -p "$PROJ_DIR/checkpoints"
CKPT_DIR="$MJLAB_DIR/logs/rsl_rl/t1_amp"
if [ -d "$CKPT_DIR" ]; then
  LATEST_CKPT=$(find "$CKPT_DIR" -name "model_*.pt" -type f | sort -t_ -k2 -n | tail -1)
  if [ -n "$LATEST_CKPT" ]; then
    cp "$LATEST_CKPT" "$PROJ_DIR/checkpoints/"
    echo "  -> Copied $(basename "$LATEST_CKPT")"
  else
    echo "  -> No checkpoint found"
  fi
else
  echo "  -> Checkpoint directory not found, skipping"
fi

# ------------------------------------------------------------------
# 清理旧的冗余文件 (已集成到 src/mjlab 中)
# ------------------------------------------------------------------
echo ""
echo "Cleaning up redundant files..."
rm -rf "$PROJ_DIR/configs" "$PROJ_DIR/assets" "$PROJ_DIR/motions"
echo "  -> Removed old configs/, assets/, motions/ (now inside src/mjlab/)"

# ------------------------------------------------------------------
# 统计
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo " Package build complete!"
echo "=========================================="
TOTAL_FILES=$(find "$PROJ_DIR" -type f | wc -l)
TOTAL_SIZE=$(du -sh "$PROJ_DIR" | cut -f1)
echo "Total files: $TOTAL_FILES"
echo "Total size:  $TOTAL_SIZE"
echo ""
echo "Project structure:"
echo "  src/mjlab/                  <- 训练框架 (mjlab完整源码，含MuJoCo/AMP/T1)"
echo "  general_motion_retargeting/ <- 动作重定向 (GMR完整源码)"
echo "  gmr_assets/                 <- GMR 机器人模型 (T1 URDF/mesh)"
echo "  scripts/                    <- sim2sim 键盘控制脚本"
echo "  retarget/                   <- 重定向脚本 + IK配置"
echo "  checkpoints/                <- 预训练策略"
echo ""
echo "Next steps:"
echo "  pip install -e .            # 安装 mjlab + GMR + 所有依赖"
echo "  # 或者"
echo "  pip install uv && uv sync   # 使用 UV 安装"
