#!/bin/bash
# Batch retarget ACCAD walking motions to Booster T1 (23 DOF)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACCAD_WALK="$SCRIPT_DIR/../ACCAD/Female1Walking_c3d"
OUT_DIR="$SCRIPT_DIR/../t1_retarget_output"

declare -A MOTIONS
MOTIONS["female_walk1"]="$ACCAD_WALK/B3_-_walk1_stageii.npz"
MOTIONS["female_stand_to_walk"]="$ACCAD_WALK/B1_-_stand_to_walk_stageii.npz"
MOTIONS["female_walk_to_stand"]="$ACCAD_WALK/B2_-_walk_to_stand_stageii.npz"
MOTIONS["female_walk_backwards"]="$ACCAD_WALK/B5_-_walk_backwards_stageii.npz"
MOTIONS["female_walk_turn_left_90"]="$ACCAD_WALK/B9_-_walk_turn_left_(90)_stageii.npz"
MOTIONS["female_walk_turn_left_45"]="$ACCAD_WALK/B10_-_walk_turn_left_(45)_stageii.npz"
MOTIONS["female_walk_turn_right_90"]="$ACCAD_WALK/B12_-_walk_turn_right_(90)_stageii.npz"
MOTIONS["female_walk_turn_right_45"]="$ACCAD_WALK/B13_-_walk_turn_right_(45)_stageii.npz"

for name in "${!MOTIONS[@]}"; do
  src="${MOTIONS[$name]}"
  dst="$OUT_DIR/${name}.pkl"
  echo "=== Retargeting: $name ==="
  echo "  src: $src"
  echo "  dst: $dst"
  python "$SCRIPT_DIR/smplx_to_robot.py" \
    --smplx_file "$src" \
    --robot booster_t1 \
    --save_path "$dst"
  echo "  Done: $dst"
  echo ""
done

echo "All retargeting complete. Output in: $OUT_DIR"
