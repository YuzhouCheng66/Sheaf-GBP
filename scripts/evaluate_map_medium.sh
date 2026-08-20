#!/usr/bin/env bash
set -euo pipefail

agents="${1:-12}"
python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
episodes="${EPISODES:-4096}"

case "$agents" in
  12) seed=20261791 ;;
  8) seed=20261931 ;;
  *)
    echo "supported agent counts: 8 or 12" >&2
    exit 2
    ;;
esac

run_root="runs/map_medium_${agents}"
out_path="artifacts/map_medium_${agents}_paired_${episodes}.json"
mkdir -p "$(dirname "$out_path")"

"$python_bin" -m piano_movers.compare \
  --checkpoints \
    "$run_root/sheaf_gbp/sheaf_gbp_best.pt" \
    "$run_root/raw_full/raw_full_best.pt" \
    "$run_root/comm_matched/comm_matched_best.pt" \
  --device "$device" \
  --episodes "$episodes" \
  --batch-size 512 \
  --seed "$seed" \
  --out "$out_path"
