#!/usr/bin/env bash
set -euo pipefail

arm="${1:-sheaf_gbp}"
python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
source_checkpoint="${SOURCE_CHECKPOINT:-runs/map_medium_12/${arm}/${arm}_best.pt}"
out_dir="${OUT_DIR:-runs/map_medium_8/${arm}}"

if [[ ! -f "$source_checkpoint" ]]; then
  echo "missing 12-agent checkpoint: $source_checkpoint" >&2
  echo "run scripts/train_map_medium_12.sh $arm first" >&2
  exit 2
fi

case "$arm" in
  sheaf_gbp)
    arm_args=(
      --gbp-steps 1
      --edge-dim 12
      --temporal-window 4
      --temporal-precision 0.25
      --restriction-residual-scale 0.08
      --force-residual-scale 1.0
      --analytic-force-scale 0.25
      --decoder-extra-context
      --edge-conditioned-restrictions
    )
    ;;
  comm_matched|raw_full)
    arm_args=(--edge-dim 12)
    ;;
  *)
    echo "transfer is defined for sheaf_gbp, raw_full, and comm_matched" >&2
    exit 2
    ;;
esac

"$python_bin" -m piano_movers.train \
  --arm "$arm" \
  --device "$device" \
  --out-dir "$out_dir" \
  --init-checkpoint "$source_checkpoint" \
  --allow-init-env-config-mismatch \
  --allow-init-arm-config-mismatch \
  --init-compatible-weights-only \
  --hidden-dim 128 \
  --comm-rounds 1 \
  --n-agents 8 \
  --max-force 2.55 \
  --world-x 3.90 \
  --world-y 1.60 \
  --wall-x-abs 0.95 \
  --source-x -2.65 \
  --goal-x 2.65 \
  --max-steps 110 \
  --comm-radius 0.90 \
  --gap-half-base 0.78 \
  --gap-half-difficulty-scale 0.06 \
  --gap-half-jitter 0.05 \
  --steps 2000 \
  --batch-size 768 \
  --eval-every 100 \
  --eval-episodes 1024 \
  --eval-batch-size 512 \
  --dagger-start 50 \
  --dagger-every 1 \
  --dagger-rollout-steps 110 \
  --rollout-loss-weight 1.0 \
  --rollout-horizon 30 \
  --rollout-batch-size 256 \
  --rollout-initial-fraction 0.50 \
  --rollout-state-weight 5.0 \
  --rollout-target-weight 0.05 \
  --rollout-progress-weight 2.0 \
  --rollout-safety-weight 8.0 \
  --rollout-source-margin-weight 5.0 \
  --rollout-source-margin 0.012 \
  --rollout-clearance-margin-weight 50.0 \
  --rollout-clearance-margin 0.002 \
  --rollout-goal-capture-weight 1.0 \
  --scenario-difficulty 3.0 \
  --strict-local-observation \
  --wall-sensor-range 1.05 \
  --seed 20261871 \
  "${arm_args[@]}"
