#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  Reproduce Table 1: Cross-Center Diagnostic Performance         ║
# ║  Author: Teerapong Panboonyuen (Kao Panboonyuen)               ║
# ╚══════════════════════════════════════════════════════════════════╝
set -e

DATA_ROOT="data/cervical_cytology"
OUT_DIR="outputs"
DEVICE="cuda"

BACKBONES=("resnet18" "resnet50" "densenet121" "vit_l16")

echo "============================================================"
echo "  TABLE 1: Cross-Center Diagnostic Performance"
echo "============================================================"

for BACKBONE in "${BACKBONES[@]}"; do

  echo ""
  echo ">>> Training BASELINE: $BACKBONE"
  python scripts/train.py \
    --backbone "$BACKBONE" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUT_DIR/baseline_${BACKBONE}" \
    --device "$DEVICE" --amp

  echo ">>> Training CERVSHORT: $BACKBONE"
  python scripts/train.py \
    --backbone "$BACKBONE" \
    --use_cervshort \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUT_DIR/cervshort_${BACKBONE}" \
    --device "$DEVICE" --amp

  echo ">>> Evaluating BASELINE: $BACKBONE"
  python scripts/evaluate.py \
    --checkpoint "$OUT_DIR/baseline_${BACKBONE}/best.pth" \
    --backbone "$BACKBONE" \
    --data_root "$DATA_ROOT" \
    --output_json "$OUT_DIR/results_baseline_${BACKBONE}.json" \
    --device "$DEVICE"

  echo ">>> Evaluating CERVSHORT: $BACKBONE"
  python scripts/evaluate.py \
    --checkpoint "$OUT_DIR/cervshort_${BACKBONE}/best.pth" \
    --backbone "$BACKBONE" \
    --use_cervshort \
    --data_root "$DATA_ROOT" \
    --output_json "$OUT_DIR/results_cervshort_${BACKBONE}.json" \
    --device "$DEVICE"

done

echo ""
echo "Table 1 reproduction complete. Results in outputs/results_*.json"
