#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  Reproduce Table 3: Ablation Study                             ║
# ║  Author: Teerapong Panboonyuen (Kao Panboonyuen)               ║
# ╚══════════════════════════════════════════════════════════════════╝
set -e
echo "TABLE 3: Ablation Study"
python scripts/ablation.py \
  --data_root data/cervical_cytology \
  --checkpoint_dir outputs/ablation_checkpoints \
  --device cuda
