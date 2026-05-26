#!/usr/bin/env bash
# Overfitting ablation — fold 0 only, 40 epochs each
# Tests: model size (d_h) × CBFC × dropout
# Run: bash ablation_overfit.sh 2>&1 | tee ablation_overfit.log

set -e
BASE="conda run -n hopeful python train_iemocap.py --max_folds 1 --epochs 40 --batch_size 12"

echo "========================================"
echo "A: baseline  d_h=512 dropout=0.3 cbfc=ON   (reference)"
echo "========================================"
$BASE --d_h 512 --dropout 0.3

echo "========================================"
echo "B: dh256     d_h=256 dropout=0.3 cbfc=ON   (size fix)"
echo "========================================"
$BASE --d_h 256 --dropout 0.3

echo "========================================"
echo "C: nocbfc    d_h=512 dropout=0.3 cbfc=OFF  (CBFC isolated)"
echo "========================================"
$BASE --d_h 512 --dropout 0.3 --no_cbfc

echo "========================================"
echo "D: dh256_nocbfc  d_h=256 cbfc=OFF          (both)"
echo "========================================"
$BASE --d_h 256 --dropout 0.3 --no_cbfc

echo "========================================"
echo "E: dh256_drop05  d_h=256 dropout=0.5 cbfc=ON  (more dropout)"
echo "========================================"
$BASE --d_h 256 --dropout 0.5

echo "========================================"
echo "All ablations complete."
echo "========================================"
