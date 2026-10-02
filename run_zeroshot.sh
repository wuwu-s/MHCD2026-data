#!/usr/bin/env bash
# SAM3 零样本直测 MHCD2022 → 生产 mask 标注
# 用法: bash data/run_zeroshot.sh    （日志: {OUT}/run.log）
set -e
cd "$(dirname "$0")"

PY=/root/miniconda3/envs/sam3/bin/python
OUT=/root/autodl-tmp/sam3_masks_mhcd2022

if ! nvidia-smi >/dev/null 2>&1; then
  echo "❌ 未检测到 GPU（nvidia-smi 无设备）。"
  exit 1
fi

mkdir -p "$OUT"
nohup "$PY" sam3_zeroshot_annotate.py --out "$OUT" > "$OUT/run.log" 2>&1 &
echo "已后台启动，日志: $OUT/run.log"
echo "查看进度: tail -f $OUT/run.log"
