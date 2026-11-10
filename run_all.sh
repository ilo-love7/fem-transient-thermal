#!/usr/bin/env bash
# N6 v1.1 一键复现：校验 → v1.0 基线 → v1.1 深化 A/B/C → 汇总（纯 CPU，约 5 分钟）
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
echo "== pytest =="
"$PY" -m pytest tests/ -q
for s in 00_verify_env 01_baseline_validation 02_param_sweep 03_cooling_design 04_animation 05_summary 06_axisym_spreading 07_nonlinear_waveform 08_inverse_identification 09_summary_v11; do
  echo "== scripts/$s.py =="
  "$PY" "scripts/$s.py"
done
echo "== done: results/ + figures/ 已生成 =="
