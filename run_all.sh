#!/usr/bin/env bash
# N6 v1.2 一键复现：校验 → v1.0 基线 → v1.1 深化 A/B/C → v1.2 深化 D/E/F → 汇总（纯 CPU）
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
echo "== pytest =="
"$PY" -m pytest tests/ -q
for s in 00_verify_env 01_baseline_validation 02_param_sweep 03_cooling_design 04_animation 05_summary 06_axisym_spreading 07_nonlinear_waveform 08_inverse_identification 09_summary_v11 10_nonlinear_transient 11_dynamic_spreading 12_radiative_contact 13_summary_v12; do
  echo "== scripts/$s.py =="
  "$PY" "scripts/$s.py"
done
echo "== done: results/ + figures/ 已生成 =="
