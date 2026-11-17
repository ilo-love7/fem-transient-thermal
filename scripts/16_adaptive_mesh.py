"""16 v1.3 深化 E18：后验误差指示子驱动的 h-自适应网格。

回应任务卡「网格自适应」与 §12.3「完整三维/非轴对称热源」的二维对照。
对局部热源（a/R=0.2~0.6）与全宽热源（严格 1D，负结果对照）：

  1) 粗网格 (nx=24) 稳态解 -> 单元能量范数指示子 eta=Ae|grad T|²；
  2) 聚合到 x 列，标记 top fraction 区间插入中点 -> 自适应网格；
  3) 与**同自由度**均匀网格比较结温（参考解 nx=240），量化精度增益；
  4) 扫描热源宽度与标记比例，给出自适应的有效边界。

预设：局部热源下自适应优于同自由度均匀；全宽热源下 x 加密零收益（负结果）。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fem import case_from_config, default_config_path, load_config  # noqa: E402
from fem.adaptive import adaptivity_study  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RES = ROOT / "results"
FIG = ROOT / "figures"


def dump(name, obj):
    (RES / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")


def main():
    t0 = time.time()
    cfg = load_config(default_config_path())
    base = case_from_config(cfg)

    out = {"protocol": {
        "coarse_nx": 24, "reference_nx": 240, "ny_per_layer": 8,
        "error_metric": "|Tj_adapt - Tj_ref| / (Tj_ref - T_amb)",
        "indicator": "eta_e = Ae * |grad T_e|^2 (P1 精确)",
        "source_ratios": [1.0, 0.6, 0.4, 0.2],
        "refine_fractions": [0.25, 0.5, 0.75],
    }}

    # --- 1) 热源宽度 × 标记比例扫描 ---
    scan = []
    for ratio in (1.0, 0.6, 0.4, 0.2):
        case = replace(base, heat_source_width=ratio * base.width,
                       nx=24, ny_per_layer=8)
        for frac in (0.25, 0.5, 0.75):
            r = adaptivity_study(case, coarse_nx=24, refine_fraction=frac,
                                 reference_nx=240)
            scan.append({
                "source_ratio": ratio, "refine_fraction": frac,
                "adaptive_nx": r["adaptive_nx"],
                "rel_err_coarse": r["rel_err_coarse"],
                "rel_err_adaptive": r["rel_err_adaptive"],
                "rel_err_uniform_same_dof": r["rel_err_uniform_same_dof"],
                "speedup_over_uniform": r["speedup_over_uniform"],
                "refined_near_heat_source": r["refined_near_heat_source"],
                "interval_eta_max_over_min": r["interval_eta_max_over_min"],
                "t_junction_ref_C": r["t_junction_ref_C"],
            })
    out["scan"] = scan

    # --- 2) 裁决 ---
    local = [r for r in scan if r["source_ratio"] < 1.0]
    full = [r for r in scan if r["source_ratio"] == 1.0]
    wins = [r for r in local if r["rel_err_adaptive"] < r["rel_err_uniform_same_dof"]]
    best = max(local, key=lambda r: r["speedup_over_uniform"])
    full_max_err = max(max(r["rel_err_adaptive"], r["rel_err_uniform_same_dof"])
                       for r in full)
    out["verdict"] = {
        "local_source_adaptive_wins_count": len(wins),
        "local_source_total": len(local),
        "best": {"source_ratio": best["source_ratio"],
                 "refine_fraction": best["refine_fraction"],
                 "speedup_over_uniform": best["speedup_over_uniform"],
                 "rel_err_adaptive": best["rel_err_adaptive"],
                 "rel_err_uniform": best["rel_err_uniform_same_dof"]},
        "full_width_max_rel_err": full_max_err,
        "full_width_zero_gain": full_max_err < 2e-5,
        "note": "全宽热源=严格1D，x 加密在数值精度内无收益（负结果）",
    }

    # --- 图 1：热源宽度扫描 ---
    ratios = sorted(set(r["source_ratio"] for r in scan))
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=130)
    for frac, col, ls in ((0.25, "#1f77b4", ":"),
                          (0.5, "#2ca02c", "-"),
                          (0.75, "#9467bd", "--")):
        rr = [r for r in scan if abs(r["refine_fraction"] - frac) < 1e-9]
        ax.plot([r["source_ratio"] for r in rr],
                [r["rel_err_adaptive"] for r in rr], ls, color=col,
                marker="o", lw=1.8, label=f"自适应 frac={frac}")
        ax.plot([r["source_ratio"] for r in rr],
                [r["rel_err_uniform_same_dof"] for r in rr], ls, color=col,
                marker="x", alpha=0.5, lw=1.2)
    ax.set_xlabel("热源宽度比 a/R")
    ax.set_ylabel("结温相对误差")
    ax.set_yscale("log")
    ax.set_title("E18 自适应网格 vs 同自由度均匀（○=自适应，×=均匀）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v13_adaptive_sweep.png")
    plt.close(fig)

    # --- 图 2：最佳工况的精度-自由度对照 ---
    case = replace(base, heat_source_width=best["source_ratio"] * base.width,
                   nx=24, ny_per_layer=8)
    from fem.adaptive import (build_mesh_nonuniform, gradient_indicator,
                              interval_indicators, refine_intervals)
    from fem.assembly import assemble, solve_steady
    coarse_x = np.linspace(0, base.width, 25)
    ref_x = np.linspace(0, base.width, 241)
    m0 = build_mesh_nonuniform(case, coarse_x)
    T0 = solve_steady(assemble(case, m0), case.power_per_length)
    eta = interval_indicators(m0, gradient_indicator(m0, T0))
    x_ad, marked = refine_intervals(coarse_x, eta, best["refine_fraction"])
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=130)
    ax.bar(np.arange(m0.nx), eta / max(eta.max(), 1e-30),
           color=["#d62728" if i in marked else "#cccccc" for i in range(m0.nx)])
    ax.axvline((m0.nx + 1) * 0.5 - (m0.nx + 1) * best["source_ratio"] / 2,
               color="#1f77b4", ls="--", lw=1.5, label="热源左边缘")
    ax.axvline((m0.nx + 1) * 0.5 + (m0.nx + 1) * best["source_ratio"] / 2,
               color="#1f77b4", ls="--", lw=1.2)
    ax.set_xlabel("x 区间索引")
    ax.set_ylabel("归一化误差指示子 eta")
    ax.set_title(f"后验指示子与标记区间（a/R={best['source_ratio']:g}）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v13_adaptive_indicator.png")
    plt.close(fig)

    out["wall_s"] = time.time() - t0
    dump("v13_adaptive.json", out)
    print("[16] local wins =", out["verdict"]["local_source_adaptive_wins_count"],
          "/", out["verdict"]["local_source_total"])
    print("[16] best speedup =", round(best["speedup_over_uniform"], 2),
          "× at a/R =", best["source_ratio"], "frac =", best["refine_fraction"])
    print("[16] full-width max rel err =", full_max_err,
          "zero gain =", out["verdict"]["full_width_zero_gain"])
    print("[16] 用时", round(out["wall_s"], 1), "s；已写 JSON + 2 图")


if __name__ == "__main__":
    main()
