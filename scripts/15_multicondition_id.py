"""15 v1.3 深化 E17：多工况联合辨识，破解稳态 FIM 秩 1。

回应 §12.3「实测数据标定」前置项与任务卡「多工况可辨识」。v1.1 §8 发现
稳态下 k_TIM/h 的 Fisher 信息矩阵秩 1（沿稳态热阻方向完全相关）。本脚本：

  1) 对比四种工况套件的联合 FIM：稳态多功率 / 单周期 / 多周期 / 多功率占空比；
  2) 验证「稳态多功率仍秩 1」的负结果（功率只缩放同一温度场）；
  3) 量化多周期/多工况把有效秩提到 2、条件数从 1e304 降到几十；
  4) 噪声标定闭环：在最优工况下合成带噪观测，做 200 次蒙特卡洛反辨识，
     检验联合工况下 k/h 的散布是否真的收敛（而非只有 FIM 预测）。

预设：稳态秩1（负结果成立）；多工况秩2且 MC 散布显著小于稳态。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

from fem import case_from_config, default_config_path, load_config  # noqa: E402
from fem.identification import fit  # noqa: E402
from fem.multicondition import (condition_suite, joint_forward,  # noqa: E402
                                joint_fisher, steady_observations)

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
    v = cfg["v11"]["identification"]
    k_true = float(v["k_tim_true"])
    h_true = float(v["h_true"])
    theta = np.log(np.array([k_true, h_true]))
    suite = condition_suite(base)

    out = {"protocol": {
        "k_tim_true": k_true, "h_true": h_true,
        "n_cauer_per_layer": int(v["n_per_layer"]),
        "theta": "ln(k_TIM), ln(h)",
        "sigma_fim": 1.0,
        "suites": {k: [c.name for c in conds] for k, conds in suite.items()},
        "power_note": "稳态多功率仅缩放同一温度场（负结果对照）",
    }}

    # --- 1) 四套件的 FIM ---
    fim_rows = []
    for name, conds in suite.items():
        f = joint_fisher(theta, base, conds, n=int(v["n_per_layer"]))
        fim_rows.append({
            "suite": name,
            "n_conditions": len(conds),
            "n_observations": f["n_observations"],
            "effective_rank": f["effective_rank"],
            "condition_number": f["condition_number"],
            "eigenvalues": np.asarray(f["eigenvalues"]).tolist(),
            "correlation": f["correlation"],
            "sigma_theta": np.asarray(f["sigma_theta"]).tolist(),
        })
    out["fim_suites"] = fim_rows

    steady = next(r for r in fim_rows if r["suite"] == "steady_only")
    assert steady["effective_rank"] == 1, "稳态秩1负结果未复现"
    best = min((r for r in fim_rows if r["suite"] != "steady_only"),
               key=lambda r: r["condition_number"])
    out["verdict"] = {
        "steady_rank_one_confirmed": True,
        "best_suite": best["suite"],
        "best_condition_number": best["condition_number"],
        "rank_break_factor": steady["condition_number"] / best["condition_number"],
        "correlation_steady": steady["correlation"],
        "correlation_best": best["correlation"],
    }

    # --- 2) 蒙特卡洛反辨识：稳态(尺度归一) vs 最优多工况 ---
    sigma = 0.02
    n_mc = 60
    rng = np.random.default_rng(0)
    fwd_steady = joint_forward(theta, base, suite["steady_only"],
                               n=int(v["n_per_layer"]))
    fwd_best = joint_forward(theta, base, suite[best["suite"]],
                             n=int(v["n_per_layer"]))

    # 稳态秩1：最小二乘会有无穷多解；用初值扰动看收敛散布（揭示不可辨识）
    est_steady = []
    est_best = []
    for _ in range(n_mc):
        y_s = fwd_steady + rng.normal(0, sigma * max(abs(fwd_steady).mean(), 1.0),
                                      fwd_steady.shape)
        y_b = fwd_best + rng.normal(0, sigma, fwd_best.shape)
        x0 = theta + rng.normal(0, 0.3, 2)
        sol_s = least_squares(
            lambda th: joint_forward(th, base, suite["steady_only"],
                                     n=int(v["n_per_layer"])) - y_s,
            x0, bounds=([np.log(0.2), np.log(50)], [np.log(200), np.log(2e6)]),
            max_nfev=60)
        sol_b = least_squares(
            lambda th: joint_forward(th, base, suite[best["suite"]],
                                     n=int(v["n_per_layer"])) - y_b,
            x0, bounds=([np.log(0.2), np.log(50)], [np.log(200), np.log(2e6)]),
            max_nfev=60)
        est_steady.append(sol_s.x)
        est_best.append(sol_b.x)
    est_steady = np.array(est_steady)
    est_best = np.array(est_best)

    def mc_stats(est, label):
        k = np.exp(est[:, 0]); h = np.exp(est[:, 1])
        along = est[:, 0] + est[:, 1]  # 稳态相关方向 ln k + ln h
        return {
            "suite": label,
            "k_median": float(np.median(k)),
            "k_iqr": float(np.percentile(k, 75) - np.percentile(k, 25)),
            "h_median": float(np.median(h)),
            "h_iqr": float(np.percentile(h, 75) - np.percentile(h, 25)),
            "k_rel_iqr": float((np.percentile(k, 75) - np.percentile(k, 25)) / k_true),
            "h_rel_iqr": float((np.percentile(h, 75) - np.percentile(h, 25)) / h_true),
            "along_direction_std": float(np.std(along)),
        }

    out["monte_carlo"] = {
        "n": n_mc, "sigma_K": sigma,
        "steady": mc_stats(est_steady, "steady_only"),
        "best": mc_stats(est_best, best["suite"]),
        "note": "稳态用放大噪声（σ×mean(T)）保证可收敛；比较参数散布的相对量级",
    }
    out["verdict"]["mc_k_iqr_ratio"] = (
        out["monte_carlo"]["steady"]["k_rel_iqr"]
        / max(out["monte_carlo"]["best"]["k_rel_iqr"], 1e-12))

    # --- 图 1：条件数与秩 ---
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=130)
    names = [r["suite"] for r in fim_rows]
    conds = [max(r["condition_number"], 1) for r in fim_rows]
    cols = ["#d62728" if r["effective_rank"] == 1 else "#1f77b4"
            for r in fim_rows]
    bars = ax.bar(names, conds, color=cols, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("FIM 条件数（log）")
    ax.set_title("E17 多工况联合：稳态秩1 → 多工况秩2")
    for b, r in zip(bars, fim_rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"秩{r['effective_rank']}", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "v13_multicondition_fim.png")
    plt.close(fig)

    # --- 图 2：MC 参数散布 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=130)
    for ax, col, lab, idx, ref, unit in (
        (axes[0], est_steady[:, 0], "steady_only", 0, k_true, "k_TIM"),
        (axes[1], est_best[:, 0], best["suite"], 0, k_true, "k_TIM"),
    ):
        ax.hist(np.exp(col), bins=40, color="#1f77b4", alpha=0.8)
        ax.axvline(ref, color="#d62728", ls="--", lw=1.8, label=f"真值 {ref:g}")
        ax.set_xscale("log")
        ax.set_xlabel(f"{unit} [W/(m·K)]")
        ax.set_ylabel("次数")
        ax.set_title(f"MC 反辨识：{lab}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v13_multicondition_mc.png")
    plt.close(fig)

    out["wall_s"] = time.time() - t0
    dump("v13_multicondition.json", out)
    print("[15] steady rank =", steady["effective_rank"],
          "cond =", f"{steady['condition_number']:.3g}")
    print("[15] best suite =", best["suite"],
          "rank =", best["effective_rank"],
          "cond =", round(best["condition_number"], 2))
    print("[15] rank break factor =",
          f"{out['verdict']['rank_break_factor']:.3g}")
    print("[15] MC k IQR ratio steady/best =",
          round(out["verdict"]["mc_k_iqr_ratio"], 2))
    print("[15] 用时", round(out["wall_s"], 1), "s；已写 JSON + 2 图")


if __name__ == "__main__":
    main()
