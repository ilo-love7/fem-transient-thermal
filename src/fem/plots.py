"""可视化：温度场热图、结温瞬态、误差/收敛曲线、参数热力图、散热方案对比。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from .materials import ThermalCase
from .mesh import StructuredMesh
from .solver import TransientResult


def plot_field(mesh: StructuredMesh, T: np.ndarray, case: ThermalCase,
               title: str, path: Path, cmap: str = "inferno"):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
    cn = ax.tricontourf(mesh.nodes[:, 0] * 1e3, mesh.nodes[:, 1] * 1e3,
                        mesh.elems, T, levels=24, cmap=cmap)
    for y in case.y_interfaces[1:-1]:
        ax.axhline(y * 1e3, color="w", lw=0.8, ls="--", alpha=0.6)
    x0, x1 = case.heat_source_bounds
    if x1 - x0 < case.width - 1e-12:
        ax.plot([x0 * 1e3, x1 * 1e3], [case.height * 1e3] * 2,
                color="cyan", lw=3, solid_capstyle="butt")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    fig.colorbar(cn, ax=ax, label="T [°C]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_transient(curves: dict[str, TransientResult], title: str, path: Path,
                   show_power: bool = True):
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, (name, r) in enumerate(curves.items()):
        ax.plot(r.t, r.t_junction, lw=1.8 if i == 0 else 1.3,
                color=colors[i % len(colors)], label=name,
                ls="-" if i == 0 else "--", alpha=0.95 if i == 0 else 0.85)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("结温 T_j [°C]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if show_power:
        ax2 = ax.twinx()
        r0 = next(iter(curves.values()))
        ax2.fill_between(r0.t, 0, r0.power / max(r0.power.max(), 1e-9),
                         color="gray", alpha=0.12, step="post", label="脉冲功率")
        ax2.set_ylabel("功率（归一化）", color="gray")
        ax2.set_ylim(0, 3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_convergence(x: np.ndarray, err: np.ndarray, p: float,
                     xlabel: str, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.loglog(x, err, "o-", color="#1f77b4", label="实测误差")
    if np.isfinite(p) and len(x) >= 2:
        c = err[0] / x[0] ** p
        ax.loglog(x, c * np.array(x) ** p, "k--", lw=1, label=f"斜率 p={p:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("误差 [K]")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_heatmap(x: np.ndarray, y: np.ndarray, Z: np.ndarray,
                 xlabel: str, ylabel: str, title: str, path: Path,
                 zlabel: str = "T_j 峰值 [°C]", cmap: str = "inferno"):
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=120)
    im = ax.pcolormesh(x, y, Z, shading="auto", cmap=cmap)
    cs = ax.contour(x, y, Z, levels=8, colors="w", linewidths=0.6, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=zlabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars(names: list[str], values: np.ndarray, ylabel: str,
              title: str, path: Path, color: str = "#1f77b4"):
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=120)
    bars = ax.bar(names, values, color=color, alpha=0.85)
    ax.bar_label(bars, fmt="%.1f", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_error_vs_h(h: np.ndarray, err: np.ndarray, bi: np.ndarray,
                    title: str, path: Path, threshold: float = 5.0):
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=120)
    ax.semilogx(h, err, "o-", color="#d62728", label="单节 Cauer 峰值误差")
    ax.axhline(threshold, color="k", ls="--", lw=1, label=f"{threshold:.0f}% 目标线")
    ax.set_xlabel("对流系数 h [W/(m² K)]")
    ax.set_ylabel("周期稳态峰值相对误差 [%]")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax2 = ax.twiny()
    ax2.semilogx(bi, err, alpha=0)
    ax2.set_xlabel("TIM 层 Bi = h·L/k")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def animate_field(mesh: StructuredMesh, case: ThermalCase,
                  snaps: dict[float, np.ndarray], path: Path,
                  vmin: float | None = None, vmax: float | None = None,
                  fps: int = 8):
    """温度场 GIF。matplotlib 3.11 的 TriContourSet 不再暴露 .collections，
    故每帧清空坐标轴重画（帧数 <=100 时开销可忽略，且最稳健）。"""
    times = sorted(snaps)
    allT = np.concatenate([snaps[t] for t in times])
    vmin = float(allT.min()) if vmin is None else vmin
    vmax = float(allT.max()) if vmax is None else vmax
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=100)
    cn0 = ax.tricontourf(mesh.nodes[:, 0] * 1e3, mesh.nodes[:, 1] * 1e3,
                         mesh.elems, snaps[times[0]], levels=24,
                         cmap="inferno", vmin=vmin, vmax=vmax)
    fig.colorbar(cn0, ax=ax, label="T [°C]")
    x_km, y_km = mesh.nodes[:, 0] * 1e3, mesh.nodes[:, 1] * 1e3

    def update(i):
        ax.clear()
        ax.tricontourf(x_km, y_km, mesh.elems, snaps[times[i]], levels=24,
                       cmap="inferno", vmin=vmin, vmax=vmax)
        for y in case.y_interfaces[1:-1]:
            ax.axhline(y * 1e3, color="w", lw=0.8, ls="--", alpha=0.6)
        x0, x1 = case.heat_source_bounds
        ax.plot([x0 * 1e3, x1 * 1e3], [case.height * 1e3] * 2,
                color="cyan", lw=3, solid_capstyle="butt")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"二维局部热源温度场 · t = {times[i]:.2f} s")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(times),
                                   interval=1000 // fps, blit=False)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)
