# -*- coding: utf-8 -*-
"""make_figures.py — 生成 docs/decoding-internals.md 的四联图（真实数据版）。

Panel A: HMM 状态构成（min_intron_length=20, S=170，实测计数）
Panel B: 转移矩阵稀疏度——A 条件矩阵的 28,900 个格子中只有少数有限边（真实边表数据）
Panel C: Viterbi 内循环每碱基工作量（稠密 S² vs 稀疏 E/5，对数轴）
Panel D: 端到端解码耗时基准（参考实现 vs 本仓库，24 线程）

用法：python make_figures.py  （在 docs/assets/ 下运行，输出 decoding_figures.png）
"""
import os
import sys
import types

# ---- numba 桩：本脚本只用矩阵构建函数（纯 numpy），JIT 内核不需要执行 ----
_numba = types.ModuleType("numba")


def _njit(*args, **kwargs):
    def deco(fn):
        return fn
    if args and callable(args[0]):
        return args[0]
    return deco


_numba.njit = _njit
sys.modules.setdefault("numba", _numba)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.HMM import _build_hmm_artifacts  # noqa: E402

# ---- 真实数据 ----
ART = _build_hmm_artifacts(20, None, None)
S = ART["num_states"]
E_TOTAL = ART["n_edges"]
SYM = 0  # A 条件矩阵
counts_sym = int(ART["to_ptr"][SYM, -1] - ART["to_ptr"][SYM, 0])
to_idx = np.repeat(np.arange(S), np.diff(ART["to_ptr"][SYM]))
fr_idx = ART["edge_from"][ART["to_ptr"][SYM, 0]:ART["to_ptr"][SYM, -1]]
DENSE = S * S
SPARSE_AVG = E_TOTAL / 5

# ---- 配色 ----
C_REF = "#9aa5b1"      # 参考实现 / 稠密
C_NEW = "#2e6f9e"      # 本仓库 / 稀疏
C_HOT = "#c74440"      # 强调（计数器、倍数）
C_GRID = "#d8dde3"

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
})

fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6))
fig.subplots_adjust(hspace=0.42, wspace=0.28)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#666")
    ax.spines["bottom"].set_color("#666")
    ax.tick_params(colors="#444")


# ================= Panel A: 状态构成 =================
ax = axes[0, 0]
families = [
    ("Intergenic", 1), ("Start codon (ATG)", 3), ("CDS (phase-tagged)", 6),
    ("DSS / ASS motifs", 12), ("Stop codon", 4), ("Splice helpers", 24),
    ("Intron length counters", 120),
]
labels = [f[0] for f in families][::-1]
vals = [f[1] for f in families][::-1]
colors = [C_REF] * 6 + [C_HOT]
bars = ax.barh(labels, vals, color=colors, height=0.62, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_width() + 2.5, b.get_y() + b.get_height() / 2, str(v),
            va="center", fontsize=9.5, color="#333", zorder=4)
ax.annotate("grammar states: 50", xy=(30, 1.0), fontsize=9.5, color="#555",
            style="italic")
ax.annotate("counters: 120\n(70% of S)", xy=(78, 5.4), fontsize=9.5,
            color=C_HOT, fontweight="bold")
ax.set_xlim(0, 140)
ax.set_xlabel("states")
ax.set_title("A  What the 170-state machine is made of", loc="left",
             fontweight="bold")
_style(ax)
ax.grid(axis="x", color=C_GRID, lw=0.7, zorder=0)

# ================= Panel B: 稀疏度（真实边表） =================
ax = axes[0, 1]
ax.set_facecolor("#fbfcfd")
# 空格子底纹：用极淡网格表示 S×S
ax.set_xlim(0, S)
ax.set_ylim(0, S)
ax.scatter(fr_idx, to_idx, s=2.4, c=C_NEW, alpha=0.75, linewidths=0, zorder=3)
ax.set_xticks([0, 40, 80, 120, 169])
ax.set_yticks([0, 40, 80, 120, 169])
ax.set_xticklabels(["0", "40", "80", "120", "S−1"])
ax.set_yticklabels(["0", "40", "80", "120", "S−1"])
ax.set_xlabel("from state")
ax.set_ylabel("to state")
ax.set_title("B  The transition matrix is almost empty", loc="left",
             fontweight="bold")
ax.text(0.03, 0.05,
        f"A-conditional matrix:\n{counts_sym:,} finite edges out of {DENSE:,} cells"
        f" ({100*counts_sym/DENSE:.1f}%)",
        transform=ax.transAxes, fontsize=9.5, va="bottom", color="#333",
        bbox=dict(facecolor="white", edgecolor=C_GRID, alpha=0.9, pad=4))
ax.text(0.97, 0.955,
        "5 conditional matrices (A/T/C/G/N)\nshare 879 edges in total",
        transform=ax.transAxes, fontsize=9, va="top", ha="right", color="#555",
        style="italic")
_style(ax)

# ================= Panel C: 每碱基工作量 =================
ax = axes[1, 0]
names = ["dense O(L·S²)\nS² = 170² comparisons", "sparse O(L·E)\n≈ E/5 edge visits"]
vals2 = [DENSE, SPARSE_AVG]
bars = ax.bar(names, vals2, color=[C_REF, C_NEW], width=0.5, zorder=3)
ax.set_yscale("log")
ax.set_ylim(60, 4e5)
for b, v in zip(bars, vals2):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.30, f"{v:,.0f}",
            ha="center", fontsize=10, color="#333", zorder=4)
ax.annotate("", xy=(1, 320), xytext=(0, 42000),
            arrowprops=dict(arrowstyle="-|>", color=C_HOT, lw=1.6,
                            connectionstyle="arc3,rad=-0.18"))
ax.text(0.52, 0.50, "≈ 164× fewer", transform=ax.transAxes, ha="center",
        fontsize=13, color=C_HOT, fontweight="bold")
ax.text(0.52, 0.40, "candidate visits per base", transform=ax.transAxes,
        ha="center", fontsize=9, color="#666")
ax.set_ylabel("work per base (log scale)")
ax.set_title("C  What the rewrite saves per base", loc="left",
             fontweight="bold")
_style(ax)
ax.grid(axis="y", color=C_GRID, lw=0.7, zorder=0)

# ================= Panel D: 端到端基准 =================
ax = axes[1, 1]
x = np.arange(2)
ref = [17.9, 59.4]
fast = [3.4, 22.3]
w = 0.34
b1 = ax.bar(x - w / 2, ref, w, label="reference decoder", color=C_REF, zorder=3)
b2 = ax.bar(x + w / 2, fast, w, label="this repo", color=C_NEW, zorder=3)
for b, v in zip(b1, ref):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v}s", ha="center",
            fontsize=9.5, color="#333", zorder=4)
for b, v in zip(b2, fast):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v}s", ha="center",
            fontsize=9.5, color="#333", zorder=4)
# 加速箭头
ax.annotate("", xy=(0 + w / 2, 5.2), xytext=(0 - w / 2, 18.9),
            arrowprops=dict(arrowstyle="-|>", color=C_HOT, lw=1.6,
                            connectionstyle="arc3,rad=0.35"))
ax.annotate("", xy=(1 + w / 2, 24), xytext=(1 - w / 2, 60.8),
            arrowprops=dict(arrowstyle="-|>", color=C_HOT, lw=1.6,
                            connectionstyle="arc3,rad=0.35"))
ax.text(0.06, 0.88, "5.3×", transform=ax.transAxes, fontsize=14,
        color=C_HOT, fontweight="bold")
ax.text(0.60, 0.72, "2.7×", transform=ax.transAxes, fontsize=14,
        color=C_HOT, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["synthetic 2×5 Mb\n6,047 genes", "A. thaliana genome\n24,625 genes"],
                   fontsize=9.5)
ax.set_ylabel("wall time (s, 24 threads)")
ax.set_ylim(0, 72)
ax.legend(frameon=False, loc="upper left")
ax.set_title("D  What it buys end-to-end (output identical)", loc="left",
             fontweight="bold")
_style(ax)
ax.grid(axis="y", color=C_GRID, lw=0.7, zorder=0)

fig.savefig("decoding_figures.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print(f"saved decoding_figures.png  (S={S}, E_total={E_TOTAL}, "
      f"A-matrix edges={counts_sym}, dense/cell={DENSE})")
