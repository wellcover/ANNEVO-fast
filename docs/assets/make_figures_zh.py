# -*- coding: utf-8 -*-
"""make_figures_zh.py — 生成中文版四联图 decoding_figures_zh.png（内容与英文版一致）。"""
import os
import sys
import types

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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ART = _build_hmm_artifacts(20, None, None)
S = ART["num_states"]
E_TOTAL = ART["n_edges"]
SYM = 0
counts_sym = int(ART["to_ptr"][SYM, -1] - ART["to_ptr"][SYM, 0])
to_idx = np.repeat(np.arange(S), np.diff(ART["to_ptr"][SYM]))
fr_idx = ART["edge_from"][ART["to_ptr"][SYM, 0]:ART["to_ptr"][SYM, -1]]
DENSE = S * S
SPARSE_AVG = E_TOTAL / 5

C_REF, C_NEW, C_HOT, C_GRID = "#9aa5b1", "#2e6f9e", "#c74440", "#d8dde3"

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


# ---- A 状态构成 ----
ax = axes[0, 0]
families = [
    ("基因间区", 1), ("起始密码子 (ATG)", 3), ("CDS（相位标记）", 6),
    ("供体/受体 motif", 12), ("终止密码子", 4), ("剪接辅助态", 24),
    ("内含子长度计数器", 120),
]
labels = [f[0] for f in families][::-1]
vals = [f[1] for f in families][::-1]
colors = [C_REF] * 6 + [C_HOT]
bars = ax.barh(labels, vals, color=colors, height=0.62, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_width() + 2.5, b.get_y() + b.get_height() / 2, str(v),
            va="center", fontsize=9.5, color="#333", zorder=4)
# 括号标注上方六个语法家族（行 1–6）
bx = 31
ax.plot([bx, bx], [0.6, 6.45], color="#888", lw=1.0, zorder=2)
ax.plot([bx - 1.4, bx], [6.45, 6.45], color="#888", lw=1.0, zorder=2)
ax.plot([bx - 1.4, bx], [0.6, 0.6], color="#888", lw=1.0, zorder=2)
ax.text(bx + 2.5, 3.5, "语法状态合计：50", fontsize=9.5, color="#555",
        style="italic", va="center", zorder=4)
# 计数器占比写进柱体内部
ax.text(55, 0, "占 S 的 70%", fontsize=9.5, color="white", fontweight="bold",
        va="center", zorder=4)
ax.set_xlim(0, 140)
ax.set_xlabel("状态数")
ax.set_title("A  170 个状态由什么构成", loc="left", fontweight="bold")
_style(ax)
ax.grid(axis="x", color=C_GRID, lw=0.7, zorder=0)

# ---- B 稀疏度 ----
ax = axes[0, 1]
ax.set_facecolor("#fbfcfd")
ax.set_xlim(0, S)
ax.set_ylim(0, S)
ax.scatter(fr_idx, to_idx, s=2.4, c=C_NEW, alpha=0.75, linewidths=0, zorder=3)
ax.set_xticks([0, 40, 80, 120, 169])
ax.set_yticks([0, 40, 80, 120, 169])
ax.set_xticklabels(["0", "40", "80", "120", "S−1"])
ax.set_yticklabels(["0", "40", "80", "120", "S−1"])
ax.set_xlabel("来源状态 (from)")
ax.set_ylabel("目标状态 (to)")
ax.set_title("B  转移矩阵几乎是空的", loc="left", fontweight="bold")
ax.text(0.03, 0.05,
        f"A 条件矩阵：\n28,900 个格子中仅 {counts_sym} 条有限边（0.6%）",
        transform=ax.transAxes, fontsize=9.5, va="bottom", color="#333",
        bbox=dict(facecolor="white", edgecolor=C_GRID, alpha=0.9, pad=4))
ax.text(0.97, 0.955,
        "5 张条件矩阵（A/T/C/G/N）\n共有 879 条边",
        transform=ax.transAxes, fontsize=9, va="top", ha="right", color="#555",
        style="italic")
_style(ax)

# ---- C 每碱基工作量 ----
ax = axes[1, 0]
names = ["稠密 O(L·S²)\nS² = 170² 次比较", "稀疏 O(L·E)\n≈ E/5 次边访问"]
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
ax.text(0.52, 0.50, "≈ 164× 减少", transform=ax.transAxes, ha="center",
        fontsize=13, color=C_HOT, fontweight="bold")
ax.text(0.52, 0.40, "每碱基候选访问次数", transform=ax.transAxes,
        ha="center", fontsize=9, color="#666")
ax.set_ylabel("每碱基工作量（对数轴）")
ax.set_title("C  重写后每碱基省下多少", loc="left", fontweight="bold")
_style(ax)
ax.grid(axis="y", color=C_GRID, lw=0.7, zorder=0)

# ---- D 端到端基准 ----
ax = axes[1, 1]
x = np.arange(2)
ref = [17.9, 59.4]
fast = [3.4, 22.3]
w = 0.34
b1 = ax.bar(x - w / 2, ref, w, label="官方对照", color=C_REF, zorder=3)
b2 = ax.bar(x + w / 2, fast, w, label="本仓库", color=C_NEW, zorder=3)
for b, v in zip(b1, ref):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v} 秒", ha="center",
            fontsize=9.5, color="#333", zorder=4)
for b, v in zip(b2, fast):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v} 秒", ha="center",
            fontsize=9.5, color="#333", zorder=4)
# 括号式加速标注
for xi, r, f, sp in ((0, 17.9, 3.4, "5.3×"), (1, 59.4, 22.3, "2.7×")):
    xr = xi + w / 2 + 0.07
    ax.hlines(r, xi - w / 2, xr, colors="#888", linestyles=(0, (3, 2)),
              lw=0.9, zorder=2)
    ax.hlines(f, xi + w / 2, xr, colors="#888", linestyles=(0, (3, 2)),
              lw=0.9, zorder=2)
    ax.annotate("", xy=(xr, f + 1.6), xytext=(xr, r - 1.6),
                arrowprops=dict(arrowstyle="<|-|>", color=C_HOT, lw=1.5,
                                mutation_scale=11), zorder=4)
    ax.text(xr + 0.045, (r + f) / 2, sp, fontsize=13, color=C_HOT,
            fontweight="bold", va="center", zorder=4)
ax.set_xticks(x)
ax.set_xticklabels(["合成数据 2×5 Mb\n6,047 个基因", "拟南芥全基因组\n24,625 个基因"],
                   fontsize=9.5)
ax.set_ylabel("耗时（秒，24 线程）")
ax.set_xlim(-0.55, 1.82)
ax.set_ylim(0, 72)
ax.legend(frameon=False, loc="upper left")
ax.set_title("D  端到端的实际收益（输出完全一致）", loc="left",
             fontweight="bold")
_style(ax)
ax.grid(axis="y", color=C_GRID, lw=0.7, zorder=0)

fig.savefig("decoding_figures_zh.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print(f"saved decoding_figures_zh.png  (S={S}, E_total={E_TOTAL}, "
      f"A-matrix edges={counts_sym})")
