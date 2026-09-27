# -*- coding: utf-8 -*-
"""make_figures.py — 生成 docs/decoding-internals.md 的三联对比图。

Panel A: HMM 状态构成（min_intron_length=20, S=170，实测计数）
Panel B: Viterbi 内循环每碱基工作量（稠密 S² vs 稀疏 E/5，对数轴）
Panel C: 端到端解码耗时基准（参考实现 vs 本仓库，24 线程）

用法：python make_figures.py  （在 docs/assets/ 下运行，输出 decoding_figures.png）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
for ax in axes:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

# ---------------- Panel A: state inventory ----------------
ax = axes[0]
families = [
    ("Intergenic", 1), ("Start codon", 3), ("CDS (phase)", 6),
    ("DSS/ASS motifs", 12), ("Stop codon", 4), ("Splice helpers", 24),
    ("Intron counters", 120),
]
labels = [f[0] for f in families][::-1]
vals = [f[1] for f in families][::-1]
colors = ["#b0b8c4"] * 6 + ["#d9534f"]
bars = ax.barh(labels, vals, color=colors, height=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2, str(v),
            va="center", fontsize=9, color="#333")
ax.set_xlabel("Number of states")
ax.set_title(f"A  State inventory (S = 170)\ncounters dominate the machine",
             fontsize=10.5, loc="left")
ax.grid(axis="x", alpha=0.25, linewidth=0.6)
ax.grid(axis="y", alpha=0)
ax.set_xlim(0, 138)

# ---------------- Panel B: inner-loop work per base ----------------
ax = axes[1]
names = ["Dense O(L·S²)\nS² comparisons", "Sparse O(L·E)\nE/5 edge visits"]
vals = [170 ** 2, 879 / 5]
bars = ax.bar(names, vals, color=["#b0b8c4", "#2e6f9e"], width=0.5)
ax.set_yscale("log")
ax.set_ylim(80, 2e5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.35, f"{v:,.0f}",
            ha="center", fontsize=10, color="#333")
ax.annotate("~164× fewer\nper-base visits", xy=(1, 200), xytext=(0.42, 3000),
            fontsize=10, color="#d9534f", ha="center",
            arrowprops=dict(arrowstyle="->", color="#d9534f", lw=1.2))
ax.set_ylabel("work per base (log scale)")
ax.set_title("B  Viterbi inner loop\nmin_intron = 20, 879 edges total",
             fontsize=10.5, loc="left")

# ---------------- Panel C: wall-clock benchmark ----------------
ax = axes[2]
import numpy as np
x = np.arange(2)
ref = [17.9, 59.4]
fast = [3.4, 22.3]
w = 0.36
b1 = ax.bar(x - w / 2, ref, w, label="Reference decoder", color="#b0b8c4")
b2 = ax.bar(x + w / 2, fast, w, label="This repo", color="#2e6f9e")
for b, v in zip(b1, ref):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v}s", ha="center",
            fontsize=9.5, color="#333")
for b, v in zip(b2, fast):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v}s", ha="center",
            fontsize=9.5, color="#333")
ax.annotate("5.3×", xy=(0, 12), ha="center", fontsize=12, color="#d9534f",
            fontweight="bold")
ax.annotate("2.7×", xy=(1, 36), ha="center", fontsize=12, color="#d9534f",
            fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Synthetic 2×5 Mb\n6,047 genes", "A. thaliana genome\n24,625 genes"],
                   fontsize=9.5)
ax.set_ylabel("wall time (s, 24 threads)")
ax.set_ylim(0, 70)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("C  End-to-end decoding\ngene content identical", fontsize=10.5,
             loc="left")

fig.tight_layout(pad=1.2)
fig.savefig("decoding_figures.png", dpi=200, bbox_inches="tight")
print("saved decoding_figures.png")
