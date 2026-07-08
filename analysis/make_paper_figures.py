"""
Generate publication figures for the ECPPM 2026 paper.

Produces three vector (PDF) figures under analysis/figures/:
  1. fig_category_heatmap.pdf      -> \\label{fig:category-placeholder}
  2. fig_efficiency_tradeoff.pdf   -> \\label{fig:efficiency-placeholder}
  3. fig_criteria_faithfulness.pdf -> \\label{fig:criteria-placeholder}

All numbers are the frozen 507-question test-split results taken from
analysis/data_representation_comparison_report.md (sections 4, 5, 6), so the
script is self-contained and does not need the exported result CSVs.

Run (ephemeral env, no repo venv touched):
    env -u VIRTUAL_ENV uv run --with matplotlib --with numpy \
        python analysis/make_paper_figures.py
"""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style: Times-like serif (STIX) to match the paper's mathptmx font.
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType, avoid Type-3 (publisher-safe)
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.edgecolor": "#333333",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# One consistent colour per representation across every figure (Okabe-Ito,
# colour-blind safe). Order = descending overall accuracy.
REPS = ["Native IFC", "File system", "Graph database", "SQL database"]
COLOR = {
    "Native IFC":     "#0072B2",  # blue   (best overall)
    "File system":    "#E69F00",  # orange
    "Graph database": "#009E73",  # green
    "SQL database":   "#D55E00",  # vermillion
}


def _txt_color(rgb):
    """Black or white label depending on background luminance."""
    r, g, b = rgb[:3]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if lum > 0.55 else "white"


# ===========================================================================
# Figure 1 -- category accuracy heatmap (report section 4)
# rows = representation, cols = question category
# ===========================================================================
def fig_category_heatmap():
    cats = ["C1\nDirect\nretrieval",
            "C2\nComputational\naggregation",
            "C3\nGeometric /\nspatial",
            "C4\nIncomplete\ninformation"]
    counts = [65, 288, 62, 92]
    # acc[rep][cat]  (%)
    acc = {
        "Native IFC":     [86.2, 74.7, 87.1, 80.4],
        "File system":    [83.1, 74.0, 75.8, 79.3],
        "Graph database": [83.1, 72.6, 80.6, 76.1],
        "SQL database":   [83.1, 70.5, 88.7, 71.7],
    }
    M = np.array([acc[r] for r in REPS])

    fig = plt.figure(figsize=(6.8, 2.7))
    ax = fig.add_axes([0.165, 0.22, 0.70, 0.72])
    cmap = mpl.colormaps["YlGnBu"]
    vmin, vmax = 68, 90
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # cell annotations with auto-contrast text
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            rgb = cmap((M[i, j] - vmin) / (vmax - vmin))
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    fontsize=9.5, color=_txt_color(rgb))

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, counts)],
                       fontsize=8)
    ax.set_yticks(range(len(REPS)))
    ax.set_yticklabels(REPS)
    ax.tick_params(length=0)
    # thin white gridlines between cells
    ax.set_xticks(np.arange(-.5, len(cats), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(REPS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    for s in ax.spines.values():
        s.set_visible(False)

    cax = fig.add_axes([0.880, 0.22, 0.022, 0.72])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Accuracy (%)", fontsize=8.5)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=8, length=2)

    fig.savefig(OUT / "fig_category_heatmap.pdf", bbox_inches=None)
    plt.close(fig)
    print("wrote", OUT / "fig_category_heatmap.pdf")


# ===========================================================================
# Figure 2 -- efficiency trade-off scatter (report section 6)
# x = mean tool calls, y = mean latency; marker size encodes accuracy
# ===========================================================================
def fig_efficiency_tradeoff():
    # rep: (tool_calls, latency_s, accuracy_%)
    data = {
        "Native IFC":     (8.8,  73.2, 78.7),
        "SQL database":   (12.3, 105.6, 74.6),
        "File system":    (20.4, 81.4, 76.3),
        "Graph database": (23.3, 129.7, 75.5),
    }
    # label placement offsets (points) tuned to avoid overlap
    offs = {
        "Native IFC":     (10, 8,  "left"),
        "SQL database":   (10, -4, "left"),
        "File system":    (0, 12,  "center"),
        "Graph database": (-10, 6, "right"),
    }

    # marker area encodes overall accuracy (linear over the observed range,
    # suppressed zero so the small 74.6-78.7 spread is visible)
    lo, hi = 74.6, 78.7

    def acc_to_size(acc):
        return 200 + (acc - lo) / (hi - lo) * 460

    fig, ax = plt.subplots(figsize=(5.6, 3.5))

    for rep, (tc, lat, acc) in data.items():
        ax.scatter(tc, lat, s=acc_to_size(acc), color=COLOR[rep],
                   edgecolor="black", linewidth=0.7, zorder=3, alpha=0.92)
        dx, dy, ha = offs[rep]
        ax.annotate(f"{rep}\n{acc:.1f}%", (tc, lat),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", fontsize=8.3,
                    linespacing=1.15)

    ax.set_xlabel("Mean tool calls per question")
    ax.set_ylabel("Mean latency (s) per question")
    ax.set_xlim(4, 27)
    ax.set_ylim(60, 140)
    ax.grid(True, linewidth=0.4, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

    fig.savefig(OUT / "fig_efficiency_tradeoff.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_efficiency_tradeoff.pdf")


# ===========================================================================
# Figure 3 -- faithfulness as the universal bottleneck (report section 5)
# (a) criterion pass rates per representation
# (b) share of wrong answers that fail faithfulness
# ===========================================================================
def fig_criteria_faithfulness():
    criteria = ["Faithful-\nness", "Complete-\nness", "Transpar-\nency", "Relevance"]
    passrate = {
        "Native IFC":     [84.2, 93.2, 96.0, 96.6],
        "File system":    [82.1, 90.8, 93.6, 96.6],
        "Graph database": [82.1, 90.3, 93.8, 96.4],
        "SQL database":   [79.1, 90.8, 94.2, 95.8],
    }
    # faithfulness failures among wrong answers: (count, total, pct)
    faith_fail = {
        "Native IFC":     (80, 107, 74.8),
        "File system":    (90, 115, 78.3),
        "Graph database": (89, 116, 76.7),
        "SQL database":   (105, 125, 84.0),
    }

    # single-column figure: two panels stacked vertically
    fig, (axA, axB) = plt.subplots(
        2, 1, figsize=(3.4, 4.5),
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.55})
    fig.subplots_adjust(left=0.17, right=0.965, top=0.93, bottom=0.16)

    # ---- (a) grouped criterion pass rates ----
    x = np.arange(len(criteria))
    w = 0.20
    bar_handles = []
    for k, rep in enumerate(REPS):
        h = axA.bar(x + (k - 1.5) * w, passrate[rep], width=w,
                    color=COLOR[rep], edgecolor="black", linewidth=0.4,
                    label=rep, zorder=3)
        bar_handles.append(h)
    axA.set_xticks(x)
    axA.set_xticklabels(criteria, fontsize=7.3)
    axA.set_ylim(70, 100)
    axA.set_ylabel("Pass rate (%)")
    axA.set_title("(a) Judge-criterion pass rates", fontsize=8.5)
    axA.grid(axis="y", linewidth=0.4, color="#dddddd", zorder=0)
    axA.set_axisbelow(True)
    axA.tick_params(labelsize=8)

    # ---- (b) faithfulness failures among wrong answers (horizontal bars) ----
    yb = np.arange(len(REPS))[::-1]   # first representation at the top
    vals = [faith_fail[r][2] for r in REPS]
    bars = axB.barh(yb, vals, height=0.62,
                    color=[COLOR[r] for r in REPS],
                    edgecolor="black", linewidth=0.4, zorder=3)
    for b, r in zip(bars, REPS):
        pct = faith_fail[r][2]
        axB.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2,
                 f"{pct:.0f}%", ha="left", va="center", fontsize=7.8)
    axB.set_yticks(yb)
    axB.set_yticklabels(REPS, fontsize=8)
    axB.set_xlim(60, 92)
    axB.set_xlabel("Errors failing faithfulness (%)")
    axB.set_title("(b) Faithfulness failure share", fontsize=8.5)
    axB.grid(axis="x", linewidth=0.4, color="#dddddd", zorder=0)
    axB.set_axisbelow(True)
    axB.tick_params(labelsize=8)

    # shared representation legend along the bottom
    fig.legend(handles=[h[0] for h in bar_handles], labels=REPS,
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.2,
               handlelength=1.1, handletextpad=0.4, fontsize=7.5)

    fig.savefig(OUT / "fig_criteria_faithfulness.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_criteria_faithfulness.pdf")


if __name__ == "__main__":
    fig_category_heatmap()
    fig_efficiency_tradeoff()
    fig_criteria_faithfulness()
    print("done ->", OUT)
