from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY, TEAL, BLUE, RED, AMBER = "#23384D", "#24738A", "#52718E", "#B75450", "#D49B38"
INK, MUTED, LINE, PAPER = "#263645", "#647586", "#AAB7C2", "#F6F8FA"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 6.3, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def rounded(ax, x, y, w, h, face="white", edge=LINE, lw=0.8, radius=0.018, z=2, linestyle="solid", hatch=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.008,rounding_size={radius}",
                       facecolor=face, edgecolor=edge, linewidth=lw, linestyle=linestyle, hatch=hatch, zorder=z)
    ax.add_patch(p)
    return p


def stage(ax, x, number, title, color):
    ax.add_patch(Circle((x, 0.915), 0.020, facecolor=color, edgecolor="none", zorder=5))
    ax.text(x, 0.915, str(number), ha="center", va="center", color="white", weight="bold", fontsize=6.5, zorder=6)
    ax.text(x + 0.029, 0.915, title, ha="left", va="center", color=INK, weight="bold", fontsize=7.1)


def arrow(ax, start, end, color=MUTED, lw=1.15, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8.5,
                                 linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=4))


def pill(ax, x, y, text, color, width):
    rounded(ax, x, y, width, 0.047, face=color, edge=color, radius=0.020, z=4)
    ax.text(x + width / 2, y + 0.0235, text, ha="center", va="center", color="white", weight="bold", fontsize=5.7, zorder=5)


fig, ax = plt.subplots(figsize=(7.1, 2.82))
ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="none", zorder=0))
ax.plot([0.048, 0.95], [0.915, 0.915], color="#D5DDE3", linewidth=1.0, zorder=1)
stage(ax, 0.055, 1, "Virtual sensing", TEAL); stage(ax, 0.245, 2, "Temporal summaries", TEAL)
stage(ax, 0.455, 3, "LOPO isolation", BLUE); stage(ax, 0.687, 4, "Fold outputs", BLUE)
stage(ax, 0.855, 5, "Screening", RED)

rounded(ax, 0.025, 0.18, 0.168, 0.64, face=PAPER, edge=TEAL, lw=1.0)
ax.text(0.109, 0.775, "Software-derived channels", ha="center", va="center", color=TEAL, weight="bold", fontsize=6.8)
channels = [("Content", "resource exposure"), ("Navigation", "course structure"), ("Assessment", "task engagement"),
            ("Communication", "help seeking"), ("Collaboration", "joint activity"), ("Enrichment", "supplementary tools")]
for i, (name, desc) in enumerate(channels):
    yy = 0.704 - i * 0.082
    ax.add_patch(Circle((0.047, yy), 0.008, facecolor=TEAL, edgecolor="none"))
    ax.text(0.061, yy + 0.010, name, ha="left", va="center", color=INK, weight="bold", fontsize=6.0)
    ax.text(0.061, yy - 0.014, desc, ha="left", va="center", color=MUTED, fontsize=5.3)
ax.text(0.109, 0.205, "No physical sensor deployment", ha="center", va="center", color=TEAL, style="italic", fontsize=5.5)

rounded(ax, 0.218, 0.18, 0.188, 0.64, face=PAPER, edge=TEAL, lw=1.0)
ax.text(0.312, 0.775, "Weekly channel streams", ha="center", va="center", color=TEAL, weight="bold", fontsize=6.8)
for row, vals in enumerate([[3, 5, 5, 7, 6, 4, 3, 2], [1, 2, 4, 3, 5, 5, 6, 7], [7, 5, 2, 6, 4, 3, 5, 2]]):
    y0 = 0.68 - row * 0.083
    for j, val in enumerate(vals):
        ax.add_patch(Rectangle((0.243 + j * 0.017, y0), 0.011, val * 0.0047,
                               facecolor=TEAL, alpha=0.26 + 0.07 * row, edgecolor="none"))
ax.text(0.312, 0.445, "Auditable descriptors", ha="center", va="center", color=INK, weight="bold")
for i, label in enumerate(["Volume + active weeks", "Slope + variability", "Recency + diversity"]):
    pill(ax, 0.246, 0.382 - i * 0.066, label, TEAL if i < 2 else BLUE, 0.132)
ax.text(0.312, 0.205, "Landmarks: days 28 / 56 / 84", ha="center", va="center", color=MUTED, fontsize=5.5)

rounded(ax, 0.431, 0.13, 0.228, 0.72, face="white", edge=BLUE, lw=1.2)
ax.text(0.545, 0.805, "One held-out presentation per fold", ha="center", va="center", color=NAVY, weight="bold", fontsize=6.9)
ax.plot([0.451, 0.639], [0.735, 0.735], color=LINE, linewidth=0.7, linestyle=(0, (3, 2)))
rounded(ax, 0.454, 0.472, 0.182, 0.225, face="#EEF3F7", edge=BLUE, lw=0.9, linestyle="solid")
pill(ax, 0.468, 0.646, "TRAINING PRESENTATIONS", BLUE, 0.154)
for i, line in enumerate(["fit scaler + classifier", "fit and order mixture model", "estimate training-only rules"]):
    ax.text(0.477, 0.600 - i * 0.043, f"•  {line}", ha="left", va="center", color=INK, fontsize=5.7)
rounded(ax, 0.454, 0.205, 0.182, 0.208, face="#FFF7EC", edge=AMBER, lw=1.0, linestyle=(0, (4, 2)))
pill(ax, 0.468, 0.363, "HELD-OUT PRESENTATION", AMBER, 0.154)
for i, line in enumerate(["exclude prior withdrawals", "transform and score once", "assign descriptive profile"]):
    ax.text(0.477, 0.319 - i * 0.043, f"•  {line}", ha="left", va="center", color=INK, fontsize=5.7)
ax.text(0.545, 0.155, "No cross-presentation fitting", ha="center", va="center", color=RED, weight="bold", fontsize=5.7)

rounded(ax, 0.683, 0.55, 0.140, 0.22, face="#EEF3F7", edge=BLUE, lw=0.9)
ax.text(0.753, 0.728, "Predictive output", ha="center", va="center", color=BLUE, weight="bold", fontsize=6.5)
ax.text(0.753, 0.665, "Out-of-fold risk score", ha="center", va="center", color=INK, fontsize=5.8)
ax.text(0.753, 0.613, "AUROC • AUPRC • Brier", ha="center", va="center", color=MUTED, fontsize=5.4)
rounded(ax, 0.683, 0.25, 0.140, 0.22, face="#F3F1F7", edge="#766A91", lw=0.9)
ax.text(0.753, 0.428, "Explanatory output", ha="center", va="center", color="#665A82", weight="bold", fontsize=6.5)
ax.text(0.753, 0.365, "Longitudinal profile", ha="center", va="center", color=INK, fontsize=5.8)
ax.text(0.753, 0.313, "Description, not identity", ha="center", va="center", color=MUTED, fontsize=5.4)

rounded(ax, 0.849, 0.18, 0.126, 0.64, face="#FFF7F5", edge=RED, lw=1.0, linestyle=(0, (2, 1)))
ax.text(0.912, 0.775, "Decision layer", ha="center", va="center", color=RED, weight="bold", fontsize=6.8)
items = [("1", "Probability strata", "Low / moderate / high"), ("2", "Capacity rule", "Select top K scores"),
         ("3", "Instructor review", "Trends + profile context"), ("4", "Monitoring", "Precision@K / Recall@K")]
for i, (num, title, desc) in enumerate(items):
    yy = 0.682 - i * 0.119
    ax.add_patch(Circle((0.872, yy), 0.013, facecolor=RED, edgecolor="none", zorder=4))
    ax.text(0.872, yy, num, ha="center", va="center", color="white", weight="bold", fontsize=5.3)
    ax.text(0.893, yy + 0.012, title, ha="left", va="center", color=INK, weight="bold", fontsize=5.7)
    ax.text(0.893, yy - 0.017, desc, ha="left", va="center", color=MUTED, fontsize=5.0)
ax.text(0.912, 0.213, "Retrospective screening\n≠ intervention benefit", ha="center", va="center", color=RED, style="italic", fontsize=5.4, linespacing=1.2)

arrow(ax, (0.193, 0.50), (0.218, 0.50), color=TEAL); arrow(ax, (0.406, 0.50), (0.431, 0.50), color=BLUE)
arrow(ax, (0.659, 0.59), (0.683, 0.65), color=BLUE, rad=-0.05); arrow(ax, (0.659, 0.39), (0.683, 0.36), color="#766A91", rad=0.04)
arrow(ax, (0.823, 0.65), (0.849, 0.60), color=RED, rad=0.04); arrow(ax, (0.823, 0.36), (0.849, 0.43), color=RED, rad=-0.04)
ax.text(0.5, 0.055, "Unit: student-module-presentation   •   22-fold leave-one-presentation-out (LOPO)   •   training-fold fitting only",
        ha="center", va="center", color=MUTED, fontsize=5.8)

fig.tight_layout(pad=0.08)
base = OUT / "Fig0_framework"
fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
with Image.open(base.with_suffix(".tiff")) as im:
    assert im.format == "TIFF" and im.info.get("dpi", (0, 0))[0] >= 599
print(base.with_suffix(".tiff"))
