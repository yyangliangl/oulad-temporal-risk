from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def box(ax, xy, wh, title, lines, color, title_color="white"):
    x, y = xy; w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                           linewidth=0.8, edgecolor=color, facecolor="#F8FAFC")
    ax.add_patch(patch)
    ax.add_patch(FancyBboxPatch((x, y+h-0.08), w, 0.08, boxstyle="round,pad=0.012,rounding_size=0.015",
                                linewidth=0, facecolor=color))
    ax.text(x+w/2, y+h-0.04, title, ha="center", va="center", color=title_color, fontweight="bold")
    ax.text(x+0.025, y+h-0.105, "\n".join(lines), ha="left", va="top", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.0, color="#52606D"))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.025, label, ha="center", va="bottom", color="#52606D")


fig, ax = plt.subplots(figsize=(7.1, 2.45))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

box(ax, (0.02, 0.15), (0.19, 0.70), "Virtual sensor layer",
    ["Content", "Navigation", "Assessment", "Communication", "Collaboration", "Enrichment"], "#397C96")
box(ax, (0.27, 0.15), (0.20, 0.70), "Temporal feature engine",
    ["Weekly channel streams", "Volume and active weeks", "Slope and variability", "Recent activity share", "Channel entropy"], "#397C96")
box(ax, (0.53, 0.52), (0.20, 0.33), "Leakage-safe prediction",
    ["TRAIN: fit scaler + model", "TEST: transform + score", "One held-out presentation"], "#596D83")
box(ax, (0.53, 0.15), (0.20, 0.27), "Longitudinal profiling",
    ["TRAIN: fit/order GMM", "TEST: assign profile", "Days 28 / 56 / 84"], "#596D83")
box(ax, (0.79, 0.15), (0.19, 0.70), "Decision layer",
    ["Probability strata", "Descriptive profiles", "Top-K allocation policy", "Proposed dashboard", "Periodic model review"], "#B85450")

arrow(ax, 0.21, 0.50, 0.27, 0.50)
arrow(ax, 0.47, 0.58, 0.53, 0.68)
arrow(ax, 0.47, 0.42, 0.53, 0.29)
arrow(ax, 0.73, 0.68, 0.79, 0.60)
arrow(ax, 0.73, 0.29, 0.79, 0.40)
ax.text(0.50, 0.96, "IoT-inspired virtual sensing: no physical devices required", ha="center",
        va="center", fontweight="bold", color="#263746")
ax.text(0.50, 0.045, "Observation windows: days 0-28 / 0-56 / 0-84   |   Unit: student-module-presentation",
        ha="center", va="center", color="#52606D")
fig.tight_layout(pad=0.2)

tiff = OUT / "Fig0_framework.tiff"
fig.savefig(tiff, dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(OUT / "Fig0_framework.svg", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "Fig0_framework.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "Fig0_framework.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
with Image.open(tiff) as im:
    assert im.format == "TIFF" and im.info.get("dpi", (0, 0))[0] >= 599
print(tiff)
