# make_fig4.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from pathlib import Path

ROOT = Path(r"D:\Dynamic Behavioral Profiling")
OUT = ROOT / "outputs"

# ---------- 读取 oof_predictions.csv ----------
df = pd.read_csv(OUT / "oof_predictions.csv")

# 概率与真实标签
prob = df["risk_probability"].values
y = df["target"].values

# ---------- 计算可靠性曲线 ----------
frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10)

# ---------- 绘图 ----------
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": 600,
})
fig, ax = plt.subplots(figsize=(3.4, 3.2))
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
ax.plot(mean_pred, frac_pos, "o-", color="#2B6F8A",
        markersize=4, linewidth=1.2, label="LR-temporal")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.grid(axis="both", color="#E5E7EB", linewidth=0.5)
ax.legend(loc="upper left", fontsize=7, frameon=False)
ax.set_title("Day-56 reliability curve", fontsize=8)
fig.tight_layout()

fig.savefig(OUT / "Fig4_reliability_day56.tiff", dpi=600,
            bbox_inches="tight", facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(OUT / "Fig4_reliability_day56.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.close(fig)

print("Saved:", OUT / "Fig4_reliability_day56.tiff")