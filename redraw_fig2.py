from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "legend.frameon": False, "figure.facecolor": "white",
})

profiles = pd.read_csv(OUT / "profile_stats.csv")
strata = pd.read_csv(OUT / "risk_strata.csv")
capacity = pd.read_csv(OUT / "capacity_policy_comparison.csv")

fig, axes = plt.subplots(1, 3)
profiles["label"] = [f"P{i+1}" for i in profiles.profile]
axes[0].bar(profiles.label, profiles.risk_rate, color=["#7FB0C2", "#C85A54"][:len(profiles)])
axes[0].set(xlabel="Behavioral profile", ylabel="Observed academic-risk rate", ylim=(0, 1))
axes[1].bar(strata.risk_stratum.astype(str), strata.observed_risk, color=["#B7D5DF", "#E4B363", "#C85A54"])
axes[1].set(xlabel="Predicted probability stratum", ylabel="Observed academic-risk rate", ylim=(0, 1))
for policy, marker, color in [("Global probability ranking", "o", "#2B6F8A"),
                              ("Profile-balanced quota", "s", "#C85A54")]:
    z = capacity[capacity.policy == policy]
    axes[2].plot(z.capacity_fraction * 100, z.precision_at_k, marker=marker, color=color, label=policy)
axes[2].set(xlabel="Intervention capacity (% of cohort)", ylabel="Precision@K", ylim=(0, 1))
axes[2].legend(fontsize=6)
for i, ax in enumerate(axes):
    ax.grid(axis="y", color="#E5E7EB", lw=.5)
    ax.text(-.16, 1.04, chr(97+i), transform=ax.transAxes, fontweight="bold", fontsize=9)
fig.set_size_inches(7.1, 2.35)
fig.tight_layout(w_pad=1.2)
tiff = FIG / "Fig2_profiles_and_stratification.tiff"
fig.savefig(tiff, dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(FIG / "Fig2_profiles_and_stratification.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
with Image.open(tiff) as im:
    assert im.format == "TIFF" and im.info.get("dpi", (0, 0))[0] >= 599
