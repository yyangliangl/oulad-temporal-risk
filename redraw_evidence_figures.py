from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
RNG_SEED = 20260820

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
})

BLUE = "#2B6F8A"
RED = "#C85A54"
GREY = "#9AA3AD"
LIGHT_BLUE = "#A7CAD6"
GOLD = "#D9A441"


def mean_ci(values, rng, n_boot=3000):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return values.mean(), np.quantile(draws, 0.025), np.quantile(draws, 0.975)


def save_figure(fig, stem, width=7.1, height=2.45):
    fig.set_size_inches(width, height)
    fig.tight_layout(w_pad=1.25)
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(FIG / f"{stem}.tiff") as im:
        assert im.format == "TIFF" and im.info.get("dpi", (0, 0))[0] >= 599


# Figure 2: presentation-level uncertainty and paired fold differences.
folds = pd.read_csv(OUT / "fold_metrics.csv")
folds = folds[folds.model == "Logistic regression"].copy()
rng = np.random.default_rng(RNG_SEED)
fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1.05, 1.05, 1.15]})
for metric, ax in zip(["AUROC", "AUPRC"], axes[:2]):
    for label, color, marker in [("Cumulative", GREY, "s"), ("Temporal virtual sensing", BLUE, "o")]:
        centers, lower, upper = [], [], []
        for horizon in [28, 56, 84]:
            vals = folds[(folds.horizon_days == horizon) & (folds.feature_set == label)][metric]
            center, lo, hi = mean_ci(vals, rng)
            centers.append(center); lower.append(center - lo); upper.append(hi - center)
        ax.errorbar([28, 56, 84], centers, yerr=[lower, upper], color=color, marker=marker,
                    linewidth=1.5, capsize=2.2, markersize=4.5, label=label)
    ax.set(xlabel="Observation horizon (days)", ylabel=f"Mean fold {metric}", xticks=[28, 56, 84])
    ax.grid(axis="y", color="#E5E7EB", linewidth=.5)
axes[0].legend(loc="lower right", fontsize=6.2)

ax = axes[2]
for xpos, metric, color in [(0, "AUROC", BLUE), (1, "AUPRC", RED)]:
    a = folds[(folds.horizon_days == 56) & (folds.feature_set == "Cumulative")].sort_values("held_out")
    b = folds[(folds.horizon_days == 56) & (folds.feature_set == "Temporal virtual sensing")].sort_values("held_out")
    delta = b[metric].to_numpy() - a[metric].to_numpy()
    jitter = np.linspace(-.08, .08, len(delta))
    ax.scatter(np.full(len(delta), xpos) + jitter, delta, s=12, color=color, alpha=.58, edgecolor="none")
    center, lo, hi = mean_ci(delta, rng)
    ax.errorbar(xpos, center, yerr=[[center-lo], [hi-center]], color="black", marker="D",
                markersize=4, capsize=3, linewidth=1.1)
ax.axhline(0, color="#667085", linewidth=.8, linestyle="--")
ax.set(xticks=[0, 1], xticklabels=["AUROC", "AUPRC"], ylabel="Temporal − cumulative\n(fold difference)",
       xlabel="56-day paired comparison")
ax.grid(axis="y", color="#E5E7EB", linewidth=.5)
for i, ax in enumerate(axes):
    ax.text(-.16, 1.04, chr(97+i), transform=ax.transAxes, fontweight="bold", fontsize=8)
save_figure(fig, "Fig1_predictive_performance")


# Figure 3: cluster-bootstrap uncertainty for descriptive and operational strata.
pred = pd.read_csv(OUT / "oof_predictions.csv")
pred["presentation"] = pred.code_module.astype(str) + "-" + pred.code_presentation.astype(str)
presentations = pred.presentation.unique()


def cluster_bootstrap_profiles(frame, n_boot=1000):
    rng = np.random.default_rng(RNG_SEED + 1)
    profile_vals = {p: [] for p in sorted(frame.profile.unique())}
    stratum_vals = {s: [] for s in ["Low", "Moderate", "High"]}
    policy_vals = {(f, policy): [] for f in [.05, .10, .15, .20, .30]
                   for policy in ["Global probability ranking", "Profile-balanced quota"]}
    grouped = {g: frame[frame.presentation == g] for g in presentations}
    for _ in range(n_boot):
        sampled = rng.choice(presentations, size=len(presentations), replace=True)
        z = pd.concat([grouped[g] for g in sampled], ignore_index=True)
        probs = z.risk_probability.to_numpy()
        targets = z.target.to_numpy()
        global_order = np.argsort(-probs)
        profile_orders = {}
        for p in profile_vals:
            cand = np.flatnonzero(z.profile.to_numpy() == p)
            profile_orders[p] = cand[np.argsort(-probs[cand])]
        for p in profile_vals:
            q = z[z.profile == p]
            if len(q): profile_vals[p].append(q.target.mean())
        for s in stratum_vals:
            q = z[z.risk_stratum == s]
            if len(q): stratum_vals[s].append(q.target.mean())
        for frac in [.05, .10, .15, .20, .30]:
            k = int(np.ceil(len(z) * frac))
            global_ix = global_order[:k]
            policy_vals[(frac, "Global probability ranking")].append(targets[global_ix].mean())
            selected = []
            quota = k // len(profile_vals)
            for p in profile_vals:
                take = profile_orders[p][:min(quota, len(profile_orders[p]))]
                selected.extend(take.tolist())
            selected = list(dict.fromkeys(selected))
            if len(selected) < k:
                selected_set = set(selected)
                remaining = [i for i in global_order if i not in selected_set]
                selected.extend(remaining[:k-len(selected)])
            policy_vals[(frac, "Profile-balanced quota")].append(targets[selected[:k]].mean())
    return profile_vals, stratum_vals, policy_vals


profile_boot, stratum_boot, policy_boot = cluster_bootstrap_profiles(pred)
profile_stats = pred.groupby("profile").target.mean()
stratum_stats = pred.groupby("risk_stratum").target.mean()
fig, axes = plt.subplots(1, 3)

profiles = sorted(profile_stats.index)
profile_centers = [profile_stats[p] for p in profiles]
profile_err = []
for p, center in zip(profiles, profile_centers):
    lo, hi = np.quantile(profile_boot[p], [.025, .975])
    profile_err.append((center-lo, hi-center))
axes[0].bar(range(len(profiles)), profile_centers, color=[LIGHT_BLUE, RED][:len(profiles)])
axes[0].errorbar(range(len(profiles)), profile_centers, yerr=np.array(profile_err).T,
                 fmt="none", ecolor="black", capsize=2.5, linewidth=.9)
axes[0].set(xticks=range(len(profiles)), xticklabels=[f"P{p+1}" for p in profiles],
            xlabel="Descriptive behavioral profile", ylabel="Observed academic-risk rate", ylim=(0, 1))

strata = ["Low", "Moderate", "High"]
stratum_centers = [stratum_stats[s] for s in strata]
stratum_err = []
for s, center in zip(strata, stratum_centers):
    lo, hi = np.quantile(stratum_boot[s], [.025, .975])
    stratum_err.append((center-lo, hi-center))
axes[1].bar(range(3), stratum_centers, color=[LIGHT_BLUE, GOLD, RED])
axes[1].errorbar(range(3), stratum_centers, yerr=np.array(stratum_err).T,
                 fmt="none", ecolor="black", capsize=2.5, linewidth=.9)
axes[1].set(xticks=range(3), xticklabels=strata, xlabel="Operational probability stratum",
            ylabel="Observed academic-risk rate", ylim=(0, 1))

capacity = pd.read_csv(OUT / "capacity_policy_comparison.csv")
for policy, marker, color in [("Global probability ranking", "o", BLUE),
                              ("Profile-balanced quota", "s", RED)]:
    z = capacity[capacity.policy == policy].sort_values("capacity_fraction")
    centers = z.precision_at_k.to_numpy(); lower, upper = [], []
    for frac, center in zip(z.capacity_fraction, centers):
        lo, hi = np.quantile(policy_boot[(frac, policy)], [.025, .975])
        lower.append(center-lo); upper.append(hi-center)
    axes[2].errorbar(z.capacity_fraction * 100, centers, yerr=[lower, upper], marker=marker,
                     color=color, linewidth=1.4, capsize=2, markersize=4, label=policy)
axes[2].set(xlabel="Follow-up capacity (% of cohort)", ylabel="Precision@K", ylim=(0, 1))
axes[2].legend(fontsize=5.8)
for i, ax in enumerate(axes):
    ax.grid(axis="y", color="#E5E7EB", linewidth=.5)
    ax.text(-.16, 1.04, chr(97+i), transform=ax.transAxes, fontweight="bold", fontsize=8)
save_figure(fig, "Fig2_profiles_and_stratification")
