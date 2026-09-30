from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from run_oulad_experiments import (  # noqa: E402
    CHANNELS, DB, HORIZONS, OUT, RANDOM_STATE, load_or_build_weekly, make_features, oof_predict,
)

FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)


def mask_channels(x: pd.DataFrame, channels_by_row: np.ndarray, temporal: list[str]) -> pd.DataFrame:
    """Zero unavailable software channels and recompute cross-channel summaries."""
    z = x.copy()
    for j, channel in enumerate(CHANNELS):
        rows = channels_by_row[:, j]
        cols = [f for f in temporal if f.startswith(channel + "_")]
        z.loc[rows, cols] = 0.0
    total_cols = [f"{c}_total" for c in CHANNELS]
    z["all_total"] = z[total_cols].sum(axis=1)
    totals = z[total_cols].to_numpy(float)
    shares = totals / (totals.sum(axis=1, keepdims=True) + 1.0)
    z["channel_entropy"] = -(shares * np.log(shares + 1e-12)).sum(axis=1)
    return z


def pooled_metrics(y, p):
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
    }


def cluster_ci(y, p, groups, metric, n_boot=2000):
    rng = np.random.default_rng(RANDOM_STATE)
    ug = np.unique(groups); ix_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(ug, size=len(ug), replace=True)
        ix = np.concatenate([ix_by_g[g] for g in sampled])
        if len(np.unique(y[ix])) == 2:
            vals.append(metric(y[ix], p[ix]))
    return np.quantile(vals, [.025, .975])


def main():
    info = pd.read_csv(DB / "studentInfo.csv")
    registration = pd.read_csv(DB / "studentRegistration.csv", na_values="?")
    info = info.merge(
        registration[["code_module", "code_presentation", "id_student", "date_unregistration"]],
        on=["code_module", "code_presentation", "id_student"], how="left", validate="one_to_one"
    )
    weekly = load_or_build_weekly()
    # Recompute leakage-safe OOF trajectories so panels a-b can carry
    # presentation-cluster bootstrap uncertainty around pooled metrics.
    trajectory = []
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])
    for horizon in HORIZONS:
        dh, base_h, temporal_h = make_features(weekly, horizon, info)
        for label, features in (("Cumulative", base_h), ("Temporal virtual sensing", temporal_h)):
            ph, _ = oof_predict(dh, features, lr, "presentation_group")
            yh, gh = dh.target.to_numpy(), dh.presentation_group.to_numpy()
            row = {"horizon_days": horizon, "feature_set": label}
            for name, fn in (("AUROC", roc_auc_score), ("AUPRC", average_precision_score)):
                lo, hi = cluster_ci(yh, ph, gh, fn)
                row[name] = fn(yh, ph); row[f"{name}_low"] = lo; row[f"{name}_high"] = hi
            trajectory.append(row)
    trajectory = pd.DataFrame(trajectory)
    trajectory.to_csv(OUT / "performance_trajectory_ci.csv", index=False)

    data, _, temporal = make_features(weekly, 56, info)
    y = data.target.to_numpy()
    groups = data.presentation_group.to_numpy()

    full = np.zeros(len(data))
    channel_pred = {c: np.zeros(len(data)) for c in CHANNELS}
    severities = (0.10, 0.20, 0.30)
    n_seeds = 10
    random_pred = {(q, s): np.zeros(len(data)) for q in severities for s in range(n_seeds)}
    logo = LeaveOneGroupOut()
    for fold, (tr, te) in enumerate(logo.split(data, y, groups)):
        fitted = clone(lr).fit(data.iloc[tr][temporal], y[tr])
        xte = data.iloc[te][temporal].copy().reset_index(drop=True)
        full[te] = fitted.predict_proba(xte)[:, 1]
        for j, channel in enumerate(CHANNELS):
            mask = np.zeros((len(te), len(CHANNELS)), dtype=bool)
            mask[:, j] = True
            channel_pred[channel][te] = fitted.predict_proba(mask_channels(xte, mask, temporal))[:, 1]
        for q in severities:
            for seed in range(n_seeds):
                rng = np.random.default_rng(RANDOM_STATE + 1000 * fold + 100 * seed + int(q * 100))
                mask = rng.random((len(te), len(CHANNELS))) < q
                random_pred[(q, seed)][te] = fitted.predict_proba(mask_channels(xte, mask, temporal))[:, 1]

    baseline = pooled_metrics(y, full)
    rows = [{"condition": "Full", "channel": "None", "severity": 0.0, "seed": -1, **baseline}]
    for channel, p in channel_pred.items():
        rows.append({"condition": "Single-channel outage", "channel": channel,
                     "severity": 1 / len(CHANNELS), "seed": -1, **pooled_metrics(y, p)})
    for (q, seed), p in random_pred.items():
        rows.append({"condition": "Random channel outage", "channel": "Random",
                     "severity": q, "seed": seed, **pooled_metrics(y, p)})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "channel_dropout_results.csv", index=False)

    # Presentation-level paired differences for the deterministic channel outages.
    fold_rows = []
    for channel, p in channel_pred.items():
        for g in np.unique(groups):
            ix = groups == g
            fold_rows.append({
                "channel": channel, "presentation": g,
                "delta_AUROC": roc_auc_score(y[ix], p[ix]) - roc_auc_score(y[ix], full[ix]),
                "delta_AUPRC": average_precision_score(y[ix], p[ix]) - average_precision_score(y[ix], full[ix]),
                "delta_Brier": brier_score_loss(y[ix], p[ix]) - brier_score_loss(y[ix], full[ix]),
            })
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "channel_dropout_fold_differences.csv", index=False)

    # Cluster bootstrap across the 22 held-out presentations and paired
    # two-sided Wilcoxon tests; presentation is the independent unit.
    rng = np.random.default_rng(RANDOM_STATE)
    summary_rows = []
    for channel in CHANNELS:
        z = folds[folds.channel == channel]
        row = {"channel": channel, "n_presentations": len(z)}
        for metric in ("delta_AUROC", "delta_AUPRC", "delta_Brier"):
            v = z[metric].to_numpy()
            boot = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(5000)])
            row[f"{metric}_mean"] = v.mean()
            row[f"{metric}_ci_low"] = np.quantile(boot, .025)
            row[f"{metric}_ci_high"] = np.quantile(boot, .975)
            row[f"{metric}_p_raw"] = wilcoxon(v, alternative="two-sided", zero_method="wilcox").pvalue
        summary_rows.append(row)
    channel_summary = pd.DataFrame(summary_rows)
    # Holm correction within each metric family of six channel tests.
    for metric in ("delta_AUROC", "delta_AUPRC", "delta_Brier"):
        p = channel_summary[f"{metric}_p_raw"].to_numpy()
        order_ix = np.argsort(p); adj = np.empty_like(p); running = 0.0
        for rank, ix in enumerate(order_ix):
            running = max(running, (len(p) - rank) * p[ix]); adj[ix] = min(running, 1.0)
        channel_summary[f"{metric}_p_holm"] = adj
    channel_summary.to_csv(OUT / "channel_dropout_summary.csv", index=False)

    # Publication figure: performance trajectory plus channel-outage audit.
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    colors = {"Cumulative": "#9AA3AD", "Temporal virtual sensing": "#176B87"}
    for metric, ax in zip(("AUROC", "AUPRC"), axes[:2]):
        for fs in ("Cumulative", "Temporal virtual sensing"):
            z = trajectory[trajectory.feature_set == fs].sort_values("horizon_days")
            yerr = np.vstack([z[metric] - z[f"{metric}_low"], z[f"{metric}_high"] - z[metric]])
            ax.errorbar(z.horizon_days, z[metric], yerr=yerr, marker="o", lw=1.5, capsize=2,
                        color=colors[fs], label=fs)
        ax.set(xlabel="Observation horizon (days)", ylabel=f"Pooled {metric}", xticks=list(HORIZONS))
        ax.grid(axis="y", color="#E5E7EB", lw=.5)
    axes[0].legend(loc="lower right", fontsize=6)

    order = channel_summary.sort_values("delta_AUPRC_mean").channel.tolist()
    ypos = np.arange(len(order))
    cs = channel_summary.set_index("channel").reindex(order)
    means_pr = cs["delta_AUPRC_mean"]
    means_roc = cs["delta_AUROC_mean"]
    axes[2].axvline(0, color="#6B7280", lw=.7)
    xerr_pr = np.vstack([means_pr - cs["delta_AUPRC_ci_low"], cs["delta_AUPRC_ci_high"] - means_pr])
    xerr_roc = np.vstack([means_roc - cs["delta_AUROC_ci_low"], cs["delta_AUROC_ci_high"] - means_roc])
    axes[2].errorbar(means_pr, ypos - .12, xerr=xerr_pr, fmt="o", ms=3.5, capsize=2,
                     color="#C4473D", lw=.8, label="ΔAUPRC")
    axes[2].errorbar(means_roc, ypos + .12, xerr=xerr_roc, fmt="s", ms=3.2, capsize=2,
                     mfc="white", color="#176B87", lw=.8, label="ΔAUROC")
    axes[2].set(yticks=ypos, yticklabels=[x.capitalize() for x in order], xlabel="Mean held-out change after channel outage")
    axes[2].grid(axis="x", color="#E5E7EB", lw=.5)
    axes[2].legend(loc="lower right", fontsize=6)
    for i, ax in enumerate(axes):
        ax.text(-.15, 1.04, chr(97 + i), transform=ax.transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout(w_pad=1.0)
    stem = FIG / "Fig2_performance_and_channel_outage"
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Full:", baseline)
    print(results.to_string(index=False))
    print("\nChannel outage summary:\n", channel_summary.to_string(index=False))
    print("\nMean paired presentation differences:\n", folds.groupby("channel").mean(numeric_only=True).sort_values("delta_AUPRC"))


if __name__ == "__main__":
    main()
