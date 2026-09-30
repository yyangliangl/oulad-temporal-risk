from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "Database"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

HORIZONS = (28, 56, 84)
MAX_HORIZON = max(HORIZONS)
RANDOM_STATE = 20260814
GMM_COMPONENTS = 2

CHANNEL_MAP = {
    "oucontent": "content",
    "resource": "content",
    "page": "content",
    "url": "content",
    "folder": "content",
    "htmlactivity": "content",
    "dualpane": "content",
    "subpage": "navigation",
    "homepage": "navigation",
    "sharedsubpage": "navigation",
    "repeatactivity": "navigation",
    "quiz": "assessment",
    "externalquiz": "assessment",
    "questionnaire": "assessment",
    "forumng": "communication",
    "glossary": "communication",
    "oucollaborate": "collaboration",
    "ouwiki": "collaboration",
    "ouelluminate": "collaboration",
    "dataplus": "enrichment",
}
CHANNELS = ["content", "navigation", "assessment", "communication", "collaboration", "enrichment"]


def load_or_build_weekly() -> pd.DataFrame:
    cache = OUT / "weekly_channel_counts.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    vle = pd.read_csv(DB / "vle.csv", usecols=["id_site", "activity_type"])
    site_to_channel = vle.drop_duplicates("id_site").set_index("id_site")["activity_type"].map(CHANNEL_MAP)
    parts = []
    usecols = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    for chunk in pd.read_csv(DB / "studentVle.csv", usecols=usecols, chunksize=1_000_000):
        chunk = chunk[(chunk["date"] >= 0) & (chunk["date"] < MAX_HORIZON)].copy()
        chunk["channel"] = chunk["id_site"].map(site_to_channel).fillna("enrichment")
        chunk["week"] = (chunk["date"] // 7).astype("int8")
        agg = chunk.groupby(
            ["code_module", "code_presentation", "id_student", "channel", "week"], observed=True
        )["sum_click"].sum().reset_index()
        parts.append(agg)
    weekly = pd.concat(parts, ignore_index=True)
    weekly = weekly.groupby(
        ["code_module", "code_presentation", "id_student", "channel", "week"], observed=True
    )["sum_click"].sum().reset_index()
    weekly.to_parquet(cache, index=False)
    return weekly


def make_features(weekly: pd.DataFrame, horizon: int, info: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    n_weeks = horizon // 7
    w = weekly[weekly["week"] < n_weeks]
    idx = ["code_module", "code_presentation", "id_student"]
    pivot = w.pivot_table(index=idx, columns=["channel", "week"], values="sum_click", aggfunc="sum", fill_value=0)
    full_cols = pd.MultiIndex.from_product([CHANNELS, range(n_weeks)], names=["channel", "week"])
    pivot = pivot.reindex(columns=full_cols, fill_value=0)
    feat = pd.DataFrame(index=pivot.index)
    baseline, temporal = [], []
    x = np.arange(n_weeks, dtype=float)
    x_center = x - x.mean()
    denom = np.square(x_center).sum()
    for channel in CHANNELS:
        arr = pivot[channel].to_numpy(dtype=float)
        total = arr.sum(axis=1)
        names = {
            f"{channel}_total": total,
            f"{channel}_active_weeks": (arr > 0).sum(axis=1),
            f"{channel}_weekly_sd": arr.std(axis=1),
            f"{channel}_slope": (arr @ x_center) / denom,
            f"{channel}_recent_share": arr[:, n_weeks // 2 :].sum(axis=1) / (total + 1.0),
            f"{channel}_last_week": arr[:, -1],
        }
        for name, values in names.items():
            feat[name] = values
            (baseline if name.endswith("_total") else temporal).append(name)
    feat["all_total"] = feat[[f"{c}_total" for c in CHANNELS]].sum(axis=1)
    baseline.append("all_total")
    feat["channel_entropy"] = 0.0
    shares = feat[[f"{c}_total" for c in CHANNELS]].to_numpy() / (feat["all_total"].to_numpy()[:, None] + 1.0)
    feat["channel_entropy"] = -(shares * np.log(shares + 1e-12)).sum(axis=1)
    temporal.append("channel_entropy")
    feat = feat.reset_index()
    merged = info.merge(feat, on=idx, how="left").fillna({c: 0 for c in baseline + temporal})
    # Landmark design: predict only among students still enrolled at the cutoff.
    merged = merged[merged["date_unregistration"].isna() | (merged["date_unregistration"] > horizon)].copy()
    merged["target"] = merged["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    merged["presentation_group"] = merged["code_module"] + "-" + merged["code_presentation"]
    return merged, baseline, baseline + temporal


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "F1": f1_score(y, pred),
        "Balanced accuracy": balanced_accuracy_score(y, pred),
        "Brier": brier_score_loss(y, p),
    }


def oof_predict(data: pd.DataFrame, features: list[str], model, group_col: str, target_col: str = "target") -> tuple[np.ndarray, pd.DataFrame]:
    y = data[target_col].to_numpy()
    groups = data[group_col].to_numpy()
    p = np.zeros(len(data), dtype=float)
    fold_rows = []
    logo = LeaveOneGroupOut()
    for fold, (tr, te) in enumerate(logo.split(data, y, groups)):
        fitted = clone(model).fit(data.iloc[tr][features], y[tr])
        p[te] = fitted.predict_proba(data.iloc[te][features])[:, 1]
        row = {"fold": fold, "held_out": groups[te][0], **metrics(y[te], p[te])}
        fold_rows.append(row)
    return p, pd.DataFrame(fold_rows)


def bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, groups: np.ndarray, fn, n=1000) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    unique_groups = np.unique(groups)
    group_indices = {g: np.flatnonzero(groups == g) for g in unique_groups}
    vals = []
    for _ in range(n):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        ix = np.concatenate([group_indices[g] for g in sampled_groups])
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(fn(y[ix], p_b[ix]) - fn(y[ix], p_a[ix]))
    return float(np.mean(vals)), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm step-down family-wise error correction."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, ix in enumerate(order):
        running = max(running, (m - rank) * p[ix])
        adjusted[ix] = min(running, 1.0)
    return adjusted.tolist()


def profile_augmented_oof(data, features, profile_features, model):
    """Fit scaler/GMM and the downstream classifier only on each LOPO training split."""
    y = data.target.to_numpy()
    groups = data.presentation_group.to_numpy()
    p = np.zeros(len(data), dtype=float)
    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(data, y, groups)):
        ps = StandardScaler().fit(data.iloc[tr][profile_features])
        xtr_p, xte_p = ps.transform(data.iloc[tr][profile_features]), ps.transform(data.iloc[te][profile_features])
        rng = np.random.default_rng(RANDOM_STATE + fold)
        fit_ix = rng.choice(len(xtr_p), size=min(2500, len(xtr_p)), replace=False)
        gmm = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type="diag", reg_covar=1e-4,
                              n_init=2, max_iter=250, random_state=RANDOM_STATE).fit(xtr_p[fit_ix])
        tr_lab, te_lab = gmm.predict(xtr_p), gmm.predict(xte_p)
        train_risk = pd.Series(y[tr]).groupby(tr_lab).mean().sort_values()
        rank = {old: new for new, old in enumerate(train_risk.index)}
        tr_lab = np.array([rank[v] for v in tr_lab]); te_lab = np.array([rank[v] for v in te_lab])
        tr_df, te_df = data.iloc[tr][features].copy(), data.iloc[te][features].copy()
        for j in range(GMM_COMPONENTS):
            tr_df[f"profile_{j}"] = (tr_lab == j).astype(int)
            te_df[f"profile_{j}"] = (te_lab == j).astype(int)
        fitted = clone(model).fit(tr_df, y[tr])
        p[te] = fitted.predict_proba(te_df)[:, 1]
    return p


def leakage_safe_profiles(data, profile_features):
    y = data.target.to_numpy(); groups = data.presentation_group.to_numpy()
    profile = np.full(len(data), -1, dtype=int)
    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(data, y, groups)):
        scaler = StandardScaler().fit(data.iloc[tr][profile_features])
        xtr = scaler.transform(data.iloc[tr][profile_features]); xte = scaler.transform(data.iloc[te][profile_features])
        rng = np.random.default_rng(RANDOM_STATE + fold)
        fit_ix = rng.choice(len(xtr), size=min(2500, len(xtr)), replace=False)
        gmm = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type="diag", reg_covar=1e-4,
                              n_init=2, max_iter=250, random_state=RANDOM_STATE).fit(xtr[fit_ix])
        train_labels = gmm.predict(xtr)
        order = pd.Series(y[tr]).groupby(train_labels).mean().sort_values().index.to_list()
        rank = {old: new for new, old in enumerate(order)}
        profile[te] = np.array([rank[v] for v in gmm.predict(xte)])
    return profile


def save_tiff(fig: plt.Figure, name: str, width_in: float, height_in: float) -> None:
    fig.set_size_inches(width_in, height_in)
    path = FIG / f"{name}.tiff"
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(path) as im:
        assert im.format == "TIFF" and im.info.get("dpi", (0, 0))[0] >= 599


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "legend.frameon": False, "figure.facecolor": "white",
    })
    info = pd.read_csv(DB / "studentInfo.csv")
    registration = pd.read_csv(DB / "studentRegistration.csv", na_values="?")
    info = info.merge(registration[["code_module", "code_presentation", "id_student", "date_unregistration"]],
                      on=["code_module", "code_presentation", "id_student"], how="left", validate="one_to_one")
    weekly = load_or_build_weekly()
    lr = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])
    rf = RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=10, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)
    hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=.06, max_leaf_nodes=15,
                                         l2_regularization=1.0, random_state=RANDOM_STATE)

    summary_rows, fold_frames = [], []
    saved = {}
    for horizon in HORIZONS:
        data, base, temporal = make_features(weekly, horizon, info)
        for feature_set, features in [("Cumulative", base), ("Temporal virtual sensing", temporal)]:
            p, folds = oof_predict(data, features, lr, "presentation_group")
            row = {"horizon_days": horizon, "feature_set": feature_set, "model": "Logistic regression", **metrics(data.target.to_numpy(), p)}
            summary_rows.append(row)
            folds["horizon_days"], folds["feature_set"], folds["model"] = horizon, feature_set, "Logistic regression"
            fold_frames.append(folds)
            saved[(horizon, feature_set, "LR")] = (data, features, p)

    data, base, temporal = make_features(weekly, 56, info)
    for feature_set, features in [("Cumulative", base), ("Temporal virtual sensing", temporal)]:
        p, folds = oof_predict(data, features, rf, "presentation_group")
        summary_rows.append({"horizon_days": 56, "feature_set": feature_set, "model": "Random forest", **metrics(data.target.to_numpy(), p)})
        folds["horizon_days"], folds["feature_set"], folds["model"] = 56, feature_set, "Random forest"
        fold_frames.append(folds)
        saved[(56, feature_set, "RF")] = (data, features, p)
    for feature_set, features in [("Cumulative", base), ("Temporal virtual sensing", temporal)]:
        p, folds = oof_predict(data, features, hgb, "presentation_group")
        summary_rows.append({"horizon_days": 56, "feature_set": feature_set, "model": "HistGradientBoosting", **metrics(data.target.to_numpy(), p)})
        folds["horizon_days"], folds["feature_set"], folds["model"] = 56, feature_set, "HistGradientBoosting"
        fold_frames.append(folds)

    endpoint_rows = []
    for endpoint, target_col in [("Failure", "target_fail"), ("Withdrawal", "target_withdrawn")]:
        data[target_col] = (data["final_result"] == ("Fail" if endpoint == "Failure" else "Withdrawn")).astype(int)
        for feature_set, features in [("Cumulative", base), ("Temporal virtual sensing", temporal)]:
            p, _ = oof_predict(data, features, lr, "presentation_group", target_col=target_col)
            endpoint_rows.append({"endpoint": endpoint, "feature_set": feature_set, **metrics(data[target_col].to_numpy(), p)})

    # Ablate feature families from the 56-day temporal representation.
    families = {
        "Full": temporal,
        "Without trends": [f for f in temporal if not (f.endswith("_slope") or f.endswith("_recent_share"))],
        "Without variability": [f for f in temporal if not f.endswith("_weekly_sd")],
        "Cumulative only": base,
    }
    ablation = []
    for name, features in families.items():
        p, _ = oof_predict(data, features, lr, "presentation_group")
        ablation.append({"feature_set": name, **metrics(data.target.to_numpy(), p)})
    channel_addition = []
    for channel in CHANNELS:
        channel_features = base + [f for f in temporal if f.startswith(channel + "_") and f not in base]
        p, _ = oof_predict(data, channel_features, lr, "presentation_group")
        channel_addition.append({"channel": channel, **metrics(data.target.to_numpy(), p)})

    # Cross-module transfer is a deliberately harder external-validity test.
    p_module, module_folds = oof_predict(data, temporal, lr, "code_module")
    transfer = {"leave_one_module_out": metrics(data.target.to_numpy(), p_module)}

    # Leakage-safe behavioral profiles: scaler and GMM fitted within every held-out presentation fold.
    profile_features = [f for f in temporal if f.endswith(("_total", "_slope", "_recent_share"))] + ["channel_entropy"]
    y = data.target.to_numpy()
    groups = data.presentation_group.to_numpy()
    profile = leakage_safe_profiles(data, profile_features)
    data["profile"] = profile
    profile_stats = data.groupby("profile").agg(n=("target", "size"), risk_rate=("target", "mean"), withdrawn_rate=("final_result", lambda s: (s == "Withdrawn").mean()), fail_rate=("final_result", lambda s: (s == "Fail").mean())).reset_index()

    # Descriptive sensitivity analysis for the number of mixture components.
    # This diagnostic is not used to tune any held-out prediction.
    gmm_rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    xall = StandardScaler().fit_transform(data[profile_features])
    fit_ix = rng.choice(len(xall), size=min(5000, len(xall)), replace=False)
    xdiag = xall[fit_ix]
    sample_ix = rng.choice(len(xdiag), size=min(1500, len(xdiag)), replace=False)
    for k in range(2, 7):
        g1 = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4,
                             n_init=5, max_iter=500, random_state=RANDOM_STATE).fit(xdiag)
        g2 = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4,
                             n_init=5, max_iter=500, random_state=RANDOM_STATE + 17).fit(xdiag)
        lab1, lab2 = g1.predict(xdiag), g2.predict(xdiag)
        counts = np.bincount(lab1, minlength=k)
        gmm_rows.append({"k": k, "bic_per_student": g1.bic(xdiag) / len(xdiag),
                         "silhouette": silhouette_score(xdiag[sample_ix], lab1[sample_ix]),
                         "ari_two_seeds": adjusted_rand_score(lab1, lab2),
                         "min_cluster_fraction": counts.min() / len(xdiag)})

    p_profile_aug = profile_augmented_oof(data, temporal, profile_features, lr)
    profile_gain = pd.DataFrame([
        {"feature_set": "Temporal virtual sensing", **metrics(y, saved[(56, "Temporal virtual sensing", "LR")][2])},
        {"feature_set": "Temporal + profile indicators", **metrics(y, p_profile_aug)},
    ])

    # Repeated landmark profiles support a genuinely dynamic interpretation.
    profile_landmarks = []
    id_cols = ["code_module", "code_presentation", "id_student"]
    for horizon in HORIZONS:
        dh, fh, _ = saved[(horizon, "Temporal virtual sensing", "LR")]
        pfh = [f for f in fh if f.endswith(("_total", "_slope", "_recent_share"))] + ["channel_entropy"]
        tmp = dh[id_cols].copy()
        tmp["horizon_days"] = horizon
        tmp["profile"] = leakage_safe_profiles(dh, pfh)
        profile_landmarks.append(tmp)
    landmark_profiles = pd.concat(profile_landmarks, ignore_index=True)
    wide_profiles = landmark_profiles.pivot(index=id_cols, columns="horizon_days", values="profile")
    transition_rows = []
    for a, b in [(28, 56), (56, 84)]:
        z = wide_profiles[[a, b]].dropna().astype(int)
        tab = pd.crosstab(z[a], z[b])
        for origin in tab.index:
            for dest in tab.columns:
                transition_rows.append({"from_day": a, "to_day": b, "from_profile": origin,
                                        "to_profile": dest, "n": int(tab.loc[origin, dest]),
                                        "row_fraction": float(tab.loc[origin, dest] / tab.loc[origin].sum())})

    # Operational stratification from OOF probabilities and fixed intervention capacity.
    p_main = saved[(56, "Temporal virtual sensing", "LR")][2]
    data["risk_probability"] = p_main
    data["risk_stratum"] = pd.qcut(p_main, [0, .5, .8, 1], labels=["Low", "Moderate", "High"], duplicates="drop")
    strata = data.groupby("risk_stratum", observed=True).agg(n=("target", "size"), observed_risk=("target", "mean"), mean_probability=("risk_probability", "mean")).reset_index()
    capacity = []
    for frac in (.05, .10, .15, .20, .30):
        k = int(np.ceil(len(data) * frac))
        ix = np.argsort(-p_main)[:k]
        capacity.append({"capacity_fraction": frac, "k": k, "precision_at_k": y[ix].mean(), "recall_at_k": y[ix].sum() / y.sum()})
    capacity_compare = []
    for frac in (.05, .10, .15, .20, .30):
        k = int(np.ceil(len(data) * frac))
        global_ix = np.argsort(-p_main)[:k]
        selected = []
        quota = k // GMM_COMPONENTS
        for prof in range(GMM_COMPONENTS):
            cand = np.flatnonzero(profile == prof)
            selected.extend(cand[np.argsort(-p_main[cand])[:min(quota, len(cand))]].tolist())
        selected = list(dict.fromkeys(selected))
        if len(selected) < k:
            remaining = np.setdiff1d(np.argsort(-p_main), np.asarray(selected), assume_unique=False)
            selected.extend(remaining[:k-len(selected)].tolist())
        for policy, ix in [("Global probability ranking", global_ix), ("Profile-balanced quota", np.asarray(selected[:k]))]:
            capacity_compare.append({"capacity_fraction": frac, "k": k, "policy": policy,
                                     "precision_at_k": y[ix].mean(), "recall_at_k": y[ix].sum() / y.sum()})

    # Feature-level distribution shift across anonymous modules.
    shift_rows = []
    for feature in temporal:
        vals = []
        for module in sorted(data.code_module.unique()):
            a = data.loc[data.code_module == module, feature].to_numpy()
            b = data.loc[data.code_module != module, feature].to_numpy()
            vals.append(ks_2samp(a, b).statistic)
        shift_rows.append({"feature": feature, "mean_one_vs_rest_ks": np.mean(vals), "max_one_vs_rest_ks": np.max(vals)})

    summary = pd.DataFrame(summary_rows)
    folds_all = pd.concat(fold_frames, ignore_index=True)

    # Paired presentation-fold tests for temporal vs cumulative representations.
    infer_rows = []
    for metric_name in ["AUROC", "AUPRC"]:
        raw = []
        tmp = []
        for horizon in HORIZONS:
            a = folds_all[(folds_all.horizon_days == horizon) & (folds_all.feature_set == "Cumulative") & (folds_all.model == "Logistic regression")].sort_values("held_out")
            b = folds_all[(folds_all.horizon_days == horizon) & (folds_all.feature_set == "Temporal virtual sensing") & (folds_all.model == "Logistic regression")].sort_values("held_out")
            stat, pval = wilcoxon(b[metric_name].to_numpy(), a[metric_name].to_numpy(), alternative="greater", zero_method="wilcox")
            raw.append(pval); tmp.append((horizon, stat, pval, (b[metric_name].to_numpy()-a[metric_name].to_numpy()).mean()))
        adj = holm_adjust(raw)
        for row, padj in zip(tmp, adj):
            infer_rows.append({"metric": metric_name, "horizon_days": row[0], "wilcoxon_statistic": row[1],
                               "mean_fold_difference": row[3], "p_raw": row[2], "p_holm": padj, "n_presentations": 22})

    summary.to_csv(OUT / "model_summary.csv", index=False)
    folds_all.to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(ablation).to_csv(OUT / "ablation.csv", index=False)
    pd.DataFrame(channel_addition).to_csv(OUT / "channel_addition.csv", index=False)
    pd.DataFrame(endpoint_rows).to_csv(OUT / "endpoint_analysis.csv", index=False)
    module_folds.to_csv(OUT / "module_transfer_folds.csv", index=False)
    profile_stats.to_csv(OUT / "profile_stats.csv", index=False)
    pd.DataFrame(gmm_rows).to_csv(OUT / "gmm_k_diagnostics.csv", index=False)
    profile_gain.to_csv(OUT / "profile_predictive_gain.csv", index=False)
    landmark_profiles.to_csv(OUT / "profile_landmarks.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(OUT / "profile_transitions.csv", index=False)
    strata.to_csv(OUT / "risk_strata.csv", index=False)
    pd.DataFrame(capacity).to_csv(OUT / "capacity_policy.csv", index=False)
    pd.DataFrame(capacity_compare).to_csv(OUT / "capacity_policy_comparison.csv", index=False)
    pd.DataFrame(shift_rows).sort_values("mean_one_vs_rest_ks", ascending=False).to_csv(OUT / "module_feature_shift.csv", index=False)
    pd.DataFrame(infer_rows).to_csv(OUT / "paired_inference.csv", index=False)
    data[["code_module", "code_presentation", "id_student", "final_result", "target", "profile", "risk_probability", "risk_stratum"]].to_csv(OUT / "oof_predictions.csv", index=False)

    y56 = data.target.to_numpy()
    p_base = saved[(56, "Cumulative", "LR")][2]
    deltas = {
        "AUROC": bootstrap_delta(y56, p_base, p_main, data.presentation_group.to_numpy(), roc_auc_score),
        "AUPRC": bootstrap_delta(y56, p_base, p_main, data.presentation_group.to_numpy(), average_precision_score),
    }
    audit = {
        "n_enrolments": int(len(info)), "n_unique_students": int(info.id_student.nunique()),
        "landmark_sample_sizes": {str(h): int(len(saved[(h, "Temporal virtual sensing", "LR")][0])) for h in HORIZONS},
        "n_presentations": int(info.groupby(["code_module", "code_presentation"]).ngroups),
        "outcomes": info.final_result.value_counts().to_dict(), "channel_map": CHANNEL_MAP,
        "bootstrap_temporal_minus_cumulative": deltas, "cross_module": transfer,
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # Figure 1: performance trajectory and ablation evidence.
    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    palette = {"Cumulative": "#9AA3AD", "Temporal virtual sensing": "#2B6F8A"}
    lr_s = summary[summary.model == "Logistic regression"]
    for metric, ax in zip(["AUROC", "AUPRC"], axes[:2]):
        for fs in ["Cumulative", "Temporal virtual sensing"]:
            z = lr_s[lr_s.feature_set == fs]
            ax.plot(z.horizon_days, z[metric], marker="o", lw=1.6, color=palette[fs], label=fs)
        ax.set(xlabel="Observation horizon (days)", ylabel=metric, xticks=HORIZONS)
        ax.grid(axis="y", color="#E5E7EB", lw=.5)
    abl = pd.DataFrame(ablation).sort_values("AUPRC")
    axes[2].barh(abl.feature_set, abl.AUPRC, color=["#2B6F8A" if x == "Full" else "#B8C0C8" for x in abl.feature_set])
    axes[2].set(xlabel="AUPRC", ylabel="", title="56-day ablation")
    axes[0].legend(loc="lower right")
    for i, ax in enumerate(axes): ax.text(-.16, 1.04, chr(97+i), transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout(w_pad=1.2)
    save_tiff(fig, "Fig1_predictive_performance", 7.1, 2.35)

    # Figure 2: behavioral profiles, risk strata, and capacity-constrained decisions.
    fig, axes = plt.subplots(1, 3)
    ps = profile_stats.copy(); ps["profile_label"] = [f"P{i+1}" for i in ps.profile]
    axes[0].bar(ps.profile_label, ps.risk_rate, color=["#7FB0C2", "#C85A54"][:len(ps)])
    axes[0].set(xlabel="Behavioral profile", ylabel="Observed academic-risk rate", ylim=(0, 1))
    axes[1].bar(strata.risk_stratum.astype(str), strata.observed_risk, color=["#B7D5DF", "#E4B363", "#C85A54"])
    axes[1].set(xlabel="Predicted risk stratum", ylabel="Observed academic-risk rate", ylim=(0, 1))
    cap = pd.DataFrame(capacity_compare)
    for policy, marker, color in [("Global probability ranking", "o", "#2B6F8A"),
                                  ("Profile-balanced quota", "s", "#C85A54")]:
        z = cap[cap.policy == policy]
        axes[2].plot(z.capacity_fraction * 100, z.precision_at_k, marker=marker, color=color, label=policy)
    axes[2].set(xlabel="Intervention capacity (% of cohort)", ylabel="Precision@K", ylim=(0, 1))
    axes[2].legend()
    for i, ax in enumerate(axes):
        ax.grid(axis="y", color="#E5E7EB", lw=.5)
        ax.text(-.16, 1.04, chr(97+i), transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.tight_layout(w_pad=1.2)
    save_tiff(fig, "Fig2_profiles_and_stratification", 7.1, 2.35)

    print(summary.to_string(index=False))
    print("\nAblation\n", pd.DataFrame(ablation).to_string(index=False))
    print("\nProfiles\n", profile_stats.to_string(index=False))
    print("\nStrata\n", strata.to_string(index=False))
    print("\nCapacity\n", pd.DataFrame(capacity).to_string(index=False))
    print("\nAudit\n", json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
