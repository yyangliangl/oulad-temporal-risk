from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from run_oulad_experiments import load_or_build_weekly, make_features, holm_adjust


ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "Database"
OUT = ROOT / "outputs"
RNG = np.random.default_rng(20260820)
N_BOOT = 1000


def hl_shift(x: np.ndarray, y: np.ndarray, iterations: int = 42) -> float:
    """Approximate the two-sample Hodges-Lehmann shift median(x_i-y_j).

    The bisection counts pairwise differences without materializing the full
    n_x by n_y matrix. Values are accurate to floating-point precision for the
    count-valued and summary features used here.
    """
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    lo = float(x[0] - y[-1])
    hi = float(x[-1] - y[0])
    target = (len(x) * len(y) - 1) // 2
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        # x-y <= mid iff y >= x-mid.
        count = np.sum(len(y) - np.searchsorted(y, x - mid, side="left"))
        if count <= target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def cluster_boot_hl(data: pd.DataFrame, feature: str) -> tuple[float, float]:
    groups = data["presentation_group"].unique()
    by_group = {g: data[data["presentation_group"] == g] for g in groups}
    vals: list[float] = []
    for _ in range(N_BOOT):
        sampled = RNG.choice(groups, size=len(groups), replace=True)
        boot = pd.concat([by_group[g] for g in sampled], ignore_index=True)
        x = boot.loc[boot.final_result == "Fail", feature].to_numpy()
        y = boot.loc[boot.final_result == "Withdrawn", feature].to_numpy()
        if len(x) and len(y):
            vals.append(hl_shift(x, y))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def fmt_median_iqr(s: pd.Series) -> str:
    q = s.quantile([0.25, 0.5, 0.75])
    return f"{q.loc[0.5]:.2f} [{q.loc[0.25]:.2f}, {q.loc[0.75]:.2f}]"


def main() -> None:
    info = pd.read_csv(DB / "studentInfo.csv")
    reg = pd.read_csv(DB / "studentRegistration.csv", na_values="?")
    info = info.merge(
        reg[["code_module", "code_presentation", "id_student", "date_unregistration"]],
        on=["code_module", "code_presentation", "id_student"],
        how="left",
        validate="one_to_one",
    )
    data, _, _ = make_features(load_or_build_weekly(), 56, info)
    data = data[data.final_result.isin(["Fail", "Withdrawn"])].copy()

    # Construct seven prespecified, student-level endpoint descriptors.
    weekly = load_or_build_weekly()
    w = weekly[weekly.week < 8].groupby(
        ["code_module", "code_presentation", "id_student", "week"], observed=True
    ).sum_click.sum().unstack(fill_value=0).reindex(columns=range(8), fill_value=0)
    arr = w.to_numpy(float)
    xc = np.arange(8, dtype=float) - 3.5
    agg = pd.DataFrame({
        "active_weeks": (arr > 0).sum(axis=1),
        "recent_activity_share": arr[:, 4:].sum(axis=1) / (arr.sum(axis=1) + 1.0),
        "activity_slope": arr @ xc / np.square(xc).sum(),
        "weekly_variability": arr.std(axis=1),
    }, index=w.index).reset_index()
    data = data.merge(agg, on=["code_module", "code_presentation", "id_student"], how="left")
    derived = ["active_weeks", "recent_activity_share", "activity_slope", "weekly_variability"]
    data[derived] = data[derived].fillna(0)

    features = [
        ("Cumulative activity", "all_total"),
        ("Active weeks", "active_weeks"),
        ("Recent-activity share", "recent_activity_share"),
        ("Activity slope", "activity_slope"),
        ("Weekly variability", "weekly_variability"),
        ("Assessment-channel activity", "assessment_total"),
        ("Communication-channel activity", "communication_total"),
    ]
    rows = []
    raw_p = []
    for label, feature in features:
        fail = data.loc[data.final_result == "Fail", feature].to_numpy(float)
        wd = data.loc[data.final_result == "Withdrawn", feature].to_numpy(float)
        test = mannwhitneyu(fail, wd, alternative="two-sided", method="asymptotic")
        delta = 2.0 * test.statistic / (len(fail) * len(wd)) - 1.0
        estimate = hl_shift(fail, wd)
        ci_lo, ci_hi = cluster_boot_hl(data, feature)
        raw_p.append(float(test.pvalue))
        rows.append({
            "Feature": label,
            "Fail median [IQR]": fmt_median_iqr(pd.Series(fail)),
            "Withdrawn median [IQR]": fmt_median_iqr(pd.Series(wd)),
            "HL shift (Fail-Withdrawn)": estimate,
            "Cluster-bootstrap 95% CI low": ci_lo,
            "Cluster-bootstrap 95% CI high": ci_hi,
            "Cliff delta": delta,
            "Mann-Whitney p": float(test.pvalue),
        })
    adjusted = holm_adjust(raw_p)
    for row, p in zip(rows, adjusted):
        row["Holm-adjusted p"] = p

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "endpoint_exploratory_day56.csv", index=False)
    data[["code_module", "code_presentation", "id_student", "final_result"] + [f for _, f in features]].to_parquet(
        OUT / "endpoint_exploratory_day56_source.parquet", index=False
    )
    print(result.to_string(index=False))
    print(data.final_result.value_counts().to_string())


if __name__ == "__main__":
    main()
