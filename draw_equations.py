from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "equations"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8.0})

EQUATIONS = {
    "eq_landmark": [
        r"$R_h=\{i\mid u_i>h\},\qquad y_i=\mathbb{1}\!\left[o_i\in\{\mathrm{Fail},\mathrm{Withdrawn}\}\right]$",
        r"$\mathbf{x}_{i,h}=\phi\!\left(\{e\in E_i\mid 0\leq t_e<h\}\right)$",
    ],
    "eq_temporal": [
        r"$V_{ic}=\sum_{t=0}^{T_h-1}x_{ict},\qquad A_{ic}=\sum_{t=0}^{T_h-1}\mathbb{1}(x_{ict}>0)$",
        r"$S_{ic}=\dfrac{\sum_{t=0}^{T_h-1}(t-\bar t)x_{ict}}{\sum_{t=0}^{T_h-1}(t-\bar t)^2}$",
    ],
    "eq_entropy": [
        r"$Q_{ic}=\dfrac{\sum_{t=\lfloor T_h/2\rfloor}^{T_h-1}x_{ict}}{V_{ic}+1},\qquad p_{ic}=\dfrac{V_{ic}}{\sum_{c'=1}^{6}V_{ic'}+1}$",
        r"$H_i=-\sum_{c=1}^{6}p_{ic}\log\!\left(p_{ic}+10^{-12}\right)$",
    ],
    "eq_capacity": [
        r"$K_{h,\kappa}=\lceil\kappa|R_h|\rceil,\qquad \mathcal{S}_{h,\kappa}=\mathrm{TopK}_{K_{h,\kappa}}\!\left(\{\hat p_i:i\in R_h\}\right)$",
        r"$\mathrm{Precision@K}=\dfrac{\sum_{i\in\mathcal{S}_{h,\kappa}}y_i}{K_{h,\kappa}},\quad \mathrm{Recall@K}=\dfrac{\sum_{i\in\mathcal{S}_{h,\kappa}}y_i}{\sum_{i\in R_h}y_i}$",
    ],
}

SPECS = {
    "eq_landmark": (0.56, 7.1, (0.76, 0.22)),
    "eq_temporal": (0.78, 6.4, (0.80, 0.18)),
    "eq_entropy": (0.78, 6.2, (0.80, 0.18)),
    "eq_capacity": (0.78, 5.9, (0.80, 0.18)),
}

for name, lines in EQUATIONS.items():
    height, fontsize, ys = SPECS[name]
    fig, ax = plt.subplots(figsize=(3.35, height))
    ax.axis("off")
    for line, y in zip(lines, ys):
        ax.text(0.5, y, line, ha="center", va="center", color="black", fontsize=fontsize)
    fig.savefig(OUT / f"{name}.png", dpi=600, transparent=True,
                bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

print(OUT)
