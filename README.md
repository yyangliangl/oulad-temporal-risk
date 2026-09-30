# oulad-temporal-risk

Analysis code and derived aggregate outputs for the ICETM 2026 paper:

**"Log-Derived IoT-Inspired Temporal Representation for Capacity-Aware STEM Risk Screening in Online Education"**

## Code

- `run_oulad_experiments.py` — main LOPO experiments, profiles, strata, capacity policies
- `run_channel_dropout.py` — channel-outage audit
- `endpoint_exploratory_analysis.py` — endpoint-specific bootstrap analysis
- `draw_framework.py`, `draw_framework_v2.py` — Figure 1
- `draw_equations.py` — display equations
- `redraw_evidence_figures.py`, `redraw_fig2.py` — Figures 2 and 3
- `make_fig4.py` — Figure 4 (calibration reliability curve)
- `build_manuscript.py` — manuscript generation
- `extract_reference_metadata.py`, `render_pdf_pages.py` — auxiliary utilities

## Derived outputs

Aggregate CSV outputs for model comparison, ablation, paired inference, endpoint analysis, module transfer, feature shift, profiles, strata, capacity policies, channel dropout, and calibration. Two files (`oof_predictions.csv`, `profile_landmarks.csv`) provide per-enrolment predictions for full reproducibility of the profile and capacity analyses.

## Data

This repository does not redistribute the OULAD dataset. Download it from the Open University:

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. Scientific Data, 4, 170171. https://doi.org/10.1038/sdata.2017.171

## License

MIT License.