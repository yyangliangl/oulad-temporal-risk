from pathlib import Path
from copy import deepcopy

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
EQ = OUT / "equations"
TEMPLATE = ROOT / "template.docx"
DEST = OUT / "ICETM2026_Log_Derived_IoT_Virtual_Sensing_Submission_Updated.docx"

TITLE = "Log-Derived IoT-Inspired Virtual Sensing for Capacity-Aware Academic Risk Screening in Online Education: Methodological Considerations for STEM Learning Analytics"


def clear_document(doc):
    body = doc._element.body
    sect = body.sectPr
    for child in list(body):
        if child is not sect:
            body.remove(child)


def set_columns(section, n=2, space_twips=360):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(space_twips))


def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)


def body(doc, text, bold_lead=None):
    p = doc.add_paragraph(style="Body Text")
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.0
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_lead and text.startswith(bold_lead):
        r = p.add_run(bold_lead); r.bold = True
        p.add_run(text[len(bold_lead):])
    else:
        p.add_run(text)
    return p


def equation_image(doc, image, alt_text):
    """Insert a compact, high-resolution display equation."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.0
    shape = p.add_run().add_picture(str(image), width=Inches(3.25))
    shape._inline.docPr.set("descr", alt_text)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph(text, style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(4 if level == 1 else 2)
    p.paragraph_format.space_after = Pt(1)
    return p


def caption(doc, text):
    p = doc.add_paragraph(text, style="figure caption")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(2)
    return p


def table_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(7)
    return p


def add_full_width_figure(doc, image, caption_text, alt_text, width=7.0):
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(sec, 1)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(0)
    shape = p.add_run().add_picture(str(image), width=Inches(width))
    shape._inline.docPr.set("descr", alt_text)
    caption(doc, caption_text)
    sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(sec2, 2)


def add_full_width_table(doc, caption_text, headers, rows, widths):
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(sec, 1)
    table_caption(doc, caption_text)
    table = add_table(doc, headers, rows, widths=widths, font_size=6.8)
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
    set_columns(sec2, 2)
    return table


def add_table(doc, headers, rows, widths=None, font_size=7.0):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    # The ICETM template omits Word's built-in "Table Grid" style.
    # Keep the template's default table style and add explicit borders below.
    try:
        table.style = "Table Grid"
    except KeyError:
        pass
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "B7C3CC")
        borders.append(border)
    tbl_pr.append(borders)
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = h
        set_cell_shading(cell, "D9E4EA")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for r in cell.paragraphs[0].runs:
            r.bold = True; r.font.size = Pt(font_size)
        if widths: cell.width = Inches(widths[j])
    set_repeat_table_header(table.rows[0])
    for row in rows:
        cells = table.add_row().cells
        for j, value in enumerate(row):
            cells[j].text = str(value)
            cells[j].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths: cells[j].width = Inches(widths[j])
            for p in cells[j].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs: r.font.size = Pt(font_size)
    return table


doc = Document(TEMPLATE)
clear_document(doc)
sec0 = doc.sections[0]
set_columns(sec0, 1)

# Title and author block.
p = doc.add_paragraph(style="paper title")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run(TITLE)
authors = [
    ("Liang Yang", "yang_li@mail.rmutt.ac.th", False),
    ("Sumeth Theskul*", "sumeth_t@rmutt.ac.th", True),
]
table = doc.add_table(rows=1, cols=2)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
table.autofit = False
table.allow_autofit = False
tblPr = table._tbl.tblPr
borders = OxmlElement("w:tblBorders")
for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
    el = OxmlElement(f"w:{edge}")
    el.set(qn("w:val"), "nil")
    borders.append(el)
tblPr.append(borders)
for cell, (name, email, corresponding) in zip(table.rows[0].cells, authors):
    cell.width = Inches(3.50)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
    p = cell.paragraphs[0]
    p.style = doc.styles["Author"]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(name + "\n")
    r.italic = False
    r = p.add_run("Faculty of Technical Education\nRajamangala University of Technology\nThanyaburi\n")
    r.italic = True
    r = p.add_run("Pathum Thani, Thailand\n" + email)
    r.italic = False
    if corresponding:
        r = p.add_run("\n*Corresponding author")
        r.italic = False

# The abstract remains full width to make the long title and framework readable in this draft.
abstract = (
    "Abstract—Learning-management-system logs are often represented by cumulative activity counts that obscure whether "
    "engagement is persistent, volatile, or declining. This study evaluates a log-derived, Internet-of-Things-inspired "
    "virtual-sensing representation for academic-risk screening in online education. Virtual sensing is used as a conceptual "
    "organization of software event streams; no physical sensors or hardware IoT devices are deployed. Six behavioral "
    "channels were summarized using auditable temporal descriptors at days 28, 56, and 84. Academic risk was defined as a "
    "subsequent Fail or Withdrawn outcome. Experiments used 32,593 Open University Learning Analytics Dataset enrolments and "
    "strict 22-fold leave-one-presentation-out validation, with preprocessing, Gaussian mixture profiling, and prediction "
    "restricted to training presentations. Temporal features produced no reliable improvement at day 28, but increased "
    "logistic-regression AUROC/AUPRC from 0.671/0.575 to 0.721/0.652 at day 56 and reached 0.751/0.693 at day 84. Descriptive "
    "profiles added negligible predictive information, while global probability ranking outperformed profile-balanced "
    "allocation under fixed follow-up capacity. Withdrawal remained harder to predict than Failure, and cross-module "
    "evaluation revealed limited portability. The resulting leakage-safe workflow offers an auditable template for "
    "retrospective capacity-constrained screening; it does not establish intervention effectiveness or STEM-specific "
    "empirical effects."
)
p = doc.add_paragraph(style="Abstract")
p.paragraph_format.space_after = Pt(2)
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.add_run(abstract)
p = doc.add_paragraph(style="Keywords")
p.add_run("Keywords—IoT-inspired virtual sensing, learning analytics, landmark prediction, academic risk, capacity-aware screening")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
shape = p.add_run().add_picture(str(ROOT / "tmp" / "Fig1.png"), width=Inches(6.1))
shape._inline.docPr.set("descr", "Framework connecting six virtual sensor channels to temporal features, fold-isolated prediction and profiling, and capacity-aware decisions.")
caption(doc, "Log-derived, leakage-safe workflow. Landmark risk sets exclude prior withdrawals. No preprocessing, profiling, or model fitting crosses the presentation boundary; each held-out presentation is transformed and scored once. Profiles, probability strata, and allocation policies have distinct roles.")

sec = doc.add_section(WD_SECTION.CONTINUOUS)
set_columns(sec, 2)

heading(doc, "INTRODUCTION")
body(doc, "Learning analytics and educational big data provide institutions with a basis for predictive analytics, dropout prevention, and data-driven student support. Online education produces detailed records of content access, navigation, assessment activity, and peer interaction. These traces create an opportunity to identify students who may benefit from timely support, but operational value depends on making predictions before an adverse outcome occurs and under the resource constraints faced by instructors. Prior studies have shown that learning-management-system (LMS) behavior can predict performance, although predictor relevance and model portability vary substantially across courses [1]–[4].")
body(doc, "A common representation is the cumulative number of clicks or active events observed before a cutoff. Such aggregates are inexpensive, yet identical totals can describe markedly different trajectories: regular participation, short bursts, gradual decline, or late re-engagement. Sequence models, including recurrent and transformer-based architectures, can preserve fine-grained temporal ordering and may capture patterns omitted by summary statistics. Their complexity, data requirements, and limited transparency, however, may exceed what is necessary for lightweight instructor-facing screening. The descriptors evaluated here—volume, persistence, slope, variability, recency, and diversity—represent an auditable middle ground; the present study compares them with cumulative counts but does not claim superiority over deep sequence models.")
body(doc, "The IoT-inspired framing is an organizational heuristic rather than a sensing-hardware contribution. Our earlier work operationalized virtual sensing through inter-activity-time descriptors of behavioral rhythm, intensity, and stability [17]. The present study extends that research direction using a different representation and evaluation objective: functional VLE event streams are organized as six software-derived channels and mapped to auditable measures of volume, persistence, slope, variability, recency, and diversity. This resembles soft sensing from observable process variables [12], while remaining distinct from physical educational sensors [14]. Its value is transparent organization of log evidence, not direct measurement of a latent learning state.")
body(doc, "This study makes three bounded contributions to learning analytics and educational-big-data research for resource-constrained academic-risk screening. First, it evaluates a six-channel temporal representation against cumulative counts at three landmarks and across linear, forest, and gradient-boosting models. Second, it integrates landmark eligibility, intact-presentation separation, fold-isolated preprocessing, profiling, and prediction within leave-one-presentation-out (LOPO) validation. Third, it connects evaluation to constrained screening by separating explanatory profiles from probability strata and comparing allocation policies under fixed follow-up capacity. The contribution is a decision-oriented evaluation architecture rather than a new cross-validation or clustering algorithm.")

heading(doc, "RELATED WORK")
body(doc, "Prior research has established that learning-platform interactions can predict student performance, but it has also exposed substantial variation across courses. Conijn et al. reported between-course differences in LMS predictor utility [2], while Gašević et al. showed that predictive relationships depend on instructional conditions [3]. OULAD studies also differ in outcome definitions, modules, covariates, and validation protocols; recent work, for example, uses a record-level 60/20/20 split with multimodal inputs [16]. Such values are not directly comparable with an intact-presentation holdout. Course-heterogeneity evidence [2], [3] therefore motivates our scheme; we do not claim LOPO as a new cross-validation principle.")
body(doc, "Clustering has been used to characterize engagement patterns and learner heterogeneity [5], [6], [13]. Such profiles can support interpretation, but cluster membership need not contain information beyond its input variables and should not be treated automatically as a stable learner identity. Operational use raises a separate decision problem: conventional discrimination metrics do not specify how scarce instructor follow-up should be allocated. We therefore distinguish descriptive profiles, probability strata, and allocation policies, and evaluate targeting under explicit capacity constraints.")
body(doc, "Deployment requires more than discrimination. Large-scale implementations depend on teacher adoption [8], and recent dashboard research highlights interpretability, privacy, and workload constraints [15]. Our decision layer reports retrospective targeting trade-offs rather than intervention effects, excludes demographic attributes from modeling, and treats predictions as prompts for contextual human review [9].")

heading(doc, "METHODS")
heading(doc, "Data, Outcomes, and Landmark Risk Sets", 2)
body(doc, "We used the Open University Learning Analytics Dataset (OULAD), which contains 32,593 enrolments, 28,785 unique students, 22 module presentations, and 10,655,280 daily VLE interaction records [1]. The released module identifiers are anonymized. Pass and Distinction were coded as non-risk; Fail and Withdrawn were coded as academic risk. The unit of analysis was a student–module–presentation enrolment, not an individual click.")
body(doc, "Predictions were made at days 28, 56, and 84 after presentation start. For each enrolment, the notation below denotes unregistration day u (set to infinity if no date was recorded), final outcome o, dated event history E, and the day t of each event e. At landmark h, eligibility, outcome, and temporal availability were")
equation_image(doc, EQ / "eq_landmark.png", "Landmark risk set, binary academic-risk outcome, and pre-landmark feature vector definitions.")
body(doc, "Thus, the risk set contains only enrolments still registered immediately before the cutoff, and the feature vector contains no event on or after the landmark. Risk-set size decreased from 27,538 to 26,522 and 25,724 as already unregistered students were progressively removed rather than retained as predictable cases.")
body(doc, "The landmarks represent different evidence–actionability conditions. Day 28 provides the greatest potential follow-up time but only four weeks of behavioral history; day 56 provides a longer trajectory while retaining a substantial portion of the presentation; and day 84 is treated as a later-term checkpoint rather than an unqualified early-warning point. Comparisons therefore quantify both information accumulation and reduced response time.")

heading(doc, "Six-Channel Virtual Sensing", 2)
body(doc, "VLE activity types were mapped a priori to six functional channels: content (resource exposure), navigation (course-structure access), assessment (formative-task engagement), communication (help seeking and social participation), collaboration (joint activity), and enrichment (supplementary tools). This behaviorally interpretable mapping is informed by engagement and self-regulated-learning research [6], but the channels are proxies rather than direct measurements of latent SRL constructs. Clicks were aggregated into non-overlapping weeks, and the mapping was fixed before model fitting.")
body(doc, "Unlike the IAT-based contextual proxies evaluated in our earlier study [17], the present representation summarizes weekly activity separately within six predefined functional channels and evaluates it under landmark-specific, intact-presentation holdout. The prior and present feature sets therefore address related temporal-representation questions but are not reused as equivalent predictors.")
body(doc, "Weekly click counts were indexed by enrolment i, channel c, and week t over the landmark-specific observation window. Cumulative volume, active weeks, and least-squares slope were")
equation_image(doc, EQ / "eq_temporal.png", "Cumulative volume, active weeks, and least-squares activity-slope definitions.")
body(doc, "Weekly standard deviation and last-week activity captured volatility and recency. Recent-half share and cross-channel entropy were computed as")
equation_image(doc, EQ / "eq_entropy.png", "Recent-half activity share, normalized channel share, and cross-channel entropy definitions.")
body(doc, "The +1 denominators and small logarithmic stabilizer match the implementation. Together these descriptors capture volume, persistence, direction, volatility, recency, and behavioral diversity. No demographic feature or assessment score was used.")

heading(doc, "Prediction and Leakage Control", 2)
body(doc, "Four constraints governed leakage control. First, temporal availability restricted features to days [0,h) at landmark h. Second, risk-set eligibility excluded enrolments withdrawn by h. Third, presentation separation placed every enrolment from one complete module presentation on only one side of each of 22 LOPO folds. Fourth, fit isolation restricted scaling, Gaussian mixture model (GMM) fitting, profile ordering, and classifier fitting to training presentations; the held-out presentation was transformed and scored once. Logistic regression used inverse-frequency class weights and C=1, while tree-model hyperparameters were fixed without test-fold tuning.")
body(doc, "We report AUROC, AUPRC, F1, balanced accuracy, and Brier score, with AUPRC as the principal ranking metric [10]. Temporal-minus-cumulative differences used 1,000 paired cluster-bootstrap resamples of presentations. One-sided paired Wilcoxon tests compared fold-level AUROC and AUPRC at each landmark; Holm correction controlled the six within-metric horizon comparisons. Leave-one-module-out validation across seven modules tested transfer, and one-versus-rest Kolmogorov–Smirnov (KS) statistics described feature shift.")

heading(doc, "Endpoint-Specific Exploratory Analysis", 2)
body(doc, "To examine why Withdrawal was less predictable than Failure, we conducted an endpoint-specific descriptive analysis among enrolments remaining in the day-56 risk set (n=26,522). Subsequent Failure (n=7,044) and Withdrawal (n=4,093) cases were compared using only behavior observed during days 0–56. Prespecified dimensions were cumulative activity, active weeks, recent-activity share, activity slope, weekly variability, assessment-channel activity, and communication-channel activity. Skewed distributions were summarized by medians and interquartile ranges. Between-endpoint differences were expressed as two-sample Hodges–Lehmann shifts (Fail minus Withdrawn), with 95% intervals from 500 bootstrap resamples of the 22 presentations, and Cliff's delta. Two-sided Mann–Whitney tests were exploratory and Holm-corrected across the seven dimensions. Effect sizes and intervals were prioritized over null-hypothesis tests; the analysis does not identify causal mechanisms of withdrawal.")

heading(doc, "Longitudinal Profiles and Capacity-Aware Stratification", 2)
body(doc, "Within each LOPO fold, standardized channel totals, slopes, recent-activity shares, and entropy entered a diagonal-covariance GMM [11]. A fixed 5,000-enrolment sensitivity sample compared k=2–6 without tuning held-out predictions. BIC decreased through k=6, but descriptive separation deteriorated: relative to k=2 (silhouette=0.483; minimum share=10.2%), k=3 had silhouette=0.126 and a 3.2% minority component, whereas k=4–6 had silhouettes of 0.031–0.054 and minimum shares below 1%. The k=5 solution was initialization-sensitive (two-seed adjusted Rand index [ARI]=0.393). For k=2, ARI=1.000 means that two random initializations yielded identical assignments on this fixed sample, not perfect external validity. We retained k=2 as a coarse descriptive partition rather than evidence of two natural learner types.")
body(doc, "Training-fold components were ordered by risk rate before assignment to the held-out presentation. Profiles were recomputed at days 28, 56, and 84, and transitions were evaluated only for enrolments present at both landmarks.")
body(doc, "Terminology was kept role-specific: behavioral profiles are descriptive GMM clusters; probability strata are Low, Moderate, and High operational categories derived from out-of-fold 56-day probabilities; and allocation policies determine selection. For capacity fraction κ and out-of-fold risk scores, the global policy used")
equation_image(doc, EQ / "eq_capacity.png", "Capacity-constrained global ranking, Precision at K, and Recall at K definitions.")
body(doc, "A profile-balanced comparator allocated equal quotas to the two profiles and filled unused quota by global rank. These metrics quantify retrospective targeting, not intervention benefit.")

p = heading(doc, "RESULTS")
p.paragraph_format.page_break_before = True
add_full_width_table(
    doc,
    "TABLE I. Landmark LOPO performance. Metrics are pooled out-of-fold estimates; p values are one-sided paired Wilcoxon tests across 22 held-out presentations with Holm correction within each metric.",
    ["Day", "Representation", "n", "AUROC", "AUPRC", "Brier", "Holm p (ROC / PR)"],
    [
        ["28", "Cumulative", "27,538", "0.649", "0.569", "0.237", "—"],
        ["28", "Temporal virtual sensing", "27,538", "0.670", "0.607", "0.228", "0.111 / 0.869"],
        ["56", "Cumulative", "26,522", "0.671", "0.575", "0.233", "—"],
        ["56", "Temporal virtual sensing", "26,522", "0.721", "0.652", "0.213", "6.68×10⁻⁶ / 7.15×10⁻⁷"],
        ["84", "Cumulative", "25,724", "0.689", "0.580", "0.229", "—"],
        ["84", "Temporal virtual sensing", "25,724", "0.751", "0.693", "0.201", "3.58×10⁻⁶ / 9.54×10⁻⁷"],
    ],
    [0.42, 1.85, 0.72, 0.72, 0.72, 0.65, 1.65],
)
heading(doc, "Temporal Features Improved Landmark Prediction", 2)
body(doc, "Pooled out-of-fold discrimination favored temporal virtual sensing (Fig. 2). At day 28, AUROC/AUPRC changed from 0.649/0.569 to 0.670/0.607, but presentation-level gains were not reliable after Holm correction (p=0.111 and 0.869). Thus, four weeks of history did not provide a stable cross-presentation advantage for slope, variability, and recency summaries. At day 56, AUROC increased from 0.671 to 0.721 and AUPRC from 0.575 to 0.652; cluster-bootstrap gains were 0.0486 (95% CI, 0.0371–0.0585) and 0.0738 (0.0593–0.0896). At day 84, AUROC/AUPRC reached 0.751/0.693 (Holm-adjusted p<10⁻⁵ for both), with less follow-up time remaining.")
body(doc, "At day 56, the same feature ordering occurred for random forest (0.684/0.607 versus 0.734/0.683) and histogram gradient boosting (0.685/0.611 versus 0.732/0.681). Thus, the feature advantage was not confined to a linear classifier. Absolute performance remained moderate under the strict risk-set definition and exclusion of assessment scores and demographic variables.")

add_full_width_figure(doc, FIG / "Fig1_predictive_performance.tiff", "Presentation-level predictive evidence. (a,b) Mean AUROC and AUPRC across 22 held-out presentations; bars show 95% bootstrap CIs over presentations. (c) Paired 56-day fold differences; diamonds and bars show the mean and 95% CI. Landmark n=27,538/26,522/25,724. Holm-adjusted tests are reported in Table I.", "Line charts show presentation-level AUROC and AUPRC with uncertainty across three landmarks; a paired-difference panel shows 56-day temporal-minus-cumulative changes.")

heading(doc, "Trends Were a Distinct but Partial Contributor", 2)
body(doc, "Ablation at 56 days showed that no single temporal family explained the entire gain. Removing slopes and recent-activity shares reduced AUROC/AUPRC from 0.721/0.652 to 0.716/0.641. Removing weekly variability yielded 0.717/0.647. Both remained above the cumulative-only result of 0.671/0.575, indicating that persistence, recency, variability, and channel diversity contributed complementary information rather than one engineered statistic dominating the model.")

heading(doc, "Longitudinal Profiles Explained Heterogeneity but Did Not Improve Prediction", 2)
body(doc, "The retained day-56 profiles contained 4,678 and 21,844 enrolments with risk rates of 0.279 and 0.450 (Fig. 3a). Between days 56 and 84, 27.1% of the lower-risk profile moved to the higher-risk profile and 23.5% moved in the opposite direction, supporting a time-varying rather than fixed typology.")
body(doc, "AUROC remained 0.721 with and without leakage-safe profile indicators, while AUPRC remained 0.652. Profiles therefore supplied context but no independent predictive gain. Probability strata gave a sharper risk gradient (0.264, 0.481, and 0.718). Profile-balanced allocation reduced Precision@K by 12.5–19.7 percentage points across capacities of 5%–30%, with a 15.7-point loss at 10% capacity. These retrospective targeting differences do not estimate intervention benefit.")

add_full_width_figure(doc, FIG / "Fig2_profiles_and_stratification.tiff", "Descriptive profiles and operational allocation at day 56 (n=26,522). (a) Risk across leakage-safe profiles. (b) Risk across out-of-fold probability strata. (c) Precision@K under global and profile-balanced allocation. Error bars are 95% presentation-cluster bootstrap CIs. Values quantify retrospective targeting, not intervention effects.", "Risk bars with presentation-cluster uncertainty distinguish descriptive profiles from operational probability strata; a line chart compares targeting precision under two allocation policies.")

heading(doc, "Endpoint and Transfer Audits Exposed Boundaries", 2)
body(doc, "Separate temporal models performed better for Failure (AUROC/AUPRC 0.708/0.473) than for Withdrawal (0.607/0.207); cumulative counterparts achieved 0.675/0.428 and 0.557/0.170. Exploratory day-56 comparisons (Table II) found that subsequent Failure generally involved less activity than subsequent Withdrawal: cumulative activity was 166 [IQR 50–369] versus 243 [101–490], with a Hodges–Lehmann shift of −60 clicks (presentation-bootstrap 95% CI, −83 to −39; Cliff's δ=−0.173). Active weeks, variability, assessment activity, and communication activity showed similarly small standardized differences (|δ|=0.134–0.173), while recent-activity share and slope showed weaker separation (|δ|=0.067–0.090). Assessment activity had a cluster-bootstrap shift interval that included zero despite a small record-level Mann–Whitney p value, illustrating presentation heterogeneity. No observed dimension showed a large effect.")

add_full_width_table(
    doc,
    "TABLE II. Endpoint-specific day-56 behavior among subsequent Failure (n=7,044) and Withdrawal (n=4,093). Values are median [IQR]. Hodges–Lehmann shifts are Fail minus Withdrawn; 95% CIs resample 22 presentations. Mann–Whitney p values are Holm-adjusted exploratory comparisons.",
    ["Feature", "Failure", "Withdrawal", "Hodges–Lehmann shift [95% CI]", "Cliff δ", "Holm p"],
    [
        ["Cumulative activity", "166 [50, 369]", "243 [101, 490]", "−60 [−83, −39]", "−0.173", "<0.001"],
        ["Active weeks", "6 [3, 7]", "7 [4, 8]", "−1 [−1, 0]", "−0.173", "<0.001"],
        ["Recent-activity share", "0.30 [0.06, 0.48]", "0.33 [0.16, 0.48]", "−0.019 [−0.036, −0.002]", "−0.067", "<0.001"],
        ["Activity slope", "−1.52 [−5.80, 0]", "−2.56 [−8.00, 0]", "0.762 [0.387, 1.071]", "0.090", "<0.001"],
        ["Weekly variability", "23.42 [9.05, 45.52]", "30.86 [14.92, 54.64]", "−6.35 [−8.91, −3.76]", "−0.152", "<0.001"],
        ["Assessment activity", "3 [0, 41]", "11 [1, 81]", "−1 [−4, 0]", "−0.134", "<0.001"],
        ["Communication activity", "21 [0, 80]", "35 [6, 111]", "−6 [−14, −2]", "−0.138", "<0.001"],
    ],
    [1.36, 1.18, 1.18, 1.70, 0.60, 0.62],
)
body(doc, "The uniformly small effects indicate that weaker Withdrawal prediction cannot be attributed to one dominant VLE dimension. Combining endpoints therefore supports broad screening but does not imply a shared behavioral mechanism.")
body(doc, "Leave-one-module-out evaluation reduced temporal logistic-regression AUROC from 0.721 under LOPO to 0.696 and AUPRC from 0.652 to 0.610. Assessment-channel features showed the largest one-versus-rest shifts (mean KS=0.567 for weekly variability and 0.560 for total activity), followed by collaboration and content variability. The decline is therefore consistent with module-specific assessment and activity design rather than only classifier instability [2], [3]. Thresholds and profiles require local monitoring before transfer.")

heading(doc, "DISCUSSION")
body(doc, "The day-28 result admits two competing explanations. Four weekly observations may be insufficient for stable slope and variability estimates; alternatively, accurately measured early behavior may be dominated by orientation, resource exploration, or assessment onboarding rather than outcome-specific trajectories. The analysis cannot distinguish estimation instability from adaptation-period noise because it did not model onboarding events or alternative early-window resolutions. Day 28 is therefore a boundary of this representation, not evidence that early temporal behavior is generally uninformative.")
body(doc, "The endpoint analysis likewise narrows but does not explain weaker Withdrawal prediction. Under a measurement-insufficiency hypothesis, relevant information lies mainly outside VLE traces; under an outcome-heterogeneity hypothesis, the Withdrawn label combines behaviorally distinct pathways. Without withdrawal-reason labels or off-platform covariates, the data cannot adjudicate between these hypotheses. Economic, occupational, health, or administrative circumstances are examples of unobserved information, not established causes in this study.")
body(doc, "The day-84 result is not a recommendation to delay screening. It provides a later-term reference under the same leakage-safe protocol and shows that predictive information continues to accumulate, motivating sequential score updates. Day 28 could support low-intensity initial screening, day 56 a principal review point, and day 84 confirmation or escalation. This three-landmark framing is a design orientation for future dashboards, not a tested intervention strategy; the present data neither establish optimal escalation thresholds nor show that review at any landmark changes outcomes.")
body(doc, "The log-derived virtual-sensing representation organizes heterogeneous platform events into auditable signal streams; it neither introduces physical sensing nor treats clicks as direct learning-state measurements. Profiles similarly provide context rather than learner identities. The decision layer converts scores into a review list under explicit capacity, but the dashboard and sequential-review workflow remain design hypotheses rather than implemented or evaluated systems.")
body(doc, "The framework offers methodological considerations for STEM-focused learning analytics, where cumulative counts may obscure sustained problem solving, assessment-driven spikes, or gradual disengagement. However, OULAD provides no authoritative disciplinary labels. The results are not STEM-specific empirical findings, and explicitly identified STEM courses are required before disciplinary generalization.")
body(doc, "Several limitations bound the claims. OULAD comes from one institution in 2013–2014; changes in platforms and online-learning practice may limit temporal transfer. Omitting text, marks, demographics, and off-platform circumstances supports data minimization but limits discrimination, particularly for Withdrawal. Excluding demographic attributes does not establish fairness because behavioral traces may encode proxy differences; subgroup calibration and error disparities were not evaluated. The k=2 GMM is a coarse partition, not a natural learner typology, and discrete snapshots are not a joint dynamic-state model. Deep sequence models were not evaluated. No operational dashboard or instructor study assessed usability, workload, adoption, or outcomes. All allocation results remain retrospective and do not establish intervention effectiveness.")

heading(doc, "CONCLUSION")
body(doc, "This study evaluated a log-derived IoT-inspired representation within a leakage-safe, decision-oriented screening workflow. Temporal summaries provided no reliable cross-presentation advantage at day 28 but improved discrimination at days 56 and 84. Profiles described heterogeneity without improving prediction, while profile-balanced allocation reduced retrospective targeting precision. Endpoint and transfer audits delimited portability. The workflow offers methodological considerations for future STEM learning analytics, but explicitly identified STEM data and prospective evaluation are required before disciplinary or intervention claims.")

heading(doc, "DATA AND CODE AVAILABILITY")
body(doc, "OULAD is available under CC BY 4.0 from the Open University [1]. Analysis scripts and derived aggregate outputs are available from the corresponding author upon reasonable request. No attempt was made to re-identify students.")

heading(doc, "REFERENCES", 5)
refs = [
    "[1] J. Kuzilek, M. Hlosta, and Z. Zdrahal, “Open University Learning Analytics dataset,” Scientific Data, vol. 4, Art. no. 170171, 2017, doi: 10.1038/sdata.2017.171.",
    "[2] R. Conijn, C. Snijders, A. Kleingeld, and U. Matzat, “Predicting student performance from LMS data: A comparison of 17 blended courses using Moodle LMS,” IEEE Trans. Learn. Technol., vol. 10, no. 1, pp. 17–29, 2017, doi: 10.1109/TLT.2016.2616312.",
    "[3] D. Gašević, S. Dawson, T. Rogers, and D. Gasevic, “Learning analytics should not promote one size fits all: The effects of instructional conditions in predicting academic success,” Internet Higher Educ., vol. 28, pp. 68–84, 2016, doi: 10.1016/j.iheduc.2015.10.002.",
    "[4] M. Hlosta, Z. Zdrahal, and J. Zendulka, “Ouroboros: Early identification of at-risk students without models based on legacy data,” in Proc. 7th Int. Learn. Analytics Knowl. Conf., 2017, pp. 6–15, doi: 10.1145/3027385.3027449.",
    "[5] J. Pecuchova and M. Drlik, “Enhancing the early student dropout prediction model through clustering analysis of students’ digital traces,” IEEE Access, vol. 12, pp. 159336–159367, 2024, doi: 10.1109/ACCESS.2024.3486762.",
    "[6] Ü. Çakiroğlu, M. Kokoç, and M. Atabay, “Online learners’ self-regulated learning skills regarding LMS interactions: A profiling study,” J. Comput. Higher Educ., vol. 36, no. 1, pp. 220–241, 2024, doi: 10.1007/s12528-024-09397-2.",
    "[7] B. T.-m. Wong, K. C. Li, and S. K. S. Cheung, “An analysis of learning analytics in personalised learning,” J. Comput. Higher Educ., vol. 35, pp. 371–390, 2023, doi: 10.1007/s12528-022-09324-3.",
    "[8] C. Herodotou et al., “The scalable implementation of predictive learning analytics at a distance learning university: Insights from a longitudinal case study,” Internet Higher Educ., vol. 45, Art. no. 100725, 2020, doi: 10.1016/j.iheduc.2020.100725.",
    "[9] S. Slade and P. Prinsloo, “Learning analytics: Ethical issues and dilemmas,” Am. Behav. Sci., vol. 57, no. 10, pp. 1510–1529, 2013, doi: 10.1177/0002764213479366.",
    "[10] T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” PLoS ONE, vol. 10, no. 3, Art. no. e0118432, 2015, doi: 10.1371/journal.pone.0118432.",
    "[11] A. P. Dempster, N. M. Laird, and D. B. Rubin, “Maximum likelihood from incomplete data via the EM algorithm,” J. R. Stat. Soc. Ser. B, vol. 39, no. 1, pp. 1–38, 1977.",
    "[12] P. Kadlec, B. Gabrys, and S. Strandt, “Data-driven soft sensors in the process industry,” Comput. Chem. Eng., vol. 33, no. 4, pp. 795–814, 2009, doi: 10.1016/j.compchemeng.2008.12.012.",
    "[13] D. H. Setiabudi and M. Santoso, “Effect of students’ activities on academic performance using clustering evolution analysis,” CommIT J., vol. 17, no. 2, pp. 209–219, 2023.",
    "[14] X. Zhang, Y. Ding, X. Huang, W. Li, L. Long, and S. Ding, “Smart classrooms: How sensors and AI are shaping educational paradigms,” Sensors, vol. 24, no. 17, Art. no. 5487, 2024, doi: 10.3390/s24175487.",
    "[15] I. Possaghi et al., “Integrating multi-modal learning analytics dashboard in K-12 education: Insights for enhancing orchestration and teacher decision-making,” Smart Learn. Environ., vol. 12, Art. no. 53, 2025, doi: 10.1186/s40561-025-00410-4.",
    "[16] U. Tran Van, B. H. Tieu, and D. H. Tran, “A multimodal learning analytics model based on deep learning for predicting student performance using tabular and time-series data fusion,” Discover Artif. Intell., vol. 6, Art. no. 232, 2026, doi: 10.1007/s44163-026-00990-1.",
    "[17] L. Yang, S. Theskul, and K. Juikumjorn, “Integrating IoT-inspired contextual features with learning analytics for technology-enhanced STEM teaching support,” in Proc. 8th Int. Conf. Computer Science and Technologies in Education (CSTE), Wuhan, China, 2026, pp. 1026–1031, doi: 10.1109/CSTE69562.2026.11649792.",
]
for ref in refs:
    p = doc.add_paragraph(ref.split("] ", 1)[1], style="references")
    p.paragraph_format.space_after = Pt(0)

# Normalize fonts without overriding the template's intended hierarchy.
for style_name in ["Body Text", "Abstract", "Keywords", "references", "figure caption"]:
    if style_name in doc.styles:
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")

doc.core_properties.title = TITLE
doc.core_properties.subject = "ICETM 2026 conference manuscript"
doc.core_properties.keywords = "IoT-inspired virtual sensing; learning analytics; landmark prediction; academic risk; capacity-aware screening"
doc.core_properties.author = "Liang Yang; Sumeth Theskul"
doc.save(DEST)
print(DEST)
