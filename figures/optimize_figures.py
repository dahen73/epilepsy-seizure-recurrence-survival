# -*- coding: utf-8 -*-
"""Publication optimisation of the two IDEAS survival-analysis figures.

Figure contracts
----------------
Figure 1 (quantitative grid): recurrence is front-loaded; pathology, but not
site, stratifies outcome. Panel a is the hero estimate; b/c are stratification.
Figure 2 (quantitative grid): imaging aggregates and resection extent add no
detected incremental value. Panel a is decisive internal validation; b shows
coefficient uncertainty; c bounds the time interaction; d tests robustness.

Structural adaptation of the supplied Python workflow. No observations are
excluded beyond the original analytic and complete-case definitions.
"""
from pathlib import Path
import json
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("IDEAS_DATA_DIR", REPO_ROOT / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("IDEAS_OUTPUT_DIR", REPO_ROOT / "outputs" / "longitudinal")).resolve()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import NullFormatter, NullLocator
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
# Editable text: svg.fonttype='none'; pdf.fonttype=42.
plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 5.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "axes.grid": False,
    "savefig.facecolor": "white",
})

WIDTH_MM = 183

DATA = DATA_ROOT
TAB = RESULTS_ROOT
OUT = REPO_ROOT / "outputs" / "figures"
QA = OUT / "qa"
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)

BLUE = "#0F4D92"
TEAL = "#2A8C94"
ORANGE = "#D97706"
RED = "#B64342"
GREY = "#737373"
LIGHT_GREY = "#D4D4D4"
DARK = "#272727"
PATH_COLORS = {
    "HS": "#0072B2",
    "DNT": "#E69F00",
    "CAV": "#009E73",
    "FCD": "#CC79A7",
    "DUAL": "#56B4E9",
    "Other": "#D55E00",
}
SITE_COLORS = {"颞叶": "#0072B2", "额叶": "#E69F00", "其他脑叶外": "#009E73"}


def panel_label(ax, label):
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction",
                xytext=(-19, 11), textcoords="offset points",
                ha="left", va="bottom", fontsize=9, fontweight="bold",
                color=DARK, annotation_clip=False)


def clean_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color="#E7E7E7", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, pad=2)


def export_figure(fig, stem):
    """Alignment is a blocking gate; preserve exact 183-mm page width."""
    fig.canvas.draw()
    exemptions = []
    if stem == "Figure_2":
        exemptions = [{
            "panels": ["a_delta_aic"],
            "checks": ["row"],
            "reason": "Secondary plot within composite panel a; the composite carries one shared label.",
        }]
    if require_matplotlib_panel_alignment is not None:
        require_matplotlib_panel_alignment(
            fig,
            json_out=str(QA / f"{stem}.alignment.json"),
            overlay_svg=str(QA / f"{stem}.alignment.svg"),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
            exemptions=exemptions,
        )
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    fig.savefig(OUT / f"{stem}.tiff", dpi=600)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Source follow-up: one row per patient, first recurrence, terminal censoring.
# ---------------------------------------------------------------------------
fu = pd.read_csv(TAB / "person_period_long.csv")
assert len(fu) == 1066 and fu["ID"].nunique() == 427 and int(fu["event"].sum()) == 252
pat = fu.groupby("ID").agg(duration=("stop", "max"), event=("event", "max")).reset_index()
pat["event"] = pat["event"].astype(int)
baseline = fu.drop_duplicates("ID").set_index("ID")
pat["PathoGrp"] = pat["ID"].map(baseline["PathoGrp"])
pat["Lobe"] = pat["ID"].map(baseline["Lobe"])

# Figure 1: 183 mm × 150 mm, overall hero above two aligned stratified panels.
fig = plt.figure(figsize=(WIDTH_MM / 25.4, 150 / 25.4), layout="constrained")
gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0])
a = fig.add_subplot(gs[0, :])
b = fig.add_subplot(gs[1, 0])
c = fig.add_subplot(gs[1, 1])
for ax, label in [(a, "a"), (b, "b"), (c, "c")]:
    ax.set_gid(label)
    panel_label(ax, label)

kmf = KaplanMeierFitter()
kmf.fit(pat["duration"], pat["event"])
kmf.plot_survival_function(ax=a, ci_show=True, color=BLUE, linewidth=1.6,
                           ci_alpha=0.18)
a.get_legend().remove()  # The in-panel annotation already identifies the estimate and CI.
a.set_xlim(0, 5)
a.set_xticks(range(6))
a.set_ylim(0, 1.02)
a.set_xlabel("Years after surgery")
a.set_ylabel("Seizure-free probability (ILAE class 1)")
a.set_title("Overall seizure freedom (n = 427)", pad=4)
a.text(0.985, 0.94, "Kaplan–Meier estimate and 95% CI", transform=a.transAxes,
       ha="right", va="top", fontsize=6.2, color=GREY)
clean_axis(a)

path_order = ["HS", "DNT", "CAV", "FCD", "DUAL", "Other"]
path_names = {"HS": "Hippocampal sclerosis", "DNT": "DNT", "CAV": "Cavernous angioma",
              "FCD": "FCD", "DUAL": "Dual pathology", "Other": "Other pathology"}
for name in path_order:
    sub = pat.loc[pat["PathoGrp"] == name]
    km = KaplanMeierFitter()
    km.fit(sub["duration"], sub["event"])
    km.plot_survival_function(ax=b, ci_show=False, color=PATH_COLORS[name],
                              linewidth=1.25,
                              label=f"{path_names[name]} (n={len(sub)})")
b.set_xlim(0, 5)
b.set_xticks(range(6))
b.set_ylim(0, 1.02)
b.set_xlabel("Years after surgery")
b.set_ylabel("Seizure-free probability")
b.set_title("By histopathology", pad=4)
lr_path = multivariate_logrank_test(pat["duration"], pat["PathoGrp"], pat["event"])
assert abs(lr_path.p_value - 0.0092) < 0.0001
b.text(0.97, 0.96, f"Log-rank p = {lr_path.p_value:.4f}", transform=b.transAxes,
       ha="right", va="top", fontsize=6.2)
b.legend(loc="lower left", ncol=2, columnspacing=0.9, handlelength=1.6,
         borderaxespad=0.15, labelspacing=0.35)
clean_axis(b)

site_order = ["颞叶", "额叶", "其他脑叶外"]
site_names = {"颞叶": "Temporal", "额叶": "Frontal", "其他脑叶外": "Extratemporal-other"}
for name in site_order:
    sub = pat.loc[pat["Lobe"] == name]
    km = KaplanMeierFitter()
    km.fit(sub["duration"], sub["event"])
    km.plot_survival_function(ax=c, ci_show=False, color=SITE_COLORS[name],
                              linewidth=1.25,
                              label=f"{site_names[name]} (n={len(sub)})")
c.set_xlim(0, 5)
c.set_xticks(range(6))
c.set_ylim(0, 1.02)
c.set_xlabel("Years after surgery")
c.set_title("By surgical site", pad=4)
lr_site = multivariate_logrank_test(pat["duration"], pat["Lobe"], pat["event"])
c.text(0.97, 0.96, f"Log-rank p = {lr_site.p_value:.2f}", transform=c.transAxes,
       ha="right", va="top", fontsize=6.2)
c.legend(loc="lower left", handlelength=1.6, borderaxespad=0.15, labelspacing=0.4)
clean_axis(c)
export_figure(fig, "Figure_1")

# ---------------------------------------------------------------------------
# Refit only the prespecified M3 to obtain its coefficient covariance matrix.
# ---------------------------------------------------------------------------
TEMPORAL = ["bankssts", "entorhinal", "fusiform", "inferiortemporal",
            "middletemporal", "parahippocampal", "superiortemporal",
            "temporalpole", "transversetemporal"]
tz = pd.read_csv(DATA / "zscores" / "thick_zscores_patients.csv")
vz = pd.read_csv(DATA / "zscores" / "vol_zscores_patients.csv")
cli = pd.read_csv(DATA / "tables_metadata" / "Metadata_Release_Anon.csv", dtype={"ID": int}).drop_duplicates("ID")
side = cli.set_index("ID")["Op_Side"].astype(str).str.upper().str.strip().str[0]
img = tz[["ID"]].copy()
thick_cols = [col for col in tz.columns if col.endswith("_thickness")]
img["ThickZ_global"] = tz[thick_cols].mean(axis=1)
img["side"] = img["ID"].map(side)
img = img[img["side"].isin(["L", "R"])].copy()
img["h_ipse"] = np.where(img["side"] == "L", "lh", "rh")
img["h_contra"] = np.where(img["side"] == "L", "rh", "lh")

def temporal_mean(ids, hemispheres):
    values = []
    for patient_id, hemisphere in zip(ids, hemispheres):
        cols = [f"{hemisphere}_{region}_thickness" for region in TEMPORAL]
        values.append(float(tz.loc[tz["ID"] == patient_id, cols].mean(axis=1).iloc[0]))
    return values

img["ThickZ_ipseTL"] = temporal_mean(img["ID"], img["h_ipse"])
img["ThickZ_contraTL"] = temporal_mean(img["ID"], img["h_contra"])
hip_col = {"L": "Left_Hippocampus_volume", "R": "Right_Hippocampus_volume"}
img["HippZ_ipse"] = [vz.loc[vz["ID"] == i, hip_col[s]].iloc[0] for i, s in zip(img["ID"], img["side"])]
res_tab = pd.read_csv(DATA / "resected_table" / "table_resected.csv", index_col=0)
res_tab.columns = [int(float(col)) for col in res_tab.columns]
res_tab.index = [str(x).strip().strip("'").strip('"').lstrip("\ufeff") for x in res_tab.index]

def resect_pct(patient_id, regions):
    if patient_id not in res_tab.columns:
        return np.nan
    row = res_tab.loc[res_tab.index.isin(regions), patient_id]
    return float(pd.to_numeric(row, errors="coerce").mean())

img["Resect_ipseTL"] = [resect_pct(i, [f"ctx-{h}-{r}" for r in TEMPORAL] +
                                  [("Left-Hippocampus" if s == "L" else "Right-Hippocampus"),
                                   ("Left-Amygdala" if s == "L" else "Right-Amygdala")])
                       for i, s, h in zip(img["ID"], img["side"], img["h_ipse"])]
img["Resect_ipseHipp"] = [resect_pct(i, ["Left-Hippocampus" if s == "L" else "Right-Hippocampus"])
                         for i, s in zip(img["ID"], img["side"])]
img_vars = ["ThickZ_global", "ThickZ_ipseTL", "ThickZ_contraTL", "HippZ_ipse",
            "Resect_ipseTL", "Resect_ipseHipp"]
merged = fu.merge(img[["ID"] + img_vars], on="ID", how="inner")
print(f"M3 before complete-case filter: {len(merged)} intervals, {merged['ID'].nunique()} patients")
m = merged.dropna(subset=["OnsetMid", "SurgAgeMid", "ASM"] + img_vars).copy()
print(f"M3 after complete-case filter: {len(m)} intervals, {m['ID'].nunique()} patients")
assert m["ID"].nunique() == 406 and len(m) == 1009 and int(m["event"].sum()) == 238
m["OnsetMid_z"] = (m["OnsetMid"] - m["OnsetMid"].mean()) / m["OnsetMid"].std()
m["SurgAgeMid_z"] = (m["SurgAgeMid"] - m["SurgAgeMid"].mean()) / m["SurgAgeMid"].std()
m["ASM_z"] = (m["ASM"] - m["ASM"].mean()) / m["ASM"].std()
m["logt"] = np.log(m["stop"])
m["Resect_ipseTL_s"] = (m["Resect_ipseTL"] - m["Resect_ipseTL"].mean()) / m["Resect_ipseTL"].std()
m["Resect_x_logt"] = m["Resect_ipseTL_s"] * m["logt"]
CLIN = ["FCD(参照HS)", "DNT(参照HS)", "CAV(参照HS)", "双重病理(参照HS)", "其他病理(参照HS)",
        "额叶(参照颞叶)", "其他脑叶外(参照颞叶)", "发病年龄(z)", "手术年龄(z)", "术前ASM数(z)",
        "FBTCS", "SE史", "男性"]
X = pd.get_dummies(m[["PathoGrp", "Lobe"]], drop_first=False)
X = X[["PathoGrp_FCD", "PathoGrp_DNT", "PathoGrp_CAV", "PathoGrp_DUAL", "PathoGrp_Other",
       "Lobe_额叶", "Lobe_其他脑叶外"]].astype(float)
X.columns = CLIN[:7]
X["发病年龄(z)"] = m["OnsetMid_z"]
X["手术年龄(z)"] = m["SurgAgeMid_z"]
X["术前ASM数(z)"] = m["ASM_z"]
X["FBTCS"] = m["FBTCS"]
X["SE史"] = m["SE"]
X["男性"] = m["Male"]
for label, col in {"全脑皮层厚度z": "ThickZ_global", "术侧颞叶厚度z": "ThickZ_ipseTL",
                   "对侧颞叶厚度z": "ThickZ_contraTL", "术侧海马体积z": "HippZ_ipse",
                   "术侧颞叶切除范围(z)": "Resect_ipseTL_s", "术侧颞叶切除×log(时间)": "Resect_x_logt"}.items():
    X[label] = m[col].astype(float).values
X = X.astype(float)
base = m[["ID", "start", "stop", "event"]].reset_index(drop=True)
ctv = CoxTimeVaryingFitter()
ctv.fit(base.join(X.reset_index(drop=True)), id_col="ID", event_col="event",
        start_col="start", stop_col="stop")
refit_hr = np.exp(ctv.summary["coef"])
saved_res = pd.read_csv(TAB / "enhanced_cox_results.csv").set_index("covariate")
assert np.allclose(refit_hr.loc[saved_res.index].values, saved_res["HR"].values, rtol=1e-6, atol=1e-8)

# Figure 2: 183 mm × 185 mm, hero validation above coefficient/boundary panels.
fig = plt.figure(figsize=(WIDTH_MM / 25.4, 185 / 25.4), layout="constrained")
gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1.15, 1.15], width_ratios=[1.18, 1.0])
top = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :], width_ratios=[1.35, 1.0])
a1 = fig.add_subplot(top[0, 0])
a2 = fig.add_subplot(top[0, 1])
b = fig.add_subplot(gs[1:, 0])
c = fig.add_subplot(gs[1, 1])
d = fig.add_subplot(gs[2, 1])
for ax, label in [(a1, "a"), (b, "b"), (c, "c"), (d, "d")]:
    ax.set_gid(label)
    panel_label(ax, label)
a2.set_gid("a_delta_aic")

cmp_df = pd.read_csv(TAB / "enhanced_model_comparison.csv")
revision = json.loads((TAB / "revision" / "revision_results.json").read_text(encoding="utf-8"))
boot = revision["bootstrap_C"]
x = np.arange(4)
a1.plot(x, cmp_df["C_index"], color=TEAL, marker="o", markersize=4.2,
        linewidth=1.3, label="Apparent")
a1.plot([0, 3], [boot["M0"]["corrected"], boot["M3"]["corrected"]],
        color=ORANGE, marker="D", markersize=4.5, linewidth=1.2,
        linestyle="--", label="Optimism-corrected")
a1.axhline(0.5, color=LIGHT_GREY, linestyle=":", linewidth=0.8)
a1.set_xticks(x, ["M0", "M1", "M2", "M3"])
a1.set_ylim(0.48, 0.66)
a1.set_ylabel("C-index")
a1.set_title("Discrimination", pad=4)
for i, value in enumerate(cmp_df["C_index"]):
    a1.text(i, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8, color=TEAL)
a1.text(0, boot["M0"]["corrected"] - 0.008, "0.580", ha="center", va="top", fontsize=6.2, color=ORANGE)
a1.text(3, boot["M3"]["corrected"] - 0.008, "0.571", ha="center", va="top", fontsize=6.2, color=ORANGE)
a1.legend(loc="lower left", ncol=2, columnspacing=1.0, handlelength=1.8)
clean_axis(a1)

delta_aic = cmp_df["AIC"] - cmp_df["AIC"].iloc[0]
a2.hlines(x, 0, delta_aic, color=LIGHT_GREY, linewidth=1.2)
a2.plot(x, delta_aic, "o", color=BLUE, markersize=4.2)
a2.axhline(0, color=GREY, linewidth=0.7)
a2.set_xticks(x, ["M0", "M1", "M2", "M3"])
a2.set_ylim(-0.7, 11.5)
a2.set_ylabel("ΔAIC vs M0")
a2.set_title("Relative model fit (lower is better)", pad=4)
for i, value in enumerate(delta_aic):
    a2.text(i, value + 0.45, f"{value:.1f}", ha="center", va="bottom", fontsize=5.8)
clean_axis(a2)

EN_NAME = {
    "FCD(参照HS)": "FCD (vs HS)", "DNT(参照HS)": "DNT (vs HS)",
    "CAV(参照HS)": "Cavernous angioma (vs HS)", "双重病理(参照HS)": "Dual pathology (vs HS)",
    "其他病理(参照HS)": "Other pathology (vs HS)", "额叶(参照颞叶)": "Frontal (vs temporal)",
    "其他脑叶外(参照颞叶)": "Extratemporal-other (vs temporal)",
    "发病年龄(z)": "Age at onset", "手术年龄(z)": "Age at surgery", "术前ASM数(z)": "ASM load",
    "FBTCS": "FBTCS history", "SE史": "Status epilepticus history", "男性": "Male sex",
    "全脑皮层厚度z": "Global thickness z", "术侧颞叶厚度z": "Ipsilateral temporal thickness z",
    "对侧颞叶厚度z": "Contralateral temporal thickness z", "术侧海马体积z": "Ipsilateral hippocampal volume z",
    "术侧颞叶切除范围(z)": "Resection extent", "术侧颞叶切除×log(时间)": "Resection extent × log(time)",
}
target_names = ["全脑皮层厚度z", "术侧颞叶厚度z", "对侧颞叶厚度z", "术侧海马体积z",
                "术侧颞叶切除范围(z)", "术侧颞叶切除×log(时间)"]
ordered_names = target_names + [name for name in saved_res.index if name not in target_names]
y_map = {name: 19 - i if i < 6 else 18 - i for i, name in enumerate(ordered_names)}
# Clinical rows are separated from the six target rows by one empty line.
b.axhspan(13.45, 19.55, color="#EAF1F7", zorder=0)
b.axvline(1, color=GREY, linestyle="--", linewidth=0.8, zorder=1)
for name in ordered_names:
    is_target = name in target_names
    color = BLUE if is_target else GREY
    yi = y_map[name]
    row = saved_res.loc[name]
    b.hlines(yi, row["CI_low"], row["CI_high"], color=color, linewidth=1.05, zorder=2)
    b.plot(row["HR"], yi, "o", color=color, markersize=3.1, zorder=3)
    b.text(4.35, yi, f"{row['p']:.3f}", va="center", ha="left", fontsize=5.2, color=color)
b.set_yticks([y_map[name] for name in ordered_names], [EN_NAME[name] for name in ordered_names])
b.set_xscale("log")
b.set_xlim(0.25, 7)
b.set_ylim(-0.8, 20.1)
b.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
b.xaxis.set_minor_formatter(NullFormatter())
b.xaxis.set_minor_locator(NullLocator())
b.set_xlabel("Hazard ratio (95% CI, log scale)")
b.set_title("M3 coefficients (n = 406, 238 events)", pad=4)
b.text(4.35, 20.0, "p-value", fontsize=5.5, fontweight="bold", va="center")
b.text(0.985, 0.975, "Imaging and resection", transform=b.transAxes,
       ha="right", va="top", fontsize=5.8, color=BLUE, fontweight="bold")
clean_axis(b, "x")

br = float(ctv.summary.loc["术侧颞叶切除范围(z)", "coef"])
bi = float(ctv.summary.loc["术侧颞叶切除×log(时间)", "coef"])
var_idx = [ctv.summary.index.get_loc(name) for name in
           ["术侧颞叶切除范围(z)", "术侧颞叶切除×log(时间)"]]
var = ctv.variance_matrix_.iloc[var_idx, var_idx].values
ts = np.linspace(1, 5, 100)
log_hr = br + bi * np.log(ts)
se = np.sqrt(var[0, 0] + 2 * np.log(ts) * var[0, 1] + np.log(ts) ** 2 * var[1, 1])
hr = np.exp(log_hr)
lo = np.exp(log_hr - 1.96 * se)
hi = np.exp(log_hr + 1.96 * se)
c.fill_between(ts, lo, hi, color=BLUE, alpha=0.16, linewidth=0)
c.plot(ts, hr, color=BLUE, linewidth=1.35)
saved_time = pd.read_csv(TAB / "resect_time_hr.csv")
c.plot(saved_time["year"], saved_time["HR"], "o", color=BLUE, markersize=3.2)
assert np.allclose(np.exp(br + bi * np.log(saved_time["year"].values)), saved_time["HR"].values, rtol=1e-6)
c.axhline(1, color=GREY, linestyle="--", linewidth=0.8)
c.set_xlim(1, 5)
c.set_xticks([1, 2, 3, 4, 5])
c.set_xlabel("Years after surgery")
c.set_ylabel("HR per +1 SD resection extent")
c.set_title("Resection extent × time", pad=4)
p_int = float(saved_res.loc["术侧颞叶切除×log(时间)", "p"])
c.text(0.04, 0.95, f"Interaction p = {p_int:.3f}\nLine and band: HR and 95% CI",
       transform=c.transAxes, ha="left", va="top", fontsize=5.7, color=GREY)
clean_axis(c)

hs_cmp = pd.read_csv(TAB / "hs_subgroup_comparison.csv")
lrt_rows = [
    ("Full: imaging aggregates", float(cmp_df["LRT_vs_prev_p"].iloc[1])),
    ("Full: resection extent", float(cmp_df["LRT_vs_prev_p"].iloc[2])),
    ("Full: resection × time", float(cmp_df["LRT_vs_prev_p"].iloc[3])),
    ("Temporal only: imaging", float(revision["temporal_only"]["LRT_img"])),
    ("Temporal only: resection", float(revision["temporal_only"]["LRT_resect"])),
    ("HS subgroup: imaging", float(hs_cmp["LRT_vs_prev_p"].iloc[1])),
    ("HS subgroup: resection", float(hs_cmp["LRT_vs_prev_p"].iloc[2])),
]
y = np.arange(len(lrt_rows))[::-1]
pvals = np.array([row[1] for row in lrt_rows])
d.hlines(y, 0, -np.log10(pvals), color=LIGHT_GREY, linewidth=1.1)
d.plot(-np.log10(pvals), y, "o", color=TEAL, markersize=3.5)
d.axvline(-np.log10(0.05), color=RED, linestyle="--", linewidth=0.8)
d.set_yticks(y, [row[0] for row in lrt_rows])
d.set_xlim(0, 1.6)
d.set_xlabel("-log10(likelihood-ratio p)")
d.set_title("Incremental model tests", pad=4)
d.text(-np.log10(0.05) + 0.03, len(lrt_rows) - 0.45, "p = 0.05", fontsize=5.5, color=RED, va="center")
for yi, pval in zip(y, pvals):
    d.text(1.57, yi, f"{pval:.2f}", ha="right", va="center", fontsize=5.4, color=GREY)
clean_axis(d, "x")

export_figure(fig, "Figure_2")
print("OPTIMISED FIGURES EXPORTED")
print(f"M3 coefficient verification: maximum relative HR difference = {np.max(np.abs(refit_hr.loc[saved_res.index].values / saved_res['HR'].values - 1)):.3e}")
print(f"Figure 2 panel c: 95% CI at year 5 = {lo[-1]:.3f}–{hi[-1]:.3f}")
