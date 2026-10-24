# -*- coding: utf-8 -*-
"""
Revision analyses responding to the three-reviewer report (2026-10-11).
Covers: R1-M1/M2/M3, R2-M1..M7, R3-M1/M3/M4 (P0/P1 statistical items).
Output: outputs/longitudinal/revision/ (override with IDEAS_OUTPUT_DIR).
"""
import os, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2
from lifelines import KaplanMeierFitter, CoxTimeVaryingFitter
from lifelines.utils import concordance_index
import statsmodels.api as sm

warnings.filterwarnings("ignore")
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("IDEAS_DATA_DIR", REPO_ROOT / "data")).resolve()
OUT = Path(os.environ.get("IDEAS_OUTPUT_DIR", REPO_ROOT / "outputs" / "longitudinal")).resolve()
OUT2 = os.path.join(OUT, "revision")
os.makedirs(OUT2, exist_ok=True)
R = {}

TEMPORAL = ["bankssts", "entorhinal", "fusiform", "inferiortemporal",
            "middletemporal", "parahippocampal", "superiortemporal",
            "temporalpole", "transversetemporal"]

# ---------- base data (identical to longitudinal_analysis.py) ----------
df = pd.read_csv(os.path.join(BASE, "tables_metadata/Metadata_Release_Anon.csv"), dtype={"ID": int})
R["released_rows"] = int(len(df))
df = df.drop_duplicates(subset="ID").reset_index(drop=True)
R["unique_patients"] = int(len(df))

patho_map = {"HS": "HS", "FCD": "FCD", "DNT": "DNT", "CAV": "CAV", "DUAL": "DUAL"}
df["PathoGrp"] = df["Pathology"].map(patho_map).fillna("Other")
lobe_map = {"T Lx": "Temporal", "T Lesx": "Temporal", "T P Lx": "Temporal", "T O Lesx": "Temporal",
            "F Lx": "Frontal", "F Lesx": "Frontal",
            "P Lesx": "Extratemporal", "P Lx": "Extratemporal", "O Lx": "Extratemporal",
            "O Lesx": "Extratemporal", "O P Lx": "Extratemporal", "Hx": "Extratemporal"}
df["Lobe"] = df["Op_Type"].map(lobe_map)
onset_mid = {"Less than 1": 0.5, "1 to 2": 1.5, "3 to 4": 3.5, "5 to 7": 6.0,
             "8 to 10": 9.0, "11 to 14": 12.5, "15 to 19": 17.0, "20 to 24": 22.0,
             "25 to 29": 27.0, "30 to 34": 32.0, "35 to 39": 37.0, "Over 40": 45.0}
surg_mid = {"15 to 20": 17.5, "20 to 24": 22.0, "25 to 29": 27.0, "30 to 34": 32.0,
            "35 to 39": 37.0, "40 to 44": 42.0, "45 to 49": 47.0, "50 to 54": 52.0,
            "55 to 59": 57.0, "60 to 64": 62.0, "65 to 70": 67.0}
df["OnsetMid"] = df["Binned_Onset_Age"].map(onset_mid)
df["SurgAgeMid"] = df["Binned_Age_at_Surgery"].map(surg_mid)
df["ASM"] = pd.to_numeric(df["Number_ASMs"], errors="coerce")
def to_bin(s):
    if s.dtype == bool:
        return s.astype(int)
    return (s.astype(str).str.upper() == "TRUE").astype(int)
df["FUS_b"] = to_bin(df["FUS"]); df["FBTCS_b"] = to_bin(df["FBTCS"]); df["SE_b"] = to_bin(df["SE"])
df["Male"] = (df["Sex"] == "M").astype(int)

# ---------- A. pathology counts & Other composition (R1-M3, R1-m2, R2-m1) ----------
pc = df["Pathology"].value_counts().to_dict()
R["pathology_raw"] = pc
R["pathology_grouped"] = df["PathoGrp"].value_counts().to_dict()

def build_pp(thr=2):
    """person-period + patient summary; event = first year ILAE >= thr; years with
    ILAE < thr remain at risk non-event; truncated at first missing record."""
    rows, psum = [], []
    for _, p in df.iterrows():
        st = []
        for y in range(1, 6):
            v = p[f"ILAE_Year{y}"]
            st.append(None if pd.isna(v) else (1 if float(v) >= thr else 0))
        last = 5
        for i, s in enumerate(st):
            if s is None:
                last = i; break
        event_year = None; event_ilae = None
        for i, s in enumerate(st[:last]):
            if s == 1:
                event_year = i + 1
                event_ilae = float(p[f"ILAE_Year{event_year}"])
                break
        n_obs = last  # consecutive observed records from year 1
        if last == 0:
            later = any(s is not None for s in st)  # first record missing but later observed?
            psum.append(dict(ID=int(p["ID"]), n_fu=0, event_year=np.nan, censored=np.nan,
                             n_obs=0, later_obs=int(later), PathoGrp=p["PathoGrp"],
                             Lobe=p["Lobe"], SurgAgeMid=p["SurgAgeMid"]))
            continue
        obs_last = event_year if event_year else last
        for y in range(1, obs_last + 1):
            rows.append(dict(ID=int(p["ID"]), start=y - 1, stop=y,
                             event=1 if (event_year == y) else 0,
                             PathoGrp=p["PathoGrp"], Lobe=p["Lobe"],
                             OnsetMid=p["OnsetMid"], SurgAgeMid=p["SurgAgeMid"],
                             ASM=p["ASM"], FUS=p["FUS_b"], FBTCS=p["FBTCS_b"],
                             SE=p["SE_b"], Male=p["Male"]))
        # intermittent gap: missing year followed by an observed year
        gap = any(st[i] is None and i < 4 and st[i + 1] is not None for i in range(last if last < 5 else 5))
        psum.append(dict(ID=int(p["ID"]), n_fu=obs_last,
                         event_year=event_year if event_year else np.nan,
                         censored=1 if event_year is None else 0,
                         n_obs=last, later_obs=0, gap=int(gap),
                         event_ilae=event_ilae, PathoGrp=p["PathoGrp"],
                         Lobe=p["Lobe"], SurgAgeMid=p["SurgAgeMid"]))
    return pd.DataFrame(rows), pd.DataFrame(psum)

long2, ps2 = build_pp(2)
R["n_analytic"] = int((ps2["n_fu"] > 0).sum())
R["n_no_fu"] = int((ps2["n_fu"] == 0).sum())
R["n_no_fu_with_later_obs"] = int(ps2.loc[ps2["n_fu"] == 0, "later_obs"].sum())
R["n_events"] = int(ps2["event_year"].notna().sum())
R["n_censored_sf"] = int(ps2["censored"].sum())
R["n_person_periods"] = int(len(long2))

# events per Pathology (grouped and raw)
ev = ps2[ps2["event_year"].notna()]
R["events_by_patho_group"] = ev["PathoGrp"].value_counts().to_dict()
R["events_by_patho_raw_other"] = ev.loc[ev["PathoGrp"] == "Other", "ID"].map(
    df.set_index("ID")["Pathology"]).value_counts().to_dict()
R["n_by_patho_raw_other"] = df.loc[df["PathoGrp"] == "Other", "Pathology"].value_counts().to_dict()

# ---------- F. ILAE class distribution / aura-only counts (R1-M1) ----------
ilae2 = {}
for y in range(1, 6):
    vc = df[f"ILAE_Year{y}"].value_counts().to_dict()
    ilae2[f"year{y}"] = {str(int(k)): int(v) for k, v in vc.items() if pd.notna(k)}
R["ilae_distribution"] = ilae2
R["n_event_ilae2"] = int((ps2["event_ilae"] == 2).sum())   # aura-only first recurrences
R["n_event_ilae3plus"] = int((ps2["event_ilae"] >= 3).sum())

# ---------- B. missingness characterization (R1-M2, R2-M2, R2-m6) ----------
an = ps2[ps2["n_fu"] > 0]
R["n_obs_records_distribution"] = an["n_obs"].value_counts().sort_index().to_dict()
R["n_complete_5yr_records"] = int((an["n_obs"] == 5).sum())
R["n_intermittent_gaps"] = int(an["gap"].sum())
# excluded-15 vs analytic comparison
excl = ps2[ps2["n_fu"] == 0]
R["excluded15_patho"] = excl["PathoGrp"].value_counts().to_dict()
R["excluded15_temporal_pct"] = round(float((excl["Lobe"] == "Temporal").mean()) * 100, 1)
R["analytic_temporal_pct"] = round(float((an["Lobe"] == "Temporal").mean()) * 100, 1)

# risk table with censored per year (Table 2 revision)
rt = []
for y in range(1, 6):
    at_risk = int(((ps2["n_fu"] >= y) & (ps2["event_year"].isna() | (ps2["event_year"] >= y))).sum())
    events = int((ps2["event_year"] == y).sum())
    censored = int(((ps2["n_fu"] == y) & (ps2["censored"] == 1)).sum()) if y < 5 else \
               int(((ps2["n_fu"] == y) & (ps2["censored"] == 1)).sum())
    rt.append(dict(year=y, at_risk=at_risk, events=events, censored=censored,
                   hazard=round(events / at_risk * 100, 1) if at_risk else np.nan))
R["risk_table"] = rt

# ---------- C. exact KM with CI (R2-m2, R1-m5) ----------
def km_fit(sub):
    d = sub[sub["n_fu"] > 0].copy()
    d["duration"] = d["event_year"].fillna(d["n_fu"])
    d["event"] = 1 - d["censored"].astype(int)
    kmf = KaplanMeierFitter().fit(d["duration"], d["event"])
    return kmf, d
kmf, dkm = km_fit(ps2)
ci = kmf.confidence_interval_survival_function_
R["km_1yr"] = round(float(kmf.predict(1)) * 100, 1)
R["km_5yr"] = round(float(kmf.predict(5)) * 100, 1)
R["km_5yr_ci"] = [round(float(ci.iloc[-1, 0]) * 100, 1), round(float(ci.iloc[-1, 1]) * 100, 1)]
R["km_implied_product_hazards"] = round(float(np.prod([1 - r["hazard"] / 100 for r in rt])) * 100, 1)

# ---------- D. best/worst-case dropout bounds (R1-M2, R2-M2) ----------
d = dkm.copy()
d_worst = d.copy(); d_worst["event"] = 1  # all censored = recurrence at censor time
kmf_w = KaplanMeierFitter().fit(d_worst["duration"], d_worst["event"])
R["km_5yr_worst_all_censored_event"] = round(float(kmf_w.predict(5)) * 100, 1)
# alternative worst: censored patients recur the year after their last record
d_w2 = d.copy()
d_w2["dur2"] = np.where(d_w2["event"] == 0, np.minimum(d_w2["duration"] + 1, 5), d_w2["duration"])
kmf_w2 = KaplanMeierFitter().fit(d_w2["dur2"], np.ones(len(d_w2)))
R["km_5yr_worst_next_year_event"] = round(float(kmf_w2.predict(5)) * 100, 1)
# best case: censored remain seizure-free through year 5 (KM as observed, i.e., optimistic bound)
R["km_5yr_best_as_observed"] = R["km_5yr"]

# ---------- clinical model helpers ----------
def clin_design(m):
    m = m.copy()
    m["OnsetMid_z"] = (m["OnsetMid"] - m["OnsetMid"].mean()) / m["OnsetMid"].std()
    m["SurgAgeMid_z"] = (m["SurgAgeMid"] - m["SurgAgeMid"].mean()) / m["SurgAgeMid"].std()
    m["ASM_z"] = (m["ASM"] - m["ASM"].mean()) / m["ASM"].std()
    X = pd.get_dummies(m[["PathoGrp", "Lobe"]], drop_first=False)
    X = X[["PathoGrp_FCD", "PathoGrp_DNT", "PathoGrp_CAV", "PathoGrp_DUAL", "PathoGrp_Other",
           "Lobe_Frontal", "Lobe_Extratemporal"]].astype(float)
    X.columns = ["FCD", "DNT", "CAV", "DUAL", "Other", "Frontal", "Extratemporal"]
    X["OnsetAge_z"] = m["OnsetMid_z"]; X["SurgAge_z"] = m["SurgAgeMid_z"]
    X["ASM_z"] = m["ASM_z"]; X["FBTCS"] = m["FBTCS"]; X["SE"] = m["SE"]; X["Male"] = m["Male"]
    return m, X.astype(float)

def cox_fit(m, X, robust=False):
    ctv = CoxTimeVaryingFitter()
    fitdf = m[["ID", "start", "stop", "event"]].reset_index(drop=True).join(X.reset_index(drop=True))
    ctv.fit(fitdf, id_col="ID", event_col="event", start_col="start", stop_col="stop",
            robust=robust, show_progress=False)
    return ctv

def cox_table(ctv):
    s = ctv.summary
    return pd.DataFrame({"HR": np.exp(s["coef"]), "CI_low": np.exp(s["coef lower 95%"]),
                         "CI_high": np.exp(s["coef upper 95%"]), "p": s["p"]})

# clinical model sample & SDs (R2-m5, R2-m7)
mC, XC = clin_design(long2.dropna(subset=["OnsetMid", "SurgAgeMid", "ASM"]))
R["clinical_model_n"] = int(mC["ID"].nunique())
R["clinical_model_events"] = int(mC["event"].sum())
R["clinical_model_person_periods"] = int(len(mC))
R["standardisation_SDs"] = {"OnsetAge": round(float(long2.dropna(subset=["OnsetMid","SurgAgeMid","ASM"])["OnsetMid"].std()), 2),
                            "SurgAge": round(float(long2.dropna(subset=["OnsetMid","SurgAgeMid","ASM"])["SurgAgeMid"].std()), 2),
                            "ASM": round(float(long2.dropna(subset=["OnsetMid","SurgAgeMid","ASM"])["ASM"].std()), 2)}
R["onset_bins_scheme"] = sorted(df["Binned_Onset_Age"].dropna().unique().tolist())
R["surg_bins_scheme"] = sorted(df["Binned_Age_at_Surgery"].dropna().unique().tolist())
R["onset_median_bin"] = df["Binned_Onset_Age"].mode()[0]
R["pct_onset_before_15"] = round(float(df["OnsetMid"].le(14).mean()) * 100, 1)

# ---------- G. M0 model-based SE + cluster-robust SE via cloglog GLM (R2-M7) ----------
# lifelines CoxTimeVaryingFitter does not implement robust SE; use the equivalent
# Prentice-Gloeckler grouped-data Cox (cloglog GLM) with cluster-robust covariance.
ctv_mb = cox_fit(mC, XC, robust=False)
t_mb = cox_table(ctv_mb)
Xg_mb = sm.add_constant(XC)
glm_cr = sm.GLM(mC["event"], Xg_mb, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit(
    cov_type="cluster", cov_kwds={"groups": mC["ID"].values})
glm_cr_base = sm.GLM(mC["event"], Xg_mb, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
R["M0_model_based"] = t_mb.round(4).to_dict("index")
R["M0_cloglog_cluster_robust"] = {k: dict(HR=round(float(np.exp(v)), 3),
                                          CI=[round(float(np.exp(v - 1.96 * glm_cr.bse[k])), 3),
                                              round(float(np.exp(v + 1.96 * glm_cr.bse[k])), 3)],
                                          p=round(float(glm_cr.pvalues[k]), 4))
                                  for k, v in glm_cr.params.items() if k != "const"}
R["M0_cloglog_modelbased_HR"] = {k: dict(HR=round(float(np.exp(v)), 3),
                                         p=round(float(glm_cr_base.pvalues[k]), 4))
                                 for k, v in glm_cr_base.params.items() if k != "const"}
R["M0_loglik"] = float(ctv_mb.log_likelihood_)

# ---------- H. PH diagnostics: covariate x log(t) LRTs (R2-M1) ----------
mC = mC.copy(); mC["logt"] = np.log(mC["stop"])
ph_tests = {}
ll0, k0 = ctv_mb.log_likelihood_, XC.shape[1]
for cov in XC.columns:
    Xi = XC.copy(); Xi[f"{cov}_x_logt"] = (XC[cov] * mC["logt"]).values
    ctv_i = cox_fit(mC, Xi)
    stat = 2 * (ctv_i.log_likelihood_ - ll0)
    ph_tests[cov] = round(float(chi2.sf(stat, 1)), 4)
R["PH_interaction_LRT_p"] = ph_tests

# ---------- I. FDR (BH) across clinical coefficients (R2-M6) ----------
from statsmodels.stats.multitest import multipletests
pvals = t_mb["p"].values
rej, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
t_mb["p_fdr"] = p_adj
R["M0_with_FDR"] = t_mb.round(4).to_dict("index")

# ---------- E. ILAE>=3 sensitivity (R1-M1) ----------
long3, ps3 = build_pp(3)
rt3 = []
for y in range(1, 6):
    at_risk = int(((ps3["n_fu"] >= y) & (ps3["event_year"].isna() | (ps3["event_year"] >= y))).sum())
    events = int((ps3["event_year"] == y).sum())
    rt3.append(dict(year=y, at_risk=at_risk, events=events,
                    hazard=round(events / at_risk * 100, 1) if at_risk else np.nan))
R["ilae3_risk_table"] = rt3
kmf3, d3 = km_fit(ps3)
R["ilae3_km_1yr"] = round(float(kmf3.predict(1)) * 100, 1)
R["ilae3_km_5yr"] = round(float(kmf3.predict(5)) * 100, 1)
R["ilae3_n_events"] = int(ps3["event_year"].notna().sum())
m3s, X3s = clin_design(long3.dropna(subset=["OnsetMid", "SurgAgeMid", "ASM"]))
ctv3 = cox_fit(m3s, X3s)
R["ilae3_cox"] = cox_table(ctv3).round(4).to_dict("index")

# ---------- J. cloglog GEE (R2-M5) ----------
Xg = sm.add_constant(XC)
gee_logit = sm.GEE(mC["event"], Xg, groups=mC["ID"],
                   family=sm.families.Binomial(),
                   cov_struct=sm.cov_struct.Exchangeable()).fit()
gee_cloglog = sm.GEE(mC["event"], Xg, groups=mC["ID"],
                     family=sm.families.Binomial(link=sm.families.links.CLogLog()),
                     cov_struct=sm.cov_struct.Exchangeable()).fit()
R["GEE_logit_OR"] = {k: dict(OR=round(float(np.exp(v)), 3), p=round(float(gee_logit.pvalues[k]), 4))
                     for k, v in gee_logit.params.items() if k != "const"}
R["GEE_cloglog_HR"] = {k: dict(HR=round(float(np.exp(v)), 3), p=round(float(gee_cloglog.pvalues[k]), 4))
                       for k, v in gee_cloglog.params.items() if k != "const"}
R["GEE_n_patients"] = int(mC["ID"].nunique())

# ---------- K. enhanced (imaging) sample rebuild ----------
tz = pd.read_csv(os.path.join(BASE, "zscores/thick_zscores_patients.csv"))
vz = pd.read_csv(os.path.join(BASE, "zscores/vol_zscores_patients.csv"))
cli = df.set_index("ID")
side = cli["Op_Side"].astype(str).str.upper().str.strip().str[0]
img = tz[["ID"]].copy()
thick_cols = [c for c in tz.columns if c.endswith("_thickness")]
img["ThickZ_global"] = tz[thick_cols].mean(axis=1)
img["side"] = img["ID"].map(side)
img = img[img["side"].isin(["L", "R"])].copy()
img["h_ipse"] = np.where(img["side"] == "L", "lh", "rh")
img["h_contra"] = np.where(img["side"] == "L", "rh", "lh")
tz_idx = tz.set_index("ID")
for tag, hcol in [("ThickZ_ipseTL", "h_ipse"), ("ThickZ_contraTL", "h_contra")]:
    vals = []
    for i, h in zip(img["ID"], img[hcol]):
        cols = [f"{h}_{r}_thickness" for r in TEMPORAL]
        vals.append(float(tz_idx.loc[i, cols].astype(float).mean(axis=0) if hasattr(tz_idx.loc[i, cols], "mean") else np.mean(tz_idx.loc[i, cols])))
    img[tag] = vals
vz_idx = vz.set_index("ID")
hip_col = {"L": "Left_Hippocampus_volume", "R": "Right_Hippocampus_volume"}
img["HippZ_ipse"] = [float(vz_idx.loc[i, hip_col[s]]) for i, s in zip(img["ID"], img["side"])]

res_tab = pd.read_csv(os.path.join(BASE, "resected_table/table_resected.csv"), index_col=0)
res_tab.columns = [int(float(c)) for c in res_tab.columns]
res_tab.index = [str(x).strip().strip("'").strip('"').lstrip("\ufeff") for x in res_tab.index]
def resect_pct(sid, regions):
    if sid not in res_tab.columns:
        return np.nan
    row = res_tab.loc[res_tab.index.isin(regions), sid]
    return float(pd.to_numeric(row, errors="coerce").mean())
img["Resect_ipseTL"] = [resect_pct(i, [f"ctx-{h}-{r}" for r in TEMPORAL] +
                                   [("Left-Hippocampus" if s == "L" else "Right-Hippocampus"),
                                    ("Left-Amygdala" if s == "L" else "Right-Amygdala")])
                        for i, s, h in zip(img["ID"], img["side"], img["h_ipse"])]
img_vars = ["ThickZ_global", "ThickZ_ipseTL", "ThickZ_contraTL", "HippZ_ipse", "Resect_ipseTL"]
mg = long2.merge(img[["ID"] + img_vars], on="ID", how="inner")
mg = mg.dropna(subset=["OnsetMid", "SurgAgeMid", "ASM"] + img_vars).copy()
# focal indices (R3-M1): per-patient max |z| in ipsilateral temporal regions, ipsi-contra asymmetry
fmax, asym = {}, {}
for i, h_i, h_c in zip(img["ID"], img["h_ipse"], img["h_contra"]):
    zi = [float(tz_idx.loc[i, f"{h_i}_{r}_thickness"]) for r in TEMPORAL]
    zc = [float(tz_idx.loc[i, f"{h_c}_{r}_thickness"]) for r in TEMPORAL]
    fmax[i] = float(np.max(np.abs(zi)))
    asym[i] = float(np.mean(zi) - np.mean(zc))
mg["FocalMaxZ_ipseTL"] = mg["ID"].map(fmax)
mg["AsymZ_TL"] = mg["ID"].map(asym)

mg, XE = clin_design(mg)
mg["logt"] = np.log(mg["stop"])
mg["Resect_s"] = (mg["Resect_ipseTL"] - mg["Resect_ipseTL"].mean()) / mg["Resect_ipseTL"].std()
mg["Resect_x_logt"] = mg["Resect_s"] * mg["logt"]
R["enhanced_n"] = int(mg["ID"].nunique())
R["enhanced_events"] = int(mg["event"].sum())
R["enhanced_person_periods"] = int(len(mg))
# attrition trail (R1-m3, R2-m7, R3-m4)
analytic_ids = set(ps2.loc[ps2["n_fu"] > 0, "ID"])
R["trail"] = dict(
    unique=442,
    analytic=R["n_analytic"],
    clinical_model=R["clinical_model_n"],
    with_imaging=int(len(set(ps2.loc[ps2["n_fu"] > 0, "ID"]) & set(img["ID"]))),
    complete_case=R["enhanced_n"])
# why 425->406
an_ids_img = analytic_ids & set(img["ID"])
cc_ids = set(mg["ID"].dropna().unique()) if False else set(mg["ID"])
missing_reason = {}
for v in img_vars:
    sub = img[img["ID"].isin(an_ids_img - cc_ids)]
    missing_reason[v] = int(sub[v].isna().sum())
R["attrition_missing_reasons"] = missing_reason
R["attrition_n_excluded_425_406"] = len(an_ids_img - cc_ids)

# imaging predictor distributions (R3-M1 restricted-variance check)
R["img_predictor_sd"] = {v: dict(SD=round(float(mg[v].std()), 2),
                                 min=round(float(mg[v].min()), 2),
                                 max=round(float(mg[v].max()), 2)) for v in img_vars}

# M0..M3 on enhanced sample with coefficient CIs + minimum detectable HR (R2-M3, R3-M3)
IMGZ = {"ThickZ_global": "ThickZ_global", "ThickZ_ipseTL": "ThickZ_ipseTL",
        "ThickZ_contraTL": "ThickZ_contraTL", "HippZ_ipse": "HippZ_ipse"}
def fit_enh(extra):
    Xf = XE.copy()
    for lab, col in extra.items():
        Xf[lab] = mg[col].astype(float).values
    ctv = cox_fit(mg, Xf)
    ph = np.asarray(ctv.predict_partial_hazard(Xf)).ravel()
    cidx = concordance_index(mg["stop"].astype(float), -ph, mg["event"].astype(int))
    return ctv, Xf, cidx
mods = {"M0": fit_enh({}), "M1": fit_enh(IMGZ),
        "M2": fit_enh({**IMGZ, "Resect_s": "Resect_s"}),
        "M3": fit_enh({**IMGZ, "Resect_s": "Resect_s", "Resect_x_logt": "Resect_x_logt"})}
cmp_rows, prev = [], None
for name, (ctv, Xf, cidx) in mods.items():
    ll, k = ctv.log_likelihood_, Xf.shape[1]
    lrt = np.nan
    if prev is not None:
        lrt = float(chi2.sf(2 * (ll - prev[0]), k - prev[1]))
    cmp_rows.append(dict(model=name, k=k, loglik=round(ll, 2),
                         AIC=round(-2 * ll + 2 * k, 2), C_index=round(cidx, 3),
                         LRT_p=lrt))
    prev = (ll, k)
R["enhanced_model_comparison"] = cmp_rows
tM3 = cox_table(mods["M3"][0])
seM3 = mods["M3"][0].summary["se(coef)"]
tM3["min_detectable_HR_80pct"] = np.exp(2.8 * seM3)
R["M3_coefficients"] = tM3.round(3).to_dict("index")

# minimum detectable HR for imaging block (per SD, 80% power) & Other pathology
R["min_detectable"] = {
    "Other_pathology": round(float(np.exp(2.8 * seM3["Other"])), 2),
    "ThickZ_global": round(float(np.exp(2.8 * seM3["ThickZ_global"])), 2),
    "ThickZ_ipseTL": round(float(np.exp(2.8 * seM3["ThickZ_ipseTL"])), 2),
    "ThickZ_contraTL": round(float(np.exp(2.8 * seM3["ThickZ_contraTL"])), 2),
    "HippZ_ipse": round(float(np.exp(2.8 * seM3["HippZ_ipse"])), 2),
    "Resect": round(float(np.exp(2.8 * seM3["Resect_s"])), 2)}

# temporal-only resection sensitivity (R3-M4): design without site dummies
mgT = mg[mg["Lobe"] == "Temporal"].copy()
mgT["OnsetMid_z"] = (mgT["OnsetMid"] - mgT["OnsetMid"].mean()) / mgT["OnsetMid"].std()
mgT["SurgAgeMid_z"] = (mgT["SurgAgeMid"] - mgT["SurgAgeMid"].mean()) / mgT["SurgAgeMid"].std()
mgT["ASM_z"] = (mgT["ASM"] - mgT["ASM"].mean()) / mgT["ASM"].std()
XT = pd.get_dummies(mgT[["PathoGrp"]], drop_first=False)[
    ["PathoGrp_FCD", "PathoGrp_DNT", "PathoGrp_CAV", "PathoGrp_DUAL", "PathoGrp_Other"]].astype(float)
XT.columns = ["FCD", "DNT", "CAV", "DUAL", "Other"]
XT["OnsetAge_z"] = mgT["OnsetMid_z"].values; XT["SurgAge_z"] = mgT["SurgAgeMid_z"].values
XT["ASM_z"] = mgT["ASM_z"].values; XT["FBTCS"] = mgT["FBTCS"].values
XT["SE"] = mgT["SE"].values; XT["Male"] = mgT["Male"].values
XT = XT.astype(float)
mgT["Resect_s"] = (mgT["Resect_ipseTL"] - mgT["Resect_ipseTL"].mean()) / mgT["Resect_ipseTL"].std()
def fit_T(extra):
    Xf = XT.copy()
    for lab, col in extra.items():
        Xf[lab] = mgT[col].astype(float).values
    ctv = cox_fit(mgT, Xf)
    return ctv, Xf
tm0, _ = fit_T({}); tm1, _ = fit_T(IMGZ); tm2, _ = fit_T({**IMGZ, "Resect_s": "Resect_s"})
R["temporal_only"] = dict(
    n=int(mgT["ID"].nunique()), events=int(mgT["event"].sum()),
    LRT_img=round(float(chi2.sf(2 * (tm1.log_likelihood_ - tm0.log_likelihood_), 4)), 4),
    LRT_resect=round(float(chi2.sf(2 * (tm2.log_likelihood_ - tm1.log_likelihood_), 1)), 4),
    resect_HR=round(float(np.exp(tm2.summary.loc["Resect_s", "coef"])), 3),
    resect_p=round(float(tm2.summary.loc["Resect_s", "p"]), 4),
    resect_CI=[round(float(np.exp(tm2.summary.loc["Resect_s", "coef lower 95%"])), 3),
               round(float(np.exp(tm2.summary.loc["Resect_s", "coef upper 95%"])), 3)])

# all-68-region block + focal indices via cloglog GLM (R3-M1)
Xg0 = sm.add_constant(XE)
glm0 = sm.GLM(mg["event"], Xg0, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
# build 68-region matrix per patient, map into person-period
region_cols = thick_cols  # 68 (lh+rh) thickness columns
zreg = tz[["ID"] + region_cols].set_index("ID")
mg_reg = mg.copy()
for c in region_cols:
    mg_reg[c] = mg_reg["ID"].map(zreg[c])
mg_reg["LeftHippZ"] = mg_reg["ID"].map(vz_idx["Left_Hippocampus_volume"])
mg_reg["RightHippZ"] = mg_reg["ID"].map(vz_idx["Right_Hippocampus_volume"])
Xfull = pd.concat([Xg0.reset_index(drop=True),
                   mg_reg[region_cols + ["LeftHippZ", "RightHippZ"]].reset_index(drop=True)], axis=1)
glmF = sm.GLM(mg_reg["event"].values, Xfull, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
R["all_region_LRT"] = dict(
    df=70, chi2=round(2 * (glmF.llf - glm0.llf), 2),
    p=round(float(chi2.sf(2 * (glmF.llf - glm0.llf), 70)), 4))
# focal indices
Xfoc = Xg0.copy()
Xfoc["FocalMaxZ_ipseTL"] = mg["FocalMaxZ_ipseTL"].values
Xfoc["AsymZ_TL"] = mg["AsymZ_TL"].values
glmFoc = sm.GLM(mg["event"], Xfoc, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
R["focal_LRT"] = dict(df=2, chi2=round(2 * (glmFoc.llf - glm0.llf), 2),
                      p=round(float(chi2.sf(2 * (glmFoc.llf - glm0.llf), 2)), 4))

# HS subgroup events (R2-M3)
hs_ids = set(ps2.loc[(ps2["PathoGrp"] == "HS") & (ps2["n_fu"] > 0), "ID"])
hs_cc = mg[mg["PathoGrp"] == "HS"]
R["hs_subgroup"] = dict(analytic_HS=len(hs_ids), complete_case=int(hs_cc["ID"].nunique()),
                        events=int(hs_cc.groupby("ID")["event"].max().sum()))

# ---------- bootstrap C-index validation (R2-M4) ----------
rng = np.random.default_rng(42)
ids = mg["ID"].unique()
apparent = {"M0": mods["M0"][2], "M3": mods["M3"][2]}
boot = {"M0": [], "M3": []}
boot_errors = []
B = 150
mg_indexed = mg.reset_index(drop=True)
for b in range(B):
    pick = rng.choice(ids, size=len(ids), replace=True)
    parts = []
    for j, pid in enumerate(pick):
        sub = mg_indexed[mg_indexed["ID"] == pid].copy()
        sub["ID"] = j  # unique bootstrap id
        parts.append(sub)
    mb = pd.concat(parts).reset_index(drop=True)
    for name, extra in [("M0", {}), ("M3", {**IMGZ, "Resect_s": "Resect_s", "Resect_x_logt": "Resect_x_logt"})]:
        try:
            Xb = clin_design(mb)[1]
            for lab, col in extra.items():
                if col == "Resect_s":
                    Xb[lab] = ((mb["Resect_ipseTL"] - mb["Resect_ipseTL"].mean()) / mb["Resect_ipseTL"].std()).values
                elif col == "Resect_x_logt":
                    rs = (mb["Resect_ipseTL"] - mb["Resect_ipseTL"].mean()) / mb["Resect_ipseTL"].std()
                    Xb[lab] = (rs * np.log(mb["stop"])).values
                else:
                    Xb[lab] = mb[col].astype(float).values
            ctv_b = cox_fit(mb, Xb)
            ph_b = np.asarray(ctv_b.predict_partial_hazard(Xb)).ravel()
            c_app = concordance_index(mb["stop"].astype(float), -ph_b, mb["event"].astype(int))
            XE_test = XE.copy()
            if name == "M3":
                XE_test["Resect_s"] = mg["Resect_s"].values
                XE_test["Resect_x_logt"] = mg["Resect_x_logt"].values
                for lab, col in IMGZ.items():
                    XE_test[lab] = mg[col].astype(float).values
            ph_t = np.asarray(ctv_b.predict_partial_hazard(XE_test.reset_index(drop=True))).ravel()
            c_test = concordance_index(mg["stop"].astype(float), -ph_t, mg["event"].astype(int))
            boot[name].append(c_app - c_test)
        except Exception as e:
            if len(boot_errors) < 3:
                boot_errors.append(f"{name}: {type(e).__name__}: {e}")
    if (b + 1) % 25 == 0:
        print(f"bootstrap {b+1}/{B} (M0={len(boot['M0'])}, M3={len(boot['M3'])})", flush=True)
if boot_errors:
    print("BOOT ERRORS:", boot_errors)
R["bootstrap_C"] = {}
for name in ["M0", "M3"]:
    diffs = boot[name]
    tests = [apparent[name] - d for d in diffs]
    R["bootstrap_C"][name] = dict(
        B=len(diffs), apparent=round(apparent[name], 3),
        optimism=round(float(np.mean(diffs)), 4),
        corrected=round(apparent[name] - float(np.mean(diffs)), 3),
        test_CI=[round(float(np.percentile(tests, 2.5)), 3),
                 round(float(np.percentile(tests, 97.5)), 3)] if tests else None)

# ---------- M. lesional/MRI column check (R3-m5) ----------
mri_cols = [c for c in df.columns if "mri" in c.lower() or "lesion" in c.lower()]
R["mri_related_columns"] = mri_cols
if mri_cols:
    c = mri_cols[0]
    R["lesional_by_patho"] = df.groupby("PathoGrp")[c].value_counts(dropna=False).to_dict()

# ---------- z-score provenance (R3-M2) ----------
R["zscores_files"] = os.listdir(os.path.join(BASE, "zscores"))
R["zscores_columns_sample"] = tz.columns[:6].tolist() + ["..."] + tz.columns[-3:].tolist()
R["resected_table_shape"] = list(res_tab.shape)

with open(os.path.join(OUT2, "revision_results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)

# console summary
print(json.dumps({k: R[k] for k in ["released_rows", "unique_patients", "n_analytic", "n_events",
                                    "n_no_fu", "n_no_fu_with_later_obs", "n_person_periods",
                                    "n_event_ilae2", "n_event_ilae3plus"]},
                 ensure_ascii=False, indent=1))
print("\nRISK TABLE:", json.dumps(R["risk_table"], ensure_ascii=False))
print("\nKM:", R["km_1yr"], R["km_5yr"], R["km_5yr_ci"], "| product:", R["km_implied_product_hazards"])
print("\nBOUNDS: worst(next-year)=", R["km_5yr_worst_next_year_event"],
      " worst(all)=", R["km_5yr_worst_all_censored_event"], " best=", R["km_5yr_best_as_observed"])
print("\nMissingness: n_obs dist=", R["n_obs_records_distribution"],
      " complete5=", R["n_complete_5yr_records"], " gaps=", R["n_intermittent_gaps"])
print("\nILAE3:", json.dumps(R["ilae3_risk_table"], ensure_ascii=False),
      " KM1/5=", R["ilae3_km_1yr"], R["ilae3_km_5yr"], " events=", R["ilae3_n_events"])
print("\nILAE3 Cox (key):", {k: v for k, v in R["ilae3_cox"].items() if k in ["Other", "FBTCS", "DNT"]})
print("\nPH LRT p:", R["PH_interaction_LRT_p"])
print("\nM0 robust Other:", R["M0_robust"]["Other"])
print("\nFDR (key):", {k: (v["p"], v["p_fdr"]) for k, v in R["M0_with_FDR"].items() if k in ["Other", "FBTCS", "SE", "DNT"]})
print("\nGEE logit Other:", R["GEE_logit_OR"]["Other"], "| cloglog Other:", R["GEE_cloglog_HR"]["Other"])
print("\nTrail:", R["trail"], "| attrition reasons:", R["attrition_missing_reasons"])
print("\nEnhanced cmp:", json.dumps(R["enhanced_model_comparison"], ensure_ascii=False))
print("\nM3 imaging coefs:", {k: v for k, v in R["M3_coefficients"].items()
                              if k in IMGZ or k in ["Resect_s", "Resect_x_logt", "Other"]})
print("\nMinDetectable:", R["min_detectable"])
print("\nTemporal-only:", R["temporal_only"])
print("\nAll-region LRT:", R["all_region_LRT"], "| Focal LRT:", R["focal_LRT"])
print("\nHS subgroup:", R["hs_subgroup"])
print("\nBootstrap C:", json.dumps(R["bootstrap_C"], ensure_ascii=False))
print("\nMRI cols:", R["mri_related_columns"])
print("\nOnset bins:", R["onset_bins_scheme"])
print("\nOnset median bin:", R["onset_median_bin"], "| pct<15:", R["pct_onset_before_15"])
print("\nOther composition:", R["n_by_patho_raw_other"], "events:", R["events_by_patho_raw_other"])
print("\nimg SD:", R["img_predictor_sd"])
print("\nDONE -> revision_results.json")
