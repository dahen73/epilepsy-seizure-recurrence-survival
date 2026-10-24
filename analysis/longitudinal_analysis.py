# -*- coding: utf-8 -*-
"""
IDEAS (ds005602) 术后发作结局纵向分析
- 离散时间生存分析: KM + log-rank
- 纵向风险建模: 离散时间 Cox (CoxTimeVaryingFitter) + GEE 敏感性分析
输出: outputs/longitudinal/（可用 IDEAS_OUTPUT_DIR 修改）
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxTimeVaryingFitter
from lifelines.statistics import multivariate_logrank_test

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("IDEAS_DATA_DIR", REPO_ROOT / "data")).resolve()
BASE = DATA_ROOT / "tables_metadata"
OUT = Path(os.environ.get("IDEAS_OUTPUT_DIR", REPO_ROOT / "outputs" / "longitudinal")).resolve()
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 数据准备 ----------
df = pd.read_csv(os.path.join(BASE, "Metadata_Release_Anon.csv"), dtype={"ID": int})
df = df.drop_duplicates(subset="ID").reset_index(drop=True)

patho_map = {"HS": "HS", "FCD": "FCD", "DNT": "DNT", "CAV": "CAV", "DUAL": "DUAL"}
df["PathoGrp"] = df["Pathology"].map(patho_map).fillna("Other")
lobe_map = {"T Lx": "颞叶", "T Lesx": "颞叶", "T P Lx": "颞叶", "T O Lesx": "颞叶",
            "F Lx": "额叶", "F Lesx": "额叶",
            "P Lesx": "其他脑叶外", "P Lx": "其他脑叶外", "O Lx": "其他脑叶外",
            "O Lesx": "其他脑叶外", "O P Lx": "其他脑叶外", "Hx": "其他脑叶外"}
df["Lobe"] = df["Op_Type"].map(lobe_map)

onset_mid = {"Less than 1": 0.5, "1 to 2": 1.5, "3 to 4": 3.5, "5 to 7": 6.0,
             "8 to 10": 9.0, "11 to 14": 12.5, "15 to 19": 17.0, "20 to 24": 22.0,
             "25 to 29": 27.0, "30 to 34": 32.0, "35 to 39": 37.0, "Over 40": 45.0}
df["OnsetMid"] = df["Binned_Onset_Age"].map(onset_mid)
surg_mid = {"15 to 20": 17.5, "20 to 24": 22.0, "25 to 29": 27.0, "30 to 34": 32.0,
            "35 to 39": 37.0, "40 to 44": 42.0, "45 to 49": 47.0, "50 to 54": 52.0,
            "55 to 59": 57.0, "60 to 64": 62.0, "65 to 70": 67.0}
df["SurgAgeMid"] = df["Binned_Age_at_Surgery"].map(surg_mid)
df["ASM"] = pd.to_numeric(df["Number_ASMs"], errors="coerce")
def to_bin(s):
    if s.dtype == bool:
        return s.astype(int)
    return (s.astype(str).str.upper() == "TRUE").astype(int)
df["FUS_b"] = to_bin(df["FUS"])
df["FBTCS_b"] = to_bin(df["FBTCS"])
df["SE_b"] = to_bin(df["SE"])
df["Male"] = (df["Sex"] == "M").astype(int)

# person-period 长格式（在首个 NA 处删失截断；终点=首次复发 ILAE>=2）
yrs = [f"ILAE_Year{y}" for y in range(1, 6)]
rows, patient_sum = [], []
for _, p in df.iterrows():
    st = []
    for y in range(1, 6):
        v = p[f"ILAE_Year{y}"]
        if pd.isna(v):
            st.append(None)
        else:
            st.append(0 if float(v) == 1.0 else 1)  # ILAE=1 无发作, >=2 复发
    last = 5
    for i, s in enumerate(st):
        if s is None:
            last = i
            break
    event_year = None
    for i, s in enumerate(st[:last]):
        if s == 1:
            event_year = i + 1
            break
    if last == 0:
        patient_sum.append(dict(ID=p["ID"], n_fu=0, event_year=np.nan, censored=np.nan))
        continue
    obs_last = event_year if event_year else last
    for y in range(1, obs_last + 1):
        rows.append(dict(ID=int(p["ID"]), start=y - 1, stop=y,
                         event=1 if (event_year == y) else 0,
                         PathoGrp=p["PathoGrp"], Lobe=p["Lobe"],
                         OnsetMid=p["OnsetMid"], SurgAgeMid=p["SurgAgeMid"],
                         ASM=p["ASM"], FUS=p["FUS_b"], FBTCS=p["FBTCS_b"],
                         SE=p["SE_b"], Male=p["Male"]))
    patient_sum.append(dict(ID=int(p["ID"]), n_fu=obs_last,
                            event_year=event_year if event_year else np.nan,
                            censored=1 if event_year is None else 0))

long = pd.DataFrame(rows)
ps = pd.DataFrame(patient_sum).merge(df[["ID", "PathoGrp", "Lobe"]], on="ID", how="left")
n_total = len(df)
n_nofu = int((ps["n_fu"] == 0).sum())
n_event = int(ps["event_year"].notna().sum())
n_cens = int((ps["censored"] == 1).sum())
long.to_csv(os.path.join(OUT, "person_period_long.csv"), index=False, encoding="utf-8-sig")

# ---------- 2. 逐年风险表 ----------
haz = []
for y in range(1, 6):
    at_risk = int(((ps["n_fu"] >= y) & (ps["event_year"].isna() | (ps["event_year"] >= y))).sum())
    ev = int((ps["event_year"] == y).sum())
    haz.append(dict(year=y, at_risk=at_risk, events=ev,
                    hazard=round(ev / at_risk * 100, 1) if at_risk else np.nan))
haz_df = pd.DataFrame(haz)
haz_df.to_csv(os.path.join(OUT, "yearly_hazard.csv"), index=False, encoding="utf-8-sig")

# ---------- 3. KM 曲线 ----------
# KM：duration = 复发年(event_year) 或删失时点(n_fu)；event=复发
def prep_km(sub):
    d = sub[(sub["n_fu"] > 0)].copy()
    d["duration"] = d["event_year"].fillna(d["n_fu"])
    d["event"] = 1 - d["censored"].astype(int)
    return d

def km_plot(groups, fname, title):
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=150)
    ax.set_prop_cycle(color=["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"])
    for name, sub in groups:
        d = prep_km(sub)
        if len(d) == 0:
            continue
        kmf = KaplanMeierFitter(label=f"{name} (n={len(d)})")
        kmf.fit(d["duration"], d["event"])
        kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2)
    ax.set_xlim(0, 5); ax.set_xticks(range(0, 6))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("术后年数"); ax.set_ylabel("持续无发作概率 (ILAE=1)")
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

km_plot([("全体患者", ps)], "km_overall.png", "术后至首次复发生存曲线（全体，n=427）")
km_plot([(k, ps[ps["PathoGrp"] == k]) for k in ["HS", "FCD", "DNT", "CAV", "DUAL", "Other"]],
        "km_pathology.png", "按病理类型分层的无发作生存曲线（log-rank p 见报告）")
km_plot([(k, ps[ps["Lobe"] == k]) for k in ["颞叶", "额叶", "其他脑叶外"]],
        "km_lobe.png", "按手术部位分层的无发作生存曲线")

# log-rank
def logrank_by(col, order):
    d = prep_km(ps)
    groups = [d[d[col] == k]["duration"] for k in order]
    evs = [d[d[col] == k]["event"] for k in order]
    from lifelines.statistics import multivariate_logrank_test as mlt
    sub = d[d[col].isin(order)]
    r = mlt(sub["duration"], sub[col], sub["event"])
    return round(r.p_value, 4), {k: len(d[d[col] == k]) for k in order}

p_patho, n_patho = logrank_by("PathoGrp", ["HS", "FCD", "DNT", "CAV", "DUAL", "Other"])
p_lobe, n_lobe = logrank_by("Lobe", ["颞叶", "额叶", "其他脑叶外"])

# ---------- 4. 离散时间 Cox ----------
m = long.dropna(subset=["OnsetMid", "SurgAgeMid", "ASM"]).copy()
m["OnsetMid_z"] = (m["OnsetMid"] - m["OnsetMid"].mean()) / m["OnsetMid"].std()
m["SurgAgeMid_z"] = (m["SurgAgeMid"] - m["SurgAgeMid"].mean()) / m["SurgAgeMid"].std()
m["ASM_z"] = (m["ASM"] - m["ASM"].mean()) / m["ASM"].std()
X = pd.get_dummies(m[["PathoGrp", "Lobe"]], drop_first=False)
# 参照组: HS / 颞叶
X = X[["PathoGrp_FCD", "PathoGrp_DNT", "PathoGrp_CAV", "PathoGrp_DUAL", "PathoGrp_Other",
       "Lobe_额叶", "Lobe_其他脑叶外"]]
X.columns = ["FCD(参照HS)", "DNT(参照HS)", "CAV(参照HS)", "双重病理(参照HS)", "其他病理(参照HS)",
             "额叶(参照颞叶)", "其他脑叶外(参照颞叶)"]
X["发病年龄(z)"] = m["OnsetMid_z"]; X["手术年龄(z)"] = m["SurgAgeMid_z"]
X["术前ASM数(z)"] = m["ASM_z"]; X["FBTCS"] = m["FBTCS"]; X["SE史"] = m["SE"]; X["男性"] = m["Male"]
X = X.astype(float)

ctv = CoxTimeVaryingFitter()
ctv.fit(m[["ID", "start", "stop", "event"]].join(X), id_col="ID", event_col="event", start_col="start", stop_col="stop")
summ = ctv.summary
res = pd.DataFrame({"HR": np.exp(summ["coef"]), "CI_low": np.exp(summ["coef lower 95%"]),
                    "CI_high": np.exp(summ["coef upper 95%"]), "p": summ["p"]})
res.to_csv(os.path.join(OUT, "cox_results.csv"), encoding="utf-8-sig")
print("=== 离散时间Cox（至首次复发） ===")
print(res.round(3).to_string())

# GEE 敏感性分析（population-averaged, exchangeable）
try:
    import statsmodels.api as sm
    import patsy
    Xg = sm.add_constant(X.astype(float))
    gee = sm.GEE(m["event"], Xg, groups=m["ID"], family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Exchangeable()).fit()
    gee_res = pd.DataFrame({"OR": np.exp(gee.params), "p": gee.pvalues})
    gee_res.to_csv(os.path.join(OUT, "gee_results.csv"), encoding="utf-8-sig")
    print("=== GEE (OR) ===")
    print(gee_res.round(3).to_string())
except Exception as e:
    print("GEE skipped:", e)
    gee_res = None

# ---------- 5. 森林图 ----------
fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=150)
rr = res.iloc[::-1]
ypos = np.arange(len(rr))
colors = ["#dc2626" if p < 0.05 and h > 1 else ("#059669" if p < 0.05 else "#94a3b8")
          for h, p in zip(rr["HR"], rr["p"])]
ax.hlines(ypos, rr["CI_low"], rr["CI_high"], color=colors, linewidth=2)
ax.scatter(rr["HR"], ypos, color=colors, s=42, zorder=3)
ax.axvline(1, color="#475569", linestyle="--", linewidth=1)
ax.set_yticks(ypos); ax.set_yticklabels(rr.index, fontsize=10)
ax.set_xscale("log"); ax.set_xlim(0.3, 4)
ax.set_xlabel("复发风险比 HR (95% CI, 对数轴)")
ax.set_title("至首次复发的离散时间 Cox 回归", fontsize=13)
for y, (h, lo, hi, p) in enumerate(zip(rr["HR"], rr["CI_low"], rr["CI_high"], rr["p"])):
    ax.text(4.0 / 3.75, y, f"HR={h:.2f}, p={p:.3f}", va="center", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "cox_forest.png")); plt.close(fig)

# ---------- 6. 汇总 JSON ----------
summary = dict(n_total=n_total, n_nofu=n_nofu, n_event=n_event, n_cens=n_cens,
               n_person_periods=int(len(long)),
               yearly_hazard=haz, p_patho=p_patho, p_lobe=p_lobe,
               n_patho=n_patho, n_lobe=n_lobe,
               sf_by_group={})
for col, order in [("PathoGrp", ["HS", "FCD", "DNT", "CAV", "DUAL", "Other"]),
                   ("Lobe", ["颞叶", "额叶", "其他脑叶外"])]:
    for k in order:
        d = prep_km(ps[ps[col] == k])
        if len(d) == 0: continue
        kmf = KaplanMeierFitter().fit(d["duration"], d["event"])
        summary["sf_by_group"][f"{col}:{k}"] = dict(
            n=len(d), sf1=round(float(kmf.predict(1)), 3), sf5=round(float(kmf.predict(5)), 3))
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print("=== 逐年复发风险 ===")
print(haz_df.to_string(index=False))
print(f"log-rank 病理 p={p_patho}, 手术部位 p={p_lobe}")
print("DONE")
