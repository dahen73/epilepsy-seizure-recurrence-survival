# -*- coding: utf-8 -*-
"""
IDEAS (ds005602) 影像增强版离散时间 Cox
在临床基线模型上加入:
  A. 影像 z 分数(ComBat 调和): 全脑/术侧颞叶/对侧颞叶皮层厚度 z、术侧海马体积 z
  B. 切除范围: 术侧颞叶(含海马、杏仁核)切除百分比、术侧海马切除百分比
  C. 切除范围 x log(time) 交互(剂量-时间效应)
模型比较: M0 临床 -> M1 +影像z -> M2 +切除范围 -> M3 +交互
输出: outputs/longitudinal/（可用 IDEAS_OUTPUT_DIR 修改）
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from scipy.stats import chi2

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("IDEAS_DATA_DIR", REPO_ROOT / "data")).resolve()
OUT = Path(os.environ.get("IDEAS_OUTPUT_DIR", REPO_ROOT / "outputs" / "longitudinal")).resolve()
os.makedirs(OUT, exist_ok=True)

TEMPORAL = ["bankssts", "entorhinal", "fusiform", "inferiortemporal",
            "middletemporal", "parahippocampal", "superiortemporal",
            "temporalpole", "transversetemporal"]

# ---------- 1. 影像 z 分数(患者) ----------
tz = pd.read_csv(os.path.join(BASE, "zscores/thick_zscores_patients.csv"))
vz = pd.read_csv(os.path.join(BASE, "zscores/vol_zscores_patients.csv"))
img = tz[["ID"]].copy()
thick_cols = [c for c in tz.columns if c.endswith("_thickness")]
img["ThickZ_global"] = tz[thick_cols].mean(axis=1)

cli = pd.read_csv(os.path.join(BASE, "tables_metadata/Metadata_Release_Anon.csv"), dtype={"ID": int})
cli = cli.drop_duplicates(subset="ID")
side = cli.set_index("ID")["Op_Side"].astype(str).str.upper().str.strip().str[0]
img["side"] = img["ID"].map(side)
img = img[img["side"].isin(["L", "R"])].copy()
img["h_ipse"] = np.where(img["side"] == "L", "lh", "rh")
img["h_contra"] = np.where(img["side"] == "L", "rh", "lh")

def tl_mean(frame_ids, hcol):
    out = []
    for i, h in zip(frame_ids, hcol):
        cols = [f"{h}_{r}_thickness" for r in TEMPORAL]
        out.append(float(tz.loc[tz["ID"] == i, cols].mean(axis=1).iloc[0]))
    return out

img["ThickZ_ipseTL"] = tl_mean(img["ID"], img["h_ipse"])
img["ThickZ_contraTL"] = tl_mean(img["ID"], img["h_contra"])
hip_col = {"L": "Left_Hippocampus_volume", "R": "Right_Hippocampus_volume"}
img["HippZ_ipse"] = [vz.loc[vz["ID"] == i, hip_col[s]].iloc[0]
                     for i, s in zip(img["ID"], img["side"])]

# ---------- 2. 切除范围 ----------
res_tab = pd.read_csv(os.path.join(BASE, "resected_table/table_resected.csv"), index_col=0)
res_tab.columns = [int(float(c)) for c in res_tab.columns]
res_tab.index = [str(x).strip().strip("'").strip('"').lstrip("\ufeff") for x in res_tab.index]

def resect_pct(sid, regions):
    if sid not in res_tab.columns:
        return np.nan
    row = res_tab.loc[res_tab.index.isin(regions), sid]
    return float(pd.to_numeric(row, errors="coerce").mean())

img["Resect_ipseTL"] = [
    resect_pct(i, [f"ctx-{h}-{r}" for r in TEMPORAL] +
               [("Left-Hippocampus" if s == "L" else "Right-Hippocampus"),
                ("Left-Amygdala" if s == "L" else "Right-Amygdala")])
    for i, s, h in zip(img["ID"], img["side"], img["h_ipse"])]
img["Resect_ipseHipp"] = [
    resect_pct(i, ["Left-Hippocampus" if s == "L" else "Right-Hippocampus"])
    for i, s in zip(img["ID"], img["side"])]

img_vars = ["ThickZ_global", "ThickZ_ipseTL", "ThickZ_contraTL", "HippZ_ipse",
            "Resect_ipseTL", "Resect_ipseHipp"]
print("=== 影像预测因子描述 ===")
print(img[img_vars].describe().round(2).to_string())
print("完整病例:", int(img[img_vars].dropna().shape[0]), "/", len(img))

# ---------- 3. 合并 person-period ----------
long = pd.read_csv(os.path.join(OUT, "person_period_long.csv"))
long = long.merge(img[["ID"] + img_vars], on="ID", how="inner")
m = long.dropna(subset=["OnsetMid", "SurgAgeMid", "ASM"] + img_vars).copy()
m["OnsetMid_z"] = (m["OnsetMid"] - m["OnsetMid"].mean()) / m["OnsetMid"].std()
m["SurgAgeMid_z"] = (m["SurgAgeMid"] - m["SurgAgeMid"].mean()) / m["SurgAgeMid"].std()
m["ASM_z"] = (m["ASM"] - m["ASM"].mean()) / m["ASM"].std()
m["logt"] = np.log(m["stop"])
m["Resect_ipseTL_s"] = (m["Resect_ipseTL"] - m["Resect_ipseTL"].mean()) / m["Resect_ipseTL"].std()
m["Resect_x_logt"] = m["Resect_ipseTL_s"] * m["logt"]
print(f"建模样本: {m['ID'].nunique()} 例, {len(m)} 人-年, 事件 {m['event'].sum()}")

CLIN = ["FCD(参照HS)", "DNT(参照HS)", "CAV(参照HS)", "双重病理(参照HS)", "其他病理(参照HS)",
        "额叶(参照颞叶)", "其他脑叶外(参照颞叶)", "发病年龄(z)", "手术年龄(z)", "术前ASM数(z)",
        "FBTCS", "SE史", "男性"]
Xc = pd.get_dummies(m[["PathoGrp", "Lobe"]], drop_first=False)
Xc = Xc[["PathoGrp_FCD", "PathoGrp_DNT", "PathoGrp_CAV", "PathoGrp_DUAL", "PathoGrp_Other",
         "Lobe_额叶", "Lobe_其他脑叶外"]].astype(float)
Xc.columns = CLIN[:7]
Xc["发病年龄(z)"] = m["OnsetMid_z"]; Xc["手术年龄(z)"] = m["SurgAgeMid_z"]
Xc["术前ASM数(z)"] = m["ASM_z"]; Xc["FBTCS"] = m["FBTCS"]; Xc["SE史"] = m["SE"]; Xc["男性"] = m["Male"]
Xc = Xc.astype(float)

IMGZ = {"全脑皮层厚度z": "ThickZ_global", "术侧颞叶厚度z": "ThickZ_ipseTL",
        "对侧颞叶厚度z": "ThickZ_contraTL", "术侧海马体积z": "HippZ_ipse"}
REST = {"术侧颞叶切除范围(z)": "Resect_ipseTL_s"}

def fit_block(extra_dict):
    Xf = Xc.copy()
    for lab, col in extra_dict.items():
        Xf[lab] = m[col].astype(float).values
    base = m[["ID", "start", "stop", "event"]].reset_index(drop=True)
    ctv = CoxTimeVaryingFitter()
    ctv.fit(base.join(Xf.reset_index(drop=True)), id_col="ID", event_col="event",
            start_col="start", stop_col="stop")
    ph = np.asarray(ctv.predict_partial_hazard(Xf)).ravel()
    from lifelines.utils import concordance_index
    cidx = concordance_index(m["stop"].astype(float), -ph, m["event"].astype(int))
    return ctv, Xf, cidx

def cox_table(ctv):
    s = ctv.summary
    return pd.DataFrame({"HR": np.exp(s["coef"]), "CI_low": np.exp(s["coef lower 95%"]),
                         "CI_high": np.exp(s["coef upper 95%"]), "p": s["p"]})

models = {}
models["M0_临床"] = fit_block({})
models["M1_+影像z"] = fit_block(IMGZ)
models["M2_+切除范围"] = fit_block({**IMGZ, **REST})
models["M3_+切除×时间"] = fit_block({**IMGZ, **REST, "术侧颞叶切除×log(时间)": "Resect_x_logt"})

print("\n=== 模型比较 ===")
cmp_rows, prev_ll, prev_k, prev_name = [], None, None, None
for name, (ctv, Xf, cidx) in models.items():
    k = Xf.shape[1]
    ll = ctv.log_likelihood_
    aic = -2 * ll + 2 * k
    lrt_p = np.nan
    if prev_ll is not None:
        stat = 2 * (ll - prev_ll); df_diff = k - prev_k
        lrt_p = float(chi2.sf(stat, df_diff))
    cmp_rows.append(dict(model=name, n=m["ID"].nunique(), k=k, loglik=round(ll, 2),
                         AIC=round(aic, 2), C_index=round(cidx, 3),
                         LRT_vs_prev_p=lrt_p))
    prev_ll, prev_k = ll, k
cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(os.path.join(OUT, "enhanced_model_comparison.csv"), index=False, encoding="utf-8-sig")
print(cmp_df.to_string(index=False))

final_ctv, final_X, final_cidx = models["M3_+切除×时间"]
res = cox_table(final_ctv)
res.to_csv(os.path.join(OUT, "enhanced_cox_results.csv"), encoding="utf-8-sig")
print("\n=== 最终模型 (M3) ===")
print(res.round(3).to_string())

# 切除范围在各自变量下的时变 HR
b_r = float(final_ctv.summary.loc["术侧颞叶切除范围(z)", "coef"])
b_i = float(final_ctv.summary.loc["术侧颞叶切除×log(时间)", "coef"])
hr_curve = pd.DataFrame({"year": [1, 2, 3, 4, 5],
                         "HR": [np.exp(b_r + b_i * np.log(t)) for t in [1, 2, 3, 4, 5]]})
hr_curve.to_csv(os.path.join(OUT, "resect_time_hr.csv"), index=False, encoding="utf-8-sig")
print("\n=== 术侧颞叶切除范围的时变 HR ===")
print(hr_curve.round(3).to_string(index=False))

# ---------- 4s. 补充: HS 亚组(最均质病理组) ----------
mhs = m[m["PathoGrp"] == "HS"].copy().reset_index(drop=True)
Xc_hs = pd.DataFrame(index=mhs.index)
Xc_hs["发病年龄(z)"] = ((mhs["OnsetMid"] - m["OnsetMid"].mean()) / m["OnsetMid"].std()).values
Xc_hs["手术年龄(z)"] = ((mhs["SurgAgeMid"] - m["SurgAgeMid"].mean()) / m["SurgAgeMid"].std()).values
Xc_hs["术前ASM数(z)"] = ((mhs["ASM"] - m["ASM"].mean()) / m["ASM"].std()).values
Xc_hs["FBTCS"] = mhs["FBTCS"].values; Xc_hs["SE史"] = mhs["SE"].values; Xc_hs["男性"] = mhs["Male"].values
Xc_hs = Xc_hs.astype(float)
mhs["logt"] = np.log(mhs["stop"])
mhs["Resect_ipseTL_s"] = (mhs["Resect_ipseTL"] - m["Resect_ipseTL"].mean()) / m["Resect_ipseTL"].std()
mhs["Resect_x_logt"] = mhs["Resect_ipseTL_s"] * mhs["logt"]

def fit_hs(extra):
    Xf = Xc_hs.copy()
    for lab, col in extra.items():
        Xf[lab] = mhs[col].astype(float).values
    base = mhs[["ID", "start", "stop", "event"]].reset_index(drop=True)
    ctv = CoxTimeVaryingFitter()
    ctv.fit(base.join(Xf.reset_index(drop=True)), id_col="ID", event_col="event",
            start_col="start", stop_col="stop")
    ph = np.asarray(ctv.predict_partial_hazard(Xf)).ravel()
    from lifelines.utils import concordance_index
    cidx = concordance_index(mhs["stop"].astype(float), -ph, mhs["event"].astype(int))
    return ctv, Xf, cidx

hs_models = {"HS_M0_临床": fit_hs({}), "HS_M1_+影像z": fit_hs(IMGZ),
             "HS_M2_+切除": fit_hs({**IMGZ, **REST})}
hs_cmp, pll, pk = [], None, None
for name, (ctv, Xf, cidx) in hs_models.items():
    ll, k = ctv.log_likelihood_, Xf.shape[1]
    lrt = np.nan if pll is None else float(chi2.sf(2 * (ll - pll), k - pk))
    hs_cmp.append(dict(model=name, n=int(mhs["ID"].nunique()), loglik=round(ll, 2),
                       C_index=round(cidx, 3), LRT_vs_prev_p=lrt))
    pll, pk = ll, k
hs_cmp_df = pd.DataFrame(hs_cmp)
hs_cmp_df.to_csv(os.path.join(OUT, "hs_subgroup_comparison.csv"), index=False, encoding="utf-8-sig")
hs_ctv, _, _ = hs_models["HS_M2_+切除"]
hs_res = cox_table(hs_ctv)
hs_res.to_csv(os.path.join(OUT, "hs_subgroup_cox.csv"), encoding="utf-8-sig")
print("\n=== HS 亚组(n=%d) ===" % mhs["ID"].nunique())
print(hs_cmp_df.to_string(index=False))
print(hs_res.loc[[i for i in hs_res.index if "z" in i or "切除" in i]].round(3).to_string())

# 影像变量共线性检查
corr = m[["ThickZ_ipseTL", "HippZ_ipse", "ThickZ_global"]].corr().iloc[0, 1]
print(f"\n术侧颞叶厚度z 与 术侧海马体积z 相关系数: {corr:.2f}")

# ---------- 4. 可视化 ----------
# 4a. 最终模型森林图
fig, ax = plt.subplots(figsize=(8.8, 6.6), dpi=150)
rr = res.iloc[::-1]
ypos = np.arange(len(rr))
colors = ["#dc2626" if p < 0.05 and h > 1 else ("#059669" if p < 0.05 else "#94a3b8")
          for h, p in zip(rr["HR"], rr["p"])]
ax.hlines(ypos, rr["CI_low"], rr["CI_high"], color=colors, linewidth=2)
ax.scatter(rr["HR"], ypos, color=colors, s=42, zorder=3)
ax.axvline(1, color="#475569", linestyle="--", linewidth=1)
ax.set_yticks(ypos); ax.set_yticklabels(rr.index, fontsize=9.5)
ax.set_xscale("log"); ax.set_xlim(0.2, 5)
ax.set_xlabel("复发风险比 HR (95% CI, 对数轴)")
ax.set_title("影像增强版离散时间 Cox(最终模型 M3)", fontsize=13)
for y, (h, lo, hi, p) in enumerate(zip(rr["HR"], rr["CI_low"], rr["CI_high"], rr["p"])):
    ax.text(5 / 4.6, y, f"HR={h:.2f}, p={p:.3f}", va="center", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "enh_forest.png")); plt.close(fig)

# 4b. 模型比较(C-index 柱状 + AIC)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=150)
names = [r["model"] for r in cmp_rows]
cidx = [r["C_index"] for r in cmp_rows]
a1.bar(range(len(names)), cidx, color=["#94a3b8", "#60a5fa", "#2563eb", "#1e3a8a"])
a1.set_xticks(range(len(names))); a1.set_xticklabels(names, fontsize=9, rotation=12)
a1.set_ylim(0.5, max(cidx) + 0.03); a1.set_ylabel("一致性指数 C-index")
a1.set_title("预测判别力逐步提升", fontsize=12)
for i, c in enumerate(cidx):
    a1.text(i, c + 0.004, f"{c:.3f}", ha="center", fontsize=9.5)
aics = [r["AIC"] for r in cmp_rows]
a2.bar(range(len(names)), aics, color=["#94a3b8", "#60a5fa", "#2563eb", "#1e3a8a"])
a2.set_xticks(range(len(names))); a2.set_xticklabels(names, fontsize=9, rotation=12)
a2.set_ylabel("AIC (越低越好)"); a2.set_title("模型简约性比较", fontsize=12)
for i, a in enumerate(aics):
    a2.text(i, a + 2, f"{a:.0f}", ha="center", fontsize=9.5)
fig.suptitle("模型比较(同一完整病例样本, 似然比检验见报告)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "enh_model_compare.png")); plt.close(fig)

# 4c. 按术侧颞叶切除范围分组的 KM
ps_img = img.dropna(subset=["Resect_ipseTL"]).copy()
ps_img = ps_img.merge(cli[["ID"]], on="ID", how="left")
fu = pd.read_csv(os.path.join(OUT, "person_period_long.csv"))
pat = fu.groupby("ID").agg(stop_max=("stop", "max"), any_event=("event", "max")).reset_index()
pat.columns = ["ID", "n_fu", "any_event"]
ps_img = ps_img.merge(pat, on="ID", how="inner")
ps_img["event_year"] = np.where(ps_img["any_event"] == 1, ps_img["n_fu"], np.nan)
ps_img["duration"] = ps_img["event_year"].fillna(ps_img["n_fu"])
ps_img["event"] = ps_img["any_event"].astype(int)
ps_img["grp"] = pd.qcut(ps_img["Resect_ipseTL"], 3, labels=["低切除(下1/3)", "中切除", "高切除(上1/3)"])
fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=150)
for name, sub in ps_img.groupby("grp", observed=True):
    kmf = KaplanMeierFitter(label=f"{name} (n={len(sub)})")
    kmf.fit(sub["duration"], sub["event"])
    kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=2)
ax.set_xlim(0, 5); ax.set_xticks(range(0, 6)); ax.set_ylim(0, 1.02)
ax.set_xlabel("术后年数"); ax.set_ylabel("持续无发作概率 (ILAE=1)")
ax.set_title("按术侧颞叶切除范围三分位的无发作生存曲线", fontsize=13)
ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "enh_km_resect.png")); plt.close(fig)

# 4d. 切除范围×时间交互: 时变 HR 曲线
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
ts = np.linspace(0.5, 5, 60)
hrs = np.exp(b_r + b_i * np.log(ts))
ax.plot(ts, hrs, color="#2563eb", linewidth=2.4)
ax.axhline(1, color="#475569", linestyle="--", linewidth=1)
ax.set_xticks([0.5, 1, 2, 3, 4, 5])
ax.set_xlabel("术后时间 (年)"); ax.set_ylabel("每+1 SD 术侧颞叶切除范围的瞬时复发 HR")
ax.set_title(f"切除范围的时间依赖效应 (交互项 β={b_i:.3f}, p={res.loc['术侧颞叶切除×log(时间)', 'p']:.3f})", fontsize=12)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "enh_resect_time.png")); plt.close(fig)

# ---------- 5. 汇总 ----------
summary = dict(
    n_analytic=int(m["ID"].nunique()), n_person_periods=int(len(m)),
    n_events=int(m["event"].sum()),
    model_comparison=cmp_rows,
    resect_hr_by_year=hr_curve.to_dict("records"),
    interaction_coef=dict(beta=b_i, p=float(res.loc["术侧颞叶切除×log(时间)", "p"])),
    img_complete=int(img[img_vars].dropna().shape[0]), img_total=len(img))
with open(os.path.join(OUT, "enhanced_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print("DONE")
