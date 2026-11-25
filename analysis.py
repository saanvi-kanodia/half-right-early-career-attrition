"""
Early-career attrition survey: single reproducible analysis.

Replaces eda.py, PCA.py, PCA_likert.py, factor_analysis.py, logistic.py,
enhanced_analysis.py and research_project/analysis_code.ipynb.

Usage:
    .venv/bin/python analysis.py [path/to/survey_data.csv]

Writes all tables to results_clean/report.txt and figures to results_clean/.

Key corrections vs. the earlier scripts:
  * Age groups are mapped explicitly (LabelEncoder sorted them alphabetically,
    putting 37-42 into "Early-Career" and <25 into "Senior").
  * Outcome items (Intent_to_Stay, Job_Search) are excluded from the PCA that
    feeds the regression.
  * Predictive performance is leave-one-out cross-validated, not in-sample.
  * KMO, Bartlett and a proper parallel analysis are reported.
  * Group comparisons use Mann-Whitney U with Holm correction (ordinal data).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

HERE = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "survey_data_anonymized.csv"
OUT = HERE / "results_clean"
OUT.mkdir(exist_ok=True)
SEED = 0

# --------------------------------------------------------------------------
# 1. Load and rename (match on question-text prefixes, robust to dash variants)
# --------------------------------------------------------------------------
ITEMS = {
    "My manager gives clear goals": "Clarity",
    "I receive helpful coaching": "Feedback",
    "I can usually predict my work": "Predictability",
    "Last-minute changes": "Unpredictability",
    "I experience intense workload": "Workload_Spikes",
    "I have autonomy to choose": "Autonomy",
    "I have adequate flexibility": "Flexibility",
    "When I request flexibility": "Flexibility_Approval",
    "Performance evaluations": "Fair_Evaluation",
    "My total compensation": "Fair_Compensation",
    "Incentive or target rules": "Incentive_Clarity",
    "Good performance is recognized": "Recognition",
    "I often feel emotionally exhausted": "Burnout",
    "My current job allows me": "WLB",
    "I expect to be working": "Intent_to_Stay",
    "In the past month I have actively": "Job_Search",
    "your colleagues have actively": "Colleague_Influence",
}
META = {
    "Gender": "Gender",
    "Age group": "Age",
    "Years of total work experience": "Experience",
    "Tenure at current company": "Tenure",
    "Average weekly hours": "Avg_Hours",
    "Total daily commute": "Commute_Min",
    "What were your main reasons": "Reasons_Leaving",
}

raw = pd.read_csv(DATA)


def find(prefix):
    hits = [c for c in raw.columns if c.strip().lower().startswith(prefix.lower())]
    if len(hits) != 1:
        raise KeyError(f"Expected one column starting with {prefix!r}, found {hits}")
    return hits[0]


df = pd.DataFrame({new: raw[find(p)] for p, new in {**META, **ITEMS}.items()})
LIKERT = list(ITEMS.values())
df[LIKERT] = df[LIKERT].apply(pd.to_numeric, errors="coerce")
df["Experience"] = pd.to_numeric(df["Experience"].replace({"< 1 year": "0.5"}), errors="coerce")
for c in ["Tenure", "Avg_Hours", "Commute_Min"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Explicit age mapping (fixes the LabelEncoder bug in eda.py)
EARLY_AGES = ["<25", "25-30", "31-36"]
df["Age"] = df["Age"].astype(str).str.strip()
df["Early_Career"] = df["Age"].isin(EARLY_AGES).astype(int)
df["Exp_lt5"] = (df["Experience"] < 5).astype(int)
df["High_Risk"] = (df["Intent_to_Stay"] <= 2).astype(int)

OUTCOMES = ["Intent_to_Stay", "Job_Search"]
PREDICTORS = [c for c in LIKERT if c not in OUTCOMES + ["Colleague_Influence"]]

lines = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def section(t):
    out("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------
# 2. Sample description
# --------------------------------------------------------------------------
section("1. SAMPLE")
out(f"N = {len(df)} (Likert items complete: {df[LIKERT].notna().all(axis=1).sum()})")
out("Gender:", df["Gender"].value_counts().to_dict())
out("Age:", df["Age"].value_counts().reindex(EARLY_AGES + ["37-42", "43-48", "49+"]).to_dict())
out(f"Age <=36: {df.Early_Career.sum()}  |  Experience <5 yrs: {df.Exp_lt5.sum()}  |  both: {(df.Early_Career & df.Exp_lt5).sum()}")
out("\nContinuous variables:\n" + df[["Avg_Hours", "Commute_Min", "Tenure", "Experience"]].describe().round(2).to_string())
out("\nLikert items (mean, SD, median):\n" + df[LIKERT].agg(["mean", "std", "median"]).T.round(2).to_string())

reasons = df["Reasons_Leaving"].dropna().str.split(r",\s*(?=[A-Z])", regex=True).explode().str.strip()
reasons = reasons[reasons != ""]
rt = pd.DataFrame({"n": reasons.value_counts()})
rt["% of respondents"] = (rt.n / df["Reasons_Leaving"].notna().sum() * 100).round(1)
rt["% of mentions"] = (rt.n / rt.n.sum() * 100).round(1)
out("\nReasons for leaving (multi-select):\n" + rt.head(8).to_string())

# --------------------------------------------------------------------------
# 3. Reliability
# --------------------------------------------------------------------------
section("2. RELIABILITY (Cronbach's alpha)")


def alpha(X):
    X = np.asarray(X, float)
    k = X.shape[1]
    return k / (k - 1) * (1 - X.var(0, ddof=1).sum() / X.sum(1).var(ddof=1))


SCALES = {
    "Managerial Quality": ["Clarity", "Feedback", "Fair_Evaluation"],
    "Workload Pressure (incl. Burnout)": ["Unpredictability", "Workload_Spikes", "Burnout"],
    "Workload Pressure (excl. Burnout)": ["Unpredictability", "Workload_Spikes"],
    "Job Conditions": ["Flexibility", "Flexibility_Approval", "Fair_Compensation",
                       "Incentive_Clarity", "Recognition", "Autonomy"],
}
for name, cols in SCALES.items():
    out(f"{name:38s} k={len(cols)}  alpha={alpha(df[cols].dropna()):.3f}")

# --------------------------------------------------------------------------
# 4. Factorability, parallel analysis, PCA (predictors only)
# --------------------------------------------------------------------------
section("3. DIMENSION REDUCTION (predictor items only; outcomes excluded)")
d = df.dropna(subset=PREDICTORS + ["Intent_to_Stay"]).reset_index(drop=True)
Z = StandardScaler().fit_transform(d[PREDICTORS])
kmo = calculate_kmo(Z)[1]
chi2, bp = calculate_bartlett_sphericity(Z)
out(f"KMO = {kmo:.3f}   Bartlett chi2 = {chi2:.1f}, p = {bp:.2e}   (items={len(PREDICTORS)}, N={len(d)})")

ev = np.linalg.eigvalsh(np.corrcoef(Z.T))[::-1]
rng = np.random.default_rng(SEED)
rand = np.array([np.linalg.eigvalsh(np.corrcoef(rng.normal(size=Z.shape).T))[::-1] for _ in range(1000)])
thr = np.percentile(rand, 95, axis=0)
out(f"Eigenvalues: {np.round(ev[:6], 2).tolist()}")
out(f"Kaiser (>1) retains {int((ev > 1).sum())}; parallel analysis (95th pct, 1000 sims) retains {int((ev > thr).sum())}")

pca = PCA(n_components=3).fit(Z)
scores = pca.transform(Z)
# Orient components so Factor 1 = higher managerial quality, Factor 2 = higher workload
signs = np.sign([pca.components_[0][PREDICTORS.index("Clarity")],
                 pca.components_[1][PREDICTORS.index("Unpredictability")],
                 pca.components_[2][PREDICTORS.index("Burnout")]])
scores = scores * signs
loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_) * signs,
                        index=PREDICTORS, columns=["F1", "F2", "F3"])
out("Explained variance:", np.round(pca.explained_variance_ratio_, 3).tolist())
out("\nLoadings (component x sqrt(eigenvalue)):\n" + loadings.round(2).to_string())
for i in range(3):
    d[f"F{i+1}"] = scores[:, i]

# --------------------------------------------------------------------------
# 5. Age-group comparisons
# --------------------------------------------------------------------------
section("4. AGE-GROUP COMPARISONS (<=36 vs 37+)")
wp_cols = SCALES["Workload Pressure (excl. Burnout)"]
d["Workload_Raw"] = d[wp_cols].mean(1)
e, s = d[d.Early_Career == 1], d[d.Early_Career == 0]
out(f"n early = {len(e)}, n senior = {len(s)}")
for label, col in [("Workload composite (Unpred+Spikes)", "Workload_Raw"), ("Workload factor score (F2)", "F2")]:
    t = stats.ttest_ind(e[col], s[col], equal_var=False)
    u = stats.mannwhitneyu(e[col], s[col])
    pooled = np.sqrt((e[col].var(ddof=1) + s[col].var(ddof=1)) / 2)
    dd = (e[col].mean() - s[col].mean()) / pooled
    diff = e[col].mean() - s[col].mean()
    se = np.sqrt(e[col].var(ddof=1) / len(e) + s[col].var(ddof=1) / len(s))
    ci = stats.t.ppf(.975, t.df) * se
    out(f"{label}: early M={e[col].mean():.2f} (SD {e[col].std():.2f}), senior M={s[col].mean():.2f} (SD {s[col].std():.2f}); "
        f"Welch t({t.df:.1f})={t.statistic:.3f}, p={t.pvalue:.3f}; diff 95% CI [{diff-ci:.2f}, {diff+ci:.2f}]; "
        f"d={dd:.2f}; Mann-Whitney p={u.pvalue:.3f}")

rows = []
for v in LIKERT:
    u = stats.mannwhitneyu(e[v], s[v])
    rows.append((v, e[v].mean(), s[v].mean(), e[v].median(), s[v].median(), u.pvalue))
cmp_ = pd.DataFrame(rows, columns=["item", "mean_early", "mean_senior", "med_early", "med_senior", "p"])
cmp_["p_holm"] = multipletests(cmp_.p, method="holm")[1]
out("\nAll items, Mann-Whitney U with Holm correction:\n" + cmp_.round(3).to_string(index=False))

ct = pd.crosstab(d.Early_Career, d.High_Risk)
orr, fp = stats.fisher_exact(ct)
out(f"\nHigh attrition risk (Intent_to_Stay <= 2) by age group:\n{ct.rename(index={0:'Senior',1:'Early'}, columns={0:'Low',1:'High'}).to_string()}")
out(f"Fisher exact: OR = {orr:.2f}, p = {fp:.3f}")
ct2 = pd.crosstab(d.Exp_lt5, d.High_Risk)
or2, fp2 = stats.fisher_exact(ct2)
out(f"Same by experience <5 yrs: OR = {or2:.2f}, p = {fp2:.3f}  (table {ct2.values.tolist()})")

# --------------------------------------------------------------------------
# 6. Logistic regression
# --------------------------------------------------------------------------
section("5. LOGISTIC REGRESSION: High_Risk ~ F1 + F2 + F3 (factors from predictors only)")
y = d.High_Risk
out(f"Events: {int(y.sum())}/{len(y)}  (events per predictor = {y.sum()/3:.1f}; <10 means estimates are unstable)")
X = sm.add_constant(d[["F1", "F2", "F3"]].apply(lambda c: (c - c.mean()) / c.std()))
m = sm.Logit(y, X).fit(disp=0)
tab = pd.DataFrame({"coef": m.params, "OR": np.exp(m.params),
                    "OR_lo": np.exp(m.conf_int()[0]), "OR_hi": np.exp(m.conf_int()[1]), "p": m.pvalues})
out(tab.round(3).to_string())
out(f"McFadden pseudo-R2 = {m.prsquared:.3f}; LR chi2({int(m.df_model)}) = {2*(m.llf-m.llnull):.2f}, p = {m.llr_pvalue:.2e}")
out(f"F1 per-SD change in odds: {(np.exp(m.params['F1'])-1)*100:.1f}%")

Xa = sm.add_constant(pd.concat([X.drop(columns="const"), d.Early_Career], axis=1))
ma = sm.Logit(y, Xa).fit(disp=0)
m0 = sm.Logit(y, sm.add_constant(d.Early_Career)).fit(disp=0)
out(f"\nDoes F1 account for the age gap?  Age OR alone = {np.exp(m0.params['Early_Career']):.2f} (p={m0.pvalues['Early_Career']:.3f}); "
    f"Age OR controlling F1-F3 = {np.exp(ma.params['Early_Career']):.2f} (p={ma.pvalues['Early_Career']:.3f}); "
    f"F1 OR with age = {np.exp(ma.params['F1']):.2f} (p={ma.pvalues['F1']:.3f})")

out("\nOut-of-sample performance (leave-one-out CV; scaler + PCA refit inside each fold):")
pipe = make_pipeline(StandardScaler(), PCA(n_components=3), LogisticRegression(C=1e6, max_iter=5000))
prob = cross_val_predict(pipe, d[PREDICTORS], y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
pred = (prob >= .5).astype(int)
out(f"AUC={roc_auc_score(y, prob):.3f}  Acc={accuracy_score(y, pred):.3f}  Prec={precision_score(y, pred):.3f}  "
    f"Rec={recall_score(y, pred):.3f}  F1={f1_score(y, pred):.3f}")
out("Confusion matrix [[TN FP],[FN TP]]:", confusion_matrix(y, pred).tolist())
ins = m.predict(X)
out(f"(In-sample, for comparison only: AUC={roc_auc_score(y, ins):.3f}, Acc={accuracy_score(y, ins>=.5):.3f})")

# --------------------------------------------------------------------------
# 7. Figures
# --------------------------------------------------------------------------
def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


fig, axes = plt.subplots(4, 5, figsize=(15, 11))
for ax, v in zip(axes.flat, LIKERT):
    counts = d[v].value_counts().reindex(range(1, 6), fill_value=0)
    ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.set_title(v, fontsize=10)
    ax.set_xticks(range(1, 6))
for ax in list(axes.flat)[len(LIKERT):]:
    ax.axis("off")
fig.suptitle(f"Distribution of Likert items (N={len(d)}; 1=Strongly disagree, 5=Strongly agree)")
save(fig, "fig1_likert_distributions.png")

fig, axes = plt.subplots(4, 5, figsize=(15, 11))
for ax, v in zip(axes.flat, LIKERT):
    ax.boxplot([s[v], e[v]], tick_labels=[f"37+\n(n={len(s)})", f"<=36\n(n={len(e)})"])
    p = cmp_.set_index("item").loc[v, "p_holm"]
    ax.set_title(f"{v}\nHolm p={p:.2f}", fontsize=9)
    ax.set_ylim(0.5, 5.5)
for ax in list(axes.flat)[len(LIKERT):]:
    ax.axis("off")
fig.suptitle("Likert items by age group (explicit mapping: <25, 25-30, 31-36 = early)")
save(fig, "fig2_items_by_age.png")

C = d[LIKERT].corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(LIKERT)), LIKERT, rotation=45, ha="right")
ax.set_yticks(range(len(LIKERT)), LIKERT)
for i in range(len(LIKERT)):
    for j in range(len(LIKERT)):
        ax.text(j, i, f"{C.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im)
ax.set_title("Pearson correlations between Likert items")
save(fig, "fig3_correlations.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
k = np.arange(1, len(ev) + 1)
ax.plot(k, ev, "o-", label="Observed")
ax.plot(k, thr, "s--", label="Random (95th pct)")
ax.axhline(1, color="grey", lw=.8)
ax.set_xlabel("Component")
ax.set_ylabel("Eigenvalue")
ax.set_title("Scree plot with parallel analysis")
ax.legend()
save(fig, "fig4_scree_parallel.png")

fig, ax = plt.subplots(figsize=(6, 7))
im = ax.imshow(loadings, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
ax.set_yticks(range(len(PREDICTORS)), PREDICTORS)
ax.set_xticks(range(3), ["F1 Managerial/Job", "F2 Workload", "F3 Stress"])
for i in range(len(PREDICTORS)):
    for j in range(3):
        ax.text(j, i, f"{loadings.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("PCA loadings (predictor items only)")
save(fig, "fig5_loadings.png")

rate = d.groupby("Early_Career").High_Risk.agg(["mean", "sum", "count"])
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.bar(["37+", "<=36"], rate["mean"].reindex([0, 1]) * 100, color=["#4C72B0", "#DD8452"])
for i, g in enumerate([0, 1]):
    ax.text(i, rate.loc[g, "mean"] * 100 + 1, f"{int(rate.loc[g,'sum'])}/{int(rate.loc[g,'count'])}", ha="center")
ax.set_ylabel("% high attrition risk")
ax.set_title(f"High attrition risk by age group\nFisher exact OR={orr:.1f}, p={fp:.3f}")
save(fig, "fig6_risk_by_age.png")

# Appendix B: sample description charts
def hbar(series, title, xlabel, name, pct_of=None):
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(s) + 1.2))
    ax.barh(s.index, s.values, color="#4C72B0")
    for y, v in enumerate(s.values):
        lab = f"{v} ({v / pct_of * 100:.0f}%)" if pct_of else str(v)
        ax.text(v + 0.2, y, lab, va="center", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, s.max() * 1.2)
    save(fig, name)


age_order = ["<25", "25-30", "31-36", "37-42", "43-48", "49+"]
ages = df["Age"].value_counts().reindex(age_order, fill_value=0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(ages.index, ages.values, color=["#DD8452"] * 3 + ["#4C72B0"] * 3)
for x, v in enumerate(ages.values):
    ax.text(x, v + 0.2, str(v), ha="center")
ax.set_xlabel("Age group (orange = early-career, 36 or under)")
ax.set_ylabel("Respondents")
ax.set_title(f"Age distribution of survey respondents (N = {len(df)})")
save(fig, "figB1_age.png")

ind = raw[find("Which industry")].astype(str).str.strip()
ind = ind.where(ind.map(ind.value_counts()) >= 2, "Other (1 respondent each)")
hbar(ind.value_counts(), f"Industry of current employer (N = {len(df)})", "Respondents", "figB2_industry.png", len(df))
hbar(rt["n"][rt["n"] >= 2], f"Reasons for leaving most recent job (multi-select, N = {len(df)})",
     "Respondents selecting reason", "figB3_reasons.png", len(df))

(OUT / "report.txt").write_text("\n".join(lines))
d.to_csv(OUT / "factor_scores.csv", index=False, columns=["Age", "Early_Career", "High_Risk", "F1", "F2", "F3"])
print(f"\nSaved report and figures to {OUT}")
