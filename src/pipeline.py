"""
Corroded-steel residual yield force pipeline — deterministic seeds, lean code.
Target: ln(Fy). Compares a code-style linear capacity law against ML models,
and isolates pitting-notch effects from uniform mass loss.
ponytail: no custom classes, no registry, single file.
"""
import json, time, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, Matern, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb, lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
np.random.seed(42)
ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
LAST = {}

CAT_COLS = ["Specimen_Type", "Load_History", "Bar_Type", "Steel_Origin",
            "Bar_Grade", "Corrosion_Method", "Cathode_Type", "Cleaning_Method"]
NUM_COLS = ["Nominal Diameter", "Gauge Length", "Current Density",
            "Exposure Duration", "Solution Concentration",
            "Mass Loss", "Corroded Length", "Corrosion Penetration Depth"]
PIT_FAMILY = ["Corrosion Penetration Depth", "pit_depth_ratio",
              "notch_severity", "log_current", "log_exposure"]

# ── data ─────────────────────────────────────────────────────────────────────
def load_data(path=None):
    path = path or ROOT / "data" / "Data.csv"
    # keep_default_na=False: 'None' (load history) and 'N/A' (cleaning)
    # are real category labels, not missing values
    df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"No.": "No", "Author, Specimen Name": "Author_Spec",
                            "ln(Fy)": "lnFy"})
    df["Author"] = df["Author_Spec"].str.split(",").str[0].str.strip()
    for c in CAT_COLS:
        df[c] = df[c].astype(str).str.strip().replace("", "Missing")
    for c in NUM_COLS + ["Fy"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce").fillna(0)
    return df

def build_features(df):
    """Numericals + ordinal categoricals + fracture-flavoured engineered terms."""
    X = df[NUM_COLS].copy().astype(float)
    D = X["Nominal Diameter"].clip(lower=1e-6)
    eta = (X["Mass Loss"] / 100.0).clip(0, 0.95)
    depth = X["Corrosion Penetration Depth"].clip(lower=0)
    X["eta"] = eta
    X["area_unif"] = 1.0 - eta                       # uniform-thinning area fraction
    X["pit_depth_ratio"] = depth / D                 # pit depth over bar diameter
    X["notch_severity"] = (depth / D) / np.sqrt(eta + 0.005)  # deep pit at low mass loss = sharp notch
    X["log_current"] = np.log1p(X["Current Density"].clip(lower=0))
    X["log_exposure"] = np.log1p(X["Exposure Duration"].clip(lower=0))
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[CAT_COLS] = enc.fit_transform(df[CAT_COLS].astype(str))
    return X

# ── code-style baseline: Fy = Fy0 * (1 - beta * eta) ─────────────────────────
def code_baseline_predict(df_train, df_test):
    """Fy0 = median control Fy per (grade, diameter); one global beta by least squares."""
    ctrl = df_train[df_train["Mass Loss"] == 0]
    fy0_map = ctrl.groupby(["Bar_Grade", "Nominal Diameter"])["Fy"].median()
    fy0_fallback = ctrl["Fy"].median() if len(ctrl) else df_train["Fy"].median()
    def fy0_of(df):
        key = list(zip(df["Bar_Grade"], df["Nominal Diameter"]))
        return np.array([fy0_map.get(k, fy0_fallback) for k in key], dtype=float)
    eta_tr = (df_train["Mass Loss"] / 100.0).clip(0, 0.95).values
    r_tr = 1.0 - df_train["Fy"].values / fy0_of(df_train)
    beta = float(np.dot(eta_tr, r_tr) / max(np.dot(eta_tr, eta_tr), 1e-9))
    beta = float(np.clip(beta, 0.2, 3.0))
    eta_te = (df_test["Mass Loss"] / 100.0).clip(0, 0.95).values
    fy_pred = fy0_of(df_test) * np.clip(1 - beta * eta_te, 0.05, None)
    return np.log(fy_pred), beta

# ── models ───────────────────────────────────────────────────────────────────
def make_gpr(n_feat):
    # isotropic composite kernel: 5 hyperparameters, fits in seconds.
    # (ARD with 29 dims took 7+ min per fit on 1200 points: measured, cut.)
    k = (C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
         + C(1.0, (1e-2, 1e2)) * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
         + WhiteKernel(1e-3))
    return Pipeline([("sc", StandardScaler()),
                     ("gpr", GaussianProcessRegressor(kernel=k, normalize_y=True,
                                                      n_restarts_optimizer=2, random_state=42))])

def make_models(X_tr, y_tr, X_val, y_val):
    models = {}
    models["MLR"] = LinearRegression().fit(X_tr, y_tr)
    models["RF"] = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                         n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    mc = {c: (0 if c not in ("Mass Loss", "eta") else -1) for c in X_tr.columns}
    models["XGBoost"] = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=1500,
                                         subsample=0.8, colsample_bytree=0.8,
                                         monotone_constraints=mc, random_state=42,
                                         verbosity=0, early_stopping_rounds=50).fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    lgbm = lgb.LGBMRegressor(num_leaves=63, learning_rate=0.05, n_estimators=1500,
                             min_child_samples=10, random_state=42, verbose=-1)
    # LightGBM moved this argument twice: 4.7 wants eval_X/eval_y, older releases
    # only accept eval_set. Try both so the pipeline runs on any recent version.
    try:
        models["LightGBM"] = lgbm.fit(
            X_tr, y_tr, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
    except TypeError:
        models["LightGBM"] = lgbm.fit(
            X_tr, y_tr, eval_X=X_val, eval_y=y_val,
            callbacks=[lgb.early_stopping(50, verbose=False)])
    models["MLP"] = Pipeline([("sc", StandardScaler()),
                              ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.01,
                                                   max_iter=2000, random_state=42))]).fit(X_tr, y_tr)
    models["GPR"] = make_gpr(X_tr.shape[1]).fit(X_tr.values, y_tr.values)
    return models

# ── metrics ──────────────────────────────────────────────────────────────────
def reg_metrics(y_true, y_pred):
    fy_t, fy_p = np.exp(y_true), np.exp(y_pred)
    return dict(R2=round(r2_score(y_true, y_pred), 4),
                RMSE=round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
                MAE=round(float(mean_absolute_error(y_true, y_pred)), 4),
                MAPE=round(float(np.mean(np.abs(fy_t - fy_p) / fy_t) * 100), 2))

def prob_metrics(y_true, y_pred, y_std, z=1.96):
    lo, hi = y_pred - z * y_std, y_pred + z * y_std
    return dict(PICP95=round(float(np.mean((y_true >= lo) & (y_true <= hi))), 4),
                MPIW95=round(float(np.mean(hi - lo)), 4))

def cv_table(X, y, n_splits=10):
    """Single-repeat 10-fold CV, mean ± std of R2/RMSE per model (fast; source paper used 100-fold)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc = {}
    for tr, te in kf.split(X):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        Xtr2, Xva, ytr2, yva = train_test_split(Xtr, ytr, test_size=0.15, random_state=42)
        ms = make_models(Xtr2, ytr2, Xva, yva)
        for name, m in ms.items():
            p = m.predict(Xte.values) if name == "GPR" else m.predict(Xte)
            acc.setdefault(name, []).append([r2_score(yte, p),
                                             np.sqrt(mean_squared_error(yte, p))])
    return {k: {"R2_mean": round(float(np.mean([a[0] for a in v])), 4),
                "R2_std": round(float(np.std([a[0] for a in v])), 4),
                "RMSE_mean": round(float(np.mean([a[1] for a in v])), 4),
                "RMSE_std": round(float(np.std([a[1] for a in v])), 4)}
            for k, v in acc.items()}

# ── figures ──────────────────────────────────────────────────────────────────
def parity_plot(y_test, preds):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    lim = (float(y_test.min()) - 0.2, float(y_test.max()) + 0.2)
    for ax, (name, pr) in zip(axes, preds.items()):
        ax.scatter(y_test, pr, s=10, alpha=0.5)
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_title(f"{name}  (R2={r2_score(y_test, pr):.3f})", fontsize=11)
        ax.set_xlabel("observed ln(Fy)")
    axes[0].set_ylabel("predicted ln(Fy)")
    fig.suptitle("Parity on the held-out test set")
    fig.tight_layout(); fig.savefig(FIG / "parity.png", dpi=150)
    return fig

def notch_residual_plot(df_test, code_resid, xgb_resid):
    """The money plot: code-law residuals grow with pit depth; XGB residuals do not."""
    depth = df_test["Corrosion Penetration Depth"].values
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, resid, name in zip(axes, [code_resid, xgb_resid],
                               ["Code law  Fy0(1-beta eta)", "XGBoost (full features)"]):
        hb = ax.hexbin(depth, resid, gridsize=30, cmap="Blues", mincnt=1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("corrosion penetration depth [mm]")
        ax.set_title(name)
    axes[0].set_ylabel("residual ln(Fy), observed minus predicted")
    fig.colorbar(hb, ax=axes, label="count")
    fig.suptitle("Residuals against pit depth: what the linear code law misses, the ML model absorbs")
    fig.tight_layout(); fig.savefig(FIG / "notch_residuals.png", dpi=150)
    return fig

def pd_depth_plot(model, X_train, mass_levels=(2, 8, 20)):
    """Partial dependence on penetration depth at fixed mass-loss levels."""
    med = X_train.median()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    grid = np.linspace(0, X_train["Corrosion Penetration Depth"].quantile(0.98), 60)
    for ml in mass_levels:
        Xp = pd.DataFrame([med] * len(grid))
        Xp["Corrosion Penetration Depth"] = grid
        Xp["Mass Loss"] = ml
        eta = np.clip(ml / 100, 0, 0.95)
        D = Xp["Nominal Diameter"].clip(lower=1e-6)
        Xp["eta"] = eta
        Xp["area_unif"] = 1 - eta
        Xp["pit_depth_ratio"] = grid / D.values
        Xp["notch_severity"] = (grid / D.values) / np.sqrt(eta + 0.005)
        ax.plot(grid, model.predict(Xp), label=f"mass loss {ml}%")
    ax.set_xlabel("penetration depth [mm]")
    ax.set_ylabel("predicted ln(Fy)")
    ax.legend(); ax.set_title("Same mass loss, deeper pit, lower strength")
    fig.tight_layout(); fig.savefig(FIG / "pd_depth.png", dpi=150)
    return fig

def calibration_plot(y_test, mu, sd):
    from scipy.stats import norm
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    zs = np.linspace(0.5, 2.5, 21)
    cov = [np.mean(np.abs(np.asarray(y_test) - mu) <= z * sd) for z in zs]
    ax.plot(2 * (1 - norm.cdf(zs)), cov, "o-", ms=4, label="GPR")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.legend(); ax.set_title("Predictive interval calibration, GPR")
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=150)
    return fig

def shap_plots(model, X_test, names):
    expl = shap.TreeExplainer(model)
    sv = expl(X_test)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, show=False, max_display=14)
    plt.tight_layout(); plt.savefig(FIG / "shap_xgb.png", dpi=150); plt.close()
    for feat in ("Mass Loss", "Corrosion Penetration Depth", "notch_severity"):
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.dependence_plot(feat, sv.values, X_test, show=False, ax=ax)
        plt.tight_layout()
        plt.savefig(FIG / f"shap_dep_{feat[:8]}.png", dpi=150); plt.close()
    vals = np.abs(sv.values).mean(0)
    order = np.argsort(vals)[::-1]
    return [(names[i], round(float(vals[i]), 4)) for i in order[:14]]

def print_table(results):
    cols = ["R2", "RMSE", "MAE", "MAPE"]
    header = f"{'Model':<28}" + "".join(f"{c:>10}" for c in cols)
    sep = "-" * len(header)
    print(sep); print(header); print(sep)
    for name, m in results.items():
        print(f"{name:<28}" + "".join(f"{m.get(c, ''):>10}" for c in cols))
    print(sep)

# ── full pipeline ────────────────────────────────────────────────────────────
def run_pipeline(df=None):
    t0 = time.time()
    if df is None:
        df = load_data()
    X = build_features(df)
    y = np.log(df["Fy"].astype(float))
    all_results = {}

    print("10-fold CV over all models ...")
    all_results["cv"] = cv_table(X, y)

    X_train, X_test, df_train, df_test = train_test_split(
        X, df, test_size=0.2, random_state=42)
    y_train = np.log(df_train["Fy"].astype(float))
    y_test = np.log(df_test["Fy"].astype(float))
    Xtr, Xval, ytr, yval = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    print("fitting all models on the train split ...")
    models = make_models(Xtr, ytr, Xval, yval)
    preds = {n: (m.predict(X_test.values) if n == "GPR" else m.predict(X_test))
             for n, m in models.items()}

    print("code-law baseline ...")
    code_pred, beta = code_baseline_predict(df_train, df_test)
    all_results["code_beta"] = round(beta, 4)
    preds["CodeLaw"] = code_pred

    for name, pr in preds.items():
        all_results[name] = reg_metrics(y_test.values, pr)
    gpr_mu = preds["GPR"]
    _, gpr_sd = models["GPR"].predict(X_test.values, return_std=True)
    all_results["GPR"].update(prob_metrics(y_test.values, gpr_mu, gpr_sd))

    print("pitting ablation: XGBoost with and without the pit-depth family ...")
    keep = [c for c in X.columns if c not in PIT_FAMILY]
    m_nopit = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=1500,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbosity=0,
                               early_stopping_rounds=50).fit(
        Xtr[keep], ytr, eval_set=[(Xval[keep], yval)], verbose=False)
    all_results["XGBoost_no_pit"] = reg_metrics(y_test.values, m_nopit.predict(X_test[keep]))

    print("campaign-grouped split: 20 campaigns train, 6 unseen ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tri, tei = next(gss.split(X, y, df["Author"]))
    mg = xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=800,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbosity=0).fit(X.iloc[tri], y.iloc[tri])
    all_results["XGBoost_grouped"] = reg_metrics(y.iloc[tei], mg.predict(X.iloc[tei]))

    print("figures ...")
    yt = y_test.values
    parity_plot(yt, {"XGBoost": preds["XGBoost"], "GPR": preds["GPR"], "CodeLaw": preds["CodeLaw"]})
    notch_residual_plot(df_test.reset_index(drop=True), yt - code_pred, yt - preds["XGBoost"])
    pd_depth_plot(models["XGBoost"], X_train)
    calibration_plot(yt, gpr_mu, gpr_sd)
    all_results["shap_top"] = shap_plots(models["XGBoost"], X_test, list(X.columns))

    LAST.update(y_test=yt, X_test=X_test, models=models, preds=preds,
                gpr_sd=gpr_sd, df_test=df_test.reset_index(drop=True))
    elapsed = round(time.time() - t0, 1)
    print(f"\npipeline finished in {elapsed} s\n")
    held = {k: v for k, v in all_results.items()
            if k not in ("cv", "code_beta", "shap_top")}
    print_table(held)
    with open(RES / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    return all_results, elapsed


if __name__ == "__main__":
    run_pipeline()
