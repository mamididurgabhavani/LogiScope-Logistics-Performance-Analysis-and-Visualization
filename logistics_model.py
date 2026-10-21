"""Week 4 - Optimization in Logistics Systems (LogiScope)
Predict shipment delay, validate the model, then use predictions to optimize
carrier allocation and proactive intervention."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid")
SEED = 42
OUT = "charts4/"
R = {}

# ------------------------------------------------------------------
# 1. DATA PREPARATION
# ------------------------------------------------------------------
df = pd.read_csv("logistics_data.csv", parse_dates=["ship_date"])
df["is_peak"] = df.month.isin([10, 11, 12]).astype(int)
df["weekday_num"] = df.ship_date.dt.dayofweek

TARGET = "delay_days"
num_feats = ["distance_km", "weight_kg", "volume_m3", "planned_days", "month", "is_peak"]
cat_feats = ["origin_hub", "transport_mode", "carrier", "priority"]
FEATS = num_feats + cat_feats
X, y = df[FEATS], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
R["n_train"], R["n_test"] = len(X_train), len(X_test)

prep = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
])

def make(model):
    return Pipeline([("prep", prep), ("model", model)])

# ------------------------------------------------------------------
# 2. MODEL COMPARISON (5-fold CV on the training set)
# ------------------------------------------------------------------
models = {
    "Mean baseline": None,
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, m in models.items():
    if m is None:
        pred_cv = np.full(len(y_train), y_train.mean())
        rows.append({"model": name, "cv_rmse": float(np.sqrt(mean_squared_error(y_train, pred_cv))),
                     "cv_rmse_std": 0.0, "cv_r2": 0.0})
        continue
    pipe = make(m)
    rmse = -cross_val_score(pipe, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
    r2 = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2")
    rows.append({"model": name, "cv_rmse": rmse.mean(), "cv_rmse_std": rmse.std(), "cv_r2": r2.mean()})
cv_tbl = pd.DataFrame(rows)
R["cv"] = cv_tbl.round(3).to_dict("records")

# ------------------------------------------------------------------
# 3. HYPERPARAMETER TUNING (GridSearchCV)
# ------------------------------------------------------------------
gb_grid = {"model__n_estimators": [100, 200, 300], "model__max_depth": [2, 3, 4],
           "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.8, 1.0]}
gs = GridSearchCV(make(GradientBoostingRegressor(random_state=SEED)), gb_grid, cv=kf,
                  scoring="neg_root_mean_squared_error", n_jobs=-1)
gs.fit(X_train, y_train)
R["best_params"] = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
R["tuned_cv_rmse"] = round(-gs.best_score_, 3)
R["grid_size"] = int(np.prod([len(v) for v in gb_grid.values()]))
best = gs.best_estimator_

# ------------------------------------------------------------------
# 4. FINAL TEST-SET EVALUATION
# ------------------------------------------------------------------
def evaluate(name, pipe):
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    return {"model": name, "mae": mean_absolute_error(y_test, p),
            "rmse": float(np.sqrt(mean_squared_error(y_test, p))), "r2": r2_score(y_test, p)}, p

test_rows = [{"model": "Mean baseline",
              "mae": mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())),
              "rmse": float(np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))),
              "r2": 0.0}]
preds = {}
for name in ["Linear Regression", "Decision Tree", "Random Forest"]:
    r, p = evaluate(name, make(models[name]))
    test_rows.append(r); preds[name] = p
rb, pb = evaluate("Gradient Boosting (tuned)", best)
test_rows.append(rb); preds["Gradient Boosting (tuned)"] = pb
test_tbl = pd.DataFrame(test_rows)
R["test"] = test_tbl.round(3).to_dict("records")
R["y_test_std"] = round(float(y_test.std()), 3)

# Fig 1: model comparison
fig, ax = plt.subplots(figsize=(8.5, 4.3))
order = test_tbl.sort_values("rmse", ascending=False)
bars = ax.barh(order.model, order.rmse, color=["#9DB4C0" if "baseline" in m else "#2E6F9E" for m in order.model])
for b_, v in zip(bars, order.rmse):
    ax.text(v + 0.01, b_.get_y() + b_.get_height() / 2, f"{v:.3f}", va="center")
ax.set(title="Test-set RMSE by Model (lower is better)", xlabel="RMSE (days)")
fig.tight_layout(); fig.savefig(OUT + "m1_models.png", dpi=160); plt.close(fig)

# Fig 2: actual vs predicted + residuals
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(y_test, pb, alpha=0.5, s=18, color="#2E6F9E")
lim = [0, max(y_test.max(), pb.max()) + 0.2]
axes[0].plot(lim, lim, "r--", label="Perfect prediction")
axes[0].set(title="Actual vs Predicted Delay (test set)", xlabel="Actual delay (days)", ylabel="Predicted delay (days)")
axes[0].legend()
res = y_test.values - pb
sns.histplot(res, bins=30, kde=True, color="#C0392B", ax=axes[1])
axes[1].axvline(0, color="k", lw=1)
axes[1].set(title="Residual Distribution", xlabel="Residual (actual - predicted, days)")
fig.tight_layout(); fig.savefig(OUT + "m2_fit.png", dpi=160); plt.close(fig)
R["resid_mean"] = round(float(res.mean()), 3)
R["resid_std"] = round(float(res.std()), 3)

# Fig 3: permutation importance (on original features)
pi = permutation_importance(best, X_test, y_test, n_repeats=15, random_state=SEED,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
imp = pd.DataFrame({"feature": FEATS, "importance": pi.importances_mean, "std": pi.importances_std}) \
        .sort_values("importance")
R["importance"] = imp.sort_values("importance", ascending=False).round(3).to_dict("records")
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.barh(imp.feature, imp.importance, xerr=imp["std"], color="#2E6F9E")
ax.set(title="Permutation Feature Importance (increase in RMSE when shuffled)", xlabel="RMSE increase (days)")
fig.tight_layout(); fig.savefig(OUT + "m3_importance.png", dpi=160); plt.close(fig)

# Fig 4: tuning results - depth x learning rate
cvr = pd.DataFrame(gs.cv_results_)
piv = cvr.groupby(["param_model__max_depth", "param_model__learning_rate"])["mean_test_score"].max().unstack() * -1
fig, ax = plt.subplots(figsize=(6.5, 3.8))
sns.heatmap(piv.astype(float), annot=True, fmt=".3f", cmap="viridis_r", cbar_kws={"label": "CV RMSE"}, ax=ax)
ax.set(title="Grid Search: Best CV RMSE by Depth and Learning Rate", xlabel="Learning rate", ylabel="Max depth")
fig.tight_layout(); fig.savefig(OUT + "m4_grid.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------------
# 5. OPTIMIZATION A: CARRIER ALLOCATION (linear programming)
# ------------------------------------------------------------------
# Refit the tuned model on all data, then predict each shipment's delay under every carrier.
final = make(GradientBoostingRegressor(random_state=SEED, **{k: v for k, v in R["best_params"].items()}))
final.fit(X, y)
carriers = sorted(df.carrier.unique())
PENALTY = 25.0   # assumed cost (USD) per day of delay per shipment
pred_matrix = np.column_stack([final.predict(df[FEATS].assign(carrier=c)) for c in carriers])
cost_matrix = pred_matrix * PENALTY                      # shipments x carriers
n, k = cost_matrix.shape
cur_idx = df.carrier.map({c: i for i, c in enumerate(carriers)}).values
cur_cost = cost_matrix[np.arange(n), cur_idx].sum()
cur_counts = np.bincount(cur_idx, minlength=k)
cap = np.ceil(cur_counts * 1.25).astype(int)             # each carrier can absorb +25% volume
# decision variable x[i,j] in [0,1]; each shipment fully assigned; carrier capacity limits
c_vec = cost_matrix.flatten()
A_eq = np.zeros((n, n * k)); 
for i in range(n): A_eq[i, i * k:(i + 1) * k] = 1
A_ub = np.zeros((k, n * k))
for j in range(k): A_ub[j, j::k] = 1
res_lp = linprog(c_vec, A_ub=A_ub, b_ub=cap, A_eq=A_eq, b_eq=np.ones(n), bounds=(0, 1), method="highs")
X_opt = res_lp.x.reshape(n, k)
new_idx = X_opt.argmax(axis=1)
new_counts = np.bincount(new_idx, minlength=k)
new_cost = cost_matrix[np.arange(n), new_idx].sum()
exp_delay_cur = pred_matrix[np.arange(n), cur_idx]
exp_delay_new = pred_matrix[np.arange(n), new_idx]
R["lp"] = {
    "status": res_lp.message, "carriers": carriers,
    "cur_counts": cur_counts.tolist(), "new_counts": new_counts.tolist(), "cap": cap.tolist(),
    "cur_delay_total": round(float(exp_delay_cur.sum()), 0), "new_delay_total": round(float(exp_delay_new.sum()), 0),
    "cur_delay_avg": round(float(exp_delay_cur.mean()), 3), "new_delay_avg": round(float(exp_delay_new.mean()), 3),
    "cur_cost": round(float(cur_cost), 0), "new_cost": round(float(new_cost), 0),
    "saving": round(float(cur_cost - new_cost), 0), "saving_pct": round(float((cur_cost - new_cost) / cur_cost * 100), 1),
    "moved": int((new_idx != cur_idx).sum()),
    "frac_integral": round(float(((X_opt > 1e-6) & (X_opt < 1 - 1e-6)).any(axis=1).mean() * 100), 1),
    "penalty": PENALTY,
}
fig, ax = plt.subplots(figsize=(8.5, 4.3))
xs = np.arange(k); wd = 0.38
ax.bar(xs - wd / 2, cur_counts, wd, label="Current", color="#9DB4C0")
ax.bar(xs + wd / 2, new_counts, wd, label="Optimized", color="#2E6F9E")
ax.scatter(xs, cap, marker="_", s=900, color="#C0392B", label="Capacity limit (+25%)", zorder=3)
ax.set_xticks(xs); ax.set_xticklabels(carriers)
ax.set(title="Carrier Allocation: Current vs Optimized", ylabel="Shipments")
ax.legend(); fig.tight_layout(); fig.savefig(OUT + "m5_alloc.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------------
# 6. OPTIMIZATION B: RISK-BASED PROACTIVE INTERVENTION
# ------------------------------------------------------------------
# Out-of-sample predictions for every shipment (5-fold) so the policy is not evaluated in-sample.
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(make(GradientBoostingRegressor(random_state=SEED, **R["best_params"])), X, y, cv=kf)
INTERV_COST, REDUCTION = 15.0, 0.30   # assumed: $15 per intervention, cuts delay by 30%
thr_rows = []
for thr in np.arange(0.4, 3.01, 0.1):
    flag = oof > thr
    benefit = (y.values[flag] * REDUCTION * PENALTY).sum()
    cost_ = flag.sum() * INTERV_COST
    late = (y.values > 1.0)
    prec = (late & flag).sum() / max(flag.sum(), 1)
    rec = (late & flag).sum() / late.sum()
    thr_rows.append({"thr": round(float(thr), 1), "flagged": int(flag.sum()), "precision": prec, "recall": rec,
                     "net": benefit - cost_})
thr_df = pd.DataFrame(thr_rows)
best_row = thr_df.loc[thr_df.net.idxmax()]
R["policy"] = {"best_thr": float(best_row.thr), "flagged": int(best_row.flagged),
               "precision": round(float(best_row.precision), 3), "recall": round(float(best_row.recall), 3),
               "net": round(float(best_row.net), 0), "all_net": round(float((y.values * REDUCTION * PENALTY).sum() - n * INTERV_COST), 0),
               "interv_cost": INTERV_COST, "reduction": REDUCTION,
               "table": thr_df.iloc[::3].round(3).to_dict("records")}
# sensitivity of the best policy to the assumed intervention cost and effectiveness
sens = []
for ic in [5.0, 10.0, 15.0]:
    for red in [0.3, 0.5]:
        best_ = max(((((y.values[oof > t] * red * PENALTY).sum() - (oof > t).sum() * ic), t, int((oof > t).sum()))
                     for t in np.arange(0.4, 3.01, 0.1)), key=lambda z: z[0])
        sens.append({"cost": ic, "reduction": red, "net": round(float(best_[0]), 0),
                     "thr": round(float(best_[1]), 1), "flagged": best_[2]})
R["policy_sens"] = sens
fig, ax1 = plt.subplots(figsize=(8.5, 4.3))
ax1.plot(thr_df.thr, thr_df.net, marker="o", color="#2E6F9E", label="Net benefit ($)")
ax1.axvline(best_row.thr, color="#C0392B", ls="--", label=f"Best threshold = {best_row.thr:.1f} d")
ax1.set(xlabel="Flag shipment if predicted delay exceeds (days)", ylabel="Net benefit (USD)",
        title="Choosing the Intervention Threshold")
ax2 = ax1.twinx(); ax2.plot(thr_df.thr, thr_df.flagged, color="#7F8C8D", ls=":", label="Shipments flagged"); ax2.grid(False)
ax2.set_ylabel("Shipments flagged")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower center")
fig.tight_layout(); fig.savefig(OUT + "m6_policy.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------------
# 7. SCENARIO: combined effect on predicted delay
# ------------------------------------------------------------------
flag_best = oof > best_row.thr
d_after_lp = exp_delay_new.copy()
R["scenario"] = {
    "baseline_avg": round(float(exp_delay_cur.mean()), 3),
    "lp_avg": round(float(exp_delay_new.mean()), 3),
    "n_late_cur": int((exp_delay_cur > 1.0).sum()), "n_late_new": int((exp_delay_new > 1.0).sum()),
}
R["data_n"] = n
R["peak_share_flagged"] = round(float(df.is_peak.values[flag_best].mean() * 100), 1)
R["mumbai_delhi_share_flagged"] = round(float(df.origin_hub.isin(["Mumbai", "Delhi"]).values[flag_best].mean() * 100), 1)
with open("stats4.json", "w") as f:
    json.dump(R, f, indent=1, default=float)
print("done")
