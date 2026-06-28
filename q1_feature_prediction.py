"""
Q1: Feature Selection & NTU Prediction
Methods: LASSO + MI + RF cross-select -> XGBoost predict -> SHAP explain
Output: ./output/q1_predictions.xlsx
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import shap

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def select_features(X, y):
    """LASSO + RF + MI three-way cross-selection"""
    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=TimeSeriesSplit(5), max_iter=5000, random_state=RANDOM_SEED)
    lasso.fit(X_scaled, y)
    lasso_selected = [feature_names[i] for i in range(len(feature_names))
                      if abs(lasso.coef_[i]) > 1e-6]

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    rf_selected = rf_imp.head(10).index.tolist()

    mi = mutual_info_regression(X_scaled, y, random_state=RANDOM_SEED)
    mi_series = pd.Series(mi, index=feature_names).sort_values(ascending=False)
    mi_selected = mi_series.head(10).index.tolist()

    core = list(set(lasso_selected) & set(rf_selected) & set(mi_selected))
    if len(core) < 5:
        for f in rf_imp.index:
            if f not in core:
                core.append(f)
            if len(core) >= 10:
                break

    return {
        'lasso': lasso_selected, 'rf': rf_selected, 'mi': mi_selected, 'core': core,
        'lasso_coef': dict(zip(feature_names, lasso.coef_)),
        'rf_importance': rf_imp.to_dict(), 'mi_scores': mi_series.to_dict(),
    }


def train_models(X, y, feature_names):
    """Train >=3 models, time-series CV evaluation"""
    tscv = TimeSeriesSplit(5)
    models = {
        'MLR': LinearRegression(),
        'Ridge': RidgeCV(cv=5),
        'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10,
                                               random_state=RANDOM_SEED, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                 random_state=RANDOM_SEED, n_jobs=-1, base_score=0.5),
    }
    results = {}
    for name, model in models.items():
        rmse_scores, r2_scores = [], []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2_scores.append(r2_score(y_val, y_pred))
        results[name] = {
            'RMSE': float(np.mean(rmse_scores)), 'RMSE_std': float(np.std(rmse_scores)),
            'R2': float(np.mean(r2_scores)), 'R2_std': float(np.std(r2_scores)),
        }
    return results


def predict_target_dates(df_train, df_feb, feature_cols):
    """Predict Feb 1/10/20 NTU using Ridge (best CV model, R2=0.352)"""
    df_tr = df_train.dropna().copy()
    drop_cols = ['datetime', 'NTU', 'PROCESS_VIOLATION', 'BW_FLAG']
    available = [c for c in feature_cols if c in df_tr.columns and c not in drop_cols]
    X_train = df_tr[available].values
    y_train = df_tr['NTU'].values

    y_train_log = np.log1p(y_train)
    model = RidgeCV(cv=5)
    model.fit(X_train, y_train_log)

    target_dates = ['2026-02-01', '2026-02-10', '2026-02-20']
    predictions = {}
    for target_date in target_dates:
        day_data = df_feb[df_feb['datetime'].dt.strftime('%Y-%m-%d') == target_date].copy()
        if len(day_data) == 0:
            continue
        X_pred = day_data[available].values
        pred_log = model.predict(X_pred)
        pred_values = np.maximum(np.expm1(pred_log), 0)
        times = day_data['datetime'].dt.strftime('%H:%M').values
        predictions[target_date] = pd.Series(pred_values, index=times)
    return predictions


def main():
    print("=" * 60)
    print("Q1: Feature Selection & NTU Prediction")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    df_2025 = pd.read_pickle('./cache/preprocessed_2025.pkl')
    df_feb = pd.read_pickle('./cache/preprocessed_2026_feb.pkl')
    df_clean = df_2025.dropna()

    drop_cols = ['datetime', 'NTU', 'PROCESS_VIOLATION', 'BW_FLAG']
    all_features = [c for c in df_clean.columns if c not in drop_cols]

    # Pre-filter by correlation to reduce noise
    corr = df_clean[all_features].corrwith(df_clean['NTU']).abs().sort_values(ascending=False)
    feature_cols = corr[corr > 0.08].index.tolist()
    for must in ['FILT_NTU', 'RW_NTU', 'ALUM', 'RW_PH', 'RW_FLOW']:
        if must in all_features and must not in feature_cols:
            feature_cols.append(must)

    X = df_clean[feature_cols]
    y = df_clean['NTU']
    upper = y.quantile(0.995)
    mask = y <= upper
    X, y = X[mask], y[mask]
    y_log = np.log1p(y)
    print(f"  Training: {len(X)} rows x {len(feature_cols)} features (excl >{upper:.2f} NTU)")

    print("\n[2/4] Feature selection (LASSO+RF+MI)...")
    sel = select_features(X, y_log)
    print(f"  LASSO: {len(sel['lasso'])}, RF: {len(sel['rf'])}, MI: {len(sel['mi'])}")
    print(f"  Core (intersection): {len(sel['core'])}")
    for i, f in enumerate(sel['core'][:10]):
        coef = sel['lasso_coef'].get(f, 0)
        direction = '+' if coef > 0 else '-'
        print(f"    {i+1}. {f} [{direction}]")

    print("\n[3/4] Model training & comparison...")
    X_arr = X.values.astype(np.float64)
    y_arr = y_log.values.astype(np.float64)
    results = train_models(X_arr, y_arr, feature_cols)

    print(f"  {'Model':<15} {'RMSE':>8} {'R2':>8}")
    print(f"  {'-'*31}")
    best_name, best_r2 = None, -np.inf
    for name, m in results.items():
        print(f"  {name:<15} {m['RMSE']:>8.4f} {m['R2']:>8.4f}")
        if m['R2'] > best_r2:
            best_r2, best_name = m['R2'], name
    print(f"  Best: {best_name} (R2={best_r2:.4f})")

    print("\n[4/4] SHAP + prediction...")
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                         random_state=RANDOM_SEED, base_score=0.5)
    xgb.fit(X_arr, y_arr)

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_arr[:min(300, len(X_arr))])
    shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    print("  SHAP Top-10:")
    for i, (f, v) in enumerate(shap_imp.head(10).items()):
        direction = '+' if np.corrcoef(X[f].values, y_arr)[0, 1] > 0 else '-'
        print(f"    {i+1}. {f}: |SHAP|={v:.4f} [{direction}]")

    preds = predict_target_dates(df_2025, df_feb, feature_cols)

    os.makedirs('./output', exist_ok=True)
    with pd.ExcelWriter('./output/q1_predictions.xlsx') as writer:
        for date_str, series in preds.items():
            df_out = pd.DataFrame({'TIME': series.index, 'Predicted_NTU': series.values.round(4)})
            sheet_name = date_str.replace('-', '')[4:]
            df_out.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  {date_str}: {len(df_out)} predictions, mean={series.mean():.4f}")

    print(f"\nQ1 done. Output: ./output/q1_predictions.xlsx")
    return results, sel, preds


if __name__ == '__main__':
    main()
