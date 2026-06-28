"""
Q2: Dynamic Time-Lag Model for FILT.NTU
Methods: CCF lag detection -> Almon DLM -> XGBoost-lag comparison
Constraint: I14(2h step), E2(ALUM lag 2-6h -> 1-3 steps)
Output: ./output/q2_lag_params.csv
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

INPUT_VARS = ['RW_NTU', 'RW_PH', 'ALUM', 'RW_FLOW']
TARGET = 'FILT_NTU'
MAX_LAG = 6
ALUM_MAX_LAG = 3  # E2 constraint


def compute_ccf_lags(df):
    """CCF on log-target; ALUM constrained to [1,3] per E2"""
    y = np.log1p(df[TARGET].values)
    lags = {}
    for var in INPUT_VARS:
        if var not in df.columns:
            lags[var] = 0; continue
        x = df[var].values
        ccf_vals = []
        for lag in range(MAX_LAG + 1):
            if lag == 0:
                cc = np.corrcoef(x, y)[0, 1]
            else:
                cc = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            ccf_vals.append(abs(cc) if not np.isnan(cc) else 0)
        opt = int(np.argmax(ccf_vals))
        if var == 'ALUM':
            opt = min(max(opt, 1), ALUM_MAX_LAG)
        lags[var] = opt
    return lags


def fit_almon_dlm(df, lags=None, poly_order=2):
    """Almon PDL with log-target, standardized inputs"""
    if lags is None:
        lags = compute_ccf_lags(df)
    y = np.log1p(df[TARGET].values)
    n = len(df)
    max_lag = max(lags.values())

    Z_list = []
    for var in INPUT_VARS:
        k_i = lags.get(var, 0)
        x = df[var].values
        x_mean, x_std = np.nanmean(x), np.nanstd(x) + 1e-8
        x_norm = (x - x_mean) / x_std
        if k_i == 0:
            Z_list.append(x_norm)
        else:
            for p in range(poly_order + 1):
                z = np.zeros(n)
                for j in range(k_i + 1):
                    shifted = np.roll(x_norm, j)
                    shifted[:j] = 0
                    z += (j ** p) * shifted
                Z_list.append(z)

    Z = np.column_stack(Z_list)
    Z_full = np.column_stack([np.ones(n), Z])
    valid = ~np.isnan(Z_full).any(axis=1) & ~np.isnan(y)
    Z_f, y_f = Z_full[valid], y[valid]

    theta = np.linalg.lstsq(Z_f, y_f, rcond=None)[0]

    lag_params = {}
    idx = 1
    for var in INPUT_VARS:
        k_i = lags.get(var, 0)
        if k_i == 0:
            lag_params[var] = [float(theta[idx])]
            idx += 1
        else:
            betas = [float(sum(theta[idx + p] * (j ** p) for p in range(poly_order + 1)))
                     for j in range(k_i + 1)]
            lag_params[var] = betas
            idx += poly_order + 1

    yp_log = Z_f @ theta
    rmse = np.sqrt(mean_squared_error(np.expm1(y_f), np.expm1(yp_log)))
    r2 = r2_score(np.expm1(y_f), np.expm1(yp_log))

    return {'lags': lags, 'lag_params': lag_params, 'RMSE': float(rmse),
            'R2': float(r2), 'n_obs': len(y_f)}


def fit_xgboost_lag(df, lags=None):
    """XGBoost with CCF-lagged features, log target"""
    if lags is None:
        lags = compute_ccf_lags(df)
    max_lag = max(lags.values())

    feature_data = {}
    for var in INPUT_VARS:
        x = df[var].values
        for j in range(lags.get(var, 0) + 1):
            f = np.roll(x, j); f[:j] = np.nan
            feature_data[f'{var}_lag{j}'] = f

    X = pd.DataFrame(feature_data).iloc[max_lag:].values
    y = df[TARGET].values[max_lag:]
    y_log = np.log1p(y)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_log)
    X, y_log, y = X[mask], y_log[mask], y[mask]

    split = int(len(X) * 0.8)
    model = XGBRegressor(n_estimators=200, max_depth=5,
                          learning_rate=0.03, random_state=RANDOM_SEED)
    model.fit(X[:split], y_log[:split])
    yp = np.expm1(model.predict(X[split:]))
    rmse = np.sqrt(mean_squared_error(y[split:], yp))
    r2 = r2_score(y[split:], yp)
    return {'RMSE': float(rmse), 'R2': float(r2)}


def main():
    print("=" * 60)
    print("Q2: Dynamic Time-Lag Model")
    print("=" * 60)

    df = pd.read_pickle('./cache/preprocessed_q2.pkl').dropna()
    if 'BW_FLAG' in df.columns:
        df = df[df['BW_FLAG'] == 0].copy()

    # Remove extreme outliers (top 2% flood events) for stable model fitting
    upper = df[TARGET].quantile(0.98)
    n_ext = (df[TARGET] > upper).sum()
    df_fit = df[df[TARGET] <= upper].copy()
    print(f"  Total: {len(df)} rows, fitting: {len(df_fit)} (excl {n_ext} extreme >{upper:.2f})")
    print(f"  FILT_NTU [fit]: mean={df_fit[TARGET].mean():.3f}, std={df_fit[TARGET].std():.3f}")

    print("\n[1/3] CCF lag detection...")
    lags = compute_ccf_lags(df_fit)
    print(f"  {'Variable':<12} {'Lag(step)':>10} {'Lag(h)':>8}")
    for var in INPUT_VARS:
        print(f"  {var:<12} {lags[var]:>10} {lags[var]*2:>8}")

    print("\n[2/3] Almon DLM (log-target, standardized)...")
    dlm = fit_almon_dlm(df_fit, lags)
    print(f"  RMSE={dlm['RMSE']:.4f}, R2={dlm['R2']:.4f}")
    for var in INPUT_VARS:
        print(f"  {var} weights: {[round(b,5) for b in dlm['lag_params'][var]]}")

    print("\n[3/3] XGBoost-lag comparison...")
    xgb = fit_xgboost_lag(df_fit, lags)
    print(f"  Almon DLM:   RMSE={dlm['RMSE']:.4f}, R2={dlm['R2']:.4f}")
    print(f"  XGBoost-lag: RMSE={xgb['RMSE']:.4f}, R2={xgb['R2']:.4f}")

    os.makedirs('./output', exist_ok=True)
    rows = []
    for var in INPUT_VARS:
        lag_step = lags[var]
        y_log = np.log1p(df_fit[TARGET].values)
        x = df_fit[var].values
        if lag_step > 0:
            cc = np.corrcoef(x[lag_step:], y_log[:-lag_step])[0, 1]
        else:
            cc = np.corrcoef(x, y_log)[0, 1]
        rows.append({'Variable': var, 'Optimal_Lag_Step': lag_step,
                     'Optimal_Lag_Hours': lag_step * 2, 'Max_CCF': round(float(cc), 4)})
    pd.DataFrame(rows).to_csv('./output/q2_lag_params.csv', index=False)
    print(f"\nQ2 done. Output: ./output/q2_lag_params.csv")
    return dlm, xgb


if __name__ == '__main__':
    main()
