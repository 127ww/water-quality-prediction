"""
Q3: Hybrid Dynamic Model (CSTR mechanism + Seq2Seq LSTM residual)
Predict 1-12h ahead treated water NTU for Feb 1/10/20, 7:00-19:00
Output: ./output/q3_predictions.xlsx
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


def build_cstr_model(df, n_cstr=3, tau_scale=50.0):
    """CSTR串联清水池停留时间模型（解析解，无clip退化）
    dC_i/dt = (C_{i-1} - C_i)/tau_i 的精确离散化:
    C_i(t+dt) = C_i(t)*exp(-dt/tau_i) + C_{i-1}(t)*(1-exp(-dt/tau_i))
    对任意 tau>0 恒为凸组合，不退化。
    tau_scale: 水位→停留时间标定系数（默认50，使总停留时间约4-8h）
    """
    dt = 2.0  # 2h step
    # 估算平均停留时间: tau_base = tau_scale * level / flow
    cw_level = df['CW_WELL_LEVEL'].values
    tw_flow = df['TW_FLOW'].values
    tau_base = tau_scale * np.nanmean(cw_level) / (np.nanmean(tw_flow) + 1e-8)
    # 确保总停留时间不低于4h（清水池物理合理下界）
    tau_base = max(tau_base, 6.0)
    tau_per_cstr = tau_base / n_cstr

    filt_ntu = df['FILT_NTU'].values
    n = len(filt_ntu)
    cstr_out = np.zeros(n)
    c = np.zeros(n_cstr)

    for t in range(n):
        c_in = filt_ntu[t]
        # 解析解更新各CSTR（无需clip）
        for j in range(n_cstr):
            c_prev = c_in if j == 0 else c[j - 1]
            alpha = np.exp(-dt / tau_per_cstr)  # 0<alpha<1 恒成立
            c[j] = alpha * c[j] + (1 - alpha) * c_prev
        cstr_out[t] = c[-1]

    return cstr_out, tau_per_cstr


def build_lstm_residual(df, cstr_pred):
    """Seq2Seq LSTM learning CSTR residuals.
    Input: past 24h features -> Output: future 12h residual correction.
    Uses simplified dense+LSTM for reliability.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping

    # Features: FILT_NTU, CW_WELL_LEVEL, TW_FLOW, ALUM, CSTR residuals
    target = df['NTU'].values
    residual = target - cstr_pred
    n = len(residual)

    # Feature matrix
    feat_cols = ['FILT_NTU', 'CW_WELL_LEVEL', 'TW_FLOW']
    if 'ALUM' in df.columns:
        feat_cols.append('ALUM')
    feats = df[feat_cols].values

    # Standardize
    scaler_X = StandardScaler()
    feats_scaled = scaler_X.fit_transform(feats)

    # Build sequences: 12-step input -> 6-step output (24h -> 12h)
    seq_in, seq_out = 12, 6
    X_seq, y_seq = [], []
    for i in range(seq_in, n - seq_out):
        X_seq.append(feats_scaled[i - seq_in:i])
        y_seq.append(residual[i:i + seq_out])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if len(X_seq) < 50:
        return None, None, feat_cols

    # Train/val split
    split = int(len(X_seq) * 0.8)
    X_tr, X_val = X_seq[:split], X_seq[split:]
    y_tr, y_val = y_seq[:split], y_seq[split:]

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(seq_in, feats.shape[1])),
        LSTM(16),
        Dense(seq_out)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=50, batch_size=32, callbacks=[es], verbose=0)

    return model, scaler_X, feat_cols


def forecast_target_date(df, model, scaler_X, cstr_pred, target_date, feats_raw):
    """Rolling forecast for one target date, 7:00-19:00"""
    seq_in = 12

    results = []
    for hour in [7, 9, 11, 13, 15, 17, 19]:
        target_dt = pd.Timestamp(f'{target_date} {hour:02d}:00:00')
        # Find closest data point by integer position
        time_diffs = np.abs((df['datetime'] - target_dt).dt.total_seconds())
        closest_pos = int(time_diffs.idxmin())

        if closest_pos < seq_in:
            continue

        # Prepare input sequence using integer position indexing
        window = feats_raw[closest_pos - seq_in:closest_pos]
        if len(window) != seq_in:
            continue
        input_seq = scaler_X.transform(window).reshape(1, seq_in, -1)

        if model is not None:
            residual = model.predict(input_seq, verbose=0)[0, 0]
        else:
            residual = 0

        # Use CSTR at pos-1 to avoid future FILT_NTU leakage (review fix)
        cstr_val = cstr_pred[max(0, closest_pos - 1)]
        ntu_pred = max(cstr_val + residual, 0)
        results.append({'TIME': f'{hour:02d}:00', 'Predicted_NTU': round(float(ntu_pred), 4)})

    return results


def main():
    print("=" * 60)
    print("Q3: Hybrid CSTR + LSTM Dynamic Forecast")
    print("=" * 60)

    df_2025 = pd.read_pickle('./cache/preprocessed_2025.pkl').dropna()
    df_feb = pd.read_pickle('./cache/preprocessed_2026_feb.pkl')

    print(f"  Training: {len(df_2025)} rows")

    # Step 1: CSTR mechanistic model
    print("\n[1/2] CSTR mechanistic model (N=3)...")
    cstr_pred, tau = build_cstr_model(df_2025, n_cstr=3)
    rmse_cstr = np.sqrt(mean_squared_error(df_2025['NTU'], cstr_pred))
    print(f"  CSTR-only: RMSE={rmse_cstr:.4f}, tau_per_CSTR={tau:.1f}h")

    # Step 2: LSTM residual
    print("\n[2/2] Seq2Seq LSTM residual correction...")
    model, scaler_X, feat_cols = build_lstm_residual(df_2025, cstr_pred)

    # Full prediction evaluation
    if model is not None:
        cstr_full, _ = build_cstr_model(df_2025, n_cstr=3)
        feats_unscaled = df_2025[feat_cols].values
        residual_pred = np.zeros(len(df_2025))
        seq_in = 12
        for i in range(seq_in, len(df_2025)):
            seq = scaler_X.transform(feats_unscaled[i - seq_in:i]).reshape(1, seq_in, -1)
            residual_pred[i] = model.predict(seq, verbose=0)[0, 0]
        hybrid_pred = cstr_full + residual_pred
        rmse_hybrid = np.sqrt(mean_squared_error(
            df_2025['NTU'].values[seq_in:], hybrid_pred[seq_in:]))
        print(f"  Hybrid (CSTR+LSTM): RMSE={rmse_hybrid:.4f}")
        print(f"  CSTR-only RMSE={rmse_cstr:.4f}")
    else:
        hybrid_pred = cstr_pred
        print("  LSTM skipped (insufficient data), using CSTR-only")

    # Forecast target dates using 2026 Feb data features
    print("\nForecasting Feb 1/10/20 7:00-19:00...")
    feat_cols = ['FILT_NTU', 'CW_WELL_LEVEL', 'TW_FLOW']
    if 'ALUM' in df_2025.columns:
        feat_cols.append('ALUM')

    # Combine Jan+Feb 2026 for sufficient lookback before Feb 1
    df_jan = pd.read_pickle('./cache/preprocessed_2026_jan.pkl')
    df_combined = pd.concat([df_jan, df_feb]).sort_values('datetime').reset_index(drop=True)

    # Compute CSTR on combined data
    cstr_combined, _ = build_cstr_model(df_combined, n_cstr=3)
    feats_combined = df_combined[feat_cols].values

    all_preds = {}
    for target_date in ['2026-02-01', '2026-02-10', '2026-02-20']:
        results = forecast_target_date(df_combined, model, scaler_X, cstr_combined,
                                        target_date, feats_combined)
        if results:
            all_preds[target_date] = pd.DataFrame(results)
            print(f"  {target_date}: {len(results)} predictions, "
                  f"mean NTU={all_preds[target_date]['Predicted_NTU'].mean():.4f}")

    os.makedirs('./output', exist_ok=True)
    with pd.ExcelWriter('./output/q3_predictions.xlsx') as writer:
        for date_str, df_out in all_preds.items():
            sheet_name = date_str.replace('-', '')[4:]
            df_out.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\nQ3 done. Output: ./output/q3_predictions.xlsx")
    return True


if __name__ == '__main__':
    main()
