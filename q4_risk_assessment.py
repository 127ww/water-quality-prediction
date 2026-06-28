"""
Q4: Water Quality Risk Assessment
S = integral of (NTU-1) exceedance, D = max continuous exceedance duration
K-Means (K=4) classification -> Safe/Low/Medium/High
Constraint: E1 (<=1 NTU), I15 (MECE)
Output: ./output/q4_risk_classification.xlsx
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def calc_risk_metrics(ntu_series):
    """Calculate severity S (integral) and duration D (max consecutive hours) per day"""
    vals = ntu_series.values
    exceed = np.maximum(vals - 1.0, 0)
    S = sum(exceed) * 2.0  # sum * dt(2h) for physical integral (NTU-hours)

    # Max consecutive exceedance (in hours = steps * 2)
    exceed_flag = (vals > 1.0).astype(int)
    max_run, cur = 0, 0
    for v in exceed_flag:
        if v == 1:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    D = max_run * 2  # hours

    return float(S), float(D)


def classify_risk(df_mar):
    """K-Means (K=4) classification of daily NTU into risk levels"""
    daily_metrics = []
    for day in df_mar['datetime'].dt.date.unique():
        day_data = df_mar[df_mar['datetime'].dt.date == day]
        S, D = calc_risk_metrics(day_data['NTU'])
        ntu_max = day_data['NTU'].max()
        ntu_mean = day_data['NTU'].mean()
        daily_metrics.append({
            'Date': day, 'Severity_S': S, 'Duration_D_h': D,
            'NTU_Max': ntu_max, 'NTU_Mean': ntu_mean,
        })

    df_daily = pd.DataFrame(daily_metrics)

    # K-Means clustering
    X = StandardScaler().fit_transform(df_daily[['Severity_S', 'Duration_D_h']])
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=20)
    df_daily['Cluster'] = kmeans.fit_predict(X)

    # Map clusters to risk levels by sorting by (S+D) center magnitude
    centers = kmeans.cluster_centers_
    risk_order = np.argsort(centers.sum(axis=1))
    risk_map = {risk_order[0]: 'Safe', risk_order[1]: 'Low',
                risk_order[2]: 'Medium', risk_order[3]: 'High'}
    df_daily['Risk_Level'] = df_daily['Cluster'].map(risk_map)

    # I15 MECE verification
    assert df_daily['Risk_Level'].notna().all(), "Unmapped clusters"
    total_days = len(df_daily)
    for level in ['Safe', 'Low', 'Medium', 'High']:
        count = (df_daily['Risk_Level'] == level).sum()
        pct = count / total_days * 100
        print(f"  {level}: {count} days ({pct:.1f}%)")

    return df_daily


def main():
    print("=" * 60)
    print("Q4: Water Quality Risk Assessment")
    print("=" * 60)

    df_mar = pd.read_pickle('./cache/preprocessed_2026_mar.pkl')
    print(f"  March 2026: {len(df_mar)} readings, "
          f"NTU mean={df_mar['NTU'].mean():.3f}, max={df_mar['NTU'].max():.3f}")

    # Overview: how many readings exceed 1 NTU
    n_exceed = (df_mar['NTU'] > 1.0).sum()
    print(f"  NTU > 1.0: {n_exceed}/{len(df_mar)} readings ({n_exceed/len(df_mar)*100:.1f}%)")

    print("\n[1/2] Computing S (severity integral) and D (duration)...")
    df_risk = classify_risk(df_mar)

    # Display sample
    print(f"\n[2/2] Risk classification summary:")
    print(f"  Total days: {len(df_risk)}")
    for _, row in df_risk.iterrows():
        if row['Risk_Level'] in ['Medium', 'High'] or row['NTU_Max'] > 1.5:
            print(f"  {row['Date']}: {row['Risk_Level']}, "
                  f"S={row['Severity_S']:.3f}, D={row['Duration_D_h']:.0f}h, "
                  f"NTU_max={row['NTU_Max']:.3f}")

    os.makedirs('./output', exist_ok=True)
    df_risk.to_excel('./output/q4_risk_classification.xlsx', index=False,
                      columns=['Date', 'Risk_Level', 'Severity_S', 'Duration_D_h',
                               'NTU_Max', 'NTU_Mean'])
    print(f"\nQ4 done. Output: ./output/q4_risk_classification.xlsx")
    return df_risk


if __name__ == '__main__':
    main()
