import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)

sns.set_theme(style="whitegrid")
VENV_PYTHON = ".venv/bin/python3"

conn = duckdb.connect("../aml.duckdb")

# ─────────────────────────────────────────────
# STEP 1: BUILD ACCOUNT-LEVEL FEATURES IN SQL
# ─────────────────────────────────────────────
# We compute sender + receiver aggregates once and store as a table.
# MAX(is_laundering) flags an account if ANY tx is laundering.
print("Building account features...")
conn.execute("""
CREATE OR REPLACE TABLE account_features AS

WITH sender AS (
    SELECT
        from_account                                                        AS account,
        COUNT(*)                                                            AS sender_tx_count,
        AVG(amount_paid)                                                    AS sender_avg_amount,
        MAX(amount_paid)                                                    AS sender_max_amount,
        STDDEV(amount_paid)                                                 AS sender_std_amount,
        COUNT(DISTINCT to_account)                                          AS sender_distinct_receivers,
        COUNT(DISTINCT payment_format)                                      AS sender_distinct_formats,
        COUNT(DISTINCT payment_currency)                                    AS sender_distinct_currencies,
        ROUND(SUM(CASE WHEN amount_paid BETWEEN 8000 AND 10000 THEN 1 ELSE 0 END)
              * 100.0 / COUNT(*), 4)                                        AS sender_pct_near_10k
    FROM transactions
    GROUP BY from_account
),

receiver AS (
    SELECT
        to_account                                                          AS account,
        COUNT(*)                                                            AS receiver_tx_count,
        AVG(amount_received)                                                AS receiver_avg_amount,
        MAX(amount_received)                                                AS receiver_max_amount,
        COUNT(DISTINCT from_account)                                        AS receiver_distinct_senders,
        COUNT(DISTINCT payment_format)                                      AS receiver_distinct_formats
    FROM transactions
    GROUP BY to_account
),

velocity AS (
    SELECT
        from_account                                                        AS account,
        AVG(tx_count_last_24h)                                              AS sender_avg_velocity_24h,
        MAX(tx_count_last_24h)                                              AS sender_max_velocity_24h
    FROM (
        SELECT
            from_account,
            COUNT(*) OVER (
                PARTITION BY from_account
                ORDER BY CAST(timestamp AS TIMESTAMP)
                RANGE BETWEEN INTERVAL 1 DAY PRECEDING AND CURRENT ROW
            ) AS tx_count_last_24h
        FROM transactions
    )
    GROUP BY from_account
)

SELECT
    COALESCE(s.account, r.account)      AS account,
    COALESCE(s.sender_tx_count, 0)      AS sender_tx_count,
    s.sender_avg_amount,
    s.sender_max_amount,
    s.sender_std_amount,
    COALESCE(s.sender_distinct_receivers, 0)    AS sender_distinct_receivers,
    COALESCE(s.sender_distinct_formats, 0)      AS sender_distinct_formats,
    COALESCE(s.sender_distinct_currencies, 0)   AS sender_distinct_currencies,
    COALESCE(s.sender_pct_near_10k, 0)          AS sender_pct_near_10k,
    COALESCE(v.sender_avg_velocity_24h, 0)      AS sender_avg_velocity_24h,
    COALESCE(v.sender_max_velocity_24h, 0)      AS sender_max_velocity_24h,
    COALESCE(r.receiver_tx_count, 0)            AS receiver_tx_count,
    r.receiver_avg_amount,
    r.receiver_max_amount,
    COALESCE(r.receiver_distinct_senders, 0)    AS receiver_distinct_senders,
    COALESCE(r.receiver_distinct_formats, 0)    AS receiver_distinct_formats
FROM sender s
FULL OUTER JOIN receiver r  ON s.account = r.account
LEFT JOIN velocity v        ON s.account = v.account
""")
print("  account_features table created.")


# ─────────────────────────────────────────────
# STEP 2: BUILD TRANSACTION-LEVEL FEATURE SET
# ─────────────────────────────────────────────
print("Building transaction feature set...")
df = conn.execute("""
SELECT
    -- labels + split key
    t.timestamp,
    t.is_laundering,

    -- raw transaction features
    t.amount_paid,
    t.amount_received,
    LOG10(NULLIF(t.amount_paid, 0))                                 AS log_amount_paid,
    LOG10(NULLIF(t.amount_received, 0))                             AS log_amount_received,
    CASE WHEN t.amount_paid BETWEEN 8000 AND 10000 THEN 1 ELSE 0 END AS near_10k_flag,
    CASE WHEN t.payment_currency != t.receiving_currency THEN 1 ELSE 0 END AS cross_currency_flag,

    -- categorical (will be label-encoded below)
    t.payment_format,
    t.payment_currency,
    t.receiving_currency,

    -- sender account features
    sf.sender_tx_count,
    sf.sender_avg_amount,
    sf.sender_max_amount,
    sf.sender_std_amount,
    sf.sender_distinct_receivers,
    sf.sender_distinct_formats,
    sf.sender_distinct_currencies,
    sf.sender_pct_near_10k,
    sf.sender_avg_velocity_24h,
    sf.sender_max_velocity_24h,

    -- receiver account features
    rf.receiver_tx_count,
    rf.receiver_avg_amount,
    rf.receiver_max_amount,
    rf.receiver_distinct_senders,
    rf.receiver_distinct_formats

FROM transactions t
LEFT JOIN account_features sf ON t.from_account = sf.account
LEFT JOIN account_features rf ON t.to_account   = rf.account
ORDER BY CAST(t.timestamp AS TIMESTAMP)
""").df()

print(f"  Dataset shape: {df.shape}")
print(f"  Laundering rate: {df['is_laundering'].mean()*100:.3f}%")


# ─────────────────────────────────────────────
# STEP 3: ENCODE CATEGORICALS
# ─────────────────────────────────────────────
cat_cols = ['payment_format', 'payment_currency', 'receiving_currency']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


# ─────────────────────────────────────────────
# STEP 4: TIME-BASED SPLIT (no data leakage)
# ─────────────────────────────────────────────
# Sort is already done in SQL. Use earliest 80% for train, latest 20% for test.
split_idx = int(len(df) * 0.80)

feature_cols = [c for c in df.columns if c not in ['timestamp', 'is_laundering']]

X_train = df.iloc[:split_idx][feature_cols]
y_train = df.iloc[:split_idx]['is_laundering']
X_test  = df.iloc[split_idx:][feature_cols]
y_test  = df.iloc[split_idx:]['is_laundering']

print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")
print(f"Train laundering rate: {y_train.mean()*100:.3f}%")
print(f"Test  laundering rate: {y_test.mean()*100:.3f}%")


# ─────────────────────────────────────────────
# STEP 5: TRAIN BASELINE — RANDOM FOREST
# class_weight='balanced' handles imbalance without resampling
# ─────────────────────────────────────────────
print("\nTraining Random Forest baseline...")
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)
print("  Done.")


# ─────────────────────────────────────────────
# STEP 6: HONEST METRICS
# For imbalanced data: accuracy is useless.
# Key metrics: PR-AUC, ROC-AUC, F1 at 0.5 threshold
# ─────────────────────────────────────────────
y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)

print("\n=== CLASSIFICATION REPORT (threshold=0.5) ===")
print(classification_report(y_test, y_pred, target_names=["Clean", "Laundering"], digits=4))

print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}  <-- primary metric for imbalanced problems")


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# -- Confusion Matrix --
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Pred Clean', 'Pred Launder'],
            yticklabels=['True Clean', 'True Launder'])
axes[0].set_title('Confusion Matrix (threshold=0.5)')

# -- ROC Curve --
RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[1], name=f"RF (AUC={roc_auc:.3f})")
axes[1].set_title('ROC Curve')
axes[1].plot([0,1],[0,1],'k--', linewidth=0.8)

# -- Precision-Recall Curve --
PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[2], name=f"RF (PR-AUC={pr_auc:.3f})")
axes[2].set_title('Precision-Recall Curve')
axes[2].axhline(y_test.mean(), color='red', linestyle='--', linewidth=0.8, label='Baseline (random)')
axes[2].legend()

plt.tight_layout()
plt.savefig("baseline_metrics.png", dpi=150)
plt.show()
print("Saved: baseline_metrics.png")


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False).head(15)

print("\n=== TOP 15 FEATURES ===")
print(importance_df.to_markdown(index=False))

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis', ax=ax)
ax.set_title('Top 15 Feature Importances (Random Forest)')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Saved: feature_importance.png")
