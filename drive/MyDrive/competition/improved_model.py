import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data ──────────────────────────────────────────────────────────────
train = pd.read_csv("input/train.csv", index_col="Id")
test  = pd.read_csv("input/test.csv",  index_col="Id")

target = train["Drafted"].copy()
train  = train.drop(columns=["Drafted"])

# ── 2. Position-Specific Imputation ──────────────────────────────────────────
# Fill missing combine values with the median for that specific Position.
# A DT and a CB have very different distributions, so global medians are misleading.
combine_cols = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
                "Broad_Jump", "Agility_3cone", "Shuttle"]

# Compute position medians from train only (no leakage)
pos_medians = train.groupby("Position")[combine_cols].median()

def impute_by_position(df, pos_medians, global_medians):
    df = df.copy()
    for col in combine_cols:
        # Map each row's Position to its position-level median
        pos_fill = df["Position"].map(pos_medians[col])
        # Where position median is also NaN (unseen position), fall back to global
        df[col] = df[col].fillna(pos_fill).fillna(global_medians[col])
    return df

global_medians = train[combine_cols].median()
train = impute_by_position(train, pos_medians, global_medians)
test  = impute_by_position(test,  pos_medians, global_medians)

# ── 3. Feature Engineering ────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()

    # BMI
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # Speed score (combines weight and 40-yard dash)
    df["Speed_Score"] = (df["Weight"] * 200) / (df["Sprint_40yd"] ** 4)

    # Burst score (vertical + broad jump)
    df["Burst_Score"] = df["Vertical_Jump"] + df["Broad_Jump"]

    # Agility score (lower is better, so invert)
    df["Agility_Score"] = df["Agility_3cone"] + df["Shuttle"]

    # Athletic composite (normalised sum of key metrics)
    df["Athletic_Composite"] = (
        df["Vertical_Jump"].fillna(df["Vertical_Jump"].median()) +
        df["Broad_Jump"].fillna(df["Broad_Jump"].median()) / 10 +
        df["Bench_Press_Reps"].fillna(df["Bench_Press_Reps"].median())
    )

    # Weight-to-height ratio
    df["Weight_Height_Ratio"] = df["Weight"] / df["Height"]

    # Draft year recency (more recent = potentially different scouting)
    df["Year_Recency"] = df["Year"] - 2009

    # Age flag: missing age
    df["Age_Missing"] = df["Age"].isna().astype(int)

    return df

train = add_features(train)
test  = add_features(test)

# ── 3. Target Encoding for School ─────────────────────────────────────────────
# Use 5-fold out-of-fold to avoid leakage
school_te = np.zeros(len(train))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for tr_idx, val_idx in skf.split(train, target):
    school_mean = (
        train.iloc[tr_idx]
        .assign(Drafted=target.iloc[tr_idx])
        .groupby("School")["Drafted"]
        .mean()
    )
    school_te[val_idx] = train.iloc[val_idx]["School"].map(school_mean).fillna(target.mean())

train["School_TE"] = school_te

# For test: use full train mean
school_mean_full = train.assign(Drafted=target).groupby("School")["Drafted"].mean()
test["School_TE"] = test["School"].map(school_mean_full).fillna(target.mean())

# Drop raw School column
train = train.drop(columns=["School"])
test  = test.drop(columns=["School"])

# ── 4. Label Encoding for categoricals ───────────────────────────────────────
cat_cols = ["Player_Type", "Position_Type", "Position"]
le = LabelEncoder()
for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

# ── 5. Train LightGBM with cross-validation ───────────────────────────────────
feature_cols = [c for c in train.columns]

lgb_params = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "verbose":          -1,
    "random_state":     42,
}

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))
n_splits   = 5
skf        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(train, target)):
    X_tr, y_tr   = train.iloc[tr_idx][feature_cols], target.iloc[tr_idx]
    X_val, y_val = train.iloc[val_idx][feature_cols], target.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr,  label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
    )

    oof_preds[val_idx]  = model.predict(X_val[feature_cols])
    test_preds          += model.predict(test[feature_cols]) / n_splits

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

overall_auc = roc_auc_score(target, oof_preds)
print(f"\nOverall OOF AUC: {overall_auc:.4f}")

# ── 6. Create Submission ──────────────────────────────────────────────────────
submission = pd.read_csv("input/sample_submission.csv")
submission["Drafted"] = test_preds
submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")
print(submission.head())
