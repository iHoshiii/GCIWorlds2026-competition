import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

SEED     = 42
N_SPLITS = 10

# ── 1. Load data ───────────────────────────────────────────────────────────────
train = pd.read_csv("input/train.csv", index_col="Id")
test  = pd.read_csv("input/test.csv",  index_col="Id")

target = train["Drafted"].copy()
train  = train.drop(columns=["Drafted"])

# ── 2. Missing-count feature (before imputation — captures who skipped combine) ─
combine_cols = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
                "Broad_Jump", "Agility_3cone", "Shuttle"]

train["Missing_Count"] = train[combine_cols].isnull().sum(axis=1)
test["Missing_Count"]  = test[combine_cols].isnull().sum(axis=1)

# ── 3. Position-specific imputation ───────────────────────────────────────────
pos_medians    = train.groupby("Position")[combine_cols].median()
global_medians = train[combine_cols].median()

def impute_by_position(df, pos_medians, global_medians):
    df = df.copy()
    for col in combine_cols:
        pos_fill = df["Position"].map(pos_medians[col])
        df[col]  = df[col].fillna(pos_fill).fillna(global_medians[col])
    return df

train = impute_by_position(train, pos_medians, global_medians)
test  = impute_by_position(test,  pos_medians, global_medians)

# ── 4. Position z-scores ───────────────────────────────────────────────────────
pos_stats = train.groupby("Position")[combine_cols].agg(["mean", "std"])

def add_position_zscores(df, pos_stats):
    df = df.copy()
    for col in combine_cols:
        mu  = df["Position"].map(pos_stats[col]["mean"])
        sig = df["Position"].map(pos_stats[col]["std"])
        df[f"{col}_z"] = (df[col] - mu) / (sig + 1e-6)
    return df

train = add_position_zscores(train, pos_stats)
test  = add_position_zscores(test,  pos_stats)

# ── 5. Conference tier ─────────────────────────────────────────────────────────
sec   = {'Alabama','LSU','Georgia','Florida','Auburn','Tennessee','Mississippi',
         'Mississippi St.','Arkansas','South Carolina','Kentucky','Vanderbilt',
         'Missouri','Texas A&M'}
big10 = {'Ohio St.','Michigan','Penn St.','Wisconsin','Iowa','Michigan St.',
         'Nebraska','Minnesota','Northwestern','Indiana','Purdue','Illinois',
         'Maryland','Rutgers'}
acc   = {'Clemson','Florida St.','Miami (FL)','Virginia Tech','North Carolina',
         'North Carolina St.','Boston Col.','Virginia','Georgia Tech','Pittsburgh',
         'Syracuse','Wake Forest','Louisville','Duke'}
big12 = {'Oklahoma','Texas','Baylor','TCU','Oklahoma St.','Kansas St.',
         'Iowa St.','West Virginia','Kansas','Texas Tech'}
pac12 = {'USC','UCLA','Oregon','Washington','Stanford','California',
         'Arizona St.','Arizona','Utah','Colorado','Oregon St.','Washington St.'}

def conf_tier(s):
    if s in sec or s in big10: return 3
    if s in acc or s in big12 or s in pac12: return 2
    return 1

train["Conference_Tier"] = train["School"].map(conf_tier)
test["Conference_Tier"]  = test["School"].map(conf_tier)

# ── 6. Engineered features ─────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    df["BMI"]                = df["Weight"] / (df["Height"] ** 2)
    df["Speed_Score"]        = (df["Weight"] * 200) / (df["Sprint_40yd"] ** 4)
    df["Burst_Score"]        = df["Vertical_Jump"] + df["Broad_Jump"]
    df["Agility_Score"]      = df["Agility_3cone"] + df["Shuttle"]
    df["Weight_Height_Ratio"]= df["Weight"] / df["Height"]
    df["Year_Recency"]       = df["Year"] - 2009
    df["Age_Missing"]        = df["Age"].isna().astype(int)
    # Relative agility: lower 3cone+shuttle vs position peers = better
    df["Agility_Rank_Proxy"] = df["Agility_3cone"] / (df["Shuttle"] + 1e-6)
    return df

train = add_features(train)
test  = add_features(test)

# ── 7. Smoothed target encoding (Bayesian / additive smoothing) ────────────────
# Formula: (n_pos * mean_pos + k * global_mean) / (n_pos + k)
# k controls how much we shrink rare categories toward the global mean.
SMOOTH_K = 20
global_mean = target.mean()

def smoothed_te_train(train_df, target_s, col, k=SMOOTH_K):
    """OOF smoothed TE for training set."""
    oof = np.zeros(len(train_df))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in skf.split(train_df, target_s):
        stats = (
            train_df.iloc[tr_idx]
            .assign(_y=target_s.iloc[tr_idx])
            .groupby(col)["_y"]
            .agg(["sum", "count"])
        )
        stats["te"] = (stats["sum"] + k * global_mean) / (stats["count"] + k)
        oof[val_idx] = train_df.iloc[val_idx][col].map(stats["te"]).fillna(global_mean)
    return oof

def smoothed_te_test(train_df, target_s, col, k=SMOOTH_K):
    """Full-train smoothed TE for test set."""
    stats = (
        train_df.assign(_y=target_s)
        .groupby(col)["_y"]
        .agg(["sum", "count"])
    )
    stats["te"] = (stats["sum"] + k * global_mean) / (stats["count"] + k)
    return test[col].map(stats["te"]).fillna(global_mean)

for col in ["School", "Position", "Position_Type", "Player_Type"]:
    train[f"{col}_TE"] = smoothed_te_train(train, target, col)
    test[f"{col}_TE"]  = smoothed_te_test(train, target, col)

# Drop raw high-cardinality categoricals
train = train.drop(columns=["School"])
test  = test.drop(columns=["School"])

# ── 8. Interaction features ────────────────────────────────────────────────────
for df in [train, test]:
    df["Speed_x_PosTE"]   = df["Speed_Score"]   * df["Position_TE"]
    df["BMI_x_PosTE"]     = df["BMI"]            * df["Position_TE"]
    df["Burst_x_PosTE"]   = df["Burst_Score"]    * df["Position_TE"]
    df["Agility_x_PosTE"] = df["Agility_Score"]  * df["Position_TE"]
    df["School_x_Conf"]   = df["School_TE"]      * df["Conference_Tier"]

# ── 9. Label-encode remaining categoricals ─────────────────────────────────────
cat_cols = ["Player_Type", "Position_Type", "Position"]
le = LabelEncoder()
for col in cat_cols:
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))

# ── 10. Train LightGBM (10-fold, tuned params) ────────────────────────────────
feature_cols = list(train.columns)
print(f"Total features: {len(feature_cols)}")

lgb_params = {
    "objective":          "binary",
    "metric":             "auc",
    "learning_rate":      0.02,
    "num_leaves":         127,
    "max_depth":          -1,
    "min_child_samples":  15,
    "feature_fraction":   0.7,
    "bagging_fraction":   0.8,
    "bagging_freq":       1,
    "reg_alpha":          0.05,
    "reg_lambda":         1.0,
    "min_split_gain":     0.01,
    "verbose":            -1,
    "random_state":       SEED,
}

oof_preds  = np.zeros(len(train))
test_preds = np.zeros(len(test))
skf        = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_idx, val_idx) in enumerate(skf.split(train, target)):
    X_tr,  y_tr  = train.iloc[tr_idx][feature_cols], target.iloc[tr_idx]
    X_val, y_val = train.iloc[val_idx][feature_cols], target.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr,  label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)],
    )

    oof_preds[val_idx] += model.predict(X_val[feature_cols])
    test_preds         += model.predict(test[feature_cols]) / N_SPLITS

    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold+1:2d}  AUC: {fold_auc:.4f}  best_iter: {model.best_iteration}")

overall_auc = roc_auc_score(target, oof_preds)
print(f"\nOverall OOF AUC: {overall_auc:.4f}")

# ── 11. Create submission ──────────────────────────────────────────────────────
submission = pd.read_csv("input/sample_submission.csv")
submission["Drafted"] = test_preds
submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")
print(submission.head())
print(f"Prediction range: {test_preds.min():.4f} — {test_preds.max():.4f}")
