from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

# =======================
# SETTINGS
# =======================
DATA_PATH = Path("joint_dataset.csv")   

TEST_CHILD_FRAC = 0.30
RANDOM_SEED = 42

# Choose which target to train: "stress", "reaction", or "both"
TARGET_MODE = "both"

# Logistic Regression params
LOGREG_KWARGS = dict(
    solver="liblinear",        # stable for small/medium datasets, supports class_weight
    class_weight="balanced",   # helps if classes imbalanced
    max_iter=2000,
)

# =======================
# HELPERS
# =======================

def encode_binary_label(series: pd.Series, positive_value: str, negative_value: str) -> pd.Series:
    s = series.astype(str).str.strip()
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s == positive_value] = 1.0
    out[s == negative_value] = 0.0
    return out


def get_feature_columns(df):
    drop_exact = {
        "group", "child", "game",
        "stress_label", "reaction_label",
        "label_time_sec", "n_samples",
        "y_stress", "y_reaction",
    }

    feat_cols = []
    for c in df.columns:
        cl = c.lower()
        if c in drop_exact:
            continue
        # IMPORTANT: remove any time/window info (it leaks the protocol)
        if "time" in cl or "window" in cl:
            continue
        feat_cols.append(c)

    return feat_cols


def child_split(df: pd.DataFrame, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    children = df["child"].dropna().astype(str).unique()
    children = np.array(sorted(children))

    n_test = int(np.ceil(len(children) * test_frac))
    test_children = set(rng.choice(children, size=n_test, replace=False).tolist())

    test_df = df[df["child"].astype(str).isin(test_children)].copy()
    train_df = df[~df["child"].astype(str).isin(test_children)].copy()
    return train_df, test_df


def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, y_col: str, title: str):
    feat_cols = get_feature_columns(train_df)

    print("First 30 feature cols:", feat_cols[:30])
    assert all(("time" not in c.lower() and "window" not in c.lower()) for c in feat_cols)


    # Convert features to numeric (robust)
    X_train = train_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X_test  = test_df[feat_cols].apply(pd.to_numeric, errors="coerce")

    y_train = train_df[y_col].astype(int).values
    y_test  = test_df[y_col].astype(int).values

    # Simple missing handling: fill NaNs with train medians
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**LOGREG_KWARGS)),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # predict_proba exists for liblinear
    y_prob = model.predict_proba(X_test)[:, 1]

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    print(f"Num features: {len(feat_cols)}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"F1:               {f1:.4f}")
    print(f"ROC-AUC:           {auc:.4f}")
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)

    # Encode labels
    df["y_stress"] = encode_binary_label(df["stress_label"], "Stress", "No Stress")
    df["y_reaction"] = encode_binary_label(df["reaction_label"], "Reaction", "No Reaction")

    # Drop rows that don't have the label we need (done per task)
    # But we must do the child split on the full df so the same children go to test for both tasks.
    train_df, test_df = child_split(df, TEST_CHILD_FRAC, RANDOM_SEED)

    print("Children total:", df["child"].nunique())
    print("Children train:", train_df["child"].nunique())
    print("Children test: ", test_df["child"].nunique())
    print("Test child fraction (actual):", test_df["child"].nunique() / max(1, df["child"].nunique()))

    if TARGET_MODE in ("stress", "both"):
        tr = train_df.dropna(subset=["y_stress"]).copy()
        te = test_df.dropna(subset=["y_stress"]).copy()
        train_and_eval(tr, te, "y_stress", "Logistic Regression — Stress vs No Stress")

    if TARGET_MODE in ("reaction", "both"):
        tr = train_df.dropna(subset=["y_reaction"]).copy()
        te = test_df.dropna(subset=["y_reaction"]).copy()
        train_and_eval(tr, te, "y_reaction", "Logistic Regression — Reaction vs No Reaction")


if __name__ == "__main__":
    main()
