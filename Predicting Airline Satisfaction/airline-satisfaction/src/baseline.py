"""
Faz 02 — Baseline modeller (feature engineering YOK).

Amaç: Sonraki her değişikliği karşılaştıracağımız bir referans skor.

Çalıştırma (proje klasöründen):
    python src/baseline.py          # iki model
    python src/baseline.py lr       # sadece Logistic Regression
    python src/baseline.py lgbm     # sadece LightGBM
"""
import sys

import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CATEGORICAL, ID, SEED, load_data, run_cv, save_submission


def make_logreg(numeric_cols):
    # Linear model: categorical → one-hot, numeric → NaN doldur + ölçekle.
    # Ölçekleme şart: Flight Distance (~1000) ile puanlar (0–5) aynı ölçekte olmalı.
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
    ])
    return lambda: make_pipeline(pre, LogisticRegression(max_iter=1000))


def make_lgbm():
    # Tree model: ölçekleme gerekmez, NaN'ı ve 'category' dtype'ı kendisi halleder.
    # Bilerek default parametreler — tuning Faz 05'te.
    return lambda: lgb.LGBMClassifier(random_state=SEED, verbose=-1)


def main():
    which = sys.argv[1:] or ["lr", "lgbm"]
    train, test, y = load_data()
    X = train.drop(columns=[ID])
    X_test = test.drop(columns=[ID])
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]
    print(f"X: {X.shape}, X_test: {X_test.shape}")

    if "lr" in which:
        _, pred, _ = run_cv(make_logreg(numeric_cols), X, y, X_test, "LogReg")
        save_submission(test[ID], pred, "faz02_logreg.csv")
    if "lgbm" in which:
        _, pred, _ = run_cv(make_lgbm(), X, y, X_test, "LightGBM")
        save_submission(test[ID], pred, "faz02_lgbm.csv")


if __name__ == "__main__":
    main()
