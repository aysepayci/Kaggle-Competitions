"""Projede tekrar tekrar kullanılan sabitler ve yardımcı fonksiyonlar."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SUBMISSIONS = ROOT / "submissions"

TARGET = "satisfaction"
ID = "id"
SEED = 42
N_FOLDS = 5

CATEGORICAL = ["Gender", "Customer Type", "Type of Travel", "Class"]
SERVICE = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness",
]


def load_data():
    """train, test ve y (0/1) döndürür. Categorical sütunlar 'category' dtype olur."""
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    y = train.pop(TARGET).astype(int)
    for col in CATEGORICAL:
        # train ve test aynı kategori listesini kullansın (kodlar tutarlı olsun)
        cats = sorted(train[col].unique())
        train[col] = pd.Categorical(train[col], categories=cats)
        test[col] = pd.Categorical(test[col], categories=cats)
    return train, test, y


def run_cv(make_model, X, y, X_test, name=""):
    """
    Stratified K-Fold CV.
    - make_model: her fold'da yeni (eğitilmemiş) model döndüren fonksiyon
    - Döndürür: oof (train için out-of-fold tahminler), test tahmini (fold ortalaması), CV AUC
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        model = make_model()
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS
        print(f"  [{name}] fold {fold}: AUC = {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.5f}")

    cv_auc = roc_auc_score(y, oof)
    print(f"  [{name}] CV AUC = {cv_auc:.5f}")
    return oof, test_pred, cv_auc


def fold_aucs(y, oof):
    """OOF tahminlerinden fold bazında AUC (run_cv ile aynı split'ler)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return np.array([roc_auc_score(y.iloc[va], oof[va]) for _, va in skf.split(oof, y)])


def save_submission(test_ids, pred, filename):
    SUBMISSIONS.mkdir(exist_ok=True)
    path = SUBMISSIONS / filename
    pd.DataFrame({ID: test_ids, TARGET: pred}).to_csv(path, index=False)
    print(f"  kaydedildi → {path.relative_to(ROOT)}")
