"""
Faz 04b — LightGBM / XGBoost / CatBoost karşılaştırması (default parametreler).

Her modelin OOF ve test tahminleri oof/<model>/ altına fold fold kaydedilir:
- Ensemble (src/ensemble.py) bu dosyaları kullanır, modelleri yeniden eğitmez.
- Checkpoint: Script yarıda kesilirse tekrar çalıştırınca biten fold'ları atlar.

Çalıştırma (proje klasöründen):
    python src/models.py lgbm
    python src/models.py xgb
    python src/models.py cat
    python src/models.py cat --budget 150   # ~150 sn sonra durur, sonra kaldığı yerden devam
"""
import argparse
import time

import catboost as cb
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import CATEGORICAL, ID, N_FOLDS, ROOT, SEED, load_data

OOF_DIR = ROOT / "oof"

# Üçü de gradient boosting; farkları ağacı nasıl kurdukları:
# - LightGBM: leaf-wise büyür (en çok kazandıran yaprağı böler) → hızlı, derin/asimetrik ağaçlar
# - XGBoost : level-wise büyür (seviye seviye) → daha dengeli ağaçlar
# - CatBoost: symmetric (oblivious) ağaçlar + categorical'lar için ordered target encoding
MODELS = {
    "lgbm": lambda: lgb.LGBMClassifier(random_state=SEED, verbose=-1),
    "xgb": lambda: xgb.XGBClassifier(enable_categorical=True, tree_method="hist", random_state=SEED),
    "cat": lambda: cb.CatBoostClassifier(cat_features=CATEGORICAL, random_seed=SEED, verbose=0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("--budget", type=float, default=None, help="saniye; dolunca yeni fold'a başlamaz")
    args = ap.parse_args()

    train, test, y = load_data()
    X, X_test = train.drop(columns=[ID]), test.drop(columns=[ID])
    out = OOF_DIR / args.model
    out.mkdir(parents=True, exist_ok=True)

    start = time.time()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)  # run_cv ile aynı fold'lar
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        if (out / f"fold{fold}_test.npy").exists():
            print(f"  fold {fold}: zaten var, atlandı")
            continue
        if args.budget and time.time() - start > args.budget:
            print("  süre bütçesi doldu → tekrar çalıştırınca kaldığı yerden devam eder")
            return
        t = time.time()
        model = MODELS[args.model]()
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        val = model.predict_proba(X.iloc[va_idx])[:, 1]
        np.save(out / f"fold{fold}_val.npy", val)
        np.save(out / f"fold{fold}_test.npy", model.predict_proba(X_test)[:, 1])
        print(f"  fold {fold}: AUC = {roc_auc_score(y.iloc[va_idx], val):.5f} ({time.time() - t:.0f} sn)")

    # Tüm fold'lar bitti → birleştir
    oof = np.zeros(len(X))
    for fold, (_, va_idx) in enumerate(skf.split(X, y), 1):
        oof[va_idx] = np.load(out / f"fold{fold}_val.npy")
    test_pred = np.mean([np.load(out / f"fold{f}_test.npy") for f in range(1, N_FOLDS + 1)], axis=0)
    np.save(OOF_DIR / f"{args.model}_oof.npy", oof)
    np.save(OOF_DIR / f"{args.model}_test.npy", test_pred)
    print(f"  [{args.model}] CV AUC = {roc_auc_score(y, oof):.5f}")


if __name__ == "__main__":
    main()
