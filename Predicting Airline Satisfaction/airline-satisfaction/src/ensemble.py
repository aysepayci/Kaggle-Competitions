"""
Faz 04b — Ensemble: modellerin OOF tahminlerini birleştir.

Önce src/models.py ile modelleri çalıştır (oof/<model>_oof.npy ve _test.npy oluşur).

Denenen yöntemler:
1. Simple average   : olasılıkların ortalaması
2. Rank average     : önce her modeli sıralamaya (rank) çevir, sonra ortala.
                      AUC sadece sıralamaya baktığı için, modellerin olasılık ölçekleri
                      farklıysa (biri 0.4–0.6, diğeri 0–1 arası) daha adil birleştirir.
3. Weighted average : OOF üzerinde AUC'yi en yükselten ağırlıklar (grid search).

Çalıştırma (proje klasöründen):
    python src/ensemble.py                # lgbm xgb cat
    python src/ensemble.py lgbm cat       # seçili modeller
"""
import itertools
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import ID, ROOT, fold_aucs, load_data, save_submission

OOF_DIR = ROOT / "oof"


def rank(a):
    return rankdata(a) / len(a)


def best_weights(oofs, y, step=0.05):
    """Toplamı 1 olan ağırlıkları grid ile dener (model sayısı az olduğu için yeterli)."""
    grid = np.arange(0, 1 + 1e-9, step)
    best = (None, -1)
    for w in itertools.product(grid, repeat=len(oofs) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        w = np.array([*w, 1 - sum(w)])
        score = roc_auc_score(y, sum(wi * rank(o) for wi, o in zip(w, oofs)))
        if score > best[1]:
            best = (w, score)
    return best


def main():
    names = sys.argv[1:] or ["lgbm", "xgb", "cat"]
    train, test, y = load_data()
    oofs = [np.load(OOF_DIR / f"{n}_oof.npy") for n in names]
    tests = [np.load(OOF_DIR / f"{n}_test.npy") for n in names]

    print("Tek modeller:")
    for n, o in zip(names, oofs):
        print(f"  {n:6s} CV AUC = {roc_auc_score(y, o):.5f}")

    # Modeller ne kadar benzer hata yapıyor? Correlation ~1 ise ensemble az kazandırır.
    print("\nOOF correlation (Spearman = sıralama benzerliği):")
    print(pd.DataFrame(dict(zip(names, oofs))).corr(method="spearman").round(4).to_string())

    simple = np.mean(oofs, axis=0)
    ranked = np.mean([rank(o) for o in oofs], axis=0)
    w, w_score = best_weights(oofs, y)
    best_single = max(roc_auc_score(y, o) for o in oofs)

    print("\nEnsemble:")
    for label, score in [("simple average", roc_auc_score(y, simple)),
                         ("rank average", roc_auc_score(y, ranked)),
                         (f"weighted {dict(zip(names, map(float, w.round(2))))}", w_score)]:
        print(f"  {label:40s} CV AUC = {score:.5f} (en iyi tek modele göre {score - best_single:+.5f})")

    weighted_oof = sum(wi * rank(o) for wi, o in zip(w, oofs))
    print("\nWeighted ensemble fold AUC'leri:", fold_aucs(y, weighted_oof).round(5))

    pred = sum(wi * rank(t) for wi, t in zip(w, tests))
    save_submission(test[ID], pred, f"faz04_ensemble_{'_'.join(names)}.csv")


if __name__ == "__main__":
    main()
