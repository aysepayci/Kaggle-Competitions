"""
Faz 03 — Her feature grubunu baseline'a TEK TEK ekleyip LightGBM CV AUC'yi ölçer.

Neden tek tek? Hepsini birden eklersek hangisinin işe yaradığını bilemeyiz.
Neden fold bazında karşılaştırma? Aynı fold'larda baseline'dan kaç kez daha iyi
olduğuna bakarak küçük farkların gürültü olup olmadığını anlarız.

Çalıştırma (proje klasöründen):
    python src/fe_experiments.py
    python src/fe_experiments.py travel_class zero_flags   # sadece seçilenler
"""
import sys
import time

import lightgbm as lgb
import pandas as pd

from common import ID, ROOT, SEED, fold_aucs, load_data, run_cv
from features import FEATURE_SETS, add_all


def lgbm():
    return lgb.LGBMClassifier(random_state=SEED, verbose=-1)


def evaluate(name, X, y, X_test):
    t = time.time()
    oof, _, cv = run_cv(lgbm, X, y, X_test, name)
    return {"experiment": name, "n_features": X.shape[1], "cv_auc": cv,
            "folds": fold_aucs(y, oof), "sec": round(time.time() - t)}


def main():
    names = sys.argv[1:] or ["baseline", *FEATURE_SETS, "all"]
    train, test, y = load_data()
    X0, T0 = train.drop(columns=[ID]), test.drop(columns=[ID])

    results = []
    for name in names:
        if name == "baseline":
            X, T = X0, T0
        elif name == "all":
            X, T = add_all(X0), add_all(T0)
        else:
            X, T = FEATURE_SETS[name](X0), FEATURE_SETS[name](T0)
        results.append(evaluate(name, X, y, T))

    df = pd.DataFrame(results)
    out = ROOT / "notes" / "faz03_results.csv"
    # Önceki çalıştırmalarla birleştir (aynı deney tekrar çalışırsa üzerine yazar)
    if out.exists():
        old = pd.read_csv(out)
        old = old[~old.experiment.isin(df.experiment)]
    else:
        old = pd.DataFrame()
    save = df.assign(folds=df.folds.apply(lambda f: " ".join(f"{a:.5f}" for a in f)))
    pd.concat([old, save]).to_csv(out, index=False)
    print(f"\nkaydedildi → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
