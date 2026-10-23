"""
Faz 04a — Orijinal veriyi ek training verisi olarak eklemek işe yarıyor mu?

Orijinal satırlar SADECE training fold'larına eklenir (bkz. common.run_cv),
validation hep yarışma verisi → CV skoru baseline ile adil karşılaştırılabilir.

Çalıştırma (proje klasöründen):
    python src/original_data.py   (~4 dk)
"""
import lightgbm as lgb
import numpy as np

from common import ID, SEED, fold_aucs, load_data, load_original, run_cv


def lgbm():
    return lgb.LGBMClassifier(random_state=SEED, verbose=-1)


def main():
    train, test, y = load_data()
    X, X_test = train.drop(columns=[ID]), test.drop(columns=[ID])
    X_orig, y_orig = load_original(X.columns)
    print(f"yarışma: {X.shape}, orijinal: {X_orig.shape}")

    oof_base, _, cv_base = run_cv(lgbm, X, y, X_test, "LGBM")
    oof_orig, _, cv_orig = run_cv(lgbm, X, y, X_test, "LGBM + original", extra=(X_orig, y_orig))
    report(y, oof_base, oof_orig, cv_base, cv_orig)

    # Varyant: modele satırın nereden geldiğini söyle (is_original) → ağaç iki kaynağı ayırabilsin
    oof_flag, _, cv_flag = run_cv(lgbm, X.assign(is_original=0), y, X_test.assign(is_original=0),
                                  "LGBM + original + flag", extra=(X_orig.assign(is_original=1), y_orig))
    report(y, oof_base, oof_flag, cv_base, cv_flag)

    diagnose(X, y, X_orig, y_orig)


def report(y, oof_base, oof_new, cv_base, cv_new):
    delta = fold_aucs(y, oof_new) - fold_aucs(y, oof_base)
    print(f"  Δ CV = {cv_new - cv_base:+.5f} | fold farkları (×10⁻⁴): {np.round(delta * 1e4, 1)}"
          f" | {int((delta > 0).sum())}/5 fold'da daha iyi\n")


def diagnose(X, y, X_orig, y_orig):
    """İki veri ne kadar 'aynı dünya'? Birinde eğitip diğerinde ölç."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    p = cross_val_predict(lgbm(), X_orig, y_orig, cv=5, method="predict_proba")[:, 1]
    print("orijinal, kendi içinde CV AUC :", round(roc_auc_score(y_orig, p), 4))
    m = lgbm().fit(X_orig, y_orig)
    print("orijinalle eğit → yarışma AUC :", round(roc_auc_score(y, m.predict_proba(X)[:, 1]), 4))
    m = lgbm().fit(X, y)
    print("yarışmayla eğit → orijinal AUC:", round(roc_auc_score(y_orig, m.predict_proba(X_orig)[:, 1]), 4))


if __name__ == "__main__":
    main()
