"""
Faz 05 — Optuna ile hyperparameter tuning.

Strateji (hız için):
- Tuning tek bir holdout üzerinde: CV'deki fold 1 (560k train / 140k validation).
  5-fold yerine 1 split → her deneme 5 kat hızlı. En iyi parametreler sonra
  `python src/models.py <model> --tuned` ile tam 5-fold CV'de doğrulanır.
- learning_rate sabit (0.1), ağaç sayısını early stopping bulur (en fazla 3000).
  lr ile ağaç sayısı birbirine bağlı; ikisini birden aramak boşa deneme harcar.
- Çalışma SQLite'a kaydedilir (oof/optuna.db) → script kesilirse tekrar çalıştırınca
  denemeler kaldığı yerden devam eder.

Çalıştırma (proje klasöründen):
    python src/tune.py lgbm --trials 30
    python src/tune.py cat --trials 25 --timeout 3600   # en fazla 1 saat
Sonuç: params/<model>.json
"""
import argparse
import json
import warnings

import catboost as cb
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import ID, N_FOLDS, ROOT, SEED, load_data
from models import OOF_DIR, PARAMS_DIR, build

warnings.filterwarnings("ignore", category=FutureWarning)  # lightgbm eval_set uyarısı

LR = 0.1
MAX_TREES = 3000
PATIENCE = 100  # 100 ağaç boyunca validation AUC artmazsa dur


def search_space(trial, name):
    """Her model için aranacak parametreler ve aralıkları."""
    if name == "lgbm":
        return dict(
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),          # ağaç karmaşıklığı
            min_child_samples=trial.suggest_int("min_child_samples", 10, 500, log=True),  # yaprakta min satır → gürültüye karşı
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),     # her ağaçta feature oranı
            subsample=trial.suggest_float("subsample", 0.5, 1.0),                   # her ağaçta satır oranı
            subsample_freq=1,
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),         # L1
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),       # L2
        )
    if name == "xgb":
        return dict(
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_weight=trial.suggest_float("min_child_weight", 1, 100, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 1e-3, 5, log=True),                  # split için min kazanç
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
    if name == "cat":
        return dict(
            depth=trial.suggest_int("depth", 4, 10),                                # symmetric ağaç derinliği
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0, 1),
        )
    raise ValueError(name)


def fit_early_stopping(name, params, X_tr, y_tr, X_va, y_va):
    """Early stopping ile eğitir; (validation AUC, en iyi ağaç sayısı) döndürür."""
    if name == "lgbm":
        m = build(name, {**params, "learning_rate": LR, "n_estimators": MAX_TREES})
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        n = m.best_iteration_
    elif name == "xgb":
        m = build(name, {**params, "learning_rate": LR, "n_estimators": MAX_TREES,
                         "early_stopping_rounds": PATIENCE, "eval_metric": "auc"})
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        n = m.best_iteration + 1
    else:
        m = build(name, {**params, "learning_rate": LR, "iterations": MAX_TREES,
                         "early_stopping_rounds": PATIENCE, "eval_metric": "AUC"})
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        n = m.get_best_iteration() + 1
    return roc_auc_score(y_va, m.predict_proba(X_va)[:, 1]), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["lgbm", "xgb", "cat"])
    ap.add_argument("--trials", type=int, default=30, help="toplam hedef deneme sayısı")
    ap.add_argument("--timeout", type=float, default=None, help="bu çalıştırma için saniye sınırı")
    args = ap.parse_args()

    train, _, y = load_data()
    X = train.drop(columns=[ID])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(iter(skf.split(X, y)))  # fold 1
    X_tr, y_tr, X_va, y_va = X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx]

    OOF_DIR.mkdir(exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = f"sqlite:///{(OOF_DIR / 'optuna.db').as_posix()}"
    study = optuna.create_study(study_name=args.model, direction="maximize",
                                storage=storage, load_if_exists=True)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    # TPE: önceki denemelerden öğrenerek arar. Seed'e tamamlanan deneme sayısını ekliyoruz;
    # yoksa script her yeniden başladığında aynı denemeleri tekrar üretir.
    study.sampler = optuna.samplers.TPESampler(seed=SEED + done)
    if done == 0:
        # 1. deneme: default parametreler → tuning'in en azından default'u geçtiğini görelim
        study.enqueue_trial({}, skip_if_exists=True)

    def objective(trial):
        auc, n = fit_early_stopping(args.model, search_space(trial, args.model), X_tr, y_tr, X_va, y_va)
        trial.set_user_attr("n_estimators", n)
        return auc

    def log(study, trial):
        best = study.best_trial
        print(f"  deneme {trial.number:3d}: AUC = {trial.value:.5f} ({trial.user_attrs['n_estimators']} ağaç)"
              f" | en iyi = {best.value:.5f} (#{best.number})", flush=True)

    remaining = max(args.trials - done, 0)
    print(f"[{args.model}] tamamlanan: {done}, hedef: {args.trials}")
    study.optimize(objective, n_trials=remaining, timeout=args.timeout, callbacks=[log])

    best = study.best_trial
    tree_key = "iterations" if args.model == "cat" else "n_estimators"
    params = {**search_space(optuna.trial.FixedTrial(best.params), args.model),
              "learning_rate": LR, tree_key: best.user_attrs["n_estimators"]}
    PARAMS_DIR.mkdir(exist_ok=True)
    out = PARAMS_DIR / f"{args.model}.json"
    out.write_text(json.dumps({"holdout_auc": best.value, "trial": best.number,
                               "n_trials": len(study.trials), "params": params}, indent=2))
    print(f"\nen iyi holdout AUC = {best.value:.5f} → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
