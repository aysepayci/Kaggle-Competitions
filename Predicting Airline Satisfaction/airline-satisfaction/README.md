# Predicting Airline Satisfaction

Kaggle Playground Series — Season 6, Episode 10
Yarışma: https://www.kaggle.com/competitions/playground-series-s6e10

## Amaç
Yolcu bilgileri ve uçuş deneyimi puanlarından yolcunun memnun olup olmadığını tahmin etmek.

## Yol Haritası
- [x] Faz 00 — Problemi tanı (veri, hedef, metrik)
- [x] Faz 01 — EDA (keşifsel veri analizi)
- [x] Faz 02 — Baseline model
- [x] Faz 03 — Feature engineering
- [x] Faz 04 — Gradient boosting modelleri + ensemble
- [x] Faz 05 — Hyperparameter tuning
- [ ] Faz 06 — Ensemble ve final submission

## Klasör Yapısı
```
data/raw/        # Kaggle verisi (repoya eklenmez)
data/processed/  # işlenmiş veri
notebooks/       # faz notebook'ları
notes/           # öğrenme notları
src/             # tekrar kullanılabilir kod
submissions/     # Kaggle'a gönderilen dosyalar
```

## Sonuçlar
| Faz | Model | CV | Public LB |
|-----|-------|----|-----------|
| 02 | Logistic Regression | 0.92156 | 0.92227 |
| 02 | LightGBM (default) | 0.95779 | 0.95723 |
| 03 | LogReg + one-hot features | 0.94877 | |
| 03 | LightGBM + tüm FE (gürültü, kullanılmadı) | 0.95780 | |
| 04a | LightGBM + orijinal veri (kötüleştirdi, kullanılmadı) | 0.95737 | |
| 04b | XGBoost (default) | 0.95798 | |
| 04b | CatBoost (default) | 0.95850 | |
| 04b | Ensemble: weighted rank (LGBM+XGB+Cat) | 0.95877 | 0.95837 |
| 05 | LightGBM (tuned) | 0.95893 | |
| 05 | XGBoost (tuned) | 0.95887 | |
| 05 | Ensemble: LGBM tuned + XGB tuned + Cat | **0.95919** | |
