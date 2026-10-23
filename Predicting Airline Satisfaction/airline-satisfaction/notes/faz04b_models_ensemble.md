# Faz 04b — XGBoost, CatBoost ve Ensemble

Kod: `src/models.py` (modeller, OOF'ları `oof/` altına kaydeder) · `src/ensemble.py`

## Üç model, aynı aile, farklı ağaçlar
| Model | Ağaç nasıl büyür | Categorical | Süre / fold* |
|---|---|---|---|
| LightGBM | **leaf-wise**: en çok kazandıran yaprağı böler → derin, asimetrik | native (`category` dtype) | ~7 sn |
| XGBoost | **level-wise**: seviye seviye → dengeli | native (`enable_categorical`) | ~4 sn |
| CatBoost | **symmetric (oblivious)**: aynı seviyede her yerde aynı split | **ordered target encoding** | ~140 sn |

\* 2 çekirdekli ortamda, default parametreler. CatBoost default 1000 ağaç kurar (diğerleri 100) → yavaş ama güçlü.

## Tek model sonuçları (default parametreler)
| Model | CV AUC | Δ LightGBM | Fold kazanma |
|---|---|---|---|
| LightGBM | 0.95779 | — | — |
| XGBoost | 0.95798 | +0.00019 | 5/5 |
| **CatBoost** | **0.95850** | **+0.00071** | 5/5 |

CatBoost'un üstünlüğü 5 fold'da da tutarlı → gerçek fark. Muhtemel nedenler: daha çok ağaç ve symmetric ağaçların gürültülü (synthetic) veride daha az overfit etmesi.
**Dikkat:** Default'lar eşit şartlar değil (1000 vs 100 ağaç). Adil karşılaştırma Faz 05 tuning'den sonra.

## Ensemble
Modellerin OOF tahminleri arasındaki Spearman correlation **~0.955–0.967** → çok benzer ama birebir aynı değil; farklı hatalar yapıyorlar. Ensemble'ın işe yaraması için gereken şey bu.

| Yöntem | CV AUC | Δ CatBoost |
|---|---|---|
| Simple average (olasılık) | 0.95870 | +0.00020 |
| Rank average | 0.95872 | +0.00021 |
| **Weighted rank** (lgbm 0.25, xgb 0.20, cat 0.55) | **0.95877** | **+0.00027** |

Weighted ensemble, CatBoost'tan **5 fold'un 5'inde** daha iyi.

### Neden rank average?
AUC sadece sıralamaya bakar. Modellerin olasılık ölçekleri farklı olabilir (biri hep 0.3–0.7, diğeri 0–1). Olasılıkları direkt ortalarsak geniş ölçekli model baskın çıkar; sıralamaya çevirmek her modele eşit söz hakkı verir.

### Weighted'da overfitting riski
Ağırlıklar OOF üzerinde seçildi, yani aynı veride hem seçip hem ölçüyoruz → skor biraz iyimser. 3 ağırlık ve 700k satır olduğu için risk küçük; rank average ile farkı da sadece +0.00005. Şüphede kalırsan basit rank average daha güvenli.

## Neden OOF'ları kaydediyoruz?
- Ensemble denemek için modelleri tekrar eğitmeye gerek kalmıyor (CatBoost 12 dk sürüyor!).
- `models.py` fold bazında checkpoint tutuyor: yarıda kesilirse kaldığı yerden devam ediyor.

## Sonraki: Faz 05 — Hyperparameter tuning
Default'lar adil değil (100 vs 1000 ağaç). Optuna ile her modeli tune edip ensemble'ı tekrar kuracağız.
