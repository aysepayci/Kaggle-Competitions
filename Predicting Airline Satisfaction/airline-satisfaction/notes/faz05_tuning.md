# Faz 05 — Hyperparameter Tuning (Optuna)

Kod: `src/tune.py` · Sonuçlar: `params/<model>.json` · Doğrulama: `python src/models.py <model> --tuned`

## Strateji ve nedenleri
| Karar | Neden | Alternatif ve neden değil |
|---|---|---|
| **Optuna (TPE sampler)** | Önceki denemelerden öğrenir, iyi bölgelere yoğunlaşır | Grid search: 7 parametrede patlar. Random search: öğrenmez, daha çok deneme ister |
| **Tek holdout** (fold 1) üzerinde tune | Her deneme 5 kat hızlı | 5-fold içinde tune: daha güvenilir ama 5 kat yavaş. Sonuçları zaten 5-fold'da doğruluyoruz |
| **learning_rate sabit 0.1** + **early stopping** | Ağaç sayısını early stopping buluyor; lr ve ağaç sayısı birbirine bağlı, ikisini aramak boşa deneme | Daha düşük lr (0.02) biraz daha iyi olabilir ama 5 kat yavaş — Faz 06'da denenebilir |
| **SQLite'a kayıt** | Script kesilirse kaldığı yerden devam | — |
| **İlk deneme = default parametreler** | Tuning'in default'u geçtiğini görmek için referans | — |

### Bir hata ve dersi
İlk versiyonda sampler'ın seed'i sabitti → script her yeniden başlatıldığında **aynı denemeleri tekrar üretti** (deneme 5–9 = 0–4). Çözüm: seed'e tamamlanan deneme sayısını eklemek. Ders: resume edilebilir sistemlerde rastgeleliğin de "kaldığı yerden" devam etmesi gerekir.

## Sonuçlar (5-fold CV)
| Model | Default | Tuned | Δ | Holdout (fold 1) |
|---|---|---|---|---|
| LightGBM | 0.95779 | **0.95893** | +0.00114 | 0.95781 → 0.95931 |
| XGBoost | 0.95798 | **0.95887** | +0.00089 | 0.95792 → 0.95922 |
| CatBoost | 0.95850 | *bilgisayarında çalıştırılacak* | | |

- Tuned LightGBM ve XGBoost artık default CatBoost'u (0.95850) ve hatta eski 3'lü ensemble'ı (0.95877) tek başına geçiyor.
- Kazancın büyük kısmı **daha çok ağaçtan** geldi: LightGBM default (100 ağaç) → aynı parametrelerle early stopping (311 ağaç) tek başına +0.0011 (holdout). Parametre araması üstüne ~+0.0004 ekledi.
- Holdout'taki kazanç 5-fold'da da tuttu → holdout'a overfit etmemişiz.

### Bulunan parametrelerden çıkarımlar
- **LightGBM:** `min_child_samples=136` (default 20), `colsample_bytree=0.57` → yapraklarda daha çok örnek ve her ağaçta daha az feature = daha güçlü **regularization**. Gürültülü synthetic veri için mantıklı.
- **XGBoost:** `max_depth=10` ama `min_child_weight=35` ve `reg_alpha=7.3` → derin ağaçlar, ama güçlü regularization ile dengelenmiş.

## Ensemble (tuned LGBM + tuned XGB + default Cat)
| Yöntem | CV AUC |
|---|---|
| Rank average | 0.95917 |
| **Weighted** (lgbm 0.45, xgb 0.35, cat 0.20) | **0.95919** |

Tuned modeller birbirine daha çok benziyor (Spearman 0.978 vs önce 0.955) → ensemble kazancı benzer kaldı (+0.00026).
