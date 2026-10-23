# Faz 02 — Baseline

Kod: `src/common.py` (CV altyapısı) · `src/baseline.py` (modeller)

## Neden baseline?
Feature engineering ve tuning'e geçmeden önce bir **referans skor** lazım. Yoksa bir değişikliğin işe yarayıp yaramadığını bilemeyiz.

## Validation stratejisi: 5-fold Stratified CV
- **K-Fold:** Veriyi 5 parçaya bölüyoruz. Her seferinde 4 parçayla eğitip 1 parçada ölçüyoruz → 5 skorun ortalaması tek bir train/val split'ten daha güvenilir.
- **Stratified:** Her fold'da True/False oranı aynı kalır (%44).
- **OOF (out-of-fold):** Her train satırı için, o satırı *görmemiş* modelin tahmini. CV AUC bunlardan hesaplanır. Faz 06'da ensemble için de kullanacağız.
- **Test tahmini:** 5 fold modelinin ortalaması → tek modelden daha stabil.
- EDA'da train/test dağılımı aynı çıktığı için CV skoru leaderboard'u iyi yansıtmalı.

## Sonuçlar
| Model | CV AUC | Fold std | Süre |
|---|---|---|---|
| Sabit tahmin (sample_submission) | 0.5000 | — | — |
| Logistic Regression | 0.92156 | ~0.0005 | ~20 sn |
| LightGBM (default) | **0.95779** | ~0.0006 | ~40 sn |

## Neden LightGBM bu kadar önde?
EDA'daki bulgular doğrudan açıklıyor:
- **Eşikler** (puan 3→4, mesafe ~1300 km): Linear model bunları düz bir eğim olarak görür, ağaç ise tek bir split ile yakalar.
- **0 puanı:** LR 0'ı "en kötü puan" sanıyor. Oysa wifi=0 yolcularının %89'u memnun.
- **Interaction'lar** (Personal Travel × Class): LR'ye bunu elle vermemiz gerekir, ağaçlar kendisi bulur.
- **Ters U şeklindeki Age:** LR sadece "yaş arttıkça artar/azalır" diyebilir.

## Model seçimleri — neden bunlar?
| Seçim | Neden | Alternatif ve neden şimdilik değil |
|---|---|---|
| LogReg | Basit, hızlı, "linear sınır" referansı | SVM/KNN: 700k satırda çok yavaş |
| LightGBM | Tabular veride en güçlü aile, hızlı, NaN ve categorical'ı kendisi halleder | XGBoost/CatBoost: aynı aile, Faz 04'te karşılaştıracağız |
| Default parametreler | Baseline'ın amacı adil bir referans olmak | Tuning: Faz 05 |

## Faz 03 için fikirler
- LR farkını kapatmak için `is_zero` flag'leri ve one-hot puanlar → linear model ne kadar yaklaşır?
- `travel_class` interaction, hizmet kümelerinin ortalaması, `arr - dep` gecikme farkı
- Her yeni feature'ı **tek tek** ekleyip LightGBM CV AUC'sindeki değişimi ölçmek
