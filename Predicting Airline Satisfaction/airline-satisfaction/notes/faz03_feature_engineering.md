# Faz 03 — Feature Engineering

Kod: `src/features.py` · `src/fe_experiments.py` (LightGBM) · `src/lr_onehot.py` (LogReg)
Ham sonuçlar: `notes/faz03_results.csv`

## Tahmin
"En çok **cluster_means** (hizmet kümelerinin ortalaması) artırır."

## LightGBM sonuçları (her grup baseline'a tek tek eklendi)
| Deney | CV AUC | Δ baseline | Fold kazanma | Fold farkları (×10⁻⁴) |
|---|---|---|---|---|
| baseline | 0.95779 | — | — | — |
| travel_class | 0.95783 | +0.00004 | 3/5 | +0.1 −0.5 +0.5 −0.1 +2.2 |
| zero_flags | 0.95777 | −0.00002 | 2/5 | −0.9 −0.4 −0.1 +0.1 +0.4 |
| cluster_means | 0.95775 | −0.00004 | 1/5 | −0.7 −0.7 −0.9 −0.9 +1.1 |
| delay_diff | 0.95775 | −0.00004 | 2/5 | −0.2 −1.1 +0.1 +0.2 −0.7 |
| all | 0.95780 | +0.00001 | 2/5 | −0.1 −0.2 −1.0 +0.7 +1.0 |

**Sonuç: Hiçbiri anlamlı bir fark yaratmadı.** Tüm farklar ±0.0001 içinde ve fold'larda yönleri karışık → **gürültü**.
Tahmin ❌: cluster_means en kötülerden biri oldu (5 fold'un 4'ünde baseline'dan kötü).

### Neden?
Bu feature'ların hepsi **mevcut sütunlardan türetilmiş** bilgi. Tree modeller bunları zaten split'lerle öğreniyor:
- `travel_class` → ağaç önce Type of Travel'a, sonra Class'a bölerek aynı şeyi bulur
- `is_zero` → `wifi <= 0` tek bir split
- `cluster_means` → ağaç zaten tek tek puanlara bakıyor; ortalama bilgi kaybettirebilir (hangi hizmetin kötü olduğu kaybolur)
- `delay_diff` → gecikmelerin zaten zayıf etkisi var

## Logistic Regression'da ise büyük fark
| Model | CV AUC |
|---|---|
| LogReg baseline (puanlar sayı olarak) | 0.92156 |
| LogReg + one-hot puanlar + Age/mesafe binning + travel_class | **0.94877** (+0.027) |

Aynı bilgi, sadece **farklı temsil**. Linear model eşikleri, "0 = N/A"yı ve ters U'yu kendisi öğrenemez; one-hot/binning ile bunları açıkça verince fark 0.036'dan 0.009'a indi.

## Dersler
1. **Feature engineering modele bağlıdır.** Linear modelde hayati olan şey, tree modelde gereksiz olabilir.
2. **Küçük farklara fold bazında bak.** +0.00004 tek başına "iyileşme" gibi görünür ama 5 fold'un 2'sinde kötüleşiyorsa gürültüdür.
3. **Negatif sonuç da sonuçtur.** Ne işe yaramadığını bilmek, ileride zaman kaybetmemizi önler.

## Karar
LightGBM için şimdilik feature eklemiyoruz (`travel_class` kenarda dursun). Bundan sonraki kazanç başka yerden gelmeli:
- **Farklı modeller** (XGBoost, CatBoost) ve onların **ensemble**'ı — Faz 04
- **Orijinal veri seti** (Kaggle "Airline Passenger Satisfaction", ~130k gerçek satır) ek training verisi olarak — Playground yarışmalarında yaygın ve etkili bir teknik
- **Hyperparameter tuning** — Faz 05
