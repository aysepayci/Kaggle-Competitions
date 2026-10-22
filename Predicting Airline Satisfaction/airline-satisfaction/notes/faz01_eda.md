# Faz 01 — EDA Özeti

Detaylar ve grafikler: `notebooks/faz01_eda.ipynb`

## En önemli bulgular
| # | Bulgu | Modele etkisi |
|---|---|---|
| 1 | **Online boarding** en güçlü feature (single-feature AUC 0.84). Sonra Inflight entertainment (0.75), Seat comfort (0.73) | Feature importance'ta üstte görmeyi bekliyoruz |
| 2 | Puanlarda **eşik etkisi**: 1–3 düşük, 4–5 yüksek memnuniyet | İlişki doğrusal değil → tree modeller avantajlı |
| 3 | **0 puanı N/A değil, sinyal taşıyor**: wifi=0 → %89 memnun (genel %44). 0'lar birlikte geliyor ve çoğunlukla Business travel/Business class yolcular | 0'ları silme/NaN yapma. Linear modelde `is_zero` flag ekle |
| 4 | **Personal Travel** → ~%10 memnun (class fark etmiyor). **Business travel + Business class** → %73 | Güçlü interaction → Faz 03'te `travel_class` feature'ı dene |
| 5 | **Disloyal** müşteriler her durumda ~%20 | Önemli categorical |
| 6 | **Age** ters U: 40–60 yaş ~%60, <20 ve 60+ ~%15 | Monoton değil → tree modeller yakalar |
| 7 | **Flight Distance** ~1300 km'de eşik: altı ~%30, üstü %60–80 | |
| 8 | Gecikmelerin etkisi zayıf, uçuşların %91'i gecikmesiz. Departure–Arrival correlation 0.84 | Redundant. `arr - dep` denenebilir |
| 9 | Gate location, Departure/Arrival time convenient, Gender ≈ bilgisiz | İleride çıkarmayı dene |
| 10 | Train/test dağılımları birebir aynı, duplicate yok | **Local CV'ye güvenebiliriz** |

## Hipotez kontrolü
- **Hipotez (Faz 00):** "0 = N/A, etkisi olmamalı."
- **Sonuç:** ❌ Yanlış. 0 belirli bir yolcu profilini gösteriyor ve güçlü bir sinyal.
- **Ders:** Domain varsayımlarını (0 = N/A) her zaman veriyle test et.

## Faz 02 planı
Hiçbir feature engineering yapmadan **baseline** kur:
1. Logistic Regression (basit, yorumlanabilir referans)
2. LightGBM default parametrelerle
3. 5-fold Stratified CV ile AUC ölç → ilk submission
