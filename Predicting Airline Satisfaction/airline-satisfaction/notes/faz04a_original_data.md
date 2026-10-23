# Faz 04a — Orijinal Veriyi Eklemek

Kod: `src/original_data.py` · Veri: `data/raw/original/` (Kaggle "Airline Passenger Satisfaction", train + test = 129,880 satır)

## Fikir
Playground verisi bu orijinal veriden bir generative model ile üretildi. Gerçek satırları ek training verisi olarak eklemek Playground yarışmalarında sık işe yarayan bir teknik.

## Nasıl ekledik (önemli)
Orijinal satırlar **sadece training fold'larına** eklendi, validation'a asla. Validation hep yarışma verisi kalmalı; yoksa CV skoru test'i yansıtmaz ve baseline ile karşılaştırılamaz.
Hazırlık: orijinaldeki fazladan `Inflight service` sütunu atıldı, target `"satisfied"` → 1.

## Sonuçlar (LightGBM default)
| Deney | CV AUC | Δ | Fold kazanma |
|---|---|---|---|
| baseline | 0.95779 | — | — |
| + original | 0.95737 | −0.00042 | **0/5** |
| + original + `is_original` flag | 0.95746 | −0.00033 | — |

Fark Faz 03'teki gürültüden (~±0.0001) 4 kat büyük ve 5 fold'un hepsinde aynı yönde → **gerçek bir kötüleşme**.

## Neden kötüleştirdi? — Teşhis
| Deney | AUC |
|---|---|
| Orijinal, kendi içinde CV | **0.9945** |
| Orijinalle eğit → yarışma verisinde ölç | 0.9531 |
| Yarışmayla eğit → orijinalde ölç | 0.9923 |

- Orijinal veri çok **temiz** (0.9945). Synthetic veri ise generative model yüzünden **gürültülü** (0.958 civarı tavan).
- Örnek: wifi=0 → orijinalde %100 memnun, yarışmada %89. Desenler aynı ama synthetic'te daha bulanık.
- Orijinalle eğitilen model bu "keskin" kuralları öğreniyor; gürültülü test verisinde aşırı emin ve yanlış oluyor.
- Yarışma verisiyle eğitilen model orijinali zaten 0.992 ile tahmin ediyor → orijinal **yeni bilgi** getirmiyor, sadece farklı bir dağılım getiriyor.
- `is_original` flag'i kısmen yardım etti ama yetmedi.

## Ders
- "Daha fazla veri = daha iyi" her zaman doğru değil. Ek veri **test ile aynı dağılımdan** gelmeli.
- Popüler bir teknik her yarışmada çalışmaz → her zaman CV ile ölç.

## Karar
Orijinal veriyi kullanmıyoruz. Faz 04b: XGBoost ve CatBoost.
