# Faz 00 — Problemi Tanı

## 1. Görev
Bir yolcunun uçuştan **memnun (True)** olup olmadığını tahmin etmek → **binary classification**.
Ama submission'da etiket değil, **probability** gönderiyoruz (`id,satisfaction` → 0.0–1.0).

## 2. Veri
| | Satır | Sütun |
|---|---|---|
| train | 699,635 | 23 (22 feature + `satisfaction`) |
| test | 299,844 | 22 |

Playground Series verileri **synthetic**'tir: gerçek bir veri setinden (Kaggle'daki "Airline Passenger Satisfaction", ~130k satır) bir generative model ile üretilir. Bu yüzden bazı ilişkiler orijinaldekinden biraz "gürültülü" olabilir.

### Sütun grupları
| Grup | Sütunlar | Tip |
|---|---|---|
| Yolcu | Gender, Customer Type, Age | categorical + numeric |
| Yolculuk | Type of Travel, Class, Flight Distance | categorical + numeric |
| Hizmet puanları (14 adet) | Inflight wifi, Online boarding, Seat comfort, Cleanliness… | **ordinal**, 0–5 |
| Gecikmeler | Departure Delay, Arrival Delay (dakika) | numeric, çok çarpık (skewed) |

**Ordinal** = sayı ama sıralı kategori. 4 puan, 2 puanın "iki katı" değil, sadece "daha iyi". Orijinal veride **0 = "değerlendirilmedi / not applicable"** demek → 0'ı "en kötü" gibi yorumlamak yanlış olabilir. EDA'da kontrol edeceğiz.

### İlk gözlemler
- **Target dengesi:** %55.6 False / %44.4 True → neredeyse dengeli. `imbalanced-learn` gerekmez.
- **Missing value:** Sadece `Arrival Delay in Minutes`'ta (train 292, test 130). Çok az.
- `Class`: Business / Eco / **Eco Plus (sadece %4)** → nadir kategori.

## 3. Metric: ROC-AUC
**Soru:** Rastgele bir *memnun* ve bir *memnun olmayan* yolcu seçersem, model memnun olana **daha yüksek** olasılık verme ihtimali nedir?

- 1.0 → mükemmel sıralama
- 0.5 → yazı-tura (rastgele)

### Önemli sonuçlar
1. **AUC sadece sıralamaya bakar, olasılığın kendisine değil.** `[0.1, 0.9]` ile `[0.49, 0.51]` aynı AUC'yi verir. → Threshold (0.5) seçmeye gerek yok, **calibration** önemli değil.
2. Bu yüzden `predict()` değil, **`predict_proba()[:, 1]`** kullanacağız. Etiket (0/1) gönderirsek sıralama bilgisi kaybolur, skor düşer.
3. `sample_submission.csv` herkese aynı değeri (0.4436 = train'deki True oranı) veriyor → sıralama yok → **AUC = 0.5**. Bu bizim "sıfır noktamız".
4. Neden accuracy değil? Accuracy bir threshold'a bağlı ve "ne kadar emin" bilgisini atar. AUC modelin genel ayırt etme gücünü ölçer.

### Alternatifler neden değil
| Metric | Neden seçilmedi |
|---|---|
| Accuracy | Threshold'a bağlı, olasılık bilgisini atar |
| Log loss | Calibration'ı da cezalandırır; AUC daha sade bir "sıralama" ölçüsü |
| F1 | Dengesiz veride iyi, ama burada veri zaten dengeli |

## 4. Faz 01 (EDA) için sorular
- Hangi hizmet puanı memnuniyetle en çok ilişkili? (tahmin: Online boarding, Inflight wifi)
- 0 puanları gerçekten "N/A" gibi mi davranıyor?
- Business travel + Business class yolcular daha mı memnun?
- Departure ve Arrival Delay ne kadar benzer? (muhtemelen çok yüksek correlation)
- Train ve test dağılımları aynı mı?
