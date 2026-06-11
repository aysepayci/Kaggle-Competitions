# Faz 00 — Problemi ve Evreni Tanı

## 1. SDSS (Sloan Digital Sky Survey) Nedir?

Evrenin yapısını ve genişleme tarihçesini anlamayı hedefleyen SDSS, gökyüzünü sistematik olarak haritalamayı amaçlayan devasa bir gözlem programıdır. Geleneksel astronominin "tekil hedeflere odaklanma" yaklaşımını değiştirerek 100 milyondan fazla gök cisminin konumunu ve yüz binlerce galaksinin uzaklığını ölçmeyi hedeflemiştir.

Alfred P. Sloan Vakfı'nın desteğiyle 2000 yılından bu yana süren proje, APO (Kuzey yarımküre) ve LCO (Güney yarımküre) gözlemevlerinden eşzamanlı veri toplamaktadır.

### Veri Toplama Yöntemi

Teleskop gökyüzünü takip etmek yerine **Zaman Gecikmeli Entegrasyon (TDI)** adı verilen sürüklenme taraması tekniğini kullanır. Spektroskopik gözlemler için BOSS (optik) ve APOGEE (kızılötesi) spektrografları devrededir. SDSS-V ile birlikte manuel fiber yerleştirme kaldırılarak 500 adet mikromotorlu koldan oluşan **Robotik Odak Düzlemi Sistemi (FPS)** getirilmiştir.

### Üç Ana Bilimsel Program

| Program | Odak |
|---------|------|
| Samanyolu Haritalayıcısı (MWM) | Galaksimizin oluşumu ve kimyasal yapısı |
| Kara Delik Haritalayıcısı (BHM) | Süper kütleli kara deliklerin büyüme mekanizmaları |
| Yerel Hacim Haritalayıcısı (LVM) | Gaz bulutları ve yıldız oluşum bölgeleri |

### Kozmolojik Katkıları

SDSS verileri; evrenin genişleme hızının ölçülmesi, Baryon Akustik Salınımlarının (BAO) incelenmesi ve Karanlık Enerji modellerinin doğrulanmasında kritik rol oynamıştır.

---

## 2. STAR, GALAXY ve QSO — Fiziksel Farklar

Bir gökbilimci bu nesneleri ayırt etmek için önce görüntüdeki morfolojiye (şekle), ardından spektruma bakar.

### STAR (Yıldız)

Teleskop görüntüsünde tek bir ışık noktası olarak görünür. Samanyolu içindeki gök cisimleridir; spektrumlarında atmosferlerinden kaynaklanan soğurma çizgileri bulunur.

### GALAXY (Galaksi)

Uzaya yayılmış, genişlemiş kaynaklar olarak görünür. Sarmal veya eliptik şekiller gösterebilir. Milyarlarca yıldızın toplam ışığını içerir.

### QSO (Quasar)

Aktif galaksi çekirdeğidir. Merkezinde süper kütleli kara delik barındırır; son derece parlak olmakla birlikte çok uzak oldukları için teleskopta yıldız gibi tek bir nokta olarak görünürler.

### Neden QSO Tespiti En Zordur?

Kamerada sıradan yıldızlarla aynı görünümü üretirler (görsel ikizlik). Ev sahibi galaksinin ışığı kuasar sinyaline karışabilir. Otomatik algoritmalar bazı galaksi spektral çizgilerini yanlış yorumlayarak hatalı tespit yapabilir.

---

## 3. Fotometrik Filtreler: u, g, r, i, z

SDSS kameraları evreni farklı dalga boylarında gözlemek için beş filtre kullanır.

| Filtre | Dalga Boyu | Bölge |
|--------|-----------|-------|
| u | ~355 nm | Ultraviyole |
| g | ~468 nm | Mavi-yeşil |
| r | ~616 nm | Kırmızı |
| i | ~748 nm | Yakın kızılötesi |
| z | ~893 nm | Kızılötesi |

### Renk İndeksi Nedir?

Renk indeksi, iki farklı filtrenin parlaklık farkıdır (örn. `u–g`, `g–r`, `r–i`). Bu farklar cismin sıcaklığı, kimyasal yapısı ve yaşı hakkında hızlı bilgi sağlar.

- **Düşük / negatif değer** → Mavi ve sıcak cisim
- **Yüksek değer** → Kırmızı ve soğuk cisim

---

## 4. Kızıla Kayma (Redshift) Nedir?

Bir gök cismi bizden uzaklaştıkça yaydığı ışığın dalga boyu uzar ve spektrumun kırmızı tarafına kayar. Bu olay cismin uzaklığını, uzaklaşma hızını ve evrenin genişlemesini anlamamızı sağlar.

### Makine Öğrenmesinde Neden Bu Kadar Güçlü?

Redshift değerleri doğal sınıf ayrımları oluşturur:

| Nesne | Tipik Redshift |
|-------|----------------|
| STAR | Çok düşük |
| GALAXY | Orta |
| QSO | Çok yüksek |

Yalnızca renk ve parlaklık açısından benzer görünen nesneler, redshift bilgisi eklendiğinde kolayca ayrıştırılabilir.

---

## 5. Balanced Accuracy — Neden Bu Metrik?

### Formül

Balanced Accuracy, her sınıfın recall değerinin aritmetik ortalamasıdır:

$$
\text{Balanced Accuracy} = \frac{1}{N} \sum_{i=1}^{N} Recall_i
$$

$$
Recall = \frac{TP}{TP + FN}
$$

### Hesaplama Örneği

Veri setinde 1000 GALAXY, 100 STAR, 10 QSO var. Model her şeyi GALAXY tahmin etsin.

**Normal Accuracy:**

$$
Accuracy = \frac{1000}{1110} \approx 90.09\%
$$

**Recall değerleri:**

| Sınıf | Doğru Tahmin | Toplam | Recall |
|-------|-------------|--------|--------|
| GALAXY | 1000 | 1000 | 1.00 |
| STAR | 0 | 100 | 0.00 |
| QSO | 0 | 10 | 0.00 |

**Balanced Accuracy:**

$$
\text{Balanced Accuracy} = \frac{1.00 + 0.00 + 0.00}{3} \approx 33.33\%
$$

%90 accuracy gösterirken aslında %33 değerinde bir model. Bu farkı kaçırmak sıralamayı doğrudan etkiler.

### Neden Accuracy Yeterli Değil?

Astronomik veri setleri doğası gereği dengesizdir: GALAXY sayısı çok fazla, QSO sayısı oldukça düşüktür. Sıradan accuracy, model sadece en büyük sınıfı tahmin etse bile yüksek çıkabilir. Balanced Accuracy her sınıfa eşit ağırlık vererek azınlık sınıflarını (özellikle QSO) görmezden gelen modelleri cezalandırır.

---



