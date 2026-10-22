"""
Faz 00 — Veriye ilk bakış.

Amaç: Model kurmadan önce verinin boyutunu, tiplerini, target dağılımını
ve eksik değerleri görmek. Bulgular notes/faz00_bilgi.md içinde.

Çalıştırma (proje klasöründen):
    python src/inspect_data.py
"""
from pathlib import Path

import pandas as pd

# Script nereden çalıştırılırsa çalıştırılsın doğru klasörü bulsun
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TARGET = "satisfaction"


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def main() -> None:
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    sample = pd.read_csv(RAW / "sample_submission.csv")

    section("Boyutlar")
    print("train:", train.shape)
    print("test :", test.shape)

    section("Sütun tipleri")
    print(train.dtypes.to_string())

    section("İlk 3 satır")
    print(train.head(3).T.to_string())

    # Target dengeli mi? Dengesizse farklı teknikler gerekir.
    section("Target dağılımı")
    print(train[TARGET].value_counts(normalize=True).round(4).to_string())

    section("Eksik değerler")
    print("train:\n", train.isna().sum()[lambda s: s > 0].to_string())
    print("test:\n", test.isna().sum()[lambda s: s > 0].to_string())

    # Az sayıda unique değer → categorical / ordinal olabilir
    section("Unique değer sayıları")
    print(train.nunique().to_string())

    section("Categorical sütunlar")
    for col in train.select_dtypes("object").columns:
        print(f"{col}: {train[col].value_counts().to_dict()}")

    # Herkese aynı olasılığı veren submission → sıralama yok → AUC = 0.5
    section("Sample submission")
    print(sample.head(3).to_string(index=False))
    print("unique değer sayısı:", sample[TARGET].nunique())


if __name__ == "__main__":
    main()
