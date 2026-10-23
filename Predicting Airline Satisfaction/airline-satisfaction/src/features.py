"""
Faz 03 — Feature engineering fonksiyonları.
Her fonksiyon bir DataFrame alır, yeni sütunlar eklenmiş bir kopya döndürür.
Fikirlerin kaynağı: notes/faz01_eda.md
"""
import numpy as np
import pandas as pd

from common import SERVICE

COMFORT = ["Seat comfort", "Inflight entertainment", "Cleanliness", "Food and drink"]
ONLINE = ["Inflight wifi service", "Ease of Online booking", "Gate location"]
ZERO_KEY = ["Inflight wifi service", "Ease of Online booking", "Online boarding"]


def add_travel_class(df):
    """EDA: Personal Travel her class'ta ~%10, Business travel + Business %73."""
    df = df.copy()
    df["travel_class"] = (df["Type of Travel"].astype(str) + "_" + df["Class"].astype(str)).astype("category")
    return df


def add_zero_flags(df):
    """EDA: 0 puanı 'kullanmadım' anlamında ve güçlü sinyal (wifi=0 → %89 memnun)."""
    df = df.copy()
    for col in ZERO_KEY:
        df[f"{col}_is_zero"] = (df[col] == 0).astype(int)
    df["n_zero"] = (df[SERVICE] == 0).sum(axis=1)
    return df


def add_cluster_means(df):
    """EDA: correlation heatmap'te iki küme vardı. 0'lar 'N/A' olduğu için ortalamaya katılmıyor."""
    df = df.copy()
    rated = df[SERVICE].replace(0, np.nan)
    df["comfort_mean"] = rated[COMFORT].mean(axis=1)
    df["online_mean"] = rated[ONLINE].mean(axis=1)
    df["service_mean"] = rated.mean(axis=1)
    return df


def add_delay_diff(df):
    """EDA: iki gecikme 0.84 correlated. Fark = yolda telafi edilen (−) / eklenen (+) dakika."""
    df = df.copy()
    df["delay_diff"] = df["Arrival Delay in Minutes"] - df["Departure Delay in Minutes"]
    return df


FEATURE_SETS = {
    "travel_class": add_travel_class,
    "zero_flags": add_zero_flags,
    "cluster_means": add_cluster_means,
    "delay_diff": add_delay_diff,
}


def add_all(df):
    for fn in FEATURE_SETS.values():
        df = fn(df)
    return df
