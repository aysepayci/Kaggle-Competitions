"""
Faz 03 — Aynı bilgi, farklı temsil: Logistic Regression'a feature engineering.

LightGBM'de yeni feature'lar işe yaramadı (ağaçlar zaten öğreniyor).
Linear model ise eşikleri, 0 puanını ve ters U şeklini kendisi öğrenemez →
bunları one-hot / binning ile ona açıkça veriyoruz.

Çalıştırma (proje klasöründen):
    python src/lr_onehot.py
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from common import CATEGORICAL, ID, SERVICE, load_data, run_cv
from features import add_travel_class

NUMERIC = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def make_model():
    pre = ColumnTransformer([
        # Puanları one-hot: her puan (0 dahil) kendi katsayısını alır → eşik ve "0 = N/A" sorunu çözülür
        ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL + ["travel_class"] + SERVICE),
        # Age ve mesafeyi 10 gruba böl → ters U ve ~1300 km eşiği öğrenilebilir
        ("bins", make_pipeline(SimpleImputer(strategy="median"),
                               KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile", subsample=None)),
         ["Age", "Flight Distance"]),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000))


def main():
    train, test, y = load_data()
    X = add_travel_class(train.drop(columns=[ID]))
    X_test = add_travel_class(test.drop(columns=[ID]))
    run_cv(make_model, X, y, X_test, "LogReg + one-hot")


if __name__ == "__main__":
    main()
