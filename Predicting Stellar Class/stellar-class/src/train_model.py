import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_PATH = os.path.join(BASE, 'data', 'raw', 'train.csv')
TEST_PATH = os.path.join(BASE, 'data', 'raw', 'test.csv')
SUB_PATH = os.path.join(BASE, 'submissions', 'submission.csv')

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Basic preprocessing
feature_cols = [c for c in train.columns if c != 'Star type']
X = train[feature_cols].copy()
y = train['Star type'].copy()
X_test = test[feature_cols].copy()

numeric_cols = X.select_dtypes(include='number').columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, random_state=42))
])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
val_pred = model.predict(X_val)
print('Validation accuracy:', round(accuracy_score(y_val, val_pred), 4))
print(classification_report(y_val, val_pred))

# Final training on full data
model.fit(X, y)

test_pred = model.predict(X_test)

submission = pd.DataFrame({
    'id': test['id'],
    'Star type': test_pred,
})

submission.to_csv(SUB_PATH, index=False)
print('Saved submission to', SUB_PATH)
print(submission.head())
