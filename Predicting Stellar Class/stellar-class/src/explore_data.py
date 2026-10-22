import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent.parent
train = pd.read_csv(base / 'data' / 'raw' / 'train.csv')
test = pd.read_csv(base / 'data' / 'raw' / 'test.csv')

print('train shape', train.shape)
print('test shape', test.shape)
print('train columns', train.columns.tolist())
print('test columns', test.columns.tolist())
print('dtypes:')
print(train.dtypes.to_string())
print('target counts')
print(train['Star type'].value_counts().sort_index().to_string())
print('train head')
print(train.head().to_string(index=False))
print('test head')
print(test.head().to_string(index=False))
