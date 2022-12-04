---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from os import listdir
from os.path import join
import re
from joblib import load
from features import FeatureGenerator
from sklearn.metrics import mean_squared_error
```

```python
df = pd.read_csv('../train.csv')
```

```python
targets = df.columns.difference(['text_id', 'full_text']).to_list()
```

```python
targets
```

```python
GENERATE_CSV_FILES = 0
```

```python
if GENERATE_CSV_FILES:
    manual_check = df.sample(n=100, random_state=42).copy()
    team_members = ['train', 'alex', 'abdallah', 'mohamed', 'shahinde']
    for i, member in enumerate(team_members):
        if member != 'train':
            manual_check[targets] = 0
        manual_check[20*i : 20*(i+1)].to_csv('manual_check_' + member + '.csv')
```

```python
test_size = 80
path = './human_baseline_data'
```

```python
d = []
for file in listdir(path):
    filepath = join(path, file)
    name = re.findall('[a-z]+', file)[0]
    if 'semicol' in file:
        tmp_df = pd.read_csv(filepath, sep=';')
    else:
        tmp_df = pd.read_csv(filepath)
    tmp_df['name'] = name
    d.append(tmp_df)
results = pd.concat(d, axis=0, ignore_index=True)
```

```python
df['text_id'].duplicated().any()
```

```python
results['text_id'].duplicated().any()
```

```python
df
```

```python
pd.set_option('display.max_rows', 100)
results
```

```python
pd.reset_option('display.max_rows')
```

```python
def MCRMSE(a, b):
    return (((a - b)**2).mean(axis=0)**0.5).mean()
```

```python
def get_error(r: pd.DataFrame):
    r = r.reset_index(drop=True)
    truth = df.merge(r.text_id, how='right')
    return MCRMSE(truth[targets], r[targets])
```

```python
errors = pd.DataFrame()
```

```python
errors[['name', 'error']] = [['human_baseline', get_error(results)]]
```

```python
errors
```

```python
errors = pd.concat([errors, results.groupby('name').apply(get_error).reset_index(name='error')], ignore_index=True)
```

```python
errors
```

now let's compare it with model performance

```python
test_df = df.merge(results.text_id, how='right')
```

```python
mean_squared_error(test_df[targets], results[targets], squared=False)
```

```python
train_df = df.merge(results.text_id, how='left', indicator=True)
```

```python
train_df = train_df[train_df['_merge'] == 'left_only']
```

```python
train_df = train_df.drop(columns=['_merge'])
```

```python
model = load('./xgb_linreg_rf.joblib')
```

```python
feature_generator = FeatureGenerator()
```

```python
train_df = feature_generator.generate_features(train_df)
```

```python
test_df = feature_generator.generate_features(test_df)
```

```python
features = train_df.iloc[:,8:].columns
```

```python
model.fit(train_df[features], train_df[targets])
```

```python
model_test_error = mean_squared_error(test_df[targets], model.predict(test_df[features]), squared=False)
```

```python
errors = pd.concat([errors, pd.DataFrame({'name': ['model'], 'error': [model_test_error]})], ignore_index=True)
```

```python
errors
```

```python
errors.sort_values(by='error', ascending=False).plot.barh(x='name', y='error')
```

```python

```
