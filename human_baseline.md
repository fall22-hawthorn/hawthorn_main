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
```

```python
df = pd.read_csv('../train.csv')
```

```python
targets = df.columns.difference(['text_id', 'full_text']).to_list()
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
get_error(results)
```

```python
results.groupby('name').apply(get_error)
```
