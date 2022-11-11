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
```

```python
import seaborn as sns
```

```python
df = pd.read_csv('../train.csv')
```

```python
df
```

```python
targets = df.columns.difference(['text_id', 'full_text'])
```

```python
df[targets].corr()
```

```python
sns.pairplot(df[targets])
```

```python
df['text_len'] = df.full_text.apply(lambda x: len(x))
```

```python
df.text_len.hist(bins=30)
```

```python
sns.pairplot(data=df, x_vars=targets, y_vars='text_len')
```

```python
manual_check = df.sample(n=100, random_state=42)[['text_id', 'full_text'] + features.to_list()].copy()
team_members = ['train', 'alex', 'abdallah', 'mohamed', 'shahinde']
for i, member in enumerate(team_members):
    if member != 'train':
        manual_check[features] = 0
    manual_check[20*i : 20*(i+1)].to_csv('manual_check_' + member + '.csv')
```

```python
import readability
```

```python
results = readability.getmeasures(df.full_text[0], lang='en')
print(results['readability grades']['ARI'])
```

```python
results
```

```python
def text_preprocess(text: str):
    text = text.strip()
    
    # new paragraph starts with \n\n
    # readability also requires each sentence to end with \n
    paragraphs = [p.strip().replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n') for p in text.split('\n\n')]
    
    return "\n\n".join(paragraphs)
```

```python
print(df.full_text[0])
```

```python
print(text_preprocess(df.full_text[0]))
```

```python
df['full_text'] = df['full_text'].apply(text_preprocess)
```

```python
results = readability.getmeasures(df.full_text[0], lang='en')
print(results['readability grades']['ARI'])
```

```python
results
```

```python

```
