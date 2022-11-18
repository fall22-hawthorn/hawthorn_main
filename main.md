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
import enchant
from enchant.checker import SpellChecker
```

```python
df = pd.read_csv('../train.csv')
```

```python
df
```

```python
targets = df.columns.difference(['text_id', 'full_text']).to_list()
```

```python
df[targets].corr()
```

```python
from scipy.stats import pearsonr
def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
```

```python
g = sns.pairplot(df[targets], kind='reg')
g.map_offdiag(corrfunc)
```

```python
sns.heatmap(df[targets].corr(), annot=True)
```

```python
df['text_len'] = df.full_text.apply(lambda x: len(x))
```

```python
df.text_len.hist(bins=30)
```

```python
g = sns.pairplot(data=df, x_vars=targets, y_vars='text_len', kind='reg')
g.map(corrfunc)
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

We would like to add these results as features in out dataframe. We cannot use `merge=True` because `word usage` and `sentence beginnings` have same keys which overlap each other when `merge=True`

```python
from collections import OrderedDict
```

```python
def flatten_readability(r: OrderedDict):
    out = {}
    for k, group in r.items():
        prefix = {'readability grades': '',
                  'sentence info': '',
                  'word usage': 'wu_',
                  'sentence beginnings': 'sb_'}[k]
        for var_name, value in group.items():
            out[prefix + var_name] = value
    return out
```

```python
flatten_readability(results)
```

```python
applied_df = df[['full_text']].apply(lambda row:
                                    flatten_readability(
                                        readability.getmeasures(row[0], lang='en')),
                                axis='columns',
                                result_type='expand')
```

```python
flatten_readability(readability.getmeasures(df.full_text[0], lang='en'))
```

```python
applied_df
```

```python
large_df = pd.concat([df, applied_df], axis='columns')
```

add number of misspelled words as a feature

```python
def misspelled_count(text):
    chkr = SpellChecker("en_US")
    chkr.set_text(text)
    return len(list(chkr))
```

```python
large_df['misspelled_count'] = large_df['full_text'].apply(misspelled_count)
```

```python
large_df['misspelled_ratio'] = large_df['misspelled_count'] / large_df['words']
```

```python
for col_name in ['wordtypes',
                 'long_words',
                 'complex_words',
                 'complex_words_dc',
                 'wu_tobeverb',
                 'wu_auxverb',
                 'wu_conjunction',
                 'wu_pronoun',
                 'wu_preposition',
                 'wu_nominalization',]:
    large_df[col_name + '_ratio'] = large_df[col_name] / large_df['words']

for col_name in ['sb_pronoun',
                 'sb_interrogative',
                 'sb_article',
                 'sb_subordination',
                 'sb_conjunction',
                 'sb_preposition',]:
    large_df[col_name + '_ratio'] = large_df[col_name] / large_df['sentences']
```

```python
features = large_df.columns[8:].to_list()
```

```python
g = sns.pairplot(large_df, x_vars=targets, y_vars=features, kind='reg')
g.map(corrfunc)
```

```python
fig, ax = plt.subplots(figsize=(10, 20))
sns.heatmap(large_df[targets + features].corr().iloc[len(targets):, :len(targets)], annot=True, ax=ax)
```
