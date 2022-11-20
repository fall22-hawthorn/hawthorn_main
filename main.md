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

```python
vocab_feature_corr = large_df[targets + features].corr().iloc[len(targets):, :len(targets)]['vocabulary'].sort_values(key=abs, ascending=False)
```

```python
vocab_feature_corr
```

```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
```

```python
from sklearn.linear_model import LinearRegression
```

```python
def MCRMSE(a, b):
    return (((a - b)**2).mean(axis=0)**0.5).mean()

from sklearn.metrics import make_scorer
mcrmse_score = make_scorer(MCRMSE, greater_is_better=False) #same as scoring='neg_root_mean_squared_error'
```

```python
scores = {}
```

```python
n_splits = 5
```

```python
score = cross_val_score(estimator=LinearRegression(),
                        X=large_df[features],
                        y=large_df[targets],
                        scoring='neg_root_mean_squared_error',
                        cv=n_splits,)
```

```python
score.mean()
```

```python
scores['linreg'] = score.mean()
```

```python
score = cross_val_score(estimator=make_pipeline(Normalizer(), LinearRegression()),
                        X=large_df[features],
                        y=large_df[targets],
                        scoring='neg_root_mean_squared_error',
                        cv=n_splits,)
```

```python
score.mean()
```

```python
scores['norm_linreg'] = score.mean()
```

```python
score = cross_val_score(estimator=make_pipeline(StandardScaler(), LinearRegression()),
                        X=large_df[features],
                        y=large_df[targets],
                        scoring='neg_root_mean_squared_error',
                        cv=n_splits,)
```

```python
score.mean()
```

```python
scores['std_linreg'] = score.mean()
```

```python
from sklearn.linear_model import ElasticNet
```

```python
gs_elastic = GridSearchCV(estimator=Pipeline([('std', StandardScaler()),
                                              ('en', ElasticNet(random_state=42, max_iter=2000))]),
                     param_grid={"en__alpha": [0.1, 1, 10], 'en__l1_ratio':[0, 0.25, 0.5, 0.75, 1]},
                     scoring='neg_root_mean_squared_error',
                     cv=n_splits)
gs_elastic.fit(X=large_df[features], y=large_df[targets])
```

```python
gs_elastic.best_estimator_
```

```python
gs_elastic.best_score_
```

```python
scores['gs_elastic'] = gs_elastic.best_score_
```

```python
from sklearn.ensemble import RandomForestRegressor
```

```python
score = cross_val_score(estimator=RandomForestRegressor(max_depth=5),
                        X=large_df[features],
                        y=large_df[targets],
                        scoring='neg_root_mean_squared_error',
                        cv=n_splits,)
```

```python
score.mean()
```

```python
gs_rf = GridSearchCV(estimator=RandomForestRegressor(),
                     param_grid={"max_depth": [10, 12, 14]},
                     scoring='neg_root_mean_squared_error',
                     cv=n_splits)
gs_rf.fit(X=large_df[features], y=large_df[targets])
```

```python
gs_rf.best_estimator_
```

```python
gs_rf.best_score_
```

```python
scores['gs_rf'] = gs_rf.best_score_
```

```python
from xgboost import XGBRegressor
```

```python
score = cross_val_score(estimator=XGBRegressor(max_depth=5),
                        X=large_df[features],
                        y=large_df[targets],
                        scoring='neg_root_mean_squared_error',
                        cv=n_splits,)
```

```python
score.mean()
```

```python
X_train, X_val, y_train, y_val = train_test_split(large_df[features].copy(),
                                                  large_df[targets].copy(),
                                                  test_size=0.2,
                                                  random_state=42,
                                                  )
```

```python
gs_xgb = GridSearchCV(estimator=XGBRegressor(early_stopping_rounds=10, n_estimators=100, random_seed=42),
                     param_grid={"max_depth": [2, 3, 4], 'learning_rate': [0.05, 0.1, 0.5]},
                     scoring='neg_root_mean_squared_error',
                     cv=n_splits)
gs_xgb.fit(X=X_train[features], y=y_train[targets], eval_set=[(X_val, y_val)])
```

```python
gs_xgb.best_estimator_
```

```python
gs_xgb.best_score_
```

```python
scores['gs_xgb'] = gs_xgb.best_score_
```

```python
gs_xgb.best_params_
```

now we need to figure out what's the best n_estimators for XGBRegressor

```python
gs_xgb.cv_results_
```

```python
def f(max_depth, learning_rate):
    print(learning_rate, max_depth)
f(**gs_xgb.best_params_)
```

```python
xgb_reg = XGBRegressor(**gs_xgb.best_params_, early_stopping_rounds=10, n_estimators=100)
xgb_reg.fit(X=X_train[features], y=y_train[targets], eval_set=[(X_val, y_val)])
```

ok let's keep n_estimators=100

```python
from sklearn.ensemble import VotingRegressor
```

```python
vote = VotingRegressor(estimators=[
    ('xgb',  XGBRegressor(**gs_xgb.best_params_, n_estimators=100)),
    ('linreg', LinearRegression()),
    ('rf', RandomForestRegressor(**gs_rf.best_params_))
])
```

unfortunately, VotingRegressor does not do multi-target regression

```python
score = cross_val_score(estimator=vote,
                        X=large_df[features],
                        y=large_df[targets],
                        scoring='neg_root_mean_squared_error',
                        cv=n_splits,)
```

```python
from sklearn.multioutput import MultiOutputRegressor
```

```python
score = cross_val_score(estimator=MultiOutputRegressor(vote),
                        X=large_df[features],
                        y=large_df[targets],
                        scoring='neg_root_mean_squared_error',
                        cv=n_splits,)
```

```python
score.mean()
```

```python
scores['vote_xgb_linreg_rf'] = score.mean()
```

```python
sorted(scores.items())
```

```python

```
