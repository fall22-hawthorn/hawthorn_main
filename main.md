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

feature idea: number of correctly spelled noncommon english words

```python
common_words_1k_filename = '1-1000.txt'
with open(common_words_1k_filename) as f:
    common_words_1k = set(x.strip() for x in f.readlines())

common_words_filename_10k = 'google-10000-english-no-swears.txt'
with open(common_words_filename_10k) as f:
    common_words_10k = set(x.strip() for x in f.readlines())

# make sure common_words are a subset as well
common_words_10k.update(common_words_1k)
```

```python
from nltk.tokenize import word_tokenize
import enchant

enchant_dict = enchant.Dict("en_US")

def get_noncommon_words_count(text, common_words_dict):
    # first, tokenize the text
    # second, iterate over tokens and see whether
    # a. it is a word
    # b. not in common_words
    # c. correctly spelled
    # d. does not have underscore '_'

    def is_noncommon_word(w):
        return  len(w) > 2 and\
                w not in common_words_dict and\
                '_' not in w and\
                enchant_dict.check(w)

    return sum(is_noncommon_word(x) for x in word_tokenize(text.lower()))

def get_noncommon_words_count_1k(text):
    return get_noncommon_words_count(text, common_words_1k)

def get_noncommon_words_count_10k(text):
    return get_noncommon_words_count(text, common_words_10k)
```

```python
large_df['noncommon_words_1k'] = large_df['full_text'].apply(get_noncommon_words_count_1k)
large_df['noncommon_words_10k'] = large_df['full_text'].apply(get_noncommon_words_count_10k)
```

```python
large_df['noncommon_words_1k_ratio'] = large_df['noncommon_words_1k'] / large_df['words']
large_df['noncommon_words_10k_ratio'] = large_df['noncommon_words_10k'] / large_df['words']
```

```python
profanity_filename = 'profanity.txt'
with open(profanity_filename) as f:
    profanity_set = set(x.strip() for x in f.readlines())

def get_profanity_count(text):
    return sum(x in profanity_set for x in word_tokenize(text.lower()))
```

```python
large_df['profanity_count'] = large_df['full_text'].apply(get_profanity_count)
```

```python
large_df['profanity_count'].hist()
```

```python
words_freq_filename = "count_1w.txt"
```

```python
words_freq = pd.read_csv(words_freq_filename,
                         names=['word', 'freq'],
                         sep='\t',
                         header=None,
                         dtype={'word': str, 'freq': int},
                         keep_default_na=False,
                         na_values=[''])
```

```python
words_freq.freq.plot.line(logy=True)
```

```python
words_freq.word.apply(lambda x: x[0].isupper()).any()
```

```python
words_freq = words_freq.set_index('word')
```

```python
def get_uncommon_words_counts(text):
    #print(text)
    word_freq_thresholds = np.array([1e8, 1e7, 1e6, 1e5], dtype=int)
    counts = np.zeros(len(word_freq_thresholds), dtype=int)
    for w in word_tokenize(text):
        if len(w) <  3 or '_' in w or not enchant_dict.check(w) or w not in words_freq.index:
            continue
        w_freq = words_freq.loc[w].values[0]
        counts += (word_freq_thresholds > w_freq)
    return counts
```

```python
get_uncommon_words_counts(large_df['full_text'][0])
```

```python
large_df[['uwc1e8', 'uwc1e7', 'uwc1e6', 'uwc1e5']] = large_df[['full_text']].apply(lambda x: get_uncommon_words_counts(x[0]), axis=1, result_type='expand')
```

```python
large_df[['uwc1e8_ratio', 'uwc1e7_ratio', 'uwc1e6_ratio', 'uwc1e5_ratio']] = large_df[['uwc1e8', 'uwc1e7', 'uwc1e6', 'uwc1e5']].div(large_df['words'], axis=0)
```

```python
import language_tool_python
```

```python
tool = language_tool_python.LanguageTool('en-US')
```

```python
# I have commented out categories giving 0 columns in train dataset
# SEMANTICS is mostly 0, ill drop it as well
LT_categories = ['CASING',
                 #'COLLOQUIALISMS',
                 'COMPOUNDING',
                 'CONFUSED_WORDS',
                 #'FALSE_FRIENDS',
                 #'GENDER_NEUTRALITY',
                 'GRAMMAR',
                 'MISC',
                 'PUNCTUATION',
                 'REDUNDANCY',
                 #'REGIONALISMS',
                 #'REPETITIONS',
                 #'REPETITIONS_STYLE',
                 #'SEMANTICS',
                 'STYLE',
                 'TYPOGRAPHY',
                 'TYPOS',
                 'TOTAL',
                ]
def get_LT_features(text):
    '''
    Generates LanguateTool features: each category count and a total number
    '''
    matches = tool.check(text)
    cat_counts = [sum(m.category == cat for m in matches) for cat in LT_categories[:-1]]
    return cat_counts + [len(matches)]
```

```python
get_LT_features(large_df['full_text'][0])
```

```python
large_df[['LT_' + x for x in LT_categories]] =  large_df[['full_text']].apply(lambda x: get_LT_features(x[0]), axis=1, result_type='expand')
```

```python
large_df[['LT_' + x + '_ratio' for x in LT_categories]] = large_df[['LT_' + x for x in LT_categories]].div(large_df['words'], axis=0)
```

feature generation complete

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
# abs heatmap
fig, ax = plt.subplots(figsize=(10, 20))
sns.heatmap(large_df[targets + features].corr().iloc[len(targets):, :len(targets)].abs(), annot=True, ax=ax)
```

```python
large_df.describe().loc['std'].sort_values()
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
score = -cross_val_score(estimator=LinearRegression(),
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
score = -cross_val_score(estimator=make_pipeline(Normalizer(), LinearRegression()),
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
score = -cross_val_score(estimator=make_pipeline(StandardScaler(), LinearRegression()),
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
-gs_elastic.best_score_
```

```python
scores['gs_elastic'] = -gs_elastic.best_score_
```

```python
from sklearn.ensemble import RandomForestRegressor
```

```python
score = -cross_val_score(estimator=RandomForestRegressor(max_depth=5),
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
-gs_rf.best_score_
```

```python
scores['gs_rf'] = -gs_rf.best_score_
```

```python
from xgboost import XGBRegressor
```

```python
score = -cross_val_score(estimator=XGBRegressor(max_depth=5),
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
-gs_xgb.best_score_
```

```python
scores['gs_xgb'] = -gs_xgb.best_score_
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
score = -cross_val_score(estimator=vote,
                        X=large_df[features],
                        y=large_df[targets],
                        scoring='neg_root_mean_squared_error',
                        cv=n_splits,)
```

```python
from sklearn.multioutput import MultiOutputRegressor
```

```python
score = -cross_val_score(estimator=MultiOutputRegressor(vote),
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
scores_df = pd.DataFrame(sorted(scores.items(), key=lambda x: x[1]), columns=['model', 'score'])
```

```python
scores_df
```

```python

```
