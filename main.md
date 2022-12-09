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
from scipy.stats import pearsonr

from features import FeatureGenerator
```

```python
# if the plots look bad, run it again
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
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

Targets are correlated to some degree

```python
df[targets].corr()
```

```python
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
feature_generator = FeatureGenerator()
```

It can take some tike to generate features

```python
large_df = feature_generator.generate_features(df)
```

```python
features = large_df.columns[8:].to_list()
```

```python
# too slow to plot
# g = sns.pairplot(large_df, x_vars=targets, y_vars=features, kind='reg')
# g.map(corrfunc)
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
```

```python
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
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
pca = make_pipeline(StandardScaler(), PCA(n_components=0.99))
pca.fit(large_df[features])
```

```python
len(pca['pca'].explained_variance_ratio_)
```

```python
scores['std_pca_linreg'] = -cross_val_score(estimator=make_pipeline(StandardScaler(),
                                                                    PCA(n_components=50),
                                                                    LinearRegression()),
                        X=large_df[features],
                        y=large_df[targets],
                        scoring='neg_root_mean_squared_error',
                        cv=n_splits,).mean()
```

```python
scores['std_pca_linreg']
```

```python
# convergence issues
# gs_elastic = GridSearchCV(estimator=Pipeline([('std', StandardScaler()),
#                                               ('en', ElasticNet(random_state=42, max_iter=2000))]),
#                      param_grid={"en__alpha": [0.1, 1, 10], 'en__l1_ratio':[0, 0.25, 0.5, 0.75, 1]},
#                      scoring='neg_root_mean_squared_error',
#                      cv=n_splits)
# gs_elastic.fit(X=large_df[features], y=large_df[targets])
```

```python
# gs_elastic.best_estimator_
```

```python
# scores['gs_elastic'] = -gs_elastic.best_score_
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

these two cell are basically same

```python
# vote = VotingRegressor(estimators=[
#     ('xgb',  XGBRegressor(**gs_xgb.best_params_, n_estimators=100)),
#     ('linreg', LinearRegression()),
#     ('rf', RandomForestRegressor(**gs_rf.best_params_))
# ])
```

```python
vote = VotingRegressor(estimators=[
    ('xgb',  XGBRegressor(learning_rate=0.1, max_depth=2, n_estimators=100)),
    ('linreg', LinearRegression()),
    ('rf', RandomForestRegressor(max_depth=12))
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
scores['xgb_linreg_rf'] = score.mean()
```

```python
scores_df = pd.DataFrame(sorted(scores.items(), key=lambda x: x[1]), columns=['model', 'score'])
```

```python
scores_df
```

```python
ax = sns.barplot(data=scores_df, x='score', y='model')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f')
ax.set_xbound(upper=0.6)
ax.set_xlabel("CV score")
```

```python
final_model = MultiOutputRegressor(vote)
final_model.fit(X=large_df[features], y=large_df[targets])
```

```python
from joblib import dump, load
dump(final_model, 'xgb_linreg_rf.joblib')
```

Feature importance

Hmm there is some problem with accessing models inside final voting regressor so we need to train each model separately.


Lots of features are highly correlated, let's drop some of them.
But how can we decide which feature to keep and which to drop?
And if we drop a feature, does it mean we can forget about it in feature importance analysis?


### ATTENTION: this linear regression model is not the same as before!

```python
def trimm_correlated(df_in, threshold):
    df_corr = df_in.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    return df_in.columns.difference(un_corr_idx)
```

```python
trimmed_df = large_df.copy()
```

```python
trimmed_df = trimmed_df.drop(columns=trimm_correlated(trimmed_df[features], 0.9))
```

```python
trimmed_df_features = trimmed_df.columns[8:].to_list()
```

```python
len(features), len(trimmed_df_features)
```

```python
print(f"we have dropped {len(features) - len(trimmed_df_features)} highly correlated features")
```

```python
std_lin_reg = make_pipeline(StandardScaler(), LinearRegression())
std_lin_reg.fit(X=trimmed_df[trimmed_df_features], y=trimmed_df[targets])
```

```python
trimmed_corr_std_linreg = -cross_val_score(estimator=make_pipeline(StandardScaler(), LinearRegression()),
                        X=trimmed_df[trimmed_df_features],
                        y=trimmed_df[targets],
                        scoring='neg_root_mean_squared_error',
                        cv=n_splits,).mean()
```

```python
scores['trimmed_corr_std_linreg'] = trimmed_corr_std_linreg
```

```python
scores_df = pd.DataFrame(sorted(scores.items(), key=lambda x: x[1]), columns=['model', 'score'])
```

```python
ax = sns.barplot(data=scores_df, x='score', y='model')
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f')
ax.set_xbound(upper=0.6)
ax.set_xlabel("CV score")
```

```python
lin_reg_feature_imp = pd.DataFrame(data=std_lin_reg['linearregression'].coef_[0], index=trimmed_df_features, columns=['feature importance'])
```

```python
lin_reg_feature_imp = lin_reg_feature_imp.sort_values(by='feature importance', key=abs)
```

```python
lin_reg_feature_imp
```

```python
fig, ax = plt.subplots(figsize=(5, 5))
lin_reg_feature_imp[50:].plot.barh(y=0, ax=ax)
ax = plt.gca()
ax.set_title("Top 20 features (linear regression)")
```

```python
fig, ax = plt.subplots(figsize=(10, 20))
lin_reg_feature_imp.plot.barh(y=0, ax=ax)
```

```python

```

Feature importance for random forest
### There are many other ways to study feature importance for random forest
https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e

```python
rf_model = RandomForestRegressor(**gs_rf.best_params_)
rf_model.fit(large_df[features], large_df[targets])
```

```python
rf_feature_imp = pd.DataFrame(data=rf_model.feature_importances_, index=features, columns=['feature importance']).sort_values(by='feature importance', key=abs)
```

```python
fig, ax = plt.subplots(figsize=(5, 5))
rf_feature_imp[60:].plot.barh(y=0, ax=ax)
ax = plt.gca()
ax.set_title("Top 10 feature (Random Forest)")
```

```python
fig, ax = plt.subplots(figsize=(10, 20))
rf_feature_imp.plot.barh(y=0, ax=ax)
```

feature importance for xgbclassifier

```python
from xgboost import plot_importance
```

```python
fig, ax = plt.subplots(figsize=(5, 5))
plot_importance(xgb_reg, ax=ax, importance_type='weight', max_num_features=20)
ax.set_title("XGBClassifier feature importance (top 20, by weight)")
```

```python
fig, ax = plt.subplots(figsize=(10, 20))
plot_importance(xgb_reg, ax=ax, importance_type='weight')
```

```python
fig, ax = plt.subplots(figsize=(5, 5))
plot_importance(xgb_reg, ax=ax, importance_type='gain', max_num_features=20)
ax.set_title("XGBClassifier feature importance (top 20, by gain)")
```

```python
fig, ax = plt.subplots(figsize=(10, 20))
plot_importance(xgb_reg, ax=ax, importance_type='gain')
```

```python
fig, ax = plt.subplots(figsize=(5, 5))
plot_importance(xgb_reg, ax=ax, importance_type='cover', max_num_features=20)
ax.set_title("XGBClassifier feature importance (top 20, by cover)")
```

```python
fig, ax = plt.subplots(figsize=(10, 20))
plot_importance(xgb_reg, ax=ax, importance_type='cover')
```

```python
def clipped_model_predict(X):
    return final_model.predict(X).clip(1, 5)
```

```python

```

```python
import shap
```

First, let's try SHAP analysis for the final model. We select a subset of the data because it takes quite a lot of time to genereta SHAP values.

```python
shap_dataset = large_df.sample(n=100, random_state=42).copy()
```

```python
explainer = shap.KernelExplainer(clipped_model_predict, shap_dataset[final_model.feature_names_in_])
```

```python
shap_values = explainer.shap_values(shap_dataset[final_model.feature_names_in_])
```

`shap_values` stores 6 shap values for each target

```python
len(shap_values)
```

```python
for i, target_name in enumerate(targets):
    shap.summary_plot(shap_values[i],
                      shap_dataset[final_model.feature_names_in_],
                      show=False
                      )
    ax = plt.gca()
    ax.set_title(target_name)
    plt.show()
```

As we can see, the most important feature now is ARI and the top 3 features are the same for each target. This may be because our voting model has linear regressor with highly correlated features and we may expect some issues in importance analysis from it.

```python
large_df[['ARI',
          'characters_per_word',
          'words_per_sentence',
          'Kincaid',
          'FleschReadingEase',
          'LIX',
          'GunningFogIndex',
          'syll_per_word',
          'long_words_ratio',
          'complex_words_ratio',
          'words',
          'characters']].corr()
```

Actually, we can analyze correlations for the whole set of features

```python
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 60))
corr = spearmanr(large_df[features]).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=features, ax=ax1, orientation='right'
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig.tight_layout()
plt.show()
```

As you can see, there are clusters of highly correlated features! Such as (syllables, characters, words, text_len), (LT_REDUNDANCY, LT_REDUNDANCY_ratio), (Kincaid, ARI, GunningFogIndex, words_per_sentence), (uwc1e5, uwc1e5_ratio), (type_token_ratio, wordtypes_ratio) and so on

```python

```

Now let's try SHAP analysis on our xgb regressor

```python
explainer_xgb_only = shap.TreeExplainer(xgb_reg)
```

```python
shap_values_xgb_only = explainer_xgb_only.shap_values(large_df[xgb_reg.feature_names_in_])
```

```python
for i, target_name in enumerate(targets):
    shap.summary_plot(shap_values_xgb_only[i],
                      large_df[xgb_reg.feature_names_in_],
                      show=False
                      )
    ax = plt.gca()
    ax.set_title(target_name)
    plt.show()
```

## They look much more reasonable
For example, for grammar, the top feature is LT_GRAMMAR_ratio.

For vocabulary, the second top feature is wordtypes (number of different word types).

Besides, it looks pretty close to the top features by weight and gain


```python

```
