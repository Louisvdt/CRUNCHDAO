#%pip install crunch-cli --upgrade --quiet --progress-bar off
#!crunch setup-notebook datacrunch-2 hTvDmQRRpIT0WNXPquwYYV6s


import os 
import joblib
import pandas as pd
import crunch
import numpy as np
import lightgbm as lgb
import sklearn  # == 1.7.2
from sklearn.linear_model import Ridge, LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


#plt.style.use('seaborn-v0_8-whitegrid')


#crunch_tools = crunch.load_notebook()


def get_feature_columns(X: pd.DataFrame):
    return [column for column in X.columns if column.startswith("Feature_")]

# This function just return a list of colomns in the dataframe which names start with "Feature_", 


# Load the data
#X_train, y_train, X_test = crunch_tools.load_data()


#X_train.head()
# No leakage risk: the id changes every moon


#stocks_per_moon = X_train.groupby("moon").size()
#print(f"Min : {stocks_per_moon.min()}")
#print(f"Max : {stocks_per_moon.max()}")
#print(f"Mean: {stocks_per_moon.mean():.0f}")

import matplotlib.pyplot as plt
#stocks_per_moon.plot(figsize=(12, 3))
#plt.title("Number of stocks per moon")
#plt.xlabel("Moon")
#plt.ylabel("Count")
#plt.show()


#feature_cols = get_feature_columns(X_train)

#unique_counts = X_train[feature_cols].nunique()

#print(f"Min unique values:    {unique_counts.min()}")
#print(f"Max unique values:    {unique_counts.max()}")
#print(f"Median unique values: {unique_counts.median()}")
#print()
#print("Distribution:")
#print(unique_counts.value_counts().sort_index().head(20))


import matplotlib.pyplot as plt

#merged_data = X_train.merge(y_train, on=["id", "moon"])

# Regarder 6 features au hasard
#sample_features = ["Feature_1", "Feature_100", "Feature_300", 
#                   "Feature_600", "Feature_900", "Feature_1150"]

#fig, axes = plt.subplots(2, 3, figsize=(15, 8))
#for ax, feat in zip(axes.flat, sample_features):
#    means = merged_data.groupby(feat)["target"].mean()
#    ax.plot(means.index, means.values, marker="o")
#    ax.set_title(feat)
#    ax.set_xlabel("Feature value (septile)")
#    ax.set_ylabel("Mean target")
#    ax.axhline(0, color="grey", linewidth=0.5)
#plt.tight_layout()
#plt.show()


#feature_cols = get_feature_columns(X_train)

# Confirm features are bounded in [0, 1]
#print(f"Global min: {X_train[feature_cols].min().min():.2f}")
#print(f"Global max: {X_train[feature_cols].max().max():.2f}")


#y_train.head()


import matplotlib.pyplot as plt

#target_props = (
#    y_train.groupby("moon")["target"]
#    .value_counts(normalize=True)
#    .unstack(fill_value=0)
#)

#fig, ax = plt.subplots(figsize=(12, 4))
#target_props.plot(ax=ax)
#ax.set_title("Target proportions over time")
#ax.set_xlabel("Moon")
#ax.set_ylabel("Proportion")
#ax.legend(title="Target", labels=["-1", "0", "+1"])
#ax.axhline(0.5, color="grey", linewidth=0.5, linestyle="--")
#plt.tight_layout()
#plt.show()


# Check for missing values
#cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
#print(f"{len(cols_with_missing)} columns with missing values — {'imputation needed' if cols_with_missing else 'no imputation needed'}")


def train(X_train: pd.DataFrame, y_train: pd.DataFrame, model_directory_path: str) -> None:
    feature_columns = get_feature_columns(X_train)

    data = X_train[["moon", "id"] + feature_columns].merge(
        y_train[["moon", "id", "target"]], on=["moon", "id"]
    )

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=2000,
        colsample_bytree=0.3,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(data[feature_columns], data["target"])
    joblib.dump(model, os.path.join(model_directory_path, "model.joblib"))

    


def infer(X_test: pd.DataFrame, model_directory_path: str) -> pd.DataFrame:
    prediction = X_test[["id", "moon"]].copy()

    model = joblib.load(os.path.join(model_directory_path, "model.joblib"))
    feature_columns = get_feature_columns(X_test)

    prediction["prediction"] = model.predict(X_test[feature_columns]).clip(-1, 1)
    return prediction


#crunch_tools.test(force_first_train=True,train_frequency=0)


#prediction = pd.read_parquet("prediction/prediction.parquet") 
#prediction


# Load the targets
#y_test = pd.read_parquet("data/y.reduced.parquet",filters=[("moon", "in", prediction["moon"].unique())])
#y_test 


# Define the scoring function (la corrélation de Pearson utilisée par CrunchDAO)
def score(group: pd.DataFrame):
    return group["prediction"].corr(group["target"], method="pearson")




# Merge the prediction with the target y (with moon and id)
#merged = y_test.merge(prediction,on=["moon", "id"])

# Compute the pearson for each moon
#pearson_values = merged.groupby("moon").apply(score, include_groups=False).fillna(0) 

#print(pearson_values) # voir si certains moons sont très différents des autres
#print(pearson_values.mean())


import seaborn as sns
import matplotlib.pyplot as plt

#fig, ax = plt.subplots(figsize=(10, 4))
#sns.barplot(x=pearson_values.index, y=pearson_values.values, ax=ax)
#ax.axhline(pearson_values.mean(), color="green", linewidth=0.8, linestyle="--", label=f"mean = {pearson_values.mean():.4f}")
#ax.set_title("Pearson correlation per moon")
#ax.set_xlabel("Moon")
#ax.set_ylabel("Pearson")
#ax.legend()
#plt.tight_layout()
#plt.show()


import joblib
import pandas as pd
import matplotlib.pyplot as plt

#model = joblib.load("resources/model.joblib")
#feature_columns = get_feature_columns(X_train)

# Two importance metrics: 'gain' = quality of splits, 'split' = number of times used
#importance_gain = pd.Series(
#    model.booster_.feature_importance(importance_type="gain"),
#    index=feature_columns,
#).sort_values(ascending=False)

#importance_split = pd.Series(
#    model.booster_.feature_importance(importance_type="split"),
#    index=feature_columns,
#).sort_values(ascending=False)

# 1. How many features are actually used?
#n_used = (importance_gain > 0).sum()
#print(f"Features actually used by the model: {n_used} / {len(feature_columns)}")
#print(f"Features completely ignored:        {len(feature_columns) - n_used}")


# 2. Top 20 features by gain
#top20 = importance_gain.head(20)
#fig, ax = plt.subplots(figsize=(10, 6))
#top20[::-1].plot(kind="barh", ax=ax)
#ax.set_title("Top 20 features by gain")
#ax.set_xlabel("Gain")
#plt.tight_layout()
#plt.show()


# 3. Cumulative importance — how many features capture 90% of the signal?
#cumulative = importance_gain.cumsum() / importance_gain.sum()

#n_50 = (cumulative <= 0.50).sum() + 1
#n_80 = (cumulative <= 0.80).sum() + 1
#n_90 = (cumulative <= 0.90).sum() + 1
#n_95 = (cumulative <= 0.95).sum() + 1

#print(f"Features needed for 50% of total gain: {n_50}")
#print(f"Features needed for 80% of total gain: {n_80}")
#print(f"Features needed for 90% of total gain: {n_90}")
#print(f"Features needed for 95% of total gain: {n_95}")

#fig, ax = plt.subplots(figsize=(10, 4))
#ax.plot(range(1, len(cumulative) + 1), cumulative.values)
#ax.axhline(0.9, color="red", linestyle="--", label="90%")
#ax.axhline(0.5, color="green", linestyle="--", label="50%")
#ax.set_xlabel("Number of features (ranked by gain)")
#ax.set_ylabel("Cumulative gain")
#ax.set_title("Cumulative feature importance")
#ax.legend()
#plt.tight_layout()
#plt.show()


# 4. Distribution of importance — are a few features dominant or is it spread out?
#fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#importance_gain[importance_gain > 0].plot(kind="hist", bins=50, ax=axes[0])
#axes[0].set_title("Distribution of feature gains (used features only)")
#axes[0].set_xlabel("Gain")

#importance_gain[importance_gain > 0].apply(lambda x: max(x, 1e-6)).plot(
#    kind="hist", bins=50, ax=axes[1], log=True
#)
#axes[1].set_title("Same, log scale")
#axes[1].set_xlabel("Gain")
#plt.tight_layout()
#plt.show()


# 5. Performance per moon — does the model work on all moons or just some?
#pred_train = X_train[["id", "moon"]].copy()
#pred_train["prediction"] = model.predict(X_train[feature_columns]).clip(-1, 1)
#merged_train = y_train.merge(pred_train, on=["id", "moon"])

#pearson_per_moon = merged_train.groupby("moon").apply(
#    lambda g: g["prediction"].corr(g["target"]), include_groups=False
#).fillna(0)

#fig, ax = plt.subplots(figsize=(12, 4))
#pearson_per_moon.plot(ax=ax)
#ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
#ax.axhline(pearson_per_moon.mean(), color="green", linestyle="--",
#           label=f"mean = {pearson_per_moon.mean():.4f}")
#ax.set_title("In-sample Pearson per moon")
#ax.set_xlabel("Moon")
#ax.set_ylabel("Pearson")
#ax.legend()
#plt.tight_layout()
#plt.show()
