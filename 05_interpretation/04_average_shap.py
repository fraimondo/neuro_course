# %%
from pathlib import Path
import seaborn as sns
from seaborn import load_dataset
from julearn import PipelineCreator, run_cross_validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.inspection import permutation_importance

# import shap

# shap.initjs()
# %%
df_iris = load_dataset("iris")

df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

N_VARS_TO_ADD = 10
N_RAND_RUNS = 20

for n_var in range(1, N_VARS_TO_ADD + 1):
    random = np.random.randn(len(df_iris)) * 0.1
    new_var = df_iris["petal_length"] + random
    r, p = pearsonr(df_iris["petal_length"], new_var)
    print(f"At iteration {n_var}: r={r} p={p}")
    df_iris[f"correlated_var{n_var}"] = new_var

df_train, df_test = train_test_split(df_iris, test_size=0.2, random_state=42)

model_name = "rf"
coef_name = "Importance"

# %% Set-up models

y = "species"
X = ["sepal_length", "petal_length"] + [
    f"correlated_var{i}" for i in range(1, N_VARS_TO_ADD + 1)
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_types = {"continuous": [".*"]}

# %% Run random models
importances = {}
coefs = {}
scores = {}
models = []
for i_rand in range(N_RAND_RUNS):
    creator = PipelineCreator(problem_type="classification")
    creator.add("rf", n_estimators=100, random_state=i_rand)

    print("Testing original model")
    t_score, t_model = run_cross_validation(
        X=X,
        y=y,
        data=df_train,
        model=creator,
        return_estimator="final",
        X_types=X_types,
    )
    print(f"\tScores: {t_score['test_score'].mean()}")
    r = permutation_importance(
        t_model, df_test[X], df_test[y], n_repeats=30, random_state=0
    )
    print(f"\tImportances: {r.importances_mean}")
    importances[f"run_{i_rand}"] = r.importances_mean
    if model_name == "svm":
        coefs[f"run_{i_rand}"] = t_model.named_steps["svm"].coef_
    elif model_name == "rf":
        coefs[f"run_{i_rand}"] = t_model.named_steps["rf"].feature_importances_
    scores[f"run_{i_rand}"] = t_score["test_score"].mean()
    models.append(t_model)
    print(f"\tcoefs/importances: {coefs[f'run_{i_rand}']}")

df_coefs = pd.DataFrame(coefs, index=X)
df_coefs.index.name = "Variable"
df_coefs = (
    df_coefs.stack().reset_index().rename(columns={0: coef_name, "level_1": "Run"})
)
df_importances = pd.DataFrame(importances, index=X)
df_importances.index.name = "Variable"
df_importances = (
    df_importances.stack()
    .reset_index()
    .rename(columns={0: "Permutation Importance", "level_1": "Run"})
)
# %%

fig, ax = plt.subplots(1, 1)
sns.stripplot(
    data=df_coefs,
    x=coef_name,
    y="Variable",
    ax=ax,
    # hue="Run",
    # palette="viridis",
)
sns.boxplot(
    data=df_coefs,
    x=coef_name,
    y="Variable",
    showfliers=False,
    boxprops=dict(facecolor="none"),
    ax=ax,
)

# %%
fig, ax = plt.subplots(1, 1)
sns.stripplot(
    data=df_importances,
    x="Permutation Importance",
    y="Variable",
    ax=ax,
    # hue="Run",
    # palette="viridis",
)
sns.boxplot(
    data=df_importances,
    x="Permutation Importance",
    y="Variable",
    showfliers=False,
    boxprops=dict(facecolor="none"),
    ax=ax,
)

# %%
import shap
all_shap_values = []
for t_model in models:
    explainer = shap.TreeExplainer(t_model.named_steps["rf"])
    shap_values = explainer(df_test[X])
    shap.summary_plot(shap_values[..., 0], df_test[X])
    all_shap_values.append(shap_values[..., 0].values)
mean_shap_values = np.mean(all_shap_values, axis=0)
shap.summary_plot(mean_shap_values, df_test[X])
# %%
