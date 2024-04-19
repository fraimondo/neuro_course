# %%
from pathlib import Path
import seaborn as sns
from seaborn import load_dataset
from julearn import PipelineCreator, run_cross_validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# import shap

# shap.initjs()
# %%
df_iris = load_dataset("iris")

df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

N_VARS_TO_ADD = 10

for n_var in range(1, N_VARS_TO_ADD + 1):
    random = np.random.randn(len(df_iris)) * 0.1
    new_var = df_iris["petal_length"] + random
    r, p = pearsonr(df_iris["petal_length"], new_var)
    print(f"At iteration {n_var}: r={r} p={p}")
    df_iris[f"correlated_var{n_var}"] = new_var

df_train, df_test = train_test_split(df_iris, test_size=0.2, random_state=42)

model_name = "svm"

importances = {}
coefs = {}


# %% Set up model

y = "species"
X_types = {"continuous": [".*"]}
creator = PipelineCreator(problem_type="classification")
# creator.add("zscore")
if model_name == "svm":
    creator.add("svm", kernel="linear", probability=True)
elif model_name == "rf":
    creator.add("rf", n_estimators=100)

# %% Run original model
print("Testing original model")
X = ["sepal_length", "petal_length"]
scores, model = run_cross_validation(
    X=X,
    y=y,
    data=df_train,
    model=creator,
    return_estimator="final",
    X_types=X_types,
)

print(f"\tScores: {scores['test_score'].mean()}")

r = permutation_importance(
    model, df_test[X], df_test[y], n_repeats=30, random_state=0
)
print(f"\tImportances: {r.importances_mean}")
importances["original"] = r.importances_mean
if model_name == "svm":
    coefs["original"] = model.named_steps["svm"].coef_
elif model_name == "rf":
    coefs["original"] = model.named_steps["rf"].feature_importances_
print(f"\tcoefs/importances: {coefs['original']}")

# %% Plot decision function
if model_name == "svm":
    ax = sns.scatterplot(
        x="sepal_length", y="petal_length", hue=y, data=df_train, s=20
    )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    a = ax.contour(
        XX, YY, Z, colors="k", levels=[0], alpha=0.5, linestyles=["-"]
    )
    ax.set_title("Data with SVM decision function boundaries")
# %% Run model with correlated variables

for n_var in range(0, N_VARS_TO_ADD + 1):
    X = ["sepal_length", "petal_length"] + [
        f"correlated_var{i}" for i in range(1, n_var + 1)
    ]
    scores, model = run_cross_validation(
        X=X,
        y=y,
        data=df_train,
        model=creator,
        return_estimator="final",
        X_types=X_types,
    )

    print(f"\tScores: {scores['test_score'].mean()}")

    r = permutation_importance(
        model, df_test[X], df_test[y], n_repeats=30, random_state=0
    )
    print(f"\tImportances: {r.importances_mean}")
    importances[f"vars_{n_var}"] = r.importances_mean
    if model_name == "svm":
        coefs[f"vars_{n_var}"] = model.named_steps["svm"].coef_
    elif model_name == "rf":
        coefs[f"vars_{n_var}"] = model.named_steps["rf"].feature_importances_
    print(f"\tcoefs/importances: {coefs[f'vars_{n_var}']}")
# %%

all_data = {
    "Permutation Importance": [],
    "Added Variables": [],
    "Variable": [],
}
if model_name == "svm":
    all_data["SVM Coefficient"] = []
elif model_name == "rf":
    all_data["RF Importance"] = []

for t_set, t_values in importances.items():
    t_importances = np.ones(N_VARS_TO_ADD + 2) * np.nan
    t_importances[: len(t_values)] = t_values
    n_vars = [len(t_values) - 2] * len(t_importances)
    t_coefs = np.ones(N_VARS_TO_ADD + 2) * np.nan
    t_coefs[: len(t_values)] = coefs[t_set]
    all_data["Permutation Importance"].extend(t_importances)
    if model_name == "svm":
        all_data["SVM Coefficient"].extend(t_coefs)
    elif model_name == "rf":
        all_data["RF Importance"].extend(t_coefs)
    all_data["Added Variables"].extend(n_vars)
    all_data["Variable"].extend(X)

df = pd.DataFrame(all_data)

# %%
fig_out = Path(__file__).parent / "figures"
sns.set(style="white", palette="viridis")

if model_name == "svm":
    x = "SVM Coefficient"
elif model_name == "rf":
    x = "RF Importance"

plt.figure()
sns.stripplot(
    data=df,
    x=x,
    y="Variable",
    hue="Added Variables",
    palette="viridis",
)
if model_name != "svm":
    plt.xlim(-0.05, 1)
plt.tight_layout()
plt.savefig(fig_out / f"{model_name}_coef.png")
# %%
plt.figure()
sns.stripplot(
    data=df[df["Added Variables"] == 0],
    x=x,
    y="Variable",
    hue="Added Variables",
    palette=sns.color_palette("viridis"),
)
if model_name != "svm":
    plt.xlim(-0.05, 1)
plt.tight_layout()
plt.savefig(fig_out / f"{model_name}_coef0.png")
# %%
plt.figure()
sns.stripplot(
    data=df[df["Added Variables"] == 10],
    x=x,
    y="Variable",
    hue="Added Variables",
    palette="viridis_r",
)
plt.tight_layout()
plt.savefig(fig_out / f"{model_name}_coef10.png")

# %%
plt.figure()
sns.stripplot(
    data=df,
    x="Permutation Importance",
    y="Variable",
    hue="Added Variables",
    palette="viridis",
)
if model_name != "svm":
    plt.xlim(-0.05, 1)
plt.tight_layout()
plt.savefig(fig_out / f"{model_name}_permimp.png")

# %%
plt.figure()
sns.stripplot(
    data=df[df["Added Variables"] == 0],
    x="Permutation Importance",
    y="Variable",
    hue="Added Variables",
    palette=sns.color_palette("viridis"),
)
plt.xlim(-0.05, 1)
plt.tight_layout()
plt.savefig(fig_out / f"{model_name}_permimp0.png")

# %%

plt.figure()
sns.stripplot(
    data=df[df["Added Variables"] == 10],
    x="Permutation Importance",
    y="Variable",
    hue="Added Variables",
    palette="viridis_r",
)
plt.tight_layout()
plt.savefig(fig_out / f"{model_name}_permimp10.png")

# %%
import shap
from julearn.inspect import preprocess
shap.initjs()
df_preprocessed = preprocess(model, data=df_test, X=X)

explainer = shap.Explainer(model.named_steps["svm"], df_train[X])
shap_values = explainer(df_preprocessed)

# %% Waterfall plots of some samples
shap.plots.waterfall(shap_values[0], max_display=12)

shap.plots.waterfall(shap_values[10], max_display=12)

# %%
shap.summary_plot(shap_values, df_preprocessed)

# %%
shap.summary_plot(shap_values, df_preprocessed, plot_type="bar")

# %% Force plot
shap.force_plot(explainer.expected_value, shap_values=shap_values.values)

# %% Clustering plot
clust = shap.utils.hclust(
    df_preprocessed, (df_test[y] == "virginica").astype(int), linkage="single"
)
fig = shap.plots.bar(
    shap_values, clustering=clust, clustering_cutoff=0.9, show=False
)

# %%
