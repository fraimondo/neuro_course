# %%
import seaborn as sns
from seaborn import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from julearn.pipeline import PipelineCreator
from julearn import run_cross_validation
from julearn.viz import plot_scores

# %% Get dataset
cars = sns.load_dataset("mpg")
cars.dropna(inplace=True)
scoring = ["neg_mean_squared_error", "r2", "neg_mean_absolute_error"]

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=42)
creator1 = PipelineCreator(problem_type="regression")
creator1.add("zscore")
creator1.add("linreg")

scores1 = run_cross_validation(
    X=["weight"],
    y="mpg",
    data=cars,
    model=creator1,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
)
scores1["model"] = "LR-no-CR"
# %%
creator2 = PipelineCreator(problem_type="regression")
creator2.add("confound_removal")
creator2.add("zscore")
creator2.add("linreg")

X_types = {"continuous": ["weight"], "confound": ["horsepower"]}

scores2 = run_cross_validation(
    X=["weight", "horsepower"],
    y="mpg",
    data=cars,
    X_types=X_types,
    model=creator2,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
)
scores2["model"] = "LR-CR"

# %% dummy model

creator3 = PipelineCreator(problem_type="regression")
creator3.add("dummy")

X_types = {"continuous": ["weight"]}

scores3 = run_cross_validation(
    X=["weight"],
    y="mpg",
    data=cars,
    X_types=X_types,
    model=creator3,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
)
scores3["model"] = "Dummy"

# %%
plot_scores(scores1, scores2, scores3)
# %%
