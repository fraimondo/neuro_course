# %%
from seaborn import load_dataset
import julearn
from julearn import run_cross_validation
from julearn import PipelineCreator
from julearn.utils import configure_logging

from sklearn.model_selection import StratifiedKFold

# %%
configure_logging(level="INFO")

df_iris = load_dataset("iris")

df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

# %% Set-up the variables

X = ["sepal_length", "sepal_width", "petal_length"]
y = "species"

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %%
creator1 = PipelineCreator(problem_type="classification")
creator1.add("zscore")
creator1.add("svm", kernel="linear", C=[0.1, 1, 10, 100])

inner_cv = StratifiedKFold(n_splits=3)

# %%
scores, model, inspector = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model=creator1,
    search_params={"cv": inner_cv},
    return_train_score=True,
    return_inspector=True,
    cv=cv,
)

# %%

creator2 = PipelineCreator(problem_type="classification")
creator2.add("zscore")
creator2.add("svm", kernel=["rbf", "sigmoid"], C=[0.1, 1, 10], gamma=[0.1, 1, 10])


scores, model, inspector = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model=[creator1, creator2],
    return_train_score=True,
    return_inspector=True,
)
# %%
