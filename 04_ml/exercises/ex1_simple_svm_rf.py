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
creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm")

# %%
scores = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    model=creator,
    cv=cv,
    return_train_score=True,
)
scores["model"] = "SVM"

# %%
creator2 = PipelineCreator(problem_type="classification")
creator2.add("zscore")
creator2.add("rf")

# %%
scores2 = run_cross_validation(
    X=X,
    y=y,
    data=df_iris,
    cv=cv,
    model=creator2,
    return_train_score=True,
)
scores2["model"] = "RF"
# %%

from julearn.viz import plot_scores
plot_scores(scores, scores2)

# %%
