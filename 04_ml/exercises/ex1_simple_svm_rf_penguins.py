# %%
from seaborn import load_dataset
import julearn
from julearn import run_cross_validation
from julearn import PipelineCreator
from julearn.utils import configure_logging

from sklearn.model_selection import KFold

# %%
configure_logging(level="INFO")

df_penguins = load_dataset("penguins")
df_penguins.dropna(inplace=True)
# %% Set-up the variables

X = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"]
y = "body_mass_g"

scoring = "neg_mean_absolute_error"
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# %%
creator = PipelineCreator(problem_type="regression")
creator.add("zscore")
creator.add("svm", kernel="linear")

# %%
scores = run_cross_validation(
    X=X,
    y=y,
    data=df_penguins,
    model=creator,
    scoring=scoring,
    cv=cv,
    return_train_score=True,
)
scores["model"] = "SVM"

# %%
creator2 = PipelineCreator(problem_type="regression")
creator2.add("zscore")
creator2.add("rf")

# %%
scores2 = run_cross_validation(
    X=X,
    y=y,
    data=df_penguins,
    scoring=scoring,
    cv=cv,
    model=creator2,
    return_train_score=True,
)
scores2["model"] = "RF"
# %%

creator3 = PipelineCreator(problem_type="regression")
creator3.add("dummy")

# %%
scores3 = run_cross_validation(
    X=X,
    y=y,
    data=df_penguins,
    scoring=scoring,
    cv=cv,
    model=creator3,
    return_train_score=True,
)
scores3["model"] = "Dummy"

# %%
from julearn.viz import plot_scores
plot_scores(scores, scores2, scores3)

# %%
