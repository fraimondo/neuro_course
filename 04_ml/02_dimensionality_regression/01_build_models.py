# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# %%  Fit a linear regression on random data
samples = [10, 30, 50, 70, 90]  # Number of samples to try
max_dimensions = 100  # Maximum number of dimensions to try

results = {}
results["n_samples"] = []
results["n_dimensions"] = []
results["r2_score"] = []
np.random.seed(42)  # Set the seed to ensure reproducibility
for n_samples in samples:
    y = np.random.rand(n_samples)
    X = np.random.rand(n_samples, max_dimensions)
    for t_d in range(1, max_dimensions + 1):
        model = LinearRegression()
        model.fit(X[:, :t_d], y)
        y_pred = model.predict(X[:, :t_d])
        r2 = r2_score(y, y_pred)
        results["n_samples"].append(n_samples)
        results["n_dimensions"].append(t_d)
        results["r2_score"].append(r2)

df = pd.DataFrame(results)
df_90 = df[df["n_samples"] == 90]  # only with 90 samples for the first plot
# %%  Plot R2 vs number of dimensions
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.lineplot(data=df_90, x="n_dimensions", y="r2_score", ax=ax)
ax.set_title("R2 Score vs Number of Dimensions")
fig.savefig("r2_score_vs_dimensions.png")
# %%  Plot the real vs predicted values (in-sample)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.lineplot(x=y, y=y_pred, ax=ax, ls="none", marker="o")
ax.set_title("Original vs Predicted values")
fig.savefig("original_vs_predicted_max_dimensions_90_samples.png")

# %%  Plot the R2 vs number of dimensions for all sample sizes tested
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.lineplot(
    data=df,
    x="n_dimensions",
    y="r2_score",
    ax=ax,
    hue="n_samples",
    palette="colorblind",
)

for t_sample, t_color in zip(samples, sns.color_palette("colorblind")):
    ax.axvline(t_sample, color=t_color, ls="--", alpha=0.5)

ax.set_title("R2 Score vs Number of Dimensions")
fig.savefig("r2_score_vs_dimensions_samples.png")
# %%
