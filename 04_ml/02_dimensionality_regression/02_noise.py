# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# %%  Fit a linear regression on random data
n_samples = 100
n_dimensions = 100  # Maximum number of dimensions to try

results = {}
results["n_samples"] = []
results["n_dimensions"] = []
results["r2_score"] = []
np.random.seed(42)  # Set the seed to ensure reproducibility

y = np.random.rand(n_samples)
X = np.random.rand(n_samples, n_dimensions)
y_mod = y + np.random.normal(0, 0.1, n_samples)  # add some noise to the data
X[:, 0] = y_mod  # make the first dimension the same as the target variable

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)


# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.histplot(model.coef_, ax=ax, bins=20)
ax.set_title("Histogram of Coefficients (100 features)")
ax.set_xlabel("Coefficient Value")
fig.savefig("noise_histogram_coefficients_100d.png")
ax.axvline(model.coef_[0], color="r", ls="--", label="Good coefficient")
fig.savefig("noise_histogram_coefficients_100d_w.png")
# %%
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.scatterplot(x=y, y=y_mod, ax=ax)
ax.set_xlabel("Target")
ax.set_ylabel("Good Feature")
ax.set_title("Good Feature vs Target")
fig.savefig("noise_good_feature.png")
# %%
model = LinearRegression()
model.fit(X[:, :20], y)
y_pred = model.predict(X[:, :20])
r2 = r2_score(y, y_pred)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.histplot(model.coef_, ax=ax, bins=20)
ax.set_title("Histogram of Coefficients (20 features)")
ax.set_xlabel("Coefficient Value")
ax.axvline(model.coef_[0], color="r", ls="--", label="Good coefficient")
fig.savefig("noise_histogram_coefficients_20d.png")
# %%
