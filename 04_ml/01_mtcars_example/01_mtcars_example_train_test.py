# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# %% Load data
cars = sns.load_dataset("mpg")
short_cars = cars.sample(n=50, random_state=42)
short_cars["set"] = ["train" if x == 1 else "test" 
                     for x in np.random.randint(0, 2, 50)]
short_cars_train = short_cars[short_cars["set"] == "train"]
short_cars_test = short_cars[short_cars["set"] == "test"]
model = LinearRegression()
model.fit(short_cars_train[["weight"]], short_cars_train["mpg"])
y_pred_train = model.predict(short_cars_train[["weight"]])
# %% Plot weight vs mpg
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=short_cars_train, x="weight", y="mpg", ax=ax)
xlims = np.array(ax.get_xlim())  # Get the bounds of the x-axis
y = model.intercept_ + model.coef_[0] * xlims  # Compute the y values
plt.plot(xlims, y)
fig.savefig("train_weight_vs_mpg_regression.png")

# %% Fit a linear regression model and check the parameters


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x=short_cars_train["mpg"], y=y_pred_train, ax=ax)
ax.set_xlabel("MPG (Real Value)")
ax.set_ylabel("MPG (Predicted Value)")
mse = mean_squared_error(short_cars_train["mpg"], y_pred_train)
ax.set_title(f"Real vs Predicted MPG (MSE={mse:.2f})")
ax.set_xlim(10, 50)
ax.set_ylim(10, 50)
ax.plot([10, 50], [10, 50], color="k", alpha=0.5, ls="--")
fig.savefig("real_vs_pred.png")

# %%
model = LinearRegression()

model.fit(short_cars_train[["weight"]], short_cars_train["mpg"])
y_pred_train = model.predict(short_cars_train[["weight"]])
y_pred_test = model.predict(short_cars_test[["weight"]])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(
    x=short_cars_train["mpg"], y=y_pred_train, ax=ax, color="gray", alpha=0.5, label="Train"
)
sns.scatterplot(
    x=short_cars_test["mpg"], y=y_pred_test, ax=ax, color="red", label="Test"
)
ax.set_xlabel("MPG (Real Value)")
ax.set_ylabel("MPG (Predicted Value)")
mse = mean_squared_error(short_cars_test["mpg"], y_pred_test)
ax.set_title(f"Real vs Predicted MPG (MSE={mse:.2f})")
ax.set_xlim(10, 50)
ax.set_ylim(10, 50)
ax.plot([10, 50], [10, 50], color="k", alpha=0.5, ls="--")
fig.savefig("real_vs_pred_train.png")

# %%
