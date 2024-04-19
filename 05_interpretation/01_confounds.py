# %%
import seaborn as sns
from seaborn import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# %% Get dataset
cars = sns.load_dataset("mpg")
cars.dropna(inplace=True)
short_cars = cars.sample(n=40, random_state=42)

# %% Check original model
model = LinearRegression()
model.fit(short_cars[["weight"]], short_cars["mpg"])
mse = mean_squared_error(
    y_true=short_cars["mpg"],
    y_pred=model.predict(short_cars[["weight"]])
)
print(f"Alpha={model.intercept_}, Beta={model.coef_}, mse={mse}")

# %% Plot the data and regression line
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=short_cars, x="weight", y="mpg", ax=ax)
xlims = np.array(ax.get_xlim())  # Get the bounds of the x-axis
y = model.intercept_ + model.coef_[0] * xlims  # Compute the y values
ax.plot(xlims, y)
ax.set_title(f"Weights vs MPG (MSE: {mse:.2f})")

# %%  Check the relation with the horsepower
fig = plt.figure()
sns.relplot(y="weight", x="horsepower", data=short_cars)

# %%  Fit a linear regression with the horsepower
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x="horsepower", y="weight", data=short_cars, ax=ax)

model = LinearRegression().fit(
    X=short_cars[["horsepower"]], y=short_cars["weight"]
)
xlims = np.array(ax.get_xlim())  # Get the bounds of the x-axis
y = model.intercept_ + model.coef_[0] * xlims  # Compute the y values
ax.plot(xlims, y)
print(f"Alpha={model.intercept_}, Beta={model.coef_}, mse={mse}")

# %%  Remove the effect of the horsepower
y_pred = model.predict(X=short_cars[["horsepower"]])
short_cars["weight_pred"] = y_pred
short_cars["weight_cr"] = short_cars["weight"] - y_pred

# %% Plot the residuals
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x="horsepower", y="weight", data=short_cars, ax=ax)
xlims = np.array(ax.get_xlim())  # Get the bounds of the x-axis
y = model.intercept_ + model.coef_[0] * xlims  # Compute the y values
ax.plot(xlims, y)
x = np.c_[short_cars["horsepower"], short_cars["horsepower"]].T
y = np.c_[short_cars["weight"], short_cars["weight_pred"]].T
ax.plot(x, y, color="red", ls=":")


# %%  Plot the relation between horspower and weight_cr
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x="horsepower", y="weight_cr", data=short_cars, ax=ax)

# %%  Now fit a regression with the corrected weight and mpg
model = LinearRegression()
model.fit(short_cars[["weight_cr"]], short_cars["mpg"])
mse = mean_squared_error(
    y_true=short_cars["mpg"],
    y_pred=model.predict(short_cars[["weight_cr"]])
)
print(f"Alpha={model.intercept_}, Beta={model.coef_}, mse={mse}")

# %% Plot the regression line
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=short_cars, x="weight_cr", y="mpg", ax=ax)
xlims = np.array(ax.get_xlim())  # Get the bounds of the x-axis
y = model.intercept_ + model.coef_[0] * xlims  # Compute the y values
ax.plot(xlims, y)
ax.set_title(f"Weights vs MPG (MSE: {mse:.2f})")
# %%

creator1 = PipelineCreator(problem_type="regression")
creator1.add("zscore")
creator1.add("linear", fit_intercept=False)

