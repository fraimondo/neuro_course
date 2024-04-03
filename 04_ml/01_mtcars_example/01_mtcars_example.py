# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
from scipy.special import expit

# %% Load data
cars = sns.load_dataset("mpg")
short_cars = cars.sample(n=20)

# %% Plot weight vs mpg
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=short_cars, x="weight", y="mpg", ax=ax)
fig.savefig("weight_vs_mpg.png")

# %% Fit a linear regression model and check the parameters
model = LinearRegression()
model.fit(short_cars[["weight"]], short_cars["mpg"])
print(f"Alpha={model.intercept_}, Beta={model.coef_}")

# %% Plot the regression line
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=short_cars, x="weight", y="mpg", ax=ax)
xlims = np.array(ax.get_xlim())  # Get the bounds of the x-axis
y = model.intercept_ + model.coef_[0] * xlims  # Compute the y values
plt.plot(xlims, y)
fig.savefig("weight_vs_mpg_regression.png")

# %%  Predict some values
to_predict = np.array([2130, 3300, 4460])
y_pred = model.predict(to_predict.reshape(-1, 1))

# %% Do some animations
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=short_cars, x="weight", y="mpg", ax=ax)
xlims = ax.get_xlim()
ylims = ax.get_ylim()


def init():
    plt.plot(np.array(xlims), y, c="k")


def update(frame):
    i_pred = frame // 3
    i_stage = frame % 3
    print("i_pred", i_pred, "i_stage", i_stage)
    if i_stage == 0:
        plt.plot((to_predict[i_pred]), (ylims[0]), c="r", marker="o")
    elif i_stage == 1:
        plt.plot(
            (to_predict[i_pred], to_predict[i_pred]),
            (ylims[0], y_pred[i_pred]),
            c="r",
            ls="--",
            marker="o",
        )
    elif i_stage == 2:
        plt.plot(
            (xlims[0], to_predict[i_pred]),
            (y_pred[i_pred], y_pred[i_pred]),
            c="r",
            ls="--",
            marker="o",
        )
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)


ani = FuncAnimation(
    fig, update, frames=len(to_predict) * 3, init_func=init, repeat=False
)
ani.save("weight_vs_mpg_regression.gif", writer="ffmpeg", fps=0.7, dpi=300)

# %% Now the logistic regression
short_cars = cars.sample(n=30, random_state=42)  # reample so we see some overlap
short_cars["cyl_bin"] = short_cars["cylinders"].apply(lambda x: "8" if x > 4 else "4/6")
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.swarmplot(x="cyl_bin", y="weight", data=short_cars, ax=ax, hue="cyl_bin")
fig.savefig("weight_vs_cylinders_swarm.png")
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data=short_cars, x="weight", y="cyl_bin", ax=ax, hue="cyl_bin")
ax.invert_yaxis()
model = LogisticRegression()
model.fit(short_cars[["weight"]], short_cars["cyl_bin"])
print(f"Alpha={model.intercept_}, Beta={model.coef_}")

xlims = ax.get_xlim()
x = np.linspace(xlims[0], xlims[1], 300)

y = expit(x * model.coef_ + model.intercept_).ravel()
plt.plot(x, y, label="Logistic Regression Model", color="red", linewidth=1)
ax.set_yticks(np.linspace(0, 1, 11))
ax.set_yticklabels([f"{x:.2f}" for x in np.linspace(0, 1, 11)])
ax.set_ylabel("Probability of 8 cylinders")
ax.hlines(0.5, *xlims, linestyles="--", color="gray")
mid_thresh_idx = np.argmin(np.abs(y - 0.5))
ax.vlines(x[mid_thresh_idx], 0, 1, linestyles="--", color="gray")
fig.savefig("weight_vs_cylinders.png")
# %%
