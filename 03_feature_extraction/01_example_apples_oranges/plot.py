# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv", sep=";")
df1 = df.query("Sabor == 'Dulce'")
df2 = df.query("Sabor != 'Riquísimo'")
# df3 = df.query("Sabor == 'Riquísimo'")
# %% Shop 1

fig, ax = plt.subplots(1, 1, figsize=(8, 2))
sns.scatterplot(data=df1, x="Peso", y=0, hue="Fruta", ax=ax)
ax.set_ylabel("")
ax.set_yticks([])
ax.set_ylim(-0.2, 0.5)
ax.set_xlim(199, 330)
fig.savefig("shop1.png")
# %% Shop 1 + 2

fig, ax = plt.subplots(1, 1, figsize=(8, 2))
sns.scatterplot(data=df2, x="Peso", y=0, hue="Fruta", ax=ax)
ax.set_ylabel("")
ax.set_yticks([])
ax.set_ylim(-0.2, 0.5)
ax.set_xlim(199, 330)
fig.savefig("shop12.png")
# %% All shops
fig, ax = plt.subplots(1, 1, figsize=(8, 2))
sns.scatterplot(data=df, x="Peso", y=0, hue="Fruta", ax=ax)
ax.set_ylabel("")
ax.set_yticks([])
ax.set_ylim(-0.2, 0.5)
ax.set_xlim(199, 330)
fig.savefig("shopall.png")
# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
sns.scatterplot(data=df, x="Peso", y="Sabor", hue="Fruta", ax=ax)
ax.set_xlim(199, 330)
fig.savefig("taste.png")
# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
sns.scatterplot(data=df, x="Peso", y="Volumen", hue="Fruta", style="Sabor", ax=ax)
ax.set_xlim(199, 330)
fig.savefig("taste_vol.png")
# %%
