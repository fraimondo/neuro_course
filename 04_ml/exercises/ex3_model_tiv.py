# %%
from junifer.storage import HDF5FeatureStorage
from julearn.api import run_cross_validation
from julearn.pipeline import PipelineCreator
import pandas as pd
import seaborn as sns

# %%
storage = HDF5FeatureStorage("./data/ds003097_GMD.hdf5")
# %%
df = storage.read_df("VBM_GM_Schaefer1000x7_Mean")
# %%
df_demographics = pd.read_csv("./data/participants.tsv", sep="\t")
df_demographics.rename(columns={"participant_id": "subject"}, inplace=True)
# %%
df_volumne = pd.read_csv(
    "./data/data-subcortical_type-aseg_measure-volume_hemi-both.tsv", sep="\t"
)
df_volumne.rename(columns={"Measure:volume": "subject"}, inplace=True)
# %%
X = list(df.columns)
df_full = df.merge(df_demographics, on="subject")
df_full = df_full.merge(df_volumne, on="subject")
# %%
creator = PipelineCreator(problem_type="classification")
creator.add("zscore")
creator.add("svm", C=[0.1, 1, 10])
# %%
scores, model, inspector = run_cross_validation(
    X=X,
    y="sex",
    data=df_full,
    model=creator,
    return_train_score=True,
    return_inspector=True,
)

# %%
predictions = inspector.folds.predict()
to_merge = df_full[["sex", "EstimatedTotalIntraCranialVol"]].iloc[
    predictions.index
]
pd.concat([predictions, to_merge], axis=1)
# %%
to_plot = pd.concat([predictions, to_merge], axis=1)
to_plot["correct"] = to_plot["repeat0_p0"] == to_plot["target"]
# %%
sns.boxplot(
    data=to_plot, x="sex", hue="correct", y="EstimatedTotalIntraCranialVol"
)

# %%
