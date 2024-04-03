# %%
from nilearn import datasets, plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# %%
dataset = datasets.fetch_development_fmri(n_subjects=1)

# print basic information on the dataset
print(f"First subject functional nifti image (4D) is at: {dataset.func[0]}")
dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
labels = [
    "Posterior Cingulate Cortex",
    "Left Temporoparietal junction",
    "Right Temporoparietal junction",
    "Medial prefrontal cortex",
]

from nilearn.maskers import NiftiSpheresMasker

masker = NiftiSpheresMasker(
    dmn_coords,
    radius=8,
    detrend=True,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    low_pass=0.1,
    high_pass=0.01,
    t_r=2,
    memory="nilearn_cache",
    memory_level=1,
    verbose=2,
    clean__butterworth__padtype="even",  # kwarg to modify Butterworth filter
)

# Additionally, we pass confound information to ensure our extracted
# signal is cleaned from confounds.

func_filename = dataset.func[0]
confounds_filename = dataset.confounds[0]

time_series = masker.fit_transform(
    func_filename, confounds=[confounds_filename]
)
# %%


for time_serie, label in zip(time_series.T, labels):
    plt.plot(time_serie, label=label)

plt.title("Default Mode Network Time Series")
plt.xlabel("Scan number")
plt.ylabel("Normalized signal")
plt.legend()
plt.tight_layout()

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.regplot(x=time_series[:, 0], y=time_series[:, 1], ax=ax)
ax.set_xlabel(labels[0])
ax.set_ylabel(labels[1])
r, _ = pearsonr(time_series[:, 0], time_series[:, 1])
ax.annotate(f"r = {r:.2f}", (0.1, 0.9), xycoords="axes fraction")

# %%


from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(
    kind="correlation",
)
correlation_matrix = correlation_measure.fit_transform([time_series])[0]


# Make a large figure
# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)
# The labels we have start with the background (0), hence we skip the
# first label
# matrices are ordered for block-like representation
plotting.plot_matrix(
    correlation_matrix,
    figure=(3, 3),
    labels=labels,
    vmax=0.8,
    vmin=-0.8,
    # title="Correlation",
    reorder=True,
)
# %%
