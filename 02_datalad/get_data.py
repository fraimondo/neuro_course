# %%
import datalad.api as dl

# %% Instalamos el dataset
db_path = "/Users/fraimondo/test_datalad"
ds_uri = "https://github.com/OpenNeuroDatasets/ds003097.git"

ds = dl.clone(path=db_path, source=ds_uri)

# %% Descargamos los datos anatomicos del sujeto sub-0001
ds.get("sub-0001/anat/sub-0001_run-1_T1w.nii.gz")

# %% Descartamos los datos anatomicos del sujeto sub-0001
ds.drop("sub-0001/anat/sub-0001_run-1_T1w.nii.gz")

