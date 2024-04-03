#!/bin/bash

# Lugar donde vamos a instalar el dataset
cd /Users/fraimondo/test_datalad

# Instalamos el dataset
datalad install https://github.com/OpenNeuroDatasets/ds003097.git

# Tenemos que estar en el directorio del dataset para poder trabajar con él
cd ds003097

# Descargamos los datos anatomicos del sujeto sub-0001
datalad get sub-0001/anat/sub-0001_run-1_T1w.nii.gz

# Descartamos los datos anatomicos del sujeto sub-0001
datalad drop sub-0001/anat/sub-0001_run-1_T1w.nii.gz

