# Testing audio compression autoencoders for seismology: moving toward foundation models 

Authors: Laura Laurenti, Christopher W. Johnson, Daniele Trappolini, Elisa Tinti, Fabio Galasso, Chris Marone. 

The models are available in folder "models" at https://zenodo.org/records/15484940

Models are trained on earthquakes from datasets STEAD and INSTANCE, and noise from STEAD when named "STEAD_INSTANCE_chX"; and only on STEAD when named "STEAD_chX"

We make available the following folders for the experimental tests: 
- reconstruction: the notebook uses a subset from our test set from Stead as evaluation available at https://zenodo.org/records/15484940
The purpose is to get the embedded version of the input signal, and the reconstruction made by the model, with or without the quantizer.



- Norcia 
Data for this task are publicly available at: https://zenodo.org/records/12806081
Needed data are: 
  - dataframe_pre_NRCA
  - dataframe_visso_NRCA
  - dataframe_post_NRCA



- ground_motion
Data for this task are publicly available at: https://zenodo.org/records/5767221 and https://github.com/StefanBloemheuvel/GCNTimeseriesRegression/tree/main/data

Needed data are: 
  - chosenStations.pkl
  - inputs_ci.npy
  - meta.npy
  - station_coords.npy
  - targets.npy


- phase_detection: To make the code runnable and user-friendly, we train and test this code on on a small subset of the test set from STEAD, available at https://zenodo.org/records/15484940, which was also used in the reconstruction examples.
The full experiments and analyses described in the paper were conducted using the entire STEAD and INSTANCE datasets, which are publicly available at:
STEAD: https://github.com/smousavi05/STEAD
INSTANCE: https://github.com/INGV/instance
Models in eval_models folder are intended to be used in testing, in the eval_phase_detection notebook.





packages version:
pandas                    2.1.3
torch                     2.1.1+cu121
numpy                     1.26.2
matplotlib                3.8.2
transformers              4.38.0.dev0
scikit-learn              1.3.2 

