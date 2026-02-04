# PG-FKN
Abstract
The inversion of 3D ocean temperature and salinity fields is critical for understanding underwater acoustics and mesoscale dynamics. However, observational data from Argo floats is spatially sparse, while climatological data (WOA) lacks real-time precision. To address this, we propose the PG-FKN (Physics-Guided Residual FNO-KAN) framework.This model integrates Fourier Neural Operators (FNO) for capturing global frequency-domain features with Kolmogorov-Arnold Networks (KAN) for precise local non-linear activation. Adopting a residual learning strategy, the model utilizes the World Ocean Atlas (WOA) as a background field and learns the deviation correction using sparse in-situ data. By embedding physical constraints—including smoothness regularization and dynamic loss weighting—PG-FKN effectively reconstructs high-resolution ocean fields from limited observations.Experiments demonstrate that this architecture significantly reduces Root Mean Square Error (RMSE) compared to traditional interpolation and standard deep learning baselines, ensuring physically consistent vertical profiles.

🔗 RUN
Python 3.11.14

PyTorch 2.9.1 (CUDA Supported)

numpy

pandas

xarray

scipy

pykrige

cartopy

matplotlib

python PG-FKN.py
