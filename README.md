# PG-FKN
# Global Spectra and Local Splines: A Physics-guided FNO-KAN Network for Reconstructing Subsurface Ocean Fields
Abstract
The inversion of 3D ocean temperature and salinity fields is critical for understanding underwater acoustics and mesoscale dynamics. However, observational data from Argo floats is spatially sparse, while climatological data (WOA) lacks real-time precision. To address this, we propose the PG-FKN (Physics-Guided Residual FNO-KAN) framework.This model integrates Fourier Neural Operators (FNO) for capturing global frequency-domain features with Kolmogorov-Arnold Networks (KAN) for precise local non-linear activation. Adopting a residual learning strategy, the model utilizes the World Ocean Atlas (WOA) as a background field and learns the deviation correction using sparse in-situ data. By embedding physical constraints—including smoothness regularization and dynamic loss weighting—PG-FKN effectively reconstructs high-resolution ocean fields from limited observations.Experiments demonstrate that this architecture significantly reduces Root Mean Square Error (RMSE) compared to traditional interpolation and standard deep learning baselines, ensuring physically consistent vertical profiles.

🔗 Python scrpits
PG-FKN.py : Main Model: Physics-Guided Residual FNO-KAN
FNO.py : Fourier Neural Operator Implementation
PINN.py : Physics-Informed Neural Networks (PINN) Baseline
U-Net.py : U-Net Baseline
kriging.py : Kriging Baseline
vs.FFPG.py : Comparative Experiments with the FFPG Method

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

🔗 Data and models
The World Ocean Atlas 2023 (WOA23) dataset used in this study is available from the NOAA National Centers for Environmental Information (NCEI) at (https://www.ncei.noaa.gov/products/world-ocean-atlas)
The global Argo float data were obtained from the Argo Global Data Assembly Center (GDAC): https://argo.ucsd.edu/data/ (accessed on April 2025)
