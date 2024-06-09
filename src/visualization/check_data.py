from pathlib import Path
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
path = Path("data/processed/Train/subtiles/Tile4/0_0")

# Load the data
landsat = xr.open_dataarray(path / "landsat_rgb.nc")
print(landsat.values.shape)
ndvi = xr.open_dataarray(path / "landsat_ndvi.nc").astype(int)
print(ndvi.values.shape)
#plots rgb image of landsat
landsat_rgb = landsat
landsat_rgb.plot.imshow()
plt.savefig("landsat_rgb.png")

#plots the NDVI of the landsat data
plt.figure()
ndvi.squeeze().plot.imshow()
plt.savefig("landsat_ndvi.png")









