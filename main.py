#%% Imports -------------------------------------------------------------------

import tifffile
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed 

# bdtools
from bdtools.norm import norm_gcn, norm_pct
from bdtools.nan import nan_replace

# Skimage
from skimage.transform import rescale
from skimage.morphology import disk
from skimage.filters import median

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Ding\data")
data_name = "Exp1.ome"

#%% 

memmap = tifffile.memmap(str(Path(data_path, data_name)))
nT, nY, nX = memmap.shape

def load(memmap):
    return rescale(memmap, 0.1, order=0)
    
stack = Parallel(n_jobs=-1)(
    delayed(load)(memmap[t,...])
    for t in range(nT)
    )
stack = np.stack(stack).astype("float32")
if data_name == "Exp1.ome":
    stack = stack[:-1]

#%%

std = np.std(stack, axis=0)
std_filt = median(std, footprint=disk(1))
std_sub = std - std_filt
outliers = std_sub > 0.1
for img in stack:
    img[outliers] = np.nan
stack = nan_replace(stack)

test = np.mean(stack, axis=0)
test = median(test, footprint=disk(3))

import napari
viewer = napari.Viewer()
viewer.add_image(stack)
viewer.add_image(test)
