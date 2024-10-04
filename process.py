#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# bdtools
from bdtools.nan import nan_replace

# Skimage
from skimage.filters import median
from skimage.morphology import disk, ball

# Scipy
from scipy.ndimage import uniform_filter1d

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Ding\data")

#%% Function(s): --------------------------------------------------------------

def process(stack, tresh=0.1):
    stack = stack.astype("float32")
    std = np.std(stack, axis=0)
    std_med = median(std, footprint=disk(1))
    std_sub = std - std_med
    outliers = std_sub > tresh
    for img in stack:
        img[outliers] = np.nan
    return nan_replace(stack)

def roll_avg(stack, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd.")
    pad = window_size // 2
    stack = np.pad(
        stack, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
    stack = uniform_filter1d(stack, size=window_size, axis=0)
    return stack[pad:-pad]

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    rf = 0.1
    window_size = 501
    
    for path in data_path.glob(f"*rf-{rf}*"):
        stack = process(io.imread(path))  
        rstack = roll_avg(stack, window_size)
        io.imsave(
            path.parent / (path.name.replace("raw", "stack")),
            stack.astype("uint8"), check_contrast=False,
            )  
        io.imsave(
            path.parent / (path.name.replace("raw", "rstack")),
            rstack.astype("float32"), check_contrast=False,
            )        

            
#%%

# from skimage.filters import gaussian

# -------------------------------------------------------------------------

# t0 = time.time()
# rstack = roll_avg(stack, 501)
# t1 = time.time()
# print(f"roll_avg() : {t1 - t0:.3f}")

# io.imsave(
#     path.parent / (path.stem.replace("stack", "rstack")),
#     rstack.astype("float32"), check_contrast=False,
#     )        

# -------------------------------------------------------------------------

# import napari
# viewer = napari.Viewer()
# viewer.add_image(stack, contrast_limits=(98, 120))
# viewer.add_image(rstack, contrast_limits=(98, 120))
        