#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path

# Functions
from model.model_functions import predict

# bdtools
from bdtools.nan import nan_filt
from bdtools.mask import get_edt
from bdtools.norm import norm_gcn, norm_pct

# Skimage
from skimage.measure import label
from skimage.morphology import remove_small_objects

# Scipy
from scipy.ndimage import binary_fill_holes, uniform_filter1d

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")
model_path = Path.cwd() / "model" /"model_normal"

# Parameters
rf = 0.1
window_size = 501
min_size = 256

#%% Function(s): --------------------------------------------------------------

def process(stack, model_path, window_size=501, min_size=64):
    
    # Nested function(s) --------------------------------------------------------
    
    def rolling_avg(stack, window_size):
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")
        pad = window_size // 2
        stack = np.pad(
            stack, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
        stack = uniform_filter1d(stack, size=window_size, axis=0)
        return stack[pad:-pad]
    
    # Execute -----------------------------------------------------------------

    # Rolling average
    rstack = rolling_avg(stack, window_size)

    # Predict
    probs = predict(
        rstack[::25], model_path, img_norm="global", patch_overlap=32)
    
    # Get mask
    mask = np.mean(probs, axis=0) > 0.5
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_size)
    mask = label(mask)
    
    # Get edt
    edt = get_edt(mask)
    
    # Filter stack
    filt = nan_filt(
        norm_pct(norm_gcn(stack), pct_low=0, pct_high=100), 
        mask=mask > 0, kernel_size=(1, 3, 3), iterations=3,
        )
                   
    return probs, mask, edt, filt
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    for path in data_path.glob(f"*rf-{rf}_stack*"):
        
        t0 = time.time()
        
        print(path.name)
        
        stack = io.imread(path)
        probs, mask, edt, filt = process(
            stack, model_path, window_size=window_size, min_size=min_size)
        
        t1 = time.time()
        print(f"runtime : {t1 - t0:.3f}s")
        
        # Save
        io.imsave(
            str(path).replace("stack", "probs"),
            probs.astype("float32"), check_contrast=False,
            )
        io.imsave(
            str(path).replace("stack", "mask"),
            mask.astype("uint8"), check_contrast=False,
            )
        io.imsave(
            str(path).replace("stack", "edt"),
            edt.astype("float32"), check_contrast=False,
            )
        io.imsave(
            str(path).replace("stack", "filt"),
            filt.astype("float32"), check_contrast=False,
            )
