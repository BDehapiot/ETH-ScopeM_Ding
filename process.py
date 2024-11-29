#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path

# Functions
from model.model_functions import predict

# bdtools
from bdtools.nan import nan_filt
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
rf = 0.05
window_size = 501
min_size = 2560 * rf # 256 for rf = 0.1

#%% Function(s): --------------------------------------------------------------

def process(stk, model_path, window_size=501, min_size=256):
    
    # Nested function(s) --------------------------------------------------------
    
    def rolling_avg(stk, window_size):
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")
        pad = window_size // 2
        stk = np.pad(
            stk, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
        stk = uniform_filter1d(stk, size=window_size, axis=0)
        return stk[pad:-pad]
    
    # Execute -----------------------------------------------------------------

    # Rolling average
    rstk = rolling_avg(stk, window_size)

    # Predict
    prb = predict(
        rstk[::25], model_path, img_norm="global", patch_overlap=32)
    
    # Get mask
    msk = np.mean(prb, axis=0) > 0.5
    msk = binary_fill_holes(msk)
    msk = remove_small_objects(msk, min_size=min_size)
    msk = label(msk)
    
    # Filter stack
    flt = nan_filt(
        norm_pct(norm_gcn(stk), pct_low=0, pct_high=100), 
        mask=msk > 0, kernel_size=(1, 3, 3), iterations=1,
        )
                   
    return prb, msk, flt
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":

    for path in data_path.glob(f"*rf-{rf}_stk*"):
                
        t0 = time.time()
        
        print(path.name)
        
        stk = io.imread(path)
        prb, msk, filt = process(
            stk, model_path, window_size=window_size, min_size=min_size)
        
        t1 = time.time()
        print(f"runtime : {t1 - t0:.3f}s")
        
        # Save
        io.imsave(
            str(path).replace("stk", "prb"),
            prb.astype("float32"), check_contrast=False,
            )
        io.imsave(
            str(path).replace("stk", "msk"),
            msk.astype("uint8"), check_contrast=False,
            )
        io.imsave(
            str(path).replace("stk", "flt"),
            filt.astype("float32"), check_contrast=False,
            )
