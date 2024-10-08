#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# Functions
from model.model_functions import predict

# bdtools
from bdtools.nan import nan_replace

# Skimage
from skimage.measure import label
from skimage.filters import median
from skimage.morphology import disk

# Scipy
from scipy.ndimage import uniform_filter1d, binary_fill_holes

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Ding\data")

#%% Function(s): --------------------------------------------------------------

def process(
        raw, 
        model_path,
        outlier_thresh=0.1, 
        window_size=501,
        ):
    
    # Nested function(s) ------------------------------------------------------
    
    def remove_outliers(stack, outlier_thresh):
        stack = stack.astype("float32")
        std = np.std(stack, axis=0)
        std_med = median(std, footprint=disk(1))
        std_sub = std - std_med
        outliers = std_sub > outlier_thresh
        for img in stack:
            img[outliers] = np.nan
        return nan_replace(stack)
    
    def rolling_avg(stack, window_size):
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")
        pad = window_size // 2
        stack = np.pad(
            stack, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
        stack = uniform_filter1d(stack, size=window_size, axis=0)
        return stack[pad:-pad]
    
    # Execute -----------------------------------------------------------------

    # Process
    stack = remove_outliers(raw, outlier_thresh)  
    rstack = rolling_avg(stack, window_size)
    probs = predict(rstack, model_path, img_norm="global", patch_overlap=0)
    mask = np.mean(probs, axis=0)
    mask = mask > 0.5
    mask = binary_fill_holes(mask)
    mask = label(mask)
    
    # Save
    io.imsave(
        path.parent / (path.name.replace("raw", "stack")),
        stack.astype("uint8"), check_contrast=False,
        )  
    io.imsave(
        path.parent / (path.name.replace("raw", "rstack")),
        rstack.astype("float32"), check_contrast=False,
        )
    io.imsave(
        path.parent / (path.name.replace("raw", "probs")),
        probs.astype("float32"), check_contrast=False,
        )
    io.imsave(
        path.parent / (path.name.replace("raw", "mask")),
        mask.astype("uint8"), check_contrast=False,
        )
    
    return stack, rstack, probs, mask
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    rf = 0.1
    outlier_thresh = 0.1
    window_size = 501
    model_path = Path.cwd() / "model" /"model_normal"
    
    for path in data_path.glob(f"*rf-{rf}_raw*"):
        
        raw = io.imread(path)
        stack, rstack, prds, mask = process(
            raw, model_path, 
            outlier_thresh=outlier_thresh, 
            window_size=window_size,
            )
