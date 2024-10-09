#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# Functions
from model.model_functions import predict

# bdtools
from bdtools.nan import nan_filt

# Skimage
from skimage.measure import label
from skimage.morphology import remove_small_objects

# Scipy
from scipy.ndimage import binary_fill_holes

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Ding\data")

#%% Function(s): --------------------------------------------------------------

def process(
        stack, rstack, 
        model_path,
        min_size=64,
        ):
    
    # Execute -----------------------------------------------------------------

    # Predict
    probs = predict(rstack, model_path, img_norm="global", patch_overlap=32)
    
    # Get mask
    mask = np.mean(probs, axis=0) > 0.5
    mask = binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=64)
    mask = label(mask)
           
    return probs, mask
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    rf = 0.1
    model_path = Path.cwd() / "model" /"model_normal"
    
    for path in data_path.glob(f"*rf-{rf}_stack*"):
        
        stack = io.imread(path)
        rstack = io.imread(str(path).replace("stack", "rstack"))
        probs, mask = process(stack, rstack, model_path)
        
        # Save
        io.imsave(
            str(path).replace("stack", "probs"),
            probs.astype("float32"), check_contrast=False,
            )
        io.imsave(
            str(path).replace("stack", "mask"),
            mask.astype("uint8"), check_contrast=False,
            )

