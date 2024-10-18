#%% Imports -------------------------------------------------------------------

import tifffile
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# Skimage
from skimage.transform import rescale

# Scipy
from scipy.ndimage import uniform_filter1d

#%% Inputs --------------------------------------------------------------------

# Path
data_path = Path("D:\local_Ding\data")

# Parameters
rf = 0.1
window_size = 501

#%% Function: extract() -------------------------------------------------------

def extract(path, rf, window_size):
    
    # Nested function(s) ------------------------------------------------------
    
    def _extract(img, rf):
        return rescale(img, rf, order=1)
    
    def rolling_avg(stack, window_size):
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")
        pad = window_size // 2
        stack = np.pad(
            stack, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
        stack = uniform_filter1d(stack, size=window_size, axis=0)
        return stack[pad:-pad]
    
    # Execute -----------------------------------------------------------------
    
    memmap = tifffile.memmap(str(path))
    stack = Parallel(n_jobs=-1)(
        delayed(_extract)(memmap[t,...], rf)
        for t in range(memmap.shape[0])
        )
    stack = np.stack(stack)
    if path.name == "Exp1.ome":
        stack = stack[:-1]

    rstack = rolling_avg(stack, window_size)
    
    return stack, rstack   

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in data_path.glob("*.ome"):
        
        stack, rstack = extract(path, rf, window_size)
                    
        io.imsave(
            data_path / f"{path.stem}_rf-{rf}_stack.tif",
            stack.astype("float32"), check_contrast=False,
            )       
        
        io.imsave(
            data_path / f"{path.stem}_rf-{rf}_rstack.tif",
            rstack.astype("float32"), check_contrast=False,
            )  