#%% Imports -------------------------------------------------------------------

import tifffile
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# Skimage
from skimage.transform import rescale

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Ding\data")
rf = 0.1

#%% Function: extract() -------------------------------------------------------

def extract(path, rf):
    
    # Nested function(s) ------------------------------------------------------
    
    def _extract(img, rf):
        return rescale(img, rf, order=0)
    
    # Execute -----------------------------------------------------------------
    
    memmap = tifffile.memmap(str(path))
    stack = Parallel(n_jobs=-1)(
        delayed(_extract)(memmap[t,...], rf)
        for t in range(memmap.shape[0])
        )
    
    return np.stack(stack)   

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in data_path.glob("*.ome"):
        stack = extract(path, rf)
        
        if path.name == "Exp1.ome":
            stack = stack[:-1]
            
        io.imsave(
            path.parent / f"{path.stem}_rf-{rf}_stack.tif",
            stack.astype("uint8"), check_contrast=False,
            )        