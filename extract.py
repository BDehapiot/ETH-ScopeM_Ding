#%% Imports -------------------------------------------------------------------

import time
import tifffile
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# Skimage
from skimage.transform import rescale

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")
# data_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Ding\data")

#%% Function: extract() -------------------------------------------------------

def extract(path, rf):
    
    # Nested function(s) ------------------------------------------------------
    
    def _extract(img, rf):
        return rescale(img, rf, order=1, preserve_range=True)
        
    # Execute -----------------------------------------------------------------
    
    memmap = tifffile.memmap(str(path))
    stk = Parallel(n_jobs=-1)(
        delayed(_extract)(memmap[t,...], rf)
        for t in range(memmap.shape[0])
        )
    stk = np.stack(stk)
    if path.name == "Exp1.ome":
        stk = stk[:-1]
    
    return stk

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in data_path.glob("*.ome"): 
        
        rf = 0.1 if "Exp4" in path.name else 0.05 
        
        t0 = time.time()
        print(f"extract : {path.name} : ", end="", flush=True)
        stk = extract(path, rf)
        t1 = time.time()
        print(f"{t1 - t0:.3f}s")
        
        # Save
        io.imsave(
            data_path / f"{path.stem}_rf-{rf}_stk.tif",
            stk.astype("float32"), check_contrast=False,
            )   