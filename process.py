#%% Imports -------------------------------------------------------------------

import re
import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# bdtools
from bdtools.nan import nan_filt
from bdtools.models import UNet

# Skimage
from skimage.morphology import remove_small_objects

# Scipy
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")
# data_path = Path.cwd() / "_local"

#%% Function(s): --------------------------------------------------------------

def process(path):
    
    # Nested function(s) --------------------------------------------------------
        
    # Asymetric least square (asl) 
    def get_als(y, lam=1e7, p=0.001, niter=5): # Parameters
        L = len(y)
        D = diags([1, -2, 1],[0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for i in range(niter):
            W = spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z
    
    def _process(y):
        sub = y - get_als(y)
        grd = np.gradient(sub, axis=0)
        grd = np.roll(grd, 1, axis=0)
        return sub, grd
    
    # Execute -----------------------------------------------------------------
        
    # Fetch rescaling factor (rf)
    match = re.search(r"rf-([-+]?\d*\.\d+|\d+)", path.name)
    if match: 
        rf = float(match.group(1))     
    
    # Load data
    stk = io.imread(path)
    
    # Predict & create mask
    unet = UNet(load_name="model_64_normal_2000-200_1")
    prd = unet.predict(np.mean(stk, axis=0), verbose=0)
    msk = prd > 0.5
    msk = remove_small_objects(msk, min_size=2048*rf)
    
    # Filter stack
    flt = nan_filt(stk, mask=msk > 0, kernel_size=(1, 3, 3), iterations=3)
    
    # Temporal analysis
    sub = np.full_like(flt, np.nan)
    grd = np.full_like(flt, np.nan)
    for lab in np.unique(msk)[1:]:
        idxs = np.where(msk == lab)
        flt_tvals = flt[:, idxs[0], idxs[1]]
        outputs = Parallel(n_jobs=-1)(
            delayed(_process)(flt_tvals[:, i])
            for i in range(flt_tvals.shape[1])
            )
        sub_tvals = np.vstack([data[0] for data in outputs]).T
        grd_tvals = np.vstack([data[1] for data in outputs]).T
        sub[:, idxs[0], idxs[1]] = sub_tvals
        grd[:, idxs[0], idxs[1]] = grd_tvals
                   
    return flt, sub, grd
    
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in data_path.glob(f"*stk.tif"):
                
        flt, sub, grd = process(path)
        
        # Save
        io.imsave(
            data_path / (str(path.name).replace("stk", "flt")),
            flt.astype("float32"), check_contrast=False,
            )
        io.imsave(
            data_path / (str(path.name).replace("stk", "sub")),
            sub.astype("float32"), check_contrast=False,
            )
        io.imsave(
            data_path / (str(path.name).replace("stk", "grd")),
            grd.astype("float32"), check_contrast=False,
            )
            
        # # Display
        # viewer = napari.Viewer()
        # viewer.add_image(flt)
        # viewer.add_image(sub)
        # viewer.add_image(grd)

