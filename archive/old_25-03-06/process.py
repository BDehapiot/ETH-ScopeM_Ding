#%% Imports -------------------------------------------------------------------

import h5py
import time
import numpy as np
from skimage import io
from pathlib import Path
from joblib import Parallel, delayed 

# bdtools
from bdtools.models import UNet
from bdtools.nan import nan_filt

# Skimage
from skimage.measure import label
from skimage.morphology import remove_small_objects

# Scipy
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import binary_fill_holes, uniform_filter1d

#%% Comments ------------------------------------------------------------------

#%% Inputs --------------------------------------------------------------------

# Paths
# data_path = Path("D:\local_Ding\data")
data_path = Path.cwd() / "_local"
model_path = Path.cwd() / "model" / "model_normal"

# Parameters
rf = 0.1
window_size = 501
min_size = 2560 * rf # 256 for rf = 0.1

#%% Function(s): --------------------------------------------------------------

def process(stk, msk, window_size=501, min_size=256):
    
    # Nested function(s) --------------------------------------------------------
    
    def rolling_avg(stk, window_size):
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")
        pad = window_size // 2
        stk = np.pad(
            stk, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
        stk = uniform_filter1d(stk, size=window_size, axis=0)
        return stk[pad:-pad]
    
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

    # # Rolling average
    # rstk = rolling_avg(stk, window_size)

    # # Predict
    # unet = UNet(load_name="model_128_normal_2000-160_1")
    # prb = unet.predict(rstk[::25], verbose=1)
        
    # Get mask
    msk = msk > 0
    msk = binary_fill_holes(msk)
    msk = remove_small_objects(msk, min_size=min_size)
    msk = label(msk)
    
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

    for path in data_path.glob(f"*rf-{rf}_stk*"):
                
        t0 = time.time()
        
        print(path.name)
        
        stk = io.imread(path)
        msk = io.imread(str(path).replace("stk", "msk"))
        filt, sub, grd = process(
            stk, msk, window_size=window_size, min_size=min_size)
        
        t1 = time.time()
        print(f"runtime : {t1 - t0:.3f}s")
        
        # Save
        # io.imsave(
        #     str(path).replace("stk", "prb"),
        #     prb.astype("float32"), check_contrast=False,
        #     )
        # io.imsave(
        #     str(path).replace("stk", "msk"),
        #     msk.astype("uint8"), check_contrast=False,
        #     )
        io.imsave(
            str(path).replace("stk", "flt"),
            filt.astype("float32"), check_contrast=False,
            )
        io.imsave(
            str(path).replace("stk", "sub"),
            sub.astype("float32"), check_contrast=False,
            )
        io.imsave(
            str(path).replace("stk", "grd"),
            grd.astype("float32"), check_contrast=False,
            )
