#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

# bdtools
from bdtools.nan import nan_filt
from bdtools.norm import norm_gcn, norm_pct

# Scipy
from scipy.signal import correlate
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Ding\data")

#%% Function(s): --------------------------------------------------------------

def analyse(stack, mask):
        
    global data, vals, vals_bsub, vals_ccr
    
    # Nested functions --------------------------------------------------------
    
    def als(y, lam=1e7, p=0.001, niter=5):
      L = len(y)
      D = diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
      w = np.ones(L)
      for i in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
      return z
  
    def _analyse(val):
        val_bsub = val - als(val)
        val_ccr = correlate(val_bsub, val_bsub, mode='full')
        val_ccr = val_ccr[val_ccr.size // 2:]
        return val_bsub, val_ccr      

    # result = correlate(signal, signal, mode='full')
    # autocorr = result[result.size // 2:]

    # Execute -----------------------------------------------------------------  
    
    data = {
        "label" : [],
        "vals" : [],
        "vals_bsub" : [],
        "vals_ccr" : [],
        }
    
    # Filter stack
    stack = norm_gcn(norm_pct(stack))
    stack_filt = nan_filt(
        stack, mask=mask > 0, kernel_size=(1, 3, 3), iterations=3)
    
    #
    stack_filt_bsub = np.zeros_like(stack_filt)
    for lab in np.unique(mask)[1:]:
        idx = np.where(mask == lab)
        vals = stack[:, idx[0], idx[1]]
        outputs = Parallel(n_jobs=-1)(
            delayed(_analyse)(vals[:, i])
            for i in range(vals.shape[1])
            )
        vals_bsub = np.vstack([data[0] for data in outputs]).T
        vals_ccr = np.vstack([data[1] for data in outputs]).T
        stack_filt_bsub[:, idx[0], idx[1]] = vals_bsub

        # Append data
        data["label"].append(lab)
        data["vals"].append(vals)
        data["vals_bsub"].append(vals_bsub)
        data["vals_ccr"].append(vals_ccr)
    
    return stack_filt, stack_filt_bsub

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    rf = 0.1
    
    for path in data_path.glob(f"*rf-{rf}_stack*"):    
        if path.name == "Exp1_rf-0.1_stack.tif":
            
            stack = io.imread(path)
            stack = norm_gcn(norm_pct(stack))
            
            
            # mask = io.imread(str(path).replace("stack", "mask"))
            # stack_filt, stack_filt_bsub = analyse(stack, mask)       
            
            # # Save
            # io.imsave(
            #     str(path).replace("stack", "stack_filt"),
            #     stack_filt.astype("float32"), check_contrast=False,
            #     )
            # io.imsave(
            #     str(path).replace("stack", "stack_filt_bsub"),
            #     stack_filt_bsub.astype("float32"), check_contrast=False,
            #     )
           
#%%

# test = data["vals_ccr"][13]
# test_avg = np.mean(test, axis=1)
# plt.figure(figsize=(10, 6))
# plt.plot(test_avg, label='test_avg')
# plt.legend()
# plt.show()

# Display
import napari
viewer = napari.Viewer()
viewer.add_image(stack)
# viewer.add_image(stack_filt)
# viewer.add_image(stack_filt_bsub)
           
#%%

# from scipy.sparse import diags, spdiags
# from scipy.sparse.linalg import spsolve

# def als(y, lam, p, niter=5):
#   L = len(y)
#   D = diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
#   w = np.ones(L)
#   for i in range(niter):
#     W = spdiags(w, 0, L, L)
#     Z = W + lam * D.dot(D.transpose())
#     z = spsolve(Z, w*y)
#     w = p * (y > z) + (1-p) * (y < z)
#   return z

# lam = 1e7 # Smoothness parameter
# p = 0.001 # Asymmetry parameter
# val = vals[:, 50]

# t0 = time.time()
# baseline = als(val, lam, p)
# t1 = time.time()
# print(f"als() : {t1 - t0:.5f}")

# plt.figure(figsize=(10, 6))
# plt.plot(val, label='Signal')
# plt.plot(baseline, label='ALS Baseline')
# plt.legend()
# plt.show()
