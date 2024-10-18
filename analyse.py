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
        bsub = val - als(val)
        accr = correlate(bsub, bsub, mode='full')
        accr = accr[accr.size // 2:]
        accr /= accr[0] # Zero lag normalization
        return bsub, accr      

    # Execute -----------------------------------------------------------------  
    
    data = {
        "label" : [],
        "vals" : [],
        "bsub" : [],
        "accr" : [],
        }
    
    # Filter stack
    stack = norm_pct(norm_gcn(stack), pct_low=0, pct_high=100)
    stack_filt = nan_filt(
        stack, mask=mask > 0, kernel_size=(1, 3, 3), iterations=3)
    
    # Analyse
    stack_bsub = np.zeros_like(stack_filt)
    for lab in np.unique(mask)[1:]:
        idx = np.where(mask == lab)
        vals = stack[:, idx[0], idx[1]]
        outputs = Parallel(n_jobs=-1)(
            delayed(_analyse)(vals[:, i])
            for i in range(vals.shape[1])
            )
        bsub = np.vstack([data[0] for data in outputs]).T
        accr = np.vstack([data[1] for data in outputs]).T
        stack_bsub[:, idx[0], idx[1]] = bsub

        # Append data
        data["label"].append(lab)
        data["vals"].append(vals)
        data["bsub"].append(bsub)
        data["accr"].append(accr)
    
    return stack_filt, stack_bsub

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    rf = 0.1
    
    for path in data_path.glob(f"*rf-{rf}_stack.tif*"):    
            
        stack = io.imread(path)
        mask = io.imread(str(path).replace("stack", "mask"))
        stack_filt, stack_bsub = analyse(stack, mask)       
        
        # Save
        io.imsave(
            str(path).replace("stack", "stack_filt"),
            stack_filt.astype("float32"), check_contrast=False,
            )
        io.imsave(
            str(path).replace("stack", "stack_bsub"),
            stack_bsub.astype("float32"), check_contrast=False,
            )
     
            # # Display
            # import napari
            # viewer = napari.Viewer()
            # viewer.add_image(stack)
            # viewer.add_image(stack_filt)
            # viewer.add_image(stack_filt_bsub)
            
#%%

idx = 5
vals_avg = np.vstack([np.mean(dat, axis=1) for dat in data["vals"]]).T
bsub_avg = np.vstack([np.mean(dat, axis=1) for dat in data["bsub"]]).T
accr_avg = np.vstack([np.mean(dat, axis=1) for dat in data["accr"]]).T

# Plot
plt.figure(figsize=(10, 12))

# vals_avg
plt.subplot(3, 1, 1)
plt.plot(vals_avg[:, idx], label="Raw Values (vals_avg)")
plt.plot(vals_avg[:, idx] - bsub_avg[:, idx], label="vals_avg - bsub_avg", linestyle='--')
plt.title("Raw Values Data")
plt.legend()
plt.xlabel("Time/Index")
plt.ylabel("Amplitude")

# bsub_avg
plt.subplot(3, 1, 2)
plt.plot(bsub_avg[:, idx], label="Baseline Subtracted (bsub_avg)")
plt.title("Baseline Subtracted Data")
plt.legend()
plt.xlabel("Time/Index")
plt.ylabel("Amplitude")

# accr_avg
plt.subplot(3, 1, 3)
plt.plot(accr_avg[:, idx], label="Autocorrelation (accr_avg)")
plt.title("Autocorrelation")
plt.legend()
plt.xlabel("Lag")
plt.ylabel("Correlation")

plt.tight_layout()
plt.show()
           
#%%

# def als(y, lam=1e7, p=0.001, niter=5):
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
