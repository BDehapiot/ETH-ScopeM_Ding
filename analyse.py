#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

# Scipy
from scipy.signal import correlate
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve

#%% Comments ------------------------------------------------------------------

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")

# Parameters
rf = 0.1

#%% Function(s): --------------------------------------------------------------

def analyse(stack, mask, filt):
    
    global data, stack_bsub

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
        accr = correlate(bsub, bsub, mode="full")
        accr = accr[accr.size // 2:]
        accr /= accr[0] # Zero lag normalization
        return bsub, accr     
    
    def _analyse2(val):
        pass

    # Execute -----------------------------------------------------------------  
    
    data = {"label" : [], "vals" : [], "bsub" : [], "accr" : []}
        
    # Analyse
    stack_bsub = np.zeros_like(filt)
    for lab in np.unique(mask)[1:]:
        idx = np.where(mask == lab)
        vals = filt[:, idx[0], idx[1]]
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
        
    # 
    for t in range(stack_bsub.shape[0]):
        stack_bsub[t, ...][mask == 0] = np.nan

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in data_path.glob(f"*rf-{rf}_stack.tif*"):    
            
        if path.name == "Exp2_rf-0.1_stack.tif":
        
            t0 = time.time()    
            
            print(path.name)
        
            stack = io.imread(path)
            mask = io.imread(str(path).replace("stack", "mask"))
            filt = io.imread(str(path).replace("stack", "filt"))   
            analyse(stack, mask, filt)
            
            t1= time.time()
            print(f"runtime : {t1 - t0:.3f}")
            
            stack_grd = np.gradient(stack_bsub, axis=0)
            stack_grd = np.roll(stack_grd, 1, axis=0)
            
            # Display
            import napari
            viewer = napari.Viewer()
            viewer.add_image(stack)
            viewer.add_image(stack_bsub)
            viewer.add_image(stack_grd, contrast_limits=[-0.2, 0.2])
              
#%%

# idx = 3
# vals_avg = np.vstack([np.mean(dat, axis=1) for dat in data["vals"]]).T
# bsub_avg = np.vstack([np.mean(dat, axis=1) for dat in data["bsub"]]).T
# accr_avg = np.vstack([np.mean(dat, axis=1) for dat in data["accr"]]).T

# # Plot
# plt.figure(figsize=(10, 12))

# # vals_avg
# plt.subplot(3, 1, 1)
# plt.plot(vals_avg[:, idx], label="Raw Values (vals_avg)")
# plt.plot(vals_avg[:, idx] - bsub_avg[:, idx], label="vals_avg - bsub_avg", linestyle='--')
# plt.title("Raw Values Data")
# plt.legend()
# plt.xlabel("Time/Index")
# plt.ylabel("Amplitude")

# # bsub_avg
# plt.subplot(3, 1, 2)
# plt.plot(bsub_avg[:, idx], label="Baseline Subtracted (bsub_avg)")
# plt.title("Baseline Subtracted Data")
# plt.legend()
# plt.xlabel("Time/Index")
# plt.ylabel("Amplitude")

# # accr_avg
# plt.subplot(3, 1, 3)
# plt.plot(accr_avg[:, idx], label="Autocorrelation (accr_avg)")
# plt.title("Autocorrelation")
# plt.legend()
# plt.xlabel("Lag")
# plt.ylabel("Correlation")

# plt.tight_layout()
# plt.show()
           
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
       