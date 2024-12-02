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

'''
- subtracted and gradient shoould be processed in process!
'''

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")

# Parameters
rf = 0.05

#%% Function(s): --------------------------------------------------------------

def analyse(flt, msk):
    
    # Nested functions --------------------------------------------------------
    
    # Asymetric least square (asl) 
    def get_als(y, lam=1e7, p=0.001, niter=5):
        L = len(y)
        D = diags([1, -2, 1],[0, -1, -2], shape=(L, L - 2))
        w = np.ones(L)
        for i in range(niter):
            W = spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z
  
    # Auto cross correlation (acc) 
    def get_acc(y):
        acc = correlate(y, y, mode="full")
        acc = acc[acc.size // 2:]
        acc /= acc[0] # Zero lag normalization        
        return acc
    
    def analyse_t(y):
        sub = y - get_als(y)
        grd = np.gradient(sub, axis=0)
        grd = np.roll(grd, 1, axis=0)
        acc = get_acc(sub)
        return sub, grd, acc

    # Execute -----------------------------------------------------------------  
    
    data = {
        "label"    : [], 
        "tInt"     : [], 
        "tInt_sub" : [], 
        "tInt_grd" : [], 
        "tInt_acc" : [],
        }
        
    # Temporal analysis
    sub = np.full_like(flt, np.nan)
    grd = np.full_like(flt, np.nan)
    for lab in np.unique(msk)[1:]:
        idxs = np.where(msk == lab)
        tInt = flt[:, idxs[0], idxs[1]]
        outputs = Parallel(n_jobs=-1)(
            delayed(analyse_t)(tInt[:, i])
            for i in range(tInt.shape[1])
            )
        tInt_sub = np.vstack([data[0] for data in outputs]).T
        tInt_grd = np.vstack([data[1] for data in outputs]).T
        tInt_acc = np.vstack([data[2] for data in outputs]).T
        sub[:, idxs[0], idxs[1]] = tInt_sub
        grd[:, idxs[0], idxs[1]] = tInt_grd
    
        # Append data
        data["label"].append(lab)
        data["tInt"].append(tInt)
        data["tInt_sub"].append(tInt_sub)
        data["tInt_grd"].append(tInt_grd)
        data["tInt_acc"].append(tInt_acc)
        
    # Spatial analysis
        
    return data, sub, grd

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in data_path.glob(f"*rf-{rf}_stk.tif*"):  
        
        # if path.name == f"Exp2_rf-{rf}_stk.tif":
                    
        t0 = time.time()    
        
        print(path.name)

        msk = io.imread(str(path).replace("stk", "msk"))
        flt = io.imread(str(path).replace("stk", "flt"))   
        data, sub, grd = analyse(flt, msk)
        
        t1= time.time()
        print(f"runtime : {t1 - t0:.3f}")
        
        # Save
        io.imsave(
            str(path).replace("stk", "sub"),
            sub.astype("float32"), check_contrast=False,
            )
        io.imsave(
            str(path).replace("stk", "grd"),
            grd.astype("float32"), check_contrast=False,
            )
        
        # # Display
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(stk)
        # viewer.add_image(sub)
        # viewer.add_image(
        #     grd, contrast_limits=[-0.2, 0.2], colormap="twilight")
              
#%%

# idx = 5
# val_avg = np.vstack([np.mean(dat, axis=1) for dat in data["tInt"]]).T
# sub_avg = np.vstack([np.mean(dat, axis=1) for dat in data["tInt_sub"]]).T
# acc_avg = np.vstack([np.mean(dat, axis=1) for dat in data["tInt_acc"]]).T

# # Plot
# plt.figure(figsize=(10, 12))

# # vals_avg
# plt.subplot(3, 1, 1)
# plt.plot(val_avg[:, idx], label="Raw Values (val_avg)")
# plt.plot(val_avg[:, idx] - sub_avg[:, idx], label="val_avg - sub_avg", linestyle='--')
# plt.title("Raw Values Data")
# plt.legend()
# plt.xlabel("Time/Index")
# plt.ylabel("Amplitude")

# # sub_avg
# plt.subplot(3, 1, 2)
# plt.plot(sub_avg[:, idx], label="Baseline Subtracted (sub_avg)")
# plt.title("Baseline Subtracted Data")
# plt.legend()
# plt.xlabel("Time/Index")
# plt.ylabel("Amplitude")

# # accr_avg
# plt.subplot(3, 1, 3)
# plt.plot(acc_avg[:, idx], label="Autocorrelation (acc_avg)")
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
       