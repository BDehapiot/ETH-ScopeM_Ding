#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# Skimage
from skimage.measure import label

#%% Comments ------------------------------------------------------------------

'''
# Experiment timepoints
1-300s     - 2mM Glucose,
301-900s   - 20mM Glucose,
901-1800s  - 20mM Glucose,
1801-2700s - 20mM Glucose + 100uM Mefloquine,
'''

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")

# Parameters
rf = 0.05
tps = [0, 300, 900, 1800, 2700] # Experiment timepoints

#%% Functions -----------------------------------------------------------------

def match_length(vals):
    matched_vals = vals.copy()
    max_dist = np.max([len(val) for val in vals])
    for i, val in enumerate(vals):
        tmp_nan = np.full((max_dist - val.shape[0]), np.nan)
        matched_vals[i] = np.hstack((val, tmp_nan))
    return matched_vals

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in data_path.glob(f"*rf-{rf}_stk.tif*"):  
        
        if path.name == f"Exp1_rf-{rf}_stk.tif":
                    
            msk = io.imread(str(path).replace("stk", "msk"))
            sub = io.imread(str(path).replace("stk", "sub")) 
            grd = io.imread(str(path).replace("stk", "grd"))

            t0 = time.time()    
            print(path.name)

            t1= time.time()
            print(f"runtime : {t1 - t0:.3f}")
            
#%%

arr = grd
thresh = 0.1

arr_msk = arr > thresh
arr_lbl = label(arr_msk)

nT = arr.shape[0]
nL = np.max(arr_lbl)
pulse_area = np.full((nT, nL), np.nan)
pulse_ints = np.full((nT, nL), np.nan)
for t in range(nT):
    vals = arr[t, ...].ravel()
    lvals = arr_lbl[t, ...].ravel()
    for lbl in np.unique(lvals)[1:]:
        valid = lvals == lbl
        pulse_area[t, lbl - 1] = np.sum(valid)
        pulse_ints[t, lbl - 1] = np.mean(vals[valid])

pulse_tmax = [] 
pulse_tmax_area = [] 
pulse_tmax_ints = [] 
pulse_area_reg = []
pulse_ints_reg = []  
for lbl in range(nL):
    area = pulse_area[:, lbl]
    ints = pulse_ints[:, lbl]
    valid = ~np.isnan(area)
    pulse_tmax.append(np.nanargmax(area))
    pulse_tmax_area.append(area[np.nanargmax(area)])
    pulse_tmax_ints.append(ints[np.nanargmax(area)])
    pulse_area_reg.append(area[valid])
    pulse_ints_reg.append(ints[valid])
pulse_tmax = np.stack(pulse_tmax)
pulse_tmax_area = np.stack(pulse_tmax_area)
pulse_tmax_ints = np.stack(pulse_tmax_ints)
pulse_area_reg = np.vstack(match_length(pulse_area_reg)).T
pulse_ints_reg = np.vstack(match_length(pulse_ints_reg)).T

pulse_tmax_sumCat = []
pulse_tmax_area_sumCat = []
pulse_tmax_ints_sumCat = []
pulse_area_reg_avgCat = []
pulse_ints_reg_avgCat = []
for tp in range(1, len(tps)):
    valid = (pulse_tmax > tps[tp - 1]) & (pulse_tmax <= tps[tp])
    pulse_tmax_sumCat.append(np.sum(valid))
    pulse_tmax_area_sumCat.append(pulse_tmax_area[valid])
    pulse_tmax_ints_sumCat.append(pulse_tmax_ints[valid])
    pulse_area_reg_avgCat.append(np.nanmean(pulse_area_reg[:, valid], axis = 1))
    pulse_ints_reg_avgCat.append(np.nanmean(pulse_ints_reg[:, valid], axis = 1))

# Plot
plt.figure(figsize=(8, 12))
cmap = plt.get_cmap("turbo", len(tps))
plt.subplot(4, 1, 1)
plt.plot(np.nanmean(pulse_area, axis=1))
plt.subplot(4, 1, 2)
for tp in range(1, len(tps)):
    plt.plot(
        pulse_area_reg_avgCat[tp - 1], 
        label=f"{tps[tp - 1]} - {tps[tp]}",
        color=cmap(tp - 1), 
        )
plt.title("Pulse Area")
plt.legend() 
plt.subplot(4, 1, 3)
for tp in range(1, len(tps)):
    plt.plot(
        pulse_ints_reg_avgCat[tp - 1], 
        label=f"{tps[tp - 1]} - {tps[tp]}",
        color=cmap(tp - 1), 
        )
plt.title("Pulse Intensity")
plt.legend() 
plt.subplot(4, 2, 4)
for tp in range(1, len(tps)):
    plt.boxplot(
        pulse_tmax_area_sumCat[tp - 1], 
        positions=[tp],
        widths=0.6
        )
plt.subplot(4, 3, 4)
for tp in range(1, len(tps)):
    plt.boxplot(
        pulse_tmax_ints_sumCat[tp - 1], 
        positions=[tp],
        widths=0.6
        )


