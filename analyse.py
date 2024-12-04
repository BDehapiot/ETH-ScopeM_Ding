#%% Imports -------------------------------------------------------------------

import time
import heapq
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

#%% Comments ------------------------------------------------------------------

'''
1-300s - 2mM Glucose,
301-900 - 20mM Glucose,
901-1800 - 20mM Glucose
1801-2700 - 20mM Glucose + 100uM Mefloquine
'''

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")

# Parameters
rf = 0.05
interval = 4

#%% Function(s): --------------------------------------------------------------

# Dijkstra distance transform (ddts)  
def analyse(msk, sub, grd, interval=4):
    
    global data
    
    # Nested functions --------------------------------------------------------
      
    # Reduce mask
    def reduce_msk(msk, interval):
        ys, xs = [], []
        idxs = np.where(msk)
        for y, x in zip(idxs[0], idxs[1]):
            if y % interval == 0 and x % interval == 0:
                ys.append(y)
                xs.append(x)
        ridxs = (ys, xs)
        rmsk = np.zeros_like(msk)
        rmsk[ridxs] = msk[ridxs]
        return rmsk
    
    # Auto cross correlation (acc) 
    def get_acc(y, mode="same"):
        acc = np.correlate(y, y, mode=mode)
        acc = acc[acc.size // 2:]
        acc /= acc[0] # zero lag normalization        
        return acc
    
    # Dijkstra distance transforms (ddts)  
    def get_ddts(msk, coord):

        # Define offsets
        s2 = 1.4142135623730951
        neighbor_offsets = [
            (-1, -1, s2), (-1, 0, 1), (-1, 1, s2),
            (0, -1, 1  ),             (0, 1, 1  ),
            (1, -1, s2 ), (1, 0, 1 ), ( 1, 1, s2),
            ]

        # Initialize
        cr, cc = coord
        nY, nX = msk.shape
        visited = np.zeros_like(msk, dtype=bool)
        ddts = np.full_like(msk, np.inf, dtype=float)
        ddts[cr, cc] = 0.0
        
        # Process
        heap = []
        heapq.heappush(heap, (0.0, (cr, cc)))
        while heap:
            current_dist, (r, c) = heapq.heappop(heap)
            if visited[r, c]:
                continue
            visited[r, c] = True
            for dr, dc, cost in neighbor_offsets:
                nr, nc = r + dr, c + dc
                if (0 <= nr < nY and 0 <= nc < nX and
                        not visited[nr, nc] and msk[nr, nc] == 1):
                    new_dist = current_dist + cost
                    if new_dist < ddts[nr, nc]:
                        ddts[nr, nc] = new_dist
                        heapq.heappush(heap, (new_dist, (nr, nc)))

        # Replace infinite distances
        ddts[np.isinf(ddts)] = np.nan

        return ddts
    
    # Interpolated intensity profiles (iips)
    def process_svals(vals, sorts):
        svals, svals_acc = [], []
        vals = [vals[sort] for sort in sorts]
        for val, dist, xval in zip(vals, dists, xvals):
            sval = np.interp(xval, dist, val)
            sval_acc = get_acc(sval)
            svals.append(sval)
            svals_acc.append(sval_acc)
        return svals, svals_acc
    
    # Execute -----------------------------------------------------------------  
       
    data = []
    rmsk = reduce_msk(msk, interval)    
    
    for lab in np.unique(rmsk)[1:]:
        
        ridxs = np.where(rmsk == lab)
        
        # Temporal analysis
        sub_tvals = list(sub[:, ridxs[0], ridxs[1]].T)
        grd_tvals = list(grd[:, ridxs[0], ridxs[1]].T)
        sub_tvals_acc = [get_acc(tval) for tval in sub_tvals] 
        grd_tvals_acc = [get_acc(tval) for tval in grd_tvals]  
        
        # Spatial analysis
        ddts = Parallel(n_jobs=-1)(
            delayed(get_ddts)(msk == lab, (y, x))
            for y, x in zip(ridxs[0], ridxs[1])
            )
        dists = [ddt[ridxs] for ddt in ddts]
        sorts = [np.argsort(dist) for dist in dists]  
        dists = [dist[sort] for (dist, sort) in zip(dists, sorts)]
        xvals = [np.arange(0, int(np.max(dist)), 1) for dist in dists]
        
        sub_svals, grd_svals = [], []  
        sub_svals_acc, grd_svals_acc = [], [] 
        for t in range(sub.shape[0]):
            svals, svals_acc = process_svals(sub[t, ...][ridxs], sorts)
            sub_svals.append(svals)
            sub_svals_acc.append(svals_acc)
            svals, svals_acc = process_svals(grd[t, ...][ridxs], sorts)
            grd_svals.append(svals)
            grd_svals_acc.append(svals_acc)
        
        # Append data
        data.append({
            
            "label" : lab, 
            "ridxs" : ridxs,
            "nT"    : sub.shape[0], 
            "nP"    : len(ridxs[0]),
            
            # Temporal
            "sub_tvals"     : sub_tvals,
            "grd_tvals"     : grd_tvals,
            "sub_tvals_acc" : sub_tvals_acc,
            "grd_tvals_acc" : grd_tvals_acc,
            
            # Spatial
            "ddts"          : ddts,
            "dists"         : dists,
            "sorts"         : sorts,
            "sub_svals"     : sub_svals,
            "grd_svals"     : grd_svals,
            "sub_svals_acc" : sub_svals_acc,
            "grd_svals_acc" : grd_svals_acc,
            
            })
        
#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in data_path.glob(f"*rf-{rf}_stk.tif*"):  
        
        if path.name == f"Exp2_rf-{rf}_stk.tif":
                    
            msk = io.imread(str(path).replace("stk", "msk"))
            sub = io.imread(str(path).replace("stk", "sub")) 
            grd = io.imread(str(path).replace("stk", "grd"))

            t0 = time.time()    
            print(path.name)
            
            analyse(msk, sub, grd, interval=interval)
            
            t1= time.time()
            print(f"runtime : {t1 - t0:.3f}")
            
            # # Display
            # import napari 
            # viewer = napari.Viewer()
            # viewer.add_labels(rmsk)
            
#%%

lab = 4
t0, t1 = 300, 1800
sub_tresh = 1.0
grd_tresh = 0.1

def match_length(vals):
    matched_vals = vals.copy()
    max_dist = np.max([len(val) for val in vals])
    for i, val in enumerate(vals):
        tmp_nan = np.full((max_dist - val.shape[0]), np.nan)
        matched_vals[i] = np.hstack((val, tmp_nan))
    return matched_vals

sub_svals_acc_pavg = []
grd_svals_acc_pavg = []
for t in range(data[lab]["nT"]):
    
    sub_valid = sub[t, ...][data[lab]["ridxs"]] > sub_tresh
    grd_valid = grd[t, ...][data[lab]["ridxs"]] > grd_tresh

    sub_svals_acc_pavg.append(np.nanmean(
        np.stack(match_length(data[lab]["sub_svals_acc"][t]))[sub_valid], axis=0))
    grd_svals_acc_pavg.append(np.nanmean(
        np.stack(match_length(data[lab]["grd_svals_acc"][t]))[grd_valid], axis=0))
    
# -----------------------------------------------------------------------------    
    
sub_svals_acc_pavg_t0avg = np.nanmean(
    np.stack(sub_svals_acc_pavg[t0:t1]), axis=0)
grd_svals_acc_pavg_t0avg = np.nanmean(
    np.stack(grd_svals_acc_pavg[t0:t1]), axis=0)
sub_svals_acc_pavg_t1avg = np.nanmean(
    np.stack(sub_svals_acc_pavg[t1:-1]), axis=0)
grd_svals_acc_pavg_t1avg = np.nanmean(
    np.stack(grd_svals_acc_pavg[t1:-1]), axis=0)

# -----------------------------------------------------------------------------

plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(sub_svals_acc_pavg_t0avg)
plt.plot(sub_svals_acc_pavg_t1avg)

plt.subplot(2, 1, 2)
plt.plot(grd_svals_acc_pavg_t0avg)
plt.plot(grd_svals_acc_pavg_t1avg)

#%%

# sub_svals_acc_tavg = []
# grd_svals_acc_tavg = []
# for p in range(data[lab]["nP"]):
#     sub_svals_acc_tavg.append(np.nanmean(
#         np.stack([data[lab]["sub_svals_acc"][t][p] 
#                   for t in range(data[lab]["nT"])]), axis=0))
#     grd_svals_acc_tavg.append(np.nanmean(
#         np.stack([data[lab]["grd_svals_acc"][t][p] 
#                   for t in range(data[lab]["nT"])]), axis=0))
# sub_svals_acc_tavg = match_length(sub_svals_acc_tavg)
# grd_svals_acc_tavg = match_length(grd_svals_acc_tavg)

# sub_svals_acc_tavg_pavg = np.nanmean(np.stack(sub_svals_acc_tavg), axis=0)
# grd_svals_acc_tavg_pavg = np.nanmean(np.stack(grd_svals_acc_tavg), axis=0)