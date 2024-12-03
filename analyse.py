#%% Imports -------------------------------------------------------------------

import time
import heapq
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
    
    global data, rmsk, sub_tvals, grd_tvals, sub_tvals_acc, grd_tvals_acc
    
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
    def get_acc(y):
        acc = np.correlate(y, y, mode="same")
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
    def interpolate_svals(svals):
        iips = []
        svals = [svals[sort] for sort in sorts]
        for sval, dist, xval in zip(svals, dists, xvals):
            iips.append(np.interp(xval, dist, sval))
        return iips
    
    # Execute -----------------------------------------------------------------  
       
    data = []
    nT = sub.shape[0]
    rmsk = reduce_msk(msk, interval)    
    
    for lab in np.unique(rmsk)[1:]:
        
        ridxs = np.where(rmsk == lab)
        
        # Temporal analysis
        sub_tvals = list(sub[:, ridxs[0], ridxs[1]].T)
        grd_tvals = list(grd[:, ridxs[0], ridxs[1]].T)
        sub_tvals_acc = [get_acc(tval) for tval in sub_tvals] 
        grd_tvals_acc = [get_acc(tval) for tval in grd_tvals]  
        sub_tvals_acc_avg = np.mean(sub_tvals_acc, axis=0)
        grd_tvals_acc_avg = np.mean(grd_tvals_acc, axis=0)
        
        # Spatial analysis
        ddts = Parallel(n_jobs=-1)(
            delayed(get_ddts)(msk == lab, (y, x))
            for y, x in zip(ridxs[0], ridxs[1])
            )
        dists = [ddt[ridxs] for ddt in ddts]
        sorts = [np.argsort(dist) for dist in dists]  
        dists = [dist[sort] for (dist, sort) in zip(dists, sorts)]
        xvals = [np.arange(0, int(np.max(dist)), 1) for dist in dists]
        
        iips = []   
        for t in range(arr.shape[0]):
            iip, acc = get_iip(arr[t, ...][idxs])
            iips.append(iip)
            accs.append(acc)
        
        # Append data
        data.append({
            "label" : lab, "nT" : nT, "nP" : len(ridxs[0]),
            
            # Temporal
            "sub_tvals"         : sub_tvals,
            "grd_tvals"         : grd_tvals,
            "sub_tvals_acc"     : sub_tvals_acc,
            "grd_tvals_acc"     : grd_tvals_acc,
            "sub_tvals_acc_avg" : sub_tvals_acc_avg,
            "grd_tvals_acc_avg" : grd_tvals_acc_avg,
            
            # Spatial
            "ddts" : ddts,
            
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

# lab = 2
# p = np.random.randint(0, data[lab]["nP"])
# plt.plot(data[lab]["sub_tvals"][p])
# plt.plot(data[lab]["grd_tvals"][p])
