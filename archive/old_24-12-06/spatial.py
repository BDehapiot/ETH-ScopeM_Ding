#%% Imports -------------------------------------------------------------------

import math
import time
import heapq
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, Memory 

# Scipy
from scipy.signal import correlate

#%% Comments ------------------------------------------------------------------

'''
- The idea is to compute all ddts for an object (all pixel agains all pixel) 
just one time and then do analysis over time in parallel. Indeed, the ddts will
be the same for all time points.
- Another idea would be to compute things on a crop version of arrays, indeed 
just considering the object. This should maybe improve performances even for 
the temporal analysis.
'''

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")

# Parameters
rf = 0.05
interval = 4
    
#%% Function(s) ---------------------------------------------------------------

def get_smsk(msk, interval):
    ys, xs = [], []
    idxs = np.where(msk)
    for y, x in zip(idxs[0], idxs[1]):
        if y % interval == 0 and x % interval == 0:
            ys.append(y)
            xs.append(x)
    sidxs = (ys, xs)
    smsk = np.zeros_like(msk)
    smsk[sidxs] = msk[sidxs]
    return smsk

# # Dijkstra distance transform (ddt)  
def get_ddts(msk, smsk):

    # Nested function(s) ------------------------------------------------------
    def get_ddt(msk, coord):

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
        ddt = np.full_like(msk, np.inf, dtype=float)
        ddt[cr, cc] = 0.0
        
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
                    if new_dist < ddt[nr, nc]:
                        ddt[nr, nc] = new_dist
                        heapq.heappush(heap, (new_dist, (nr, nc)))

        # Replace infinite distances
        ddt[np.isinf(ddt)] = np.nan

        return ddt

    # Execute -----------------------------------------------------------------
    ddts = []
    for lab in np.unique(msk)[1:]:
        lab_msk = msk == lab
        lab_smsk = smsk == lab
        idxs = np.where(lab_smsk)
        outputs = Parallel(n_jobs=-1)(
            delayed(get_ddt)(lab_msk, (y, x))
            for y, x in zip(idxs[0], idxs[1])
            )
        ddts.append(np.stack(outputs))

    return ddts    

#%% Execute -------------------------------------------------------------------

for path in data_path.glob(f"*rf-{rf}_stk.tif*"):    
        
    if path.name == f"Exp2_rf-{rf}_stk.tif":
    
        # Open data
        msk = io.imread(str(path).replace("stk", "msk"))
        sub = io.imread(str(path).replace("stk", "sub"))
        grd = io.imread(str(path).replace("stk", "grd"))    
    
        t0 = time.time()   
        print(path.name)
                 
        smsk = get_smsk(msk, interval)
        ddts = get_ddts(msk, smsk)
                
        t1= time.time()
        print(f"get_ddts() : {t1 - t0:.5f}")
        
        break
    
# import napari
# viewer = napari.Viewer()
# viewer.add_image(ddts[1])

#%%

t = 0
lab = 5
arr = grd

# -----------------------------------------------------------------------------

def get_acc(y):
    acc = np.correlate(y, y, mode="full")
    acc = acc[acc.size // 2:]
    acc /= acc[0] # Zero lag normalization        
    return acc

def get_iip(vals, min_int=0.02):
    iip, acc = [], []
    vals = [vals[sort] for sort in sorts]
    for val, dist, xval in zip(vals, dists, xvals):
        if val[0] > min_int:
            tmp_iip = np.interp(xval, dist, val)
            tmp_acc = get_acc(tmp_iip)
        else:
            tmp_iip = np.full(xval[-1] + 1, np.nan)
            tmp_acc = np.full((xval[-1] + 1) , np.nan)
        iip.append(tmp_iip)
        acc.append(tmp_acc)
    return iip, acc

# -----------------------------------------------------------------------------

# For all labels
t0 = time.time()  

idxs = np.where(smsk == lab)
dists = [ddt[idxs] for ddt in ddts[lab - 1]]
sorts = [np.argsort(dist) for dist in dists]  
dists = [dist[sort] for (dist, sort) in zip(dists, sorts)]
# xvals = [np.arange(0, int(np.max(dist)), 1) for dist in dists]
xvals = [np.arange(0, 80, 1) for dist in dists]

t1= time.time()
print(f"runtime #1: {t1 - t0:.5f}") 

# For all timepoints
t0 = time.time()  

# iip = []
# vals = arr[t, ...][idxs]
# vals = [vals[sort] for sort in sorts]
# for val, dist, xval in zip(vals, dists, xvals):
#     iip.append(np.interp(xval, dist, val))
    
iips, accs = [], []   
for t in range(arr.shape[0]):
    iip, acc = get_iip(arr[t, ...][idxs])
    iips.append(iip)
    accs.append(acc)

# outputs = Parallel(n_jobs=-1)(
#     delayed(get_iip)(arr[t, ...][idxs])
#     for t in range(arr.shape[0])
#     )

t1= time.time()
print(f"runtime #2: {t1 - t0:.5f}") 

#%%

iips_avg, accs_avg = [], []
for iip, acc in zip(iips, accs):
    iips_avg.append(np.nanmean(np.stack(iip), axis=0))
    accs_avg.append(np.nanmean(np.stack(acc), axis=0))
iips_avg = np.stack(iips_avg)
accs_avg = np.stack(accs_avg)
accs_avg_avg = np.nanmean(accs_avg, axis=0)

plt.plot(accs_avg_avg)

# plt.figure(figsize=(6, 6))
# plt.imshow(accs_avg, cmap='viridis', interpolation='nearest', aspect='auto')