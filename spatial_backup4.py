#%% Imports -------------------------------------------------------------------

import math
import time
import heapq
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

# Scipy
from scipy.signal import correlate
from scipy.interpolate import interp1d

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
    
#%% Function(s) ---------------------------------------------------------------

# Dijkstra distance transform (ddt)  
def get_ddts(msk):
    
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
        idxs = np.where(lab_msk)            
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
                 
        ddts = get_ddts(msk)
                
        t1= time.time()
        print(f"get_ddts() : {t1 - t0:.5f}")
        
        break

#%%

t = 0
lab = 2
arr = sub

# -----------------------------------------------------------------------------

def get_smsk(msk, interval):
    ys, xs = [], []
    idxs = np.where(msk)
    for y, x in zip(idxs[0], idxs[1]):
        if y % 4 == 0 and x % 4 == 0:
            ys.append(y)
            xs.append(x)
    sidxs = (ys, xs)
    smsk = np.zeros_like(msk)
    smsk[sidxs] = msk[sidxs]
    return smsk

# -----------------------------------------------------------------------------

# For all labels
t0 = time.time()  

smsk = get_smsk(msk, 4)
idxs = np.where(smsk == lab)
dists = [ddt[idxs] for ddt in ddts[lab - 1]]
sorts = [np.argsort(dist) for dist in dists]  
dists = [dist[sort] for (dist, sort) in zip(dists, sorts)]
xvals = [np.arange(np.min(dist), np.max(dist) + 1, 1) for dist in dists]

t1= time.time()
print(f"runtime #1: {t1 - t0:.5f}") 

# For all timepoints
t0 = time.time()  

iips = []
vals = [arr[t, ...][idxs][sort] for sort in sorts]
for dist, val, xval in zip(dists, vals, xvals):
    f = interp1d(dist, val, fill_value="extrapolate", assume_sorted=True)
    iips.append(f(xval))
    
# outputs = Parallel(n_jobs=-1)(
#     delayed(get_iip)(sub[t, ...], idxs, dists, sorts)
#     for t in range(sub.shape[0])
#     )

t1= time.time()
print(f"runtime #2: {t1 - t0:.5f}") 

#%%

# from numba import njit, prange

# # -----------------------------------------------------------------------------

# t = 0
# lab = 2
# arr = sub

# # -----------------------------------------------------------------------------

# # def get_iip(arr, idxs, dists, sorts):
# #     iips = []
# #     vals = [arr[idxs][sort] for sort in sorts]
# #     for dist, val in zip(dists, vals):
# #         x = np.arange(np.min(dist), np.max(dist) + 1, 1)
# #         f = interp1d(dist, val, fill_value="extrapolate", assume_sorted=True)
# #         iips.append(f(x))
# #     return iips
    
# # -----------------------------------------------------------------------------

# t0 = time.time()  

# # For all labels
# idxs = np.where(msk == lab)
# dists = [ddt[idxs] for ddt in ddts[lab - 1]]
# sorts = [np.argsort(dist) for dist in dists]  
# dists = [dist[sort] for (dist, sort) in zip(dists, sorts)]
# xvals = [np.arange(np.min(dist), np.max(dist) + 1, 1) for dist in dists]

# t1= time.time()
# print(f"runtime #1: {t1 - t0:.5f}") 

# t0 = time.time()  

# # For all timepoints
# iips = []
# vals = [arr[t, ...][idxs][sort] for sort in sorts]
# for dist, val, xval in zip(dists, vals, xvals):
#     f = interp1d(dist, val, fill_value="extrapolate", assume_sorted=True)
#     iips.append(f(xval))
    
# # outputs = Parallel(n_jobs=-1)(
# #     delayed(get_iip)(sub[t, ...], idxs, dists, sorts)
# #     for t in range(sub.shape[0])
# #     )

# t1= time.time()
# print(f"runtime #2: {t1 - t0:.5f}") 