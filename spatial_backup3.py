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

def analyse(sub, grd, msk):
    
    # Nested function(s) ------------------------------------------------------
    
    # Dijkstra distance transform (ddt)   
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
    
    def get_ddt_map(msk):
        ddt = [] 
        idxs = np.where(msk)
        print(idxs[0].shape)
        for coord in zip(idxs[0], idxs[1]):
            ddt.append(get_ddt(msk, coord))
        return ddt
    
    # Auto cross correlation (acc) 
    def get_acc(y):
        acc = correlate(y, y, mode="full")
        acc = acc[acc.size // 2:]
        acc /= acc[0] # Zero lag normalization
        return acc
    
    def analyse_s(sub, grd, msk):
        
        idxs = np.where(msk)
        coord = (idxs[0][0], idxs[1][0])
        ddt = get_ddt(msk, coord)
        
        for coord in zip(idxs[0], idxs[1]):
            if coord[0] % 5 == 0 and coord[1] % 5 == 0:
                
                # Get Dijkstra distance transform (ddt) 
                ddt = get_ddt(msk, coord)
            
                # Sort distances
                dist = ddt[idxs]
                sort = np.argsort(dist)
                dist = dist[sort]

    # Execute -----------------------------------------------------------------

    # 
    tmp_msk = msk == 3
    ddt = get_ddt_map(tmp_msk)

    return ddt
    
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

if __name__ == "__main__":
    
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
            print(f"runtime : {t1 - t0:.5f}")

#%%

from numba import njit, prange

@njit(parallel=True)
def apply_sorts(vals, sorts):
    n_rows, n_cols = vals.shape
    sorted_vals = np.empty_like(vals)  # Preallocate memory for results
    for i in prange(n_rows):  # Parallel over rows
        sorted_vals[i] = vals[i, sorts[i]]  # Sort each row based on sorts[i]
    return sorted_vals

# Interpolated intensity profiles (iip) 
def get_iips(msk, arr, ddts):
    
    global dists, sorts, vals
    
    # Nested function(s) ------------------------------------------------------
        
    # Execute -----------------------------------------------------------------
           
    lab = 2
    idxs = np.where(msk == lab)    
    dists = [ddt[idxs] for ddt in ddts[lab - 1]]
    sorts = [np.argsort(dist) for dist in dists]   
    dists = [dist[sort] for (dist, sort) in zip(dists, sorts)]
    vals = np.stack([ar[idxs] for ar in arr])
    vals = apply_sorts(vals, sorts)
       
    for dist, val in zip(dists, vals):
        interpx = np.arange(np.min(dist), np.max(dist) + 1, 1)
        interpf = interp1d(dist, val, fill_value="extrapolate", assume_sorted=True)
        iip.append(interpf(interpx))
        
    # interpx = np.arange(np.min(dist), np.max(dist) + 1, 1)
    # vals = [val[sort] for val in vals]
    
    # iip = []
    # for val in vals:
    #     interpf = interp1d(dist, val, fill_value="extrapolate", assume_sorted=True)
    #     iip.append(interpf(interpx))
    
# -----------------------------------------------------------------------------
    
t0 = time.time()    

iips = get_iips(msk, sub, ddts)

t1= time.time()
print(f"runtime : {t1 - t0:.5f}")