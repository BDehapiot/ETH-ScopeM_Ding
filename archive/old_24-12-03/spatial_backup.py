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

    
    tmp_msk = msk == 1
    outputs = Parallel(n_jobs=-1)(
        delayed(analyse_s)(sub[t,...], grd[t,...], tmp_msk)
        for t in range(200)
        # for t in range(sub.shape[0])
        )

    # for lab in np.unique(msk)[1:]:
    #     outputs = Parallel(n_jobs=-1)(
    #         delayed(analyse_s)(sub[t,...], grd[t,...], msk == lab)
    #         for t in range(1)
    #         )
        
    # # Get profiles
    # prf, acc = [], []
    # for t in range(arr.shape[0]):
    #     vals = arr[t, ...][idxs][sort]
    #     distance_grid = np.arange(np.min(dist), np.max(dist) + 1, 1)
    #     interp_function = interp1d(
    #         dist, vals, fill_value="extrapolate", assume_sorted=True)
    #     prf.append(interp_function(distance_grid))
    #     acc.append(get_acc(interp_function(distance_grid)))
    
#%% Execute -------------------------------------------------------------------

from scipy.interpolate import interp1d
from skimage.transform import rescale

if __name__ == "__main__":
    
    for path in data_path.glob(f"*rf-{rf}_stk.tif*"):    
            
        if path.name == f"Exp1_rf-{rf}_stk.tif":
        
            # Open data
            msk = io.imread(str(path).replace("stk", "msk"))
            sub = io.imread(str(path).replace("stk", "sub"))
            grd = io.imread(str(path).replace("stk", "grd"))    
        
            t0 = time.time()    
            print(path.name)
            analyse(sub, grd, msk)

            # idxs = np.where(msk == 2)
            # test_msk = np.full_like(msk, 0)
            # for coord in zip(idxs[0], idxs[1]):
            #     if coord[0] % 4 == 0 and coord[1] % 4 == 0:
            #         test_msk[coord] = 255

            t1= time.time()
            print(f"runtime : {t1 - t0:.5f}")
            
            # Display
            # import napari
            # viewer = napari.Viewer()
            # viewer.add_labels(msk)
            # viewer.add_labels(test_msk)
            # viewer.add_image(ddt, colormap="turbo")
            # viewer.add_image(tmp_mask, colormap="red")
            # viewer.add_image(tmp_skel, blending="additive")
            
#%% 

# def avg_val_dist(arr, idxs, ddt):
    
#     dist = ddt[idxs].astype(int)
#     unique_dist = np.unique(dist)
#     masks = {d: dist == d for d in unique_dist}

#     vals = []
#     for t in range(arr.shape[0]):
#         tmp_vals = arr[t, ...][idxs]
#         avg_vals = [np.mean(tmp_vals[mask]) for d, mask in masks.items()]
#         vals.append(avg_vals)
    
#     return vals, unique_dist