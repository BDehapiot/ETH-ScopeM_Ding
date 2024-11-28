#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path

# Skimage
from skimage.graph import route_through_array

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")

# Parameters
rf = 0.1

#%% Function(s) ---------------------------------------------------------------

def shortest_path(mask, label):
    pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    for path in data_path.glob(f"*rf-{rf}_stack.tif*"):    
            
        if path.name == "Exp2_rf-0.1_stack.tif":
        
            t0 = time.time()    
            
            print(path.name)
            
            stack = io.imread(path)
            mask = io.imread(str(path).replace("stack", "mask"))
            
            # 
            lbl = 1
            tmp_mask = mask == lbl
            idx = np.where(mask == lbl)
            sub_idx = (idx[0][::10], idx[1][::10])
            tmp_mask = tmp_mask.astype("uint8")
            tmp_mask *= 255
            tmp_mask[sub_idx] = 128

            t1= time.time()
            print(f"runtime : {t1 - t0:.3f}")
            
            # Display
            import napari
            viewer = napari.Viewer()
            viewer.add_image(tmp_mask)
            
#%% 

# from skimage.graph import route_through_array

# lbl = 2
# tmp_mask = mask == lbl
# cost_array = np.where(tmp_mask == 1, 1, np.inf)
# idxs = np.where(tmp_mask)
# start = (idxs[0][0], idxs[1][0])
# end = (idxs[0][-1], idxs[1][-1])


# t0 = time.time()   
# indices, weight = route_through_array(
#     cost_array, start, end, fully_connected=True)
# t1= time.time()
# print(f"route_through_array() : {t1 - t0:.5f}")

# path_idx = (
#     [ind[0] for ind in indices],
#     [ind[1] for ind in indices],
#     )

# tmp_mask = tmp_mask.astype("uint8") * 255
# tmp_mask[path_idx] = 128

# import napari
# viewer = napari.Viewer()
# viewer.add_image(tmp_mask)