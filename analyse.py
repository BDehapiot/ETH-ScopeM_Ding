#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

data_path = Path("D:\local_Ding\data")

#%% Function(s): --------------------------------------------------------------

# def analyse(stack, mask):
    
#     global intensities
    
#     # Nested functions --------------------------------------------------------
       
#     # Execute -----------------------------------------------------------------  
    
#     nT = stack.shape[0]
    
#     intensities = []
#     for lab in np.unique(mask):
#         intensity = np.zeros(nT)
#         for t in range(nT):
#             intensity[t] = np.mean(stack[t, ...][mask == lab])
#         intensities.append(intensity)
#     intensities = np.stack(intensities).T

#     pass

def analyse(stack, mask):
    
    global intensities
    
    # Nested functions --------------------------------------------------------
       
    # Execute -----------------------------------------------------------------  
    
    nT = stack.shape[0]
    
    intensities = []
    for lab in np.unique(mask):
        
        intensity = np.zeros(nT)
        for t in range(nT):
            intensity[t] = np.mean(stack[t, ...][mask == lab])
        intensities.append(intensity)
    intensities = np.stack(intensities).T

    pass

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    rf = 0.1
    
    for path in data_path.glob(f"*rf-{rf}_raw*"):
        
        if path.name == "Exp1_rf-0.1_raw.tif":
        
            stack = io.imread(path.parent / (path.name.replace("raw", "stack")))
            mask = io.imread(path.parent / (path.name.replace("raw", "mask")))
            analyse(stack, mask)
    
    pass

#%%

from skimage.morphology import disk, binary_dilation


rSize = 3
rDist = rSize * 2

rois = np.zeros_like(mask)
nY, nX = rois.shape
y = np.arange(rSize, nY, rDist)
x = np.arange(rSize, nX, rDist)
y_grid, x_grid = np.meshgrid(y, x)
idx = (y_grid.ravel(), x_grid.ravel())
rois[idx] = 1
rois = binary_dilation(rois, footprint=disk(rSize))

# Display
import napari
viewer = napari.Viewer()
viewer.add_labels(mask, blending="additive")
viewer.add_image(rois, blending="additive", contrast_limits=(0, 1))

#%%

# idx = np.where(mask == 9)
# test = stack[:, idx[0], idx[1]]
# # test = np.mean(test, axis=1)
