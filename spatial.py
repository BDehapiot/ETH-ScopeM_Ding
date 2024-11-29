#%% Imports -------------------------------------------------------------------

import time
import numpy as np
from skimage import io
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

# Paths
data_path = Path("D:\local_Ding\data")

# Parameters
rf = 0.1

#%% Function(s) ---------------------------------------------------------------

import numpy as np
import heapq
import math

def compute_shortest_paths_dijkstra(binary_image, source, connectivity=8):
    """
    Computes the shortest path distances from the source pixel to all other pixels within the object,
    considering the correct costs for diagonal and orthogonal movements.

    Parameters:
    - binary_image: 2D numpy array where object pixels have value 1 and background pixels have value 0.
    - source: Tuple (row, col) of the source pixel.
    - connectivity: 4 or 8 for pixel connectivity.

    Returns:
    - distance_map: 2D numpy array of shortest distances from the source within the object.
    """
    if connectivity not in [4, 8]:
        raise ValueError("Connectivity must be either 4 or 8.")

    rows, cols = binary_image.shape
    distance_map = np.full_like(binary_image, np.inf, dtype=float)
    visited = np.zeros_like(binary_image, dtype=bool)
    heap = []

    # Validate source pixel
    sr, sc = source
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError("Source pixel is out of bounds.")
    if binary_image[sr, sc] == 0:
        raise ValueError("Source pixel is not within the object.")

    # Initialize
    distance_map[sr, sc] = 0.0
    heapq.heappush(heap, (0.0, (sr, sc)))

    # Define neighbor offsets and their respective costs
    if connectivity == 8:
        neighbor_offsets = [
            (-1, -1, math.sqrt(2)), (-1, 0, 1), (-1, 1, math.sqrt(2)),
            (0, -1, 1),                          (0, 1, 1),
            (1, -1, math.sqrt(2)),  (1, 0, 1),  (1, 1, math.sqrt(2))
        ]
    else:  # connectivity == 4
        neighbor_offsets = [
            (-1, 0, 1),
            (0, -1, 1),           (0, 1, 1),
            (1, 0, 1)
        ]

    while heap:
        current_dist, (r, c) = heapq.heappop(heap)
        if visited[r, c]:
            continue
        visited[r, c] = True

        for dr, dc, cost in neighbor_offsets:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                not visited[nr, nc] and binary_image[nr, nc] == 1):

                new_dist = current_dist + cost
                if new_dist < distance_map[nr, nc]:
                    distance_map[nr, nc] = new_dist
                    heapq.heappush(heap, (new_dist, (nr, nc)))

    # Replace infinite distances with -1 to indicate unreachable pixels
    distance_map[np.isinf(distance_map)] = -1

    return distance_map

#%% Execute -------------------------------------------------------------------

from skimage.morphology import skeletonize
from skimage.graph import route_through_array

if __name__ == "__main__":
    
    for path in data_path.glob(f"*rf-{rf}_stack.tif*"):    
            
        if path.name == "Exp2_rf-0.1_stack.tif":
        
            t0 = time.time()    
            
            print(path.name)
            
            stack = io.imread(path)
            mask = io.imread(str(path).replace("stack", "mask"))
            
            
            # 
            lbl = 2
            tmp_mask = mask == lbl
            idxs = np.where(tmp_mask)
            distance_map = compute_shortest_paths_dijkstra(
                tmp_mask, (idxs[0][0], idxs[1][0]), connectivity=8)

            t1= time.time()
            print(f"runtime : {t1 - t0:.3f}")
            
            # Display
            import napari
            viewer = napari.Viewer()
            viewer.add_image(distance_map)
            # viewer.add_image(tmp_mask, colormap="red")
            # viewer.add_image(tmp_skel, blending="additive")
            
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