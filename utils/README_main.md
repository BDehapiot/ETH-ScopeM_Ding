## Procedure

### `extract.py`
Read `.ome` image stacks from the `data_path` folder and perform:
1) **Spatial rescaling**  
Reduce spatial resolution, according to `rf` parameter to improve computing 
speed and signal to noise ratio. 
2) **Temporal roll-averaging**  
Rolling average over time according to `window_size` to reduce noise and 
facilitate later segmentation steps.

```bash
# Parameters
- data_path         # str, path to data directory
- rf                # float, rescaling factor (should be between 0 and 1)
- window_size       # int, temporal window size for rollling average (should be odd)

# Outputs
- ..._stack.tif     # float32 ndarray, rescaled stack
- ..._rstack.tif    # float32 ndarray, rescaled + rolled avg. stack
```

### `process.py`
Read `..._rstack.tif` from `data_path` folder and perform:
1) **Predictions**   
Predict object localization using the embedded deep-learning model. 
2) **Labeled mask**  
Average probabilities over time and define labelled objects as contiguous 
regions where probabilities are > 0.5. 


```bash
# Parameters
- data_path        # str, path to data directory
- min_size         # int, minimum size in pixels for considering segmented objects

# Outputs
- ..._probs.tif    # float32 ndarray, probabilities of deep-learning predictions
- ..._mask.tif     # uint8 ndarray, labelled segmented objects
```

### `analyse.py`
Read `..._stack.tif` and `..._mask.tif` from `data_path` folder and perform:
1) **Spatial filtering**   
Kernel mean filtering ignoring pixels outside of segmented objects.
2) **Temporal filtering**  
Pixel level Asymmetric Least Square (ALS) baseline (Low frequency components) subtraction.

```bash
# Parameters 
- data_path             # str, path to data directory

# Outputs
- ..._stack_filt.tif    # float32 ndarray, probabilities of deep-learning predictions
- ..._stack_bsub.tif    # uint8 ndarray, labelled segmented objects

# Note : filter parameters are fixed for the moment
```