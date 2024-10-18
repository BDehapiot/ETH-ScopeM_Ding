## Procedure

### `extract.py`
Read `.ome` image stacks from `data_path` folder and perform rescaling and rolling average over time. The averaged data is used for the extraction of segmentation masks in later stages of the procedure.

- **Paths**
```bash
- data_path       # str, path to data directory
```

- **Parameters**
```bash
- rf              # float, rescaling factor (should be between 0 and 1)
- window_size     # int, temporal window size for rollling average (should be odd)
```

- **Outputs**
```bash
- ..._stack.tif   # float32 ndarray, rescaled stack
- ..._rstack.tif  # float32 ndarray, rescaled + rolled avg. stack
```

### `process.py`

- **Paths**
```bash
- data_path       # str, path to data directory
```

- **Parameters**
```bash
- min_size        # int, minimum size in pixels for considering segmented objects
```

- **Outputs**
```bash
- ..._probs.tif   # float32 ndarray, probabilities of deep-learning predictions
- ..._mask.tif    # uint8 ndarray, labelled segmented objects
```



### `analyse.py`