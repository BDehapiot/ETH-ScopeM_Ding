# first line: 197
@memory.cache
def get_iip_cached(t_idx, arr, idxs, dists, sorts, xvals):
    vals = arr[t_idx, ...][idxs]
    vals = [vals[sort] for sort in sorts]
    iips = []
    for val, dist, xval in zip(vals, dists, xvals):
        f = interp1d(dist, val, fill_value="extrapolate", assume_sorted=True)
        iips.append(f(xval))
    return iips
