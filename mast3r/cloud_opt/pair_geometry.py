"""Bounded samples of the exact 3D comparisons used by the MASt3R optimiser."""
import gzip
import json
import os
import numpy as np


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def comparisons(source, target, xy_a, xy_b, confidence, limit=64):
    source, target, xy_a, confidence = map(array, (source, target, xy_a, confidence))
    xy_b = None if xy_b is None else array(xy_b)
    if not (len(source) == len(target) == len(xy_a) == len(confidence)):
        raise ValueError('pair geometry rows disagree')
    if xy_b is not None and len(xy_b) != len(source):
        raise ValueError('matched target pixels disagree')
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    valid &= np.isfinite(xy_a).all(axis=1) & np.isfinite(confidence) & (confidence > 0)
    if xy_b is not None:
        valid &= np.isfinite(xy_b).all(axis=1)
    ids = np.flatnonzero(valid)
    if len(ids) > limit:
        ids = ids[np.linspace(0, len(ids) - 1, limit, dtype=int)]
    return [dict(source=source[k].tolist(), target=target[k].tolist(),
                 pixel_a=xy_a[k].tolist(), pixel_b=None if xy_b is None else xy_b[k].tolist(),
                 confidence=float(confidence[k]), distance=float(np.linalg.norm(source[k] - target[k])))
            for k in ids]


def write_geometry(path, result, sizes, matched, pointmaps, predictions, images, weight, sources):
    """`sources` {view: (points, pixels)}: what the pointmap fallback compares with its targets, row
    for row -- each correspondence's ANCHOR point and the anchor's pixel, where the target was
    predicted (sparse_ga). Not result['pts3d'], which
    are the points at the correspondence pixels."""
    points = [array(p) for p in result['pts3d']]
    poses = array(result['cam2w'])
    sizes = array(sizes)
    rows = {}

    def entry(i, j):
        return rows.setdefault(f'{i}-{j}', dict(
            cameras=[poses[i, :3, 3].tolist(), poses[j, :3, 3].tolist()],
            sizes=[sizes[i].tolist(), sizes[j].tolist()], matched=[], pointmap=[],
            pointmap_weight=float(weight)))

    for pair in matched:
        i, j = int(pair.img1), int(pair.img2)
        entry(i, j)['matched'] = comparisons(points[i][pair.slice1], points[j][pair.slice2],
                                             pair.pix1, pair.pix2, pair.confs)
        entry(i, j)['matched_count'] = len(pair.confs)
    for pair in pointmaps:
        i, j = int(pair.img1), int(pair.img2)
        predicted, confidence = predictions[images[j]][images[i]]
        predicted = array(predicted)
        # Exactly the target transform in loss_dust3r; no nearest-neighbour matching.
        target = predicted @ poses[j, :3, :3].T + poses[j, :3, 3]
        source, pixel = sources[i]
        entry(i, j)['pointmap'] = comparisons(source, target, pixel, None, confidence)
        entry(i, j)['pointmap_count'] = len(predicted)
    # version 2: a pointmap row's source and pixel_a are its ANCHOR's (above); in a
    # version-1 file they are the correspondence's, and its target the prediction at another pixel
    data = dict(version=2, coordinates='solver world', units='solver units, not metres',
                state='returned final solver evaluation', sample_limit=64, pairs=rows)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        # dumps, not dump: the same text, through the C encoder in one write -- dump streams the pure-Python
        # one through ~3 M writes into gzip (3.75 -> 1.98 s on a solve of 147 views)
        f.write(json.dumps(data, allow_nan=False, separators=(',', ':')))
    os.replace(tmp, path)
