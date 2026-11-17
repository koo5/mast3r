"""The cache codec these tests install. hillview's hvz.install hands sparse_ga the real one (the .hvz
container and its content digest); these tests are about the cache's paths, locks and identities, so
they set a stand-in explicitly: the entries are torch pickles under the real CACHE_EXT name, which
keeps `torch.load` usable for asserting what was written."""
import hashlib
import io
import os
import pickle

import torch

from mast3r.cloud_opt import sparse_ga as sga

EXT = '.hvz'
LEGACY = '.pth'


_PRED = '.predictions'


def _name(path):
    """hillview's hvz.name, restated: X@h.pth.predictions.pth -> X@h.hvz.predictions.hvz"""
    if path.endswith(EXT):
        return path
    if path.endswith(LEGACY + _PRED + LEGACY):
        return path[:-len(LEGACY + _PRED + LEGACY)] + EXT + _PRED + EXT
    return path[:-len(LEGACY)] + EXT


def _candidates(path):
    """the hillview rule (hvz.candidates), restated: the .hvz, then the .pth of the same key"""
    name = _name(path)
    if name.endswith(EXT + _PRED + EXT):
        return name, name[:-len(EXT + _PRED + EXT)] + LEGACY + _PRED + LEGACY
    return name, name[:-len(EXT)] + LEGACY


def _dumps(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def install():
    sga.CONTENT_DIGEST = lambda obj: hashlib.sha256(pickle.dumps(obj)).hexdigest()
    sga.CACHE_EXT = EXT
    sga.CACHE_DUMPS = _dumps
    sga.CACHE_LOAD = lambda path, device: torch.load(path, map_location=device)
    sga.CACHE_CANDIDATES = _candidates
    sga.CACHE_ENTRY_REL = lambda path, root: os.path.relpath(path, root)   # one root: no read-only ones
