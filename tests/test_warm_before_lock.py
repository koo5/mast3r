"""NEVER WAIT FOR THE NETWORK UNDER A PAIR'S LOCK: a lazily loaded model -- hillview's, whose first
use waits for room on the box -- is loaded before a pair's lock is taken, for a miss and for a canary drawn on a
locked hit alike; a run needing no inference never loads it. Real forward_mast3r and matcher, a stand-in network."""
import hashlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga as sga  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _standin_codec  # noqa: E402
from test_canary import RECORD, COMPUTE, H, W, fake_inference  # noqa: E402

_standin_codec.install()
HELD = []                                   # the pair locks held right now


class Lazy:
    """hillview's lazily loaded model as sparse_ga sees it: `loaded`, and `warm()` -- which on a box waits for room"""

    def __init__(self):
        self.loaded = False
        self.warmed_holding = []            # the pair locks held at each load

    def warm(self):
        self.warmed_holding.append(len(HELD))
        self.loaded = True


def inference(model, *a, **k):
    if not model.loaded:                    # the old path: the first inference loads it, wherever that is
        model.warm()
    return fake_inference(model, *a, **k)


@pytest.fixture
def tracked(monkeypatch, tmp_path):
    monkeypatch.setenv('MAST3R_PAIR_POST_THREADS', '0')
    monkeypatch.setattr(sga, 'CANARY_RECORD', {'results': [], 'mask_checked': 0, 'mask_on': [], 'done': set(),
                                               'fresh': {}})
    monkeypatch.setattr(sga, 'CANARY_COMPUTE', COMPUTE)
    monkeypatch.setattr(sga, 'CANARY_SALT', 'run-1')
    monkeypatch.setattr(sga, 'CANARY_FRESH_ROOT', str(tmp_path / 'fresh'))
    original = sga._PairLock

    class Tracked(original):
        def acquire(self, block):
            got = super().acquire(block)
            if got:
                HELD.append(self)
            return got

        def __exit__(self, *exc):
            HELD.remove(self)
            return super().__exit__(*exc)

    monkeypatch.setattr(sga, '_PairLock', Tracked)
    HELD.clear()
    return sga


def run(cache, model, rate=0.0, masks=None):
    pairs = [({'instance': 'view-a'}, {'instance': 'view-b'})]
    prov = {'record': dict(RECORD), 'require': None, 'vid': 'v', 'reads': []}
    kw = {}
    if masks is not None:
        ident = {v: (hashlib.sha256(v.encode()).hexdigest() if m is not None else sga.MASK_IDENTITY_NONE)
                 for v, m in masks.items()}
        kw['corr_masks'] = lambda i1, i2: (masks[i1], masks[i2], ident[i1], ident[i2])
    with patch.object(sga, 'CANARY_RATE', rate), patch.object(sga, 'symmetric_inference', inference):
        return sga.forward_mast3r(pairs, model, cache, device='cpu', subsample=4, provenance=prov, **kw)[0]


MASKS = {'view-a': np.pad(np.ones((6, W), bool), ((0, H - 6), (0, 0))), 'view-b': None}


def test_a_miss_loads_the_model_before_its_lock(tracked):
    with tempfile.TemporaryDirectory() as cache:
        m = Lazy()
        run(cache, m)                                           # every pair a miss: computed under its lock
        assert sga.LAST_FORWARD_PROFILE['computed_pairs'] == 1
        assert m.warmed_holding == [0], m.warmed_holding        # loaded once, holding no pair's lock


def test_a_canary_on_a_locked_hit_loads_the_model_before_its_lock(tracked):
    """a masked hit takes its lock (its run-local copy), and a canary drawn on it runs the network"""
    with tempfile.TemporaryDirectory() as cache:
        run(cache, Lazy(), masks=MASKS)                         # the cache, made
        m = Lazy()
        run(cache, m, rate=1.0, masks=MASKS)                    # every hit a canary
        assert sga.LAST_FORWARD_PROFILE['canary_pairs'] == 1 and sga.LAST_FORWARD_PROFILE['computed_pairs'] == 0
        assert m.warmed_holding == [0], m.warmed_holding


def test_a_run_needing_no_inference_never_loads_it(tracked):
    with tempfile.TemporaryDirectory() as cache:
        run(cache, Lazy(), masks=MASKS)
        m = Lazy()
        run(cache, m, masks=MASKS)                              # all hits, no canary: the lock taken, no network
        assert m.warmed_holding == [] and not m.loaded
