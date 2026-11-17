"""A nearest-neighbour query answering an index outside its database is refused by name (fast_nn._nn_checked):
an RX 580 answered 4475493158418731731 for a db of 782, and -1 (an all-NaN row)
would wrap silently to the last row."""
import hashlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r import fast_nn  # noqa: E402
from mast3r.cloud_opt import sparse_ga as sga  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _standin_codec  # noqa: E402

_standin_codec.install()
H = W = 16


def descs(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(H, W, 8, generator=g), dim=-1)


def match():
    return fast_nn.fast_reciprocal_NNs(descs(1), descs(2), subsample_or_initxy1=4, ret_xy=False, device='cpu',
                                       dist='dot', block_size=2 ** 13)


def test_a_sound_query_matches_as_before():
    got = match()
    with patch.object(fast_nn, '_nn_checked', lambda idx, n, side, device: np.asarray(idx)):
        ref = match()
    assert all(np.array_equal(a, b) for a, b in zip(got, ref)) and len(got[0])


def test_a_hop_with_nothing_left_to_ask_on_a_masked_view():
    """hop 1 converges every remaining seed, so hop 2 asks zero rows and gets [] back; with view 1 masked (keep1)
    keep1[next1] must still take it -- an empty float array was an IndexError there"""
    real, empty = fast_nn.cdistMatcher.query, []

    def query(self, queries, k=1, **kw):
        if len(queries) == 0:
            empty.append(1)
        return real(self, queries, k, **kw)
    m1 = np.zeros((H, W), bool)
    m1[:5] = True                                        # view 1's top rows masked: keep1 is set
    with patch.object(fast_nn.cdistMatcher, 'query', query):
        for seed in range(1, 400, 2):                    # the first descriptor pair whose loop ends on hop 1
            got = fast_nn.fast_reciprocal_NNs(descs(seed), descs(seed + 1), subsample_or_initxy1=4, ret_xy=False,
                                              device='cpu', dist='dot', block_size=2 ** 13, seed_mask=m1)
            if empty:
                break
    assert empty, 'no pair reached an empty hop: this test proves nothing'
    assert len(got[0]) and not m1.ravel()[got[0]].any()


def _answering(bad_side, value):
    real = fast_nn.cdistMatcher.query
    calls = []

    def query(self, queries, k=1, **kw):
        dis, nn = real(self, queries, k, **kw)
        calls.append(len(calls))
        side = 2 if len(calls) % 2 == 1 else 1           # the loop asks view 2's tree, then view 1's
        if side == bad_side:
            nn = (nn.detach().cpu().numpy() if isinstance(nn, torch.Tensor) else np.asarray(nn)).astype(np.int64)
            nn[0] = value
        return dis, nn
    return patch.object(fast_nn.cdistMatcher, 'query', query)


def test_garbage_from_the_device_is_refused_naming_the_side():
    with _answering(2, 4475493158418731731), pytest.raises(fast_nn.NNIndexOutOfRange) as e:
        match()
    assert "view 2's 256 rows" in str(e.value) and '4475493158418731731' in str(e.value) and 'cpu' in str(e.value)


def test_an_all_nan_row_is_refused_not_wrapped_to_the_last_row():
    with _answering(1, -1), pytest.raises(fast_nn.NNIndexOutOfRange) as e:
        match()
    assert "view 1's 256 rows" in str(e.value) and '[-1]' in str(e.value)


def fake_inference(model, img1, img2, device=None, encodings=None):
    def g(key):
        return torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    a, b = img1['instance'], img2['instance']
    return [{'pts3d': torch.randn(1, H, W, 3, generator=g(k)), 'conf': torch.rand(1, H, W, generator=g(k)) + 1,
             'desc': torch.nn.functional.normalize(torch.randn(1, H, W, 8, generator=g(v)), dim=-1),
             'desc_conf': torch.rand(1, H, W, generator=g(k + 'q')) + 1}
            for v, k in ((a, f'{a}|{b}|1'), (b, f'{a}|{b}|2'), (b, f'{b}|{a}|1'), (a, f'{b}|{a}|2'))]


def test_the_forward_pass_names_the_pair():
    with tempfile.TemporaryDirectory() as cache, _answering(2, 10 ** 12), \
            patch.object(sga, 'symmetric_inference', fake_inference), pytest.raises(fast_nn.NNIndexOutOfRange) as e:
        sga.forward_mast3r([({'instance': 'view-a'}, {'instance': 'view-b'})], object(), cache, device='cpu',
                           subsample=4)
    assert str(e.value).startswith('view-a - view-b: ')
