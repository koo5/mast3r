"""The NN reduction verifies its own answers (fast_nn._verified): an index whose distance is not the best the reduction
returned with it -- an output never written, stale memory that last held another hop's indices (met on an RX 580)
-- is marked NN_UNVERIFIED and refused by name; a sound answer, fp32 or TF32, passes untouched."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r import fast_nn  # noqa: E402

H = W = 16


def unit(n, seed, d=24):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


@pytest.mark.parametrize('dist', ['dot', 'l2'])
@pytest.mark.parametrize('block', [None, 64])          # one matmul / the blocked merge
def test_sound_answers_pass_untouched(dist, block):
    A, B = unit(300, 1), unit(500, 2)
    got = fast_nn.bruteforce_nns_A(A, B, device='cpu', dist=dist, block_size=block)
    with patch.object(fast_nn, '_verified', lambda A, B, best, nn, dist: nn):
        ref = fast_nn.bruteforce_nns_A(A, B, device='cpu', dist=dist, block_size=block)
    assert np.array_equal(got, ref) and (got >= 0).all()


def test_a_stale_in_range_index_is_marked():
    A, B = unit(50, 3), unit(80, 4)
    sim, nn = torch.max(A @ B.T, dim=1)
    stale = nn.roll(7)                                  # another hop's indices: in range, plausible, wrong
    out = fast_nn._verified(A, B, sim.neg(), stale, 'dot')
    wrong = stale != nn
    assert wrong.any() and (out[wrong] == fast_nn.NN_UNVERIFIED).all() and (out[~wrong] == nn[~wrong]).all()


def test_an_unwritten_reduction_is_refused_in_the_loop():
    """torch.max hands back an index buffer it did not write -- in range here, so only the check can see it"""
    real = torch.max

    def stale_max(X, dim):
        v, i = real(X, dim=dim)
        return v, i.roll(3)
    with patch.object(fast_nn.torch, 'max', stale_max), pytest.raises(fast_nn.NNIndexOutOfRange) as e:
        fast_nn.fast_reciprocal_NNs(unit(H * W, 5).reshape(H, W, 24), unit(H * W, 6).reshape(H, W, 24),
                                    subsample_or_initxy1=4, ret_xy=False, device='cpu', dist='dot', block_size=2 ** 13)
    assert str(fast_nn.NN_UNVERIFIED) in str(e.value) and 'did not write' in str(e.value)


def tf32(x):
    """x with TF32's 10-bit mantissa (round to nearest): what a card's --tf32 matmul multiplies"""
    b = x.contiguous().view(torch.int32)
    return ((b + (1 << 12)) & ~((1 << 13) - 1)).view(torch.float32)


def test_a_tf32_reduction_is_judged_at_tf32s_precision():
    """a --tf32 run's best differs from the fp32 gather by more than fp32's tolerance: it must not refuse every pair"""
    A, B = unit(2000, 7), unit(3000, 8)
    sim, nn = torch.max(tf32(A) @ tf32(B).T, dim=1)     # the card's TF32 answer
    worst = float(((A * B[nn]).sum(1) - sim).abs().max())
    assert 1e-4 < worst < 4e-3, worst                    # past fp32's tolerance, inside TF32's
    assert (fast_nn._verified(A, B, sim.neg(), nn, 'dot') == fast_nn.NN_UNVERIFIED).any()   # judged as fp32: refused
    with patch.object(fast_nn, '_verify_tolerance', lambda A: 4e-3):                         # as a TF32 card judges
        assert torch.equal(fast_nn._verified(A, B, sim.neg(), nn, 'dot'), nn)
