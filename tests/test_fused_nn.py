"""fast_nn's fused NN query (NN_KERNEL 'fused'): the db row each seed picks, ties to the first.

On a card the compiled kernels run; elsewhere Triton's interpreter runs the same kernels on CPU tensors
(TRITON_INTERPRET=1), which checks the tiling, the chunking and the tie rule but not the FMA rounding (the
interpreter rounds the product and the sum separately) -- the card check is the bench's nn_fused_check.py.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CUDA = torch.cuda.is_available()             # no card: conftest.py has set TRITON_INTERPRET
pytest.importorskip('triton')
from mast3r import fast_nn  # noqa: E402

DEV = 'cuda' if CUDA else 'cpu'


def descriptors(n, seed, d=24):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture(params=['dot', 'fma'])
def kernel(request, monkeypatch):
    monkeypatch.setitem(fast_nn.NN_FUSED_TILES, 'kernel', request.param)
    return request.param


def fused(A, B):
    A, B = torch.from_numpy(A).to(DEV), torch.from_numpy(B).to(DEV)
    return fast_nn._fused_launch(A, B.T.contiguous()).cpu().numpy()


@pytest.mark.parametrize('M,N', [(1, 1), (5, 127), (17, 129), (64, 1000), (300, 5000)])
def test_the_best_row_wherever_it_is_clear(M, N, kernel):
    A, B = descriptors(M, 1), descriptors(N, 2)
    sims = A.astype(np.float64) @ B.astype(np.float64).T
    top2 = np.sort(sims, axis=1)[:, -2:] if N > 1 else np.c_[sims - 1, sims]
    clear = top2[:, 1] - top2[:, 0] > 1e-5             # a near-tie may go either way under other rounding
    got = fused(A, B)
    assert got.dtype == np.int64 and got.shape == (M,)
    assert np.array_equal(got[clear], sims.argmax(1)[clear])
    assert clear.mean() > 0.9


@pytest.mark.parametrize('M,N', [(3, 1000), (40, 1000), (300, 1000), (3, 100000)])
def test_ties_go_to_the_first_row(M, N, kernel):
    """the same row at 5, 7 (one tile), 200 (another chunk at N 1000, the same chunk at N 100000) and 900
    (another chunk): the first wins. Off a card only 'fma': the interpreter's tl.dot is numpy's matmul,
    which may round two equal columns differently by their place in the tile -- not a tie any more"""
    if kernel == 'dot' and not CUDA:
        pytest.skip("the interpreter's tl.dot rounds equal columns apart (BLAS); a card's FMA dot does not")
    A = descriptors(M, 3)
    B = descriptors(N, 4) * 0.5                        # every other row scores half as well
    B[[5, 7, 200, 900]] = A[0]
    got = fused(A, B)
    assert got[0] == 5
    B[5] = descriptors(1, 5)[0] * 0.5
    assert fused(A, B)[0] == 7
    B[7] = B[5]
    assert fused(A, B)[0] == 200


def test_agrees_with_the_blocked_query_off_the_near_ties(kernel):
    A, B = descriptors(500, 6), descriptors(20000, 7)
    ref = fast_nn.bruteforce_nns_A(torch.from_numpy(A), torch.from_numpy(B), device='cpu', dist='dot',
                                   block_size=2**13)
    sims = A.astype(np.float64) @ B.astype(np.float64).T
    top2 = np.sort(sims, axis=1)[:, -2:]
    clear = top2[:, 1] - top2[:, 0] > 1e-5
    assert np.array_equal(fused(A, B)[clear], ref[clear])


def test_an_empty_db_is_refused():
    with pytest.raises(ValueError, match='empty db'):
        fast_nn._fused_launch(torch.from_numpy(descriptors(3, 10)).to(DEV), torch.zeros((24, 0), device=DEV))


def test_cpu_is_refused_not_served_the_other_kernel():
    A, B = descriptors(4, 8), descriptors(10, 9)
    with pytest.raises(ValueError, match='CUDA kernel'):
        fast_nn.fused_nns_A(A, torch.from_numpy(B).T.contiguous(), 'cpu')
    saved = fast_nn.NN_KERNEL
    try:
        fast_nn.NN_KERNEL = 'fused'
        m = fast_nn.cdistMatcher(torch.from_numpy(B), device='cpu')
        with pytest.raises(ValueError, match='CUDA kernel'):
            m.query(torch.from_numpy(A), dist='dot', block_size=2**13)
        with pytest.raises(ValueError, match="dist='dot' only"):
            m.query(torch.from_numpy(A), dist='l2')
        fast_nn.NN_KERNEL = 'tiled'
        with pytest.raises(ValueError, match="'torch' or 'fused'"):
            fast_nn.cdistMatcher(torch.from_numpy(B), device='cpu')
    finally:
        fast_nn.NN_KERNEL = saved


def test_selftest_finds_each_row(monkeypatch, kernel):
    if not CUDA:                                       # the interpreter runs the same kernels on CPU tensors
        monkeypatch.setattr(fast_nn, 'fused_nns_A', lambda A, B_T, device: fast_nn._fused_launch(
            torch.as_tensor(A), B_T).cpu().numpy())
    fast_nn.fused_selftest(DEV)
