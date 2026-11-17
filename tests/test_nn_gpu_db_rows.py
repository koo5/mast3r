"""fast_nn.NN_GPU_DB_ROWS: the db's block on a card, NN_CPU_DB_ROWS's twin for every other device.

There is no card here, so this checks the ROUTE and the block arithmetic on CPU tensors: a stand-in for fast_nn's
`torch` answers device(...).type as a card would and records the shape of every block the reduction sees. On a card
another block size may pick another GEMM kernel; that is the card's many-pairs check against float64, not this."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r import fast_nn  # noqa: E402


def descriptors(n, seed, d=24):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True))


class _Torch:
    """fast_nn's `torch`, but device(...) says `kind` and torch.max records the (seeds, db rows) of each block"""

    def __init__(self, kind):
        self.kind, self.blocks = kind, []

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, d):
        return SimpleNamespace(type=self.kind)

    def max(self, x, dim):
        self.blocks.append(tuple(x.shape))
        return torch.max(x, dim=dim)


def run(monkeypatch, kind, A, B, gpu=None, cpu=None, block_size=2**13):
    shim = _Torch(kind)
    monkeypatch.setattr(fast_nn, 'torch', shim)
    monkeypatch.setattr(fast_nn, 'NN_GPU_DB_ROWS', gpu)
    monkeypatch.setattr(fast_nn, 'NN_CPU_DB_ROWS', cpu)
    # the tensors stay on the CPU: only the device TYPE the knobs read is a card's
    got = fast_nn.bruteforce_nns_A(A, B, device='cpu', dist='dot', block_size=block_size)
    return got, shim.blocks


@pytest.mark.parametrize('rows', [97, 512, 2048, 8192, 30000])
def test_on_a_card_the_db_goes_in_blocks_of_rows_with_todays_answer(rows, monkeypatch):
    A, B = descriptors(1100, 1), descriptors(20000, 2)
    B[[700, 1500, 9000]] = A[3]                         # one exact tie across block edges at most sizes
    A_big = descriptors(9000, 3)                        # more seeds than one 8192-row seed block
    for a in (A, A_big):
        ref, ref_blocks = run(monkeypatch, 'cuda', a, B)
        got, blocks = run(monkeypatch, 'cuda', a, B, gpu=rows)
        assert np.array_equal(ref, got)
        if rows == 8192 and len(a) * len(B) <= 8192**2:
            assert blocks == ref_blocks == [(len(a), len(B))]      # = block_size: today's one dense block
            continue
        # seeds in blocks of block_size (8192), the db in blocks of `rows`, every db row seen once per seed block
        seed_blocks = [min(8192, len(a) - i) for i in range(0, len(a), 8192)]
        db_blocks = [min(rows, len(B) - j) for j in range(0, len(B), rows)]
        assert blocks == [(s, d) for s in seed_blocks for d in db_blocks]
    got, _ = run(monkeypatch, 'cuda', A, B, gpu=rows)
    assert got[3] == 700                                # ties to the FIRST row, across block edges


def test_none_is_todays_path_on_a_card(monkeypatch):
    A, B = descriptors(1100, 4), descriptors(20000, 5)
    got, blocks = run(monkeypatch, 'cuda', A, B)        # 1100 x 20000 <= 8192**2: one dense block, as before
    assert blocks == [(1100, 20000)]
    got_none, blocks_none = run(monkeypatch, 'cuda', A, B, block_size=None)
    assert blocks_none == [(1100, 20000)] and np.array_equal(got, got_none)


def test_each_knob_on_its_own_device(monkeypatch):
    A, B = descriptors(1100, 6), descriptors(20000, 7)
    _, card = run(monkeypatch, 'cuda', A, B, gpu=4096, cpu=1000)
    assert card == [(1100, 4096)] * 4 + [(1100, 20000 - 4 * 4096)]
    _, cpu = run(monkeypatch, 'cpu', A, B, gpu=4096, cpu=1000)
    assert cpu == [(1100, 1000)] * 20
    _, cpu_alone = run(monkeypatch, 'cpu', A, B, gpu=4096)
    assert cpu_alone == [(1100, 20000)]                 # the card's knob never blocks a CPU query


def test_no_seed_block_given_takes_all_seeds_at_once(monkeypatch):
    A, B = descriptors(300, 8), descriptors(5000, 9)
    ref = (A @ B.T).argmax(1).numpy()
    got, blocks = run(monkeypatch, 'cuda', A, B, gpu=2048, block_size=None)
    assert blocks == [(300, 2048), (300, 2048), (300, 904)] and np.array_equal(got, ref)


@pytest.mark.parametrize('bad', [0, -5, 2.5])
def test_a_bad_value_is_refused(bad, monkeypatch):
    with pytest.raises(ValueError, match='NN_GPU_DB_ROWS .*positive number of rows'):
        run(monkeypatch, 'cuda', descriptors(4, 10), descriptors(10, 11), gpu=bad)
