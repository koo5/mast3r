"""fast_nn.NN_CPU_DB_ROWS (on CPU boxes): the db's block on a CPU is a speed knob -- the same nn_A
as today's blocking, ties to the first row across block edges included."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r import fast_nn  # noqa: E402


def descriptors(n, seed, d=24):
    x = np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)
    return torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True))


@pytest.mark.parametrize('rows', [97, 512, 1000, 2048, 8192, 30000])
def test_same_rows_as_todays_blocking(rows, monkeypatch):
    A, B = descriptors(1100, 1), descriptors(20000, 2)
    B[[700, 1500, 9000]] = A[3]                         # one exact tie across block edges at most sizes
    A_big = descriptors(9000, 3)                        # more seeds than one 8192-row seed block
    ref = [fast_nn.bruteforce_nns_A(a, B, device='cpu', dist='dot', block_size=2**13) for a in (A, A_big)]
    monkeypatch.setattr(fast_nn, 'NN_CPU_DB_ROWS', rows)
    got = [fast_nn.bruteforce_nns_A(a, B, device='cpu', dist='dot', block_size=2**13) for a in (A, A_big)]
    assert ref[0][3] == 700
    assert all(np.array_equal(r, g) for r, g in zip(ref, got))
    got_none = fast_nn.bruteforce_nns_A(A, B, device='cpu', dist='dot', block_size=None)
    assert np.array_equal(got_none, ref[0])


@pytest.mark.parametrize('bad', [0, -5, 2.5])
def test_a_bad_value_is_refused(bad, monkeypatch):
    monkeypatch.setattr(fast_nn, 'NN_CPU_DB_ROWS', bad)
    with pytest.raises(ValueError, match='positive number of rows'):
        fast_nn.bruteforce_nns_A(descriptors(4, 4), descriptors(10, 5), device='cpu', dist='dot', block_size=2**13)
