"""fast_nn.NN_SMALL_QUERY_CPU_ROWS: a query of fewer rows is answered on the CPU, verified there,
the rest on the matcher's device; None changes nothing; a bad value is refused. No card here: the matcher is told its
device is a card and the call it makes is observed -- the route is what is under test, the CPU answer is real."""
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


def card_matcher(db, monkeypatch):
    """a cdistMatcher that believes its device is 'cuda' (its tensors stay on the CPU), and the calls it makes"""
    calls = []
    real = fast_nn.bruteforce_nns_A

    def observed(A, B, device='cuda', **kw):
        calls.append((len(A), device, A.device.type, B.device.type))
        return real(A, B, device='cpu', **kw)
    monkeypatch.setattr(fast_nn, 'bruteforce_nns_A', observed)
    m = fast_nn.cdistMatcher(db, device='cpu')
    m.device = 'cuda'
    return m, calls


def test_small_queries_go_to_the_cpu_and_get_the_same_answer(monkeypatch):
    db = descriptors(5000, 1)
    monkeypatch.setattr(fast_nn, 'NN_SMALL_QUERY_CPU_ROWS', 64)
    fast_nn.stats_reset()
    m, calls = card_matcher(db, monkeypatch)
    for n in (1, 63, 64, 300):
        q = descriptors(n, 10 + n)
        _, got = m.query(q, dist='dot', block_size=2**13)
        ref = fast_nn.bruteforce_reciprocal_nns(q, db, device='cpu', dist='dot')[0]   # nn_A of the full routine
        assert np.array_equal(got, ref), n
    assert [(n, dev) for n, dev, _a, _b in calls] == [(1, 'cpu'), (63, 'cpu'), (64, 'cuda'), (300, 'cuda')]
    assert fast_nn.STATS['small_cpu'] == 2 and m._db_cpu is not None


def test_none_routes_nothing_and_a_cpu_matcher_is_left_alone(monkeypatch):
    db = descriptors(2000, 2)
    m, calls = card_matcher(db, monkeypatch)                             # NN_SMALL_QUERY_CPU_ROWS None
    m.query(descriptors(1, 3), dist='dot', block_size=2**13)
    monkeypatch.setattr(fast_nn, 'NN_SMALL_QUERY_CPU_ROWS', 64)
    cpu = fast_nn.cdistMatcher(db, device='cpu')
    cpu.query(descriptors(1, 4), dist='dot', block_size=2**13)
    assert [dev for _n, dev, _a, _b in calls] == ['cuda', 'cpu'] and cpu._db_cpu is None


@pytest.mark.parametrize('bad', [0, -1, 2.5, True])
def test_a_bad_value_is_refused(bad, monkeypatch):
    monkeypatch.setattr(fast_nn, 'NN_SMALL_QUERY_CPU_ROWS', bad)
    with pytest.raises(ValueError, match='NN_SMALL_QUERY_CPU_ROWS'):
        fast_nn.cdistMatcher(descriptors(10, 5), device='cpu')
