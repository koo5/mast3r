"""THE CANARY: a share of cache hits recomputed and compared with what is stored (never written into a
cache), and the mask check on every masked hit. A check that does not pass is a warning, and the pair is solved on
the recompute. Real forward_mast3r and the real matcher; the network is a deterministic stand-in
(inference is not what is under test)."""
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga as sga  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _standin_codec  # noqa: E402

_standin_codec.install()
H = W = 16
COMPUTE = {'device_name': 'test-cpu', 'isa': 'AVX2', 'threads': 1, 'torch': torch.__version__}
RECORD = {'checkpoint': 'c', 'forward_version': sga.FORWARD_VERSION, 'device': 'cpu', 'precision': 'fp32',
          'kernels': 'mkl', 'compute': COMPUTE}


def _view(name, salt):
    g = torch.Generator().manual_seed(int(hashlib.sha256(f'{name}|{salt}'.encode()).hexdigest()[:8], 16))
    return g


def fake_inference(model, img1, img2, device=None, encodings=None):
    """four results (11, 21, 22, 12) as the network gives them: 11 and 21 are decoders 1 and 2 of the pass
    (img1, img2), 22 and 12 decoders 1 and 2 of the pass (img2, img1) -- so (b, a)'s four are (a, b)'s reordered,
    and the two passes' confidences differ, as the real ones do. A view's descriptors depend on the view alone."""
    a, b = img1['instance'], img2['instance']
    out = []
    for view, key in ((a, f'{a}|{b}|1'), (b, f'{a}|{b}|2'), (b, f'{b}|{a}|1'), (a, f'{b}|{a}|2')):
        g = _view(key, 'x')
        d = torch.nn.functional.normalize(torch.randn(1, H, W, 8, generator=_view(view, 'd')), dim=-1)
        out.append({'pts3d': torch.randn(1, H, W, 3, generator=g) + 5, 'conf': torch.rand(1, H, W, generator=g) + 1,
                    'desc': d, 'desc_conf': torch.rand(1, H, W, generator=_view(key, 'q')) + 1})
    return out


SUNK = []


@pytest.fixture(params=['sequential', 'pipelined'])
def canary(monkeypatch, request, tmp_path):
    # pipelined, a canary's post stage runs on a post thread (_PairPipeline): the same checks, the same record
    monkeypatch.setenv('MAST3R_PAIR_POST_THREADS', '2' if request.param == 'pipelined' else '0')
    monkeypatch.setattr(sga, 'CANARY_RECORD', {'results': [], 'mask_checked': 0, 'mask_on': [], 'done': set(),
                                              'fresh': {}})
    monkeypatch.setattr(sga, 'CANARY_COMPUTE', COMPUTE)
    monkeypatch.setattr(sga, 'CANARY_SALT', 'run-1')
    monkeypatch.setattr(sga, 'CANARY_FRESH_ROOT', str(tmp_path / 'run' / 'cache' / 'canary'))
    SUNK.clear()
    monkeypatch.setattr(sga, 'CANARY_SINK', SUNK.append)
    return sga


def run(cache, rate, masks=None, record=RECORD, model=object(), both=False):
    pairs = [({'instance': 'view-a'}, {'instance': 'view-b'})]
    if both:
        pairs.append(({'instance': 'view-b'}, {'instance': 'view-a'}))
    prov = {'record': dict(record), 'require': None, 'vid': 'v', 'reads': []}
    kw = {}
    if masks is not None:
        ident = {v: (hashlib.sha256(v.encode()).hexdigest() if m is not None else sga.MASK_IDENTITY_NONE)
                 for v, m in masks.items()}
        kw['corr_masks'] = lambda i1, i2: (masks[i1], masks[i2], ident[i1], ident[i2])
    with patch.object(sga, 'CANARY_RATE', rate), patch.object(sga, 'symmetric_inference', fake_inference):
        res, _ = sga.forward_mast3r(pairs, model, cache, device='cpu', subsample=4, provenance=prov, **kw)
    return res


def test_a_reverse_from_the_cache_owes_its_origins_bits(canary):
    """(b, a) is stored as (a, b) transposed; recomputed directly it has the same rows and other confidences"""
    with tempfile.TemporaryDirectory() as cache:
        run(cache, 0.0, both=True)
        assert sga.LAST_FORWARD_PROFILE['computed_pairs'] == 1          # (b, a) came from the cache, transposed
        run(cache, 1.0, both=True)
        res = {tuple(r['views']): r for r in sga.CANARY_RECORD['results'] if r['kind'] == 'matches'}
        assert set(res) == {('view-a', 'view-b'), ('view-b', 'view-a')}
        assert all(r['verdict'] == 'ok' and r['tier'] == 'bits' for r in res.values()), res
        assert res[('view-b', 'view-a')]['detail'].get('as_reverse') is True
        assert 'as_reverse' not in res[('view-a', 'view-b')]['detail']


def corres_of(res):
    (_p, path_corres), = res.values()
    return path_corres


def test_the_draw_is_a_function_of_salt_and_name(canary, monkeypatch):
    names = [f'corres/{i:05d}' for i in range(20000)]
    monkeypatch.setattr(sga, 'CANARY_RATE', 0.1)
    drawn = [n for n in names if sga.canary_drawn(n)]
    assert 0.09 < len(drawn) / len(names) < 0.11
    assert drawn == [n for n in names if sga.canary_drawn(n)]
    monkeypatch.setattr(sga, 'CANARY_SALT', 'run-2')
    assert drawn != [n for n in names if sga.canary_drawn(n)]
    monkeypatch.setattr(sga, 'CANARY_RATE', 0.0)
    assert not any(sga.canary_drawn(n) for n in names)


def test_tiers(canary):
    assert sga.canary_tier(dict(RECORD), RECORD) == 'bits'
    assert sga.canary_tier(dict(RECORD, compute=dict(COMPUTE, threads=8)), RECORD) == 'divergence'
    assert sga.canary_tier(dict(RECORD, precision='tf32'), RECORD) == 'precision'
    assert sga.canary_tier(None, RECORD) == 'divergence'
    assert sga.canary_tier(dict(RECORD, derived_from='x'), RECORD) == 'derived'
    gpu = dict(RECORD, device='cuda')                 # a card never owes the same bits (measured on an RX 580)
    assert sga.canary_tier(dict(gpu), gpu) == 'divergence'
    # another recorded precision: measured, never failed (a tf32 writer read by fp32)
    assert sga.canary_tier(dict(RECORD, device='cuda', precision='tf32'), RECORD) == 'precision'
    assert sga.canary_tier(dict(RECORD, precision='bf16'), RECORD) == 'precision'
    # a record that does not say its precision is no reason to waive the floor
    old = {k: v for k, v in RECORD.items() if k != 'precision'}
    assert sga.canary_tier(dict(old, device='cuda'), RECORD) == 'divergence'
    assert sga.canary_tier(dict(RECORD, device='cuda', precision='tf32'), old) == 'divergence'


def test_another_precision_is_measured_not_failed():
    """One tf32 writer's worst pair: Jaccard 0.925 against the CPU's fp32 -- a fail at fp32's floor, recorded here"""
    g = torch.Generator().manual_seed(2)
    a1, a2 = torch.randint(0, 500, (200, 2), generator=g), torch.randint(0, 500, (200, 2), generator=g)
    b2 = a2.clone()
    b2[:8] += 1000                                       # 16 of 208 rows differ: Jaccard 0.923
    fresh, stored = ((1., 1., 200), (a1, b2, torch.ones(200))), ((1., 1., 200), (a1, a2, torch.ones(200)))
    assert sga.canary_compare('matches', fresh, stored, 'divergence')[0] == 'warn'
    verdict, d = sga.canary_compare('matches', fresh, stored, 'precision')
    assert verdict == 'ok' and abs(d['jaccard'] - 192 / 208) < 1e-9 and d['rows_differ'] == 16
    X, C = torch.randn(H, W, 3, generator=g) + 5, torch.rand(H, W, generator=g) + 1.5     # every pixel trusted
    far = (X * 1.008, C, X, C)                           # pointmaps 7.6e-3 off, that writer's worst
    assert sga.canary_compare('pointmaps', far, (X, C, X, C), 'divergence')[0] == 'warn'
    assert sga.canary_compare('pointmaps', far, (X, C, X, C), 'precision')[0] == 'ok'


def test_matches_compare_as_a_set_and_the_score_within_its_tolerance():
    xy1, xy2, c = torch.tensor([[1, 2], [3, 4], [5, 6]]), torch.tensor([[7, 8], [9, 1], [2, 3]]), torch.tensor([1., 2., 3.])
    stored = ((1.5, 6.0, 3), (xy1, xy2, c))
    order = [2, 0, 1]
    fresh = ((1.5 * (1 + 1e-8), 6.0, 3), (xy1[order], xy2[order], c[order]))
    verdict, d = sga.canary_compare('matches', fresh, stored, 'bits')
    assert verdict == 'ok' and d['bit_equal'] and d['jaccard'] == 1.0
    moved = ((1.5, 6.0, 3), (xy1, torch.tensor([[7, 8], [9, 1], [2, 4]]), c))
    verdict, d = sga.canary_compare('matches', moved, stored, 'bits')
    assert verdict == 'warn' and not d['bit_equal'] and d['jaccard'] == 0.5     # 2 rows of 4: within the slack, never gross
    # divergence: one row crossing a threshold is noise on a card (2 rows differ: within CANARY_MATCH_SLACK_ROWS)...
    assert sga.canary_compare('matches', moved, stored, 'divergence')[0] == 'ok'
    assert sga.canary_compare('matches', moved, stored, 'derived')[0] == 'ok'
    # ...a set that moved is not: 200 rows, 10 replaced -> 20 differ, past both 4 rows and 1 % of the union
    g = torch.Generator().manual_seed(1)
    a1, a2 = torch.randint(0, 500, (200, 2), generator=g), torch.randint(0, 500, (200, 2), generator=g)
    b2 = a2.clone()
    b2[:10] += 1000
    many = ((1., 1., 200), (a1, b2, torch.ones(200)))
    verdict, d = sga.canary_compare('matches', many, ((1., 1., 200), (a1, a2, torch.ones(200))), 'divergence')
    assert verdict == 'warn' and d['rows_differ'] == 20
    off = ((1.5 * (1 + 1e-4), 6.0, 3), stored[1])
    assert sga.canary_compare('matches', off, stored, 'bits')[0] == 'warn'


def test_pointmaps_bits_and_divergence():
    g = torch.Generator().manual_seed(0)
    X, C = torch.randn(H, W, 3, generator=g) + 5, torch.rand(H, W, generator=g) + 1.5
    stored = (X, C, X.clone(), C.clone())
    assert sga.canary_compare('pointmaps', tuple(t.clone() for t in stored), stored, 'bits')[0] == 'ok'
    near = (X * (1 + 1e-6), C, X, C)
    assert sga.canary_compare('pointmaps', near, stored, 'bits')[0] == 'warn'
    assert sga.canary_compare('pointmaps', near, stored, 'divergence')[0] == 'ok'
    far = (X * 1.1, C, X, C)                              # a scale is the same shape: past the floor, not gross
    verdict, d = sga.canary_compare('pointmaps', far, stored, 'divergence')
    assert verdict == 'warn' and abs(d['pts_rel_max_trusted'] - 0.1) < 1e-6 and d['cos_min_trusted'] > 0.9999


def test_only_trusted_pixels_are_compared():
    """A view the network cannot see has confidence 1.0 everywhere, and its pointmap is a guess --
    not compared. Either side's confidence counts, so an entry that LOST its confidence is still compared."""
    g = torch.Generator().manual_seed(3)
    X = torch.randn(H, W, 3, generator=g) + 5
    sure, blind = torch.full((H, W), 2.0), torch.ones(H, W)
    stored = (X, sure, X, blind)
    fresh = (X, sure, X * 1.01, blind)                    # view 2 off by 1 %, and nobody trusts it
    verdict, d = sga.canary_compare('pointmaps', fresh, stored, 'divergence')
    assert verdict == 'ok' and d['trusted_px_frac'] == 0.0 and 'pts_rel' not in d['views'][1]
    assert d['views'][0]['pts_rel'] == 0.0 and d['pts_rel_max_trusted'] == 0.0
    lost = (X, sure, X * 1.01, sure)                      # the recompute trusts view 2: the stored entry lost it
    verdict, d = sga.canary_compare('pointmaps', lost, stored, 'divergence')
    assert verdict == 'warn' and d['views'][1]['trusted_frac'] == 1.0 and d['views'][1]['conf_rel'] == 1.0
    some = blind.clone()
    some[:1, :1] = 2.0                                    # 1 pixel of 256 trusted: under CANARY_TRUST_MIN_FRAC
    assert sga.canary_compare('pointmaps', (X, sure, X * 1.01, some), stored, 'divergence')[0] == 'ok'
    some[:4] = 2.0                                        # a quarter trusted: compared
    assert sga.canary_compare('pointmaps', (X, sure, X * 1.01, some), stored, 'divergence')[0] == 'warn'


def test_gross_is_gross_whatever_the_tier():
    g = torch.Generator().manual_seed(4)
    X, C = torch.randn(H, W, 3, generator=g) + 5, torch.full((H, W), 2.0)
    other = torch.randn(H, W, 3, generator=g)             # not the same values at all
    for tier in ('bits', 'divergence', 'precision'):
        verdict, d = sga.canary_compare('pointmaps', (other, C, X, C), (X, C, X, C), tier)
        assert verdict == 'gross' and d['cos_min_trusted'] < sga.CANARY_GROSS_COS, (tier, d)
    nan = X.clone()
    nan[0, 0, 0] = float('nan')                           # in the STORED entry: gross
    assert sga.canary_compare('pointmaps', (X, C, X, C), (nan, C, X, C), 'precision')[0] == 'gross'
    with pytest.raises(FloatingPointError, match='recompute'):            # in THIS box's recompute: its own fault
        sga.canary_compare('pointmaps', (nan, C, X, C), (X, C, X, C), 'divergence')
    assert sga.canary_compare('pointmaps', (X[:8], C[:8], X, C), (X, C, X, C), 'derived')[0] == 'gross'
    a1, a2 = torch.randint(0, 500, (200, 2), generator=g), torch.randint(0, 500, (200, 2), generator=g)
    stored = ((1., 1., 200), (a1, a2, torch.ones(200)))
    flipped = ((1., 1., 200), (a1, torch.flip(a2, [0]), torch.ones(200)))
    for tier in ('bits', 'divergence', 'precision'):
        assert sga.canary_compare('matches', flipped, stored, tier)[0] == 'gross', tier
    assert sga.canary_compare('matches', flipped, stored, 'derived')[0] == 'ok'     # a mask variant's rows differ
    few = ((1., 1., 3), (a1[:3], torch.flip(a2[:3], [0]), torch.ones(3)))
    assert sga.canary_compare('matches', few, ((1., 1., 3), (a1[:3], a2[:3], torch.ones(3))), 'divergence')[0] == 'ok'
    none = ((0., 0., 0), (torch.zeros(0, 2), torch.zeros(0, 2), torch.zeros(0)))
    verdict, d = sga.canary_compare('matches', none, none, 'divergence')
    assert verdict == 'ok' and d['jaccard'] == 1.0        # two empty sets are the same set


def test_a_hit_recomputed_matches_and_a_planted_entry_is_caught(canary):
    with tempfile.TemporaryDirectory() as cache:
        res = run(cache, 0.0)                                     # computes and writes the entries
        assert sga.LAST_FORWARD_PROFILE['computed_pairs'] == 1
        stored = {p: os.stat(p).st_mtime_ns for p in Path(cache).rglob('*.hvz')}
        res2 = run(cache, 1.0)                                    # every hit is a canary
        assert res2 == res and sga.LAST_FORWARD_PROFILE['canary_pairs'] == 1
        assert sga.LAST_FORWARD_PROFILE['computed_pairs'] == 0
        assert {p: os.stat(p).st_mtime_ns for p in Path(cache).rglob('*.hvz')} == stored    # nothing written
        results = sga.CANARY_RECORD['results']
        assert sorted(r['kind'] for r in results) == ['matches', 'pointmaps', 'pointmaps']
        assert all(r['verdict'] == 'ok' and r['tier'] == 'bits' and r['detail']['bit_equal'] for r in results), results
        assert SUNK == results and sga.LAST_FORWARD_PROFILE['canary_fresh_pairs'] == 0
        # the same pair is checked once per process
        run(cache, 1.0)
        assert sga.LAST_FORWARD_PROFILE['canary_pairs'] == 0
        # PLANTED: other matches under the same key, the record left as it was (a key that missed an input)
        path = corres_of(res)
        score, (xy1, xy2, c) = sga._cache_load(path, 'cpu')
        sga._save_atomic((score, (xy1, torch.flip(xy2, [0]), c)), path)
        planted = {p: os.stat(p).st_mtime_ns for p in Path(cache).rglob('*.hvz')}
        sga.CANARY_RECORD['done'].clear()
        prov = {'record': dict(RECORD), 'require': None, 'vid': 'v', 'reads': []}
        pairs = [({'instance': 'view-a'}, {'instance': 'view-b'})]
        with patch.object(sga, 'CANARY_RATE', 1.0), patch.object(sga, 'symmetric_inference', fake_inference):
            res3, _ = sga.forward_mast3r(pairs, object(), cache, device='cpu', subsample=4, provenance=prov)
        (f,) = [r for r in sga.CANARY_RECORD['results'] if r['verdict'] != 'ok']          # a warning, no stop
        assert f['kind'] == 'matches' and f['check'] == 'bits' and f['verdict'] == 'gross' and f['writer']['compute'] == COMPUTE
        assert f['file'] == os.path.relpath(path, cache) and f in SUNK
        # THE PAIR IS SOLVED ON THE RECOMPUTE, from the run's own dir; the cache is not touched
        assert {p: os.stat(p).st_mtime_ns for p in Path(cache).rglob('*.hvz')} == planted
        (paths, corres), = res3.values()
        fresh = sga.CANARY_FRESH_ROOT
        assert all(p.startswith(fresh + os.sep) for p in (*paths, corres)), (paths, corres)
        assert sga.LAST_FORWARD_PROFILE['canary_fresh_pairs'] == 1
        for got, want in zip(paths + (corres,), res[('view-a', 'view-b')][0] + (corres_of(res),)):
            if got != corres:
                assert all(torch.equal(torch.as_tensor(x), torch.as_tensor(y))
                           for x, y in zip(sga._cache_load(got, 'cpu'), sga._cache_load(want, 'cpu')))
        _s, (fx1, fx2, _c) = sga._cache_load(corres, 'cpu')
        assert torch.equal(torch.as_tensor(fx2), torch.as_tensor(xy2))       # the matches as first computed
        assert sga._prov_read(corres)['compute'] == COMPUTE
        (read,) = prov['reads']
        assert read[2] == 'canary_fresh' and read[3] is True
        # A LATER CALL IN THE SAME RUN (an adaptive round, the solve's own pass): the pair is not checked again, and
        # reads the recompute -- never the stored entry that did not pass
        prov2 = {'record': dict(RECORD), 'require': None, 'vid': 'v', 'reads': []}
        with patch.object(sga, 'CANARY_RATE', 1.0), patch.object(sga, 'symmetric_inference', fake_inference):
            res4, _ = sga.forward_mast3r(pairs, object(), cache, device='cpu', subsample=4, provenance=prov2)
        assert res4 == res3 and sga.LAST_FORWARD_PROFILE['canary_pairs'] == 0
        assert sga.LAST_FORWARD_PROFILE['computed_pairs'] == 0
        (read,) = prov2['reads']
        assert read[2] == 'canary_fresh' and read[4] == os.stat(paths[0]).st_mtime_ns


def test_another_writer_is_held_to_divergence_and_a_derived_one_only_measured(canary):
    with tempfile.TemporaryDirectory() as cache:
        res = run(cache, 0.0, record=dict(RECORD, compute=dict(COMPUTE, device_name='another box')))
        path = corres_of(res)
        score, (xy1, xy2, c) = sga._cache_load(path, 'cpu')
        sga._save_atomic((score, (xy1, torch.flip(xy2, [0]), c)), path)
        run(cache, 1.0)
        assert [(f['check'], f['tier'], f['verdict']) for f in sga.CANARY_RECORD['results'] if f['verdict'] != 'ok'] \
            == [('divergence', 'divergence', 'gross')]
        rec = sga._prov_read(path)
        sga._prov_write(path, dict(rec, derived_from='matching/other-variant'))
        sga.CANARY_RECORD.update(done=set(), results=[], fresh={})              # another process: checked anew
        run(cache, 1.0)
        assert any(r['tier'] == 'derived' and r['verdict'] == 'ok' for r in sga.CANARY_RECORD['results'])
        assert sga.LAST_FORWARD_PROFILE['canary_fresh_pairs'] == 0


def test_a_tf32_writer_is_measured_and_only_a_gross_one_is_flagged(canary):
    """A CUDA tf32 writer's entries, read by an fp32 CPU run, differ by TF32's gap -- measured,
    'ok'. A set that is not the same matches at all is 'gross' from any writer"""
    with tempfile.TemporaryDirectory() as cache:
        res = run(cache, 0.0, record=dict(RECORD, device='cuda', kernels='cuda', precision='tf32'))
        path = corres_of(res)
        score, (xy1, xy2, c) = sga._cache_load(path, 'cpu')
        n = len(xy1)
        near = torch.as_tensor(xy2).clone()
        near[: max(1, n // 10)] += 1                                          # ~10 % of the rows moved: TF32's gap
        sga._save_atomic((score, (xy1, near, c)), path)
        run(cache, 1.0)
        res_ = [r for r in sga.CANARY_RECORD['results'] if r['kind'] == 'matches']
        assert [(r['tier'], r['verdict']) for r in res_] == [('precision', 'ok')] and res_[0]['detail']['jaccard'] < 1
        sga._save_atomic((score, (xy1, torch.flip(xy2, [0]), c)), path)      # far past any floor
        sga.CANARY_RECORD.update(done=set(), results=[], fresh={})
        run(cache, 1.0)
        res_ = [r for r in sga.CANARY_RECORD['results'] if r['kind'] == 'matches']
        assert [(r['tier'], r['verdict']) for r in res_] == [('precision', 'gross')] and res_[0]['detail']['jaccard'] < 0.5


def test_the_mask_check_needs_no_model_and_measures_until_told_to_warn(canary, monkeypatch):
    ma = np.zeros((H, W), bool)
    ma[:6] = True                                                 # view a's top six rows are masked
    masks = {'view-a': ma, 'view-b': None}                        # view b: nothing masks it
    monkeypatch.setattr(sga, 'CANARY_MASK_CHECK', True)
    with tempfile.TemporaryDirectory() as cache:
        res = run(cache, 0.0, masks=masks)                        # matched under the mask: none on it
        path = corres_of(res)
        score, (xy1, xy2, c) = sga._cache_load(path, 'cpu')
        assert len(xy1) and int(torch.as_tensor(xy1)[:, 1].min()) >= 6
        run(cache, 0.0, masks=masks, model=None)                  # a hit: checked, clean
        assert sga.CANARY_RECORD['mask_checked'] == 1 and sga.CANARY_RECORD['mask_on'] == []
        # PLANTED: a match deep inside view a's mask (row 1, past the eroded rim)
        bad = torch.cat([torch.as_tensor(xy1), torch.tensor([[8, 1]])])
        sga._save_atomic((score, (bad, torch.cat([torch.as_tensor(xy2), torch.tensor([[8, 8]])]),
                                  torch.cat([torch.as_tensor(c), torch.tensor([1.])]))), path)
        run(cache, 0.0, masks=masks, model=None)                  # measured, no warning
        (m,) = sga.CANARY_RECORD['mask_on']
        assert m['verdict'] == 'ok' and m['detail']['on_mask'] == [1, 0] and m['check'] == 'mask' and SUNK == [m]
        monkeypatch.setattr(sga, 'CANARY_MASK_WARN', 0.0)
        res2 = run(cache, 0.0, masks=masks, model=None)           # a warning; the hit is still read (no model)
        assert [f['verdict'] for f in sga.CANARY_RECORD['mask_on']] == ['ok', 'warn'] and res2 == res


def test_a_canary_without_a_model_or_a_fresh_root_is_refused(canary, monkeypatch):
    with tempfile.TemporaryDirectory() as cache:
        with pytest.raises(ValueError, match='needs the model'):
            run(cache, 0.5, model=None)
        monkeypatch.setattr(sga, 'CANARY_FRESH_ROOT', None)
        with pytest.raises(ValueError, match='CANARY_FRESH_ROOT'):
            run(cache, 0.5)
        run(cache, 0.0)                                           # no rate, no root needed


def run_many(cache, rate, views, cap=None, record=RECORD):
    """forward_mast3r over consecutive pairs of `views`, at `rate` and `cap`, writing as `record`"""
    pairs = [({'instance': a}, {'instance': b}) for a, b in zip(views, views[1:])]
    prov = {'record': dict(record), 'require': None, 'vid': 'v', 'reads': []}
    with patch.object(sga, 'CANARY_RATE', rate), patch.object(sga, 'CANARY_CAP', cap), \
            patch.object(sga, 'symmetric_inference', fake_inference):
        sga.forward_mast3r(pairs, object(), cache, device='cpu', subsample=4, provenance=prov)
    return [f"{sga.hash_md5(a['instance'])}-{sga.hash_md5(b['instance'])}" for a, b in pairs]


def _checked(views=None):
    """the pairs (view-key names) whose matches were canaried"""
    return {f"{sga.hash_md5(r['views'][0])}-{sga.hash_md5(r['views'][1])}"
            for r in sga.CANARY_RECORD['results'] if r['kind'] == 'matches'}


def test_the_cap_checks_the_smallest_draws_not_the_first_pairs(canary):
    """at most CANARY_CAP a run, and the pairs with the smallest draw values -- spread over the run, not its start"""
    views = [f'view-{i:02d}' for i in range(13)]
    with tempfile.TemporaryDirectory() as cache:
        keys = run_many(cache, 0.0, views)                               # 12 pairs, now all hits
        smallest = set(sorted(keys, key=sga.canary_value)[:3])
        assert smallest != set(keys[:3]), 'the salt must make "smallest" differ from "first" for this to test it'
        run_many(cache, 1.0, views, cap=3)
        assert _checked() == smallest
        assert sga.CANARY_RECORD['cap'] == [dict(cap=3, pairs=12, drawn=12, picked=3, writers_read=0,
                                                 picked_cautious=0)]
        assert all(r['verdict'] == 'ok' for r in sga.CANARY_RECORD['results'])


def test_the_cap_is_the_runs_across_calls_and_within_the_rate(canary):
    views = [f'view-{i:02d}' for i in range(13)]
    with tempfile.TemporaryDirectory() as cache:
        run_many(cache, 0.0, views)
        run_many(cache, 1.0, views[:7], cap=4)                           # the first call takes what it draws, 4
        run_many(cache, 1.0, views[6:], cap=4)                           # the second has nothing left
        assert len(_checked()) == 4
        assert [c['picked'] for c in sga.CANARY_RECORD['cap']] == [4, 0]
        sga.CANARY_RECORD.update(results=[], done=set(), cap=[])
        keys = run_many(cache, 0.3, views, cap=100)                      # a cap above the draw: the rate decides
        assert _checked() == {k for k in keys if sga.canary_value(k) < 0.3}


def test_no_cap_is_the_rate_alone_and_a_bad_cap_is_refused(canary):
    views = [f'view-{i:02d}' for i in range(7)]
    with tempfile.TemporaryDirectory() as cache:
        run_many(cache, 0.0, views)
        run_many(cache, 1.0, views)
        assert len(_checked()) == 6 and 'cap' not in sga.CANARY_RECORD
        for bad in (0, -1, 2.5, True, '8'):
            with pytest.raises(ValueError, match='CANARY_CAP'):
                run_many(cache, 1.0, views, cap=bad)


OTHER = dict(RECORD, compute=dict(COMPUTE, device_name='another box'))    # another writer: another variant


def _variant(w):
    """the test's writer variant: the record's box, as recon_canary.variant_id separates boxes"""
    return 'unrecorded' if w is None else 'box:' + w['compute']['device_name']


def _ledger(monkeypatch, settled, cautious):
    monkeypatch.setattr(sga, 'CANARY_SETTLED', frozenset(settled))
    monkeypatch.setattr(sga, 'CANARY_CAUTIOUS_RATE', cautious)
    monkeypatch.setattr(sga, 'CANARY_VARIANT_OF', _variant)


def test_a_settled_writer_is_drawn_at_the_base_rate_any_other_cautiously(canary, monkeypatch):
    """the bench's ledger: who wrote a hit decides how often it is checked; a pair is cautious if ANY of
    its files is from a writer the ledger has not settled"""
    views = [f'view-{i:02d}' for i in range(13)]
    with tempfile.TemporaryDirectory() as cache:
        keys = run_many(cache, 0.0, views[:7])                          # pairs 0-5 by this box
        keys += run_many(cache, 0.0, views[6:], record=OTHER)           # pairs 6-11 by another
        _ledger(monkeypatch, {'box:test-cpu'}, 1.0)                     # this box settled, the other not
        run_many(cache, 1e-9, views)                                    # base rate ~0: only the cautious band draws
        assert _checked() == set(keys[6:])                              # every pair of the unsettled writer, no other
        sga.CANARY_RECORD.update(results=[], done=set(), fresh={})
        _ledger(monkeypatch, set(), 1.0)                                # nothing settled: every pair cautious
        run_many(cache, 1e-9, views)
        assert _checked() == set(keys)
        sga.CANARY_RECORD.update(results=[], done=set(), fresh={})
        _ledger(monkeypatch, {'box:test-cpu', 'box:another box'}, 1.0)  # both settled: the base rate alone
        run_many(cache, 1e-9, views)
        assert _checked() == set()


def test_under_the_cap_an_unsettled_writer_comes_first(canary, monkeypatch):
    """by draw value alone every base-rate draw precedes every cautious one, and on a big run the base draws fill
    the cap; by value OVER rate the cautious pairs are preferred, and a pair is still drawn only below its rate"""
    views = [f'view-{i:02d}' for i in range(13)]
    with tempfile.TemporaryDirectory() as cache:
        keys = run_many(cache, 0.0, views[:7]) + run_many(cache, 0.0, views[6:], record=OTHER)
        _ledger(monkeypatch, {'box:test-cpu'}, 1.0)
        rate = 0.5                                                      # half the settled pairs draw too
        rate_of = {k: (rate if i < 6 else 1.0) for i, k in enumerate(keys)}
        drawn = sorted((sga.canary_value(k) / rate_of[k], k) for k in keys if sga.canary_value(k) < rate_of[k])
        want = {k for _w, k in drawn[:3]}
        by_value = set(sorted(keys, key=sga.canary_value)[:3])
        assert want != by_value, 'the salt must make the two orders differ for this to test it'
        run_many(cache, rate, views, cap=3)
        assert _checked() == want
        (call,) = sga.CANARY_RECORD['cap']
        assert call['picked'] == 3 and call['picked_cautious'] == sum(1 for k in want if rate_of[k] == 1.0)
        assert call['writers_read'] == 12                               # every pair under the ceiling, here all


def test_a_half_set_ledger_is_refused(canary, monkeypatch):
    with tempfile.TemporaryDirectory() as cache:
        run(cache, 0.0)
        monkeypatch.setattr(sga, 'CANARY_SETTLED', frozenset({'x'}))
        with pytest.raises(ValueError, match='CANARY_CAUTIOUS_RATE'):
            run(cache, 0.5)
        monkeypatch.setattr(sga, 'CANARY_CAUTIOUS_RATE', 0.1)
        with pytest.raises(ValueError, match='CANARY_VARIANT_OF'):
            run(cache, 0.5)
        monkeypatch.setattr(sga, 'CANARY_SETTLED', ['x'])
        with pytest.raises(ValueError, match='frozenset'):
            run(cache, 0.5)
