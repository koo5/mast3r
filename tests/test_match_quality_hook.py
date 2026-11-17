"""sparse_ga.MATCH_QUALITY: a pair's matches scored as they are written, into THAT entry's record -- never a key,
never a pointmap's record; None (the default) writes records as before."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga as sga  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _standin_codec  # noqa: E402

_standin_codec.install()
H = W = 16
RECORD = {'checkpoint': 'c', 'forward_version': sga.FORWARD_VERSION, 'device': 'cpu', 'precision': 'fp32'}


def _g(key):
    return torch.Generator().manual_seed(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))


def fake_inference(model, img1, img2, device=None, encodings=None):
    a, b = img1['instance'], img2['instance']
    return [{'pts3d': torch.randn(1, H, W, 3, generator=_g(k)) + 5, 'conf': torch.rand(1, H, W, generator=_g(k)) + 1,
             'desc': torch.nn.functional.normalize(torch.randn(1, H, W, 8, generator=_g(v)), dim=-1),
             'desc_conf': torch.rand(1, H, W, generator=_g(k + 'q')) + 1}
            for v, k in ((a, f'{a}|{b}|1'), (b, f'{a}|{b}|2'), (b, f'{b}|{a}|1'), (a, f'{b}|{a}|2'))]


def run(cache, hook):
    prov = {'record': dict(RECORD), 'require': None, 'vid': 'v', 'reads': []}
    with patch.object(sga, 'MATCH_QUALITY', hook), patch.object(sga, 'symmetric_inference', fake_inference):
        res, _ = sga.forward_mast3r([({'instance': 'view-a'}, {'instance': 'view-b'})], object(), cache,
                                    device='cpu', subsample=4, provenance=prov)
    ((p1, p2), pc), = res.values()
    return [json.load(open(p + sga.PROV_SUFFIX)) for p in (p1, p2, pc)]


def test_the_entrys_record_carries_its_scores():
    seen = []

    def hook(X11, X21, corres):
        seen.append((tuple(X11.shape), tuple(X21.shape), len(corres[0]), str(X11.device)))
        return {'v': 1, 'n': len(corres[0])}
    with tempfile.TemporaryDirectory() as cache:
        r1, r2, rc = run(cache, hook)
    (x11, x21, n, dev), = seen
    assert x11 == x21 == (H, W, 3) and n > 0 and dev == 'cpu'
    assert rc['match_quality'] == {'v': 1, 'n': n} and rc['checkpoint'] == 'c'
    assert 'match_quality' not in r1 and 'match_quality' not in r2          # pointmaps: the writer alone
    assert sga.LAST_FORWARD_PROFILE['parts_s']['match_quality'] >= 0


def test_no_hook_writes_records_as_before():
    with tempfile.TemporaryDirectory() as cache:
        records = run(cache, None)
    assert all('match_quality' not in r and r['checkpoint'] == 'c' for r in records)


def test_a_failing_scorer_writes_nothing():
    """scored BEFORE the data is written: a scorer that raises leaves no matches entry without its record"""
    import pytest

    def broken(X11, X21, corres):
        raise ValueError('the scorer broke')
    with tempfile.TemporaryDirectory() as cache:
        with pytest.raises(ValueError, match='the scorer broke'):
            run(cache, broken)
        written = [p for p in Path(cache).rglob('*') if p.is_file() and 'corres' in str(p)]
        assert written == [], written
