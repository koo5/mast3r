"""sparse_ga.PAIR_TIMING: one record per pair the call inferred, after it is complete -- the device's forward time apart
from matching and saves; nothing for a cache hit; None (the default) runs as before and writes the same entries.
(The same entries by NAME: two runs of this stand-in model already differ in bytes, hook or not.)"""
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
PAIRS = [({'instance': 'view-a'}, {'instance': 'view-b'}), ({'instance': 'view-b'}, {'instance': 'view-c'})]


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
    with patch.object(sga, 'PAIR_TIMING', hook), patch.object(sga, 'symmetric_inference', fake_inference):
        res, _ = sga.forward_mast3r(PAIRS, object(), cache, device='cpu', subsample=4, provenance=prov)
    return res


def _files(cache):
    return sorted(str(p.relative_to(cache)) for p in Path(cache).rglob('*') if p.is_file() and '.lock' not in p.name)


def test_every_inferred_pair_is_reported_once_with_its_times():
    seen = []
    with tempfile.TemporaryDirectory() as cache:
        run(cache, seen.append)
    assert len(seen) == len(PAIRS), seen
    for r in seen:
        assert set(r) == {'forward_s', 'post_s', 'match_s', 'computed', 'canary', 'idx1', 'idx2', 'post_threads'}, r
        assert r['post_threads'] == 0                    # sequential here: no pipeline
        assert r['forward_s'] >= 0 and r['post_s'] >= 0 and r['match_s'] is not None and r['match_s'] >= 0
        assert r['computed'] is True and r['canary'] is False
    assert sorted((r['idx1'], r['idx2']) for r in seen) == sorted(
        (sga.hash_md5(a['instance']), sga.hash_md5(b['instance'])) for a, b in PAIRS)
    json.dumps(seen)                                     # a record goes over a socket as JSON


def test_a_cache_hit_is_not_reported():
    with tempfile.TemporaryDirectory() as cache:
        run(cache, None)
        seen = []
        run(cache, seen.append)
    assert seen == [], seen


def test_no_hook_writes_the_same_entries():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        run(a, None)
        run(b, lambda r: None)
        assert _files(a) == _files(b)
