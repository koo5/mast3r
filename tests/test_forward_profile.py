"""Warm profiling preserves cache results without invoking model inference."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mast3r.cloud_opt import sparse_ga as sga
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _standin_codec  # noqa: E402

# Cache entries need sparse_ga's codec and content digest (hillview's hvz.install sets the real ones);
# these tests are about the cache paths, so they set a stand-in explicitly (_standin_codec).
_standin_codec.install()


def test_forward_profile():
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        pairs = [({'instance': 'image-a'}, {'instance': 'image-b'})]
        a, b = map(sga.hash_md5, ('image-a', 'image-b'))
        for x, y in ((a, b), (b, a)):
            path = root / 'forward' / x / (y + '.hvz')
            path.parent.mkdir(parents=True, exist_ok=True)
            sga._save_atomic((torch.ones(1),) * 4, str(path))
        reverse = root / 'corres_conf=desc_conf_subsample=8' / (b + '-' + a + '.hvz')
        reverse.parent.mkdir(parents=True)
        xy1, xy2 = torch.tensor([[1, 2]]), torch.tensor([[3, 4]])
        sga._save_atomic(((1., 2., 1), (xy1, xy2, torch.ones(1))), str(reverse))
        with patch.object(sga, 'symmetric_inference', side_effect=AssertionError('warm inference')):
            paths, returned_root = sga.forward_mast3r(pairs, object(), scratch, device='cpu')
        assert returned_root == scratch and len(paths) == 1
        _, (x1, x2, _) = torch.load(paths['image-a', 'image-b'][1])
        assert torch.equal(x1, xy2) and torch.equal(x2, xy1)
        profile = dict(sga.LAST_FORWARD_PROFILE)
        assert profile['cached_pairs'] == 1 and profile['computed_pairs'] == 0
        assert profile['skipped_pairs'] == 0
        assert profile['wall_s'] >= sum(profile['parts_s'].values())
        with patch.object(sga.torch, 'load', side_effect=AssertionError('warm tensor load')):
            again, _ = sga.forward_mast3r(pairs, object(), scratch, device='cpu')
        assert again == paths
        sga.forward_mast3r([], None, scratch, device='cpu')
        assert sga.LAST_FORWARD_PROFILE['pairs'] == 0
        sga.forward_mast3r([({'instance': 'missing-a'}, {'instance': 'missing-b'})], None,
                          scratch, device='cpu')
        assert sga.LAST_FORWARD_PROFILE['skipped_pairs'] == 1
        assert sga.LAST_FORWARD_PROFILE['cached_pairs'] == 0
    print('PASS: actual-import forward profiling, transpose, warm no-load, empty and skipped')


if __name__ == '__main__':
    test_forward_profile()
