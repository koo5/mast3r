"""Ordered pair indexing must preserve existing canonical identities and live validation."""
import os
import sys
import tempfile
import unittest
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
from mast3r.cloud_opt.canonical_cache import cache_path, index_view_pairs


class OnceOnlyPairs(dict):
    def __init__(self, pairs):
        super().__init__(pairs)
        self.traversals = 0
    def items(self):
        self.traversals += 1
        if self.traversals > 1:
            raise AssertionError('canonical preparation rescanned the whole pair collection')
        return super().items()


class CanonicalIncidenceTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.images = ['image-a', 'image-b', 'image-c']
        self.keys = dict(zip(self.images, ['a' * 32, 'b' * 32, 'c' * 32]))
        y, x = torch.meshgrid(torch.arange(16), torch.arange(24), indexing='ij')
        points = torch.stack([(x - 12) / 8, (y - 8) / 8, torch.ones_like(x) * 3], -1).float()
        confidence = torch.ones((16, 24)) * 2
        xy = torch.tensor([[4, 4], [8, 8], [12, 12]])
        self.pairs = {}
        for a, b in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
            paths = []
            for first, second in ((a, b), (b, a)):
                path = self.root / 'forward' / self.keys[self.images[first]] / (self.keys[self.images[second]] + '.hvz')
                path.parent.mkdir(parents=True, exist_ok=True)
                sga._save_atomic((points + first * .01, confidence, points + second * .02, confidence), str(path))
                paths.append(str(path))
            corres = self.root / 'corres' / f'{a}-{b}.hvz'
            corres.parent.mkdir(exist_ok=True)
            sga._save_atomic((torch.tensor([1., 2., 3.]), (xy, xy, torch.tensor([1., 2., 3.]))), str(corres))
            self.pairs[self.images[a], self.images[b]] = (tuple(paths), str(corres))

    def key(self, img, pairs, indexed=None):
        kwargs = {} if indexed is None else {'view_pairs': indexed}
        return cache_path(str(self.root), img, pairs, self.keys, 4, 'cpu', {'mode': 'avg-angle'}, **kwargs)

    def prepare(self, pairs, local='local'):
        return sga.prepare_canonical_data(self.images, pairs, subsample=4, device='cpu', mode='avg-angle',
               cache_path=str(self.root / local), shared_cache=str(self.root), image_keys=self.keys)

    def test_keys_equal_full_scan_for_each_view_including_self_pair(self):
        a = self.images[0]
        p1, p2 = self.pairs[a, self.images[1]][0]
        self.pairs[a, a] = ((p1, p2), self.pairs[a, self.images[1]][1])
        indexed = index_view_pairs(self.pairs)
        for img in self.images:
            self.assertEqual(self.key(img, self.pairs), self.key(img, self.pairs, indexed[img]))
            expected = [item for item in self.pairs.items() if img in item[0]]
            self.assertEqual(indexed[img], expected)
        self.assertEqual(sum(img == a for pair, _ in indexed[a] for img in pair), 6)

    def test_partner_multiplicity_and_reduction_order_remain_part_of_key(self):
        a = self.images[0]
        original = self.key(a, self.pairs, index_view_pairs(self.pairs)[a])
        reversed_pairs = dict(reversed(list(self.pairs.items())))
        self.assertNotEqual(original, self.key(a, reversed_pairs, index_view_pairs(reversed_pairs)[a]))
        fewer = {pair: paths for pair, paths in self.pairs.items() if pair != (self.images[1], a)}
        self.assertNotEqual(original, self.key(a, fewer, index_view_pairs(fewer)[a]))

    def test_cold_and_warm_prepare_traverse_pairs_once(self):
        for local in ('cold', 'warm'):
            with self.subTest(local=local):
                pairs = OnceOnlyPairs(self.pairs)
                result = self.prepare(pairs, local)
                self.assertIs(result[0], pairs)
                self.assertEqual(pairs.traversals, 1)

    def test_a_value_written_as_pth_before_step_2_is_a_warm_hit(self):
        # with two containers: canonical values (and their predictions) made before .hvz are .pth; the key is the
        # same (canon-v3 is container-blind), so a run finds them and recomputes nothing
        cold = self.prepare(self.pairs, 'pth-cold')
        for value in cold[3]:
            os.rename(value, value[:-len('.hvz')] + '.pth')
            os.rename(value + '.predictions.hvz', value[:-len('.hvz')] + '.pth.predictions.pth')
        with patch.object(sga, 'canonical_view', side_effect=AssertionError('a .pth value was missed')):
            warm = self.prepare(OnceOnlyPairs(self.pairs), 'pth-warm')
        # recorded as the CACHE_EXT name in the root that holds it, read from the .pth that is there
        # (sparse_ga _same_root_name): a recorded .pth would name nothing once it is dropped
        self.assertTrue(all(v.endswith('.hvz') for v in warm[3]), warm[3])
        self.assertTrue(all(sga._found(v).endswith('.pth') for v in warm[3]), warm[3])
        self.assertTrue(torch.equal(cold[1], warm[1]))

    def test_self_pair_warm_payload_preserves_two_predictions(self):
        a = self.images[0]
        self.pairs[a, a] = self.pairs[a, self.images[1]]
        cold = self.prepare(self.pairs, 'self-cold')
        predictions = torch.load(cold[3][0] + '.predictions.hvz')
        self.assertEqual(len(predictions), 6)
        with patch.object(sga, 'canonical_view', side_effect=AssertionError('self-pair warm miss')):
            warm = self.prepare(OnceOnlyPairs(self.pairs), 'self-warm')
        self.assertEqual(cold[3], warm[3])
        self.assertTrue(torch.equal(cold[1], warm[1]))

    def test_revalidation_reads_fresh_digests_and_rejects_lost_spill(self):
        a = self.images[0]
        indexed = index_view_pairs(self.pairs)[a]
        original = self.key(a, self.pairs, indexed)
        forward = Path(self.pairs[a, self.images[1]][0][0])
        payload = torch.load(forward)
        sga._save_atomic(tuple(value + .001 for value in payload), str(forward))
        self.assertNotEqual(original, self.key(a, self.pairs, indexed))
        forward.unlink()
        forward.symlink_to(self.root / 'missing-spill')
        self.assertIsNone(self.key(a, self.pairs, indexed))

    def test_rewrite_during_warm_read_recomputes_without_publishing_stale_identity(self):
        original = self.prepare(self.pairs, 'seed')
        forward = self.pairs[self.images[0], self.images[1]][0][0]
        real_load = torch.load
        rewritten = False
        def load(path, *args, **kwargs):
            nonlocal rewritten
            result = real_load(path, *args, **kwargs)
            if str(path) == original[3][0] and not rewritten:
                rewritten = True
                payload = real_load(forward)
                sga._save_atomic(tuple(value + .001 for value in payload), forward)
            return result
        with patch.object(sga.torch, 'load', side_effect=load), \
             patch.object(sga, 'canonical_view', wraps=sga.canonical_view) as compute:
            result = self.prepare(OnceOnlyPairs(self.pairs), 'rewritten')
        self.assertTrue(rewritten)
        self.assertGreater(compute.call_count, 0)
        self.assertTrue(str(self.root / 'rewritten') in result[3][0])
        self.assertNotEqual(result[3][0], original[3][0])


if __name__ == '__main__':
    unittest.main()
