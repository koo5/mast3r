"""hillview: the pair-geometry record sparse_scene_optimizer writes (pair_geometry_path). Moved here with its writer
from hillview's scripts/enrich/recon_pair_geometry.py: the records are this module's own, and a
stock call imports nothing of the bench's."""
import gzip
import json
from collections import namedtuple
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt.pair_geometry import comparisons, write_geometry  # noqa: E402

class PairGeometryTest(unittest.TestCase):
 def test_exact_matched_positions_and_pointmap_transform(self):
  Pair = namedtuple('Pair', 'img1 img2 slice1 slice2 pix1 pix2 confs')
  xy = np.array([[5., 6.], [7., 8.]])
  pair = Pair(0, 1, slice(0, 2), slice(0, 2), xy, xy + 2, np.ones(2))
  cameras = np.repeat(np.eye(4)[None], 2, axis=0)
  cameras[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
  cameras[1, :3, 3] = [3, 4, 5]
  points = [np.array([[1., 0., 0.], [2., 0., 0.]]), np.array([[2., 1., 0.], [3., 1., 0.]])]
  predictions = {'b': {'a': (np.array([[1., 0., 0.], [0., 1., 0.]]), np.ones(2))}}
  with tempfile.TemporaryDirectory() as d:
   path = str(Path(d) / 'pair_geometry.json.gz')
   # the pointmap source is the ANCHOR's point at the anchor's pixel, not pts3d at the correspondence
   anchor_pts = np.array([[1., 0., 0.5], [2., 0., 0.5]]); anchor_xy = np.array([[4., 4.], [4., 4.]])
   write_geometry(path, dict(pts3d=points, cam2w=cameras), [[32, 24], [32, 24]], [pair], [pair], predictions, ['a', 'b'], .01,
                  {0: (anchor_pts, anchor_xy)})
   with gzip.open(path, 'rt') as f: data = json.load(f)
  row = data['pairs']['0-1']
  self.assertEqual(row['matched'][0]['target'], [2., 1., 0.])
  self.assertEqual(row['matched'][0]['pixel_b'], [7., 8.])
  self.assertEqual(row['pointmap'][0]['target'], [3., 5., 5.])
  self.assertEqual(row['pointmap'][1]['target'], [2., 4., 5.])
  self.assertIsNone(row['pointmap'][0]['pixel_b'])
  self.assertEqual(row['pointmap'][0]['source'], [1., 0., 0.5])
  self.assertEqual(row['pointmap'][0]['pixel_a'], [4., 4.])
  self.assertNotIn('1-0', data['pairs'])
  self.assertEqual(row['pointmap_weight'], .01)
  self.assertEqual(data['version'], 2)
 def test_sampling_filters_invalid_rows_without_inventing_matches(self):
  source = np.zeros((100, 3)); source[0, 0] = np.nan
  target = np.ones((100, 3)); xy = np.zeros((100, 2)); confidence = np.ones(100); confidence[1] = 0
  samples = comparisons(source, target, xy, None, confidence)
  self.assertEqual(len(samples), 64)
  self.assertTrue(all(s['pixel_b'] is None for s in samples))
  self.assertTrue(all(abs(s['distance'] - np.sqrt(3)) < 1e-9 for s in samples))
 def test_mismatched_arrays_refuse(self):
  with self.assertRaises(ValueError): comparisons(np.zeros((2,3)), np.zeros((1,3)), np.zeros((2,2)), None, np.ones(2))
if __name__ == '__main__': unittest.main()
