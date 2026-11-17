"""A reference to a canonical value records its CACHE_EXT name in the root that holds it, and every reader
resolves it where it reads it (sparse_ga _same_root_name / _found): a .pth that goes once its .hvz is
published must not leave a recorded name pointing at nothing (capture_solve, the optimizer phase's request).
The end-to-end check -- a read-root hit, the optimizer child, shared intrinsics, a drop, a replay of an
old capture -- is recorded in the commit; this pins the rule."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mast3r.cloud_opt import sparse_ga as sga

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _standin_codec  # noqa: E402

_standin_codec.install()


class CanonicalNames(unittest.TestCase):
    def test_a_found_pth_is_recorded_as_the_hvz_beside_it(self):
        self.assertEqual(sga._same_root_name("/mnt/archive/canon_views/x@s.pth"), "/mnt/archive/canon_views/x@s.hvz")
        self.assertEqual(sga._same_root_name("/srv/cache/canon_views/x@s.hvz"), "/srv/cache/canon_views/x@s.hvz")

    def test_the_recorded_name_reads_before_and_after_the_drop(self):
        with tempfile.TemporaryDirectory() as d:
            pth = os.path.join(d, "canon_views", "x@s.pth")
            os.makedirs(os.path.dirname(pth))
            value = ((torch.ones(2, 2), torch.ones(2, 2), torch.full((2, 2), 3.0)), torch.tensor(5.0))
            torch.save(value, pth)
            name = sga._same_root_name(pth)
            self.assertEqual(sga._cache_load(sga._found(name), "cpu")[1], 5.0, "only the .pth is there yet")
            sga._save_atomic(value, name)          # the converter publishes the .hvz ...
            os.remove(pth)                         # ... and the .pth is dropped
            self.assertEqual(sga._cache_load(sga._found(name), "cpu")[1], 5.0)
            self.assertEqual(sga._cache_load(sga._found(pth), "cpu")[1], 5.0,
                             "a name recorded before the fix (.pth) resolves too")


if __name__ == "__main__":
    unittest.main()
