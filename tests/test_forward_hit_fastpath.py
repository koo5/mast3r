"""Actual-import checks of the read-only hit path and the existing serialized writers."""
import os
import sys
import multiprocessing
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
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


def process_forward(root, started, waiter_entering, release, output, waiter):
    """Use independent interpreters and real flock; stub only expensive inference/matching."""
    pairs = [({'instance': 'image-a'}, {'instance': 'image-b'})]
    original_lock = sga._PairLock
    class ObservedLock(original_lock):
        # acquire, not __enter__: since 67b410f a pair's lock is TRIED (acquire(False), a held pair set aside) and
        # waited on only when nothing else is left (acquire(True)); __enter__ is acquire(True) too
        def acquire(lock, block):
            if waiter: waiter_entering.set()
            return super().acquire(block)
    def infer(*args, **kw):
        if torch.is_grad_enabled(): raise AssertionError('inference has gradients enabled')
        output.put(('inference', waiter))
        started.set()
        if not release.wait(30): raise AssertionError('writer was not released')
        return [dict(pts3d=torch.ones(1, 1, 3), conf=torch.ones(1, 1),
                     desc=torch.ones(1, 1, 2), desc_conf=torch.ones(1, 1)) for _ in range(4)]
    corres = (torch.tensor([[1, 2]]), torch.tensor([[3, 4]]), torch.ones(1))
    try:
        with patch.object(sga, '_PairLock', ObservedLock), \
             patch.object(sga, 'symmetric_inference', side_effect=infer), \
             patch.object(sga, 'extract_correspondences', return_value=corres):
            paths, _ = sga.forward_mast3r(pairs, object(), root, device='cpu')
        output.put(('result', paths))
    except Exception as error:
        output.put(('error', repr(error)))
        raise


def save_as_if_on_cuda(obj, path):
    """torch.save(obj) as a CUDA box writes it -- every storage located on cuda:0 -- on a machine with
    no CUDA: the pickle's location string is rewritten inside the zip."""
    import io
    import zipfile
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buf) as zin, zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.endswith('data.pkl'):
                assert b'X\x03\x00\x00\x00cpu' in data
                data = data.replace(b'X\x03\x00\x00\x00cpu', b'X\x06\x00\x00\x00cuda:0')
            zout.writestr(info, data)


class ForwardHitTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.pairs = [({'instance': 'image-a'}, {'instance': 'image-b'})]
        self.a, self.b = map(sga.hash_md5, ('image-a', 'image-b'))
        self.p1 = self.root / 'forward' / self.a / (self.b + '.hvz')
        self.p2 = self.root / 'forward' / self.b / (self.a + '.hvz')
        self.pc = self.root / 'corres_conf=desc_conf_subsample=8' / (self.a + '-' + self.b + '.hvz')
        self.reverse = self.pc.with_name(self.b + '-' + self.a + '.hvz')
        self.corres = ((1., 2., 1), (torch.tensor([[1, 2]]), torch.tensor([[3, 4]]), torch.ones(1)))

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        sga._save_atomic(value, str(path))

    def seed(self):
        for p in (self.p1, self.p2):
            self.save(p, (torch.ones(1),) * 4)
        self.save(self.pc, self.corres)

    def forward(self, model=None, **kw):
        return sga.forward_mast3r(self.pairs, model, str(self.root), device='cpu', **kw)

    def test_complete_hit_avoids_lock_reverse_probe_and_load(self):
        self.seed()
        provenance = {'require': None, 'vid': 'current', 'reads': []}
        with sga._PairLock(str(self.root / 'forward' / '.locks' / ('-'.join(sorted((self.a, self.b))) + '.lock'))):
            with patch.object(sga, '_PairLock', side_effect=AssertionError('warm writer lock')), \
                 patch.object(sga.torch, 'load', side_effect=AssertionError('warm tensor read')), \
                 patch.object(sga.os.path, 'isfile', side_effect=AssertionError('reverse probe')), \
                 patch.object(sga, '_stat', wraps=sga._stat) as stats:
                paths, _ = self.forward(provenance=provenance)
        self.assertEqual(stats.call_count, 3)
        self.assertEqual(paths['image-a', 'image-b'], ((str(self.p1), str(self.p2)), str(self.pc)))
        self.assertFalse(provenance['reads'][0][3])
        self.assertEqual(provenance['reads'][0][4:], (self.p1.stat().st_mtime_ns, self.p1.stat().st_size))
        self.assertEqual(sga.LAST_FORWARD_PROFILE['lock_free_pairs'], 1)
        self.assertEqual(sga.LAST_FORWARD_PROFILE['locked_pairs'], 0)
        self.assertEqual(sga.LAST_FORWARD_PROFILE['parts_s']['lock_wait'], 0)

    def test_each_missing_output_uses_locked_path(self):
        for missing in (self.p1, self.p2, self.pc):
            with self.subTest(missing=missing):
                self.seed()
                missing.unlink()
                with patch.object(sga, '_PairLock', wraps=sga._PairLock) as lock:
                    paths, _ = self.forward()
                self.assertEqual(paths, {})
                lock.assert_called_once()
                self.assertEqual(sga.LAST_FORWARD_PROFILE['skipped_pairs'], 1)

    def test_broken_spill_link_is_a_miss(self):
        self.seed()
        self.p1.unlink()
        self.p1.symlink_to(self.root / 'gone-spill')
        paths, _ = self.forward()
        self.assertEqual(paths, {})
        self.assertEqual(sga.LAST_FORWARD_PROFILE['locked_pairs'], 1)

    def test_reverse_materialization_remains_locked(self):
        self.seed()
        self.pc.rename(self.reverse)
        paths, _ = self.forward()
        _, (xy1, xy2, _) = torch.load(paths['image-a', 'image-b'][1])
        self.assertTrue(torch.equal(xy1, self.corres[1][1]))
        self.assertTrue(torch.equal(xy2, self.corres[1][0]))
        self.assertEqual(sga.LAST_FORWARD_PROFILE['locked_pairs'], 1)

    def test_reverse_written_on_a_gpu_box_materializes_on_cpu(self):
        # a GPU box's corres file names cuda:0 for its tensors; home has no CUDA
        self.seed()
        self.pc.unlink()
        save_as_if_on_cuda(self.corres, self.reverse)
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            torch.load(self.reverse)                 # the stand-in is faithful: an unmapped read fails
        paths, _ = self.forward()
        _, (xy1, xy2, _) = torch.load(paths['image-a', 'image-b'][1])
        self.assertTrue(torch.equal(xy1, self.corres[1][1]))
        self.assertTrue(torch.equal(xy2, self.corres[1][0]))

    def test_a_pth_entry_is_read_where_its_hvz_is_absent_and_a_recompute_writes_hvz(self):
        # with two containers: entries written before .hvz are .pth; read by their key, rewritten only as .hvz
        for p in (self.p1, self.p2):
            self.save_pth(p.with_suffix('.pth'), (torch.ones(1),) * 4)
        self.save_pth(self.pc.with_suffix('.pth'), self.corres)
        paths, _ = self.forward()
        (r1, r2), rc = paths['image-a', 'image-b']
        self.assertEqual((r1, r2, rc), (str(self.p1.with_suffix('.pth')), str(self.p2.with_suffix('.pth')),
                                        str(self.pc.with_suffix('.pth'))), 'a warm hit on the .pth entries')
        self.assertFalse(self.p1.exists(), 'nothing rewritten')
        # the .hvz of the same key wins where both are there
        self.save(self.p1, (torch.zeros(1),) * 4)
        paths, _ = self.forward()
        self.assertEqual(paths['image-a', 'image-b'][0][0], str(self.p1))

    def save_pth(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, str(path))

    def test_required_provenance_remains_locked_and_missing_record_rejects(self):
        self.seed()
        provenance = {'require': {'device': 'cpu'}, 'record': {'device': 'cpu'}, 'vid': 'cpu', 'reads': []}
        for p in (self.p1, self.p2, self.pc):
            sga._prov_write(str(p), provenance['record'])
        paths, _ = self.forward(provenance=provenance)
        self.assertEqual(len(paths), 1)
        self.assertEqual(sga.LAST_FORWARD_PROFILE['locked_pairs'], 1)
        self.assertEqual(provenance['reads'][0][2], 'default-verified')
        Path(str(self.p1) + sga.PROV_SUFFIX).unlink()
        paths, _ = self.forward(provenance=provenance)
        self.assertEqual(paths, {})
        self.assertEqual(sga.LAST_FORWARD_PROFILE['locked_pairs'], 1)

    def test_r2_a_required_provenance_is_read_beside_a_pth_entry(self):
        # with two containers: entries made before .hvz are .pth with their records; a require run verifies them
        # (it used to read the record on the .hvz name only and recompute every unconverted pair)
        for p in (self.p1, self.p2):
            self.save_pth(p.with_suffix('.pth'), (torch.ones(1),) * 4)
        self.save_pth(self.pc.with_suffix('.pth'), self.corres)
        provenance = {'require': {'device': 'cpu'}, 'record': {'device': 'cpu'}, 'vid': 'cpu', 'reads': []}
        for p in (self.p1, self.p2, self.pc):
            sga._prov_write(str(p.with_suffix('.pth')), provenance['record'])
        paths, _ = self.forward(provenance=provenance)
        self.assertEqual(provenance['reads'][0][2], 'default-verified')
        self.assertEqual(paths['image-a', 'image-b'][0][0], str(self.p1.with_suffix('.pth')))
        self.assertFalse(self.p1.exists(), 'nothing recomputed as a variant')

    def test_a_run_local_copy_is_one_file_in_the_cache_container(self):
        # a masked source still .pth -> the run-local copy is re-stored as .hvz, and an older local
        # .pth twin (another mask variant) in the RUN's root is removed in the same step, never left to
        # shadow it; the copy carries the source's provenance record
        self.seed()
        run = self.root / 'run'
        masks = lambda i1, i2: (None, None, 'none', 'none')
        masked, _ = sga._masked_matching_paths(str(self.root), self.a, self.b, ('none', 'none'),
                                               'desc_conf', 8, False, {})
        self.save_pth(Path(masked).with_suffix('.pth'), self.corres)
        sga._prov_write(str(Path(masked).with_suffix('.pth')), {'device': 'cpu'})
        stale_twin = run / self.pc.relative_to(self.root).with_suffix('.pth')
        self.save_pth(stale_twin, ((0., 0., 0), (torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1))))
        sga._prov_write(str(stale_twin), {'device': 'stale'})
        paths, _ = self.forward(corr_masks=masks, corres_root=str(run))
        local = paths['image-a', 'image-b'][1]
        self.assertEqual(local, str(run / self.pc.relative_to(self.root)))
        self.assertFalse(stale_twin.exists(), 'the other container\'s twin is gone')
        self.assertFalse(Path(str(stale_twin) + sga.PROV_SUFFIX).exists(), 'with its record')
        self.assertEqual(sga._prov_read(local), {'device': 'cpu'}, 'the copy carries its source\'s record')
        _, (xy1, xy2, _) = torch.load(local)
        self.assertTrue(torch.equal(xy1, self.corres[1][0]))
        self.assertTrue(Path(masked).with_suffix('.pth').exists(), 'the shared source is untouched')

    def test_without_a_corres_root_nothing_is_written_in_the_shared_root(self):
        # no corres_root: the corres root IS the shared cache; masked matches are read where they are,
        # never copied over the shared default
        self.seed()
        masks = lambda i1, i2: (None, None, 'none', 'none')
        masked, _ = sga._masked_matching_paths(str(self.root), self.a, self.b, ('none', 'none'),
                                               'desc_conf', 8, False, {})
        self.save_pth(Path(masked).with_suffix('.pth'), self.corres)
        twin = self.pc.with_suffix('.pth')
        self.save_pth(twin, ((0., 0., 0), (torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1))))
        before = {p: p.read_bytes() for p in self.pc.parent.iterdir()}
        paths, _ = self.forward(corr_masks=masks)
        self.assertEqual(paths['image-a', 'image-b'][1], str(Path(masked).with_suffix('.pth')))
        self.assertEqual({p: p.read_bytes() for p in self.pc.parent.iterdir()}, before,
                         'the shared default and its twin, untouched')

    def test_a_corres_root_that_is_the_cache_itself_is_refused(self):
        # once, hillview's adaptive pairing and expansion passed the shared root as cache_path,
        # and its wrapper made that the corres_root: the masked matches' run-local copy then landed
        # under the RAW name in the shared root. Refused before anything is read or written, under
        # any spelling of the same directory.
        self.seed()
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        masks = lambda i1, i2: (None, None, 'none', 'none')
        for same in (str(self.root), str(self.root) + '/', str(self.root / 'forward' / '..')):
            with self.subTest(corres_root=same):
                with self.assertRaisesRegex(ValueError, 'corres_root is the cache it matches into'):
                    self.forward(corr_masks=masks, corres_root=same)
        self.assertEqual({p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}, before)

    def test_r7_a_variant_never_overwrites_the_shared_default(self):
        # once, a require run landing on a provenance VARIANT, with no corres_root, copied the
        # variant's matches over the shared default (and, with the .hvz container, its record too): every default run
        # then paired the default's forward passes with the variant's matches
        default = ((1., 1., 1), (torch.ones(1, 2), torch.ones(1, 2), torch.ones(1)))
        variant_value = ((9., 9., 9), (torch.full((1, 2), 9.), torch.full((1, 2), 9.), torch.full((1,), 9.)))
        for p in (self.p1, self.p2):
            self.save(p, (torch.ones(1),) * 4)
        self.save(self.pc, default)
        for p in (self.p1, self.p2, self.pc):
            sga._prov_write(str(p), {'device': 'cpu'})
            variant = Path(sga.prov_variant(str(p), 'gpu'))
            self.save(variant, variant_value if p == self.pc else (torch.full((1,), 9.),) * 4)
            sga._prov_write(str(variant), {'device': 'cuda'})
        before = self.pc.read_bytes()
        provenance = {'require': {'device': 'cuda'}, 'record': {'device': 'cuda'}, 'vid': 'gpu', 'reads': []}
        paths, _ = self.forward(provenance=provenance)
        self.assertEqual(provenance['reads'][0][2], 'variant')
        self.assertEqual(paths['image-a', 'image-b'][1], sga.prov_variant(str(self.pc), 'gpu'), 'the variant, read in place')
        self.assertEqual(self.pc.read_bytes(), before, 'the shared default is untouched')
        self.assertEqual(sga._prov_read(str(self.pc)), {'device': 'cpu'}, 'and so is its record')

    def test_r6_a_require_run_on_shared_pth_entries_keeps_them(self):
        # once, with require and no masks, resolve_cache_paths returned the found .pth, so the
        # source looked unlike its own (.hvz) name, the "run-local copy" re-stored the SHARED corres as
        # .hvz and deleted the shared .pth + sidecars -- and the next require run had no record to read
        for p in (self.p1, self.p2, self.pc):
            self.save_pth(p.with_suffix('.pth'), (torch.ones(1),) * 4 if p != self.pc else self.corres)
            sga._prov_write(str(p.with_suffix('.pth')), {'device': 'cpu'})
        before = sorted(os.listdir(self.pc.parent))
        for run in (1, 2):
            provenance = {'require': {'device': 'cpu'}, 'record': {'device': 'cpu'}, 'vid': 'cpu', 'reads': []}
            paths, _ = self.forward(provenance=provenance)
            self.assertEqual(provenance['reads'][0][2], 'default-verified', run)
            self.assertEqual(paths['image-a', 'image-b'][1], str(self.pc.with_suffix('.pth')), run)
            self.assertEqual(sorted(os.listdir(self.pc.parent)), before, run)

    def test_masked_copy_remains_locked_and_replaces_old_variant(self):
        self.seed()
        # the pair's masks and their identities (CORR_MASKS_BY_PAIR, CORR_MASKS_BY_IDENTITY)
        masks = lambda i1, i2: (None, None, 'none', 'none')
        masked, _ = sga._masked_matching_paths(str(self.root), self.a, self.b, ('none', 'none'),
                                               'desc_conf', 8, False, {})
        different = ((1., 2., 1), (torch.tensor([[5, 6]]), torch.tensor([[7, 8]]), torch.ones(1)))
        self.save(Path(masked), different)
        paths, _ = self.forward(corr_masks=masks, corres_root=str(self.root / 'run'))
        _, (xy1, xy2, _) = torch.load(paths['image-a', 'image-b'][1])
        self.assertTrue(torch.equal(xy1, different[1][0]))
        self.assertTrue(torch.equal(xy2, different[1][1]))
        self.assertEqual(sga.LAST_FORWARD_PROFILE['locked_pairs'], 1)
        self.assertEqual(sga.LAST_FORWARD_PROFILE['lock_free_pairs'], 0)

    def test_eviction_during_probe_falls_back_to_lock(self):
        self.seed()
        original = sga._stat
        def stat(path):
            if path == str(self.p1) and self.p1.exists():
                self.p1.unlink()
            return original(path)
        with patch.object(sga, '_stat', side_effect=stat):
            paths, _ = self.forward()
        self.assertEqual(paths, {})
        self.assertEqual(sga.LAST_FORWARD_PROFILE['lock_free_pairs'], 0)

    def test_live_spill_link_can_use_read_only_hit(self):
        self.seed()
        target = self.root / 'spill.hvz'
        self.p1.rename(target)
        self.p1.symlink_to(target)
        with patch.object(sga, '_PairLock', side_effect=AssertionError('spill hit lock')):
            paths, _ = self.forward()
        self.assertEqual(len(paths), 1)
        self.assertEqual(sga.LAST_FORWARD_PROFILE['lock_free_pairs'], 1)
        self.assertEqual(sga.LAST_FORWARD_PROFILE['computed_pairs'], 0)

    def test_warm_hit_reads_complete_payload_during_atomic_replacement(self):
        self.seed()
        pending, release = threading.Event(), threading.Event()
        original_replace = sga.os.replace
        replacement = ((1., 2., 1), (torch.tensor([[9, 10]]), torch.tensor([[11, 12]]), torch.ones(1)))
        def replace(source, destination):
            if str(destination) == str(self.pc):
                pending.set()
                if not release.wait(5): raise AssertionError('publication not released')
            return original_replace(source, destination)
        with patch.object(sga.os, 'replace', side_effect=replace), ThreadPoolExecutor(max_workers=1) as pool:
            writer = pool.submit(sga._save_atomic, replacement, str(self.pc))
            try:
                self.assertTrue(pending.wait(5))
                with patch.object(sga, '_PairLock', side_effect=AssertionError('warm writer lock')):
                    paths, _ = self.forward()
                _, (xy1, _, _) = torch.load(paths['image-a', 'image-b'][1])
                self.assertTrue(torch.equal(xy1, self.corres[1][0]))
            finally:
                release.set()
            writer.result(timeout=5)
        _, (xy1, _, _) = torch.load(self.pc)
        self.assertTrue(torch.equal(xy1, replacement[1][0]))

    def test_different_provenance_uses_existing_complete_variant_under_lock(self):
        self.seed()
        provenance = {'require': {'device': 'cuda'}, 'record': {'device': 'cuda'}, 'vid': 'gpu', 'reads': []}
        for path in (self.p1, self.p2, self.pc):
            variant = Path(sga.prov_variant(str(path), 'gpu'))
            self.save(variant, self.corres if path == self.pc else (torch.ones(1),) * 4)
            sga._prov_write(str(variant), provenance['record'])
        paths, _ = self.forward(provenance=provenance)
        self.assertEqual(sga.LAST_FORWARD_PROFILE['locked_pairs'], 1)
        self.assertEqual(provenance['reads'][0][2:4], ('variant', False))
        returned = paths['image-a', 'image-b']
        self.assertTrue(all('@gpu.hvz' in path for path in returned[0]))
        # no corres_root: the variant is read in place, never materialized over the shared default
        self.assertEqual(returned[1], sga.prov_variant(str(self.pc), 'gpu'))

    def test_profile_resets_after_warm_hit_and_counts_mixed_paths(self):
        self.seed()
        pairs = self.pairs + [({'instance': 'missing-a'}, {'instance': 'missing-b'})]
        sga.forward_mast3r(pairs, None, str(self.root), device='cpu')
        profile = sga.LAST_FORWARD_PROFILE
        self.assertEqual((profile['lock_free_pairs'], profile['locked_pairs']), (1, 1))
        self.assertEqual((profile['cached_pairs'], profile['skipped_pairs']), (1, 1))
        sga.forward_mast3r([], None, str(self.root), device='cpu')
        self.assertEqual((profile['lock_free_pairs'], profile['locked_pairs']), (0, 0))

    def test_independent_processes_compute_one_pair(self):
        context = multiprocessing.get_context('spawn')
        started, entering, release = (context.Event() for _ in range(3))
        output = context.Queue()
        processes = [context.Process(target=process_forward, args=(str(self.root), started,
                      entering, release, output, waiter)) for waiter in (False, True)]
        try:
            processes[0].start()
            self.assertTrue(started.wait(30), 'first process never started inference')
            processes[1].start()
            self.assertTrue(entering.wait(30), 'second process never attempted the lock')
            release.set()
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
            messages = [output.get(timeout=5) for _ in range(3)]
            self.assertEqual(sum(kind == 'inference' for kind, value in messages), 1)
            paths = [value for kind, value in messages if kind == 'result']
            self.assertEqual(len(paths), 2)
            self.assertEqual(paths[0], paths[1])
            _, (xy1, _, _) = torch.load(paths[0]['image-a', 'image-b'][1])
            self.assertTrue(torch.equal(xy1, self.corres[1][0]))
        finally:
            release.set()
            for process in processes:
                if process.is_alive(): process.terminate()
                if process.pid is not None: process.join(5)
            output.close()
            output.join_thread()

    def test_concurrent_misses_compute_once_and_waiter_rechecks(self):
        inference_started = threading.Event()
        waiter_attempting = threading.Event()
        calls = []
        original_lock = sga._PairLock
        class ObservedLock(original_lock):
            def acquire(lock, block):           # every attempt, tried or waited (see process_forward)
                if inference_started.is_set(): waiter_attempting.set()
                return super().acquire(block)
        def infer(*args, **kw):
            calls.append(1)
            inference_started.set()
            if not waiter_attempting.wait(5): raise AssertionError('waiter never attempted lock')
            return [dict(pts3d=torch.ones(1, 1, 3), conf=torch.ones(1, 1),
                         desc=torch.ones(1, 1, 2), desc_conf=torch.ones(1, 1)) for _ in range(4)]
        with patch.object(sga, '_PairLock', ObservedLock), \
             patch.object(sga, 'symmetric_inference', side_effect=infer), \
             patch.object(sga, 'extract_correspondences', return_value=self.corres[1]), \
             ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.forward, object())
            self.assertTrue(inference_started.wait(5))
            second = pool.submit(self.forward, object())
            self.assertEqual(first.result(timeout=10), second.result(timeout=10))
        self.assertEqual(len(calls), 1)



class MaskIdentityKeyTests(unittest.TestCase):
    """hillview: masked matches are keyed by mask IDENTITY, never by mask bytes."""

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = self.scratch.name

    def key(self, i1, i2):
        return sga._masked_matching_paths(self.root, 'ka', 'kb', (i1, i2), 'desc_conf', 8, False, {})[0]

    def test_identity_not_bytes_names_the_entry(self):
        x, y = 'a' * 64, 'b' * 64
        self.assertIn('/matching/masked-v3/', self.key(x, y))
        self.assertTrue(self.key(x, y).endswith(f'ka-{x}_kb-{y}.hvz'))
        self.assertNotEqual(self.key(x, y), self.key(y, x))
        self.assertTrue(self.key('none', x).endswith(f'ka-none_kb-{x}.hvz'))

    def test_anything_but_a_digest_or_none_is_refused(self):
        import numpy as np
        for bad in (np.zeros((2, 2), bool), None, 'A' * 64, 'a' * 63, 'mask.png'):
            with self.assertRaises(ValueError):
                self.key(bad, 'none')

    def test_a_mask_under_identity_none_is_refused(self):
        import numpy as np
        pairs = [({'instance': 'image-a'}, {'instance': 'image-b'})]
        m = np.ones((1, 1), bool)
        for got in ((m, None, 'none', 'none'), (None, None)):
            with self.assertRaises(ValueError):
                sga.forward_mast3r(pairs, None, self.root, device='cpu', corr_masks=lambda i1, i2, g=got: g)


if __name__ == '__main__':
    unittest.main()
