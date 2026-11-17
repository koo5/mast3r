"""A cache write reaches the disk BEFORE its rename (a box's reboot left zero-byte entries under complete names),
and a zero-byte cache file met at the hit decision stops the call, named."""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga as sga  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _standin_codec  # noqa: E402

_standin_codec.install()


def _record(events):
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(('fsync', os.path.realpath(f'/proc/self/fd/{fd}')))
        return real_fsync(fd)

    def replace(src, dst):
        events.append(('replace', os.path.realpath(src), os.path.realpath(dst)))
        return real_replace(src, dst)
    return patch.object(sga.os, 'fsync', fsync), patch.object(sga.os, 'replace', replace)


def test_every_temporary_is_on_disk_before_its_rename_and_the_directory_after():
    with tempfile.TemporaryDirectory() as d:
        events = []
        p1, p2 = _record(events)
        path = os.path.join(d, 'k1-k2.hvz')
        with p1, p2:
            sga._save_atomic((1.0, (torch.ones(3, 2), torch.ones(3, 2), torch.ones(3))), path)
            sga._prov_write(path, {'checkpoint': 'c'})
        replaces = [e for e in events if e[0] == 'replace']
        assert [os.path.basename(e[2]) for e in replaces] == ['k1-k2.hvz', 'k1-k2.hvz.sha256', 'k1-k2.hvz.content',
                                                             'k1-k2.hvz.prov.json']
        for e in replaces:
            assert ('fsync', e[1]) in events[:events.index(e)], f'{e[1]} renamed before it was fsynced'
        last_entry_rename = events.index(replaces[2])
        assert ('fsync', os.path.realpath(d)) in events[last_entry_rename:], 'the directory was not fsynced'
        assert sga._cache_load(path, 'cpu')[0] == 1.0


def _entries(root):
    a, b = map(sga.hash_md5, ('image-a', 'image-b'))
    for x, y in ((a, b), (b, a)):
        p = root / 'forward' / x / (y + '.hvz')
        p.parent.mkdir(parents=True, exist_ok=True)
        sga._save_atomic((torch.ones(1),) * 4, str(p))
    c = root / 'corres_conf=desc_conf_subsample=8' / f'{a}-{b}.hvz'
    c.parent.mkdir(parents=True, exist_ok=True)
    sga._save_atomic(((1., 2., 1), (torch.tensor([[1, 2]]), torch.tensor([[3, 4]]), torch.ones(1))), str(c))
    return a, b, c


PAIRS = [({'instance': 'image-a'}, {'instance': 'image-b'})]


def test_an_empty_entry_is_refused_by_name_not_read_nor_recomputed():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        a, b, c = _entries(root)
        sga.forward_mast3r(PAIRS, object(), d, device='cpu')                 # a clean hit
        empty = root / 'forward' / a / (b + '.hvz')
        empty.write_bytes(b'')
        with patch.object(sga, 'symmetric_inference', side_effect=AssertionError('recomputed')):
            with pytest.raises(sga.EmptyCacheEntry) as e:
                sga.forward_mast3r(PAIRS, object(), d, device='cpu')
        assert str(empty) in str(e.value)
        assert empty.stat().st_size == 0                                       # left for the quarantine


def test_an_empty_reverse_is_named_too():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        a, b, c = _entries(root)
        rev = c.parent / f'{b}-{a}.hvz'
        c.rename(rev)                                                          # only (b, a)'s matches
        rev.write_bytes(b'')
        with patch.object(sga, 'symmetric_inference', side_effect=AssertionError('recomputed')):
            with pytest.raises(sga.EmptyCacheEntry) as e:
                sga.forward_mast3r(PAIRS, object(), d, device='cpu')
        assert str(rev) in str(e.value)
        assert not c.exists()                                                  # nothing transposed from it
