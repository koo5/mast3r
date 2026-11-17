"""A refusing kinematic tree says WHERE the pair graph split (DisconnectedGraph): each piece's
views by range and file, and the pairs asked for across the cut -- removed before the solve, or reached
with no correspondence. A 10-frame span once refused with only "2 disconnected components"."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga as sga  # noqa: E402


def chain(n, cut=()):
    """Scores of a walk 0-1-..-(n-1), each view paired with its neighbours, minus the cut edges."""
    s = torch.zeros((n, n))
    for i in range(n - 1):
        if i not in cut:
            s[i, i + 1] = s[i + 1, i] = 10. + i
    return s


def test_mst_names_its_pieces():
    with pytest.raises(sga.DisconnectedGraph) as e:
        sga.compute_min_spanning_tree(chain(5, cut={2}))
    assert isinstance(e.value, ValueError)                 # every caller catching ValueError still does
    assert e.value.components == [[0, 1, 2], [3, 4]]
    msg = str(e.value)
    assert msg.startswith('kinematic tree: correspondence graph has 2 disconnected components: '
                          '[1] 3 view(s) 0-2; [2] 2 view(s) 3-4. ')
    assert sga.compute_min_spanning_tree(chain(5))[1]       # a connected graph builds its tree as before


def test_scattered_pieces_are_ranges():
    s = chain(6)
    s[2, :] = s[:, 2] = 0                                   # view 2 alone: the walk is 0-1 | 2 | 3-4-5
    with pytest.raises(sga.DisconnectedGraph) as e:
        sga.compute_min_spanning_tree(s)
    assert e.value.components == [[0, 1], [2], [3, 4, 5]]
    assert '[2] 1 view(s) 2; [3] 3 view(s) 3-5' in str(e.value)


def test_shallow_names_its_pieces():
    with pytest.raises(sga.DisconnectedGraph) as e:
        sga.min_depth_spanning_tree(chain(6, cut={1}), log=lambda *_: None)
    assert e.value.components == [[0, 1], [2, 3, 4, 5]]
    assert str(e.value).startswith('shallow tree: the pair graph is not connected (')
    assert '[1] 2 view(s) 0-1; [2] 4 view(s) 2-5. Connect the views' in str(e.value)


@pytest.mark.parametrize('shallow', [False, True])
def test_rig_pieces_are_views_not_groups(shallow):
    # three panoramas of 2 slices (views 0-1, 2-3, 4-5) and a lone photograph (6); rigs 0-1 meet, 2 and 6 meet
    s = torch.zeros((7, 7))
    for i, j in ((1, 2), (5, 6)):
        s[i, j] = s[j, i] = 5.
    rig = [(0, None), (0, None), (1, None), (1, None), (2, None), (2, None), None]
    with pytest.raises(sga.DisconnectedGraph) as e:
        sga.rig_spanning_tree(s, rig, shallow=shallow)
    assert e.value.components == [[0, 1, 2, 3], [4, 5, 6]]


def test_name_views_logs_the_cut():
    with pytest.raises(sga.DisconnectedGraph) as e:
        sga.compute_min_spanning_tree(chain(4, cut={1}))
    lines = []
    imgs = [f'/run/slices/{i:03d}_abcd{i}_y000.jpg' for i in range(4)]
    asked = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (0, 2), (1, 3)]
    reached = [(0, 1), (1, 0), (2, 3), (3, 2), (1, 2)]     # 1-2 reached empty; 0-2 and 1-3 were gated out
    e.value.name_views(imgs, asked, reached, log=lines.append)
    assert e.value.cut == [(0, 2, False), (1, 2, True), (1, 3, False)]
    assert lines[0] == (' >> kinematic tree refused: 2 pieces; 3 pair(s) were asked for across the cut, '
                        'and none has a correspondence:')
    assert lines[2] == '    [1] 1 001_abcd1_y000.jpg  -  [2] 2 002_abcd2_y000.jpg: reached the solve with 0 correspondences'
    assert lines[1].endswith(': removed before the solve')
    assert '[1] 2 view(s) 0-1 000_abcd0_y000.jpg .. 001_abcd1_y000.jpg; [2] 2 view(s) 2-3 ' in str(e.value)


def test_sparse_global_alignment_names_the_split(capsys):
    """The real entry: sparse_global_alignment's tree site names the pieces by file, with the cut, from its
    own pairs_in and the pairs that came back from forward_mast3r (the caller's gates removed 1-2)."""
    imgs = [f'/run/slices/{i:03d}_v{i}.jpg' for i in range(4)]
    pairs_in = [({'idx': i}, {'idx': j}) for i, j in ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2))]
    back = {(imgs[i], imgs[j]): None for i, j in ((0, 1), (1, 0), (2, 3), (3, 2))}
    condensed = (None,) * 5 + ((None, None, []), None, None)
    with patch.object(sga, 'forward_mast3r', return_value=(back, '/c')), \
            patch.object(sga, 'prepare_canonical_data', return_value=({}, chain(4, cut={1}), {}, [], {})), \
            patch.object(sga, 'condense_data', return_value=condensed):
        with pytest.raises(sga.DisconnectedGraph) as e:
            sga.sparse_global_alignment(imgs, pairs_in, '/c', None, kinematic_mode='mst', device='cpu')
    assert str(e.value).startswith('kinematic tree: correspondence graph has 2 disconnected components: '
                                   '[1] 2 view(s) 0-1 000_v0.jpg .. 001_v1.jpg; [2] 2 view(s) 2-3 002_v2.jpg .. 003_v3.jpg. ')
    out = capsys.readouterr().out
    assert '1 pair(s) were asked for across the cut' in out
    assert '[1] 1 001_v1.jpg  -  [2] 2 002_v2.jpg: removed before the solve' in out
