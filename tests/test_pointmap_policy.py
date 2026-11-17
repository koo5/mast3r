"""sparse_ga's POINTMAP_POLICY (hillview; dependency injection): which pairs carry the pointmap term is the
caller's policy, passed in. This module holds upstream's rule
alone, the expression 'default' (the pairs whose matching is not trusted carry the term), so a stock call imports
nothing of the caller's. The caller's own expressions and their records are hillview's tests (enrich/recon
test_pointmap_solver.py, test_optimizer_phase.py), on this module."""
import ast
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga  # noqa: E402

PAIRS = [(0, 1), (1, 0), (1, 2), (2, 1)]
MATCHED = {(0, 1), (1, 0)}


def test_upstreams_rule_is_the_one_this_module_knows():
    got = sparse_ga.select_pointmap_terms(PAIRS, MATCHED, 'default')
    assert [(d['view_i'], d['view_j'], d['selected']) for d in got] == [
        (0, 1, False), (1, 0, False), (1, 2, True), (2, 1, True)]
    assert all(d['default'] == d['selected'] for d in got)
    for expression in ('none', 'default or neighbor_distance <= 2', ''):
        with pytest.raises(ValueError, match='needs a pointmap_policy'):
            sparse_ga.select_pointmap_terms(PAIRS, MATCHED, expression)


def test_a_policy_decides_and_answers_in_order():
    asked = []

    def policy(expression, pairs, matched, inputs):
        asked.append((expression, pairs, matched, inputs))
        return [dict(view_i=i, view_j=j, selected=i < j, note='mine') for i, j in pairs]

    got = sparse_ga.select_pointmap_terms(PAIRS, MATCHED, 'my rule', {'x': 1}, policy)
    assert asked == [('my rule', PAIRS, MATCHED, {'x': 1})]
    assert [d['selected'] for d in got] == [True, False, True, False] and all(d['note'] == 'mine' for d in got)

    def reordered(expression, pairs, matched, inputs):
        return [dict(view_i=i, view_j=j, selected=True) for i, j in reversed(pairs)]

    with pytest.raises(ValueError, match='not in the order'):
        sparse_ga.select_pointmap_terms(PAIRS, MATCHED, 'default', None, reordered)


def test_an_expression_without_a_policy_is_refused_before_the_forward_passes():
    # the check is the entry's first statement: no image, pair or model is touched before it
    with pytest.raises(ValueError, match='needs a pointmap_policy'):
        sparse_ga.sparse_global_alignment(None, None, None, None, pointmap_pairs='none')
    with pytest.raises(ValueError, match='made here, by condense_data'):
        sparse_ga.sparse_global_alignment(None, None, None, None, pointmap_selection=[])


def test_the_optimiser_takes_decisions_and_condense_data_the_policy():
    assert sparse_ga.POINTMAP_POLICY is True
    opt = inspect.signature(sparse_ga.sparse_scene_optimizer).parameters
    assert opt['pointmap_selection'].default is None and opt['pointmap_policy'].default is None
    params = inspect.signature(sparse_ga.condense_data).parameters
    assert params['pointmap_policy'].default is None and params['selection_out'].default is None


def test_a_stock_import_reads_nothing_of_the_bench():
    """The two modules the bench kept on sys.path (recon_pointmap_policy, recon_pair_geometry) are imported nowhere
    in this package: the policy is passed in, and the pair geometry writer is this package's own."""
    root = Path(sparse_ga.__file__).resolve().parents[1]
    found = []
    for path in root.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            found += [f'{path.relative_to(root)}: {n}' for n in names if n.split('.')[0].startswith('recon_')]
    assert found == []
