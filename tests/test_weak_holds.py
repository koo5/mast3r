"""sparse_ga's weak-frame holds (hillview), on synthetic cameras.

gps_rel_residual_m / weak_mid_residual_m are the terms sparse_scene_optimizer's loss_gps_rel / loss_weak_mid sum
(each through huber_m, with the gauge from gps_gauge); weak_hold_plan is what decides which steps and cameras
they hold. Checked here: each term's gradient reaches only the cameras it should, weak_mid moves only weak
cameras (its neighbours detached), scope 'weak' holds only steps with a weak end."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga  # noqa: E402
from mast3r.cloud_opt.sparse_ga import (gps_gauge, gps_rel_residual_m, huber_m, weak_hold_plan,  # noqa: E402
                                        weak_mid_residual_m)


def track(n=8, seed=0):
    """a walk: ENU fixes about 0.85 m apart, wandering a little, with heights"""
    rng = np.random.default_rng(seed)
    steps = np.c_[0.85 + 0.1 * rng.standard_normal(n - 1), 0.2 * rng.standard_normal(n - 1),
                  0.05 * rng.standard_normal(n - 1)]
    return torch.as_tensor(np.vstack([np.zeros(3), np.cumsum(steps, 0)]), dtype=torch.float64)


def cameras_on(enu, s=0.37, yaw=0.6, t=(4.0, -2.0, 1.0)):
    """camera-to-world poses whose centres are the track under a similarity (solver units != metres)"""
    c, sn = np.cos(yaw), np.sin(yaw)
    R = torch.tensor([[c, -sn, 0], [sn, c, 0], [0, 0, 1]], dtype=torch.float64)
    cam2w = torch.eye(4, dtype=torch.float64).repeat(len(enu), 1, 1)
    cam2w[:, :3, 3] = s * (enu @ R.T) + torch.tensor(t, dtype=torch.float64)
    return cam2w


def rel_loss(cam2w, enu, li, lj, zok=None, knee=1.0):
    """loss_gps_rel's body for one group: the gauge fitted detached, the residual live"""
    zok = torch.ones(len(enu), dtype=torch.bool) if zok is None else zok
    w = torch.full((len(enu),), 1.0 / len(enu), dtype=torch.float64)
    c = cam2w[:, :3, 3]
    with torch.no_grad():
        s, R, _t, enu_fit = gps_gauge(c.detach(), enu, w, zok)
    return huber_m(gps_rel_residual_m(c, li, lj, enu_fit, zok, s, R), knee).sum()


def grad_rows(loss, cam2w):
    g = torch.autograd.grad(loss, cam2w)[0]
    return set(torch.nonzero(g.abs().flatten(1).sum(1) > 0).flatten().tolist())


def idx(*v):
    return torch.as_tensor(v)


# ---------------------------------------------------------------- gps_rel
def test_rel_is_zero_on_the_track_under_any_similarity():
    enu = track()
    cam2w = cameras_on(enu)
    li, lj = idx(*range(7)), idx(*range(1, 8))
    assert float(rel_loss(cam2w, enu, li, lj)) < 1e-18


def test_rel_reads_the_gauge_not_the_solver_units():
    """the same displacement of one camera costs the same metres whatever unit the solve is in"""
    enu = track()
    vals = []
    for s in (0.37, 5.0):
        cam2w = cameras_on(enu, s=s)
        cam2w[3, :3, 3] += s * torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64)     # 0.5 m sideways, in metres
        r = gps_rel_residual_m(cam2w[:, :3, 3], idx(2, 3), idx(3, 4), enu, torch.ones(8, dtype=torch.bool),
                               *gps_gauge(cam2w[:, :3, 3], enu, torch.full((8,), 1 / 8, dtype=torch.float64),
                                          torch.ones(8, dtype=torch.bool))[:2])
        vals.append(r)
    assert torch.allclose(vals[0], vals[1], atol=1e-9)
    assert float(vals[0].min()) > 0.3                         # it is seen: ~0.5 m less the gauge's share


def test_rel_gradient_reaches_only_the_ends_of_held_steps():
    enu = track()
    cam2w = cameras_on(enu)
    cam2w[3, :3, 3] += torch.tensor([0.4, -0.3, 0.1], dtype=torch.float64)
    cam2w[6, :3, 3] += torch.tensor([0.2, 0.2, 0.0], dtype=torch.float64)       # off, but no held step touches it
    cam2w.requires_grad_(True)
    rows = grad_rows(rel_loss(cam2w, enu, idx(2, 3), idx(3, 4)), cam2w)
    assert rows == {2, 3, 4}


def test_rel_a_stretch_that_drifted_together_costs_nothing_inside_it():
    """a GPS stretch offset as a whole moves both ends of its steps: only the step INTO the stretch sees it"""
    enu = track()
    w = torch.full((8,), 1 / 8, dtype=torch.float64)
    zok = torch.ones(8, dtype=torch.bool)
    cam2w = cameras_on(enu)
    drift = enu.clone()
    drift[4:] += torch.tensor([3.0, -2.0, 0.0], dtype=torch.float64)
    c = cam2w[:, :3, 3]
    s, R, _t, fit = gps_gauge(c, drift, w, zok)
    r = gps_rel_residual_m(c, idx(*range(7)), idx(*range(1, 8)), fit, zok, s, R)
    inside, into = r[4:], r[3]
    assert float(into) > 2.0
    # inside the stretch the steps agree up to the gauge's compromise (one similarity over both pieces)
    assert float((inside - r[:3].mean()).abs().max()) < 0.5 * float(into)


def test_rel_holds_no_height_where_an_end_has_no_altitude():
    enu = track()
    c = enu.clone()                                    # identity gauge: the solve IS in metres
    c[3, 2] += 0.7                                     # camera 3 up 0.7 m
    s, R = torch.tensor(1.0, dtype=torch.float64), torch.eye(3, dtype=torch.float64)
    known = torch.ones(8, dtype=torch.bool)
    no3 = known.clone()
    no3[3] = False
    held = gps_rel_residual_m(c, idx(2, 3), idx(3, 4), enu, known, s, R)
    free = gps_rel_residual_m(c, idx(2, 3), idx(3, 4), enu, no3, s, R)
    assert torch.allclose(held, torch.full((2,), 0.7, dtype=torch.float64))
    assert float(free.abs().max()) < 1e-12
    gained = gps_rel_residual_m(c, idx(2, 3), idx(3, 4), enu, known, s, R, z_gain=2.0)
    assert torch.allclose(gained, torch.full((2,), 1.4, dtype=torch.float64))


def test_huber_is_the_gps_terms():
    x = torch.tensor([0.0, 0.5, 1.0, 3.0], dtype=torch.float64)
    assert torch.allclose(huber_m(x, 1.0), torch.tensor([0.0, 0.125, 0.5, 2.5], dtype=torch.float64))


# ---------------------------------------------------------------- weak_mid
def test_mid_gradient_reaches_only_the_weak_camera():
    enu = track()
    c = (enu * 0.37).clone()
    c[3] += torch.tensor([0.0, 0.6, 0.0], dtype=torch.float64)
    c.requires_grad_(True)
    loss = huber_m(weak_mid_residual_m(c, idx(3), idx(2), idx(4), torch.tensor(1 / 0.37, dtype=torch.float64)),
                   1.0).sum()
    g = torch.autograd.grad(loss, c)[0]
    assert set(torch.nonzero(g.abs().sum(1) > 0).flatten().tolist()) == {3}


def test_mid_is_in_metres_through_the_gauge_scale():
    c = torch.tensor([[0.0, 0, 0], [1.0, 0.5, 0], [2.0, 0, 0]], dtype=torch.float64)    # 0.5 units off the midpoint
    r = weak_mid_residual_m(c, idx(1), idx(0), idx(2), torch.tensor(4.0, dtype=torch.float64))
    assert torch.allclose(r, torch.tensor([2.0], dtype=torch.float64))


def test_mid_moves_only_weak_cameras_and_a_run_of_them_chains():
    """gradient descent on the term alone: weak 3 and 4 between 2 and 5 settle evenly spaced; nothing else moves"""
    c0 = torch.tensor([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3.4, 1.5, 0], [3.9, -1.2, 0.4], [5, 0, 0], [6, 0, 0]],
                      dtype=torch.float64)
    c = c0.clone().requires_grad_(True)
    s = torch.tensor(1.0, dtype=torch.float64)
    opt = torch.optim.SGD([c], lr=0.2)
    for _ in range(600):
        opt.zero_grad()
        huber_m(weak_mid_residual_m(c, idx(3, 4), idx(2, 3), idx(4, 5), s), 1.0).sum().backward()
        opt.step()
    c = c.detach()
    assert torch.equal(c[[0, 1, 2, 5, 6]], c0[[0, 1, 2, 5, 6]])
    assert torch.allclose(c[3], torch.tensor([3.0, 0, 0], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(c[4], torch.tensor([4.0, 0, 0], dtype=torch.float64), atol=1e-6)


# ---------------------------------------------------------------- the plan
LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8)]       # two chains: 0-5 and 6-8


def rows(views, group=0):
    return {v: (group, r) for r, v in enumerate(views)}


def test_plan_scope_weak_holds_only_steps_with_a_weak_end():
    p = weak_hold_plan({3}, LINKS, rows(range(9)), 'weak')
    assert p['rel'] == {0: [(2, 3), (3, 4)]}
    p = weak_hold_plan({3, 4}, LINKS, rows(range(9)), 'weak')
    assert p['rel'] == {0: [(2, 3), (3, 4), (4, 5)]}
    assert weak_hold_plan(set(), LINKS, rows(range(9)), 'weak')['rel'] == {}


def test_plan_scope_all_holds_every_step():
    p = weak_hold_plan(set(), LINKS, rows(range(9)), 'all')
    assert p['rel'] == {0: [(i, j) for i, j in LINKS]}


def test_plan_mid_needs_a_neighbour_each_side():
    p = weak_hold_plan({0, 3, 8, 9}, LINKS, rows(range(10)), 'weak')
    assert p['mid'] == {0: [(3, 2, 4)]}
    assert p['ends'] == [0, 8]                       # first and last of their chains
    assert p['off'] == 1                             # 9 is on no link (a rig's other view, or no photograph)


def test_plan_keeps_steps_and_triples_inside_one_gps_group():
    row_of = {**rows(range(0, 3), group=0), **rows(range(3, 9), group=1)}
    p = weak_hold_plan({2, 3}, LINKS, row_of, 'weak')
    assert p['rel'] == {0: [(1, 2)], 1: [(0, 1)]} and p['rel_out'] == 1      # 2-3 crosses the pieces
    assert p['mid'] == {} and p['mid_out'] == 2
    no_fix = rows([0, 1, 2, 4, 5, 6, 7, 8])                                   # view 3 has no fix: in no group
    p = weak_hold_plan({3}, LINKS, no_fix, 'weak')
    assert p['rel'] == {} and p['rel_out'] == 2 and p['mid_out'] == 1


def test_plan_refuses_a_view_linked_twice_on_one_side_and_an_unknown_scope():
    with pytest.raises(ValueError, match='two links on one side'):
        weak_hold_plan({3}, LINKS + [(2, 4)], rows(range(9)), 'weak')
    with pytest.raises(ValueError, match="gps_rel_scope 'some'"):
        weak_hold_plan({3}, LINKS, rows(range(9)), 'some')


def test_the_optimiser_takes_the_holds_and_says_so():
    """the caller's contract: WEAK_HOLDS, and the keywords it passes"""
    assert sparse_ga.WEAK_HOLDS is True
    params = inspect.signature(sparse_ga.sparse_scene_optimizer).parameters
    for name, default in (('weak_views', None), ('step_links', None), ('gps_rel_frac', 0.0), ('gps_rel_huber_m', 1.0),
                          ('gps_rel_scope', 'weak'), ('weak_mid_frac', 0.0), ('weak_mid_huber_m', 1.0)):
        assert params[name].default == default
