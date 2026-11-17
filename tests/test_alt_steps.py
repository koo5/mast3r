"""sparse_ga's altitude steps (hillview), on synthetic cameras.

height_step_down is the GPS gauge's down under gauge_tilt_from 'height_steps': the mounts' down corrected so the
cameras' height steps rise as the terrain's do. alt_step_residual_m is what loss_alt_step sums (through huber_m,
in that gauge). Checked here: the true down comes back from a mount that is off by degrees; a raised stretch (a
bridge, an embankment) moves it far less than it moves a fit to absolute heights; a straight track keeps the mounts'
roll about itself (it cannot see one); no lever arm keeps the mounts' down and says so; the term is zero on the
terrain and its gradient reaches only the ends of the steps."""
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga  # noqa: E402
from mast3r.cloud_opt.sparse_ga import (alt_step_residual_m, gps_gauge, height_step_down, huber_m,  # noqa: E402
                                        weighted_sim3)

DOWN = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)


def rot(axis, deg):
    a = torch.as_tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float64)
    t = math.radians(deg)
    return torch.eye(3, dtype=torch.float64) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def ride(n=40, step=5.0, turn_deg=4.0, grade=0.04, seed=0, straight=False):
    """a bike ride: ENU fixes `step` m apart along a road that turns (unless straight) and climbs at `grade`, with
    the terrain's bumps on top; the heights are the terrain's (the camera's, here, until a test raises some)"""
    rng = np.random.default_rng(seed)
    head = 0.0 if straight else np.cumsum(np.radians(turn_deg) * np.sin(np.arange(n - 1) / 6.0))
    head = np.broadcast_to(head, (n - 1,))
    xy = np.vstack([np.zeros(2), np.cumsum(step * np.c_[np.cos(head), np.sin(head)], 0)])
    s_along = np.r_[0.0, np.cumsum(np.full(n - 1, step))]
    z = grade * s_along + 0.3 * np.sin(s_along / 17.0) + 0.02 * rng.standard_normal(n)
    return torch.as_tensor(np.c_[xy, z], dtype=torch.float64)


def model_of(enu, Q, s=0.31, t=(3.0, -1.0, 2.0)):
    """camera centres in a solver frame: ENU through the rotation Q (any tilt), scale s, translation t. The
    model's true down is Q^T applied to ENU's down."""
    return (enu @ Q) / s + torch.tensor(t, dtype=torch.float64), Q.T @ DOWN


def bounce(enu, sd=0.03, seed=7):
    """the cameras over the terrain: each height off the DEM's by sd metres (the grid between fixes, the rider)"""
    cam = enu.clone()
    cam[:, 2] += torch.as_tensor(np.random.default_rng(seed).normal(0.0, sd, len(enu)), dtype=torch.float64)
    return cam


def links(n):
    return torch.arange(n - 1), torch.arange(1, n)


def deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b / (a.norm() * b.norm()))))))


Q = rot([0.3, 1.0, 0.2], 37.0) @ rot([0, 0, 1], 75.0)


def test_the_true_down_comes_back_from_a_mount_off_by_degrees():
    enu = ride()
    c, g_true = model_of(bounce(enu), Q)
    w = torch.full((len(enu),), 1.0 / len(enu), dtype=torch.float64)
    li, lj = links(len(enu))
    for axis in ([1.0, 0, 0], [0, 1.0, 0], [0.5, 0.5, 0.2]):
        g0 = rot(axis, 3.0) @ g_true                      # the mounts' mean, pitched and rolled by the grade
        g, info = height_step_down(c, enu, w, li, lj, g0, 0.5, 3.0)
        # the cameras' 3 cm bounce allows about a tenth of a degree (0.01-0.12 over seeds 0-4)
        assert deg(g, g_true) < 0.2, (axis, deg(g, g_true), info)
        assert info['why'] is None and info['beyond_knee'] == 0 and 0.01 <= info['sigma_z_m'] < 0.1, info


def test_a_raised_stretch_moves_the_steps_far_less_than_absolute_heights():
    """a bridge: the cameras run 3 m above the terrain over views 12-22. Only two steps see it, and the Huber lets
    them go; a fit to absolute heights takes the whole stretch's 3 m into its tilt"""
    enu = ride(seed=1)
    cam = bounce(enu)
    cam[12:23, 2] += 3.0
    c, g_true = model_of(cam, Q)
    w = torch.full((len(enu),), 1.0 / len(enu), dtype=torch.float64)
    li, lj = links(len(enu))
    g, info = height_step_down(c, enu, w, li, lj, rot([1, 0, 0], 2.0) @ g_true, 0.5, 3.0)
    _s, R, _t = weighted_sim3(c, enu, w)                  # absolute heights choose the tilt
    g_abs = R.T @ DOWN
    assert info['beyond_knee'] == 2, info
    # the Huber caps the two steps' pull, it does not zero it: on a turning road they leave 0.27-0.44 deg (seeds 0-4)
    # where absolute heights leave 2.4-2.5
    assert deg(g, g_true) < 0.25 * deg(g_abs, g_true), (deg(g, g_true), deg(g_abs, g_true))
    assert deg(g, g_true) < 0.6


def test_a_straight_track_keeps_the_mounts_roll_about_it():
    """steps along a straight road see its pitch and nothing of a roll about it: that stays the mounts'"""
    enu = ride(straight=True)
    c, g_true = model_of(bounce(enu), Q)
    w = torch.full((len(enu),), 1.0 / len(enu), dtype=torch.float64)
    li, lj = links(len(enu))
    along = Q.T @ torch.tensor([1.0, 0, 0], dtype=torch.float64)    # the road's direction in the model
    across = Q.T @ torch.tensor([0, 1.0, 0], dtype=torch.float64)
    g0 = rot(along, 2.0) @ rot(across, 3.0) @ g_true                  # rolled 2 deg, pitched 3 deg
    g, info = height_step_down(c, enu, w, li, lj, g0, 0.5, 3.0)
    assert info['seen'] < 1e-6, info
    def pitch(x):
        return math.degrees(math.atan2(float(x @ along), float(x @ g_true)))

    def roll(x):
        return math.degrees(math.atan2(float(x @ across), float(x @ g_true)))
    assert abs(pitch(g) - pitch(g_true)) < 0.05, (pitch(g), pitch(g_true), pitch(g0))
    # the roll stays the mounts' within a tenth of their own 3 deg (0.2-0.4 deg over seeds 0-4: a roll changes each
    # rise by 1 - cos, which the bounce hides)
    assert abs(roll(g) - roll(g0)) < 0.5, (roll(g), roll(g0))


def test_a_warm_start_reaches_the_same_down_in_fewer_steps():
    """the gauge refits every iteration from the group's last answer: the same down, sooner"""
    enu = ride()
    c, g_true = model_of(bounce(enu), Q)
    w = torch.full((len(enu),), 1.0 / len(enu), dtype=torch.float64)
    li, lj = links(len(enu))
    g0 = rot([0.5, 0.5, 0.2], 3.0) @ g_true
    g, info = height_step_down(c, enu, w, li, lj, g0, 0.5, 3.0)
    g2, info2 = height_step_down(c, enu, w, li, lj, g0, 0.5, 3.0, g_start=g)
    assert deg(g2, g) < 1e-5, deg(g2, g)
    assert info2['iters'] <= 2 < info['iters'], (info['iters'], info2['iters'])


def test_no_lever_arm_keeps_the_mounts_down_and_says_so():
    enu = ride(n=6)
    c = torch.zeros(6, 3, dtype=torch.float64) + 1.0
    w = torch.full((6,), 1.0 / 6, dtype=torch.float64)
    g0 = torch.tensor([0.1, 0.9, 0.2], dtype=torch.float64)
    g, info = height_step_down(c, enu, w, *links(6), g0, 0.5, 3.0)
    assert torch.allclose(g, g0 / g0.norm()) and info['why'] and info['iters'] == 0


def test_the_step_term_is_zero_on_the_terrain_and_reaches_only_step_ends():
    enu = ride(n=10)
    c0, g_true = model_of(enu, Q)
    w = torch.full((10,), 1.0 / 10, dtype=torch.float64)
    zok = torch.zeros(10, dtype=torch.bool)                # height_steps: the gauge reads the horizontal only
    li, lj = torch.as_tensor([2, 3, 6]), torch.as_tensor([3, 4, 7])
    dz = enu[lj, 2] - enu[li, 2]
    c = c0.clone().requires_grad_(True)
    with torch.no_grad():
        s, R, _t, _f = gps_gauge(c.detach(), enu, w, zok, g=g_true)
    r = alt_step_residual_m(c, li, lj, dz, s, R)
    assert float(r.detach().abs().max()) < 1e-9
    # one camera raised a metre: the two steps it ends read +-1 m, in metres whatever the solver's unit
    up = (Q.T @ -DOWN) / 0.31
    c2 = (c0 + torch.where(torch.arange(10)[:, None] == 3, up, torch.zeros(3, dtype=torch.float64))).requires_grad_(True)
    r2 = alt_step_residual_m(c2, li, lj, dz, s, R)
    assert torch.allclose(r2, torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64), atol=1e-9), r2
    g = torch.autograd.grad(huber_m(r2.abs(), 0.5).sum(), c2)[0]
    assert set(torch.nonzero(g.abs().sum(1) > 1e-9).flatten().tolist()) == {2, 3, 4}


def test_the_optimiser_takes_the_steps_with_the_old_behaviour_by_default():
    p = inspect.signature(sparse_ga.sparse_scene_optimizer).parameters
    for name, legacy in (('alt_step_links', None), ('gauge_tilt_from', 'mount'), ('alt_step_frac', 0.0),
                         ('alt_step_huber_m', 0.5), ('gauge_tilt_mount_sd_deg', 3.0)):
        assert p[name].default == legacy, name
    assert sparse_ga.ALT_STEPS is True
