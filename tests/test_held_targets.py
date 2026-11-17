"""sparse_ga's held targets (hillview), on synthetic cameras.

A held subset: a judged solve's views leave the optimiser and enter as FIXED points on the new views' matches.
held_reproj is the stage-2 term's projection (as loss_2d's), held_ray_closest the stage-1 term's nearest point on the
pixel's ray (as loss_3d's 3-D distance), fixed_gauge the GPS terms' gauge when it is the source's alignment, given.
Checked here: a point projects back onto the pixel it came from, from any camera; a point on its pixel's ray is at
distance 0 from it and one moved off it by d is at d; a point behind the camera is held to the centre; the gradient
reaches the camera and never the fixed point; a camera started off its true pose is brought back by the reprojection
term alone (a resection, which is what a held view's targets do); the given gauge is returned as given, with unknown
heights filled from the model; and an ordinary run is untouched (the globals' defaults)."""
import inspect
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r.cloud_opt import sparse_ga  # noqa: E402
from mast3r.cloud_opt.sparse_ga import (fixed_gauge, held_depth_scale, held_ray_closest, held_reproj,  # noqa: E402
                                        held_start)

D = torch.float64


def rot(axis, deg):
    a = torch.as_tensor(axis, dtype=D)
    a = a / a.norm()
    k = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=D)
    t = math.radians(deg)
    return torch.eye(3, dtype=D) + math.sin(t) * k + (1 - math.cos(t)) * k @ k


def camera(R, C, f=300.0, cx=256.0, cy=192.0):
    """cam2w from a rotation and a centre, its w2cam, and K"""
    c2w = torch.eye(4, dtype=D)
    c2w[:3, :3], c2w[:3, 3] = R, torch.as_tensor(C, dtype=D)
    K = torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=D)
    return c2w, torch.linalg.inv(c2w), K


def scene(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    c2w0, w2c0, K0 = camera(torch.eye(3, dtype=D), [0, 0, 0])
    c2w1, w2c1, K1 = camera(rot([0, 1, 0], 20) @ rot([1, 0, 0], -5), [1.5, -0.2, 0.3], f=280.0)
    X = torch.stack([torch.rand(n, generator=g, dtype=D) * 6 - 3, torch.rand(n, generator=g, dtype=D) * 4 - 2,
                     torch.rand(n, generator=g, dtype=D) * 10 + 5], 1)           # in front of both
    return torch.stack([w2c0, w2c1]), torch.stack([K0, K1]), torch.stack([c2w0, c2w1]), X


def project(K, w2c, X):
    x = (w2c[:3, :3] @ X.T).T + w2c[:3, 3]
    return (x @ K.T)[:, :2] / x[:, 2:3]


def test_a_point_projects_back_onto_its_pixel_from_any_camera():
    w2c, K, _c2w, X = scene()
    for v in (0, 1):
        vi = torch.full((len(X),), v, dtype=torch.long)
        uv = held_reproj(K, w2c, vi, X)
        assert torch.allclose(uv, project(K[v], w2c[v], X), atol=1e-9)


def test_the_ray_term_is_the_distance_from_the_pixels_ray():
    w2c, K, c2w, X = scene()
    v = 1
    vi = torch.full((len(X),), v, dtype=torch.long)
    pix = project(K[v], w2c[v], X)
    on = held_ray_closest(w2c, K, vi, pix, X)
    assert float((on - X).norm(dim=1).max()) < 1e-9                       # on the ray: distance 0
    # moved off the ray by d, perpendicular to it: the nearest ray point is d away
    C = c2w[v, :3, 3]
    r = (X - C) / (X - C).norm(dim=1, keepdim=True)
    side = torch.linalg.cross(r, torch.tensor([[0.0, 1.0, 0.0]], dtype=D).expand_as(r))
    side = side / side.norm(dim=1, keepdim=True)
    X2 = X + 0.37 * side
    near = held_ray_closest(w2c, K, vi, pix, X2)
    assert torch.allclose((near - X2).norm(dim=1), torch.full((len(X),), 0.37, dtype=D), atol=1e-9)


def test_a_point_behind_the_camera_is_held_to_the_centre():
    w2c, K, c2w, X = scene(n=5)
    vi = torch.zeros(5, dtype=torch.long)
    pix = project(K[0], w2c[0], X)
    behind = c2w[0, :3, 3] - (X - c2w[0, :3, 3])                             # mirrored through the centre
    near = held_ray_closest(w2c, K, vi, pix, behind)
    assert torch.allclose(near, c2w[0, :3, 3].expand_as(near), atol=1e-12)


def test_the_gradient_reaches_the_camera_never_the_fixed_point():
    w2c, K, _c2w, X = scene(n=50)
    w2c = w2c.clone().requires_grad_(True)
    vi = torch.ones(50, dtype=torch.long)
    pix = project(K[1], w2c[1].detach(), X) + 2.0                            # 2 px off: a residual to pull
    loss = (pix - held_reproj(K, w2c, vi, X)).norm(dim=1).sum()
    loss = loss + (held_ray_closest(w2c, K, vi, pix, X) - X).norm(dim=1).sum()
    loss.backward()
    assert X.grad is None and not X.requires_grad
    assert float(w2c.grad[1].abs().sum()) > 0 and float(w2c.grad[0].abs().sum()) == 0   # only the view it holds


def test_the_reprojection_term_alone_brings_a_started_off_camera_back():
    # a resection: what a held view's targets do to a new view, here through the stage-2 term with nothing else
    w2c, K, c2w, X = scene(n=300, seed=3)
    pix = project(K[1], w2c[1], X)
    vi = torch.zeros(len(X), dtype=torch.long)
    w = torch.zeros(6, dtype=D, requires_grad=True)                         # rotation vector, centre offset
    R0, C0 = c2w[1, :3, :3] @ rot([0, 0, 1], 4).T, c2w[1, :3, 3] + torch.tensor([0.4, -0.3, 0.5], dtype=D)
    opt = torch.optim.Adam([w], lr=0.01)
    steps = 1500
    for it in range(steps):
        for grp in opt.param_groups:                                         # cosine decay, as the solver's stages
            grp['lr'] = 0.01 * 0.5 * (1 + math.cos(math.pi * it / steps))
        opt.zero_grad()
        th = w[:3]
        a = th.norm().clamp(min=1e-12)
        k = torch.stack([torch.stack([torch.zeros((), dtype=D), -th[2], th[1]]),
                         torch.stack([th[2], torch.zeros((), dtype=D), -th[0]]),
                         torch.stack([-th[1], th[0], torch.zeros((), dtype=D)])]) / a
        R = (torch.eye(3, dtype=D) + torch.sin(a) * k + (1 - torch.cos(a)) * k @ k) @ R0
        C = C0 + w[3:]
        w2cv = torch.eye(4, dtype=D).clone()
        w2cv[:3, :3] = R.T
        w2cv[:3, 3] = -(R.T @ C)
        loss = (pix - held_reproj(K[1:2], w2cv[None], vi, X)).norm(dim=1).mean()
        loss.backward()
        opt.step()
    C_end = C0 + w[3:].detach()
    assert float((C_end - c2w[1, :3, 3]).norm()) < 1e-3, (C_end, c2w[1, :3, 3])
    assert float(loss.detach()) < 1e-2, float(loss.detach())


def test_the_given_gauge_is_returned_as_given_unknown_heights_from_the_model():
    c = torch.tensor([[0, 0, 0], [1, 0, 0.1], [2, 0.5, 0.2], [3, 1, 0.1]], dtype=D)
    s, R, t = 2.0, rot([0, 0, 1], 30), torch.tensor([10.0, 20.0, 200.0], dtype=D)
    enu = 2.0 * (c @ R.T) + t + torch.tensor([0.3, -0.2, 0.0], dtype=D)
    all_known = torch.ones(4, dtype=torch.bool)
    s_, R_, t_, fit = fixed_gauge(c, enu, all_known, (s, R, t))
    assert float(s_) == 2.0 and torch.equal(R_, R) and torch.equal(t_, t) and torch.equal(fit, enu)
    zok = torch.tensor([True, False, True, False])
    _s, _R, _t, fit2 = fixed_gauge(c, enu, zok, (s, R, t))
    rec = 2.0 * (c @ R.T) + t
    assert torch.equal(fit2[:, :2], enu[:, :2]) and torch.equal(fit2[zok, 2], enu[zok, 2])
    assert torch.allclose(fit2[~zok, 2], rec[~zok, 2])
    for bad in ((0.0, R, t), (2.0, R[:2], t), (2.0, R, t[:2])):
        try:
            fixed_gauge(c, enu, all_known, bad)
            raise AssertionError(f'a bad gauge was taken: {bad}')
        except ValueError:
            pass


def test_an_ordinary_run_is_untouched():
    assert sparse_ga.HELD_TARGETS is None and sparse_ga.FIXED_GAUGE is None
    src = inspect.getsource(sparse_ga.sparse_scene_optimizer)
    # every stage loss keeps its own expression where there are no targets
    assert src.count('if _held is None') + src.count('if _held is not None') >= 6
    assert '_with_held(' in src and 'HELD_TARGETS with a restore or pose_fixed' in src
    # the held start is its own branch: the placed start (initial_poses) is reached only without targets
    assert 'if _held is not None:\n                # THE START IS GIVEN' in src
    assert "HELD_TARGETS: the start is given in the targets' frame" in src


def test_the_held_start_is_taken_as_given_and_a_broken_pose_refused():
    import numpy as np
    g = np.tile(np.eye(4), (3, 1, 1))
    g[1, :3, :3] = rot([0, 1, 0], 30).numpy()
    g[2, :3, 3] = [5.0, -2.0, 40.0]
    assert np.array_equal(held_start(g, 3), g)
    for bad, why in ((g[:2], 'shape'), (np.where(np.eye(4, dtype=bool), np.nan, g), 'finite')):
        try:
            held_start(bad, 3)
        except ValueError as e:
            assert '(3, 4, 4)' in str(e), (why, e)
        else:
            raise AssertionError(why)
    refl = g.copy()
    refl[0, 0, 0] = -1.0                                        # a reflection, not a rotation
    try:
        held_start(refl, 3)
    except ValueError as e:
        assert 'view(s) [0]' in str(e)
    else:
        raise AssertionError('reflection')


def test_a_views_depth_scale_is_its_targets_over_its_own_depth_and_the_rest_take_the_median():
    import numpy as np
    sub, H, W = 8, 64, 48                                        # an 8 x 6 anchor grid
    rows, cols = len(range(sub // 2, H, sub)), len(range(sub // 2, W, sub))
    rng = np.random.default_rng(0)
    depth = [rng.uniform(1.0, 3.0, rows * cols) for _ in range(3)]
    w2c = np.tile(np.eye(4), (3, 1, 1))
    w2c[1, :3, 3] = [0.0, 0.0, -2.0]                             # view 1 sits 2 units back along its z
    K = np.array([[40.0, 0, W / 2], [0, 40.0, H / 2], [0, 0, 1.0]])
    hv, hp, hx = [], [], []
    for v, k in ((0, 3.0), (1, 5.0)):                            # the targets are k times the view's own depth
        pix = np.stack([rng.uniform(0, W, 30), rng.uniform(0, H, 30)], 1)
        cell = np.clip((pix[:, 1] // sub).astype(int), 0, rows - 1) * cols + np.clip((pix[:, 0] // sub).astype(int), 0, cols - 1)
        z = k * depth[v][cell]
        Xc = np.stack([(pix[:, 0] - K[0, 2]) / K[0, 0] * z, (pix[:, 1] - K[1, 2]) / K[1, 1] * z, z], 1)
        c2w = np.linalg.inv(w2c[v])
        hx.append(Xc @ c2w[:3, :3].T + c2w[:3, 3]); hp.append(pix); hv.append(np.full(30, v))
    held = (np.concatenate(hv), np.concatenate(hp), np.concatenate(hx), np.ones(60))
    logs, used = held_depth_scale(w2c, held, depth, [(H, W)] * 3, sub)
    assert used == {0: 30, 1: 30}
    assert np.allclose(np.exp(logs[:2]), [3.0, 5.0]) and np.isclose(np.exp(logs[2]), np.sqrt(15.0))
    try:
        held_depth_scale(w2c, held, [d[:-1] for d in depth], [(H, W)] * 3, sub)
    except ValueError as e:
        assert 'anchor grid' in str(e)
    else:
        raise AssertionError('a depth that is not the anchor grid')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
