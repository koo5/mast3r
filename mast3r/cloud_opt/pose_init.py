"""hillview: where the cameras START (--pose_init). Upstream starts every camera at the identity --
all of them on one spot, facing one way -- and lets the first steps of the coarse stage assemble the
walk out of that pile. A block held to the rest by weak ties can come out of it mirrored or twisted
(on a 40-panorama church ride, photographs 28/29: ~50 deg and 11 m off, already at the end of the coarse
stage).

    pile    the upstream start, unchanged
    visual  every photograph's pose from the pictures alone: a relative pose per photograph pair (the
            pair's matched 3D points, both already predicted in each camera's own frame), chained
            through the STRONGEST pairs (a maximum spanning tree over matching confidence, not the
            optimiser's kinematic tree), then turned so that it lies like the GPS track
    gps     the visual orientations, and every photograph that has a fix placed AT its fix
    poses   every photograph AT a pose given from outside (a pose graph's), in ENU metres; the
            pictures' own chain still sets the solver's unit (metres per canonical unit, from its fit
            to the GPS), so the depths keep their scale against the given spacing

A photograph is a rig (the slices of one panorama share a centre and turn together) or a single view.
Everything here is closed form, without gradient; the result is a list of absolute camera-to-world
poses in the solver's units (the canonical pointmaps'), which sparse_ga turns into kinematic-tree
parameters.
"""
from __future__ import annotations

import math

import numpy as np
import torch

MODES = ('pile', 'visual', 'gps', 'poses')


def sim3(src, dst, w):
    """weighted similarity with dst ~ s R src + t (numpy, float64)."""
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    S, D = src - mu_s, dst - mu_d
    U, d, Vt = np.linalg.svd((w[:, None] * D).T @ S)
    sg = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        sg[2] = -1
    R = U @ np.diag(sg) @ Vt
    s = float((d * sg).sum() / max((w * (S ** 2).sum(1)).sum(), 1e-30))
    return s, R, mu_d - s * R @ mu_s


def _down_to_z(g):
    """the smallest rotation taking unit `g` onto (0, 0, -1)."""
    g = g / max(np.linalg.norm(g), 1e-12)
    d = np.array([0.0, 0.0, -1.0])
    v = np.cross(g, d); c = float(g @ d)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def sim3_vertical(src, dst, w, g_src):
    """sim3 with the vertical known: tilt from gravity `g_src` (in src's frame), yaw and scale from
    the HORIZONTAL coordinates of dst, height offset from dst's z where finite (as sparse_ga's
    weighted_sim3_vertical; GPS heights are too noisy to choose a tilt)."""
    w = w / w.sum()
    Rt = _down_to_z(np.asarray(g_src, float))
    p = src @ Rt.T
    mu_p = w @ p
    mu_d = np.array([w @ dst[:, 0], w @ dst[:, 1], 0.0])
    S, D = p[:, :2] - mu_p[:2], dst[:, :2] - mu_d[:2]
    a = float((w * (S[:, 0] * D[:, 0] + S[:, 1] * D[:, 1])).sum())
    b = float((w * (S[:, 0] * D[:, 1] - S[:, 1] * D[:, 0])).sum())
    th = math.atan2(b, a)
    Rz = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
    s = math.hypot(a, b) / max(float((w * (S ** 2).sum(1)).sum()), 1e-30)
    R = Rz @ Rt
    t = mu_d - s * Rz @ mu_p
    zok = np.isfinite(dst[:, 2])
    if zok.any():
        rec = s * src[zok] @ R.T + t
        t[2] += float(np.average(dst[zok, 2] - rec[:, 2], weights=w[zok]))
    return s, R, t


def initial_poses(n_views, cam_pts, strong, weak, rig, gps, mode, imu_down=None, log=print,
                  weak_weight=0.01, given=None):
    """Absolute camera-to-world poses (n_views, 4, 4), float64, in the canonical pointmaps' units.

    cam_pts   per view, (A_i, 3) anchor points in that camera's own frame
    strong    [(i, j, idx_i, idx_j, conf)]: matched anchors of views i and j, one row per match
    weak      [(i, j, src_i, pts_i_in_j, conf)]: view i's points src_i (its own frame) and the same
              points as the pair predicts them in j's frame (the pointmap fallback's sources and
              targets, row for row), weighted `weak_weight` against matches -- the same 0.01 the
              objective gives them. The sources are explicit: the fallback compares
              anchor points, not cam_pts' correspondence points (sparse_ga's pointmap fallback)
    rig       per view (group, R_cam_to_rig) or None
    gps       (n_views, 3) ENU metres, NaN where unknown (z NaN: no altitude)
    imu_down  per view the measured down in the camera frame, or None: when >= 3 views carry one,
              the GPS fit takes its tilt from them, not from the heights
    given     mode 'poses' only: (n_views, 4, 4) camera-to-world in the frame of `gps` (ENU metres),
              every view; the views of one photograph must agree on its pose
    Returns (cam2w, report)."""
    assert mode in ('visual', 'gps', 'poses'), mode
    if (mode == 'poses') != (given is not None):
        raise ValueError(f'pose_init {mode}: given poses are for mode poses, and mode poses needs them')
    # ---- photographs
    ph_raw = [(r[0] if r is not None else ('view', i)) for i, r in enumerate(rig or [None] * n_views)]
    ids = {}
    photo = np.asarray([ids.setdefault(x, len(ids)) for x in ph_raw])
    n_ph = len(ids)
    R_c2r = [np.asarray(r[1], float) if r is not None else np.eye(3) for r in (rig or [None] * n_views)]
    if mode == 'poses':
        given = np.asarray(given, float)
        if given.shape != (n_views, 4, 4) or not np.isfinite(given).all():
            raise ValueError(f'pose_init poses: given poses {given.shape}, finite {np.isfinite(given).all()}; '
                             f'expected ({n_views}, 4, 4), all finite')
        Rg = given[:, :3, :3]
        bad = [v for v in range(n_views) if np.abs(Rg[v].T @ Rg[v] - np.eye(3)).max() > 1e-4 or np.linalg.det(Rg[v]) <= 0]
        if bad:
            raise ValueError(f'pose_init poses: view(s) {bad[:10]} carry no rotation (not orthonormal, or a reflection)')
        # a photograph's pose, from each of its views: R_photo = R_view R_c2r^T, one centre
        given_R, given_C = [None] * n_ph, [None] * n_ph
        for v in range(n_views):
            Rp, Cp = Rg[v] @ R_c2r[v].T, given[v, :3, 3]
            p_ = photo[v]
            if given_R[p_] is None:
                given_R[p_], given_C[p_] = Rp, Cp
            elif np.abs(given_R[p_] - Rp).max() > 1e-4 or np.linalg.norm(given_C[p_] - Cp) > 1e-3:
                raise ValueError(f'pose_init poses: view {v} disagrees with the other views of its photograph')
    P = [np.asarray(p.detach().cpu().double().numpy() if torch.is_tensor(p) else p, float) for p in cam_pts]
    # ---- every photograph pair's correspondences, in the two photographs' own frames
    acc = {}

    def add(a, b, xa, xb, w):
        if a == b or len(w) == 0:
            return
        if a > b:
            a, b, xa, xb = b, a, xb, xa
        e = acc.setdefault((a, b), {'strong': [[], [], []], 'weak': [[], [], []], 'ws': 0.0, 'ww': 0.0})
        return e, (a, b, xa, xb, w)
    for i, j, ii, jj, cf in strong:
        cf = np.asarray(cf.detach().cpu().double().numpy() if torch.is_tensor(cf) else cf, float)
        ii = np.asarray(ii.cpu() if torch.is_tensor(ii) else ii); jj = np.asarray(jj.cpu() if torch.is_tensor(jj) else jj)
        r_ = add(photo[i], photo[j], P[i][ii] @ R_c2r[i].T, P[j][jj] @ R_c2r[j].T, cf)
        if r_ is not None:
            e, (_, _, xa, xb, w) = r_
            for k_, x_ in enumerate((xa, xb, w)):
                e['strong'][k_].append(x_)
            e['ws'] += float(cf.sum())
    for i, j, src, pij, cf in weak:
        cf = np.asarray(cf.detach().cpu().double().numpy() if torch.is_tensor(cf) else cf, float) * weak_weight
        src = np.asarray(src.detach().cpu().double().numpy() if torch.is_tensor(src) else src, float)
        pij = np.asarray(pij.detach().cpu().double().numpy() if torch.is_tensor(pij) else pij, float)
        if not len(src) == len(pij) == len(cf):
            raise ValueError(f'weak tie {i}->{j}: {len(src)} sources, {len(pij)} targets, {len(cf)} weights')
        r_ = add(photo[i], photo[j], src @ R_c2r[i].T, pij @ R_c2r[j].T, cf)
        if r_ is not None:
            e, (_, _, xa, xb, w) = r_
            for k_, x_ in enumerate((xa, xb, w)):
                e['weak'][k_].append(x_)
            e['ww'] += float(cf.sum())
    # ---- a relative pose per photograph pair: X_a ~ s R X_b + t, so b sits at t in a's frame
    rel = {}
    for (a, b), e in acc.items():
        # MATCHES WHEN THERE ARE ANY: the pointmap fallback's prediction only stands in for a pair
        # the solver itself does not trust to match
        src_ = e['strong'] if e['ws'] > 0 else e['weak']
        ws, ww = e['ws'], e['ww']
        xa, xb, w = np.concatenate(src_[0]), np.concatenate(src_[1]), np.concatenate(src_[2])
        ok = np.isfinite(xa).all(1) & np.isfinite(xb).all(1) & (w > 0)
        if ok.sum() < 6:
            continue
        s, R, t = sim3(xb[ok], xa[ok], w[ok])
        res = np.linalg.norm(s * xb[ok] @ R.T + t - xa[ok], axis=1)
        rel[(a, b)] = dict(R=R, t=t, s=s, strength=ws if ws > 0 else ww, strong=ws, weak=ww,
                           resid_med=float(np.median(res)), n=int(ok.sum()))
    # ---- the strongest pairs first: a maximum spanning forest over the photographs
    parent = list(range(n_ph))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    tree = []
    # every pair with matches before any fallback-only pair, each strongest first
    for (a, b), r in sorted(rel.items(), key=lambda kv: (kv[1]['strong'] <= 0, -kv[1]['strength'])):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb; tree.append((a, b))
    comps = {}
    for p_ in range(n_ph):
        comps.setdefault(find(p_), []).append(p_)
    # ---- chain each component from its most connected photograph
    adj = {}
    for a, b in tree:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    Rph = [None] * n_ph; Cph = [None] * n_ph; via = [None] * n_ph; comp_of = np.zeros(n_ph, int)
    for k, members in enumerate(sorted(comps.values(), key=len, reverse=True)):
        root = max(members, key=lambda p_: sum(r['strength'] for (a, b), r in rel.items() if p_ in (a, b)))
        Rph[root], Cph[root] = np.eye(3), np.zeros(3)
        todo = [root]
        while todo:
            a = todo.pop()
            comp_of[a] = k
            for b in adj.get(a, []):
                if Rph[b] is not None:
                    continue
                if (a, b) in rel:      # b in a's frame
                    r = rel[(a, b)]; Rab, tab = r['R'], r['t']
                else:                  # a in b's frame: invert
                    r = rel[(b, a)]; Rab, tab = r['R'].T, -r['R'].T @ r['t']
                Rph[b] = Rph[a] @ Rab
                Cph[b] = Cph[a] + Rph[a] @ tab
                via[b] = (int(a), float(r['strength']), float(r['resid_med']))
                todo.append(b)
    n_comp = len(comps)
    # ---- turn each component to lie like the GPS track, and (gps) put photographs at their fixes
    gps = np.asarray(gps, float) if gps is not None else np.full((n_views, 3), np.nan)
    gph = np.full((n_ph, 3), np.nan)
    for v in range(n_views):
        if np.isfinite(gps[v, :2]).all() and not np.isfinite(gph[photo[v], :2]).all():
            gph[photo[v]] = gps[v]
    downs = [(v, np.asarray(d, float)) for v, d in enumerate(imu_down or []) if d is not None]
    report = dict(mode=mode, photos=n_ph, pairs=len(rel), components=n_comp, tree=[], comp=[])
    out_R, out_C = [None] * n_ph, [None] * n_ph
    s_ref = None
    for k in range(n_comp):
        mem = [p_ for p_ in range(n_ph) if comp_of[p_] == k]
        fix = [p_ for p_ in mem if np.isfinite(gph[p_, :2]).all()]
        C = np.stack([Cph[p_] for p_ in mem])
        if len(fix) >= 3 and np.ptp(gph[fix, :2], axis=0).max() > 1e-6:
            src = np.stack([Cph[p_] for p_ in fix]); dst = gph[fix]
            w = np.ones(len(fix))
            # THE TILT IS NEVER THE GPS HEIGHTS' to choose (18-21 deg off on a flat walk). It is the
            # measured or declared down where >= 3 views carry one, else the cameras held upright on
            # average (image +y is down) -- stated in the report either way
            g = [Rph[photo[v]] @ R_c2r[v] @ d for v, d in downs if comp_of[photo[v]] == k]
            if len(g) >= 3:
                tilt = f'measured/declared down on {len(g)} view(s)'
            else:
                g = [Rph[photo[v]] @ R_c2r[v] @ np.array([0.0, 1.0, 0.0]) for v in range(n_views)
                     if comp_of[photo[v]] == k]
                tilt = 'cameras upright on average (image +y down): no measured down'
            gd = np.mean(g, 0)
            for _ in range(5):        # Huber-reweighted, horizontally: one wild fix must not turn the walk
                s, R, t = sim3_vertical(src, dst, w, gd)
                r = np.linalg.norm((s * src @ R.T + t - dst)[:, :2], axis=1)
                w = np.where(r <= 5.0, 1.0, 5.0 / np.maximum(r, 1e-9))
        elif n_comp == 1 and mode == 'visual':
            s, R, t, tilt, r = 1.0, np.eye(3), np.zeros(3), 'none (no GPS)', np.zeros(0)
        else:
            raise ValueError(f'pose_init {mode}: component {k} ({len(mem)} photograph(s)) has {len(fix)} GPS fix(es); '
                             f'it cannot be placed against the others. Its photographs share no match with the rest.')
        if s_ref is None:
            s_ref = s                 # the solver's unit: the largest component's canonical unit
        world = s * C @ R.T + t       # metres, ENU
        for x, p_ in enumerate(mem):
            out_R[p_] = R @ Rph[p_]
            pos = world[x].copy()
            if mode == 'poses':
                out_R[p_] = given_R[p_]
                pos = given_C[p_].copy()
            elif mode == 'gps' and np.isfinite(gph[p_, :2]).all():
                pos[:2] = gph[p_, :2]
                if np.isfinite(gph[p_, 2]):
                    pos[2] = gph[p_, 2]
            out_C[p_] = pos / s_ref
        if len(fix):
            dis = np.linalg.norm((s * np.stack([Cph[p_] for p_ in fix]) @ R.T + t - gph[fix])[:, :2], axis=1)
        else:
            dis = np.zeros(0)
        if mode == 'poses':
            gd_ = [np.linalg.norm(world[x] - given_C[p_]) for x, p_ in enumerate(mem)]
            ga_ = [np.degrees(np.arccos(np.clip((np.trace((R @ Rph[p_]).T @ given_R[p_]) - 1) / 2, -1, 1)))
                   for p_ in mem]
            log(f' >> pose init (poses): component {k}: the given poses sit {np.median(gd_):.1f} m (median), '
                f'{max(gd_):.1f} m (worst) and {np.median(ga_):.1f} deg (median) from where the pictures put them')
        report['comp'].append(dict(photos=len(mem), fixes=len(fix), m_per_unit=float(s), tilt_from=tilt,
                                   pictures_vs_gps_m_median=float(np.median(dis)) if len(dis) else None,
                                   pictures_vs_gps_m_max=float(dis.max()) if len(dis) else None))
    for p_ in range(n_ph):
        if via[p_] is not None:
            report['tree'].append(dict(photo=p_, parent=via[p_][0], strength=via[p_][1], resid_med=via[p_][2]))
    cam2w = np.tile(np.eye(4), (n_views, 1, 1))
    for v in range(n_views):
        cam2w[v, :3, :3] = out_R[photo[v]] @ R_c2r[v]
        cam2w[v, :3, 3] = out_C[photo[v]]
    report['m_per_unit'] = float(s_ref)
    for k, c in enumerate(report['comp']):
        log(f' >> pose init ({mode}): component {k}: {c["photos"]} photograph(s), {c["fixes"]} GPS fix(es), '
            f'{c["m_per_unit"]:.3g} m per unit, tilt from {c["tilt_from"]}; the pictures put the photographs '
            + (f'{c["pictures_vs_gps_m_median"]:.1f} m (median), {c["pictures_vs_gps_m_max"]:.1f} m (worst) from their fixes'
               if c['pictures_vs_gps_m_median'] is not None else 'nowhere checkable (no fixes)'))
    log(f' >> pose init ({mode}): {n_ph} photograph(s), {len(rel)} photograph pair(s) with a relative pose, '
        f'chained through the strongest {len(report["tree"])}')
    return cam2w, report
