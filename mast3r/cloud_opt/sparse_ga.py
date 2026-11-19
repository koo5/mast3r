# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R Sparse Global Alignement
# --------------------------------------------------------
from tqdm import tqdm
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import json
import hashlib
from collections import namedtuple
from contextlib import nullcontext
import time
from functools import lru_cache
from scipy import sparse as sp
import copy
import scipy.cluster.hierarchy as sch

from mast3r.utils.misc import mkdir_for, hash_md5
from mast3r.cloud_opt.utils.losses import gamma_loss
from mast3r.cloud_opt.utils.schedules import linear_schedule, cosine_schedule
from mast3r.fast_nn import fast_reciprocal_NNs, merge_corres, NNIndexOutOfRange

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import inv, geotrf  # noqa
from dust3r.utils.device import to_cpu, to_numpy, todevice  # noqa
from dust3r.post_process import estimate_focal_knowing_depth  # noqa
from dust3r.optim_factory import adjust_learning_rate_by_lr  # noqa
from dust3r.cloud_opt.base_opt import clean_pointcloud
from dust3r.viz import SceneViz


class SparseGA():
    def __init__(self, img_paths, pairs_in, res_fine, anchors, canonical_paths, point_partners):
        def fetch_img(im):
            def torgb(x): return (x[0].permute(1, 2, 0).numpy() * .5 + .5).clip(min=0., max=1.)
            for im1, im2 in pairs_in:
                if im1['instance'] == im:
                    return torgb(im1['img'])
                if im2['instance'] == im:
                    return torgb(im2['img'])
        self.canonical_paths = canonical_paths
        # hillview: [PointPartners] -- which anchor of which view is the other end of each sparse
        # point's correspondence (sparse_global_alignment)
        self.point_partners = point_partners
        self.img_paths = img_paths
        self.imgs = [fetch_img(img) for img in img_paths]
        self.intrinsics = res_fine['intrinsics']
        self.cam2w = res_fine['cam2w']
        self.depthmaps = res_fine['depthmaps']
        self.pts3d = res_fine['pts3d']
        self.pts3d_colors = []
        self.working_device = self.cam2w.device
        for i in range(len(self.imgs)):
            im = self.imgs[i]
            x, y = anchors[i][0][..., :2].detach().cpu().numpy().T
            self.pts3d_colors.append(im[y, x])
            assert self.pts3d_colors[-1].shape == self.pts3d[i].shape
        self.n_imgs = len(self.imgs)

    def get_focals(self):
        return torch.tensor([ff[0, 0] for ff in self.intrinsics]).to(self.working_device)

    def get_principal_points(self):
        return torch.stack([ff[:2, -1] for ff in self.intrinsics]).to(self.working_device)

    def get_im_poses(self):
        return self.cam2w

    def get_sparse_pts3d(self):
        return self.pts3d

    def get_dense_pts3d(self, clean_depth=True, subsample=8):
        assert self.canonical_paths, 'cache_path is required for dense 3d points'
        device = self.cam2w.device
        confs = []
        base_focals = []
        anchors = {}
        for i, canon_path in enumerate(self.canonical_paths):
            (canon, canon2, conf), focal = _cache_load(_found(canon_path), device)
            confs.append(conf)
            base_focals.append(focal)

            H, W = conf.shape
            pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
            idxs, offsets = anchor_depth_offsets(canon2, {i: (pixels, None)}, subsample=subsample)
            anchors[i] = (pixels, idxs[i], offsets[i])

        # densify sparse depthmaps
        pts3d, depthmaps = make_pts3d(anchors, self.intrinsics, self.cam2w, [
                                      d.ravel() for d in self.depthmaps], base_focals=base_focals, ret_depth=True)

        if clean_depth:
            confs = clean_pointcloud(confs, self.intrinsics, inv(self.cam2w), depthmaps, pts3d)

        return pts3d, depthmaps, confs

    def get_pts3d_colors(self):
        return self.pts3d_colors

    def get_depthmaps(self):
        return self.depthmaps

    def get_masks(self):
        return [slice(None, None) for _ in range(len(self.imgs))]

    def show(self, show_cams=True):
        pts3d, _, confs = self.get_dense_pts3d()
        show_reconstruction(self.imgs, self.intrinsics if show_cams else None, self.cam2w,
                            [p.clip(min=-50, max=50) for p in pts3d],
                            masks=[c > 1 for c in confs])


def convert_dust3r_pairs_naming(imgs, pairs_in):
    for pair_id in range(len(pairs_in)):
        for i in range(2):
            pairs_in[pair_id][i]['instance'] = imgs[pairs_in[pair_id][i]['idx']]
    return pairs_in


def angle_between(u, v):
    """Angle in RADIANS between two unit vectors, stable at 0 and at pi.

    arccos(u.v) needs its argument clamped away from +-1 to keep the gradient finite, and that
    clamp puts a FLOOR under the reading -- the trap recon_gravity_check.py caught in the gravity
    term, where a perfectly level set read 0.026 deg instead of 0. This form is exact everywhere,
    and the epsilons keep norm()'s backward off an exact zero (antipodal, or identical, vectors).
    """
    a = torch.sqrt((u - v).pow(2).sum(-1) + 1e-12)
    b = torch.sqrt((u + v).pow(2).sum(-1) + 1e-12)
    return 2.0 * torch.atan2(a, b)


CMP_MAX_TILT = 0.996          # cos(85 deg): nearer the up axis than this and a forward has no yaw


def consensus_up(cam2w, sel):
    """The direction these views agree is up. A DIRECTION, not a gauge: no alignment, no north."""
    up = -cam2w[sel, :3, 1]                               # OpenCV camera axes: y is down
    up = up / up.norm(dim=1, keepdim=True).clamp(min=1e-9)
    m = up.mean(dim=0)
    return m / m.norm().clamp(min=1e-9)


def horizontal(v, up):
    """v with its component along `up` removed, renormalised: the part a compass can see."""
    h = v - (v * up).sum(-1, keepdim=True) * up
    return h / torch.sqrt(h.pow(2).sum(-1, keepdim=True) + 1e-12)


def compass_penalty(cam2w, i, j, tgt, slack, up_sel, max_pitch_deg=50.0, ramp_deg=20.0):
    """Mean squared YAW disagreement beyond `slack`, in radians, over pairs (i, j) whose compass
    says they stand `tgt` apart in bearing. Module level so recon_compass_check.py drives this
    exact code rather than a copy of it.

    A TILTED PHONE HAS NO USABLE BEARING: the bearing loses accuracy and eventually goes silly as
    the physical phone's pitch approaches looking straight up or down, and any pitch over 50 degrees
    from the horizon leaves it a mess.

    So this is a THRESHOLD, not a gentle decay. A first cut weighted each pair by cos(pitch) --
    the length of the device axis projected onto the horizontal plane, which is the geometry of how
    an azimuth is formed -- and that is too generous: it still gives a pair at 70 degrees a third of
    a vote, where the capture phone gives nothing worth having past 50. The projection
    argument describes the instrument; it does not describe the magnetometer and the tilt
    compensation behind it going to pieces.

    The cut is ramped over `ramp_deg` rather than dropped, because a step has no gradient and a
    camera sitting exactly on the boundary would flip in and out between iterations. Full weight to
    `max_pitch_deg - ramp_deg`, nothing past `max_pitch_deg`, linear between -- which at the
    defaults 50 and 20 is the shape wanted: full confidence to about +-30 from the horizon, then a
    decline.
    """
    up = consensus_up(cam2w, up_sel)
    f1 = cam2w[i, :3, 2]                                  # +Z is forward in these poses
    f2 = cam2w[j, :3, 2]
    f1 = f1 / f1.norm(dim=1, keepdim=True).clamp(min=1e-9)
    f2 = f2 / f2.norm(dim=1, keepdim=True).clamp(min=1e-9)
    hi = math.radians(max_pitch_deg)
    lo = math.radians(max(0.0, max_pitch_deg - max(ramp_deg, 1e-6)))

    def usable(f):
        # pitch away from horizontal: 0 level, pi/2 at the pole
        pitch = (f * up).sum(-1).clamp(-1.0, 1.0).abs().asin()
        return ((hi - pitch) / (hi - lo)).clamp(0.0, 1.0)

    w = usable(f1) * usable(f2)
    if float(w.sum()) <= 0.0:
        return cam2w.sum() * 0.0
    ang = angle_between(horizontal(f1, up), horizontal(f2, up))
    pen = ((ang - tgt).abs() - slack).clamp(min=0.0).pow(2)
    return (pen * w).sum() / w.sum().clamp(min=1e-6)


def level_penalty(cam2w, sel, target, slack):
    """Mean squared departure beyond `slack` from a forward axis sitting `target` radians from the
    views' consensus up. `target` is 90 deg minus the wanted pitch, so -20 deg of pitch is 110."""
    up = consensus_up(cam2w, sel)
    fwd = cam2w[sel, :3, 2]
    fwd = fwd / fwd.norm(dim=1, keepdim=True).clamp(min=1e-9)
    ang = angle_between(fwd, up.unsqueeze(0).expand_as(fwd))
    return ((ang - target).abs() - slack).clamp(min=0.0).pow(2).mean()


def height_penalty(cam2w, sel, slack_steps):
    """Mean squared departure of a camera's HEIGHT from the walk's median, beyond a slack.

    THE ERROR THIS IS FOR. The site: a 40-frame approach of about 40 metres, then a small picnic
    place of maybe 7 m radius, on level-ish terrain, the phone held in front of the walker. So camera
    height is CONSTANT along this walk, to within the bob of a hand. The solve
    made it a twelve-metre climb: frame 0 sat 12.24 m below frame 40 and rose monotonically to
    +0.13 by frame 36. Smooth and one-directional, so drift, not noise -- and it flattened exactly
    where the GPS votes began, because --gps_drop had removed the first twenty frames' and NOTHING
    ELSE in the objective holds a camera's height. loss_level constrains where a camera POINTS; it
    says nothing about where it IS.

    IN UNITS OF THE WALK'S OWN STEP, because the optimiser has no metres: scale_units_per_m is
    fitted afterwards, from the GPS, and this term must work on a run with no GPS at all. The median
    distance between consecutive cameras IS a step, so a slack of 0.5 steps is about half a pace of
    vertical freedom whatever the gauge. Measured under no_grad so the unit does not chase the
    gradient.

    ALONG THE CONSENSUS UP, the same direction loss_level uses -- no alignment, no north, no
    gravity vector. Like every other term here it cannot see a rigid rotation of the whole solve,
    which is the alignment's business.
    """
    up = consensus_up(cam2w, sel)
    c = cam2w[sel, :3, 3]
    h = (c * up).sum(-1)
    with torch.no_grad():
        step = (c[1:] - c[:-1]).norm(dim=1).median().clamp(min=1e-6)
        slack = slack_steps * step
        mid = h.median()
    return ((h - mid).abs() - slack).clamp(min=0.0).pow(2).mean()


def grade_penalty(rec, slack, min_step_frac=0.1):
    """Mean squared departure of the CONSECUTIVE-FRAME GRADE beyond `slack`, dimensionless.

    A drive has some maximum altitude difference between two immediately consecutive frames.

        grade_i = (rec[i+1] - rec[i]) . up / |rec[i+1] - rec[i]|        = sin(slope angle)

    A ratio, so it needs no scale and cannot be mis-weighted the way a metres-valued term can
    (loss_gps reached 140x the pixel objective); bounded in [-1, 1], so a pair that barely moved
    cannot blow it up; and LOCAL, so a real hill and a slow vertical drift both cost nothing.

    `rec` is already in the GPS GAUGE, so ENU +Z is up. It cannot use the consensus up that
    height_penalty and upright_penalty use: on a solve whose roll is scattered that mean is not
    vertical, and it reads 12.3 % median grade where the gauge reads 2.7 %.

    Slack 0.25 is measured off the DEM, not chosen, for the
    separation it buys. Pairs shorter than `min_step_frac` of the walk's median step are
    dropped: they say nothing about grade, and it keeps a panorama's slices, which share a
    centre, from voting.
    """
    if rec is None or len(rec) < 3:
        return rec.sum() * 0.0 if rec is not None else 0.0
    d = rec[1:] - rec[:-1]
    step = d.norm(dim=1)
    with torch.no_grad():
        keep = step > (min_step_frac * step.median().clamp(min=1e-9))
    if not bool(keep.any()):
        return rec.sum() * 0.0
    grade = d[:, 2] / step.clamp(min=1e-9)
    pen = (grade.abs() - slack).clamp(min=0.0).pow(2)
    return (pen * keep).sum() / keep.sum().clamp(min=1)


def axis_target(R_held, down):
    """The mount consensus: the mean of every held view's declared down, in the model frame,
    unit length, DETACHED (hillview). sparse_scene_optimizer takes it ONCE, at the start
    pose, as the injected axis, and holds every mount to that constant (it was once re-taken at
    every evaluation, which made a whole-solve tilt free). Module level so
    recon_axis_check.py checks the function the optimiser calls."""
    with torch.no_grad():
        g = (R_held.detach().double() @ down.double()[:, :, None])[:, :, 0]
        m = g.mean(0)
        return m / m.norm().clamp(min=1e-12)


def axis_rig_attitude(R_cams, D, cos_yaw, sin_yaw):
    """The MOUNT's (roll, pitch) off a target axis, radians, per view (hillview).

    `R_cams` (N,3,3) cam->world, `D` (3,) the target axis (the injected axis, fixed at the start
    pose by sparse_scene_optimizer), `cos_yaw`/`sin_yaw` (N,) the slice's own
    cut about the rig's up. Module level and not a closure because recon_axis_check.py checks THIS
    function -- a test of a re-derivation beside the code is a test of the re-derivation.

    The target axis lands in each camera's frame (`R^T D`), then is yawed back into the RIG's
    (`R_y(theta) @ .`, matching reconstruct.R_rel's `R_y(yaw) @ R_x(pitch)` for a pitch-0 slice),
    where +x is the mount's right and +z its forward. So the x component is ROLL and the z
    component PITCH, both measured against the +y component, which is +1 for a level mount.
    atan2 and not |sin|: a flipped mount then costs 180 deg instead of nothing.
    """
    v = (R_cams.transpose(1, 2) @ D[None, :, None])[:, :, 0]
    cc, ss = cos_yaw.to(v.dtype), sin_yaw.to(v.dtype)
    rx = cc * v[:, 0] + ss * v[:, 2]
    rz = -ss * v[:, 0] + cc * v[:, 2]
    vy = v[:, 1]
    vy = vy + 1e-12 * (vy == 0).to(vy.dtype)      # only a mount pitched EXACTLY 90 deg reaches this
    return torch.atan2(rx, vy), torch.atan2(rz, vy)


class InjectedAxis:
    """THE AXIS WE INJECT, held fixed (hillview): the held views' declared down
    at the FIRST call -- the start pose, before any optimiser step -- and never again. Each later
    call gives every held view's mount (roll, pitch) off that constant, so a tilt of the whole solve
    costs like any other. Every 100th call logs how far the mounts' own consensus has turned off it:
    the whole-solve tilt the term now prices, which the re-taken target let drift 25-151 deg across
    the axis arms of one 28-panorama town span. Module level so recon_axis_check.py checks the object the
    optimiser calls."""

    def __init__(self, down, cos_yaw, sin_yaw):
        self.down, self.cos_yaw, self.sin_yaw = down, cos_yaw, sin_yaw
        self.axis = None
        self.n = 0

    def __call__(self, R_held):
        self.n += 1
        if self.axis is None:
            self.axis = D = axis_target(R_held, self.down)
            print(f' >> injected axis FIXED at the start pose: ({D[0]:+.4f}, {D[1]:+.4f}, {D[2]:+.4f}), '
                  f'the mount consensus there; it does not move again')
        elif self.n % 100 == 0:
            m = axis_target(R_held, self.down)
            moved = float(torch.rad2deg(torch.arccos((m * self.axis).sum().clamp(-1.0, 1.0))))
            print(f' >> injected axis: the mount consensus is {moved:.1f} deg off it (evaluation {self.n})')
        return axis_rig_attitude(R_held, self.axis.to(R_held.dtype), self.cos_yaw, self.sin_yaw)


def rig_upright_penalty(right_enu, slack_sin):
    """THE MOUNT IS UPRIGHT, against the world rather than against itself.

    What is known of the capture: Panoramax walks are shot with cameras held (mostly)
    straight. The only ORIENTATION evidence here that does not come out
    of the solve — upright_penalty holds a camera near the CONSENSUS up (and says in its own
    docstring that a wholly inverted solve satisfies it), gravity_weight holds rigs parallel to
    each other, and the post-solve realign estimated up from the solved rotations or cloud.

    THE RIGHT AXIS, NOT THE UP AXIS. Pitch is a slice's business: pano_pitches cuts views that
    look up and down by construction, so their up axes are tilted on purpose, while pitching
    about the right axis leaves that axis horizontal. `right . up_world = 0` therefore holds for
    every slice of a level panorama at any yaw and any pitch.

    `right_enu` is each view's right axis already in the GPS gauge; the penalty is the sine of
    its angle off horizontal beyond `slack_sin`. Dimensionless, bounded in [0, 1]. It pins ROLL
    and not a 180-degree flip, which stays upright_penalty's job.

    MEASURED AND REFUTED on a 1-slice-no-rig run: it made roll_std worse, because it can only
    twist cameras individually while the gauge is detached.
    """
    if right_enu is None or len(right_enu) < 1:
        return None
    off = right_enu[:, 2].abs()                       # |sin(elevation of the right axis)|
    return (off - slack_sin).clamp(min=0.0).pow(2).mean()


def upright_penalty(cam2w, sel, slack):
    """Mean squared departure of a camera's UP from the walk's consensus up, beyond `slack` radians.

    THE REQUIREMENT: the solver must not let the cameras think the walk was made on the flip side,
    upside down like an Australian.

    loss_level does not cover this and cannot: it measures the angle between a camera's FORWARD axis
    and the consensus up, and rolling a camera about its own forward axis leaves that angle exactly
    unchanged. A camera can be upside down and perfectly level by that measure. This term is about
    the OTHER axis -- how far a camera's own up has fallen from everyone else's.

    WHAT IT CANNOT DO, said plainly: the consensus up is a mean over the cameras themselves, so if
    the WHOLE solve is inverted this term is perfectly happy -- every camera agrees with every
    other, they are simply all upside down together. recon_compass_check.py proves that invariance
    for the sibling terms and it holds here. Catching a global inversion needs an absolute
    reference, and the only ones this capture has are the GPS altitudes (60 of 80 frames) and the
    sky masks, which are already computed for every frame and say which way is up in the IMAGE.
    Neither is wired to this; the note is here so the gap is not mistaken for coverage.
    """
    up = -cam2w[sel, :3, 1]                               # OpenCV camera axes: y is down
    up = up / up.norm(dim=1, keepdim=True).clamp(min=1e-9)
    mean = up.mean(dim=0)
    mean = mean / mean.norm().clamp(min=1e-9)
    ang = angle_between(up, mean.unsqueeze(0).expand_as(up))
    return (ang - slack).clamp(min=0.0).pow(2).mean()


def horizon_penalty(cam2w, sel, max_pitch, slack):
    """Mean squared pitch ABOVE `max_pitch` radians, beyond `slack`. One-sided: looking down is free.

    THE SHAPE MATTERS, and the symmetric level prior had it wrong. Pitch stays
    between the horizon and straight down -- so the truth about a walk like this is
    a CEILING, not a target. level_weight holds a camera near 20 degrees below the horizon with a
    slack either side, which permits a few degrees ABOVE it and simultaneously argues with the
    frames that are legitimately steep -- a photographer standing on a table, looking down.

    Why it matters: measured over the runs judged good, the solved pitches run -55 to
    +22 degrees with FORTY-FIVE PERCENT of frames above the horizon, and removing the median as a
    bias leaves 49 percent. So this is not the alignment's up being tilted -- it is the solve
    splaying pitch both ways. Frames aimed up build a surface above, frames aimed down build one
    below, and that is the "double ground plane" that appears in verdict after verdict.

    Free below the ceiling, quadratic above it. `slack` keeps the hinge off an exact boundary so a
    frame sitting at the horizon does not chatter between iterations.
    """
    up = consensus_up(cam2w, sel)
    fwd = cam2w[sel, :3, 2]
    fwd = fwd / fwd.norm(dim=1, keepdim=True).clamp(min=1e-9)
    # + is above the horizon, measured against the views' own consensus up
    pitch = (fwd * up).sum(-1).clamp(-1.0, 1.0).asin()
    return (pitch - max_pitch - slack).clamp(min=0.0).pow(2).mean()


def sparse_global_alignment(imgs, pairs_in, cache_path, model, subsample=8, desc_conf='desc_conf',
                            kinematic_mode='hclust-ward', device='cuda', dtype=torch.float32, shared_intrinsics=False,
                            gpu_stage=None, canonical_cache=None, image_keys=None, inference_done=None, optimizer_phase=None, preparation_stats=None,
                            matching_only=False, pointmap_masks=None, dense_masks=None, **kw):
    """ Sparse alignment with MASt3R
        imgs: list of image paths
        cache_path: path where to dump temporary files (str)

        lr1, niter1: learning rate and #iterations for coarse global alignment (3D matching)
        lr2, niter2: learning rate and #iterations for refinement (2D reproj error)

        lora_depth: smart dimensionality reduction with depthmaps
    """
    # hillview: a pointmap policy other than upstream's needs the caller's (POINTMAP_POLICY): refused here, before the
    # forward passes, not after them. It is evaluated once, by condense_data; the optimiser gets its decisions
    pointmap_policy = kw.pop('pointmap_policy', None)
    if 'pointmap_selection' in kw:
        raise ValueError('pointmap_selection is made here, by condense_data: pass pointmap_policy instead')
    if pointmap_policy is None and kw.get('pointmap_pairs', 'default') != 'default':
        raise ValueError(f"pointmap_pairs {kw['pointmap_pairs']!r} needs a pointmap_policy: this module knows only "
                         f"upstream's rule, 'default'")
    if gpu_stage is not None:
        gpu_stage('matching pairs')
    # Convert pair naming convention from dust3r to mast3r
    pairs_in = convert_dust3r_pairs_naming(imgs, pairs_in)
    # forward pass
    # hillview: stage lines the bench turns into names ("matching pairs 2297/5272")
    print(f' >> matching pairs: {len(pairs_in)} directed pair(s); one forward pass per image pair, '
          f'the reverse direction comes from the cache')
    _t_phase = time.time()
    pairs, cache_path = forward_mast3r(pairs_in, model,
                                       cache_path=cache_path, subsample=subsample,
                                       desc_conf=desc_conf, device=device)

    print(f' >> matching pairs done in {time.time() - _t_phase:.0f} s')
    # Cache paths are the handoff: subsequent preparation/optimisation needs no
    # network weights. A lazy handle may reload on a later round's cache miss.
    if inference_done is not None:
        inference_done()
    # hillview: a MATCHING job stops here (--stop_after match): its product is the forward passes
    # and matches now in the cache, made by the very call a solve makes, so a solve elsewhere finds
    # them under the same keys. Canonical views are not part of it: they are keyed by device and
    # by this run's pair set, and cost seconds to a few minutes.
    if matching_only:
        return None
    # hillview: the pairs that reached the solve, past the caller's pair gates -- what a
    # refusing kinematic tree tells apart in its account of the cut (DisconnectedGraph.name_views)
    pairs_reached = list(pairs)

    def named_tree(build, *args_, **kws_):
        try:
            return build(*args_, **kws_)
        except DisconnectedGraph as e:
            ix = {p_: k_ for k_, p_ in enumerate(imgs)}
            e.name_views(imgs, [(p1['idx'], p2['idx']) for p1, p2 in pairs_in],
                         [(ix[a_], ix[b_]) for a_, b_ in pairs_reached])
            raise

    # extract canonical pointmaps
    print(f' >> canonical views: {len(imgs)} image(s)')
    _t_phase = time.time()
    tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21 = \
        prepare_canonical_data(imgs, pairs, subsample, cache_path=cache_path, mode='avg-angle', device=device,
                               shared_cache=canonical_cache, image_keys=image_keys, restore=kw.get('restore'))

    print(f' >> canonical views done in {time.time() - _t_phase:.0f} s')
    # Gate in the allocating process, before optimiser buffers and tensors are created.
    # The full reservation stays held through stage 2, dense extraction and late DA3.
    if gpu_stage is not None:
        gpu_stage('coarse alignment')
    # Matching has finished: resolve measurements once for preparation and loss selection.
    if callable(kw.get('pointmap_pair_inputs')):
        kw['pointmap_pair_inputs'] = kw['pointmap_pair_inputs']()
    # ...and the caller's WEAK photographs, decided from the pairs that reached the solve (weak-frame holds)
    if callable(kw.get('weak_views')):
        kw['weak_views'] = kw['weak_views']()
    if preparation_stats is not None:
        preparation_stats['pointmap_candidate_pairs'] = sum(len(row) for row in preds_21.values())
    # smartly combine all useful data
    _selection = []
    imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21 = \
        condense_data(imgs, tmp_pairs, canonical_views, preds_21, dtype,
                      pointmap_pairs=kw.get('pointmap_pairs', 'default'),
                      pointmap_pair_inputs=kw.get('pointmap_pair_inputs'),
                      matching_conf_thr=kw.get('matching_conf_thr', 5.),
                      pointmap_masks=pointmap_masks, subsample=subsample,
                      pointmap_policy=pointmap_policy, selection_out=_selection)
    kw['pointmap_selection'] = _selection
    # hillview: EVERY SPARSE POINT IS ONE END OF ONE CORRESPONDENCE. View i's anchors are
    # its correspondences with each partner concatenated (condense_data), so slice1 of view img1 and
    # slice2 of view img2 are the two ends of the same matches, index for index. Kept here, on the
    # CPU, because the optimiser may run in another process and `corres` is dropped below; the
    # caller turns them into per-point quality columns (hillview solvers.sparse_point_quality).
    point_partners = [PointPartners(int(s.img1), s.slice1, s.pix1.detach().to('cpu', torch.float32),
                                    int(s.img2), s.slice2, s.pix2.detach().to('cpu', torch.float32),
                                    s.confs.detach().to('cpu', torch.float32),
                                    pair_matching_ok(s.confs, kw.get('matching_conf_thr', 5.)))
                      for s in corres[2]]
    if preparation_stats is not None:
        preparation_stats.update(
            pointmap_selected_pairs=sum(len(row) for row in preds_21.values()),
            pointmap_target_storage_bytes=tensor_storage_bytes(preds_21),
            condensed_tensor_storage_bytes=tensor_storage_bytes(
                (imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21)))
    # These inputs are superseded by condensed tensors. Keeping canonical_views
    # alive retains per-pair pixels/confidences and full-resolution preparation
    # storage throughout both optimisation stages.
    del canonical_views, tmp_pairs, pairs

    # hillview: DENSE cross-view depth consistency (stage 2). A regular grid of sample
    # pixels per image, attached to the anchors like correspondences are, plus per-image
    # full-resolution anchor-index / offset maps so another view's depth can be read at
    # any pixel a projected point lands on. Both derive from the canonical depth, so a
    # blank wall that no correspondence ever touched still gets a say.
    dense_kw = {}
    if kw.get('dense_consistency_w', 0):
        stride = int(kw.get('dense_stride', 4))
        dense_anchors, dense_maps = {}, {}
        for i, canon_path in enumerate(canonical_paths):
            (canon, canon2, conf), focal = _cache_load(_found(canon_path), device)
            H, W = canon2.shape
            pix = torch.from_numpy(np.mgrid[stride // 2:W:stride, stride // 2:H:stride].T.reshape(-1, 2)).float().to(device)
            idxs, offsets = anchor_depth_offsets(canon2, {i: (pix, None)}, subsample=subsample)
            dense_anchors[i] = (pix, idxs[i], offsets[i])
            allpix = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
            aidx, aoff = anchor_depth_offsets(canon2, {i: (allpix, None)}, subsample=subsample)
            dense_maps[i] = (aidx[i].reshape(H, W), aoff[i].reshape(H, W).to(dtype), H, W)
        # hillview: ONLY WHAT THE PAIR'S MATCHING READS. Every directed pair the optimiser can
        # make a dense pair of (corres' slices) gets its two masks: a source sample on the source's mask
        # never enters (dense_src_keep), and one landing on the target's mask is dropped in loss_dense.
        # Precomputed as tensors: the optimiser may run in another process, where no callable reaches.
        dense_view_masks, dense_pair_var, dense_src_keep = {}, {}, {}
        if dense_masks is not None:
            _dpairs = sorted({(int(s_.img1), int(s_.img2)) for s_ in corres[2]})
            dense_view_masks, dense_pair_var = dense_pair_masks(
                _dpairs, imgs, {v: dense_maps[v][2:4] for v in dense_maps}, dense_masks, device)
            n_all = n_drop = 0
            for (i, j), (vi_, _vj) in dense_pair_var.items():
                keep = dense_source_keep(dense_anchors[i][0], dense_view_masks.get(i), vi_)
                dense_src_keep[(i, j)] = keep
                n_all += len(dense_anchors[i][0])
                n_drop += 0 if keep is None else len(dense_anchors[i][0]) - len(keep)
            print(f' >> dense term masks: {len(dense_pair_var)} directed pair(s), masks on {len(dense_view_masks)} '
                  f'view(s); {n_drop} of {n_all} source samples ({100 * n_drop / max(n_all, 1):.1f} %) sit on their '
                  f'pair\'s masked pixels and never enter; a sample landing on the target\'s mask is dropped per step')
        dense_kw = dict(dense_anchors=dense_anchors, dense_maps=dense_maps,
                        dense_masks_given=dense_masks is not None, dense_view_masks=dense_view_masks,
                        dense_pair_var=dense_pair_var, dense_src_keep=dense_src_keep,
                        dense_consistency_w=float(kw['dense_consistency_w']),
                        dense_gate=float(kw.get('dense_gate', 0.25)),
                        dense_consistency_mode=str(kw.get('dense_consistency_mode', 'symmetric')))
    for k_ in ('dense_consistency_w', 'dense_stride', 'dense_gate', 'dense_consistency_mode'):
        kw.pop(k_, None)

    # Build kinematic chain
    # (hillview) WITH A RIG, mst AND shallow ARE BUILT OVER THE RIGS, NOT THE VIEWS.
    # The view-level tree below used to be computed first and then thrown away by
    # rig_spanning_tree -- and computing it RAISED whenever the views were in pieces that the rig
    # joins: with pano_ring_pairs 0 the slices of one panorama meet only through other panoramas,
    # and at 4x90 under slice_dang 70 each yaw band is its own chain. An --unpaired
    # e2e run died in exactly that compute_min_spanning_tree, with every photograph connected.
    _rig_peek = kw.get('rig')
    _rig_tree = (_rig_peek is not None and any(r is not None for r in _rig_peek)
                 and kinematic_mode in ('mst', 'shallow'))
    if _rig_tree:
        mst = None                        # built below by rig_spanning_tree from the raw scores

    elif kinematic_mode == 'mst':
        # compute minimal spanning tree
        mst = named_tree(compute_min_spanning_tree, pairwise_scores)

    elif kinematic_mode.startswith('hclust'):
        mode, linkage = kinematic_mode.split('-')

        # Convert the affinity matrix to a distance matrix (if needed)
        n_patches = (imsizes // subsample).prod(dim=1)
        max_n_corres = 3 * torch.minimum(n_patches[:,None], n_patches[None,:])
        pws = (pairwise_scores.clone() / max_n_corres).clip(max=1)
        pws.fill_diagonal_(1)
        pws = to_numpy(pws)
        distance_matrix = np.where(pws, 1 - pws, 2)

        # Compute the condensed distance matrix
        condensed_distance_matrix = sch.distance.squareform(distance_matrix)

        # Perform hierarchical clustering using the linkage method
        Z = sch.linkage(condensed_distance_matrix, method=linkage)
        # dendrogram = sch.dendrogram(Z)

        tree = np.eye(len(imgs))
        new_to_old_nodes = {i:i for i in range(len(imgs))}
        for i, (a, b) in enumerate(Z[:,:2].astype(int)):
            # given two nodes to be merged, we choose which one is the best representant
            a = new_to_old_nodes[a]
            b = new_to_old_nodes[b]
            tree[a,b] = tree[b,a] = 1
            best = a if pws[a].sum() > pws[b].sum() else b
            new_to_old_nodes[len(imgs)+i] = best
            pws[best] = np.maximum(pws[a], pws[b]) # update the node

        pairwise_scores = torch.from_numpy(tree) # this output just gives 1s for connected edges and zeros for other, i.e. no scores or priority
        mst = compute_min_spanning_tree(pairwise_scores)

    elif kinematic_mode == 'shallow':
        # hillview: the shallowest tree the pair graph allows (min_depth_spanning_tree), against
        # the deepest one the scores produce on a walk. Rigged runs take the same route through
        # rig_spanning_tree below.
        mst = named_tree(min_depth_spanning_tree, pairwise_scores)

    elif kinematic_mode == 'chain':
        # hillview: a sequential chain in image order -- for a WALK, each camera relative to
        # the one captured before it. The clustering above builds the tree from match counts
        # and produces hubs and jumps: on one town walk frame 10 hung off 11, frame 9 off 8,
        # and the path between two consecutive frames was 9-8-11-10, with 8-11 three frames
        # long; every solve of that span then stepped BACK at frame 10. The
        # chain follows the capture; the pairs still all enter the loss.
        tree = np.eye(len(imgs))
        for i in range(len(imgs) - 1):
            tree[i, i + 1] = tree[i + 1, i] = 1
        pairwise_scores = torch.from_numpy(tree)
        mst = compute_min_spanning_tree(pairwise_scores)

    elif kinematic_mode.startswith('parents:'):
        # hillview: a PINNED tree, root included -- "0<-1 1<-5 ..." as the log's ">> kinematic tree
        # parents" line prints it (the root is the camera that is nobody's child). The default tree
        # comes out different per device and the seam follows it; pinning one
        # device's tree on the other answers whether the tree explains the whole difference.
        spec = kinematic_mode[len('parents:'):]
        par = {}
        for tok in spec.replace(',', ' ').split():
            c_, p_ = tok.split('<-')
            par[int(c_)] = int(p_)
        roots = sorted(set(range(len(imgs))) - set(par))
        if len(roots) != 1 or len(par) != len(imgs) - 1 or any(not 0 <= v < len(imgs) for v in par.values()):
            raise ValueError(f'kinematic parents must name one parent for every camera but one root '
                             f'({len(imgs)} cameras; got {len(par)} links, roots {roots})')
        children = {}
        for c_, p_ in par.items():
            children.setdefault(p_, []).append(c_)
        edges, todo, seen = [], [roots[0]], {roots[0]}
        while todo:
            n_ = todo.pop(0)
            for c_ in sorted(children.get(n_, [])):
                edges.append((n_, c_)); seen.add(c_); todo.append(c_)
        if len(seen) != len(imgs):
            raise ValueError(f'kinematic parents do not reach every camera from root {roots[0]} (a cycle?)')
        mst = (roots[0], edges)

    else:
        raise ValueError(f'bad {kinematic_mode=}')

    # remove all edges not in the spanning tree?
    # min_spanning_tree = {(imgs[i],imgs[j]) for i,j in mst[1]}
    # tmp_pairs = {(a,b):v for (a,b),v in tmp_pairs.items() if {(a,b),(b,a)} & min_spanning_tree}

    # RIG (hillview patch): views that share a camera centre and differ by a KNOWN rotation
    # (slices of one 360° frame). The kinematic tree is rebuilt so every rig is entered once
    # and its other views hang off the entry view; the optimizer then holds those edges fixed.
    rig = kw.pop('rig', None)
    if rig is not None and any(r is not None for r in rig):
        if kinematic_mode.startswith('parents:'):
            # a PINNED tree is kept as given: one replayed from a rigged run's log is already rig-shaped
            # (it was built here), and rebuilding it from match counts would silently undo the pin.
            # One that is not rig-shaped is refused, here before the optimiser's handoff
            refuse_rig_breaking_tree(mst, rig)
            print(' >> rig: the pinned kinematic tree is kept (not rebuilt around the rigs)')
        else:
            # with the chain, the only scores between two frames' views are last tile -> next frame's
            # first tile, so the tree over rigs follows the capture order too (tiles are frame-major)
            mst = named_tree(rig_spanning_tree, pairwise_scores, rig, shallow=(kinematic_mode == 'shallow'))

    if optimizer_phase is None:
        imgs, res_coarse, res_fine = sparse_scene_optimizer(
            imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21, canonical_paths, mst,
            shared_intrinsics=shared_intrinsics, cache_path=cache_path, device=device, dtype=dtype, rig=rig, **dense_kw, **kw)
    else:
        optimizer_phase.prepare(
            (imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21, canonical_paths, mst),
            dict(shared_intrinsics=shared_intrinsics, cache_path=cache_path, device=device, dtype=dtype,
                 rig=rig, **dense_kw, **kw))
        # prepare() has written the checkpoint. The child must start AFTER the
        # parent drops these tensors, not alongside a second live copy of them.
        del imsizes, pps, base_focals, core_depth, anchors, corres, corres2d, preds_21, dense_kw, kw
        if 'dense_anchors' in locals():
            del dense_anchors, dense_maps, canon, canon2, conf, focal, pix, idxs, offsets, allpix, aidx, aoff
        imgs, res_coarse, res_fine, anchors = optimizer_phase.run()

    return SparseGA(imgs, pairs_in, res_fine or res_coarse, anchors, canonical_paths, point_partners)



def weighted_sim3(src, dst, w):
    """Weighted similarity src -> dst in closed form. The gauge, computed without gradient:
    the GPS decides the shape, never which way the model happens to face."""
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    S, D = src - mu_s, dst - mu_d
    U, d, Vt = torch.linalg.svd((w[:, None] * D).T @ S)
    sgn = torch.ones(3, dtype=src.dtype, device=src.device)
    if torch.det(U) * torch.det(Vt) < 0:
        sgn[2] = -1
    R = U @ torch.diag(sgn) @ Vt
    s = (d * sgn).sum() / (w * (S ** 2).sum(1)).sum().clamp(min=1e-12)
    return s, R, mu_d - s * (R @ mu_s)


def rotation_to_down(g):
    """The smallest rotation taking the unit vector `g` onto (0, 0, -1): tilt only, no yaw."""
    g = g / g.norm().clamp(min=1e-12)
    down = torch.tensor([0.0, 0.0, -1.0], dtype=g.dtype, device=g.device)
    v = torch.linalg.cross(g, down); c = (g * down).sum()
    I = torch.eye(3, dtype=g.dtype, device=g.device)
    if float(v.norm()) < 1e-12:
        return I if float(c) > 0 else torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=g.dtype, device=g.device))
    K = torch.zeros(3, 3, dtype=g.dtype, device=g.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -v[2], v[1], v[2], -v[0], -v[1], v[0]
    return I + K + K @ K / (1.0 + c)


def weighted_sim3_vertical(src, dst, w, g_src):
    """weighted_sim3 with the VERTICAL KNOWN (hillview).

    `g_src` is gravity's direction in the model's frame, measured by the phones' accelerometers
    (the mean over cameras of R_i @ down_i). The tilt is taken from it -- 2 of the 3 rotational
    degrees of freedom -- and only yaw, scale and translation are fitted to the positions: yaw and
    scale from the HORIZONTAL coordinates, where GPS is strong, the vertical offset from the
    heights. The unconstrained fit let a flat walk's vertical GPS noise choose the tilt: 18.4 and
    21.4 deg off gravity on one phone walk, and a tilt the gauge picks for free is where a climb
    gets faked or absorbed."""
    Rt = rotation_to_down(g_src.to(src.dtype))
    p = src @ Rt.T
    mu_p = (w[:, None] * p).sum(0); mu_d = (w[:, None] * dst).sum(0)
    S, D = p[:, :2] - mu_p[:2], dst[:, :2] - mu_d[:2]
    a = (w * (S[:, 0] * D[:, 0] + S[:, 1] * D[:, 1])).sum()
    b = (w * (S[:, 0] * D[:, 1] - S[:, 1] * D[:, 0])).sum()
    th = torch.atan2(b, a); ct, st = torch.cos(th), torch.sin(th)
    Rz = torch.eye(3, dtype=src.dtype, device=src.device)
    Rz[0, 0], Rz[0, 1], Rz[1, 0], Rz[1, 1] = ct, -st, st, ct
    s = torch.sqrt(a * a + b * b) / (w * (S ** 2).sum(1)).sum().clamp(min=1e-12)
    R = Rz @ Rt
    return s, R, mu_d - s * (Rz @ mu_p)


def height_step_down(c, enu, w, li, lj, g0, knee_m, mount_sd_deg, floor_m, g_start=None, iters=20, tol_rad=1e-9):
    """The model's DOWN from the camera centres' HEIGHT STEPS (hillview): what should be constrained is the altitude
    delta.

    A step li -> lj rises -s g.(c_j - c_i) metres in a gauge whose down is g (weighted_sim3_vertical: the tilt takes g
    onto ENU's down, so ENU's up is -g in the model). Its terrain rise is enu[lj, 2] - enu[li, 2], both heights from
    one DEM. A STEP cancels a constant offset (embankment, cutting, the DEM's own bias) and puts a bridge or an
    underpass on its two end steps, which the Huber at knee_m lets go; the mounts' mean down follows the road grade,
    and absolute heights carry every offset into the tilt.

    THE FIT IS THE MOUNTS' DOWN g0 CORRECTED BY THE STEPS, each weighed by what it knows. The mounts' down is off the
    vertical by about mount_sd_deg (one sigma); the steps scatter by sigma_z metres, measured on their own residuals
    (1.4826 x the median, every iteration) but never below floor_m. So the correction (a, b) in g0's tangent plane
    minimises Huber(step residuals) + (sigma_z / mount_sd)^2 |(a, b)|^2 / 2: Gauss-Newton with IRLS, s re-read from
    the horizontal fit at each g. A tilt the steps see well moves to them; one they barely see -- a roll about a
    straight road -- stays the mounts'. An earlier cut held each direction by a FIXED share (1 %) of the best-seen
    one's information, which kept 0.15 deg of a 3 deg correction on a turning ride whose steps fit to 3 mm.

    THE FLOOR is what keeps that promise. The residuals are measured after the fit has spent its two tilt directions
    on them, so they cannot show an error the tilt absorbs: on a few steps of a gently turning road, a vertical bend
    of the solved trail against the DEM reads as a roll (the bend grows along the piece as the road's turn does), the
    residuals then fall to centimetres, and a floor that low leaves the mounts' prior weighing nothing in the direction
    the steps see worst. floor_m is the accuracy a terrain step is credited with at best (the DEM between two fixes,
    the fixes' own horizontal error across a sloping road, the mount's height); above every residual scatter a long
    piece shows, it leaves the along-the-road correction to the steps and the roll to the mounts.

    c (n, 3) model-frame centres; enu (n, 3) their fixes in ENU metres (heights read on linked rows only); w (n,) the
    horizontal fit's weights; li, lj the linked rows; floor_m > 0. g_start: where to start (the group's last answer: the gauge is
    refitted every iteration and moves little, so the fit stops in a step or two instead of running all `iters`);
    the prior stays centred on g0 either way. Stops when a step moves the down by less than
    tol_rad. -> (g, info). Called under no_grad: a gauge."""
    floor_m = float(floor_m)
    if not floor_m > 0:
        raise ValueError(f'height_step_down: floor_m must be > 0, not {floor_m!r}')
    dt = torch.float64
    c, w = c.to(dt), w.to(dt)
    flat = torch.cat([enu[:, :2].to(dt), torch.zeros_like(enu[:, :1], dtype=dt)], 1)   # s reads the horizontal only
    g0 = g0.to(dt) / g0.to(dt).norm().clamp(min=1e-12)
    e = torch.eye(3, dtype=dt, device=c.device)[int(torch.argmin(g0.abs()))]
    u = e - (e @ g0) * g0
    u = u / u.norm()
    v = torch.linalg.cross(g0, u)
    d = c[lj] - c[li]
    dz = (enu[lj, 2] - enu[li, 2]).to(dt)
    th = torch.zeros(2, dtype=dt, device=c.device)
    if g_start is not None:
        gs = g_start.to(dt)
        if float(gs @ g0) > 0.5:                            # the last answer, in g0's tangent coordinates
            q0 = gs / (gs @ g0)
            th = torch.stack([q0 @ u, q0 @ v])
    I2 = torch.eye(2, dtype=dt, device=c.device)
    prior = math.radians(float(mount_sd_deg)) ** -2
    seen = None
    n_it = 0
    sig = float('nan')
    for n_it in range(1, iters + 1):
        q = g0 + th[0] * u + th[1] * v
        nq = q.norm()
        g = q / nq
        s, _R, _t = weighted_sim3_vertical(c, flat, w, g)
        r = -s * (d @ g) - dz
        J = -s * torch.stack([d @ ((u - (g @ u) * g) / nq), d @ ((v - (g @ v) * g) / nq)], 1)
        a = r.abs()
        hw = torch.where(a <= knee_m, torch.ones_like(a), knee_m / a.clamp(min=1e-12))
        A = J.T @ (hw[:, None] * J)
        if seen is None:
            ev = torch.linalg.eigvalsh(A)
            if not float(ev[-1]) > 1e-18:
                return g0, dict(off_deg=0.0, seen=0.0, iters=0, sigma_z_m=sig, resid_med_m=float('nan'),
                                beyond_knee=0, why='the steps see no tilt (no lever arm)')
            seen = float(ev[0] / ev[-1])
        sig = max(1.4826 * float(a.median()), floor_m)
        lam = sig * sig * prior
        step = torch.linalg.solve(A + lam * I2, -(J.T @ (hw * r)) - lam * th)
        th = th + step
        if float(step.norm()) < tol_rad:
            break
    q = g0 + th[0] * u + th[1] * v
    g = q / q.norm()
    s, _R, _t = weighted_sim3_vertical(c, flat, w, g)
    r = (-s * (d @ g) - dz).abs()
    return g, dict(off_deg=math.degrees(math.acos(max(-1.0, min(1.0, float(g @ g0))))), seen=seen, iters=n_it,
                   sigma_z_m=sig, resid_med_m=float(r.median()), beyond_knee=int((r > knee_m).sum()), why=None)


def alt_step_residual_m(c, li, lj, dz, s, R):
    """Each linked step's solved height change against the terrain's dz, in metres, in the GPS gauge (s, R): ENU's
    up is R's last row (hillview --alt_step_frac). (s, R) come detached: the
    gauge is refitted every iteration, and the walk cannot shrink its way out (dz does not shrink with it)."""
    return s.to(c.dtype) * ((c[lj] - c[li]) @ R[2].to(c.dtype)) - dz.to(c.dtype)


def gyro_step_chord(cam2w, ti, tj, G):
    """How far each solved step R_i^T R_j sits from the gyro's G_ij, as the chord 2 sin(theta/2)
    (hillview).

    G_ij is the phone's own rotation between the two exposures, integrated from its gyro and taken
    into the OpenCV camera frame (C q C, C = diag(1,-1,-1), portrait). Relative, so it has no axis a
    solve could tilt to cut it (the escape a free tilt gives). The chord, not the angle, for loss_gravity's reason:
    smooth at zero with no clamp under arccos; and the norm carries an epsilon so the gradient at an
    exact match is 0, not NaN."""
    rel = cam2w[ti, :3, :3].transpose(1, 2) @ cam2w[tj, :3, :3]
    M = G.to(rel.dtype).transpose(1, 2) @ rel
    I = torch.eye(3, dtype=rel.dtype, device=rel.device)
    return torch.sqrt(((M - I) ** 2).flatten(1).sum(1) + 1e-12) / (2.0 ** 0.5)


def metres_per_unit(c, enu):
    """How many metres one solve unit is, read as the RATIO of the two walks' own sizes -- the mean
    distance of the GPS fixes from their centre over that of the camera centres from theirs
    (hillview). Differentiable in `c`, so a step measured as |c_j - c_i| * metres_per_unit(c, enu) is
    SCALE-INVARIANT: shrinking every camera toward one point shrinks the steps and grows this factor
    by the same amount. The first pace term read the GPS gauge's scale DETACHED, and on a phone
    walk the optimiser took exactly that exit -- the whole walk collapsed to a point
    (246918 units per metre, every camera in one spot, depths of 736 km) because shrinking always
    paid and the frozen scale never charged for it. It is the free-tilt escape in another term."""
    ec = (c - c.mean(0)).norm(dim=1).mean()
    eg = (enu - enu.mean(0)).norm(dim=1).mean()
    return eg / ec.clamp(min=1e-12)


def doppler_steps_m(d, L):
    """Solved step lengths `d` (solve units) in metres, with the scale taken from the Doppler's own
    expected lengths `L`: sum(L) / sum(d). A ratio, so a walk shrunk or grown uniformly reads the same
    metres; only how the length is DISTRIBUTED along the walk can differ from L (hillview)."""
    return d * (L.sum() / d.sum().clamp(min=1e-12))


def huber_m(x, knee):
    """Huber in metres, the GPS term's form: quadratic to `knee`, linear beyond it."""
    return torch.where(x <= knee, 0.5 * x ** 2, knee * (x - 0.5 * knee))


def gps_rel_residual_m(c, li, lj, enu_fit, zok, s, R, z_gain=1.0):
    """How far each consecutive step li -> lj (rows of one GPS group) is from the GPS's own step between
    the same two fixes, in metres (hillview --gps_rel_frac: the weak-frame holds' relative steps).

    Both steps are read in the GPS term's gauge: the solved step taken to ENU by gps_gauge's (s, R),
    the GPS step between its targets enu_fit -- so a stretch of fixes that drifted together moves both
    ends of every step inside it and costs nothing there, which is what separates this from the absolute
    term (only the step into the stretch sees the drift). The
    height of a step with an end whose altitude nobody recorded is not held (loss_gps's rule), and
    z_gain weighs a metre of altitude as loss_gps does. (s, R) come detached: the gauge is refitted
    every iteration and the walk cannot shrink its way out (the GPS step does not shrink with it)."""
    d = s.to(c.dtype) * ((c[lj] - c[li]) @ R.to(c.dtype).T) - (enu_fit[lj] - enu_fit[li]).to(c.dtype)
    both = zok[li] & zok[lj]
    if not bool(both.all()):
        d = torch.cat([d[:, :2], torch.where(both, d[:, 2], torch.zeros_like(d[:, 2]))[:, None]], 1)
    if z_gain != 1.0:
        d = torch.cat([d[:, :2], d[:, 2:3] * z_gain], 1)
    return d.norm(dim=1)


def weak_mid_residual_m(c, vi, pi, ni, s):
    """How far each WEAK camera vi is from the midpoint of its neighbours pi and ni, in metres through
    the GPS gauge's scale s (hillview --weak_mid_frac: the weak-frame holds' midpoint pull).
    The neighbours are DETACHED: the pull acts on the weak camera alone, never on them. It makes sense
    only in reaction to a photograph falling out of pairing: every photograph in the solve cannot just be
    pulled towards the middle of its neighbours. s comes detached too; only the weak
    camera feels this term, so it cannot shrink the walk. The FORCE is on the weak camera alone; the
    kinematic tree still carries its share of each step through that camera's ancestors and subtree,
    as it does every camera prior's, and no force from it acts on the neighbours where it settles."""
    mid = 0.5 * (c[pi] + c[ni]).detach()
    return s.to(c.dtype) * (c[vi] - mid).norm(dim=1)


def weak_hold_plan(weak, links, row_of, scope):
    """What the weak holds hold (hillview: the weak-frame holds), from the caller's weak
    views, the consecutive-photograph links (i, j) and row_of[view] = (GPS group, row in it).

    -> dict(rel={group: [(row_i, row_j)]}, rel_out, mid={group: [(row_v, row_prev, row_next)]}, ends, off,
    mid_out). rel: every link (scope 'all') or the links with a weak end ('weak'), inside one GPS group;
    rel_out counts the rest. mid: each weak view with a link on both sides, inside one group with both
    neighbours; ends names weak views at the end of their chain, off counts weak views on no link (a rig's
    other views, carried by its stand-in, or a view that stands for no photograph), mid_out the rest.
    A view with two links on one side is refused: a chain links each photograph once each way."""
    if scope not in ('weak', 'all'):
        raise ValueError(f"gps_rel_scope {scope!r}: 'weak' or 'all'")
    weak = {int(v) for v in weak}
    links = sorted({(int(i), int(j)) for i, j in links})
    prev_of, next_of = {}, {}
    for i, j in links:
        if j in prev_of or i in next_of:
            raise ValueError(f'step_links: view {j if j in prev_of else i} has two links on one side; '
                             f'a chain links each photograph to one before and one after')
        prev_of[j], next_of[i] = i, j

    def one_group(*vs):
        g_ = [row_of.get(v) for v in vs]
        return None if any(x is None for x in g_) or len({x[0] for x in g_}) != 1 else g_[0][0]
    rel, rel_out = {}, 0
    for i, j in links:
        if scope == 'weak' and i not in weak and j not in weak:
            continue
        gi = one_group(i, j)
        if gi is None:
            rel_out += 1
        else:
            rel.setdefault(gi, []).append((row_of[i][1], row_of[j][1]))
    mid, ends, off, mid_out = {}, [], 0, 0
    for v in sorted(weak):
        if v not in prev_of and v not in next_of:
            off += 1
        elif v not in prev_of or v not in next_of:
            ends.append(v)
        else:
            gi = one_group(prev_of[v], v, next_of[v])
            if gi is None:
                mid_out += 1
            else:
                mid.setdefault(gi, []).append((row_of[v][1], row_of[prev_of[v]][1], row_of[next_of[v]][1]))
    return dict(rel=rel, rel_out=rel_out, mid=mid, ends=ends, off=off, mid_out=mid_out, n_links=len(links))


def gps_gauge(c, enu, w, zok, g=None):
    """The similarity that takes the model's camera centres `c` onto the GPS track `enu`, and the
    target the residual is then measured against.

    WHAT THE GPS DOES NOT KNOW MUST NOT PULL (hillview). Most sources record no
    altitude at all -- every Panoramax frame carries `altitude: null` -- and filling that with the
    mean height asserts a flat plane the receiver never measured. `zok` marks the rows whose
    altitude IS known. Where it is not, a throwaway first fit says where the model puts that
    camera and the returned target agrees with it there, so an unknown height neither tilts the
    gauge nor appears in the residual. With every altitude known this is one fit, unchanged.

    Returns (s, R, t, enu_fit). Called under no_grad: the gauge is not differentiated.
    """
    # g: gravity's direction in the model frame. With it the tilt is measured, not fitted.
    fit = (weighted_sim3 if g is None else
           (lambda a, b, ww: weighted_sim3_vertical(a, b, ww, g.to(a.dtype))))
    if bool(zok.all()):
        s, R, t = fit(c.double(), enu.double(), w.double())
        return s, R, t, enu
    # The unknown rows have to be filled with SOMETHING before a similarity can be fitted at all,
    # and whatever they are filled with pulls the fit -- which is the whole bug. So the fill comes
    # from the model itself, and the STARTING POINT comes from the rows that are not in doubt:
    # three or more frames that do record an altitude carry full three-dimensional information, and
    # a fit on those alone is untouched by any fiction. The heights it gives the remaining frames
    # are then their fill, and a few passes over everything tighten the horizontal agreement
    # without ever reintroducing a height nobody measured. On a synthetic track this recovers the
    # exact similarity (recon_gps_check.py); filling with the mean instead left half a metre of
    # residual on the frames whose altitude WAS known, and converged no further.
    # With no altitude anywhere -- the Panoramax case -- there is nothing to seed from, and one
    # pass is already exact: every height is free, so the target's is simply the model's.
    known = torch.where(zok)[0]
    if len(known) >= 3:
        wk = w[known] / w[known].sum().clamp(min=1e-12)
        s0, R0, t0 = fit(c[known].double(), enu[known].double(), wk.double())
    else:
        z0 = enu[zok, 2].mean() if bool(zok.any()) else torch.zeros((), dtype=enu.dtype, device=enu.device)
        flat = torch.cat([enu[:, :2], torch.where(zok, enu[:, 2], z0)[:, None]], 1)
        s0, R0, t0 = fit(c.double(), flat.double(), w.double())
    rec = s0 * (c.double() @ R0.T) + t0
    enu_fit = torch.cat([enu[:, :2], torch.where(zok, enu[:, 2], rec[:, 2].to(enu.dtype))[:, None]], 1)
    s, R, t = s0, R0, t0
    for _ in range(4):
        s, R, t = fit(c.double(), enu_fit.double(), w.double())
        rec = s * (c.double() @ R.T) + t
        z_next = torch.where(zok, enu[:, 2], rec[:, 2].to(enu.dtype))
        moved = float((z_next - enu_fit[:, 2]).abs().max())
        enu_fit = torch.cat([enu[:, :2], z_next[:, None]], 1)
        if moved < 1e-9:
            break
    s, R, t = fit(c.double(), enu_fit.double(), w.double())
    return s, R, t, enu_fit


def fixed_gauge(c, enu, zok, gauge):
    """gps_gauge's answer when the gauge is GIVEN (hillview: HELD_TARGETS' frame is a judged source's,
    so its alignment (s, R, t), solver frame -> ENU, stands for every refit). An unknown height's target is the model's,
    as gps_gauge makes it, so it neither pulls nor appears in a residual. -> (s, R, t, enu_fit), float64."""
    s, R, t = (torch.as_tensor(x, dtype=torch.float64, device=c.device) for x in gauge)
    if R.shape != (3, 3) or t.shape != (3,) or s.numel() != 1 or not bool(s > 0):
        raise ValueError(f'FIXED_GAUGE: (s > 0, R 3x3, t 3) wanted, got {tuple(s.shape)}, {tuple(R.shape)}, '
                         f'{tuple(t.shape)}')
    s = s.reshape(())
    if bool(zok.all()):
        return s, R, t, enu
    rec = s * (c.double() @ R.T) + t
    return s, R, t, torch.cat([enu[:, :2], torch.where(zok, enu[:, 2], rec[:, 2].to(enu.dtype))[:, None]], 1)


def held_ray_closest(w2cam, K, vi, pix, X):
    """For each held target (hillview): the point on view vi's ray through pixel `pix` nearest the fixed point X
    (behind the camera: the centre). pix_loss(closest, X) is then the loss of X's distance from that ray, a 3-D length
    in solve units as loss_3d's, and it moves only the view: X is a constant. w2cam (V, 4, 4), K (V, 3, 3), vi (N,)
    long, pix (N, 2), X (N, 3)."""
    R = w2cam[vi, :3, :3].transpose(1, 2)                                   # camera -> world rotation
    C = -(R @ w2cam[vi, :3, 3:4]).squeeze(-1)                               # camera centre in the world
    Kv = K[vi]
    d = torch.stack([(pix[:, 0] - Kv[:, 0, 2]) / Kv[:, 0, 0], (pix[:, 1] - Kv[:, 1, 2]) / Kv[:, 1, 1],
                     torch.ones_like(pix[:, 0])], -1)
    r = (R @ d.unsqueeze(-1)).squeeze(-1)
    r = r / r.norm(dim=1, keepdim=True).clamp(min=1e-12)
    along = ((X - C) * r).sum(1, keepdim=True).clamp(min=0)
    return C + along * r


def held_reproj(K, w2cam, vi, X):
    """For each held target (hillview): the fixed point X projected into view vi, in pixels, as loss_2d's
    reprojection does it (depth clipped at 1e-3, pixels clipped to [-1000, 2000]). -> (N, 2)"""
    P = K[vi] @ w2cam[vi, :3]
    res = (P[:, :3, :3] @ X.unsqueeze(-1)).squeeze(-1) + P[:, :3, 3]
    return (res[:, 0:2] / res[:, 2:3].clip(min=1e-3)).clip(min=-1000, max=2000)


def held_start(given, n_views):
    """A held solve's start (hillview): (n_views, 4, 4) camera-to-world GIVEN in the targets' frame -- the source's
    solve frame, in its units -- every view, each rotation proper. Taken as it is: no tree and no GPS fit, which would
    put it in this run's own pictures' units (pose_init.initial_poses divides by their metres per unit), while the
    targets already fix the frame and the scale. -> float64 array."""
    g = np.asarray(given, float)
    if g.shape != (n_views, 4, 4) or not np.isfinite(g).all():
        raise ValueError(f'held start: ({n_views}, 4, 4) finite camera-to-world wanted, got {g.shape}, '
                         f'finite {bool(np.isfinite(g).all())}')
    R = g[:, :3, :3]
    bad = [v for v in range(n_views) if np.abs(R[v].T @ R[v] - np.eye(3)).max() > 1e-4 or np.linalg.det(R[v]) <= 0]
    if bad:
        raise ValueError(f'held start: view(s) {bad[:10]} carry no rotation (not orthonormal, or a reflection)')
    if np.abs(g[:, 3] - np.array([0.0, 0.0, 0.0, 1.0])).max() > 1e-9:
        raise ValueError('held start: a last row that is not (0, 0, 0, 1)')
    return g


def held_depth_scale(w2c, held, depth, shapes, subsample):
    """Each view's depth scale at a held solve's start (hillview), as log sizes. The pictures' own depths are in
    their canonical units and the targets' in the source's; a view with targets takes the median, over them, of the
    target's depth in its start camera over its own depth at the target pixel's anchor cell (the grid
    canonical[sub//2::sub, sub//2::sub]; global_scaling is 1 in a held solve, so a view's depth is linear in its size).
    A view without targets takes the median of the views that have them (one phone, one model: the best single guess,
    which its pair terms then correct). w2c (V, 4, 4) at the start; held (vi, pix, X, conf) as the solver batches them;
    depth: per view its flat depth at size 1; shapes per view (H, W). -> (log_sizes (V,), {view: targets used})."""
    hv, hp, hx = (np.asarray(x.detach().cpu() if torch.is_tensor(x) else x, float) for x in held[:3])
    hv = hv.astype(int)
    n = len(depth)
    w2c = np.asarray(w2c.detach().cpu() if torch.is_tensor(w2c) else w2c, float)
    logs, used = np.full(n, np.nan), {}
    for v in sorted(set(hv.tolist())):
        H, W = (int(x) for x in shapes[v])
        rows, cols = len(range(subsample // 2, H, subsample)), len(range(subsample // 2, W, subsample))
        d = np.asarray(depth[v].detach().cpu() if torch.is_tensor(depth[v]) else depth[v], float).ravel()
        if d.size != rows * cols:
            raise ValueError(f'held depth scale: view {v}\'s depth has {d.size} cells, its anchor grid '
                             f'{rows} x {cols} at subsample {subsample}')
        m = hv == v
        pix, X = hp[m], hx[m]
        z_t = (X @ w2c[v, :3, :3].T + w2c[v, :3, 3])[:, 2]
        r_ = np.clip((pix[:, 1] // subsample).astype(int), 0, rows - 1)
        c_ = np.clip((pix[:, 0] // subsample).astype(int), 0, cols - 1)
        z_v = d[r_ * cols + c_]
        ok = (z_t > 0) & (z_v > 0) & np.isfinite(z_t) & np.isfinite(z_v)
        if ok.sum():
            logs[v] = float(np.median(np.log(z_t[ok] / z_v[ok])))
            used[v] = int(ok.sum())
    have = np.isfinite(logs)
    if not have.any():
        raise ValueError('held depth scale: no target lies in front of its start camera on a cell with depth')
    logs[~have] = float(np.median(logs[have]))
    return logs, used


def tensor_storage_bytes(value):
    """Unique live tensor storage; shared views count once, including backing storage."""
    storages = {}
    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
    visit(value)
    return sum(storages.values())


def pack_pointmap_targets(imgs, pairs, predictions):
    """Pack targets once and retain equivalent per-pair views for the slow path.

    Rebinding entries releases the original storage even if the caller retains
    this mapping. Values, shapes, order, and dictionary keys remain unchanged.
    """
    targets = torch.cat([predictions[imgs[j]][imgs[i]][0] for i, j in pairs])
    confidence = torch.cat([predictions[imgs[j]][imgs[i]][1] for i, j in pairs])
    offset = 0
    for i, j in pairs:
        count = len(predictions[imgs[j]][imgs[i]][0])
        predictions[imgs[j]][imgs[i]] = (targets[offset:offset + count], confidence[offset:offset + count])
        offset += count
    return targets, confidence


def sparse_scene_optimizer(imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, corres, corres2d,
                           preds_21, canonical_paths, mst, cache_path,
                           lr1=0.07, niter1=300, loss1=gamma_loss(1.5),
                           lr2=0.01, niter2=300, loss2=gamma_loss(0.5),
                           lossd=gamma_loss(1.1),
                           opt_pp=True, opt_depth=True,
                           pinned_pp=None,  # hillview: per view, True where the principal point is a FACT (held)
                           known_focals=None, hold_focal=False,  # hillview: a slice's focal is KNOWN from the cut
                           schedule=cosine_schedule, depth_mode='add', exp_depth=False,
                           lora_depth=False,  # dict(k=96, gamma=15, min_norm=.5),
                           shared_intrinsics=False,
                           intrinsics_groups=None,  # hillview: shared_intrinsics='groups' -- per view (focal group, pp group or None)
                           init={}, device='cuda', dtype=torch.float32,
                           matching_conf_thr=5., loss_dust3r_w=0.01,
                           loss2_3d_w=0.0,  # hillview: keep the 3D tie through stage 2 (rigs: reprojection is blind to depth between views sharing a centre)
                           dense_anchors=None, dense_maps=None, dense_consistency_w=0.0, dense_gate=0.25,  # hillview: dense cross-view depth consistency
                           dense_consistency_mode='symmetric',  # hillview: 'near' = the view nearer a surface holds it, the farther one moves
                           dense_masks_given=False, dense_view_masks=None, dense_pair_var=None,  # hillview: the pairs' masks
                           dense_src_keep=None,  # (dense_pair_masks)
                           gps=None, gps_w=None, gps_weight=0.0, gps_groups=None, gps_huber_m=5.0,
                           gps_local_win=9,  # hillview: GPS in the objective, not only in the verdict
                           gps_frac=0.0, gps_gate_max=2.0,  # hillview: ...as a FRACTION of the pixel term, with the locality gate bounded
                           gps_z_gain=1.0,  # ...and how much a metre of ALTITUDE counts against a horizontal metre
                           gps_pull_min=0.2, gps_pull_max=20.0,  # ...and how far that local scale may swing
                           gravity_weight=0.0, gravity_slack_deg=3.0,  # hillview: the mount is upright
                           compass_pairs=None, compass_weight=0.0, compass_slack_deg=60.0,  # hillview: the capture's own bearing, as a RELATIVE yaw between frames
                           compass_max_pitch_deg=50.0, compass_ramp_deg=20.0,  # ...and none at all past a tilt the compass cannot survive
                           level_idx=None, level_weight=0.0, level_slack_deg=25.0, level_pitch_deg=-20.0,  # hillview: a hand-held walk is held roughly level
                           height_weight=0.0, height_slack_steps=0.5,  # hillview: ...and at one height, the walk being flat
                           horizon_weight=0.0, horizon_max_pitch_deg=0.0, horizon_slack_deg=5.0,  # ...and never above the horizon
                           upright_weight=0.0, upright_slack_deg=40.0,  # hillview: ...and the right way up, not antipodean
                           grade_weight=0.0, grade_slack=0.25, grade_min_step_frac=0.1,  # hillview: a drive cannot climb a wall between two consecutive frames
                           rig_upright_weight=0.0, rig_upright_slack_deg=8.0,  # hillview: the mount is upright against the WORLD, a stated fact of the capture
                           imu_down=None, gauge_gravity=False,  # hillview: the phones' measured down per view (camera frame); with gauge_gravity the GPS gauge takes its TILT from it
                           gyro_steps=None, gyro_weight=0.0, gyro_slack_deg=3.0,  # hillview: the phones' gyro between exposures, (i, j, 3x3 in the camera frame[, slack_deg])
                           gravity_cam_weight=0.0, gravity_cam_slack_deg=2.0,  # hillview: every camera's measured down agrees (reads imu_down)
                           rig_yaw=None,  # hillview: how each pano slice was cut, radians about the rig's up -- what lets roll be told from pitch
                           axis_roll_weight=0.0, axis_roll_slack_deg=2.0,  # hillview: the mount's ROLL against an INJECTED down, tight
                           axis_pitch_weight=0.0, axis_pitch_slack_deg=8.0,  # hillview: ...and its PITCH, loose -- a bike nods, it does not lean
                           pace_steps=None, pace_weight=0.0, pace_max_mps=2.5,  # hillview: consecutive exposures (i, j, seconds apart) cannot be farther apart than a walk covers
                           speed_steps=None, speed_weight=0.0,  # hillview: consecutive exposures (i, j, expected metres, tolerance metres) from the phone's Doppler speed
                           course_steps=None, course_weight=0.0,  # hillview: consecutive exposures (i, j, Doppler course east, north, course accuracy deg) -- the step's DIRECTION
                           weak_views=None, step_links=None,  # hillview: the views of WEAK photographs (decided by the caller, or a callable it answers once matching is done) and the consecutive-photograph links (i, j) the step priors hold
                           gps_rel_frac=0.0, gps_rel_huber_m=1.0, gps_rel_scope='weak',  # hillview: those steps held to the GPS's own steps, as a FRACTION of the pixel term (weak: steps with a weak end; all: every step)
                           weak_mid_frac=0.0, weak_mid_huber_m=1.0,  # hillview: a weak camera pulled to its neighbours' midpoint, the neighbours detached
                           alt_step_links=None, gauge_tilt_from='mount',  # hillview: consecutive photographs' stand-ins (i, j) whose heights are one DEM's; 'height_steps' = the GPS gauge's down from their height steps
                           gauge_tilt_mount_sd_deg=3.0,  # hillview: ...and how far the mounts' mean down may sit off the vertical (one sigma), the prior the steps correct
                           gauge_tilt_step_floor_m=None,  # hillview: ...and the accuracy a terrain step is credited with at best, metres (height_step_down's floor_m); required under 'height_steps'
                           alt_step_frac=0.0, alt_step_huber_m=0.5,  # hillview: ...and those steps' solved height changes held to the terrain's, as a FRACTION of the pixel term at one knee per step
                           pair_steps=None, pair_step_w=0.0, pair_step_huber=0.5, pair_step_depth_scale=True, pair_step_dir_only=False, pair_step_len_sep=0, pair_step_len_only=False,  # hillview: the pairs' own step beliefs in the objective
                           pose_init='pile',  # hillview: where the cameras START (pose_init.py): 'pile' (upstream), 'visual', 'gps'
                           pose_init_frac=1.0,  # hillview: how far the placed start is from the pile -- each centre moved this fraction of the way out from the cameras' mean (1 = the start as placed)
                           pose_init_poses=None,  # hillview: with pose_init 'poses', (n, 4, 4) camera-to-world in the GPS frame (ENU metres), from outside (a pose graph)
                           pose_fixed=False,  # hillview: with pose_init 'poses', the given poses are HELD: the cameras are constants and only the depths (and intrinsics) are solved
                           lr_warmup=0,  # hillview: ramp lr linearly over this many iterations of each stage (0 = upstream's cosine from the start)
                           pointmap_pairs="default", pointmap_pair_inputs=None,
                           pointmap_selection=None,  # hillview: condense_data's decisions, one per pair (POINTMAP_POLICY)
                           pointmap_policy=None,  # hillview: or the caller's policy, evaluated here (a direct caller)
                           pointmap_fallback=None, pair_geometry_path=None,  # hillview: pair-term evidence
                           trajectory=None,  # hillview: a dict this appends every iteration's camera poses to (--trajectory)
                           capture_solve=None,  # hillview: path; snapshot every input at entry for recon_solve_replay.py
                           inspect=None,  # hillview: {'out': forces.npz, 'bucket': iterations, 'chunk': correspondences, 'debug': bool} (inspect_forces.py)
                           restore=None,  # hillview: a finished run's FINAL state in place of the two stages (see below)
                           verbose=True, dbg=(), rig=None):
    _entry_locals = dict(locals())       # FIRST statement: exactly the parameters (a comprehension's locals() is not)
    if capture_solve:
        # WHAT THIS OPTIMISER RECEIVED, by name and after every wrapper, so a replay calls it with exactly this.
        # The first version captured in sparse_global_alignment, BEFORE reconstruct.py's tile_pp wrapper pinned the
        # principal points and turned opt_pp off -- a replay of it drifted 1.3 % from its own run. Anything that will
        # not pickle is NAMED in `skipped` for the replay to supply explicitly.
        import pickle
        import sys as _sys
        # the RUNNING function's own parameters: the module global sparse_scene_optimizer may be a wrapper by now
        _code = _sys._getframe().f_code
        _params = set(_code.co_varnames[:_code.co_argcount + _code.co_kwonlyargcount]) - {'capture_solve'}
        _given = {n: v for n, v in _entry_locals.items() if n in _params}
        if set(_given) != _params:
            raise RuntimeError(f'capture_solve: parameters missing from the snapshot: {sorted(_params - set(_given))}')
        keep, skipped, rebuild = {}, {}, {}
        for k_, v_ in _given.items():
            if callable(v_) and getattr(v_, '__qualname__', '').startswith('gamma_loss.<locals>'):
                # a loss closure does not pickle; record the values it closes over so the replay REBUILDS it
                cells = dict(zip(v_.__code__.co_freevars, (c_.cell_contents for c_ in v_.__closure__)))
                rebuild[k_] = dict(fn='gamma_loss', kwargs={n: cells[n] for n in ('gamma', 'mul', 'offset', 'clip')})
                continue
            if callable(v_) and getattr(v_, '__qualname__', '').startswith('meta_gamma_loss.<locals>'):
                rebuild[k_] = dict(fn='meta_gamma_loss', kwargs={})       # lambda alpha: gamma_loss(alpha)
                continue
            try:
                pickle.dumps(v_)
                keep[k_] = v_
            except Exception as e_:
                skipped[k_] = f'{type(v_).__name__}: {type(e_).__name__}'
        # a capture holds arbitrary pickles, not cache arrays: no content identity
        _save_atomic(dict(version=2, kw=keep, skipped=skipped, rebuild=rebuild), capture_solve, content=False)
        print(f' >> solve captured to {capture_solve} ({len(keep)} input(s) kept, rebuilt by the replay: '
              f'{sorted(rebuild) or "none"}, skipped: {sorted(skipped) or "none"})')
    init = copy.deepcopy(init)
    _SLOW_OFF = not os.environ.get('HV_SPARSE_GA_SLOW')
    # HV_SGA_SLOW_PARTS=rig,chain,depth,pts,l3d,l2d,d3r,quat runs those pieces the original way,
    # so a difference can be bisected to the piece that causes it
    _SLOW_PARTS = {x for x in os.environ.get('HV_SGA_SLOW_PARTS', '').split(',') if x}

    # The rig links are batched only on request: they save ~66 kernels of ~1,800 an iteration, and
    # the different order their gradients add up in was enough to send six panoramas of a ride along a church lane into
    # another local minimum (stage-1 loss 2.8656 -> 3.0387, cameras 289 mm apart). The assignment
    # itself is exact -- forward identical to the bit, gradients within 1e-6 in float32 -- so this
    # is the solve's sensitivity, not a bug; it is simply not worth it.
    if not os.environ.get('HV_SGA_FAST_RIG'):
        _SLOW_PARTS = _SLOW_PARTS | {'rig'}

    def _fastp(name):
        return _SLOW_OFF and name not in _SLOW_PARTS
    # RIG (hillview patch): rig[i] = (group, R_cam_i_to_rig) or None. For every tree edge
    # (i, j) inside one group the relative rotation is a constant, R_i^T R_j, and the
    # relative translation is whatever makes the two camera CENTRES coincide under the
    # focal-plane reparameterization below — so quats[j]/trans[j] are not free parameters.
    rig_edges = {}
    if rig is not None:
        # a rig is held only along same-rig tree edges: a tree that splits one is refused, not
        # repaired (rig_tree_breaks) -- whichever caller built or pinned it
        refuse_rig_breaking_tree(mst, rig)
        for i, j in mst[1]:
            ri, rj = rig[i], rig[j]
            if ri is not None and rj is not None and ri[0] == rj[0]:
                Ri = torch.as_tensor(np.asarray(ri[1]), dtype=dtype, device=device)
                Rj = torch.as_tensor(np.asarray(rj[1]), dtype=dtype, device=device)
                rig_edges[j] = (i, Ri.T @ Rj)
        if verbose and rig_edges:
            print(f' >> rig: {len(rig_edges)} view(s) tied to their rig by a fixed rotation; '
                  f'{len({r[0] for r in rig if r is not None})} rig(s)')
    # hillview: the rig links and the kinematic tree by depth, as index tensors, so building the
    # cameras is a few batched kernels instead of one small chain of them per camera
    _rig_js = list(rig_edges.keys())
    if _rig_js:
        _rig_j_t = torch.as_tensor(_rig_js, device=device)
        _rig_i_t = torch.as_tensor([rig_edges[j][0] for j in _rig_js], device=device)
        _rig_R_t = torch.stack([rig_edges[j][1] for j in _rig_js])
    _lvl = {mst[0]: 0}
    for i, j in mst[1]:
        _lvl[j] = _lvl[i] + 1
    _levels = []
    for L in range(1, max(_lvl.values()) + 1):
        e = [(i, j) for i, j in mst[1] if _lvl[j] == L]
        _levels.append((torch.as_tensor([i for i, _ in e], device=device),
                        torch.as_tensor([j for _, j in e], device=device)))
    # what the cameras cost per iteration: the tree's depth is how many batched products it takes
    # (a rig makes it shallow -- its slices hang off one node; a walk is a long chain)
    print(f' >> kinematic tree: {len(imgs)} cameras, depth {len(_levels)} '
          f'({len(_levels)} batched products per iteration), {len(_rig_js)} rig link(s)')
    # which tree, not just how deep: the parent of every camera, root first. The default tree
    # (hclust-ward over match counts) came out depth 6 on CUDA and depth 5 on CPU for the same
    # walk and recipe, whose two solves then ended in different minima -- a tree the
    # log does not name cannot be compared
    _par = {j: i for i, j in mst[1]}
    print(f' >> kinematic tree parents (camera<-parent, root {mst[0]}): '
          + ' '.join(f'{j}<-{_par[j]}' for j in sorted(_par)))
    _depth_flat = [None, None]   # (flat depth of all images, the per-image list it was split into)
    _CHUNK = int(os.environ.get('HV_SPARSE_GA_CHUNK', 4_000_000))   # points per batched gather

    # extrinsic parameters
    vec0001 = torch.tensor((0, 0, 0, 1), dtype=dtype, device=device)
    quats = [nn.Parameter(vec0001.clone()) for _ in range(len(imgs))]
    trans = [nn.Parameter(torch.zeros(3, device=device, dtype=dtype)) for _ in range(len(imgs))]

    # initialize
    ones = torch.ones((len(imgs), 1), device=device, dtype=dtype)
    median_depths = torch.ones(len(imgs), device=device, dtype=dtype)
    for img in imgs:
        idx = imgs.index(img)
        init_values = init.setdefault(img, {})
        if verbose and init_values:
            print(f' >> initializing img=...{img[-25:]} [{idx}] for {set(init_values)}')

        K = init_values.get('intrinsics')
        if K is not None:
            K = K.detach()
            focal = K[:2, :2].diag().mean()
            pp = K[:2, 2]
            base_focals[idx] = focal
            pps[idx] = pp
        pps[idx] /= imsizes[idx]  # default principal_point would be (0.5, 0.5)

        depth = init_values.get('depthmap')
        if depth is not None:
            core_depth[idx] = depth.detach()

        median_depths[idx] = med_depth = core_depth[idx].median()
        core_depth[idx] /= med_depth

        cam2w = init_values.get('cam2w')
        if cam2w is not None:
            rot = cam2w[:3, :3].detach()
            cam_center = cam2w[:3, 3].detach()
            quats[idx].data[:] = roma.rotmat_to_unitquat(rot)
            trans_offset = med_depth * torch.cat((imsizes[idx] / base_focals[idx] * (0.5 - pps[idx]), ones[:1, 0]))
            trans[idx].data[:] = cam_center + rot @ trans_offset
            del rot
            assert False, 'inverse kinematic chain not yet implemented'

    # intrinsics parameters
    if shared_intrinsics == 'focal':
        # hillview: ONE focal for every view, each view its OWN principal point. Tiles of one
        # photograph (and every photograph of one phone) share the focal by construction, but a
        # tile's principal point is wherever its crop put it (pinned by --tile_pin_pp). Solving 80
        # free focals for 20 photographs' tiles gave 427-599 px with the four tiles of one frame
        # 20 % apart, which free tiles hid by moving
        # and rigid ones could not.
        confs = torch.stack([_cache_load(_found(pth), 'cpu')[0][2].mean() for pth in canonical_paths]).to(pps)
        weighting = confs / confs.sum()
        pps = [nn.Parameter(pp.to(dtype)) for pp in pps]
        focal_m = weighting @ base_focals
        log_focal = nn.Parameter(focal_m.view(1).log().to(dtype))
        log_focals = [log_focal for _ in range(len(imgs))]
    elif shared_intrinsics and shared_intrinsics != 'groups':
        # Optimize a single set of intrinsics for all cameras. Use averages as init.
        confs = torch.stack([_cache_load(_found(pth), 'cpu')[0][2].mean() for pth in canonical_paths]).to(pps)
        weighting = confs / confs.sum()
        pp = nn.Parameter((weighting @ pps).to(dtype))
        pps = [pp for _ in range(len(imgs))]
        focal_m = weighting @ base_focals
        log_focal = nn.Parameter(focal_m.view(1).log().to(dtype))
        log_focals = [log_focal for _ in range(len(imgs))]
    elif shared_intrinsics == 'groups':
        # hillview: ONE FOCAL PER CAMERA, for runs that mix cameras. 'True' and 'focal'
        # put every view on one lens, which a run of two phones, or of phone photographs beside
        # panorama slices, does not have; per-view focals wander on views that cannot constrain them.
        # `intrinsics_groups[i]` is (focal group, pp group or None): views of one focal group share a
        # focal, views of one pp group a principal point, and None gives the view its own (a tile's
        # principal point is wherever its crop put it). The caller decides the groups (reconstruct
        # --intrinsics_groups camera: the manifest's camera and frame size); one group of each is
        # exactly True, one focal group with every pp None exactly 'focal'. Each group starts at the
        # confidence-weighted mean of its own views, as the single group does.
        if intrinsics_groups is None or len(intrinsics_groups) != len(imgs):
            raise ValueError(f"shared_intrinsics='groups' needs intrinsics_groups, one (focal group, pp "
                             f"group) per view: got {None if intrinsics_groups is None else len(intrinsics_groups)} "
                             f"for {len(imgs)} views")
        confs = torch.stack([_cache_load(_found(pth), 'cpu')[0][2].mean() for pth in canonical_paths]).to(pps)
        fgroups, pgroups = {}, {}
        for i, (fg, pg) in enumerate(intrinsics_groups):
            fgroups.setdefault(fg, []).append(i)
            if pg is not None:
                pgroups.setdefault(pg, []).append(i)
        for pg, idx in pgroups.items():
            if len({intrinsics_groups[i][0] for i in idx}) > 1:
                raise ValueError(f'intrinsics_groups: principal-point group {pg!r} spans focal groups '
                                 f'{sorted({str(intrinsics_groups[i][0]) for i in idx})} -- one optical '
                                 f'centre cannot belong to two lenses')
        log_focals, new_pps = [None] * len(imgs), [None] * len(imgs)
        for fg, idx in fgroups.items():
            w = confs[idx] / confs[idx].sum()
            log_focal = nn.Parameter((w @ base_focals[idx]).view(1).log().to(dtype))
            for i in idx:
                log_focals[i] = log_focal
        for pg, idx in pgroups.items():
            w = confs[idx] / confs[idx].sum()
            pp = nn.Parameter((w @ pps[idx]).to(dtype))
            for i in idx:
                new_pps[i] = pp
        pps = [new_pps[i] if new_pps[i] is not None else nn.Parameter(pps[i].to(dtype)) for i in range(len(imgs))]
        print(f' >> intrinsics: {len(fgroups)} focal group(s) ('
              + ', '.join(f'{fg}: {len(idx)} view(s)' for fg, idx in fgroups.items())
              + f'), {len(pgroups)} shared principal point(s), '
              f'{sum(1 for _, pg in intrinsics_groups if pg is None)} view(s) with their own')
    else:
        pps = [nn.Parameter(pp.to(dtype)) for pp in pps]
        log_focals = [nn.Parameter(f.view(1).log().to(dtype)) for f in base_focals]

    # THE KNOWN FOCAL (hillview, --pano_focal known). A panorama slice is a pinhole the
    # slicer rendered: its focal is (W/2)/tan(fov/2), a fact, not an estimate. Started there and,
    # with hold_focal, never moved -- so the optimiser cannot spend a free focal absorbing error
    # (the canonical estimate drifted 256 -> 226 px on a 3-panorama span, 256 -> 255.7 on 40).
    # base_focals stays the canonical estimate: it carries the scale of the canonical depth maps.
    if known_focals is not None:
        if len(known_focals) != len(imgs):
            raise ValueError(f'known_focals: {len(known_focals)} for {len(imgs)} views')
        if shared_intrinsics == 'groups':
            # per FOCAL GROUP: a group is all known (one value) or all unknown -- a known focal shared
            # with a view that has none would hold, or start, that view at another lens's value. A
            # group with no known focal is solved freely beside held ones (slices beside photographs).
            for fg in dict.fromkeys(g for g, _ in intrinsics_groups):
                kf = [known_focals[i] for i, (g, _) in enumerate(intrinsics_groups) if g == fg]
                if any(f is None for f in kf) and any(f is not None for f in kf):
                    raise ValueError(f'known_focals: focal group {fg!r} has views with a known focal and '
                                     f'views without one -- one focal cannot be both')
                if len({round(float(f), 6) for f in kf if f is not None}) > 1:
                    raise ValueError(f'known_focals differ within focal group {fg!r}: one focal cannot be two')
        elif hold_focal and any(f is None for f in known_focals):
            raise ValueError('hold_focal: every view needs a known focal -- a view with none would '
                             'be held at a guess (mixed slices and photographs are refused)')
        for i, f in enumerate(known_focals):
            if f is not None:
                if not (math.isfinite(float(f)) and float(f) > 0):     # inf passed the old `f > 0`
                    raise ValueError(f'known_focals[{i}] = {f!r} is not a focal')
                log_focals[i].data[:] = torch.tensor([float(f)], dtype=dtype, device=log_focals[i].device).log()
        if shared_intrinsics and shared_intrinsics != 'groups' \
                and len({round(float(f), 6) for f in known_focals if f is not None}) > 1:
            raise ValueError('known_focals differ but the intrinsics are SHARED: one focal cannot be two')
    _hold = [bool(hold_focal and known_focals is not None and known_focals[i] is not None)
             for i in range(len(imgs))]

    diags = imsizes.float().norm(dim=1)
    min_focals = 0.25 * diags  # diag = 1.2~1.4*max(W,H) => beta >= 1/(2*1.2*tan(fov/2)) ~= 0.26
    max_focals = 10 * diags
    # hillview: THE BOUNDS ARE FOR ESTIMATES, NOT FOR FACTS. Every
    # focal was clipped to [0.25, 10] x the image diagonal, the held known ones too: a 512 px square
    # slice cut at 120 deg has focal 147.8 px, and was held at the 181.0 px clamp -- another lens
    # than the one that rendered it, with nothing said. A HELD known focal is now used exactly (the
    # value given, not exp(log(f))), everywhere a focal is read: the cameras, the pair-step prior,
    # the pose init, the end diagnostics, inspect_forces. A free focal, a known start included,
    # keeps the bounds.
    _held_t = torch.tensor(_hold, device=device)
    _known_t = torch.tensor([float(known_focals[i]) if _hold[i] else 1.0 for i in range(len(imgs))],
                            dtype=dtype, device=device)

    def focals_of(log_focals):
        free = torch.cat(log_focals).exp().clip(min=min_focals, max=max_focals)
        return torch.where(_held_t, _known_t, free)

    assert len(mst[1]) == len(pps) - 1

    # hillview --pose_fixed: THE CAMERAS ARE GIVEN, NOT SOLVED. Started at a pose graph's poses, sparse_ga
    # pulled the walk back (on one phone walk, back-steps 1 -> 8); held through the GPS term it bulged 25-30 m
    # where MASt3R's depths are worst. Held here, each camera is the constant
    # [0] (camera-to-world, solver units) with its inverse [1]; the kinematic tree's output is not used, and the
    # depth scale is each view's own (global_scaling 1: with the cameras fixed there is no gauge for the
    # optimiser to crush, and a shared factor would tie every view's depth to the smallest view's size).
    _pose_fixed = [None, None]

    # hillview: HELD_TARGETS, as one batch (view index, pixel, fixed point, confidence) for both stages' terms
    _held = None
    if HELD_TARGETS:
        if restore is not None or pose_fixed:
            raise ValueError('HELD_TARGETS with a restore or pose_fixed: the first holds nothing new to solve, the '
                             'second holds every camera already')
        _hv, _hp, _hx, _hc = [], [], [], []
        for v, (pix_, X_, cf_) in sorted(HELD_TARGETS.items()):
            pix_, X_, cf_ = (torch.as_tensor(a, dtype=dtype, device=device) for a in (pix_, X_, cf_))
            if not 0 <= int(v) < len(imgs):
                raise ValueError(f'HELD_TARGETS: view {v} is not one of the {len(imgs)} views solved')
            if pix_.ndim != 2 or pix_.shape[1] != 2 or X_.shape != (len(pix_), 3) or cf_.shape != (len(pix_),):
                raise ValueError(f'HELD_TARGETS[{v}]: pixels (N, 2), points (N, 3), confidences (N,) wanted, got '
                                 f'{tuple(pix_.shape)}, {tuple(X_.shape)}, {tuple(cf_.shape)}')
            if not (torch.isfinite(pix_).all() and torch.isfinite(X_).all() and torch.isfinite(cf_).all()
                    and bool((cf_ > 0).all())):
                raise ValueError(f'HELD_TARGETS[{v}]: a non-finite value or a confidence <= 0')
            _hv.append(torch.full((len(pix_),), int(v), dtype=torch.long, device=device))
            _hp.append(pix_)
            _hx.append(X_)
            _hc.append(cf_)
        _held = (torch.cat(_hv), torch.cat(_hp), torch.cat(_hx), torch.cat(_hc))
        print(f' >> held targets: {len(_held[0])} fixed point(s) on {len(HELD_TARGETS)} of {len(imgs)} '
              f'view(s), confidence sum {float(_held[3].sum()):.4g}; the frame and scale are the targets\' '
              f'(global_scaling 1)' + ('; the GPS gauge is FIXED (the source\'s alignment)' if FIXED_GAUGE is not None
                                       else ''))

    _held_pile = [False]        # held targets: True only while _set_initial_poses takes a normaliser at the pile (_with_held)

    def _scaling(sizes):
        # the frame's scale factor: the smallest view's size (upstream: the optimiser crushes the scale otherwise),
        # or 1 where something else fixes it -- given cameras (pose_fixed) or fixed points (the held targets)
        return 1.0 if (_pose_fixed[0] is not None or _held is not None) else 1 / sizes.min()

    def make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth):
        # make intrinsics
        focals = focals_of(log_focals)
        pps = torch.stack(pps)
        K = torch.eye(3, dtype=dtype, device=device)[None].expand(len(imgs), 3, 3).clone()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, 0:2, 2] = pps * imsizes
        if trans is None:
            return K

        # security! optimization is always trying to crush the scale down
        sizes = torch.cat(log_sizes).exp()
        global_scaling = _scaling(sizes)

        # compute distance of camera to focal plane
        # tan(fov) = W/2 / focal
        z_cameras = sizes * median_depths * focals / base_focals

        # smart reparameterizaton of cameras (computed here already: the rig needs it)
        trans_offset = z_cameras.unsqueeze(1) * torch.cat((imsizes / focals.unsqueeze(1) * (0.5 - pps), ones), dim=-1)

        # make extrinsic
        rel_cam2cam = torch.eye(4, dtype=dtype, device=device)[None].expand(len(imgs), 4, 4).clone()
        rel_cam2cam[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(torch.stack(quats), dim=1))
        rel_cam2cam[:, :3, 3] = torch.stack(trans)
        if _fastp('rig') and _rig_js:
            # all rig links at once (below: the same, one link at a time)
            rel_cam2cam[_rig_j_t, :3, :3] = _rig_R_t
            rel_cam2cam[_rig_j_t, :3, 3] = ((_rig_R_t @ trans_offset[_rig_j_t].unsqueeze(-1)).squeeze(-1)
                                            - trans_offset[_rig_i_t])
        for j, (i, rel_R) in (rig_edges.items() if not (_fastp('rig') and _rig_js) else ()):
            # cam_j_to_cam_i: the known rotation, and the translation that puts camera j's
            # CENTRE on camera i's once both are shifted back from their focal planes:
            # c_j = t_j - R_j o_j  with  t_j = t_i + R_i rel_t,  R_j = R_i rel_R
            # c_j = c_i  <=>  rel_t = rel_R o_j - o_i
            rel_cam2cam[j, :3, :3] = rel_R
            rel_cam2cam[j, :3, 3] = rel_R @ trans_offset[j] - trans_offset[i]

        # camera are defined as a kinematic chain
        if _fastp('chain'):
            # hillview: one batched product per depth of the tree instead of one per camera
            tmp_cam2w = rel_cam2cam.clone()
            for par, ch in _levels:
                tmp_cam2w[ch] = tmp_cam2w[par] @ rel_cam2cam[ch]
        else:
            tmp_cam2w = [None] * len(K)
            tmp_cam2w[mst[0]] = rel_cam2cam[mst[0]]
            for i, j in mst[1]:
                # i is the cam_i_to_world reference, j is the relative pose = cam_j_to_cam_i
                tmp_cam2w[j] = tmp_cam2w[i] @ rel_cam2cam[j]
            tmp_cam2w = torch.stack(tmp_cam2w)

        # smart reparameterizaton of cameras
        new_trans = global_scaling * (tmp_cam2w[:, :3, 3:4] - tmp_cam2w[:, :3, :3] @ trans_offset.unsqueeze(-1))
        cam2w = torch.cat((torch.cat((tmp_cam2w[:, :3, :3], new_trans), dim=2),
                          vec0001.view(1, 1, 4).expand(len(K), 1, 4)), dim=1)
        if _pose_fixed[0] is not None:
            cam2w = _pose_fixed[0]

        if _fastp('depth') and not lora_depth and depth_mode in ('add', 'mul'):
            # hillview: all images' depth in one elementwise pass (was a loop of ~6 kernels per
            # image); the same arithmetic per element, split back into per-image views
            lens = [d.numel() for d in core_depth]
            reps = torch.as_tensor(lens, device=device)
            cd = torch.cat(core_depth)
            if exp_depth:
                cd = cd.exp()
            zc = z_cameras.repeat_interleave(reps)
            if depth_mode == 'add':
                cd = zc + (cd - 1) * (median_depths * sizes).repeat_interleave(reps)
            else:
                cd = zc * cd
            flat = global_scaling * cd
            # hillview: depth must be POSITIVE. A camera cannot see anything behind itself,
            # but depth_mode='add' is unbounded below (z_cam + (cd-1)*median*size), so a hard
            # enough pull drives a whole frame through zero -- frame 9 of one town walk:
            # 16.7% of its depthmap at z<=0, the entire lower two thirds of the image, camera
            # 0.15 m above the ground. With the defaults now shipped (mul + exp_depth) this can
            # never fire; it is here so that --depth_mode add stays merely bad, not unphysical.
            flat = flat.clamp(min=1e-3)
            depthmaps = list(torch.split(flat, lens))
            _depth_flat[0], _depth_flat[1] = flat, depthmaps
            return K, ((_pose_fixed[1] if _pose_fixed[0] is not None else inv(cam2w)), cam2w), depthmaps
        depthmaps = []
        for i in range(len(imgs)):
            core_depth_img = core_depth[i]
            if exp_depth:
                core_depth_img = core_depth_img.exp()
            if lora_depth:  # compute core_depth as a low-rank decomposition of 3d points
                core_depth_img = lora_depth_proj[i] @ core_depth_img
            if depth_mode == 'add':
                core_depth_img = z_cameras[i] + (core_depth_img - 1) * (median_depths[i] * sizes[i])
            elif depth_mode == 'mul':
                core_depth_img = z_cameras[i] * core_depth_img
            else:
                raise ValueError(f'Bad {depth_mode=}')
            depthmaps.append((global_scaling * core_depth_img).clamp(min=1e-3))  # hillview: see above

        return K, ((_pose_fixed[1] if _pose_fixed[0] is not None else inv(cam2w)), cam2w), depthmaps

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)

    if shared_intrinsics == 'groups':
        print('init focals (per group) = ' + ', '.join(
            f'{fg}: {float(K[idx[0], 0, 0].detach()):.2f}' for fg, idx in fgroups.items()))
    elif shared_intrinsics:
        print('init focal (shared) = ', to_numpy(K[0, 0, 0]).round(2))
    else:
        print('init focals =', to_numpy(K[:, 0, 0]))

    # spectral low-rank projection of depthmaps
    if lora_depth:
        core_depth, lora_depth_proj = spectral_projection_of_depthmaps(
            imgs, K, core_depth, subsample, cache_path=cache_path, **lora_depth)
    if exp_depth:
        core_depth = [d.clip(min=1e-4).log() for d in core_depth]
    # Parameters own their storage: compact preparation grids now ravel as
    # views, whereas the old strided grids ravelled into independent copies.
    core_depth = [nn.Parameter(d.ravel().to(dtype).clone()) for d in core_depth]
    log_sizes = [nn.Parameter(torch.zeros(1, dtype=dtype, device=device)) for _ in range(len(imgs))]

    # Fetch img slices
    _, confs_sum, imgs_slices = corres

    # Define which pairs are fine to use with matching
    is_matching_ok = {}
    for s in imgs_slices:
        is_matching_ok[s.img1, s.img2] = pair_matching_ok(s.confs, matching_conf_thr)

    # Prepare slices and corres for losses
    # hillview: the decisions condense_data made with the caller's policy (POINTMAP_POLICY), as data: what
    # sparse_global_alignment passes, so nothing callable crosses into an optimiser process. A direct caller passes
    # its policy instead (evaluated here), or neither, and gets upstream's rule, 'default', and only that
    _pairs = [(int(s.img1), int(s.img2)) for s in imgs_slices]
    if pointmap_selection is None:
        if callable(pointmap_pair_inputs):
            pointmap_pair_inputs = pointmap_pair_inputs()
        _selection = select_pointmap_terms(_pairs, {pair for pair, ok in is_matching_ok.items() if ok},
                                           pointmap_pairs, pointmap_pair_inputs, pointmap_policy)
    else:
        if pointmap_policy is not None:
            raise ValueError('pointmap_selection is the decisions a policy made: pass it or pointmap_policy, not both')
        if [(d['view_i'], d['view_j']) for d in pointmap_selection] != _pairs:
            raise ValueError(f'pointmap_selection holds {len(pointmap_selection)} decision(s), not one per pair of '
                             f'the {len(_pairs)} in order')
        _selection = pointmap_selection
    dust3r_slices = [s for s, decision in zip(imgs_slices, _selection) if decision['selected']]
    loss3d_slices = [s for s in imgs_slices if is_matching_ok[s.img1, s.img2]]
    if pointmap_fallback is not None:
        pointmap_fallback.update(
            matching_conf_thr=float(matching_conf_thr), loss_dust3r_w=float(loss_dust3r_w),
            n_matched=len(loss3d_slices), n_fallback=len(dust3r_slices),
            matched=[[int(s.img1), int(s.img2)] for s in loss3d_slices],
            pointmap_selection=dict(expression=pointmap_pairs, pairs=_selection),
            pairs=[dict(policy_added=bool(is_matching_ok[s.img1, s.img2]), img1=int(s.img1), img2=int(s.img2), n_corres=int(s.confs.numel()),
                        max_conf=float(s.confs.max()) if s.confs.numel() else None,
                        confsum=float(s.confs_sum)) for s in dust3r_slices])
    cleaned_corres2d = []
    for cci, (img1, pix1, confs, confsum, imgs_slices) in enumerate(corres2d):
        cf_sum = 0
        pix1_filtered = []
        confs_filtered = []
        curstep = 0
        cleaned_slices = []
        for img2, slice2 in imgs_slices:
            if is_matching_ok[img1, img2]:
                tslice = slice(curstep, curstep + slice2.stop - slice2.start, slice2.step)
                pix1_filtered.append(pix1[tslice])
                confs_filtered.append(confs[tslice])
                cleaned_slices.append((img2, slice2))
            curstep += slice2.stop - slice2.start
        if pix1_filtered != []:
            pix1_filtered = torch.cat(pix1_filtered)
            confs_filtered = torch.cat(confs_filtered)
            cf_sum = confs_filtered.sum()
        cleaned_corres2d.append((img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices))

    # what each iteration's losses run over: pairs whose matching is trusted go to the 3D loss
    # (batched below); the rest fall back to DUSt3R's pointmap regression, one slice at a time
    print(f' >> pair terms: {len(loss3d_slices)} matched pair(s) -> 3D loss, '
          f'{len(dust3r_slices)} pair(s) -> pointmap loss ({pointmap_pairs})')
    # hillview: THE POINTMAP FALLBACK CONSTRAINS EACH ANCHOR AT
    # THE ANCHOR'S OWN PIXEL. A source view's correspondences sit at arbitrary pixels -- the
    # reciprocal hops move the seeds off the grid, and seed_fill changes its stride -- while its
    # partner's prediction is stored only at the anchor pixels (sample_at_anchors: one per anchor,
    # the canonical cache keeps nothing denser). So the term no longer compares the point at a
    # correspondence's pixel with a prediction from another pixel: correspondence k pulls the point
    # of ITS ANCHOR -- the anchor's core depth, along the ray through the anchor's pixel, offset 1 --
    # toward the prediction at that same pixel. One term per correspondence as before, so the
    # weighting is unchanged; the depths and poses it reaches are the same parameters. Exact for
    # every pixel it touches, where gathering targets at the correspondence pixels would have
    # needed the full-resolution predictions, or a canonical cache keyed by the correspondences.
    _grid = {}
    for i in sorted({int(s_.img1) for s_ in dust3r_slices}):
        g_ = anchor_grid_pixels(int(imsizes[i][0]), int(imsizes[i][1]), subsample, device, dtype)
        n_ = lora_depth_proj[i].shape[0] if lora_depth else core_depth[i].numel()
        if len(g_) != n_:
            raise ValueError(f'pointmap fallback: view {i} has {n_} depth anchors but its {tuple(imsizes[i].tolist())} '
                             f'image at subsample {subsample} has {len(g_)} anchor pixels')
        _grid[i] = (g_, torch.arange(len(g_), device=device), torch.ones(len(g_), device=device, dtype=dtype))

    def grid_pts3d(K, cam2w, depthmaps):
        # {source view: the world point of each of its anchors}; make_pts3d's arithmetic at offset 1
        return dict(zip(_grid, make_pts3d(_grid, K, cam2w, depthmaps, base_focals=base_focals)))
    # hillview: ONE GATHER INSTEAD OF A SLICE PER PAIR. loss_3d sliced every pair's matched
    # points out of two images and concatenated them, loss_2d sliced every (image, other image)
    # group: thousands of tiny kernels an iteration, launched from one Python thread. On the
    # 4090 a phone walk of 5,272 pairs and 334 images ran 2.07 s/it with the card at 21 %
    # and that thread pegged. The index tensors are built here once; the values
    # gathered, their order and the sums are the same (gradients add up in another order, so
    # the last bits can differ). Frozen cameras, or anchors not keyed 0..N-1, keep the old path.
    # HV_SPARSE_GA_SLOW=1 forces the original per-pair code, for A/B (kernel counts, results)
    _fast = (_SLOW_OFF and list(anchors.keys()) == list(range(len(imgs)))
             and not any(v.get('freeze') for v in init.values()))
    if _fast:
        _off = np.concatenate([[0], np.cumsum([len(anchors[i][0]) for i in range(len(imgs))])]).tolist()

        def _rng(img, sl):
            assert sl.step in (None, 1)
            return torch.arange(_off[img] + sl.start, _off[img] + sl.stop, device=device)
        if loss3d_slices:
            _idx3d_1 = torch.cat([_rng(s.img1, s.slice1) for s in loss3d_slices])
            _idx3d_2 = torch.cat([_rng(s.img2, s.slice2) for s in loss3d_slices])
            _confs3d = torch.cat([s.confs for s in loss3d_slices])
        _idx2d = {img1: torch.cat([_rng(img2, sl2) for img2, sl2 in cleaned])
                  for img1, _p, _c, _cs, cleaned in cleaned_corres2d if cleaned}
    if _fast:
        def _chunks_of(counts):
            # consecutive images grouped into chunks of at most _CHUNK points: a few batched
            # gathers per iteration, and no per-point matrix copy larger than a chunk
            out, a, n = [], 0, 0
            for c in counts:
                if n and n + c > _CHUNK:
                    out.append((a, a + n)); a, n = a + n, 0
                n += c
            if n:
                out.append((a, a + n))
            return out
        _counts = [len(anchors[i][0]) for i in range(len(imgs))]
        _img_of = torch.repeat_interleave(torch.arange(len(imgs), device=device),
                                          torch.as_tensor(_counts, device=device))
        _pix_all = torch.cat([anchors[i][0] for i in range(len(imgs))])
        _offs_all = torch.cat([anchors[i][2] for i in range(len(imgs))])
        _pt_chunks = _chunks_of(_counts)
        _pts_didx = []
        # the pointmap fallback's targets never change: concatenate them once (one measured walk
        # had 298 of its 664 pairs here, and they were the last per-pair loop in the iteration)
        _d3 = [(s_.img1, s_.img2) for s_ in dust3r_slices]
        if _d3:
            _d3_tgt, _d3_conf = pack_pointmap_targets(imgs, _d3, preds_21)
            _d3_cfsum = _d3_conf.sum()
            _d3_src = torch.cat([torch.arange(_off[i], _off[i] + len(anchors[i][0]), device=device) for i, _ in _d3])
            _d3_cam = torch.repeat_interleave(torch.as_tensor([j for _, j in _d3], device=device),
                                              torch.as_tensor([len(anchors[i][0]) for i, _ in _d3], device=device))
            _d3_chunks = _chunks_of([len(anchors[i][0]) for i, _ in _d3])
            # the source views' anchor points packed once per iteration (_grid_pts_fast), and per
            # term the one its correspondence belongs to (see the pointmap fallback above)
            _g_views = list(_grid)
            _g_off = dict(zip(_g_views, np.concatenate([[0], np.cumsum([len(_grid[i][0]) for i in _g_views])]).tolist()))
            _g_uv1 = torch.cat([_grid[i][0] for i in _g_views])
            _g_img = torch.repeat_interleave(torch.as_tensor(_g_views, device=device),
                                             torch.as_tensor([len(_grid[i][0]) for i in _g_views], device=device))
            _g_didx = []
            _d3_gidx = torch.cat([anchors[i][1].to(device) + _g_off[i] for i, _ in _d3])
        _l2 = [(img1, pix, cf) for img1, pix, cf, _cs, _cl in cleaned_corres2d if img1 in _idx2d]
        if _l2:
            _l2_pix = torch.cat([pix for _, pix, _ in _l2])
            _l2_conf = torch.cat([cf for _, _, cf in _l2])
            _l2_idx = torch.cat([_idx2d[img1] for img1, _, _ in _l2])
            _l2_img_of = torch.repeat_interleave(torch.as_tensor([img1 for img1, _, _ in _l2], device=device),
                                                 torch.as_tensor([len(pix) for _, pix, _ in _l2], device=device))
            _l2_chunks = _chunks_of([len(pix) for _, pix, _ in _l2])

    _cat_memo = [None, None]

    def make_pts3d_fast(K, cam2w, depthmaps):
        # every image's anchor points in batched chunks: the arithmetic of make_pts3d per element
        if not _pts_didx:
            doff = np.concatenate([[0], np.cumsum([len(d) for d in depthmaps])]).tolist()
            _pts_didx.append(torch.cat([anchors[i][1] + doff[i] for i in range(len(imgs))]))
        depth_flat = _depth_flat[0] if _depth_flat[1] is depthmaps else torch.cat(depthmaps)
        ratio = base_focals / K[:, 0, 0]
        invK = inv(K)
        d_diag = invK.diagonal(dim1=1, dim2=2)
        c_col = invK[:, :, 2] * mask110(device, dtype)
        R, t = cam2w[:, :3, :3], cam2w[:, :3, 3]
        out = []
        for a, b in _pt_chunks:
            img = _img_of[a:b]
            offs = 1 + (_offs_all[a:b] - 1) * ratio[img]
            p = (depth_flat[_pts_didx[0][a:b]] * offs).unsqueeze(-1) * (_pix_all[a:b] * d_diag[img] + c_col[img])
            out.append((R[img] @ p.unsqueeze(-1)).squeeze(-1) + t[img])
        flat = (out[0] if len(out) == 1 else torch.cat(out)) if out else depth_flat.new_empty((0, 3))
        pts = list(torch.split(flat, _counts))
        _cat_memo[0], _cat_memo[1] = pts, flat
        return pts

    def _grid_pts_fast(K, cam2w, depthmaps):
        # the pointmap fallback's source points: every anchor of every source view, batched --
        # make_pts3d_fast's arithmetic with the offset 1 an anchor has at its own pixel
        if not _g_didx:
            if [len(depthmaps[i]) for i in _g_views] != [len(_grid[i][0]) for i in _g_views]:
                raise ValueError('pointmap fallback: depth maps do not match the anchor grids')
            doff = np.concatenate([[0], np.cumsum([len(d) for d in depthmaps])]).tolist()
            _g_didx.append(torch.cat([_grid[i][1] + doff[i] for i in _g_views]))
        depth_flat = _depth_flat[0] if _depth_flat[1] is depthmaps else torch.cat(depthmaps)
        invK = inv(K)
        d_diag = invK.diagonal(dim1=1, dim2=2)
        c_col = invK[:, :, 2] * mask110(device, dtype)
        img = _g_img
        p = depth_flat[_g_didx[0]].unsqueeze(-1) * (_g_uv1 * d_diag[img] + c_col[img])
        return (cam2w[:, :3, :3][img] @ p.unsqueeze(-1)).squeeze(-1) + cam2w[:, :3, 3][img]

    def _all_pts(pts3d):
        # every image's anchor points in one tensor, once per iteration however many losses ask
        if _cat_memo[0] is not pts3d:
            _cat_memo[0], _cat_memo[1] = pts3d, torch.cat(pts3d)
        return _cat_memo[1]

    def loss_dust3r(K, cam2w, depthmaps, pix_loss):
        # In the case no correspondence could be established, fallback to DUSt3R GA regression loss formulation (sparsified)
        # hillview: each term's source is its correspondence's ANCHOR point, at the pixel
        # its target was predicted for (see _grid) -- not the point at the correspondence
        if _fast and _fastp('d3r'):
            # hillview: every weak pair's pointmap in batched chunks (the same arithmetic per point)
            if not _d3:
                return 0.
            src = _grid_pts_fast(K, cam2w, depthmaps)
            loss = 0.
            for a, b in _d3_chunks:
                cam = _d3_cam[a:b]
                tgt = ((cam2w[:, :3, :3][cam] @ _d3_tgt[a:b].unsqueeze(-1)).squeeze(-1)
                       + cam2w[:, :3, 3][cam])
                loss = loss + _d3_conf[a:b] @ pix_loss(src[_d3_gidx[a:b]], tgt)
            return loss / _d3_cfsum if _d3_cfsum != 0. else 0.
        if not dust3r_slices:
            return 0.
        grid_pts = grid_pts3d(K, cam2w, depthmaps)
        loss = 0.
        cf_sum = 0.
        for s in dust3r_slices:
            if init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                continue
            # fallback to dust3r regression
            tgt_pts, tgt_confs = preds_21[imgs[s.img2]][imgs[s.img1]]
            tgt_pts = geotrf(cam2w[s.img2], tgt_pts)
            cf_sum += tgt_confs.sum()
            loss += tgt_confs @ pix_loss(grid_pts[s.img1][anchors[s.img1][1]], tgt_pts)
        return loss / cf_sum if cf_sum != 0. else 0.

    def _with_held(num, den, K, w2cam, pix_loss, stage):
        # the held targets inside the stage's own confidence-weighted mean -- a target is a correspondence
        # whose other end is a fixed point: its distance from the view's ray (stage 1), its reprojection (stage 2).
        # Called only with targets: without them every loss below keeps its own expression, unchanged.
        hv, hp, hx, hc = _held
        if _held_pile[0]:
            # a NORMALISER is being taken at the pile (_set_initial_poses): the pile sits at the source frame's origin
            # by accident of its gauge, so the targets' residuals there measure nothing; their WEIGHT stays in the
            # mean, so a hold's fraction weighs against the pair terms as they sit in this objective -- the same
            # relative weight a solve of these views alone would give it
            return num / (den + hc.sum())
        if stage == 3:
            num = num + hc @ pix_loss(held_ray_closest(w2cam, K, hv, hp, hx), hx)
        else:
            num = num + hc @ pix_loss(hp, held_reproj(K, w2cam, hv, hx))
        return num / (den + hc.sum())

    def loss_3d(K, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        if _fast and _fastp('l3d'):
            if not loss3d_slices:
                return 0. if _held is None else _with_held(0., 0., K, w2cam, pix_loss, 3)
            all_pts = _all_pts(pts3d)
            num3 = _confs3d @ pix_loss(all_pts[_idx3d_1], all_pts[_idx3d_2])
            return num3 / _confs3d.sum() if _held is None else _with_held(num3, _confs3d.sum(), K, w2cam, pix_loss, 3)
        if any(v.get('freeze') for v in init.values()):
            pts3d_1 = []
            pts3d_2 = []
            confs = []
            for s in loss3d_slices:
                if init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                    continue
                pts3d_1.append(pts3d[s.img1][s.slice1])
                pts3d_2.append(pts3d[s.img2][s.slice2])
                confs.append(s.confs)
        else:
            pts3d_1 = [pts3d[s.img1][s.slice1] for s in loss3d_slices]
            pts3d_2 = [pts3d[s.img2][s.slice2] for s in loss3d_slices]
            confs = [s.confs for s in loss3d_slices]

        if pts3d_1 != []:
            confs = torch.cat(confs)
            pts3d_1 = torch.cat(pts3d_1)
            pts3d_2 = torch.cat(pts3d_2)
            loss = confs @ pix_loss(pts3d_1, pts3d_2)
            cf_sum = confs.sum()
        else:
            loss = 0.
            cf_sum = 1.
            if _held is not None:
                return _with_held(0., 0., K, w2cam, pix_loss, 3)

        return loss / cf_sum if _held is None else _with_held(loss, cf_sum, K, w2cam, pix_loss, 3)

    def loss_2d(K, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        proj_matrix = K @ w2cam[:, :3]
        if _fast and _fastp('l2d'):
            # hillview: all images' reprojections in batched chunks (reproj2d per element)
            if not _l2:
                return 0. if _held is None else _with_held(0., 0., K, w2cam, pix_loss, 2)
            all_pts = _all_pts(pts3d)
            loss = 0.
            for a, b in _l2_chunks:
                P = proj_matrix[_l2_img_of[a:b]]
                res = (P[:, :3, :3] @ all_pts[_l2_idx[a:b]].unsqueeze(-1)).squeeze(-1) + P[:, :3, 3]
                uv = (res[:, 0:2] / res[:, 2:3].clip(min=1e-3)).clip(min=-1000, max=2000)
                loss = loss + _l2_conf[a:b] @ pix_loss(_l2_pix[a:b], uv)
            npix = _l2_conf.sum()
            if _held is not None:
                return _with_held(loss, npix, K, w2cam, pix_loss, 2)
            return loss / npix if npix != 0 else 0.
        loss = npix = 0
        for img1, pix1_filtered, confs_filtered, cf_sum, cleaned_slices in cleaned_corres2d:
            if init[imgs[img1]].get('freeze', 0) >= 1:
                continue  # no need
            if _fast and _fastp('l2d'):
                pts3d_in_img1 = _all_pts(pts3d)[_idx2d[img1]] if img1 in _idx2d else []
            else:
                pts3d_in_img1 = [pts3d[img2][slice2] for img2, slice2 in cleaned_slices]
            if (_fast and _fastp('l2d')) and len(pts3d_in_img1):
                loss += confs_filtered @ pix_loss(pix1_filtered, reproj2d(proj_matrix[img1], pts3d_in_img1))
                npix += confs_filtered.sum()
            elif not (_fast and _fastp('l2d')) and pts3d_in_img1 != []:
                pts3d_in_img1 = torch.cat(pts3d_in_img1)
                loss += confs_filtered @ pix_loss(pix1_filtered, reproj2d(proj_matrix[img1], pts3d_in_img1))
                npix += confs_filtered.sum()

        if _held is not None:
            return _with_held(loss, npix, K, w2cam, pix_loss, 2)
        return loss / npix if npix != 0 else 0.

    # hillview: the dense cross-view depth-consistency term. For every pair the solve uses,
    # take view i's regular sample grid, lift it through i's depth, project it into view j,
    # and read j's depth where it lands: the two must agree. Gated at `dense_gate` of
    # relative depth so occlusions and genuine outliers do not pull. This is the tie the
    # per-anchor refinement lacks between matched points, and the tie between panoramas a
    # repeating wall never gets from matching (seen on a church lane).
    # (loss3d_slices, not imgs_slices: the corres2d loop above rebinds that name to tuples)
    dense_pairs = sorted({(s_.img1, s_.img2) for s_ in (loss3d_slices + dust3r_slices)}) if dense_anchors else []
    _cur = {}

    # grouped by TARGET view: one geotrf/projection/gather per target over all the source
    # points that look at it, instead of one per pair (164 small pair loops took 17 s an
    # iteration on the CPU; 36 target groups take a fraction of that)
    _dense_srcs = {}
    for (i, j) in dense_pairs:
        _dense_srcs.setdefault(j, []).append(i)
    _dense_t = {'calls': 0}
    # hillview: THE PAIRS' MASKS. Each source's samples go through its pair's keep indices (none
    # on the source's mask), and per target every sample carries the variant of the target's masks its pair
    # is judged by (-1: none), so a sample landing on a masked pixel is dropped before rel and the gate.
    if dense_pairs and dense_masks_given:
        _no_mask = [p_ for p_ in dense_pairs if p_ not in dense_pair_var]
        if _no_mask:
            raise ValueError(f'dense pairs with no masks: {_no_mask[:5]}... ({len(_no_mask)}): the setup '
                             f'(corres slices) and the optimiser disagree on which pairs there are')
    _src_keep = dense_src_keep or {}
    _view_masks = dense_view_masks or {}

    def _dense_src(i, j, x):
        keep = _src_keep.get((i, j))
        return x if keep is None else x[keep]
    _dense_var = {}
    for j, srcs in _dense_srcs.items():
        if dense_masks_given and j in _view_masks:
            _dense_var[j] = torch.cat([torch.full((len(_dense_src(i, j, dense_anchors[i][0])),), dense_pair_var[(i, j)][1],
                                                  dtype=torch.long, device=dense_anchors[i][0].device) for i in srcs])

    # NEAR-VIEW AUTHORITY (dense_consistency_mode 'near', hillview).
    # 'symmetric' (upstream of this, and the default) lets both views of a disagreement move: a view
    # 80 m from a surface, seeing it a few pixels tall, pulls as hard as one 20 m from it. On
    # one phone walk views 105-112 place the walk-through tower 3-29 m short while 92-103, nearer, agree
    # to 0-2 m: the close-up confident views should overpower the far-away distant guesses. 'near' compares, per sample, the source view's own depth to the point with the target
    # view's depth there: the NEARER view is held whole (depth AND pose), and only the farther view's
    # depth and pose move onto it. Same samples, same gate (0.25: an occluder in front of a near
    # surface reads like a short far view, so a wider gate would pull far depths through occluders);
    # same normalisation except that the denominator is detached in near mode.
    if dense_consistency_mode not in ('symmetric', 'near'):
        raise ValueError(f"dense_consistency_mode {dense_consistency_mode!r}: symmetric or near")

    def loss_dense(K, w2cam, cam2w, depthmaps):
        if not dense_pairs:
            return 0.
        import time as _time
        _t0 = _time.time()
        pts = make_pts3d(dense_anchors, K, cam2w, depthmaps, base_focals=base_focals)
        _t1 = _time.time()
        total, n = 0., 0
        _near = dense_consistency_mode == 'near'
        _check = _dense_t['calls'] < 2 and dense_masks_given
        n_tgt_masked = n_bad_src = n_bad_tgt = 0
        for j, srcs in _dense_srcs.items():
            X = torch.cat([_dense_src(i, j, pts[i]) for i in srcs], 0)
            if _near:
                # each sample's depth from its OWN camera: who is nearer the surface it lies on
                zsrc = torch.cat([geotrf(w2cam[i], _dense_src(i, j, pts[i]))[:, 2] for i in srcs], 0).detach()
            Xc = geotrf(w2cam[j], X)
            z = Xc[:, 2]
            zc = z.clamp(min=1e-3)
            fx, fy, cx, cy = K[j][0, 0], K[j][1, 1], K[j][0, 2], K[j][1, 2]
            u = fx * Xc[:, 0] / zc + cx
            v = fy * Xc[:, 1] / zc + cy
            aidx, aoff, H, W = dense_maps[j]
            inside = (z > 1e-3) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
            if j in _dense_var:
                # the pixel the target's depth is read at below, rounded the same way
                masked = inside & dense_target_masked(_dense_var[j], _view_masks[j], v.detach().long().clamp(0, H - 1),
                                                      u.detach().long().clamp(0, W - 1))
                n_tgt_masked += int(masked.sum())
                inside = inside & ~masked
            if not bool(inside.any()):
                continue
            ui = u[inside].detach().long().clamp(0, W - 1)
            vi = v[inside].detach().long().clamp(0, H - 1)
            dj = depthmaps[j][aidx[vi, ui]] * aoff[vi, ui]
            if _near:
                # ROUTED BY PART, not by z: z mixes the source's point with the TARGET's
                # pose, so detaching z or dj alone either froze the farther target's pose or let the
                # nearer target's pose be dragged. Two transforms instead of one:
                #   source nearer -> the source holds (point, pose, depth); the target's depth AND pose move
                #   target nearer -> the target holds (depth, pose); the source's point and pose move
                src_near = zsrc[inside] < dj.detach()
                zs = geotrf(w2cam[j], X.detach())[:, 2][inside]
                zt = geotrf(w2cam[j].detach(), X)[:, 2][inside]
                zi_ = torch.where(src_near, zs, zt)
                dj_ = torch.where(src_near, dj, dj.detach())
                # (the denominator is detached here; symmetric keeps its gradient, as upstream)
                rel = (zi_ - dj_) / dj.detach().clamp(min=1e-3)
            else:
                rel = (z[inside] - dj) / dj.clamp(min=1e-3)
            gate = (rel.detach().abs() < dense_gate)
            if not bool(gate.any()):
                continue
            if _check:
                # the samples that enter, read back against the masks: on a masked pixel at either end, 0
                used = inside.nonzero().squeeze(1)[gate]
                if j in _dense_var:
                    # indexed here directly, not through dense_target_masked: a check that shares the drop's
                    # code cannot see the drop fail
                    vs_, vg_, ug_ = _dense_var[j][used], vi[gate], ui[gate]
                    h_ = vs_ >= 0
                    n_bad_tgt += int(_view_masks[j][vs_[h_], vg_[h_], ug_[h_]].sum())
                o_ = 0
                for i in srcs:
                    pix_i = _dense_src(i, j, dense_anchors[i][0])
                    seg = used[(used >= o_) & (used < o_ + len(pix_i))] - o_
                    vi_ = dense_pair_var[(i, j)][0]
                    if vi_ >= 0 and len(seg):
                        p_ = pix_i[seg].long()
                        n_bad_src += int(_view_masks[i][vi_][p_[:, 1], p_[:, 0]].sum())
                    o_ += len(pix_i)
            total = total + rel[gate].abs().sum()
            n += int(gate.sum())
        _dense_t['calls'] += 1
        if _dense_t['calls'] in (1, 2):
            print(f"  dense term: {len(dense_pairs)} pairs in {len(_dense_srcs)} target groups, points {_t1 - _t0:.2f}s, "
                  f"groups {_time.time() - _t1:.2f}s, {n} samples in the gate"
                  + (f"; {n_tgt_masked} landing on a target's masked pixel dropped; on a masked pixel in the term: "
                     f"{n_bad_src} source, {n_bad_tgt} target" if dense_masks_given else
                     "; NO masks given: every grid sample of every view is used"))
        return total / max(n, 1)

    # ---------------------------------------------------------------- GPS constraint --
    # (hillview) Until now GPS only judged the FINISHED solve: a similarity was fitted to it
    # and the residual reported. 334 frames of a walk therefore had nothing pulling them
    # back, and one such walk drifted into a knot while every pairwise number looked fine.
    # This puts the phone's own positions into the objective, weighted by the accuracy the
    # receiver claimed for each fix.
    #
    # The similarity between solver units and ENU metres is refitted every iteration on
    # DETACHED centres, so the term cannot be reduced by rotating or rescaling the whole
    # reconstruction — that is gauge, and the fit removes it. What is left is shape: where
    # the walk bends, stretches or folds against what the GPS says.
    #
    # One similarity per GROUP. A walk that has broken into pieces is then pulled piece by
    # piece, each by its own GPS, rather than by one compromise fit that smears every
    # piece's error over all the others.
    # BUILT FOR EITHER TERM. --grade_weight reads the vertical in this same gauge and is meant to
    # be usable with the GPS PULL OFF (gps_weight 0) -- which is the interesting arm, since the
    # pull is what wrecked a church ride's GPS-weighted solve while the vertical it leaves free is what
    # --grade_weight is for. Gating the groups on gps_weight alone made that arm silently a no-op.
    _gps_groups = []
    if gauge_tilt_from not in ('mount', 'height_steps'):
        raise ValueError(f'gauge_tilt_from {gauge_tilt_from!r}: mount or height_steps (auto is the caller\'s to resolve)')
    if ((gps_weight or gps_frac or grade_weight or rig_upright_weight or gps_rel_frac or weak_mid_frac or alt_step_frac
         or gauge_tilt_from == 'height_steps') and gps is not None):
        _g = torch.as_tensor(np.asarray(gps, dtype=np.float64), dtype=dtype, device=device)
        # A VIEW IS POSITIONED BY ITS HORIZONTAL FIX. Its ALTITUDE is a separate question, and one
        # most sources do not answer: every Panoramax frame carries altitude null. Once,
        # reconstruct.py filled an unknown altitude with 0.0, which after centring is
        # exactly the mean height of the frames that HAVE one -- so a run with no altitudes at all
        # was pulled toward a perfectly flat fictitious plane, and every Panoramax solve so far
        # has been. NaN now means unknown, the horizontal fix alone admits a view, and the height
        # of a view whose altitude nobody recorded is constrained by nothing (see loss_gps).
        _ok = torch.isfinite(_g[:, :2]).all(dim=1)
        _zok = torch.isfinite(_g[:, 2]) & _ok
        _w = (torch.as_tensor(np.asarray(gps_w, dtype=np.float64), dtype=dtype, device=device)
              if gps_w is not None else torch.ones(len(imgs), dtype=dtype, device=device))
        _w = torch.where(_ok, _w.clamp(min=0), torch.zeros_like(_w))
        grp = np.asarray(gps_groups if gps_groups is not None else np.zeros(len(imgs), int))
        for gid in sorted({int(x) for x in grp}):
            idx = torch.as_tensor(np.where(grp == gid)[0], device=device)
            sel = idx[_ok[idx]]
            if len(sel) >= 3 and float(_w[sel].sum()) > 0:
                _gps_groups.append((sel, _g[sel], _w[sel] / _w[sel].sum(), _zok[sel]))
        _n_noalt = int((_ok & ~_zok).sum())
        print(f' >> GPS gauge: {sum(len(s) for s, _, _, _ in _gps_groups)} of {len(imgs)} view(s) '
              f'in {len(_gps_groups)} group(s); '
              # SAY WHICH TERM IS ON, because there are TWO and this line named only one. With
              # `gps_frac` active and `gps_weight` 0 it read "PULL OFF (gps_weight 0)" while the
              # normalised term was pulling all along, and that line was taken as proof
              # the pull was off.
              + (f'pull weight {gps_weight} (ABSOLUTE METRES -- prefer --gps_frac), '
                 f'huber {gps_huber_m} m, {gps_pull_min}-{gps_pull_max}'
                 if gps_weight else
                 f'metres term OFF (gps_weight 0)')
              + (f'; normalised term ON at gps_frac {gps_frac:.0%} of the pixel objective'
                 if gps_frac else
                 ('' if gps_weight else ', gauge built for ' + ' + '.join(
                     n_ for n_, on_ in (('--grade_weight', grade_weight), ('--rig_upright_weight', rig_upright_weight),
                                        ('--gps_rel_frac', gps_rel_frac), ('--weak_mid_frac', weak_mid_frac),
                                        ('--alt_step_frac', alt_step_frac),
                                        ('--gauge_tilt_from height_steps', gauge_tilt_from == 'height_steps')) if on_)
                  + ' only'))
              + (f'; {_n_noalt} of them carry NO ALTITUDE, so their height is constrained by nothing'
                 if _n_noalt else '')
              + ('' if _gps_groups else ' — NOTHING to constrain (needs 3 positioned views in a group)'))
        if rig_upright_weight and _gps_groups:
            print(f' >> rig upright prior: every view\'s RIGHT axis held within '
                  f'{rig_upright_slack_deg} deg of horizontal in the GPS gauge, weight '
                  f'{rig_upright_weight} — the mount, not the solve\'s own opinion of up')
        if grade_weight and _gps_groups:
            print(f' >> grade prior: consecutive-frame rise/step held under {grade_slack:.0%}, '
                  f'weight {grade_weight}, pairs shorter than {grade_min_step_frac:.0%} of the '
                  f'median step ignored'
                  + (f' — and it is the ONLY thing holding the vertical here, since {_n_noalt} '
                     f'view(s) carry no altitude' if _n_noalt else ''))
        elif grade_weight:
            print(f' >> grade prior: asked for weight {grade_weight} but there are no GPS groups; '
                  f'the term reads the vertical in the GPS gauge, so it needs positions — not applying it')

    # GRAVITY, FROM THE MOUNT (hillview). A 360 camera on a pole, a helmet, a handlebar
    # or a car roof is UPRIGHT by construction, across walks, rides and drives -- on the church lane
    # the panoramas were shot from a bicycle ridden reasonably slowly, so pitch and roll can very much
    # be trusted. That is a fact about how the data was made, not a model of how the
    # walk moves, and it is the one thing the solve gets badly wrong at length: twelve panoramas of
    # that lane turn the rig's up by 56 and then 70 degrees at two joints where four panoramas hold
    # to 5 (pano_twist.up_turn_deg).
    #
    # The term says the rigs agree with EACH OTHER, not with a known world up: every panorama's up
    # parallel to the mean up. That needs no ENU frame, it is exactly what the metric measures, and
    # it leaves the solve free to sit at whatever absolute tilt the pixels prefer. One view per rig,
    # so a twelve-slice panorama does not outvote a four-slice one -- and any slice will do, because
    # a level ring's yaw is about the rig's own up, which leaves that axis untouched.
    #
    # SLACK, not a hard constraint: a bicycle leans into a turn by a few degrees and rights itself.
    # Below gravity_slack_deg the term is exactly zero, so the ordinary wobble of a real ride costs
    # nothing and only the folds pay.
    _grav_idx = []
    if gravity_weight and rig is not None:
        seen = {}
        for i, r in enumerate(rig):
            if r is not None and r[0] not in seen:
                seen[r[0]] = i
        _grav_idx = sorted(seen.values())
        print(f' >> gravity constraint: {len(_grav_idx)} rig(s) held upright with each other, '
              f'weight {gravity_weight}, slack {gravity_slack_deg} deg'
              + ('' if len(_grav_idx) >= 2 else ' -- NOTHING to constrain (needs 2 rigs)'))
    _grav_t = torch.as_tensor(_grav_idx, device=device) if len(_grav_idx) >= 2 else None
    # The disagreement is measured as a CHORD, |up - mean|, not as an angle: arccos needs its
    # argument clamped away from 1 to keep the gradient finite, and that clamp puts a floor under
    # the reading -- a perfectly level set came out at 0.026 deg instead of 0 (caught by
    # recon_gravity_check.py before any of this ran). The chord is 2 sin(angle/2), smooth at zero,
    # monotone in the angle, and equal to it to within 1 % out to 20 deg.
    _grav_slack = 2.0 * float(np.sin(np.radians(gravity_slack_deg) / 2.0))

    def loss_gravity(cam2w):
        up = -cam2w[_grav_t, :3, 1]                       # OpenCV camera axes: y is down
        up = up / up.norm(dim=1, keepdim=True).clamp(min=1e-9)
        mean = up.mean(dim=0)
        mean = mean / mean.norm().clamp(min=1e-9)
        return (((up - mean).norm(dim=1) - _grav_slack).clamp(min=0.0) ** 2).mean()

    # THE BEARING THE CAPTURE ALREADY RECORDED, which until now NOTHING IN THE SOLVE USED
    # (recon_heading.py says so in capitals, and it was right to: the compass's own noise is about
    # 10 deg, so a term that pulled toward the MEDIAN would drag the better estimate toward the
    # worse one -- the same mistake the GPS position prior makes).
    #
    # THE TAIL IS A DIFFERENT ANIMAL. Measured on one phone walk: honest window pairs disagree
    # with the compass by 10.2 deg median, p90 32.8; on an 80-frame span of it EVERY pair spanning
    # the U-turn read 147-174 deg, because the whole returning leg was solved 180 deg out while its
    # camera CENTRES stayed right (GPS separation 7.2 m, solved 6.7 m). Nothing could see it: GPS
    # constrains positions, gravity constrains roll and pitch, and yaw was observable only through
    # the correspondences, which were happy in the flipped minimum. The dead zone is what separates
    # the two -- anything between 45 and 90 deg passes every honest pair and catches every flip.
    #
    # RELATIVE, NOT ABSOLUTE. The target is the compass's own bearing DIFFERENCE within each pair,
    # so magnetic declination, a yawed mount and the alignment's own yaw all cancel: those are
    # gauge facts that move every frame together, and correcting them per-frame is exactly what
    # recon_heading.py warns against. No north is needed, so this works before any GPS fit.
    #
    # YAW, AND ONLY YAW. The first cut measured the raw 3-D angle between the two forward axes,
    # which CONFLATES PITCH WITH YAW: on that walk the solved pitches run from -51 to +32 deg, so two
    # cameras pointing the same way but pitched apart would read 80 deg of "bearing error" and eat
    # the whole slack. The forwards are therefore projected onto the plane perpendicular to the
    # CONSENSUS up before the angle is taken -- the same up the level prior uses, so no alignment
    # and no north is needed. A camera pointing within a few degrees of straight up or down has no
    # measurable yaw at all; its projection vanishes, and the pair is dropped rather than guessed.
    #
    # The angle is UNSIGNED, so it catches a flip but would pass a mirror (+40 deg read as -40).
    # A deliberate trade for a term with no north in it.
    _cmp = None
    if compass_weight and compass_pairs:
        _cmp = (torch.as_tensor([int(c[0]) for c in compass_pairs], device=device),
                torch.as_tensor([int(c[1]) for c in compass_pairs], device=device),
                torch.as_tensor([float(c[2]) for c in compass_pairs], device=device, dtype=dtype))
        print(f' >> compass constraint: {len(compass_pairs)} pair(s) hold their RELATIVE bearing, '
              f'weight {compass_weight}, free within {compass_slack_deg} deg')
    _cmp_slack = float(np.radians(compass_slack_deg))
    # the up used for the projection comes from every view the caller trusts to be upright
    # when there are any, and from all views otherwise -- it is a direction, not a gauge
    _cmp_up_sel = (torch.as_tensor(sorted(level_idx), device=device)
                   if level_idx is not None and len(level_idx) >= 3
                   else torch.arange(len(imgs), device=device))

    def loss_compass(cam2w):
        i, j, tgt = _cmp
        return compass_penalty(cam2w, i, j, tgt, _cmp_slack, _cmp_up_sel,
                               compass_max_pitch_deg, compass_ramp_deg)

    # A HAND-HELD WALK IS HELD ROUGHLY LEVEL -- a PRIOR, not a measurement, and it is marked as one
    # because no hillview capture records pitch or roll (the UserComment carries
    # bearing_source, location_source, location_age_ms, refined, exposure and, in later captures,
    # location_accuracy_m -- and nothing else). Level is set at about 20 deg under the horizon for
    # now, which matches the solve's own median of -10 deg over that phone walk.
    #
    # WHY IT IS NEEDED AT ALL: gravity_weight does NOT do this job and never did. It holds RIGS
    # upright with EACH OTHER, one view per rig, and auto resolves it to 0.0 on anything that is
    # not a rigged panorama -- so on that 80-frame span it was off, had no rigs to act on,
    # and never printed a line, while the solve dived to -51 deg at frames 63-64 and threw +32 deg
    # at 42. Nothing was holding pitch at all.
    #
    # AGAINST THE CONSENSUS UP, not against a world up, so it needs no alignment and no GPS: it
    # says the walker did not suddenly tilt while everyone else stayed level. Having no world
    # reference, it is invariant to a RIGID ROTATION of the whole solve -- a gauge fact, and the
    # alignment's business rather than this term's. It does NOT lose a pitch that every view
    # shares: cameras facing different ways tilt their ups in different azimuths, those mostly
    # cancel in the mean, so the consensus up stays put and the shared pitch is still charged
    # (recon_compass_check.py holds both halves of that).
    #
    # HILLVIEW ONLY: it applies to hillview captures, not to Panoramax. A sliced panorama's views
    # point deliberately up and down by construction, so the caller passes only the views this can
    # honestly speak for; level_idx is that list.
    _rig_up_slack = float(np.sin(np.radians(rig_upright_slack_deg)))
    _lvl_t = (torch.as_tensor(sorted(level_idx), device=device)
              if level_weight and level_idx is not None and len(level_idx) >= 3 else None)
    if _lvl_t is not None:
        print(f' >> level prior: {len(_lvl_t)} hand-held view(s) held near {level_pitch_deg} deg '
              f'from the horizon, weight {level_weight}, free within {level_slack_deg} deg')
    _lvl_target = float(np.radians(90.0 - level_pitch_deg))    # forward-to-up angle, not the pitch
    _lvl_slack = float(np.radians(level_slack_deg))

    def loss_level(cam2w):
        return level_penalty(cam2w, _lvl_t, _lvl_target, _lvl_slack)

    # the same views the level prior speaks for: hand-held, ours, not a panorama slice
    _hgt_t = (_lvl_t if (height_weight and _lvl_t is not None and len(_lvl_t) >= 3)
              else (torch.as_tensor(sorted(level_idx), device=device)
                    if height_weight and level_idx is not None and len(level_idx) >= 3 else None))
    if _hgt_t is not None:
        print(f' >> height prior: {len(_hgt_t)} hand-held view(s) held at one height, weight '
              f'{height_weight}, free within {height_slack_steps} of a step')

    def loss_height(cam2w):
        return height_penalty(cam2w, _hgt_t, height_slack_steps)

    _hor_t = (_hgt_t if (horizon_weight and _hgt_t is not None)
              else (torch.as_tensor(sorted(level_idx), device=device)
                    if horizon_weight and level_idx is not None and len(level_idx) >= 3 else None))
    if _hor_t is not None:
        print(f' >> horizon prior: {len(_hor_t)} hand-held view(s) kept at or below '
              f'{horizon_max_pitch_deg} deg, weight {horizon_weight}, slack {horizon_slack_deg} deg')
    _hor_max = math.radians(horizon_max_pitch_deg)
    _hor_slack = math.radians(horizon_slack_deg)

    def loss_horizon(cam2w):
        return horizon_penalty(cam2w, _hor_t, _hor_max, _hor_slack)

    _upr_t = (_hgt_t if (upright_weight and _hgt_t is not None)
              else (torch.as_tensor(sorted(level_idx), device=device)
                    if upright_weight and level_idx is not None and len(level_idx) >= 3 else None))
    if _upr_t is not None:
        print(f' >> upright prior: {len(_upr_t)} hand-held view(s) kept the right way up, weight '
              f'{upright_weight}, free within {upright_slack_deg} deg')
    _upr_slack = float(np.radians(upright_slack_deg))

    def loss_upright(cam2w):
        return upright_penalty(cam2w, _upr_t, _upr_slack)

    # ONE FIT PER GROUP PER ITERATION. loss_gps, loss_grade and loss_rig_upright all ask
    # gps_gauge the same question of the same DETACHED centres, so the answer is identical and
    # was being computed up to three times in the inner loop. Keyed on the centres' identity
    # and version, which is what changes when the optimiser steps.
    # Keyed on the GROUP only and cleared once per iteration by the optimiser loop, because
    # `cam2w[sel, :3, 3]` is a fresh tensor on every call -- keying on its identity would miss
    # every time and cache nothing.
    _gauge_cache: dict = {}
    _gps_k = [None]          # the normaliser, fixed at the first iteration and then held

    # THE GAUGE'S TILT FROM GRAVITY (hillview). imu_down[i] is "down" in view i's OpenCV frame,
    # from the phone's accelerometer; R_i @ down_i is where view i says gravity points in the model.
    # Their mean is the model's vertical, measured -- so the gauge fits yaw, scale and translation
    # only, and stops choosing a tilt from the GPS heights (18-21 deg off gravity).
    _imu_idx = _imu_d = None
    if gauge_gravity and imu_down is not None:
        _k = [i for i, d in enumerate(imu_down) if d is not None]
        if len(_k) >= 3:
            _imu_idx = torch.as_tensor(_k, device=device)
            _imu_d = torch.as_tensor([list(imu_down[i]) for i in _k], device=device, dtype=torch.float64)
            print(f' >> gauge from gravity: {len(_k)}/{len(imgs)} view(s) carry a measured down; the GPS '
                  f'fit takes yaw, scale and translation only')
        else:
            print(f' >> gauge from gravity ASKED but only {len(_k)} view(s) carry a measured down: '
                  f'the tilt is fitted to the GPS as before')

    # THE ALTITUDE STEPS (hillview): what should be constrained is the altitude delta.
    # alt_step_links are the caller's consecutive-photograph stand-ins whose two heights come from
    # one DEM: a receiver's altitude, or two DEMs, is no terrain step. A link is held inside one GPS group, both ends
    # positioned, weighted and with a height. gauge_tilt_from 'height_steps' takes each group's down from them
    # (height_step_down) instead of the mounts' mean, and drops the GPS term's ABSOLUTE heights: every row's zok goes
    # False, so loss_gps and gps_rel read the horizontal only and heights enter as steps alone. alt_step_frac holds
    # each step's solved height change to the terrain's in that gauge (alt_step_residual_m), Huber alt_step_huber_m,
    # normalised against one knee per step as weak_mid is: the GPS start puts every camera on its fix, so the term
    # starts at zero and a normaliser taken from its start would be arbitrary.
    _alt_of = {}             # group -> (rows li, rows lj, terrain dz in metres)
    _n_alt = 0
    if alt_step_frac or gauge_tilt_from == 'height_steps':
        if alt_step_links is None:
            raise ValueError('alt_step_frac / gauge_tilt_from height_steps need alt_step_links from the caller')
        _row_of_a = {}
        for gi, (sel, *_r) in enumerate(_gps_groups):
            for r_, v in enumerate(sel.tolist()):
                _row_of_a[int(v)] = (gi, r_)
        _by_g, _a_out = {}, 0
        for i, j in alt_step_links:
            a_, b_ = _row_of_a.get(int(i)), _row_of_a.get(int(j))
            if a_ is None or b_ is None or a_[0] != b_[0]:
                _a_out += 1
                continue
            _az, _aw = _gps_groups[a_[0]][3], _gps_groups[a_[0]][2]
            if not (bool(_az[a_[1]]) and bool(_az[b_[1]]) and float(_aw[a_[1]]) > 0 and float(_aw[b_[1]]) > 0):
                _a_out += 1
                continue
            _by_g.setdefault(a_[0], []).append((a_[1], b_[1]))
        for gi, rr in sorted(_by_g.items()):
            li = torch.as_tensor([a_ for a_, _ in rr], device=device)
            lj = torch.as_tensor([b_ for _, b_ in rr], device=device)
            enu_ = _gps_groups[gi][1]
            _alt_of[gi] = (li, lj, (enu_[lj, 2] - enu_[li, 2]).to(torch.float64))
        _n_alt = sum(len(rr) for rr in _by_g.values())
        print(f' >> altitude steps: {_n_alt} of {len(alt_step_links)} link(s) held inside a GPS group with both '
              f'heights' + (f'; {_a_out} not (another group, unpositioned, unweighted or no height)' if _a_out else '')
              + ''.join(f'; group {gi}: {len(_alt_of[gi][0])} step(s), terrain rise {float(_alt_of[gi][2].sum()):+.1f} m '
                        f'over |dz| median {float(_alt_of[gi][2].abs().median()):.2f} m'
                        for gi in sorted(_alt_of)))
        if gauge_tilt_from == 'height_steps':
            if _imu_idx is None:
                raise ValueError('gauge_tilt_from height_steps corrects the mounts\' down and there is none: it needs '
                                 'imu_down on >= 3 views and gauge_gravity')
            if gauge_tilt_step_floor_m is None or not float(gauge_tilt_step_floor_m) > 0:
                raise ValueError(f'gauge_tilt_from height_steps needs gauge_tilt_step_floor_m > 0 from the caller, '
                                 f'not {gauge_tilt_step_floor_m!r}')
            short = [gi for gi in range(len(_gps_groups)) if gi not in _alt_of or len(_alt_of[gi][0]) < 3]
            if short:
                raise ValueError(f'gauge_tilt_from height_steps: GPS group(s) {short} hold fewer than 3 altitude steps, '
                                 f'and their vertical would be nobody\'s')
            _gps_groups = [(sel, enu_, w_, torch.zeros_like(z_)) for sel, enu_, w_, z_ in _gps_groups]
            print(f' >> gauge tilt from HEIGHT STEPS: each group\'s down is the mounts\' (one sigma '
                  f'{gauge_tilt_mount_sd_deg} deg) corrected so its steps rise as the terrain does (huber '
                  f'{alt_step_huber_m} m, a step credited with {float(gauge_tilt_step_floor_m):g} m at best); the GPS term '
                  f'holds positions HORIZONTALLY, heights enter as steps only')
        if alt_step_frac:
            print(f' >> alt_step: {_n_alt} step(s) held to the terrain\'s height change, huber {alt_step_huber_m} m, '
                  f'{alt_step_frac:.0%} of the pixel objective at one knee per step'
                  + (' -- HOLDS NOTHING: pose_fixed holds every camera' if pose_fixed and _n_alt else '')
                  + ('' if _n_alt else ' -- HOLDS NOTHING: no link with two heights from one DEM in one GPS group'))
    _alt = list(_alt_of.items()) if (alt_step_frac and _n_alt and not pose_fixed) else None

    # THE GYRO (hillview). Each pair's solved rotation R_i^T R_j is held to the phone's
    # own, integrated from its gyro, beyond a dead zone. On one phone walk the bounded GPS term still broke
    # 35-45 steps -- nearly all in the 54 frames no pair holds strongly, whose pixels cannot say;
    # the gyro can, to 1-3 deg between exposures. A pair is (i, j, G) with the default dead zone, or
    # (i, j, G, slack_deg): pairs farther apart in time carry a wider one, since the integrated rotation
    # drifts -- which is what lets the term reach a turnaround, not only the next frame.
    # SUMMED, not averaged: one broken link must cost as much on a long walk as on a short one.
    _gyro = None
    if gyro_steps and gyro_weight > 0:
        _gi = [int(e[0]) for e in gyro_steps]; _gj = [int(e[1]) for e in gyro_steps]
        _sl = [float(e[3]) if len(e) > 3 and e[3] is not None else float(gyro_slack_deg) for e in gyro_steps]
        _gyro = (torch.as_tensor(_gi, device=device), torch.as_tensor(_gj, device=device),
                 torch.as_tensor(np.asarray([np.asarray(e[2], np.float64) for e in gyro_steps]),
                                 device=device, dtype=torch.float64),
                 torch.as_tensor(2.0 * np.sin(np.radians(_sl) / 2.0), device=device, dtype=torch.float64))
        print(f' >> gyro: {len(_gi)} pair(s) held to the phone\'s gyro, weight {gyro_weight}, dead zone '
              f'{min(_sl):.1f}-{max(_sl):.1f} deg')

    def loss_gyro(cam2w):
        ch = gyro_step_chord(cam2w, _gyro[0], _gyro[1], _gyro[2])
        return ((ch - _gyro[3].to(ch.dtype)).clamp(min=0.0) ** 2).sum()

    # EVERY CAMERA'S MEASURED DOWN (hillview). R_i @ down_i is where view i says gravity
    # points in the model; all of them must agree. Gauge-free: the target is their own mean, so
    # rotating the whole solve changes nothing, and the gauge (--gauge_gravity) is what ties that mean
    # to the world. It holds each camera's pitch and roll to the others -- the half of a break the gyro
    # term sees only relative to a neighbour. Chord, dead zone, summed, as the gyro term.
    _gcam = None
    if gravity_cam_weight > 0 and imu_down is not None:
        _k = [i for i, d in enumerate(imu_down) if d is not None]
        if len(_k) >= 3:
            _gcam = (torch.as_tensor(_k, device=device),
                     torch.as_tensor([list(imu_down[i]) for i in _k], device=device, dtype=torch.float64))
            print(f' >> gravity per camera: {len(_k)}/{len(imgs)} view(s) held to one down, weight '
                  f'{gravity_cam_weight}, slack {gravity_cam_slack_deg} deg')
    _gcam_slack = 2.0 * float(np.sin(np.radians(gravity_cam_slack_deg) / 2.0))

    def loss_gravity_cam(cam2w):
        R_ = cam2w[_gcam[0], :3, :3]
        g_ = (R_ @ _gcam[1].to(R_.dtype)[:, :, None])[:, :, 0]
        m_ = g_.mean(0); m_ = m_ / m_.norm().clamp(min=1e-12)
        ch = torch.sqrt(((g_ - m_) ** 2).sum(1) + 1e-12)
        return ((ch - _gcam_slack).clamp(min=0.0) ** 2).sum()

    # ========================== THE AXIS WE INJECT (hillview) ==========================
    # EVERY OTHER ATTITUDE TERM HERE IS RELATIVE, and that is the bug that kept showing
    # up. loss_gravity_cam pulls each camera to the cameras' own RUNNING MEAN; loss_rig_upright to
    # the GPS-FITTED gauge; upright_penalty to the walk's consensus. A solve that tips AS A WHOLE
    # satisfies all three at zero cost, which is how a bike ride comes back with a street tilted
    # and every metric content. Nothing in the objective has ever said which way is really down.
    # There is no reason to allow the cameras to invent their own up/down direction: what is
    # wanted is an up-down axis injected into the world.
    #
    # So the axis is FIXED: taken once, at the start pose, from the mount's DECLARATION
    # (--gauge_up_from_mount puts the rig's own +Y into imu_down for every level slice), and every
    # mount is held to that constant for the whole optimisation. A GPS start (pose_init) turns the
    # visual chain so that the mounts' AVERAGE declared down is ENU's down, so the axis is
    # (0, 0, -1) to four places on every run of a sweep over it -- the start's average, held; each
    # mount keeps the chain's own tilt, so a grade or a fold already in the start chain is in the
    # axis too. A whole-solve tilt during the solve costs, which is the point: it is
    # the term that was wanted, the only one that does not move with the solve.
    # For a while the target was the consensus RE-TAKEN AT EVERY STEP, on
    # the argument that the frozen one went stale after Adam's first-step swing. That made
    # this term relative like every other one: measured against the mounts' mean, a roll fold
    # about the track on a 28-panorama town span reads -9 deg on one half and +12 on the other, so a
    # relative term sees each half only part of the way off. The staleness was measured with the
    # SUMMED term (27x the pixel objective); the mean and the re-taken target landed
    # together, so a fixed axis under the mean never ran on the bench. The term needs to be about
    # an absolute up/down axis injected into the world.
    #
    # ROLL TIGHT, PITCH LOOSE, because that is what a bicycle IS. These bike rides can be
    # expected to have some pitch perhaps, and very little mean roll: more pitch slack, little roll
    # slack. One isotropic chord -- which is all
    # loss_gravity_cam has -- cannot express that: it spends the roll budget on pitch and the
    # pitch budget on roll. So the target axis is resolved into the MOUNT's two axes and each gets
    # its own dead zone.
    #
    # IN THE RIG FRAME, WHICH IS WHY rig_yaw IS NEEDED. Resolved per SLICE the two would be
    # confused: a roll of the mount reads as roll to the slice looking forward and as PITCH to the
    # slice looking sideways. That is the known worry -- an unknown axis alignment confuses
    # roll with pitch -- except that here the alignment is NOT unknown. slice_yaw is how the
    # slice was cut, so undoing it recovers the mount's roll and pitch exactly, at any yaw, and
    # the ride's HEADING never enters the term at all. A curve cannot smear one axis into the
    # other, which was the thing to be afraid of.
    #
    # AN ANGLE, NOT ITS SINE. The other terms penalise a chord or a sine, and |sin| cannot tell
    # level from INVERTED -- a camera 180 deg out scores zero, which upright_penalty says in its
    # own docstring and which is why "spans going upside down" survived every check. atan2 against
    # the declared down's own +y component makes a flip the most expensive thing in the term
    # instead of the cheapest, at no extra knob.
    _axis = None
    if (axis_roll_weight > 0 or axis_pitch_weight > 0):
        if imu_down is None or rig_yaw is None:
            print(' >> injected axis ASKED but there is no declared down or no slice yaw to undo: '
                  'needs --gauge_up_from_mount (or measured IMU) on a sliced panorama run')
        else:
            _k = [i for i in range(min(len(imu_down), len(rig_yaw)))
                  if imu_down[i] is not None and rig_yaw[i] is not None]
            if len(_k) >= 3:
                _th = torch.as_tensor([float(rig_yaw[i]) for i in _k], device=device, dtype=torch.float64)
                _axis = (torch.as_tensor(_k, device=device),
                         torch.as_tensor([list(imu_down[i]) for i in _k], device=device, dtype=torch.float64),
                         torch.cos(_th), torch.sin(_th))
                def _show(v):
                    if np.isscalar(v) or isinstance(v, (int, float)):
                        return f'{float(v):g}'
                    a = np.asarray(v, dtype=float)
                    return f'{a.min():g}-{a.max():g} (per view, median {np.median(a):g})'
                print(f' >> INJECTED AXIS: {len(_k)}/{len(imgs)} view(s) held to one fixed down, '
                      f'roll w={axis_roll_weight} slack={_show(axis_roll_slack_deg)} deg, '
                      f'pitch w={axis_pitch_weight} slack={_show(axis_pitch_slack_deg)} deg')
            else:
                print(f' >> injected axis ASKED but only {len(_k)} view(s) carry both a declared '
                      f'down and a slice yaw (needs 3): the term is OFF')
    # PER-VIEW SLACK, not one number (hillview). A bike leans into a turn and barely
    # at all on a straight, so one dead zone is wrong in both places: tight enough for the straight
    # it fights the real roll of a corner, loose enough for the corner it stops holding the
    # straight. `axis_roll_slack_deg` may therefore be a sequence, one entry per HELD VIEW in the
    # order of `_axis`'s index, and reconstruct.py builds it from the GPS track's own curvature.
    # Read on frame 29 of a ride along a church lane: the default roll slack in such turns should be
    # expanded to at least 30 degrees.
    def _slack(v, name):
        if np.isscalar(v) or isinstance(v, (int, float)):
            return float(np.radians(float(v)))
        a = np.radians(np.asarray(v, dtype=np.float64))
        if not _axis:
            return float(a.mean()) if a.size else 0.0
        held = _axis[0]
        n_held, n_all = len(held), len(imgs)
        if a.size == n_all:                 # one per IMAGE: the caller cannot know which are held,
            a = a[held.detach().cpu().numpy()]   # so it sends them all and the solver picks
        elif a.size != n_held:
            raise ValueError(f"{name}: {a.size} slacks, expected {n_held} held or {n_all} images")
        return torch.as_tensor(a, device=device)

    _axis_roll_slack = _slack(axis_roll_slack_deg, "axis_roll_slack_deg")
    _axis_pitch_slack = _slack(axis_pitch_slack_deg, "axis_pitch_slack_deg")
    # THE TARGET IS THE INJECTED AXIS, FIXED (hillview; see the section head and
    # InjectedAxis). The first evaluation is the start pose, before any optimiser step.
    _axis_fixed = InjectedAxis(*_axis[1:]) if _axis is not None else None

    def axis_components(cam2w):
        """The mount's (roll, pitch) off the injected axis, radians, per held view.
        Shared with the self-test so the thing that is checked is the thing that is optimised."""
        return _axis_fixed(cam2w[_axis[0], :3, :3])

    def loss_axis(cam2w):
        """PER HELD VIEW, not summed over them.

        THE SCALING IS WRONG BY CONSTRUCTION, which is the whole of the case for this change.
        .sum() made the weight mean "per view", so the same declared conviction pulled 8x
        harder on a 320-view rigged span than on a 40-view one, for no reason anyone intended,
        and at the documented default of 1.0 the term sits two orders of magnitude above the
        correspondence loss it is meant to accompany: 320 views only 20 deg off the axis sum
        to 31.6 against a stage-1 loss near 1.15, i.e. 27x it. A mean puts the same case at
        8.6 % of it. That is arithmetic and needs no experiment.

        WHAT THE SUM DOES TO A PLACED START, measured (coarse only,
        300 iterations, GPS start, one cache, A axis auto vs B axis 0):

            A  axis auto   gps med 12.4 m   28 36.6, 29 43.5 m   29->30 92.4 m
            B  axis 0      gps med  1.97 m  28  6.2, 29 10.3 m   29->30 48.3 m (GPS 38.3)

        BOTH swing every photograph 50-73 deg in the first iterations and both recover the
        rotations by iteration 80-150, so the swing itself is real and survivable -- it is the
        chain plus Adam's first steps, not this term. What separates them is what happens
        AFTER: B's shape relaxes back onto the GPS start (median 0.3 m, 28/29 at 0.9/1.5 m by
        iteration 299) and A's stays bent (median 3.8 m, 28/29 at 11-13 m).

        So the summed term does not throw the cameras -- it LOCKS IN the bend the swing puts
        in them. The axis target is computed once at step 0 and never again; after the swing
        it is stale, and a term 27x the correspondence loss holds the solve against a target
        that no longer describes it. (An earlier reading, that the term was throwing
        the cameras, was retracted on a 12-iteration test and is now restored in this form:
        twelve iterations ended before the divergence began.)

        THE STALENESS IS ANSWERED, AND IT WAS NOT THE SCALING'S PROBLEM TO SOLVE. Arms C and
        D settle it (coarse, 300 its, GPS start, with an lr warm-up of 30):

            A  axis auto          gps med 12.40 | 28 36.6  29 43.5 | 29|30 92.4
            B  axis 0              1.97 | 28  6.2  29 10.3 | 29|30 48.3
            C  axis 0    + wu30    1.80 | 28  3.2  29  6.9 | 29|30 44.3
            D  axis auto + wu30    1.81 | 28  4.5  29  8.2 | 29|30 45.8

        The warm-up removes the first-iteration swing outright -- mean camera move at iteration
        1 goes 13.3 units (B) to 0.45 (C) and 0.40 (D). And with no swing, the SUMMED term costs
        almost nothing: D is within 1.3 m of C on photograph 28, against A's 36.6. So the term's
        damage really did run through the swing, through the frozen target going stale, and a
        warm-up prevents it at the source without touching this scaling at all.

        SO THE CASE FOR THIS CHANGE IS NOW NARROWER THAN WHEN IT WAS WRITTEN, and honestly it is
        back to exactly what the first paragraph says and nothing more. The arithmetic is still
        wrong and this still fixes it. But a mean will not be rescuing a solve from anything:
        with the swing tamed there is nothing to rescue it from, and dividing the force by
        n_held moves D toward C -- which is the term switched OFF. That cannot be sold as the
        term working.

        WHAT WOULD ACTUALLY JUSTIFY THE TERM is the thing it exists for and that 28/29 does not
        measure: ROLL. Every other attitude prior here is relative, so a solve that tips as a
        whole pays nothing; this one is absolute. C beating D by 1.3 m on a POSITION says the
        term costs a little, not that it earns nothing. The decision needs roll metrics on a
        warmed-up bench arm with the term on, against the matching axis-0 arm. Until that
        exists, land this for the scaling and do not claim more.

        AND A MEAN MUST NOT QUIETLY BECOME A NO-OP. 1.0 is now about 1/n_held of the force it
        was, so no earlier run is comparable on that parameter and the hand-pinned recipes
        (knee5, knee10) were tuned against the sum -- they want re-reading, not re-scaling.
        Whoever tests this must check the upright actually IMPROVES against a matching axis-0
        arm, not merely that the loss stopped exploding: dividing by 320 cannot fail to do the
        latter.
        """
        roll, pitch = axis_components(cam2w)
        out = 0
        if axis_roll_weight > 0:
            out = out + axis_roll_weight * ((roll.abs() - _axis_roll_slack).clamp(min=0.0) ** 2).mean()
        if axis_pitch_weight > 0:
            out = out + axis_pitch_weight * ((pitch.abs() - _axis_pitch_slack).clamp(min=0.0) ** 2).mean()
        return out

    def _model_down(cam2w):
        R_ = cam2w[_imu_idx, :3, :3].detach().double()
        g_ = (R_ @ _imu_d[:, :, None])[:, :, 0].mean(0)
        return g_ / g_.norm().clamp(min=1e-12)

    def _gauge_positions(gi, c, enu, w, zok):
        # THE COURSE TERM'S OWN FRAME (hillview): the similarity from
        # camera CENTRES to GPS alone, never the cameras' orientations. With gauge_gravity the shared
        # gauge takes its up from the cameras' mean measured down, so a term that steers steps through
        # it turns the rotations, the up turns with them and the level it enforces turns: on one phone
        # walk every full-span course solve tipped (down spread 25.9 deg over 61 frames). Read through this
        # frame the same term at the same weight settles (8.9 deg, back-steps 9 -> 0). Cached beside
        # the shared gauge under its own key, so every refit point that clears one clears both.
        key = ('positions', gi)
        hit = _gauge_cache.get(key)
        if hit is None:
            with torch.no_grad():
                hit = (gps_gauge(c.detach(), enu, w, zok, g=None) if FIXED_GAUGE is None
                       else fixed_gauge(c.detach(), enu, zok, FIXED_GAUGE))           # held targets: the source's, given
            _gauge_cache[key] = hit
        return hit

    _alt_fits = {}           # group -> how many height-step fits so far (logged on the first and every 100th)
    _alt_last = {}           # group -> its last height-step down, where the next refit starts

    def _gauge(gi, c, enu, w, zok, cam2w=None):
        hit = _gauge_cache.get(gi)
        if hit is None and FIXED_GAUGE is not None:
            # held targets: the frame is a judged source's, its alignment GIVEN -- never refit, never tilted by the steps
            with torch.no_grad():
                hit = fixed_gauge(c.detach(), enu, zok, FIXED_GAUGE)
            _gauge_cache[gi] = hit
        if hit is None:
            with torch.no_grad():
                g_ = _model_down(cam2w) if (_imu_idx is not None and cam2w is not None) else None
                if gauge_tilt_from == 'height_steps':
                    if g_ is None:
                        raise RuntimeError('gauge_tilt_from height_steps: the gauge was asked without the poses '
                                           'whose mounts it corrects')
                    li, lj, _dz = _alt_of[gi]
                    g_, info = height_step_down(c.detach(), enu, w, li, lj, g_, float(alt_step_huber_m),
                                              float(gauge_tilt_mount_sd_deg), float(gauge_tilt_step_floor_m),
                                              g_start=_alt_last.get(gi))
                    _alt_last[gi] = g_
                    n_ = _alt_fits.get(gi, 0)
                    _alt_fits[gi] = n_ + 1
                    if n_ % 100 == 0:
                        print(f' >> height-step down, group {gi}, fit {n_} ({info["iters"]} it): {info["off_deg"]:.2f} deg off the '
                              f'mounts\' mean, the weaker tilt direction seen at {info["seen"]:.3g} of the stronger, '
                              f'steps off the terrain by median {info["resid_med_m"]:.3f} m (sigma {info["sigma_z_m"]:.3f} m), '
                              f'{info["beyond_knee"]} beyond the knee'
                              + (f' -- KEPT THE MOUNTS\': {info["why"]}' if info['why'] else ''))
                hit = gps_gauge(c.detach(), enu, w, zok, g=g_)
            _gauge_cache[gi] = hit
        return hit

    # HOW MUCH A METRE OF ALTITUDE IS WORTH (hillview --gps_z_gain). Both GPS losses reduce the
    # residual to ONE 3-D distance and Huber it, so z and xy have always counted the same and
    # --gps_frac raises both together. That is why pulling harder eventually loses the vertical:
    # measured: gps_frac 0.3 and 0.5 let camera z run to a 75 m span on a walk whose
    # ground varies 18 m, while the horizontal residual kept improving. Scaling z BEFORE the norm
    # is what lets the Huber see it, and 1.0 is exactly the old arithmetic.
    _zg = float(gps_z_gain or 1.0)

    def _z_gained(diff):
        if _zg == 1.0:
            return diff
        return torch.cat([diff[:, :2], diff[:, 2:3] * _zg], 1)

    def loss_gps(cam2w):
        total = 0
        for gi, (sel, enu, w, zok) in enumerate(_gps_groups):
            c = cam2w[sel, :3, 3]
            # the gauge, and the target the residual is measured against: gps_gauge (module
            # level, so recon_gps_check.py can drive it without a solve) leaves the height of a
            # view whose altitude nobody recorded unconstrained. Fitted under no_grad inside
            # _gauge, and shared with loss_grade / loss_rig_upright.
            s, R, t, enu_fit = _gauge(gi, c, enu, w, zok, cam2w)
            rec = s.to(c.dtype) * (c @ R.to(c.dtype).T) + t.to(c.dtype)
            # the height of a view with no recorded altitude contributes exactly zero, rather than
            # the small residue the two-pass fit would otherwise leave
            _diff = rec - enu_fit
            if not bool(zok.all()):
                _diff = torch.cat([_diff[:, :2],
                                   torch.where(zok, _diff[:, 2], torch.zeros_like(_diff[:, 2]))[:, None]], 1)
            _diff = _z_gained(_diff)
            dist = _diff.norm(dim=1)
            # Huber, in metres: one fix that wandered off across the street does not get to
            # bend the walk around itself (the GPS has been seen doing exactly that)
            hub = torch.where(dist <= gps_huber_m, 0.5 * dist ** 2,
                              gps_huber_m * (dist - 0.5 * gps_huber_m))
            # How hard to pull is not a constant. A solve sitting a few metres off its track is
            # inside the receiver's own noise, and there the pixels know better; a solve that has
            # drifted hundreds of metres is simply wrong, and the track is the only thing that can
            # say so. Each group is therefore scaled by its OWN median disagreement, measured every
            # iteration and detached — weak at metres, decisive at hundreds, and fading again as
            # the solve comes home. The Huber above still keeps ONE wild fix from bending the walk;
            # this is about the whole group being lost, which is a different failure.
            # Locality is what separates the two failures, and one number per group cannot.
            # ONE fix that wandered has neighbours that still agree, so its local median stays
            # small and the Huber above keeps it quiet. A stretch that has DRIFTED disagrees
            # together with its neighbours, and there the track is the only thing that can say
            # so. The scale is therefore per-frame, from a window of temporal neighbours (sel
            # is ascending index order, which is capture order): the model wins at metres, the
            # track wins at hundreds, frame by frame rather than walk by walk.
            with torch.no_grad():
                k = min(int(gps_local_win), len(dist))
                if k % 2 == 0:
                    k -= 1
                if k >= 3:
                    d2 = torch.nn.functional.pad(dist[None, None], (k // 2, k // 2),
                                                 mode='replicate')
                    local = d2.unfold(-1, k, 1).median(dim=-1).values.reshape(-1)[:len(dist)]
                else:
                    local = dist.median().expand_as(dist)
                # The floor and the ceiling on that swing were 0.2 and 20.0 written here, now exposed
                # so they can be moved and measured rather than guessed -- the same
                # reason pair_min_matches stopped being a literal. The floor is what a frame sitting
                # inside the receiver's own noise still contributes (0 would mute the track entirely
                # where the solve already agrees); the ceiling is how decisive a badly drifted
                # stretch is allowed to become (too high and one lost group dominates the objective).
                pull = (local / gps_huber_m).clamp(min=float(gps_pull_min), max=float(gps_pull_max))
            total = total + (pull * w * hub).sum()
        return total / max(len(_gps_groups), 1)

    # WHY THERE IS A SECOND ONE. loss_gps above is in absolute METRES, and it multiplies a
    # Huber by `pull`, which is proportional to the same distance the Huber exists to saturate
    # -- so past the knee the product is quadratic and the robustness is cancelled. Measured
    # on a church ride: the whole objective converges to 0.837 with the term off and
    # 25.2 with it on, and the term evaluated on the OFF run's own final poses is 117.05. The
    # pixel term was 0.7 % of what was being minimised, so the optimiser rationally bought a
    # 3.5x GPS reduction for a 15x pixel degradation. No constant gps_weight can be right on
    # both a 4 m hillview walk and a 600 m Panoramax span.
    #
    # This version fixes both halves. The Huber keeps its linear tail because `pull` is now a
    # bounded locality GATE -- the one thing a windowed median is genuinely for, telling a lone
    # wild fix from a stretch that drifted together -- and the whole term is rescaled ONCE, at
    # the first iteration, so that `gps_frac` is the FRACTION of the pixel objective it starts
    # at. That makes the knob mean the same thing on any walk of any length.
    def loss_gps_normalised(cam2w):
        total = 0
        for gi, (sel, enu, w, zok) in enumerate(_gps_groups):
            c = cam2w[sel, :3, 3]
            s_, R_, t_, enu_fit = _gauge(gi, c, enu, w, zok, cam2w)
            rec = s_.to(c.dtype) * (c @ R_.to(c.dtype).T) + t_.to(c.dtype)
            _diff = rec - enu_fit
            if not bool(zok.all()):
                _diff = torch.cat([_diff[:, :2],
                                   torch.where(zok, _diff[:, 2],
                                               torch.zeros_like(_diff[:, 2]))[:, None]], 1)
            _diff = _z_gained(_diff)
            dist = _diff.norm(dim=1)
            hub = torch.where(dist <= gps_huber_m, 0.5 * dist ** 2,
                              gps_huber_m * (dist - 0.5 * gps_huber_m))
            with torch.no_grad():
                k = min(int(gps_local_win), len(dist))
                if k % 2 == 0:
                    k -= 1
                if k >= 3:
                    d2 = torch.nn.functional.pad(dist[None, None], (k // 2, k // 2),
                                                 mode='replicate')
                    local = d2.unfold(-1, k, 1).median(dim=-1).values.reshape(-1)[:len(dist)]
                else:
                    local = dist.median().expand_as(dist)
                # BOUNDED. gate_max is ~2, not 20: this says "believe a drifted stretch twice as
                # much as a lone fix", and nothing about how many metres it is off.
                gate = (local / gps_huber_m).clamp(min=float(gps_pull_min),
                                                   max=float(gps_gate_max))
            total = total + (gate * w * hub).sum()
        return total / max(len(_gps_groups), 1)

    # THE PACE (hillview): how much time must it have taken a walking person to get from one pose to
    # the next? Consecutive exposures dt seconds apart
    # cannot be more than pace_max_mps * dt apart in METRES, read through metres_per_unit -- the
    # ratio of the GPS track's size to the camera walk's own, differentiable, so collapsing the walk
    # buys nothing (the detached gauge scale of the first version let a phone walk shrink to a point).
    # The same walk with the gyro held every rotation and still had 39 steps faster than 3 m/s on a 1.4 m/s
    # walk. Only the excess costs, under a 1 m Huber knee, summed; a pair outside every GPS group is
    # not constrained.
    _pace = None
    if pace_steps and pace_weight > 0 and _gps_groups:
        _grp_of = {}
        for gi, (sel, *_r) in enumerate(_gps_groups):
            for v in sel.tolist():
                _grp_of[int(v)] = gi
        _pp = [(int(i), int(j), float(t), _grp_of[int(i)]) for i, j, t in pace_steps
               if int(i) in _grp_of and _grp_of.get(int(j)) == _grp_of[int(i)] and t > 0]
        if _pp:
            _pace = (torch.as_tensor([e[0] for e in _pp], device=device), torch.as_tensor([e[1] for e in _pp], device=device),
                     torch.as_tensor([pace_max_mps * e[2] for e in _pp], device=device, dtype=torch.float64),
                     torch.as_tensor([e[3] for e in _pp], device=device))
            print(f' >> pace: {len(_pp)} consecutive step(s) held under {pace_max_mps} m/s, weight {pace_weight}')

    def loss_pace(cam2w):
        pi, pj, cap, grp = _pace
        s_g = [metres_per_unit(cam2w[sel, :3, 3].to(torch.float64), enu.to(torch.float64))
               for sel, enu, w, zok in _gps_groups]
        s_ = torch.stack(s_g)[grp]
        step = s_ * (cam2w[pj, :3, 3] - cam2w[pi, :3, 3]).to(torch.float64).norm(dim=1)
        ex = (step - cap).clamp(min=0.0)
        return torch.where(ex <= 1.0, 0.5 * ex ** 2, ex - 0.5).sum()

    # THE WALKER'S OWN SPEED (hillview): a solve must follow a walking pace. The pace cap only forbade
    # sprints: that solve still stalled at 0.1 m/s where the
    # phone walked 1.2 and ran 2.4 through a turnaround where it stood at 0.3. The phone measures its
    # speed by Doppler, to ~0.25 m/s, at every exposure, so each consecutive step has an expected
    # length L = mean(v_i, v_j) * dt and a tolerance mean(accuracy) * dt.
    #
    # THE SCALE COMES FROM THE DOPPLER TOO, not from the GPS: metres per unit = sum(L) / sum(|step|)
    # over the steps the term holds. So the term constrains only HOW THE WALK IS SPREAD ALONG ITS
    # LENGTH -- which steps are long and which short -- and leaves its overall size to the GPS gauge.
    # It is a ratio, so collapsing the walk buys nothing (the detached-gauge escape that shrank a
    # phone walk to a point), and GPS jitter does not enter it at all. Two-sided beyond the
    # tolerance, 1 m Huber knee, summed.
    _speed = None
    if speed_steps and speed_weight > 0:
        _speed = (torch.as_tensor([int(e[0]) for e in speed_steps], device=device),
                  torch.as_tensor([int(e[1]) for e in speed_steps], device=device),
                  torch.as_tensor([float(e[2]) for e in speed_steps], device=device, dtype=torch.float64),
                  torch.as_tensor([float(e[3]) for e in speed_steps], device=device, dtype=torch.float64))
        print(f' >> speed: {len(speed_steps)} consecutive step(s) held to the phone\'s Doppler speed, weight '
              f'{speed_weight}; scale from the Doppler itself (sum of expected / sum of solved)')

    def loss_speed(cam2w):
        si, sj, L, tol = _speed
        step = doppler_steps_m((cam2w[sj, :3, 3] - cam2w[si, :3, 3]).to(torch.float64).norm(dim=1), L)
        ex = ((step - L).abs() - tol).clamp(min=0.0)
        return torch.where(ex <= 1.0, 0.5 * ex ** 2, ex - 0.5).sum()

    # THE STEP'S DIRECTION FROM THE PHONE'S DOPPLER COURSE (--course_weight, hillview).
    # loss_speed holds only a step's LENGTH, so a camera that steps 3 m BACK costs it exactly what 3 m
    # forward does: on one phone walk with the speed term on, 12-13 steps per solve still went 1-4.5 m
    # backwards and 14-16 against the course (recon_imu pace.against_course): the cameras jumped back for
    # one frame instead of following the pattern of walking down the street.
    # The phone measured where it was going: fix.course_deg, with its own accuracy. Through the GPS
    # gauge (the rotation into east/north/up that loss_gps fits, shared, under no_grad) each WHOLE
    # step d (3-D: see below) is split along the level course u: what goes BACKWARDS (-d.u) costs in
    # full, what goes off it -- sideways or up/down -- beyond |d| sin(allowance) costs the excess (the allowance is the course accuracy, plus half
    # the turn when the two exposures' courses differ: reconstruct.py builds it); metres, 1 m Huber knee, summed. Speed is
    # left to loss_speed. Needs a GPS group; a step whose views sit in none is not held.
    _course = None
    if course_steps and course_weight > 0:
        if not _gps_groups:
            raise ValueError('course_weight: the Doppler course is east/north, and without a GPS group there is no '
                             'gauge to say which way that is in the solve')
        _grp_of = {}
        for gi, (sel, _e, _w, _z) in enumerate(_gps_groups):
            for v in sel.tolist():
                _grp_of[int(v)] = gi
        _cs = [e for e in course_steps if _grp_of.get(int(e[0])) is not None and _grp_of.get(int(e[0])) == _grp_of.get(int(e[1]))]
        # THE UP AXIS MUST BE MEASURED: the term charges up/down through the gauge,
        # and a view with no recorded altitude leaves the gauge's vertical unmeasured there -- charging
        # it would push along an arbitrary direction, and reading only east/north reopens the escape
        # the 3-D step closed. So a course step on a view without an altitude is refused, by name.
        _zok_of = {}
        for gi, (sel, _e, _w, zok) in enumerate(_gps_groups):
            for v, z in zip(sel.tolist(), zok.tolist()):
                _zok_of[int(v)] = bool(z)
        _noalt = [(int(e[0]), int(e[1])) for e in _cs if not (_zok_of.get(int(e[0])) and _zok_of.get(int(e[1])))]
        if _noalt:
            raise ValueError(f'course_weight: {len(_noalt)} course step(s) touch views with no recorded altitude '
                             f'(e.g. {_noalt[:3]}), where the GPS gauge has no measured up -- the term charges '
                             f'up/down and cannot hold those steps; leave course_weight off for this source')
        if _cs:
            _course = (torch.as_tensor([int(e[0]) for e in _cs], device=device),
                       torch.as_tensor([int(e[1]) for e in _cs], device=device),
                       torch.as_tensor([[float(e[2]), float(e[3])] for e in _cs], device=device, dtype=torch.float64),
                       torch.as_tensor([math.sin(math.radians(min(float(e[4]), 89.0))) for e in _cs], device=device, dtype=torch.float64),
                       torch.as_tensor([_grp_of[int(e[0])] for e in _cs], device=device))
        print(f' >> course: {len(_cs)} of {len(course_steps)} consecutive step(s) held to the phone\'s Doppler course, '
              f'weight {course_weight} (backwards in full, sideways beyond every step\'s allowance: course accuracy, plus half the turn across one; metres via the GPS gauge)')

    def loss_course(cam2w):
        si, sj, u, sin_acc, grp = _course
        total = 0
        for gi, (sel, enu, w, zok) in enumerate(_gps_groups):
            m = grp == gi
            if not bool(m.any()):
                continue
            c = cam2w[sel, :3, 3]
            s_, R_, _t, _ = _gauge_positions(gi, c, enu, w, zok)
            d = (cam2w[sj[m], :3, 3] - cam2w[si[m], :3, 3]).to(torch.float64)
            # THE WHOLE STEP, NOT ITS HORIZONTAL SHADOW: the first version
            # read only the east/north part, and with loss_speed holding the 3D length the solve
            # escaped by turning the walk on its side -- two solves of one phone walk put the track
            # plane 82-89 deg from level (z span 207-497 m) and every step's horizontal part, and so
            # this cost, near zero. The course is level, so a step's up/down is off-course like its
            # sideways part, within the same allowance (sin 6-10 deg covers a 10-17 % grade).
            d3 = s_.to(torch.float64) * (d @ R_.to(torch.float64).T)
            u3 = torch.cat([u[m], torch.zeros_like(u[m][:, :1])], 1)
            along = (d3 * u3).sum(dim=1)
            off = (d3 - along[:, None] * u3).norm(dim=1)
            ex = (-along).clamp(min=0.0) + (off - d3.norm(dim=1) * sin_acc[m]).clamp(min=0.0)
            total = total + torch.where(ex <= 1.0, 0.5 * ex ** 2, ex - 0.5).sum()
        return total

    # WEAK PHOTOGRAPHS, HELD (hillview). A photograph whose
    # correspondences in the solver's surviving pairs number fewer than one pair's worth is WEAK: no picture
    # term holds its camera. On one phone walk view 14 (89 % close bushes, every pair peaking under
    # the matching floor) sat on its own fix, a 4.6 m kink where the GPS says 0.85 m. Dropping such frames
    # is no answer: the whole walk is broken exactly because the weak frames fail to hold it
    # together. These two terms HOLD them. The caller decides the set once, before the optimiser
    # (weak_views: every view of a weak photograph), and hands the consecutive-photograph links the step
    # priors hold (step_links); both terms read that one set.
    #   gps_rel   each held step against the GPS's own step, in the GPS term's gauge (gps_rel_residual_m),
    #             Huber at gps_rel_huber_m; scope 'weak' holds the steps with a weak end, 'all' every step.
    #             Both ends move. Normalised like --gps_frac: it starts at gps_rel_frac of the pixel objective.
    #   weak_mid  a weak camera with a neighbour on each side pulled to their midpoint, the neighbours
    #             detached (weak_mid_residual_m), Huber at weak_mid_huber_m. A run of weak photographs
    #             chains, each toward its immediate neighbours; one at the end of its chain is skipped.
    #             Normalised against ONE KNEE PER HELD CAMERA, not its start: at the pile, or at a placed
    #             start, every camera sits about on its neighbours' midpoint, and a start near zero would
    #             make the factor arbitrary. weak_mid_frac is the term's share of the pixel objective at
    #             the start if every held camera were one knee off.
    # Steps and triples are held inside one GPS group (a piece), where the gauge is one similarity.
    _rel = None
    _mid = None
    _n_mid = 0
    if gps_rel_frac or weak_mid_frac:
        if pose_fixed:
            raise ValueError('gps_rel_frac / weak_mid_frac move cameras, and pose_fixed holds every camera')
        if weak_views is None or step_links is None:
            raise ValueError('gps_rel_frac / weak_mid_frac need weak_views and step_links from the caller')
        if callable(weak_views):
            weak_views = weak_views()
        if not _gps_groups:
            raise ValueError('gps_rel_frac / weak_mid_frac read the GPS gauge (metres, and for gps_rel the GPS '
                             'steps), and there is no GPS group (3 positioned views)')
        _weak = {int(v) for v in weak_views}
        _row_of = {}
        for gi, (sel, _e, _w, _z) in enumerate(_gps_groups):
            for r_, v in enumerate(sel.tolist()):
                _row_of[int(v)] = (gi, r_)
        _plan = weak_hold_plan(_weak, step_links, _row_of, gps_rel_scope)
        print(f' >> weak photographs: {len(_weak)} weak view(s) from the caller'
              + (f' ({", ".join(str(v) for v in sorted(_weak)[:20])}{" ..." if len(_weak) > 20 else ""})' if _weak else '')
              + f'; {_plan["n_links"]} consecutive-photograph link(s)')
        if gps_rel_frac:
            _rel = [(gi, torch.as_tensor([a_ for a_, _ in rr], device=device),
                     torch.as_tensor([b_ for _, b_ in rr], device=device))
                    for gi, rr in sorted(_plan['rel'].items())] or None
            print(f' >> gps_rel: {sum(len(rr) for rr in _plan["rel"].values())} of {_plan["n_links"]} consecutive '
                  f'step(s) held to the GPS\'s own steps (scope {gps_rel_scope}), huber {gps_rel_huber_m} m, starting at '
                  f'{gps_rel_frac:.0%} of the pixel objective'
                  + (f'; {_plan["rel_out"]} step(s) not inside one GPS group, not held' if _plan['rel_out'] else '')
                  + ('' if _rel is not None else
                     ' -- HOLDS NOTHING: ' + ('no weak photograph' if gps_rel_scope == 'weak' and not _weak else
                                              'no link to hold (a panorama slice stands for no photograph)')))
        if weak_mid_frac:
            _mid = [(gi,) + tuple(torch.as_tensor([t_[k_] for t_ in rr], device=device) for k_ in range(3))
                    for gi, rr in sorted(_plan['mid'].items())] or None
            _n_mid = sum(len(rr) for rr in _plan['mid'].values())
            print(f' >> weak_mid: {_n_mid} weak camera(s) pulled to the midpoint of their neighbours (detached: the '
                  f'neighbours feel nothing), huber {weak_mid_huber_m} m, {weak_mid_frac:.0%} of the pixel objective '
                  f'at one knee per camera'
                  + (f'; END of a chain, skipped: view(s) {", ".join(str(v) for v in _plan["ends"])}'
                     if _plan['ends'] else '')
                  + (f'; {_plan["off"]} weak view(s) on no link (carried by their rig\'s stand-in, or standing for '
                     f'nothing)' if _plan['off'] else '')
                  + (f'; {_plan["mid_out"]} not inside one GPS group with both neighbours, not held'
                     if _plan['mid_out'] else '')
                  + ('' if _mid is not None else ' -- HOLDS NOTHING'))

    def loss_gps_rel(cam2w):
        total = 0
        for gi, li, lj in _rel:
            sel, enu, w, zok = _gps_groups[gi]
            c = cam2w[sel, :3, 3]
            s_, R_, _t, enu_fit = _gauge(gi, c, enu, w, zok, cam2w)
            total = total + huber_m(gps_rel_residual_m(c, li, lj, enu_fit, zok, s_, R_, _zg), gps_rel_huber_m).sum()
        return total

    def loss_weak_mid(cam2w):
        total = 0
        for gi, vi, pi, ni in _mid:
            sel, enu, w, zok = _gps_groups[gi]
            c = cam2w[sel, :3, 3]
            s_, _R, _t, _f = _gauge(gi, c, enu, w, zok, cam2w)
            total = total + huber_m(weak_mid_residual_m(c, vi, pi, ni, s_), weak_mid_huber_m).sum()
        return total

    def loss_alt_step(cam2w):
        total = 0
        for gi, (li, lj, dz) in _alt:
            sel, enu, w, zok = _gps_groups[gi]
            c = cam2w[sel, :3, 3]
            s_, R_, _t, _f = _gauge(gi, c, enu, w, zok, cam2w)
            total = total + huber_m(alt_step_residual_m(c, li, lj, dz, s_, R_).abs(), alt_step_huber_m).sum()
        return total

    _rel_k = [None]          # the holds' normalisers, fixed where the GPS term's is and then held
    _mid_k = [None]
    _alt_k = [None]

    def _holds_pending():
        return ((_rel is not None and _rel_k[0] is None) or (_mid is not None and _mid_k[0] is None)
                or (_alt is not None and _alt_k[0] is None))

    def _normalise_holds(pix0, c2w0, where):
        if _rel is not None and _rel_k[0] is None:
            raw0 = float(loss_gps_rel(c2w0).detach())
            if not raw0 > 1e-12:
                raise ValueError(f'gps_rel: the held steps already equal the GPS\'s{where} (raw {raw0:.3g}), so there '
                                 f'is no start to take the fraction of')
            _rel_k[0] = float(gps_rel_frac) * pix0 / raw0
            print(f' >> gps_rel normalised{where}: x{_rel_k[0]:.4g} so it starts at {gps_rel_frac:.0%} of the pixel '
                  f'objective ({pix0:.4g}); raw {raw0:.4g} in metre-units')
        if _mid is not None and _mid_k[0] is None:
            _mid_k[0] = float(weak_mid_frac) * pix0 / (_n_mid * 0.5 * float(weak_mid_huber_m) ** 2)
            print(f' >> weak_mid normalised{where}: x{_mid_k[0]:.4g}, so {_n_mid} camera(s) one knee off would be '
                  f'{weak_mid_frac:.0%} of the pixel objective ({pix0:.4g})')
        if _alt is not None and _alt_k[0] is None:
            _alt_k[0] = float(alt_step_frac) * pix0 / (_n_alt * 0.5 * float(alt_step_huber_m) ** 2)
            print(f' >> alt_step normalised{where}: x{_alt_k[0]:.4g}, so {_n_alt} step(s) one knee off the terrain '
                  f'would be {alt_step_frac:.0%} of the pixel objective ({pix0:.4g})')

    # THE GRADE PRIOR (--grade_weight). Shares loss_gps's gauge and nothing else: it reads only
    # the VERTICAL, which is the one direction the GPS term cannot speak for on a Panoramax span
    # (every frame is altitude: null, so gps_gauge zeroes the vertical residual by design and the
    # heights float free -- measured: the DEM residual got WORSE, 17.0 -> 29.0 m sd,
    # across the A/B where the horizontal one improved). A ratio, so it is dimensionless and
    # cannot be mis-scaled the way loss_gps is; see grade_penalty for where 0.25 comes from.
    def loss_grade(cam2w):
        total = 0
        for gi, (sel, enu, w, zok) in enumerate(_gps_groups):
            c = cam2w[sel, :3, 3]
            s_, R_, t_, _ = _gauge(gi, c, enu, w, zok, cam2w)
            rec = s_.to(c.dtype) * (c @ R_.to(c.dtype).T) + t_.to(c.dtype)
            total = total + grade_penalty(rec, grade_slack, grade_min_step_frac)
        return total / max(len(_gps_groups), 1)

    # THE MOUNT IS UPRIGHT, stated of the capture rather than estimated from the solve. Shares
    # loss_gps's gauge only to learn which way is up; it touches ROTATIONS, where loss_gps
    # touches positions, so the two do not compete for the same parameters.
    def loss_rig_upright(cam2w):
        total = 0
        for gi, (sel, enu, w, zok) in enumerate(_gps_groups):
            c = cam2w[sel, :3, 3]
            _, R_, _, _ = _gauge(gi, c, enu, w, zok, cam2w)
            right = cam2w[sel, :3, 0]
            right = right / right.norm(dim=1, keepdim=True).clamp(min=1e-9)
            pen = rig_upright_penalty(right @ R_.to(right.dtype).T, _rig_up_slack)
            if pen is not None:
                total = total + pen
        return total / max(len(_gps_groups), 1)

    # PAIRWISE STEP PRIOR (hillview). Each pair's forward pass believes in a relative
    # camera translation: align view j's own pointmap onto its copy in view i's frame and the
    # translation is camera j's centre in camera i's frame, in that pass's units. Those beliefs
    # are mutually consistent to the decimetre (on one town walk: 8->9 0.95 m, 9->10 1.04, 10->11
    # 1.14) and every one of them said frame 10 was AHEAD of 9 while the solve put it 2.4 m
    # behind -- the correspondence loss loses the scale information the pointmaps carry when the
    # matches across two blocks are far-field, so per-frame scale drifts between the blocks and
    # a seam opens. This term puts the beliefs back: for pair (i, j) the solve's relative
    # translation in camera i's frame, R_i^T (c_j - c_i), must match the belief taken into solve
    # units, global_scaling * sizes_i * focal_i / base_focal_i * t_ij -- differentiable in the
    # per-image scales, which is what ties them together along the walk. Relative error per pair
    # under a Huber, so one confused pair cannot bend the rest.
    _pair_steps = None
    if pair_steps is not None and pair_step_w > 0:
        if not pair_steps:
            print(' >> pair step prior requested but NO beliefs arrived (forward cache missing?)')
        _pi, _pj, _pt = [], [], []
        # ONLY PAIRS THE SOLVE KEEPS. reconstruct.py collects a belief for every pair it matched, BEFORE
        # step consensus and the pointmap-consistency test remove pairs from the solve -- and the pairs those remove
        # are exactly the ones whose step is wrong (a repeating wall's window B taken for window A). On
        # one church ride's solve 3524 pairs were removed and their 8400-belief prior still
        # carried them: it ended with 3192 beliefs beyond the knee and broke a solved span. A belief whose pair is not
        # in the solve pulls on cameras the evidence has already disowned; it is dropped here, whatever dropped the pair.
        # (loss3d_slices + dust3r_slices, not imgs_slices: the corres2d loop rebinds that name to tuples)
        _in_solve = loss3d_slices + dust3r_slices
        _solved = {(s_.img1, s_.img2) for s_ in _in_solve} | {(s_.img2, s_.img1) for s_ in _in_solve}
        _n_out = 0
        for (a, b), t in pair_steps.items():
            try:
                ia, ib = (a if isinstance(a, int) else imgs.index(a)), (b if isinstance(b, int) else imgs.index(b))
            except ValueError:
                continue
            if (ia, ib) not in _solved:
                _n_out += 1
                continue
            _pi.append(ia); _pj.append(ib); _pt.append(list(map(float, t)))
        if _n_out:
            print(f' >> pair step prior: {_n_out} belief(s) dropped, their pairs are not in the solve')
        if pair_step_len_only and (pair_step_dir_only or not pair_step_len_sep > 0):
            raise ValueError('pair_step_len_only is its own mode: it needs pair_step_len_sep > 0 (which pairs count as near) '
                             f'and excludes pair_step_dir_only (got dir_only={pair_step_dir_only}, len_sep={pair_step_len_sep})')
        if _pi:
            _pair_steps = (torch.as_tensor(_pi, device=device), torch.as_tensor(_pj, device=device),
                           torch.as_tensor(_pt, dtype=dtype, device=device))
            print(f' >> pair step prior: {len(_pi)} pair belief(s), weight {pair_step_w}, huber {pair_step_huber}' + (f', LENGTH ONLY within {pair_step_len_sep} index steps ({int(((_pair_steps[0] - _pair_steps[1]).abs() <= pair_step_len_sep).sum())} near)' if pair_step_len_only else ', direction only' + (f' + length within {pair_step_len_sep} frames' if pair_step_len_sep > 0 else '') if pair_step_dir_only else ''))

    def loss_pairstep(cam2w):
        pi, pj, T = _pair_steps
        sizes = torch.cat(log_sizes).exp()
        global_scaling = 1 / sizes.min() if _held is None else 1.0         # held targets: as make_K_cam_depth's
        focals = focals_of(log_focals)
        k = global_scaling * sizes[pi] * focals[pi] / base_focals[pi]        # solve units per pass unit, per pair
        if pair_step_depth_scale:
            # hillview: sizes_i is NOT the whole of an image's scale in the solve. Stage 2 optimises
            # the per-anchor depths (core_depth, median 1 as initialised), so an image's entire depth
            # field can inflate or shrink there while sizes_i stays put -- and the first prior runs
            # showed exactly that: the retreat moved from 9->10 to
            # 8->9 and the early block's camera heights kept drifting 3.1 -> 0.4 m, because a belief
            # converted with sizes_i alone is satisfied by drifting the depths instead. The conversion
            # has to carry the depth field's own factor: mul mode depth = z_c * cd, factor median(cd);
            # add mode depth = z_c * (1 + (cd - 1) * base_focal / focal), factor 1 + (median(cd) - 1)
            # * base_focal / focal. Differentiable, so the prior now pulls on the depth scales too.
            meds = torch.stack([(d.exp() if exp_depth else d).median() for d in core_depth])
            s = (1 + (meds - 1) * base_focals / focals) if depth_mode == 'add' else meds
            k = k * s[pi]
        ci, cj = cam2w[pi, :3, 3], cam2w[pj, :3, 3]
        Ri = cam2w[pi, :3, :3]
        rel = torch.einsum('nji,nj->ni', Ri, cj - ci)                         # R_i^T (c_j - c_i)
        want = k.unsqueeze(1) * T
        if pair_step_len_only:
            # hillview: the belief's LENGTH only, from near pairs (measured on a church ride). The direction half
            # compares R_i^T (c_j - c_i) with the belief, so its gradient reaches camera i's ROTATION: at w 10 it
            # overrode a solved junction (xctl-ax0-pipile-leaf27-ps10-a, 27|28 consensus 1.2 -> 13.8 deg). A rotation
            # keeps norms, |R_i^T v| = |v|, so the length half alone asserts step SCALE -- what the correspondence
            # loss is measured to lose along an unrevisited leg -- and cannot turn a camera.
            near = ((pi - pj).abs() <= pair_step_len_sep).to(rel.dtype)
            len_err = (rel.norm(dim=1) - want.norm(dim=1)).abs() / want.norm(dim=1).clamp(min=1e-6)
            len_hub = torch.where(len_err <= pair_step_huber, 0.5 * len_err ** 2, pair_step_huber * (len_err - 0.5 * pair_step_huber))
            return (near * len_hub).sum() / near.sum().clamp(min=1)
        if pair_step_dir_only:
            # hillview: the belief's DIRECTION only. The end-of-solve residuals of one chain
            # solve showed the magnitudes are the unreliable half: pairs eight frames apart
            # (10->18, 12->19, 11->18) ended with the solve's step 4-6x the belief while pointing the
            # same way to 3-13 deg -- the pass compresses distance between far-apart views -- so a
            # magnitude term pulls the walk together where the beliefs are weakest. 1 - cos, under
            # the same Huber knee (0.5 = 60 deg).
            err = 1 - (rel * want).sum(1) / (rel.norm(dim=1) * want.norm(dim=1)).clamp(min=1e-9)
            if pair_step_len_sep > 0:
                # hillview: ...and the LENGTH too, but only from pairs close in the sequence. The
                # direction-only run was monotonic with frames 8, 9 and 10
                # clumped at one spot -- the photographer walked a metre between each -- because the
                # lengths were dropped for every pair, while the residuals had said they are right
                # to ~1.0x within a frame or two and 4-6x short eight frames apart. The rule is the
                # measured one: a belief's length is trusted up to pair_step_len_sep frames apart.
                near = ((pi - pj).abs() <= pair_step_len_sep).to(err.dtype)
                len_err = (rel.norm(dim=1) - want.norm(dim=1)).abs() / want.norm(dim=1).clamp(min=1e-6)
                len_hub = torch.where(len_err <= pair_step_huber, 0.5 * len_err ** 2, pair_step_huber * (len_err - 0.5 * pair_step_huber))
                hub = torch.where(err <= pair_step_huber, 0.5 * err ** 2, pair_step_huber * (err - 0.5 * pair_step_huber))
                return hub.mean() + (near * len_hub).sum() / near.sum().clamp(min=1)
        else:
            err = (rel - want).norm(dim=1) / want.norm(dim=1).clamp(min=1e-6)    # relative, dimensionless
        hub = torch.where(err <= pair_step_huber, 0.5 * err ** 2, pair_step_huber * (err - 0.5 * pair_step_huber))
        return hub.mean()

    _stage_desc = ['optimising']   # the bar's label, set before each stage

    # hillview: inspect={'out': <forces.npz>, ...} records, beside the solve and without changing it, what every
    # pair and every prior pulled on each camera and how far each actually moved it (inspect_forces).
    # The prior list mirrors the objective below term for term; the inspector checks every iteration
    # that the groups add up to the optimiser's own gradient, so a term missing here cannot hide.
    _insp = None
    if inspect:
        if not _fast:
            raise RuntimeError('inspect needs the batched path (no frozen cameras, no HV_SPARSE_GA_SLOW)')
        from mast3r.cloud_opt.inspect_forces import ForceInspector
        _priors = []
        if _gps_groups and gps_frac:
            _priors.append(('gps', lambda c: _gps_k[0] * loss_gps_normalised(c)))
        if _gps_groups and gps_weight:
            _priors.append(('gps_m', lambda c: gps_weight * loss_gps(c)))
        if _grav_t is not None:
            _priors.append(('gravity', lambda c: gravity_weight * loss_gravity(c)))
        if _cmp is not None:
            _priors.append(('compass', lambda c: compass_weight * loss_compass(c)))
        if _lvl_t is not None:
            _priors.append(('level', lambda c: level_weight * loss_level(c)))
        if _hgt_t is not None:
            _priors.append(('height', lambda c: height_weight * loss_height(c)))
        if _hor_t is not None:
            _priors.append(('horizon', lambda c: horizon_weight * loss_horizon(c)))
        if _upr_t is not None:
            _priors.append(('upright', lambda c: upright_weight * loss_upright(c)))
        if _gps_groups and grade_weight:
            _priors.append(('grade', lambda c: grade_weight * loss_grade(c)))
        if _gps_groups and rig_upright_weight:
            _priors.append(('rig_upright', lambda c: rig_upright_weight * loss_rig_upright(c)))
        if _pair_steps is not None:
            _priors.append(('pairstep', lambda c: pair_step_w * loss_pairstep(c)))
        if _gyro is not None:
            _priors.append(('gyro', lambda c: gyro_weight * loss_gyro(c)))
        if _gcam is not None:
            _priors.append(('gravity_cam', lambda c: gravity_cam_weight * loss_gravity_cam(c)))
        if _axis is not None:
            _priors.append(('axis', lambda c: loss_axis(c)))
        if _pace is not None:
            _priors.append(('pace', lambda c: pace_weight * loss_pace(c)))
        if _speed is not None:
            _priors.append(('speed', lambda c: speed_weight * loss_speed(c)))
        if _course is not None:
            _priors.append(('course', lambda c: course_weight * loss_course(c)))
        if _rel is not None:
            _priors.append(('gps_rel', lambda c: _rel_k[0] * loss_gps_rel(c)))
        if _mid is not None:
            _priors.append(('weak_mid', lambda c: _mid_k[0] * loss_weak_mid(c)))
        if _alt is not None:
            _priors.append(('alt_step', lambda c: _alt_k[0] * loss_alt_step(c)))
        _ictx = dict(device=device, dtype=dtype, make_K_cam_depth=make_K_cam_depth, focals_of=focals_of,
                     median_depths=median_depths, base_focals=base_focals,
                     depth_mode=depth_mode, exp_depth=exp_depth, core_depth=core_depth, quats=quats, trans=trans,
                     log_sizes=log_sizes, log_focals=log_focals, pps=pps, _img_of=_img_of, _pix_all=_pix_all,
                     _offs_all=_offs_all, _pts_didx=_pts_didx, _depth_flat=_depth_flat,
                     loss3d_slices=loss3d_slices, dust3r_slices=dust3r_slices, loss_dust3r_w=loss_dust3r_w,
                     lossd=lossd, loss2_3d_w=loss2_3d_w, dense_consistency_w=dense_consistency_w, priors=_priors,
                     rig=rig, slow_parts=_SLOW_PARTS, imgs=imgs, mst=mst)
        if loss3d_slices:
            _ictx.update(_idx3d_1=_idx3d_1, _idx3d_2=_idx3d_2, _confs3d=_confs3d)
        if _l2:
            _ictx.update(_l2_idx=_l2_idx, _l2_pix=_l2_pix, _l2_conf=_l2_conf, _l2_img_of=_l2_img_of)
        if _d3:
            # _d3_pix: each term's source pixel, its anchor's (see the pointmap fallback); the depth is the anchor's own
            _ictx.update(_d3_src=_d3_src, _d3_cam=_d3_cam, _d3_tgt=_d3_tgt, _d3_conf=_d3_conf, _d3_cfsum=_d3_cfsum,
                         _d3_pix=_g_uv1[_d3_gidx])
        _insp = ForceInspector(inspect['out'], int(inspect.get('bucket', 10)), len(imgs), _ictx,
                               chunk=int(inspect.get('chunk', 2_000_000)), debug=bool(inspect.get('debug')))
        print(f' >> inspect: {_insp.P} pair group(s) + {len(_priors)} prior(s) '
              f'({", ".join(n for n, _ in _priors) or "none"}) -> {inspect["out"]}')

    def optimize_loop(loss_func, lr_base, niter, pix_loss, lr_end=0):
        # create optimizer
        params = pps + log_focals + quats + trans + log_sizes + core_depth
        # each entry's name, for saying WHICH gradient went nonfinite (NonFiniteSolve)
        _names = [f'{n_}[{i_}]' for n_, lst_ in (('pp', pps), ('log_focal', log_focals), ('quat', quats),
                                                 ('trans', trans), ('log_size', log_sizes), ('core_depth', core_depth))
                  for i_ in range(len(lst_))]
        # hillview: ONE ENTRY PER PARAMETER. With shared intrinsics upstream puts the one focal and
        # the one principal point in these lists once per image (20x on a 20-frame walk, 80x for
        # tiles), and torch warns "optimizer contains a parameter group with duplicate parameters".
        # Autograd already sums the shared parameter's gradient over every use, so one Adam update
        # an iteration is the intended step; with the duplicates the optimiser applies it again
        # per copy. Measured on CPU (torch 2.12): one fused step moved a 20x-listed parameter
        # -0.162 where one step is -0.010, its step count +20. The first refinement step then threw
        # the shared focal ~15 %, and the CPU and CUDA solves of one recipe (identical coarse loss
        # 0.7324) parted right there: 1.303 -> 5.55 on CPU, 1.304 -> 3.10 on CUDA, ending 0.737 vs
        # 0.503. HV_ADAM_DUPS=1 keeps
        # the old list, for A/B.
        _n_listed = len(params)
        if not os.environ.get('HV_ADAM_DUPS'):
            _seen = set()
            _keep = [not (id(p_) in _seen or _seen.add(id(p_))) for p_ in params]
            params = [p_ for p_, k_ in zip(params, _keep) if k_]
            _names = [n_ for n_, k_ in zip(_names, _keep) if k_]
        print(f' >> optimiser: {len(params)} parameter tensors, {_n_listed - len(params)} duplicate entries removed'
              + (' (HV_ADAM_DUPS: duplicates KEPT, the old parameter list, for A/B)' if os.environ.get('HV_ADAM_DUPS') else ''))
        # hillview: Adam's own per-parameter update was 38 % of every iteration's kernels at 46
        # views (844 of 2,243: add_ 276, div 154, lerp_/mul_/addcmul_ 138 each) and — worse —
        # 139 `_local_scalar_dense` calls, one device sync per parameter, each stalling the
        # pipeline while the card waits for Python. It scales with the number of views, so the
        # 334-frame walks and the 720-view lanes pay the most, and it is a large part of why a
        # 4090 idled at 32 % through the optimiser stages. Fused collapses it into a handful of
        # multi-tensor kernels with no syncs; foreach is the fallback where fused refuses these
        # dtypes or devices, and is still far better than one kernel per parameter.
        _adam = dict(lr=1, weight_decay=0, betas=(0.9, 0.9))
        try:
            # An A/B has to be able to ASK for the old path. Without a switch both arms run
            # the same code and "equivalence" is vacuous -- which is how the GPS A/B silently
            # measured nothing. The chosen arm is printed, so the run log carries the evidence.
            if os.environ.get('HV_ADAM', '').strip().lower() == 'foreach':
                raise RuntimeError('HV_ADAM=foreach requested')
            optimizer = torch.optim.Adam(params, fused=True, **_adam)
            print(' >> Adam: fused')
        except (RuntimeError, ValueError) as e:
            print(f' >> fused Adam unavailable ({type(e).__name__}: {e}); using foreach')
            optimizer = torch.optim.Adam(params, foreach=True, **_adam)
        ploss = pix_loss if 'meta' in repr(pix_loss) else (lambda a: pix_loss)

        def objective(K, w2cam, cam2w, depthmaps, pts3d, pix_loss):
            # the stage's whole objective at one state: (total, pixel term). The loop differentiates it;
            # after the last step it is evaluated once more, at the state the solve returns
            _pix = loss_func(K, w2cam, pts3d, pix_loss)
            _gps_term = 0
            if _gps_groups and gps_frac:
                _raw = loss_gps_normalised(cam2w)
                if _gps_k[0] is None:
                    with torch.no_grad():
                        # .detach(): float() of a grad-carrying tensor warns, and the warning's
                        # echoed SOURCE LINE became the run's `now` on the bench
                        r = float(_raw.detach())
                        _gps_k[0] = (float(gps_frac) * float(_pix.detach()) / r) if r > 1e-12 else 0.0
                    print(f' >> GPS term normalised: x{_gps_k[0]:.4g} so it starts at '
                          f'{gps_frac:.0%} of the pixel objective ({float(_pix.detach()):.4g}); '
                          f'raw {r:.4g} in metre-units, gate <= {gps_gate_max}')
                _gps_term = _gps_k[0] * _raw
            if _holds_pending():
                with torch.no_grad():
                    _normalise_holds(float(_pix.detach()), cam2w, '')
            loss = (_pix
                    + loss_dust3r_w * loss_dust3r(K, cam2w, depthmaps, lossd)
                    + _gps_term
                    + (gps_weight * loss_gps(cam2w) if (_gps_groups and gps_weight) else 0)
                    + (gravity_weight * loss_gravity(cam2w) if _grav_t is not None else 0)
                    + (compass_weight * loss_compass(cam2w) if _cmp is not None else 0)
                    + (level_weight * loss_level(cam2w) if _lvl_t is not None else 0)
                    + (height_weight * loss_height(cam2w) if _hgt_t is not None else 0)
                    + (horizon_weight * loss_horizon(cam2w) if _hor_t is not None else 0)
                    + (upright_weight * loss_upright(cam2w) if _upr_t is not None else 0)
                    + (grade_weight * loss_grade(cam2w) if (_gps_groups and grade_weight) else 0)
                    + (rig_upright_weight * loss_rig_upright(cam2w)
                       if (_gps_groups and rig_upright_weight) else 0)
                    + (pair_step_w * loss_pairstep(cam2w) if _pair_steps is not None else 0)
                    + (gyro_weight * loss_gyro(cam2w) if _gyro is not None else 0)
                    + (gravity_cam_weight * loss_gravity_cam(cam2w) if _gcam is not None else 0)
                    + (loss_axis(cam2w) if _axis is not None else 0)
                    + (pace_weight * loss_pace(cam2w) if _pace is not None else 0)
                    + (speed_weight * loss_speed(cam2w) if _speed is not None else 0)
                    + (course_weight * loss_course(cam2w) if _course is not None else 0)
                    + (_rel_k[0] * loss_gps_rel(cam2w) if _rel is not None else 0)
                    + (_mid_k[0] * loss_weak_mid(cam2w) if _mid is not None else 0)
                    + (_alt_k[0] * loss_alt_step(cam2w) if _alt is not None else 0))
            return loss, _pix

        count_ops = os.environ.get('HV_COUNT_OPS')
        with tqdm(total=niter, desc=_stage_desc[0], ascii=' .') as bar:
            for iter in range(niter or 1):
                _counter = None
                if count_ops and iter == 2:
                    # hillview: how many kernels ONE iteration launches (forward, backward, step) --
                    # the number that decides the GPU's idle time, measurable on any CPU box
                    from torch.utils._python_dispatch import TorchDispatchMode

                    class _CountOps(TorchDispatchMode):
                        n = 0
                        by_op = {}

                        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                            _CountOps.n += 1
                            k = str(func.overloadpacket.__name__ if hasattr(func, 'overloadpacket') else func)
                            _CountOps.by_op[k] = _CountOps.by_op.get(k, 0) + 1
                            return func(*args, **(kwargs or {}))
                    _counter = _CountOps()
                    _counter.__enter__()
                K, (w2cam, cam2w), depthmaps = make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth)
                pts3d = (make_pts3d_fast(K, cam2w, depthmaps) if (_fast and _fastp('pts'))
                         else make_pts3d(anchors, K, cam2w, depthmaps, base_focals=base_focals))
                _cur['w2cam'], _cur['cam2w'], _cur['depthmaps'] = w2cam, cam2w, depthmaps
                if niter == 0:
                    break
                if trajectory is not None:
                    # hillview: WHERE EVERY CAMERA WAS AT EVERY ITERATION, before this iteration's step
                    # -- the camera-to-world [R|t], the stage and the step size. A few MB a run and no
                    # gradient work: it is what tells a camera that settled from one the schedule cut
                    # off still moving (recon_metrics convergence), and what the scene viewer replays.
                    # kept ON THE DEVICE: a host copy per iteration is a sync point on CUDA, every
                    # iteration of every solve; the caller moves the stack once, after the solve
                    trajectory.setdefault('poses', []).append(cam2w.detach()[:, :3, :].to(torch.float32).clone())
                    trajectory.setdefault('stage', []).append(_stage_desc[0])
                    trajectory.setdefault('it', []).append(iter)
                    trajectory.setdefault('lr', []).append(float(schedule(iter / niter, lr_base, lr_end))
                                                           * (min(1.0, (iter + 1) / lr_warmup) if lr_warmup else 1.0))

                alpha = (iter / niter)
                lr = schedule(alpha, lr_base, lr_end)
                # hillview: WARM-UP (--lr_warmup). Adam's first steps move EVERY parameter by ~lr whatever
                # its gradient (m/sqrt(v) = +-1 at step 1); the rotations are raw quaternion components and
                # the kinematic tree is up to 41 deep, so the per-link steps compound -- every photograph of
                # a 40-panorama church ride turned ~70 deg in iteration 1, gauge-free. A pile start shrugs that
                # off (its links have no length); a placed start (pose_init) has links of tens of metres
                # and the swing throws it. 0 keeps upstream's schedule exactly.
                if lr_warmup and iter < lr_warmup:
                    lr = lr * (iter + 1) / lr_warmup
                adjust_learning_rate_by_lr(optimizer, lr)
                pix_loss = ploss(1 - alpha)
                optimizer.zero_grad()
                _gauge_cache.clear()          # the cameras moved; refit the gauge once below
                loss, _pix = objective(K, w2cam, cam2w, depthmaps, pts3d, pix_loss)
                # hillview: NOTHING NONFINITE REACHES A STEP. The
                # NaN check used to come after backward and Adam.step and only broke the loop, so the
                # corrupted parameters went on into stage 2 and out as the result; inf passed unseen.
                # The loss is read before backward (the one sync per iteration the bar always made,
                # moved earlier) and every gradient before the step (one more: an inf-norm per tensor,
                # which propagates NaN). Either raises; nothing is restored or skipped.
                _lval = float(loss.detach())
                if not math.isfinite(_lval):
                    raise NonFiniteSolve(_stage_desc[0], iter, 'loss',
                                         f'{_lval}; pixel term {float(_pix.detach()) if torch.is_tensor(_pix) else _pix}')
                loss.backward()
                _gi = [k_ for k_, p_ in enumerate(params) if p_.grad is not None]
                if _gi:
                    _gmax = torch.stack(torch._foreach_norm([params[k_].grad for k_ in _gi], float('inf')))
                    if not bool(torch.isfinite(_gmax).all()):
                        _bad = [_names[_gi[k_]] for k_ in torch.nonzero(~torch.isfinite(_gmax)).flatten().tolist()]
                        raise NonFiniteSolve(_stage_desc[0], iter, 'gradient',
                                             f'{len(_bad)} of {len(_gi)} parameter tensors: {", ".join(_bad[:8])}'
                                             + (' ...' if len(_bad) > 8 else ''))
                if _insp is not None:
                    _insp.before_step(iter, niter, _stage_desc[0], 'l3d' if loss_func is loss_3d else 'l2d', pix_loss, lr)
                optimizer.step()
                if _insp is not None:
                    _insp.after_step(optimizer)

                # make sure the pose remains well optimizable
                if _fast and _fastp('quat'):
                    _q = [q.data for q in quats]
                    torch._foreach_div_(_q, torch._foreach_norm(_q))
                else:
                    for i in range(len(imgs)):
                        quats[i].data[:] /= quats[i].data.norm()
                if _counter is not None:
                    _counter.__exit__(None, None, None)
                    top = sorted(_CountOps.by_op.items(), key=lambda kv: -kv[1])[:12]
                    print(f' >> kernels in one iteration (forward + backward + step): {_CountOps.n}; '
                          f'most: {", ".join(f"{k} {v}" for k, v in top)}')

                loss = _lval
                bar.set_postfix_str(f'{lr=:.4f}, {loss=:.3f}')
                bar.update(1)

        if niter:
            # hillview: THE RESULT IS THE STATE THE LAST STEP MADE.
            # The geometry above was computed BEFORE each iteration's step, so the returned cameras,
            # depths and points were one step behind the parameters: stage 1 handed back a state other
            # than the one stage 2 started from, and the final result lacked the last refinement step
            # while the end diagnostics (focals, scales, pair-step residuals) read the newer parameters.
            # One evaluation more, without gradients, of the geometry and the objective at that state;
            # the logged final loss is that objective, under the last iteration's loss schedule.
            with torch.no_grad():
                K, (w2cam, cam2w), depthmaps = make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth)
                pts3d = (make_pts3d_fast(K, cam2w, depthmaps) if (_fast and _fastp('pts'))
                         else make_pts3d(anchors, K, cam2w, depthmaps, base_focals=base_focals))
                _cur['w2cam'], _cur['cam2w'], _cur['depthmaps'] = w2cam, cam2w, depthmaps
                _gauge_cache.clear()
                final, _pix = objective(K, w2cam, cam2w, depthmaps, pts3d, pix_loss)
                loss = float(final)
            if not math.isfinite(loss):
                raise NonFiniteSolve(_stage_desc[0], niter, 'loss at the returned state', f'{loss}')
            print(f'>> final loss = {loss} (at the returned state, after step {niter})')
        return dict(intrinsics=K.detach(), cam2w=cam2w.detach(),
                    depthmaps=[d.detach() for d in depthmaps], pts3d=[p.detach() for p in pts3d])

    def _set_initial_poses():
        # hillview: START THE CAMERAS WHERE THE PICTURES AND THE GPS PUT THEM (--pose_init), not on one
        # pile. The OBJECTIVE IS UNCHANGED: the GPS term's normaliser is taken at the pile, exactly as a
        # pile start takes it at its first iteration, so a pile and a placed start of one recipe differ
        # in where they begin and in nothing else.
        from mast3r.cloud_opt.pose_init import initial_poses
        with torch.no_grad():
            K0, (w2c0, c2w0), D0 = make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth)
            _held_pile[0] = True        # held targets: the normalisers below take no held residual (_with_held)
            try:
                if _gps_groups and gps_frac and _gps_k[0] is None:
                    pts0 = (make_pts3d_fast(K0, c2w0, D0) if (_fast and _fastp('pts'))
                            else make_pts3d(anchors, K0, c2w0, D0, base_focals=base_focals))
                    _gauge_cache.clear()
                    pl0 = loss1(1.0) if 'meta' in repr(loss1) else loss1
                    pix0 = float(loss_3d(K0, w2c0, pts0, pl0))
                    raw0 = float(loss_gps_normalised(c2w0))
                    _gauge_cache.clear()
                    _gps_k[0] = (float(gps_frac) * pix0 / raw0) if raw0 > 1e-12 else 0.0
                    print(f' >> GPS term normalised AT THE PILE: x{_gps_k[0]:.4g} (pixel {pix0:.4g}, raw {raw0:.4g}); '
                          f'the objective is the one a pile start would have')
                if _holds_pending():
                    # the weak holds' normalisers at the same state, for the same reason
                    pts0 = (make_pts3d_fast(K0, c2w0, D0) if (_fast and _fastp('pts'))
                            else make_pts3d(anchors, K0, c2w0, D0, base_focals=base_focals))
                    _gauge_cache.clear()
                    pl0 = loss1(1.0) if 'meta' in repr(loss1) else loss1
                    _normalise_holds(float(loss_3d(K0, w2c0, pts0, pl0)), c2w0, ' AT THE PILE')
                    _gauge_cache.clear()
            finally:
                _held_pile[0] = False
            if _held is not None:
                # THE START IS GIVEN IN THE TARGETS' FRAME (held_start), and each view's depth begins at the
                # scale its targets put it (held_depth_scale: the pile's depths are at size 1, so the ratio is the size)
                T = torch.as_tensor(held_start(pose_init_poses, len(imgs)), dtype=dtype, device=device)
                _shapes = [(int(imsizes[i_][1]), int(imsizes[i_][0])) for i_ in range(len(imgs))]
                _ls, _used = held_depth_scale(inv(T), _held, D0, _shapes, subsample)
                for i_ in range(len(imgs)):
                    log_sizes[i_].data[:] = float(_ls[i_])
                print(f' >> held start: {len(imgs)} view(s) at the given poses in the targets\' frame; depth '
                      f'scale from the targets on {len(_used)} view(s) (median x{float(np.exp(np.median(_ls))):.4g}, '
                      f'{float(np.exp(_ls.min())):.4g}..{float(np.exp(_ls.max())):.4g}), their median on the other '
                      f'{len(imgs) - len(_used)}')
            else:
                eye = torch.eye(4, dtype=dtype, device=device)[None].expand(len(imgs), 4, 4)
                cam_pts = make_pts3d(anchors, K0, eye, D0, base_focals=base_focals)
                strong = [(s_.img1, s_.img2, np.arange(s_.slice1.start, s_.slice1.stop),
                           np.arange(s_.slice2.start, s_.slice2.stop), s_.confs) for s_ in loss3d_slices]
                # the weak ties as the loss has them: each correspondence's ANCHOR point (see _grid)
                grid_cam = dict(zip(_grid, make_pts3d(_grid, K0, eye, D0, base_focals=base_focals)))
                weak = [(s_.img1, s_.img2, grid_cam[s_.img1][anchors[s_.img1][1]])
                        + tuple(preds_21[imgs[s_.img2]][imgs[s_.img1]][:2]) for s_ in dust3r_slices]
                target, rep_ = initial_poses(len(imgs), cam_pts, strong, weak, rig, gps, pose_init,
                                             imu_down=imu_down, log=print, weak_weight=float(loss_dust3r_w),
                                             given=pose_init_poses)
                T = torch.as_tensor(target, dtype=dtype, device=device)
                if pose_init_frac != 1.0:
                    # hillview --pose_init_frac: PART WAY BETWEEN THE PILE AND THE PLACED START. Every camera
                    # centre is moved toward the cameras' mean centre, keeping the fraction of its offset;
                    # orientations stay as placed (the pile's identity is a gauge, not an orientation worth
                    # interpolating toward). Not a similarity of the start: the depths keep their scale, so
                    # at 0.5 the cameras begin closer together than their own pointmaps say.
                    if not 0.0 <= float(pose_init_frac) <= 1.0:
                        raise ValueError(f'pose_init_frac {pose_init_frac!r}: expected 0..1')
                    c_ = T[:, :3, 3].mean(0)
                    T = T.clone()
                    T[:, :3, 3] = c_ + float(pose_init_frac) * (T[:, :3, 3] - c_)
                    print(f' >> pose init: centres at {float(pose_init_frac):g} of the placed start\'s offsets from their mean '
                          f'(0 = one spot, as the pile)')
            # INVERSE KINEMATICS: the tree's parameters that reproduce these absolute poses. A camera's
            # world centre is gs * (t - R o), o its focal-plane offset (make_K_cam_depth), so the chain
            # carries t = c / gs + R o; every link is then parent^-1 @ child. Rig links are fixed and
            # come out right by construction (slices share a centre and the rig's rotation).
            sizes = torch.cat(log_sizes).exp(); gs = _scaling(sizes)     # make_K_cam_depth's factor (1 with held targets)
            foc = focals_of(log_focals)
            z = sizes * median_depths * foc / base_focals
            off = z.unsqueeze(1) * torch.cat((imsizes / foc.unsqueeze(1) * (0.5 - torch.stack(pps)), ones), dim=-1)
            tmp = T.clone()
            tmp[:, :3, 3] = T[:, :3, 3] / gs + (T[:, :3, :3] @ off.unsqueeze(-1)).squeeze(-1)
            r0 = mst[0]
            quats[r0].data[:] = roma.rotmat_to_unitquat(tmp[r0, :3, :3])
            trans[r0].data[:] = tmp[r0, :3, 3]
            for i, j in mst[1]:
                if j in rig_edges:
                    continue
                relm = torch.linalg.inv(tmp[i]) @ tmp[j]
                quats[j].data[:] = roma.rotmat_to_unitquat(relm[:3, :3])
                trans[j].data[:] = relm[:3, 3]
            _, (_, c2w1), _ = make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth)
            spread = float(T[:, :3, 3].std(0).norm())
            e_c = float((c2w1[:, :3, 3] - T[:, :3, 3]).norm(dim=1).max())
            e_r = float((c2w1[:, :3, :3] - T[:, :3, :3]).abs().max())
            print(f' >> pose init ({pose_init}): the tree reproduces the start to {e_c:.2e} units '
                  f'(scene spread {spread:.3g}) and {e_r:.1e} in rotation')
            if not (e_c <= 1e-3 * max(spread, 1e-6) and e_r <= 1e-3):
                raise RuntimeError(f'pose init: the kinematic tree does not reproduce the start poses '
                                   f'({e_c:.3g} units, {e_r:.3g} rotation): a rig link disagrees with them')
            if pose_fixed:
                _pose_fixed[0] = T.detach().clone()
                _pose_fixed[1] = inv(_pose_fixed[0])
                print(f' >> pose fixed: the {len(imgs)} camera(s) are HELD at the given poses (pose_fixed); only the '
                      f'depths (each view its own scale) and the intrinsics are solved')

    # at start, don't optimize 3d points
    for i, img in enumerate(imgs):
        trainable = not (init[img].get('freeze'))
        pps[i].requires_grad_(False)
        log_focals[i].requires_grad_(False)
        quats[i].requires_grad_(trainable and i not in rig_edges)
        trans[i].requires_grad_(trainable and i not in rig_edges)
        log_sizes[i].requires_grad_(trainable)
        core_depth[i].requires_grad_(False)

    if pose_fixed and pose_init != 'poses':
        raise ValueError(f"pose_fixed holds GIVEN poses: it needs pose_init 'poses' (got {pose_init!r})")
    if _held is not None:
        # a held solve starts where its caller placed it, in the targets' frame (held_start); the pile and the
        # pictures' own placement (pose_init.initial_poses) are in this run's own units, which no target shares
        if pose_init != 'poses' or pose_init_poses is None:
            raise ValueError(f"HELD_TARGETS: the start is given in the targets' frame, so pose_init 'poses' with the "
                             f"poses (got {pose_init!r})")
        if float(pose_init_frac) != 1.0:
            raise ValueError(f'HELD_TARGETS with pose_init_frac {pose_init_frac}: the start is in the targets\' frame, '
                             f'and part way to a pile at its origin is nowhere')
        if rig_edges:
            raise ValueError('HELD_TARGETS with a rig: a held solve of rigged views is not built (the start would have '
                             'to agree with every rig link)')
    if pose_fixed and _pair_steps is not None:
        raise ValueError('pose_fixed with the pair step prior: the prior moves cameras, and these are held')
    if pose_fixed:
        for i in range(len(imgs)):
            quats[i].requires_grad_(False)
            trans[i].requires_grad_(False)
    if pose_init != 'pile':
        _set_initial_poses()
    # hillview (a dense arm reuses the solve). RESTORE: a finished run's final state -- its intrinsics,
    # cam2w and (flat) core depthmaps, as scene.npz keeps them at this precision, with each view's canonical entry name
    # and the anchor grid -- in place of the two stages. Everything above ran for real on THIS run's inputs (the pair
    # terms, the policy's selection, the records the caller reads), so the reuse is checked against them: the same
    # canonical views, in order, the same grid, the same anchor counts. The points come from the state by the very code
    # the stages' last evaluation uses. The canonical views ARE the source's entries:
    # prepare_canonical_data read them in place of this run's own, each checked there against this run's views and
    # its pair predictions bit for bit (_source_canonical); the names below are then equal by construction, and the
    # check stays to say so if a caller ever restores without them.
    if restore is not None:
        if inspect is not None or trajectory is not None:
            raise ValueError('restore: the inspection and the trajectory record iterations, and a restored solve runs none')
        _mine = [os.path.basename(str(c_)) for c_ in canonical_paths]
        _theirs = [os.path.basename(str(c_)) for c_ in restore['canonical']]
        if _mine != _theirs:
            _k = next((k_ for k_, (x_, y_) in enumerate(zip(_mine, _theirs)) if x_ != y_), min(len(_mine), len(_theirs)))
            raise ValueError(f"restore {restore['source']}: this run's optimiser inputs are not the source's: "
                             f"{len(_mine)} canonical views vs {len(_theirs)}, first difference at view {_k}")
        if int(restore['subsample']) != int(subsample):
            raise ValueError(f"restore {restore['source']}: anchor grid {subsample} vs the source's {restore['subsample']}")
        if [int(d_.numel()) for d_ in core_depth] != [int(np.asarray(d_).size) for d_ in restore['depthmaps']]:
            raise ValueError(f"restore {restore['source']}: anchor counts per view differ from the source's")
        print(f" >> stage 1/2 and 2/2 RESTORED from {restore['source']}: its final state, the same {len(_mine)} "
              f"canonical views; no iteration runs")
        with torch.no_grad():
            def _t(x_):
                return torch.as_tensor(np.ascontiguousarray(x_)).to(device=device, dtype=dtype)
            K_r, cam2w_r = _t(restore['intrinsics']), _t(restore['cam2w'])
            depth_r = [_t(d_) for d_ in restore['depthmaps']]
            pts_r = (make_pts3d_fast(K_r, cam2w_r, depth_r) if (_fast and _fastp('pts'))
                     else make_pts3d(anchors, K_r, cam2w_r, depth_r, base_focals=base_focals))
        res_coarse = None
        res_fine = dict(intrinsics=K_r, cam2w=cam2w_r, depthmaps=depth_r, pts3d=[p_.detach() for p_ in pts_r])
        niter2_ran = 0
    else:
        print(f' >> stage 1/2: coarse alignment on 3D matches, {niter1} iterations')
        _stage_desc[0], _t_stage = 'stage 1/2 coarse alignment', time.time()
        res_coarse = optimize_loop(loss_3d, lr_base=lr1, niter=niter1, pix_loss=loss1)
        print(f' >> stage 1/2 done in {time.time() - _t_stage:.0f} s')
        res_fine = None
        niter2_ran = niter2
    if niter2_ran:
        print(f' >> stage 2/2: refinement on 2D reprojection, {niter2} iterations')
        _stage_desc[0], _t_stage = 'stage 2/2 refinement', time.time()
        # now we can optimize 3d points
        # hillview: A PINNED PRINCIPAL POINT IS HELD, THE REST ARE REFINED. A tile's
        # principal point is where its crop put the parent's axis and a slice's is its centre: facts,
        # held. reconstruct used to turn opt_pp off for the whole run to hold them, which froze every
        # PHOTOGRAPH's beside them in a mixed span at the canonical estimate. Held per parameter, so a
        # principal point shared with a pinned view is held too (the one-lens modes share one).
        if pinned_pp is not None and len(pinned_pp) != len(imgs):
            raise ValueError(f'pinned_pp: {len(pinned_pp)} for {len(imgs)} views')
        _pp_held = {id(pps[i]) for i in range(len(imgs)) if pinned_pp is not None and pinned_pp[i]}
        for i, img in enumerate(imgs):
            if init[img].get('freeze', 0) >= 1:
                continue
            pps[i].requires_grad_(bool(opt_pp) and id(pps[i]) not in _pp_held)
            log_focals[i].requires_grad_(not _hold[i])
            core_depth[i].requires_grad_(opt_depth)

        # refinement with 2d reproj
        if loss2_3d_w or dense_consistency_w:
            # hillview: slices of a panorama share a centre, so the 2D loss between them
            # cannot see depth and their depths drift apart in this stage (measured: 0.41 m
            # at 7 m between neighbouring slices). Keep the 3D loss in the sum; and the
            # dense cross-view term when asked.
            def loss_stage2(K, w2cam, pts3d, pix_loss):
                total = loss_2d(K, w2cam, pts3d, pix_loss)
                if loss2_3d_w:
                    total = total + loss2_3d_w * loss_3d(K, w2cam, pts3d, loss1)
                if dense_consistency_w:
                    total = total + dense_consistency_w * loss_dense(K, _cur['w2cam'], _cur['cam2w'], _cur['depthmaps'])
                return total
            res_fine = optimize_loop(loss_stage2, lr_base=lr2, niter=niter2, pix_loss=loss2)
        else:
            res_fine = optimize_loop(loss_2d, lr_base=lr2, niter=niter2, pix_loss=loss2)

    if niter2_ran:
        print(f' >> stage 2/2 done in {time.time() - _t_stage:.0f} s')
    # the diagnostics below read the optimiser's PARAMETERS (core depths, sizes, focals), which a restored solve never
    # moved: they describe its start, not the state it hands back, so they are not printed (the source's log has them)
    if restore is not None:
        print(' >> the end diagnostics of the parameters (depth field scale, pair step prior, focals, per-image scale) '
              f"are the source's ({restore['source']}): a restored solve moved none")
    if niter2_ran and opt_depth and not lora_depth:
        # hillview: where an image's scale went during refinement, per image -- the median of its
        # optimised depth field against 1 as initialised (mul: the factor itself; add: the offset
        # the depths took, in the same units). A run whose camera heights drift should show it here.
        with torch.no_grad():
            _m = [float((d.exp() if exp_depth else d).median()) for d in core_depth]
        print(' >> depth field scale per image (median of the optimised core depth, 1 = as initialised, mode '
              f'{depth_mode}): ' + ' '.join(f'{m:.3f}' for m in _m))
    if _pair_steps is not None and restore is None:
        # hillview: where the solve ended against the beliefs, per pair -- the relative error the
        # prior sees, and the plain ratio |solve step| / |believed step| -- so a run's log says which
        # pairs the optimiser refused, instead of leaving that to be inferred from a walk metric.
        with torch.no_grad():
            K_, (_, cam2w_), _ = make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth)
            pi, pj, T = _pair_steps
            sizes_ = torch.cat(log_sizes).exp(); gs_ = 1 / sizes_.min()
            foc_ = focals_of(log_focals)
            k_ = gs_ * sizes_[pi] * foc_[pi] / base_focals[pi]
            if pair_step_depth_scale:
                meds_ = torch.stack([(d.exp() if exp_depth else d).median() for d in core_depth])
                k_ = k_ * ((1 + (meds_ - 1) * base_focals / foc_) if depth_mode == 'add' else meds_)[pi]
            rel_ = torch.einsum('nji,nj->ni', cam2w_[pi, :3, :3], cam2w_[pj, :3, 3] - cam2w_[pi, :3, 3])
            want_ = k_.unsqueeze(1) * T
            err_ = (rel_ - want_).norm(dim=1) / want_.norm(dim=1).clamp(min=1e-6)
            ratio_ = rel_.norm(dim=1) / want_.norm(dim=1).clamp(min=1e-6)
            cos_ = (rel_ * want_).sum(1) / (rel_.norm(dim=1) * want_.norm(dim=1)).clamp(min=1e-9)
            ang_ = torch.rad2deg(torch.acos(cos_.clamp(-1, 1)))
        n_out = int((err_ > pair_step_huber).sum())
        print(f' >> pair step prior at the end: {len(err_)} pairs, relative error median {err_.median():.2f}, '
              f'{n_out} beyond the knee {pair_step_huber}; |solve|/|belief| median {ratio_.median():.2f}; '
              f'direction off by median {ang_.median():.1f} deg')
        worst = torch.argsort(err_, descending=True)[:12].tolist()
        print(' >> pair step prior, worst pairs (i->j: rel err, |solve|/|belief|, deg): ' + ', '.join(
            f'{int(pi[w])}->{int(pj[w])}: {float(err_[w]):.2f}, {float(ratio_[w]):.2f}, {float(ang_[w]):.0f}' for w in worst))
    if restore is None:
        K = make_K_cam_depth(log_focals, pps, None, None, None, None)
        if shared_intrinsics == 'groups':
            print('Final focals (per group) = ' + ', '.join(
                f'{fg}: {float(K[idx[0], 0, 0].detach()):.2f}' for fg, idx in fgroups.items()))
        elif shared_intrinsics:
            print('Final focal (shared) = ', to_numpy(K[0, 0, 0]).round(2))
        else:
            print('Final focals =', to_numpy(K[:, 0, 0]))
        # hillview: the per-image SCALE the solve settled on -- the multiplier that places each image's
        # pointmap in the world (sizes_i, times the run's global scaling), normalised to the first
        # image. This is the direct reading of "scale drift"; the camera-height metric only infers it
        # from the ground and cannot tell a slid scale from a real slope, so the heights alone are not
        # to be read too much into. An image at 2.0 here has its
        # scene twice as large as image 0's, whatever the terrain does.
        with torch.no_grad():
            _sz = torch.cat(log_sizes).exp().detach().cpu().numpy()
        print(' >> per-image scale (pointmap multiplier, relative to image 0): ' + ' '.join(f'{v:.3f}' for v in (_sz / max(float(_sz[0]), 1e-12))))

    if pair_geometry_path is not None:
        from mast3r.cloud_opt.pair_geometry import write_geometry
        # the pointmap terms' sources as the loss compares them: anchor points at anchor pixels
        _res = res_fine or res_coarse
        with torch.no_grad():
            _gp = grid_pts3d(_res['intrinsics'], _res['cam2w'], _res['depthmaps'])
        write_geometry(pair_geometry_path, _res, imsizes,
                       loss3d_slices, dust3r_slices, preds_21, imgs, loss_dust3r_w,
                       {i: (_gp[i][anchors[i][1]], _grid[i][0][anchors[i][1], :2]) for i in _grid})
    if _insp is not None:
        _insp.save((res_fine or res_coarse)['cam2w'], imgs, mst)
    return imgs, res_coarse, res_fine


@lru_cache
def mask110(device, dtype):
    return torch.tensor((1, 1, 0), device=device, dtype=dtype)


def proj3d(inv_K, pixels, z):
    if pixels.shape[-1] == 2:
        pixels = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    return z.unsqueeze(-1) * (pixels * inv_K.diag() + inv_K[:, 2] * mask110(z.device, z.dtype))


def make_pts3d(anchors, K, cam2w, depthmaps, base_focals=None, ret_depth=False):
    focals = K[:, 0, 0]
    invK = inv(K)
    all_pts3d = []
    depth_out = []

    for img, (pixels, idxs, offsets) in anchors.items():
        # from depthmaps to 3d points
        if base_focals is None:
            pass
        else:
            # compensate for focal
            # depth + depth * (offset - 1) * base_focal / focal
            # = depth * (1 + (offset - 1) * (base_focal / focal))
            offsets = 1 + (offsets - 1) * (base_focals[img] / focals[img])

        pts3d = proj3d(invK[img], pixels, depthmaps[img][idxs] * offsets)
        if ret_depth:
            depth_out.append(pts3d[..., 2])  # before camera rotation

        # rotate to world coordinate
        pts3d = geotrf(cam2w[img], pts3d)
        all_pts3d.append(pts3d)

    if ret_depth:
        return all_pts3d, depth_out
    return all_pts3d


def make_dense_pts3d(intrinsics, cam2w, depthmaps, canonical_paths, subsample, device='cuda'):
    base_focals = []
    anchors = {}
    confs = []
    for i, canon_path in enumerate(canonical_paths):
        (canon, canon2, conf), focal = _cache_load(_found(canon_path), device)
        confs.append(conf)
        base_focals.append(focal)
        H, W = conf.shape
        pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
        idxs, offsets = anchor_depth_offsets(canon2, {i: (pixels, None)}, subsample=subsample)
        anchors[i] = (pixels, idxs[i], offsets[i])

    # densify sparse depthmaps
    pts3d, depthmaps_out = make_pts3d(anchors, intrinsics, cam2w, [
                                      d.ravel() for d in depthmaps], base_focals=base_focals, ret_depth=True)

    return pts3d, depthmaps_out, confs


# A WAIT THAT SAYS SO (the worker's idle kill): a run blocked on another run's pair lock burns no CPU and
# writes nothing, and the worker kills a solve idle for 2 h. The blocking wait polls, and says every LOCK_WAIT_SAY_S
# whom it waits on (the holder's pid and run, from /proc/locks) -- a line in run.log, so an honest wait is never
# idle, and a wait on a wedged holder names it.
LOCK_WAIT_SAY_S = 600.0
LOCK_POLL_S = 0.25


def _lock_holder(path):
    """'pid N (<its run dir>)' holding the flock on `path`, from /proc/locks; None when no holder is listed there"""
    try:
        st = os.stat(path)
        dev = f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
        with open('/proc/locks') as f:
            for line in f:                   # '1: FLOCK  ADVISORY  WRITE 1234 00:2a:5678 0 EOF'; a waiter has '->'
                p = line.split()
                if len(p) >= 6 and p[1] == 'FLOCK' and p[5] == dev:
                    try:
                        with open(f'/proc/{p[4]}/cmdline', 'rb') as c:
                            argv = c.read().split(b'\0')
                        job = argv[argv.index(b'--job') + 1].decode() if b'--job' in argv[:-1] else ''
                    except (OSError, ValueError, IndexError):
                        job = ''
                    run = os.path.basename(os.path.dirname(job)) if job else 'not a reconstruct.py --job'
                    return f'pid {p[4]} ({run})'
    except OSError:
        return None
    return None


class _PairLock:
    """hillview: one process at a time computes a given pair's forward passes and matches. The
    cache is shared and content-addressed, so two runs on one uncached span used to compute every
    pair twice, side by side (two runs of one 20-frame span: 72-89 s a pair
    each, against 17 s alone). With the lock the second run waits on each pair and then reuses it.
    flock on a small file beside the cache; released by the kernel if the holder dies."""

    def __init__(self, path):
        self.path, self.fh, self.waited = path, None, 0.0

    def __enter__(self):
        self.acquire(True)
        return self

    def acquire(self, block):
        """-> whether the lock is now held. block=False returns False at once when another holder has it
        (another run computing the pair: forward_mast3r then runs other pairs first, see _next_pair)."""
        import fcntl
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fh = open(self.path, 'a')
        t0 = said = time.time()
        while True:
            try:
                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not block:
                    self.fh.close()
                    self.fh = None
                    return False
            if time.time() - said >= LOCK_WAIT_SAY_S:          # A WAIT THAT SAYS SO (above)
                said = time.time()
                print(f'  waiting {(said - t0) / 60:.0f} min for the pair lock {os.path.basename(self.path)}: '
                      f'another run is computing this pair -- held by '
                      f'{_lock_holder(self.path) or "a process /proc/locks does not list"}', flush=True)
            time.sleep(LOCK_POLL_S)
        self.waited = time.time() - t0
        return True

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()
        return False


# hillview: every cache entry carries `<f>.content`, one line
# `sha256:<hex>` over its DECODED arrays (structure, dtypes, shapes, then each array's bytes), beside
# `<f>.sha256` over the stored bytes. The content is the entry's identity: two machines that store the
# same arrays in other bytes (the RECON_FORWARD_RAM_DIR re-deflate below, later the .hvz container)
# agree on it. ONE definition, hillview's scripts/enrich/hvz.content_digest, installed here by the
# caller (hvz.install); mast3r imports nothing of hillview, and a write with it unset is refused.
CONTENT_DIGEST = None
# hillview: the shared cache is STORED in hillview's .hvz container
# (planar, byte-shuffled, zlib: ~0.54 of a torch .pth, lossless). The codec is hillview's too, installed
# with the digest by hvz.install: CACHE_EXT names every cache entry, CACHE_DUMPS(obj) -> bytes writes
# one, CACHE_LOAD(path, device) reads one. No torch fallback: with them unset a cache read or write
# raises, and a .pth is never a cache entry any more.
CACHE_EXT = None
CACHE_DUMPS = None
CACHE_LOAD = None
# WHILE BOTH FORMATS EXIST: an entry is written under its
# CACHE_EXT name and READ from whichever container holds it. CACHE_CANDIDATES(path) -> the names one
# entry may have, in the order a reader takes them (the CACHE_EXT name first, then the torch .pth of
# the same key, in the same root -- never another root); the first that exists is the entry. Both carry
# the same .content, so this is one entry in two containers, not a fallback to other data. The
# candidates are tried with _stat, so a warm hit on a .hvz costs one stat per file, as before.
# Retiring the .pth candidate later happens inside hillview's hvz; nothing here changes.
# READ-ONLY ROOTS: hillview may install more roots than the one cache_path
# names -- a local copy prepared for this box, an archive, read after the primary and never
# written. The candidates then run over each root in turn (the rule above inside each, never mixing
# two for one entry), and the FIRST is still the name the entry is WRITTEN under, in the primary,
# whichever root a path was found in: every writer here derives its target through _write_name, and
# _save_atomic refuses any other name.
CACHE_CANDIDATES = None
# CACHE_ENTRY_REL(path, root) -> the entry's path relative to the root that HOLDS it (`root` or one
# of the read-only roots behind it): what canonical_cache keys on, so a pair found in a read-only root
# keys exactly as the same pair at home. Relative to `root` alone it would be '../../<other root>/...'.
CACHE_ENTRY_REL = None


def _codec_unset(what):
    return RuntimeError(f'sparse_ga.{what} is unset: the shared cache is read and written through '
                        "hillview's codec (scripts/enrich/hvz.install(sparse_ga) installs it)")


def _cache_load(path, device):
    """one shared-cache entry, on `device`."""
    if CACHE_LOAD is None:
        raise _codec_unset('CACHE_LOAD')
    return CACHE_LOAD(os.fspath(path), device)


def _candidates(path):
    if CACHE_CANDIDATES is None:
        raise _codec_unset('CACHE_CANDIDATES')
    return CACHE_CANDIDATES(os.fspath(path))


def _write_name(path):
    """the CACHE_EXT name of an entry given in either container"""
    return _candidates(path)[0]


def _resolve_stat(path):
    """-> (the existing file of the entry, its stat), else (its CACHE_EXT name, None)"""
    for candidate in _candidates(path):
        st = _stat(candidate)
        if st is not None:
            return candidate, st
    return _write_name(path), None


def _resolve(path):
    """the existing file of the entry named `path` (either container), or None"""
    found, st = _resolve_stat(path)
    return found if st is not None else None


def _found(path):
    """the existing file of the entry, else its name to write (CACHE_EXT)"""
    return _resolve_stat(path)[0]


def _same_root_name(path):
    """the CACHE_EXT spelling of an entry IN THE ROOT THAT HOLDS IT -- what a reference to it records
    (hillview). Not the file it was found in: a .pth goes once its .hvz is
    published, and a recorded .pth (canonical_paths, persisted by capture_solve and the optimizer
    phase's request) would then name nothing. Not _write_name either: that is the PRIMARY root's name,
    and a value found in a read-only root (a local copy, an archive) need not exist there -- the optimizer child,
    with no roots, would look for it in the wrong place. Every reader resolves this at read time
    (_found), among the two containers in that same directory."""
    stem, ext = os.path.splitext(os.fspath(path))
    return path if ext == CACHE_EXT else stem + CACHE_EXT


def _durable(f):
    """hillview: DURABLE BEFORE IT IS RENAMED. os.replace makes a write atomic for readers, not for
    a power loss: the rename can reach the disk before the data does, and a box's reboot left zero-byte
    entries and sidecars under complete names (a run died on one, 10 h in). Flushed and fsynced
    here, before the rename; the directory is fsynced after its renames."""
    f.flush()
    os.fsync(f.fileno())


def _save_atomic(obj, path, content=True):
    """hillview: write beside, then rename -- a reader in another process never sees half a file
    (isfile() was true the moment it was opened). With `content` it is a CACHE ENTRY: stored by
    CACHE_DUMPS under a CACHE_EXT name, with `<path>.content` beside it (CONTENT_DIGEST above); False
    only for what is not a cache entry (a solve capture), which stays a torch pickle."""
    import io
    import fcntl
    import hashlib   # not a module import in sparse_ga: without this every forward write raises NameError
    import tempfile
    path = os.fspath(path)
    if content:
        for name, hook in (('CONTENT_DIGEST', CONTENT_DIGEST), ('CACHE_DUMPS', CACHE_DUMPS), ('CACHE_EXT', CACHE_EXT)):
            if hook is None:
                raise _codec_unset(name)
        if not path.endswith(CACHE_EXT):
            raise ValueError(f'{path}: a cache entry is named *{CACHE_EXT}')
        # a read-only root is never written: an entry goes under its write name only (in the primary)
        if os.path.abspath(_write_name(path)) != os.path.abspath(path):
            raise ValueError(f'{path}: an entry is written as {_write_name(path)} (a read-only cache root '
                             f'is never written)')
        identity = f'sha256:{CONTENT_DIGEST(obj)}\n'.encode()
        data = CACHE_DUMPS(obj)
    else:
        identity = None
        payload = io.BytesIO()
        torch.save(obj, payload)
        data = payload.getvalue()
    ram = os.environ.get('RECON_FORWARD_RAM_DIR')
    forward = content and 'forward' in os.path.normpath(path).split(os.sep)
    spill_floor = None
    if ram and forward:
        import shutil
        # Both settings are explicit: a missing/invalid floor is a configuration error. (The spill used
        # to re-deflate the torch zip; an .hvz is already compressed and is spilled as it is.)
        spill_floor = float(os.environ['RECON_FORWARD_DISK_FLOOR_GB']) * 2**30
        if not os.path.isabs(ram) or not np.isfinite(spill_floor) or spill_floor < 0:
            raise ValueError('forward spill requires an absolute RAM directory and a finite nonnegative disk floor')
    digest = hashlib.sha256(data).hexdigest()
    folder = os.path.dirname(path) or '.'
    # Serialize writers to this directory without leaving lock-file cache entries.
    directory = os.open(folder, os.O_RDONLY)
    temporary = []
    target = None
    published = False
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        old_target = os.readlink(path) if os.path.islink(path) else None
        spill = spill_floor is not None and shutil.disk_usage(folder).free < spill_floor
        sidecars = [(digest + '\n').encode()] + ([identity] if identity is not None else [])
        for number, blob in enumerate([data] + sidecars):
            fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.tmp.', dir=folder)
            temporary.append(tmp)
            if number == 0 and spill:
                os.close(fd)
                os.unlink(tmp)
                os.makedirs(ram, exist_ok=True)
                prefix = hashlib.sha256(os.path.abspath(path).encode()).hexdigest() + '.spill.'
                fd, target = tempfile.mkstemp(prefix=prefix, suffix=CACHE_EXT, dir=ram)
                # Never modify the old target while readers still hold its open file.
                with os.fdopen(fd, 'wb') as f:
                    f.write(blob)
                    _durable(f)
                os.symlink(target, tmp)
                continue
            with os.fdopen(fd, 'wb') as f:
                f.write(blob)
                _durable(f)
        # No stale digest may identify newly replaced data, even during a crash: both sidecars go
        # first and come back after the data, so a crash leaves one missing, never one wrong.
        for side in ('.sha256', '.content'):
            try:
                os.unlink(path + side)
            except FileNotFoundError:
                pass
        os.replace(temporary[0], path)
        published = True
        os.replace(temporary[1], path + '.sha256')
        if identity is not None:
            os.replace(temporary[2], path + '.content')
        os.fsync(directory)                 # the renames themselves (DURABLE, above)
        if old_target and ram:
            prefix = hashlib.sha256(os.path.abspath(path).encode()).hexdigest() + '.spill.'
            if (os.path.dirname(old_target) == os.path.normpath(ram)
                    and os.path.basename(old_target).startswith(prefix)):
                try:
                    os.unlink(old_target)
                except FileNotFoundError:
                    pass
    finally:
        for tmp in temporary:
            if os.path.lexists(tmp):
                os.unlink(tmp)
        if target is not None and not published:
            os.unlink(target)
        os.close(directory)


# hillview: CACHE PROVENANCE. The solver has a parameter that lets a run require cache artifacts of a
# particular provenance, for special comparison runs. The address stays content-only, so every ordinary run reuses everything; what made an
# entry is recorded BESIDE it, and only a run that REQUIRES a provenance ever reads that record.
#   * every file this process writes into the cache gets `<file>.prov.json`: the producer's identity
#     (checkpoint, forward version, device, precision) plus torch, run and time. An entry without
#     one is `legacy` -- made before this existed, provenance unknown -- and never satisfies a
#     requirement.
#   * with no requirement the hit path reads NO sidecar: the cache sits on a 9p mount that is worst
#     at small-file metadata (measured at 29-35 MB/s, the solver blocked in p9 for 35 min).
#   * with one, a pair whose default entry does not carry a matching record is computed and stored
#     as a VARIANT `<file>@<producer id>.hvz` beside the default, never over it. What prunes variant
#     sets: nothing yet -- by hand, `*@*.hvz` (the cache has no policy surface).
#   * STORAGE, EXPECTED ON FIRST USE AND NOT A LEAK: the first `require` run on a span writes only
#     variants, and a later `any` run on a cold span writes the defaults, so a span touched both
#     ways carries two full sets -- on a 40-panorama church ride, 5120 directed pairs twice.
PROV_SUFFIX = '.prov.json'
FORWARD_VERSION = 'mast3r-forward-1'   # bump when a cached output changes MEANING, not on refactors


def _prov_read(path):
    try:
        with open(path + PROV_SUFFIX) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def prov_satisfies(record, require):
    """a recorded producer meets every required key (a missing record never does)."""
    return record is not None and all(str(record.get(k)) == str(v) for k, v in require.items())


def prov_variant(path, vid):
    """the variant's name for an entry given in either container: written as CACHE_EXT, and resolved
    like any entry (a .pth variant made before the .hvz container is found by its key)"""
    name = _write_name(path)
    return _found(name[:-len(CACHE_EXT)] + f'@{vid}{CACHE_EXT}')


# hillview: the requirement keys that say HOW A PAIR WAS MATCHED and nothing about its
# pointmaps -- 'matcher' (hillview --nn_kernel fused: another NN kernel, near-ties may flip). A pointmap is
# checked without them and its record never carries them, so a run matching with another kernel re-matches
# against the pointmaps it finds, or that home holds (held_fwd), as a masked re-match does; its matches alone
# are the variant (labelled whole, the fused arm re-ran the model on every pair and
# relayed every pointmap home).
MATCH_ONLY_KEYS = ('matcher',)


def forward_part(d):
    """a requirement or record without the MATCH_ONLY_KEYS: what a pointmap is judged by, and carries"""
    return {k: v for k, v in (d or {}).items() if k not in MATCH_ONLY_KEYS}


def resolve_cache_paths(default, require, vid):
    """(path1, path2, corres, corres_reverse) -> the paths this pair uses, and how they were chosen.

    'default' without a requirement (nothing read beyond what the hit path always did); with one,
    'default-verified' when both pointmaps and a correspondence file in either direction carry a
    matching record, else 'variant': the variant of all four under `vid` -- this producer's own, or,
    in a run that never makes an entry (FORWARD_MISSING 'refuse'), the one its caller names, whose
    files must then each carry a record meeting the requirement (CacheProvenanceMismatch). The
    pointmaps are judged by the requirement without its MATCH_ONLY_KEYS: when they meet that and only
    the matches fail, 'variant-matches' -- the default pointmaps and the variant correspondences."""
    if not require:
        return default, 'default'
    # the records are read beside the file that IS the entry -- a .pth made before the .hvz container carries
    # its own (read on the .hvz name only, every unconverted span recomputed a variant)
    default = tuple(_found(x) for x in default)
    p1, p2, pc, pc2 = default
    fwd_require = forward_part(require)
    fwd_ok = not fwd_require or (prov_satisfies(_prov_read(p1), fwd_require)
                                 and prov_satisfies(_prov_read(p2), fwd_require))
    if fwd_ok and (prov_satisfies(_prov_read(pc), require) or prov_satisfies(_prov_read(pc2), require)):
        return default, 'default-verified'
    if fwd_ok and fwd_require != require:
        # only a MATCH-ONLY key separates them: the pointmaps are this run's, the matches another kernel's
        variant, how = (p1, p2) + tuple(prov_variant(x, vid) for x in (pc, pc2)), 'variant-matches'
    else:
        # ALL FOUR, deliberately, even when only the correspondences failed: a pair's pointmaps and its
        # matches are one inference, and a pair read half from one producer and half from another is
        # exactly the mixed input a requirement exists to exclude
        variant, how = tuple(prov_variant(x, vid) for x in default), 'variant'
    if FORWARD_MISSING == 'refuse':
        # A RUN THAT NEVER MAKES AN ENTRY READS ANOTHER PRODUCER'S VARIANTS (a home CPU solve
        # reading a GPU box's bf16 matches; hillview --cache_read_vid). Their name is a hash of the
        # producer's identity and says nothing checkable on its own, so each file that is there must show
        # a record meeting the requirement. One without (not yet relayed, or made by another identity)
        # stops the run by name: read anyway, it would be exactly the mixed input a requirement excludes.
        # A file that is NOT there is left to the missing-pair path (ForwardCacheMissing)
        bad = [x for x, d, r in zip(variant, default, (fwd_require, fwd_require, require, require))
               if x != d and _stat(x) is not None and not prov_satisfies(_prov_read(x), r)]
        if bad:
            raise CacheProvenanceMismatch(
                f'{len(bad)} variant file(s) under @{vid} do not carry a record meeting {require}: '
                + ', '.join(f'{x} ({_prov_read(x)!r})' for x in bad))
    return variant, how


class CacheProvenanceMismatch(RuntimeError):
    """FORWARD_MISSING is 'refuse' and a variant this run reads lacks a record meeting its requirement."""


def _prov_write(path, record):
    tmp = f'{path}{PROV_SUFFIX}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(record, f, sort_keys=True)
        _durable(f)                         # a zero-byte record survived the same reboot (_durable)
    os.replace(tmp, path + PROV_SUFFIX)


def _stat(path):
    try:
        return os.stat(path)
    except OSError:
        return None


# hillview: what a view's mask identity may look like -- a sha256 hex digest of the
# record hillview composed it from (recon_pair_masks.mask_identity), or MASK_IDENTITY_NONE for a
# view nothing masks. Anything else is a caller passing bytes or a path where a name belongs.
MASK_IDENTITY_NONE = 'none'


def _check_identity(ident):
    if ident == MASK_IDENTITY_NONE:
        return ident
    if (not isinstance(ident, str) or len(ident) != 64
            or any(c not in '0123456789abcdef' for c in ident)):
        raise ValueError(f'a mask identity is a sha256 hex digest or {MASK_IDENTITY_NONE!r}, '
                         f'not {ident!r}')
    return ident


def _masked_matching_paths(cache_path, idx1, idx2, identities, desc_conf, subsample, seed_fill,
                           matching_kw):
    """Share matches only when BOTH views' mask IDENTITIES and every matching setting agree.

    Pointmaps omit descriptors, so different masks still require another inference. Cache
    completed masked matches instead: identical-mask experiments reuse the expensive work
    without retaining descriptor tensors for thousands of pairs. Keep this outside corres_conf=*
    so diagnostics cannot mistake another mask variant for this run's raw correspondences.

    hillview: keyed by each view's mask IDENTITY, not by the mask's bytes. Same-keyed
    semantic masks are interchangeable: a GPU box and a CPU box make a
    few boundary pixels differently under one mask key, as their forward passes differ by ~2e-4
    under one forward key, and a byte key made the two miss each other's matches -- a bench
    solving with --forward_cache required refused the GPU box's matches as missing. The identity
    names what the mask was made FROM (the semantic key, the rung and pair decision, the bytes of
    the deterministic layers); the caller composes it. v3: v2 entries are keyed by bytes and must
    not be read as identities.
    """
    import hashlib
    import json

    settings = json.dumps(dict(desc_conf=desc_conf, subsample=subsample,
                               seed_fill=bool(seed_fill), matching_kw=matching_kw),
                          sort_keys=True, separators=(',', ':'))
    variant = hashlib.sha256(settings.encode()).hexdigest()
    root = os.path.join(cache_path, 'matching', 'masked-v3', variant)
    if len(identities) != 2:
        raise ValueError(f'one mask identity per view, got {len(identities)}')
    key1, key2 = (f'{idx}-{_check_identity(ident)}' for idx, ident in zip((idx1, idx2), identities))
    return (os.path.join(root, f'{key1}_{key2}{CACHE_EXT}'),
            os.path.join(root, f'{key2}_{key1}{CACHE_EXT}'))


# the decorator belongs HERE: once the pair-lock class was inserted between it and this def,
# so the forward passes ran with autograd on and every uncached pair died in extract_correspondences
# ("Can't call numpy() on Tensor that requires grad")
LAST_FORWARD_PROFILE = {}  # one solve per process; read by the Hillview event adapter
# hillview: WHAT A MISSING FORWARD PASS MEANS. 'compute' (the default) runs the model for it.
# 'refuse' is for a solve whose forward passes were made elsewhere (a GPU box matches, the bench
# solves; --forward_cache required): every call collects the pairs that are not cached -- under the
# same paths a computing pass would look up, masked matches included -- and raises once it has
# looked at all of them, so no model is loaded and no CPU spends days matching by accident. One
# setting for the process, because every forward pass of a run goes through this function: the
# solve, the unpaired gate's pre-pass and the expansion rounds alike.
FORWARD_MISSING = 'compute'
# hillview: forward_mast3r takes held_fwd, the pointmap stems another cache (home) holds, and
# keeps a re-inferred pair's pointmaps there instead of writing them; the caller checks this before passing it
HELD_FWD = True
# hillview: sparse_scene_optimizer takes pose_fixed (cameras held at pose_init 'poses'). The caller refuses a checkout without it.
POSE_FIXED = True
# hillview: sparse_scene_optimizer takes weak_views + step_links and holds weak
# photographs with gps_rel_frac (consecutive steps against the GPS's own) and weak_mid_frac (a weak camera to its
# neighbours' midpoint, neighbours detached). The caller refuses a checkout without it.
WEAK_HOLDS = True
# hillview: sparse_scene_optimizer takes alt_step_links, gauge_tilt_from
# ('height_steps': the GPS gauge's down from the terrain's height steps, heights then held as steps only) and
# alt_step_frac / alt_step_huber_m (each step's solved height change held to the terrain's). The caller refuses a
# checkout without it.
ALT_STEPS = True
# hillview: height_step_down takes floor_m, and sparse_scene_optimizer gauge_tilt_step_floor_m (required under
# 'height_steps'): the accuracy a terrain step is credited with at best, so the residuals the tilt fit has absorbed
# cannot lower it. The caller refuses a checkout without it.
GAUGE_TILT_STEP_FLOOR = True
# hillview: sparse_scene_optimizer takes `restore`, a finished run's final state in
# place of the two stages, checked against this run's inputs (a dense arm, --solve_from). The caller refuses a checkout
# without it.
RESTORE = True
# hillview: REDUCED-PRECISION INFERENCE, for an A/B -- None (fp32, the
# default) or a torch dtype (torch.bfloat16, torch.float16) the encoder and the decoder run under
# autocast in. The heads stay fp32 as they always were (their autocast is off and their tokens are
# cast up), so pointmaps, confidences and descriptors leave in fp32 and matching is unchanged. Set by
# hillview's --autocast, which also labels the entries' provenance precision and refuses to run
# unless that precision is REQUIRED -- a reduced-precision entry is never written under the default
# key. An A/B, not a default: judged verdicts decide.
INFERENCE_AUTOCAST = None
# hillview: THE CANARY. A key that misses an input -- a mask it does not
# carry, an entry made by other code under the same name -- has an honest provenance record, and only
# recomputing the entry shows it. So a share of the cache HITS is recomputed here (the model and the matching,
# nothing written into a cache) and compared with what is stored. A check that passes changes no solve: the solve
# reads the STORED entry. A check that does not pass is a WARNING, never a stop: a stop halted the READER for a
# fault of the entry's WRITER, until a human cleared it -- one old entry whose second view had no confident pixel
# left a box idle with 13 runs queued. The
# pair is solved on this run's own recompute instead, kept under CANARY_FRESH_ROOT (the run's dir, never a cache),
# so no result depends on an entry under suspicion; every check goes to CANARY_SINK (reconstruct: canary.jsonl,
# gathered into the bench's ledger, where a writer going gross is stopped at its claim).
#   * WHICH HITS: sha256(CANARY_SALT | the entry's matches name) under CANARY_RATE -- the same pairs on a
#     re-run, whatever order they come in; a pair is checked once per process (CANARY_RECORD['done']).
#   * AT MOST CANARY_CAP A RUN (None: the rate alone). A rate scales with the run and detection does not: K checks
#     catch a fault hitting a share f of the entries with 1 - (1 - f)^K -- 96.6 % at f = 10 %, K = 32 -- and a code
#     change that breaks a key usually breaks it for every entry of its kind. One run (10,776 hits at 0.1) would
#     have recomputed ~1,078 pairs on a CPU box, ~30 h, and was cancelled.
#     Under a cap the draw is by the pair's two view keys (canary_pick), known before any lookup -- one run holds
#     one entry per directed pair, so the same entries -- and the pairs drawn at the rate with the SMALLEST values
#     are checked, as many as the run's cap has left: spread over the run (the first K in pair order
#     would only ever sample the start of a span), and the same pairs on a re-run. Two things a reader of the
#     records should know: the cap is spent per CALL, greedily -- a run with several
#     forward_mast3r calls (adaptive rounds, a second pass) spends it on the first call's smallest draws, so later
#     calls' pairs are under-sampled; and the picks are made before the lookup, so a picked pair that turns out a
#     MISS is computed, not checked -- a run with many misses checks fewer than its cap (CANARY_RECORD['cap']'s
#     `picked` against the checked count says so). All-hit runs, the expensive case, are unaffected.
#   * HOW COMPARED (canary_compare), by the stored file's writer record: ON A CPU, a writer whose `compute`
#     fingerprint and identity equal this box's (CANARY_COMPUTE, the record's CANARY_IDENTITY_KEYS) owe
#     the SAME BITS -- every array, matches as a set of rows -- and the score within CANARY_SCORE_RTOL. On one
#     CPU box two processes at 8 threads wrote 57/57 identical files; 8 against 4 threads changed only
#     the score's floats (a thread-dependent mean and sum). NOT ON A CARD: on an RX 580 (ROCm 5.7) two
#     processes differed in every file, pointmaps by up to 4.5e-3 on values to 27 and a match set by one
#     match, and more with cudnn deterministic; a CUDA box runs with cudnn.benchmark on
#     (set by an import), so its kernels are chosen per process too. Every other case is DIVERGENCE: pointmaps within
#     CANARY_MAX_PTS_REL (median |dX| over median |X|, over the TRUSTED pixels: below), matches differing in no
#     more rows than CANARY_MIN_JACCARD
#     allows of their union or CANARY_MATCH_SLACK_ROWS, whichever is more -- a row crossing the matcher's threshold
#     is 1 % of a 150-row pair. The RX 580 against a CPU, 181 pairs: pointmaps median 4.3e-6, matches 1,383,148
#     against 1,383,154; the RX 580 against itself, two processes: pointmaps <= 6.7e-6 (2.0e-4 with cudnn
#     deterministic, noisier on ROCm), matches 1-2 rows of ~1430. An
#     entry DERIVED from a mask variant (its record's derived_from) differs by design: measured, never failed;
#     so does a writer of ANOTHER recorded PRECISION (tf32 read by fp32, ...: canary_tier), until a floor per
#     precision pair is calibrated (one run's 183 tf32 records are the first sample).
#   * TRUSTED PIXELS: a pointmap is compared only where the network trusts it -- either
#     side's confidence at least CANARY_TRUST_CONF; either side, so an entry that LOST its confidence is still
#     caught. A view with fewer than CANARY_TRUST_MIN_FRAC trusted is not compared: the network's guess at a view
#     it cannot see is unconstrained, and showed the old-vs-current forward's gap 10x larger than the other view
#     (one old entry: 1.2e-3 on a view 99.7 % at confidence 1.000, 1e-4 on its partner; the same pair
#     from a CUDA fp32 rental agreed with a CPU box's). Weighed by canonical views at ~0.001 a pixel, it matches
#     nothing.
#   * THE VERDICT: 'ok', 'warn' (past its tier's floor: not bit-equal where bits are owed, past CANARY_MAX_PTS_REL
#     or the row slack in divergence) or 'gross' -- a stored value that is not finite, trusted pointmaps whose
#     cosine is below CANARY_GROSS_COS, a match set past the slack AND below CANARY_GROSS_JACCARD, or another
#     shape: not the same values at all, whatever the tier (a tf32 writer is measured, never excused from this).
#     A derived (mask-variant) match set differs by design and is never gross on its rows.
#   * THE MASK CHECK (CANARY_MASK_CHECK), on every masked hit and needing no model: fast_reciprocal_NNs never
#     matches a masked pixel on either side, so a stored match on the reader's own mask proves the entry was
#     made under another one. Counted inside the mask eroded by CANARY_MASK_ERODE px, since two producers'
#     same-key masks may differ at the rim; it warns on a pair only past CANARY_MASK_WARN of its matches, and
#     None (the default) measures without warning until real spans say where the rim noise ends. It has no
#     recompute to solve on (no model): a warned hit is still read, and the warning says so.
# Each check goes to CANARY_RECORD (and CANARY_SINK, as it is made); a warned pair's fresh values are in
# LAST_FORWARD_PROFILE['canary_fresh_pairs'].
CANARY_RATE = 0.0
CANARY_SALT = ''
CANARY_CAP = None
CANARY_COMPUTE = None
CANARY_IDENTITY_KEYS = ('checkpoint', 'forward_version', 'device', 'precision', 'kernels', 'matcher')
CANARY_SCORE_RTOL = 1e-6
CANARY_MAX_PTS_REL = 1e-3
CANARY_MIN_JACCARD = 0.99
CANARY_MATCH_SLACK_ROWS = 4
CANARY_MASK_CHECK = False
CANARY_MASK_ERODE = 3
CANARY_MASK_WARN = None
CANARY_TRUST_CONF = 1.5
CANARY_TRUST_MIN_FRAC = 0.01
CANARY_GROSS_COS = 0.99
CANARY_GROSS_JACCARD = 0.5
CANARY_FRESH_ROOT = None     # where a run keeps its recompute of a pair that did not pass: required with a rate
# hillview: A RESTORE READS WHAT ITS SOURCE READ. The canary is salted per run,
# so the source solved some pairs on its own recompute and an arm would solve others on its own: other matches, so
# other anchors, though every input is the same. SOURCE_FRESH maps each directed pair (idx1, idx2) the SOURCE's canary
# solved on its recompute to its three files in the source's run dir (forward 1, forward 2, matches): this run reads
# those for it, as the source did, and its canary does not check it. CANARY_SOLVE_FRESH False: a pair this run's
# canary does not pass keeps the stored files, which the source read (the check is still recorded and warned).
# Both are set by a restoring caller and checked by prepare_canonical_data(restore=...); an ordinary run leaves them.
# SOURCE_READS: every OTHER directed pair (idx1, idx2) -> the forward file the source read for it,
# located by the caller from the source's own read record (its pointmap file's mtime and size, in any cache root). A
# key's copies can differ by root (a pull put a CPU recompute of one in front of the copy the source had
# read), so resolving the name afresh is not reading what the source read. This run reads
# those files for the pair's two pointmaps; its matches are resolved as before, and a pair it would compute is refused.
SOURCE_FRESH = None
SOURCE_READS = None
CANARY_SOLVE_FRESH = True
# hillview: A HELD SUBSET. A judged solve's views
# leave the optimiser and enter as FIXED points. The caller sets HELD_TARGETS = {view index into imgs: (pixels (N, 2) at
# the loaded size, points (N, 3) in the solve frame, confidences (N,))}: each a new view's match on a held view, the
# held point read from its source's final depth (joinkit) through the source's camera. Stage 1 holds each point to its
# pixel's ray (held_ray_closest: a 3-D distance, as loss_3d's), stage 2 reprojects it (held_reproj, as loss_2d's), both
# INSIDE the stage's own confidence-weighted mean: a target is a correspondence whose other end is fixed. The frame and
# the scale are then the source's: global_scaling is 1 (as pose_fixed), and FIXED_GAUGE = (s, R, t), the source's
# alignment (solve frame -> ENU), stands for every GPS term's refit gauge (fixed_gauge). Neither goes with a restore or
# pose_fixed. An ordinary run leaves both None.
HELD_TARGETS = None
FIXED_GAUGE = None
CANARY_SINK = None           # callable(result) for every check as it is made, under the sweep's lock
# WHO WROTE IT DECIDES HOW OFTEN: a writer the bench's ledger has seen enough of --
# 200 checks, no warn or gross in 7 days, its variant in CANARY_SETTLED -- is drawn at CANARY_RATE; any other (new,
# flagged, never seen) at CANARY_CAUTIOUS_RATE. A pair is cautious if ANY of its stored files is. CANARY_VARIANT_OF
# (hillview's recon_canary.variant_id) names a writer from its record. None: CANARY_RATE for every pair (a run
# without the bench's ledger)
CANARY_SETTLED = None
CANARY_CAUTIOUS_RATE = None
CANARY_VARIANT_OF = None
CANARY_RECORD = {'results': [], 'mask_checked': 0, 'mask_on': [], 'done': set(), 'fresh': {}}


def _matching_score(C, corres):
    """a pair's stored score: (geometric mean of its four confidence maps' means, sum of match confidences,
    match count) -- one definition for the entry written and the canary comparing with it"""
    C11, C21, C22, C12 = C
    conf_score = (C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()
    return (float(conf_score), float(corres[2].sum()), len(corres[2]))


def canary_value(name):
    """the draw value of `name` in [0, 1): a pure function of (CANARY_SALT, name)"""
    h = hashlib.sha256(f'{CANARY_SALT}|{name}'.encode()).digest()
    return int.from_bytes(h[:8], 'big') / 2.0 ** 64


def canary_drawn(name):
    """whether the entry named `name` is a canary at CANARY_RATE"""
    return CANARY_RATE > 0 and canary_value(name) < CANARY_RATE


def canary_ceiling():
    """the highest rate any pair can be drawn at: a draw value at or above it is never a canary, whoever wrote it"""
    return CANARY_RATE if CANARY_SETTLED is None else max(CANARY_RATE, CANARY_CAUTIOUS_RATE)


def canary_rate_of(writers):
    """the rate a pair is drawn at, from the writer records of its stored files (None: a file without one) --
    CANARY_RATE if every one is a SETTLED variant, else the cautious rate"""
    if CANARY_SETTLED is None:
        return CANARY_RATE
    if all(CANARY_VARIANT_OF(w) in CANARY_SETTLED for w in writers):
        return CANARY_RATE
    return max(CANARY_RATE, CANARY_CAUTIOUS_RATE)


def canary_check_settings():
    """refuse a half-set ledger: CANARY_SETTLED needs the cautious rate and the hook that names a writer"""
    if CANARY_SETTLED is None:
        return
    if not isinstance(CANARY_SETTLED, frozenset) or not all(isinstance(v, str) for v in CANARY_SETTLED):
        raise ValueError(f'sparse_ga.CANARY_SETTLED: a frozenset of writer variants, or None -- not {CANARY_SETTLED!r:.80}')
    if (isinstance(CANARY_CAUTIOUS_RATE, bool) or not isinstance(CANARY_CAUTIOUS_RATE, (int, float))
            or not 0 < CANARY_CAUTIOUS_RATE <= 1):
        raise ValueError(f'sparse_ga.CANARY_CAUTIOUS_RATE {CANARY_CAUTIOUS_RATE!r}: a rate in (0, 1] with CANARY_SETTLED')
    if not callable(CANARY_VARIANT_OF):
        raise ValueError('sparse_ga.CANARY_SETTLED needs CANARY_VARIANT_OF -- the hook that names a writer variant '
                         "from its record (hillview: recon_canary.variant_id)")


def canary_pick(pairs, rate_of):
    """UNDER CANARY_CAP: the pairs ('<view key 1>-<view key 2>') this forward_mast3r call may check, as many as
    the run's cap has left (CANARY_RECORD['done'] holds what earlier calls checked). A pair is drawn when its value
    is below ITS rate (rate_of(a, b): its writers', canary_rate_of), and the drawn are taken by value OVER rate,
    smallest first -- by value alone every base-rate draw (below CANARY_RATE) would come before every cautious one,
    and on a big run the base draws would fill the cap: a new writer would get no more checks than a settled one.
    rate_of is asked only for pairs under canary_ceiling(). Each call's numbers go to CANARY_RECORD['cap']."""
    if isinstance(CANARY_CAP, bool) or not isinstance(CANARY_CAP, int) or CANARY_CAP < 1:
        raise ValueError(f'sparse_ga.CANARY_CAP {CANARY_CAP!r}: a positive number of canaries a run, or None')
    left = max(CANARY_CAP - len(CANARY_RECORD['done']), 0)
    keyed = {}
    for a, b in pairs:
        keyed.setdefault(f"{hash_md5(a['instance'])}-{hash_md5(b['instance'])}", (a, b))
    top, drawn, asked = canary_ceiling(), [], 0
    for k, (a, b) in keyed.items():
        v = canary_value(k)
        if v >= top:
            continue
        r = CANARY_RATE if CANARY_SETTLED is None else rate_of(a, b)
        asked += CANARY_SETTLED is not None
        if v < r:
            drawn.append((v / r, k, r))
    drawn.sort()
    picked = {k for _w, k, _r in drawn[:left]}
    CANARY_RECORD.setdefault('cap', []).append(dict(
        cap=CANARY_CAP, pairs=len(keyed), drawn=len(drawn), picked=len(picked), writers_read=asked,
        picked_cautious=sum(1 for _w, k, r in drawn[:left] if r > CANARY_RATE)))
    return picked


def canary_tier(writer, own):
    """'derived' (a mask variant's entry, measured only), 'precision' (a writer of ANOTHER recorded precision,
    measured only), 'bits' (a CPU box, and the same identity and compute fingerprint as the writer's: the same
    bits are owed) or 'divergence' (anything else -- a card, another writer, a writer without a record)"""
    if writer is not None and 'derived_from' in writer:
        return 'derived'
    if (writer is not None and own is not None and writer.get('precision') is not None
            and own.get('precision') is not None and str(writer['precision']) != str(own['precision'])):
        # ANOTHER PRECISION IS A LABELLED DIFFERENCE, NOT A KEY'S: a CPU box's first cautious job
        # stopped on 183 of 461 checks, every one a CUDA tf32 writer (one rental's run, relabelled tf32)
        # read by the CPU's fp32: matches Jaccard 0.925-0.979, pointmaps median 1.0e-3 to 7.6e-3 -- TF32's known,
        # measured-harmless gap, under the default key by decision. fp32's floor stopped the box for it. Measured
        # and recorded, never failed, until a floor per precision pair is calibrated from records like these. BOTH
        # must say their precision: a record without one is the least known of all, and keeps the floor
        return 'precision'
    if (writer is not None and own is not None and CANARY_COMPUTE is not None and own.get('device') == 'cpu'
            and writer.get('compute') == CANARY_COMPUTE
            and all(str(writer.get(k)) == str(own.get(k)) for k in CANARY_IDENTITY_KEYS)):
        return 'bits'
    return 'divergence'


def _canary_arrays(x):
    out = []

    def walk(v):
        if isinstance(v, (tuple, list)):
            for u in v:
                walk(u)
        elif isinstance(v, torch.Tensor):
            out.append(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            out.append(v)
    walk(x)
    return out


def _canary_pointmap_view(x, cx, y, cy):
    """one view's pointmap, fresh (x, cx) against stored (y, cy), over its TRUSTED pixels: either side's
    confidence at least CANARY_TRUST_CONF. -> {trusted_frac} alone when too few are trusted to compare."""
    m = (cx >= CANARY_TRUST_CONF) | (cy >= CANARY_TRUST_CONF)
    frac = float(m.mean()) if m.size else 0.0
    if frac < CANARY_TRUST_MIN_FRAC:
        return {'trusted_frac': frac}
    xs, ys = x[m].astype(np.float64), y[m].astype(np.float64)
    rel = float(np.median(np.linalg.norm(xs - ys, axis=-1))
                / max(float(np.median(np.linalg.norm(ys, axis=-1))), 1e-12))
    cos = float((xs * ys).sum() / max(float(np.linalg.norm(xs) * np.linalg.norm(ys)), 1e-300))
    conf_rel = float(np.median(np.abs(cx[m].astype(np.float64) - cy[m]) / np.maximum(np.abs(cy[m]), 1e-12)))
    return {'trusted_frac': frac, 'pts_rel': rel, 'cos': cos, 'conf_rel': conf_rel}


def canary_compare(kind, fresh, stored, tier):
    """-> (verdict, detail) for one file: verdict 'ok', 'warn' or 'gross' (the comment above CANARY_RATE). kind
    'pointmaps': (X, C, X, C), each pointmap over its trusted pixels; 'matches': (score, (xy1, xy2, conf)), compared
    as SETS of rows (a pair matched in the other order lists them in another order)."""
    if kind == 'pointmaps':
        a, b = _canary_arrays(fresh), _canary_arrays(stored)
        if [x.shape for x in a] != [y.shape for y in b]:
            return 'gross', {'shapes': [[list(x.shape) for x in a], [list(y.shape) for y in b]]}
        if not all(np.isfinite(x).all() for x in a):
            raise FloatingPointError('canary: THIS box\'s recompute is not finite -- its own forward is broken, '
                                     'not the entry it was to check')
        finite = all(np.isfinite(y).all() for y in b)
        bit_equal = all(np.array_equal(x, y) for x, y in zip(a, b))
        views = [_canary_pointmap_view(a[k], a[k + 1], b[k], b[k + 1]) if finite else {}
                 for k in (0, 2)]                                   # the two pointmaps, each with its C
        compared = [v for v in views if 'pts_rel' in v]
        detail = {'bit_equal': bit_equal, 'finite': finite, 'views': views,
                  'trusted_px_frac': min((v['trusted_frac'] for v in views), default=None) if finite else None,
                  'cos_min_trusted': min((v['cos'] for v in compared), default=None),
                  'pts_rel_max_trusted': max((v['pts_rel'] for v in compared), default=None)}
        if not finite or (compared and detail['cos_min_trusted'] < CANARY_GROSS_COS):
            return 'gross', detail
        if tier == 'bits':
            return ('ok' if bit_equal else 'warn'), detail
        if tier in ('derived', 'precision') or not compared:
            return 'ok', detail
        return ('ok' if detail['pts_rel_max_trusted'] <= CANARY_MAX_PTS_REL else 'warn'), detail
    (fs, (f1, f2, fc)), (ss, (s1, s2, sc)) = fresh, stored
    f1, f2, fc, s1, s2, sc = (np.asarray(_canary_arrays(v)[0]) for v in (f1, f2, fc, s1, s2, sc))
    finite = bool(all(np.isfinite(np.asarray(v, dtype=np.float64)).all() for v in (s1, s2, sc))
                  and all(np.isfinite(float(x)) for x in ss))

    def rows(xy1, xy2):
        return np.concatenate([xy1.reshape(-1, 2), xy2.reshape(-1, 2)], 1).astype(np.int64)
    rf = rows(f1, f2)
    rs = rows(s1, s2) if finite else np.zeros((0, 4), np.int64)
    of, os_ = np.lexsort(rf.T[::-1]), np.lexsort(rs.T[::-1])
    same_set = finite and rf.shape == rs.shape and np.array_equal(rf[of], rs[os_]) and np.array_equal(fc[of], sc[os_])
    score_ok = finite and len(fs) == len(ss) and all(
        abs(float(x) - float(y)) <= CANARY_SCORE_RTOL * max(abs(float(x)), abs(float(y))) for x, y in zip(fs, ss))
    kf, ks = {tuple(r) for r in rf.tolist()}, {tuple(r) for r in rs.tolist()}
    union, differ = len(kf | ks), len(kf ^ ks)
    jac = (union - differ) / union if union else 1.0                # two empty sets are the same set
    detail = {'bit_equal': bool(same_set), 'score_ok': bool(score_ok), 'finite': finite, 'n_fresh': len(rf),
              'n_stored': len(rs), 'jaccard': jac, 'rows_differ': differ,
              'score_fresh': [float(x) for x in fs], 'score_stored': [float(x) for x in ss]}
    slack = max(CANARY_MATCH_SLACK_ROWS, (1 - CANARY_MIN_JACCARD) * union)
    if not finite or (tier != 'derived' and differ > slack and jac < CANARY_GROSS_JACCARD):
        return 'gross', detail
    if tier == 'bits':
        return ('ok' if same_set and score_ok else 'warn'), detail
    return ('ok' if tier in ('derived', 'precision') or differ <= slack else 'warn'), detail


class EmptyCacheEntry(RuntimeError):
    """hillview: cache files of size zero met at the hit decision, every one named"""


class ForwardCacheMissing(RuntimeError):
    """FORWARD_MISSING is 'refuse' and some of the pairs' forward passes are not cached."""


class NonFiniteSolve(FloatingPointError):
    """hillview: the optimiser met a nonfinite loss or gradient, so the solve FAILED.

    Before, a NaN loss went through backward and Adam.step, then only broke that stage's loop:
    stage 2 ran on from the corrupted parameters and the function returned NaN poses as a
    result (an infinite loss or gradient was not even
    noticed). Raised before the step it would have poisoned, naming the stage and iteration.
    There is no restore-the-last-good-iterate policy: a solve that produced a NaN is a failed
    solve, and the run says so instead of shipping whatever came before it."""

    def __init__(self, stage, iteration, what, detail):
        self.stage, self.iteration, self.what, self.detail = stage, iteration, what, detail
        super().__init__(f'{stage}, iteration {iteration}: nonfinite {what} ({detail})')

    def __reduce__(self):
        return (type(self), (self.stage, self.iteration, self.what, self.detail))


# hillview: `corr_masks` is a callable (instance1, instance2) -> (mask1, mask2, identity1,
# identity2), not a per-image dict. The pair decides what is masked: hillview drops a
# view's FOLIAGE only when both photographs of the pair mask it, so one view carries different
# masks against different partners. The caller checks this marker and refuses an older checkout,
# whose `.get` on a callable would fail without saying why.
CORR_MASKS_BY_PAIR = True
# ...and the masked matches are keyed by the two IDENTITIES, never by the masks' bytes
# (_masked_matching_paths, matching/masked-v3). The caller refuses a checkout without this marker:
# that one unpacks two items from the callable and keys by bytes.
CORR_MASKS_BY_IDENTITY = True


# hillview: PAIRS PIPELINED INSIDE ONE WORKER, behind a knob that is OFF by default. The
# operator: "workers just need to parallelize their own pairmatching ... spawn (and stop) threads on
# command, then we can measure total throughput" -- and a second worker on a rental's last job "just
# sounds like failure to parallelize one worker". Per computed pair forward_mast3r runs the model
# (0.17-0.29 s on a 3090), matches (0.56-0.66 s before fast_nn's 0610084) and writes (0.04-0.30 s),
# strictly one after the other, so the card waits through the matching and the writes of every pair
# and only a SECOND PROCESS (two models, twice the RAM) fills it.
#
# The pipeline: the main thread looks a pair up (under its _PairLock, exactly as before) and runs the
# model; a POST thread then matches on its OWN CUDA stream, writes the pair's files, releases the lock
# and fills the pair's result slot -- while the main thread already looks up and infers the next pair.
# MAST3R_PAIR_POST_THREADS=0 (the default, on every device) is today's sequential loop, unchanged;
# N > 0 runs N post threads with at most MAST3R_PAIR_DEPTH pairs in flight. It ships
# OFF on cuda as well until it has passed on a card -- a CPU test exercises the slots, the
# locks and the queues but never torch.cuda.Stream, the event wait or record_stream, which are the
# parts that corrupt SILENTLY. A rental sets the knob explicitly.
#
# MAST3R_PAIR_CONTROL names a JSON file ({"post_threads": n, "depth": d}) re-read BETWEEN pairs, never
# mid-pair, so threads can be added or drained while a job runs and the throughput watched; a thread
# told to go finishes its current pair first. It cannot switch the pipeline off mid-call (n >= 1 there).
# hillview: LET THE CARD GO BETWEEN PAIRS. A box may share its 8 GB card with a desktop,
# and a stopped process (SIGSTOP) keeps every byte of VRAM it holds -- the driver evicts the DESKTOP's buffers
# instead (measured). GPU_YIELD, when set, is called by forward_mast3r in its main thread at
# the top of every pair, with (model, device, quiesce, held): quiesce() waits until no pair is in flight, `held`
# is the dict of view encodings kept for later pairs. The caller decides whether to yield -- move the model and
# the encodings off the card, free the cache, wait, bring them back -- and returns when the pair may run. None
# (the default) is today's loop, unchanged.
GPU_YIELD = None
# hillview: objective metrics of the card's performance, to be correlated with how much a desktop sharing it loads
# it: PAIR_TIMING, when set, is called once for every pair this call
# INFERRED (computed or canaried), after it is complete, with {forward_s, post_s, match_s, computed, canary, idx1, idx2,
# post_threads}: forward_s the model's two decoder passes on the device (the same span as the 'model' part, which waits
# for its own inference), match_s the reciprocal matching, post_s all of _post (saves, matching, canary). A CANARY pair's
# matching runs inside the canary check: its match_s is None and post_s holds it -- None is not "no matching". The
# forward of a pair is fixed work, so forward_s is the device's speed under whatever else uses it -- INCLUDING this
# run's own matching on the post threads' stream when pipelined: post_threads (the pipeline's
# target, 0 sequential) is in the record, so a comparison across runs or against the desktop's load compares like with
# like. Called from the thread that completed the pair (a post thread when pipelined), under no lock: it must be quick
# and thread-safe. The bar's postfix shows the medians of the last 50 pairs either way. None (the default) costs nothing.
PAIR_TIMING = None
# hillview (a pair's matches should look somewhat natural): MATCH_QUALITY, when
# set, is called with (X11, X21, corres) -- on the CPU -- as a pair's matches are written, and what it returns goes into
# THAT entry's record as `match_quality`, beside the writer's identity, never into a key. hillview's
# recon_match_quality scores the matches against the pair's own pointmaps, as a displacement field and against one
# epipolar geometry: a reduction that answers a self-consistent WRONG index (gfx803's GEMM, rocBLAS #1218) passes
# fast_nn._verified, not these. None (the default): records as before.
MATCH_QUALITY = None


class _PairPipeline:
    def __init__(self, post_threads, depth, control):
        import threading, queue
        self.control, self.target, self.depth = control, max(1, post_threads), max(1, depth)
        self.q, self.lock, self.cv = queue.Queue(), threading.Lock(), threading.Condition()
        self.in_flight, self.threads, self.errors = 0, [], []
        self.busy_s = self.idle_s = self.slot_wait_s = 0.0
        self.max_threads = self.target
        self._threading = threading
        self._top_up()

    @staticmethod
    def from_env():
        # THE CONTROL FILE DECIDES AT A CALL'S START TOO. A rental's workers get their
        # environment once, at container start, and the pipeline must stay off until the card's correctness
        # check (gpu_ab.py) says PASS -- so a box is rented with MAST3R_PAIR_CONTROL set and no file there
        # (= the env knobs, sequential by default), and the file is written after PASS: the next call reads
        # it here, and post_threads 0 in it means sequential. Without the env var nothing changes.
        control = os.environ.get('MAST3R_PAIR_CONTROL') or None
        n = int(os.environ.get('MAST3R_PAIR_POST_THREADS', '0') or 0)
        depth = int(os.environ.get('MAST3R_PAIR_DEPTH', '2') or 2)
        if control and os.path.isfile(control):
            try:
                with open(control) as f:
                    c = json.load(f)
                if not isinstance(c, dict):     # [2, 3], a bare 2: JSON, but not the object asked for
                    raise ValueError(f'not a JSON object: {c!r}')
                n, depth = int(c.get('post_threads', n)), int(c.get('depth', depth))
            except (OSError, ValueError, TypeError) as e:
                # NO FALLBACK: a file that is THERE says what was meant; running the
                # env's knobs instead would be a quiet guess. It is written by rename, so unreadable = a mistake
                raise ValueError(f'pair pipeline: control file {control} unreadable: {e}') from e
        if n <= 0:
            return None
        return _PairPipeline(n, depth, control)

    def refresh(self):
        """Between pairs only: the control file, then threads added (or told to drain)."""
        if self.control and os.path.isfile(self.control):
            try:
                with open(self.control) as f:
                    c = json.load(f)
                if not isinstance(c, dict):     # as in from_env: never an AttributeError mid-call
                    raise ValueError(f'not a JSON object: {c!r}')
                with self.lock:
                    self.target = max(1, int(c.get('post_threads', self.target)))
                with self.cv:
                    self.depth = max(1, int(c.get('depth', self.depth)))
                    self.cv.notify_all()
            except (OSError, ValueError, TypeError) as e:
                # as at the call's start: the main thread raises between pairs, abort() lets the pairs in
                # flight finish and stops the threads
                raise ValueError(f'pair pipeline: control file {self.control} unreadable: {e}') from e
        self._top_up()

    def _top_up(self):
        with self.lock:
            self.threads = [t for t in self.threads if t.is_alive()]
            while len(self.threads) < self.target:
                t = self._threading.Thread(target=self._run, daemon=True,
                                           name=f'pair-post-{len(self.threads)}')
                self.threads.append(t)
                t.start()
            self.max_threads = max(self.max_threads, len(self.threads))

    def quiesce(self):
        """No pair in flight: every post thread's matching and writes done (GPU_YIELD, before the card is let go).
        A post thread that failed meanwhile raises here, as it would at the next slot."""
        with self.cv:
            while self.in_flight > 0 and not self.errors:
                self.cv.wait(0.5)
            if self.errors:
                raise self.errors[0]

    def acquire_slot(self):
        """The ONLY place the main thread waits on the post threads, and it holds no _PairLock here:
        a post thread never waits on the main thread, so the wait always ends (no main <- post <- main)."""
        t0 = time.perf_counter()
        with self.cv:
            while self.in_flight >= self.depth and not self.errors:
                self.cv.wait(0.5)
            # A POST THREAD FAILED while we waited: raise BEFORE the slot is counted. Counted, it would never
            # be released (the pair is not handed over) and abort()'s _stop would wait on in_flight forever;
            # returning instead started the next pair's lookup and model on a job that had already failed
            if self.errors:
                raise self.errors[0]
            self.in_flight += 1
        self.slot_wait_s += time.perf_counter() - t0

    def release_slot(self):
        with self.cv:
            self.in_flight -= 1
            self.cv.notify_all()

    def submit(self, fn):
        self.q.put(fn)

    def _run(self):
        me = self._threading.current_thread()
        while True:
            t0 = time.perf_counter()
            fn = self.q.get()
            t1 = time.perf_counter()
            if fn is None:
                return
            try:
                fn()
            except BaseException as e:          # reported by finish(); the job fails
                with self.lock:
                    self.errors.append(e)
                with self.cv:
                    self.cv.notify_all()
            t2 = time.perf_counter()
            with self.lock:
                self.idle_s += t1 - t0; self.busy_s += t2 - t1
                # DRAIN, after a whole pair: more threads alive than the target -> this one goes
                alive = [t for t in self.threads if t.is_alive()]
                if len(alive) > self.target:
                    self.threads.remove(me)
                    return

    def raise_if_failed(self):
        if self.errors:
            raise self.errors[0]

    def _stop(self):
        with self.cv:
            while self.in_flight > 0:
                self.cv.wait(0.5)
        with self.lock:
            threads = list(self.threads)
        for _ in threads:
            self.q.put(None)
        for t in threads:
            t.join()

    def finish(self):
        self._stop()
        self.raise_if_failed()

    def abort(self):
        """The main thread is raising: let the pairs in flight finish (their locks go with them), stop the
        threads, and add nothing -- the main thread's own exception is the one that propagates."""
        self._stop()


@torch.no_grad()
def forward_mast3r(pairs, model, cache_path, desc_conf='desc_conf',
                   device='cuda', subsample=8,
                   corres_root=None, corr_masks=None, seed_fill=False,  # hillview
                   provenance=None,  # hillview: {'record': {...}, 'require': {...} | None, 'vid': str, 'reads': list}
                   held_fwd=None,    # hillview: pointmap stems ANOTHER cache holds (forward/<k1>/<k2>[@vid]), never written here
                   **matching_kw):
    res_paths = {}
    # hillview: when matching is restricted to unmasked pixels the matches are a property of
    # THIS run's masks rather than of the image pair. Identical masks/settings share completed
    # matches under a separate key; each run also gets its own raw correspondence copy.
    # Different masks still need inference: the pointmap cache contains no descriptors.
    # A run-local copy INTO THE ROOT IT IS READ FROM is never right: the copy is named by the two
    # views alone (corres_conf=.../<k1>-<k2>), which in that root is the RAW correspondences' name,
    # so a mask's or a variant's matches replaced the unmasked ones every other reader takes for
    # raw. hillview's adaptive pairing and expansion did exactly that to the shared cache once;
    # refused here so no caller can again.
    if (corres_root is not None and cache_path is not None
            and os.path.realpath(corres_root) == os.path.realpath(cache_path)):
        raise ValueError(f'forward_mast3r: corres_root is the cache it matches into ({cache_path}); '
                         f'a run-local copy there would replace the raw correspondences. Pass the '
                         f"run's own cache, or corres_root=None to copy nothing")
    corr_root = corres_root or cache_path
    _n_computed = _n_waited = _n_skipped = _n_kept_fwd = _n_held_fwd = 0; _t_waited = 0.0
    _n_lock_free = _n_locked = 0
    # WHERE THE TIME GOES in the phase that dominates a run: a 16-panorama span spent
    # ~47 min on 2472 pairs with the card at 23 % (why does the GPU sit
    # idle?). Each computed pair runs one model call, then WRITES two pointmap files and one
    # correspondence file -- at 512 px that is on the order of 18 MB a pair, tens of GB a run, onto a
    # Docker volume. So the loop times its own parts and says so at the end, instead of leaving the
    # next person to guess between the model, the matching and the disk (STRUCTURAL_TOOLS.md: make
    # the program emit its own state).
    LAST_FORWARD_PROFILE.clear()
    _t = {'model': 0.0, 'save_fwd': 0.0, 'match': 0.0, 'save_corres': 0.0,
          'lookup': 0.0, 'lock_wait': 0.0, 'reverse_corres': 0.0, 'local_corres_copy': 0.0,
          'canary': 0.0, 'mask_check': 0.0, 'match_quality': 0.0, 'model_warm': 0.0}
    # hillview: and INSIDE 'match', where reciprocal matching spends itself -- query-plus-sync
    # against numpy bookkeeping, and how many hops actually run. See fast_nn.STATS.
    from mast3r.fast_nn import STATS as _nn_stats, stats_line as _nn_line, stats_reset as _nn_reset
    _nn_reset()
    _t0 = time.perf_counter()
    if FORWARD_MISSING not in ('compute', 'refuse'):
        raise ValueError(f'FORWARD_MISSING must be compute or refuse, not {FORWARD_MISSING!r}')
    _missing = []
    _empty = []                           # zero-length cache files met at the hit decision (EmptyCacheEntry)
    # THE CANARY (CANARY_RATE above): a canary recomputes a hit, so it needs the model
    if CANARY_RATE and model is None:
        raise ValueError('forward_mast3r: CANARY_RATE > 0 needs the model -- a canary recomputes a cache hit')
    if CANARY_RATE:
        canary_check_settings()
    if CANARY_RATE and not CANARY_FRESH_ROOT:
        raise ValueError('forward_mast3r: CANARY_RATE > 0 needs CANARY_FRESH_ROOT -- where a pair whose check does '
                         'not pass is kept, recomputed, for this run to solve on')
    _n_canary = _n_mask_checked = _n_fresh = 0
    _eroded = {}                          # id(view mask) -> (mask, eroded): one erosion per mask (CANARY_MASK_ERODE)
    # hillview: ENCODE EACH VIEW ONCE per call. symmetric_inference ran the encoder on
    # both images of every computed pair, so a view was encoded once per pair it is in (~26 times on
    # a church span), and the encoder is ~24 % of a pair's inference (real checkpoint, CPU). A view's
    # encoding is kept from its first computed pair until the LAST pair of this call that names it
    # (the list is known in advance), so a call holds only the views still ahead of it -- ~3 MB each
    # at 512x384. Bit-identical on CPU (see _view_encoding).
    # MAIN THREAD ONLY: _infer fills it, the loop below evicts from it; post threads never touch it.
    _encodings, _last_use = {}, {}
    for _k, (_a, _b) in enumerate(pairs):
        _last_use[_a['instance']] = _last_use[_b['instance']] = _k
    _n_encoded = _enc_peak = 0
    # hillview: THE LOOP IN STAGES, so a pair can be pipelined (_PairPipeline, off by default).
    # _lookup (main thread: the cache check, the _PairLock, the reverse corres -- unchanged), _infer (main
    # thread: the model), _post (the pointmap writes, the matching, the corres write) and _complete (the
    # run-local copy, the lock's release, the result slot). Sequentially they run in exactly the old order,
    # with the lock held over the same span. Pipelined, _post and _complete run on a post thread.
    _pipe = _PairPipeline.from_env()
    import threading as _threading
    _tlock = _threading.Lock()            # _t and the post-side counters: post threads add to them
    _latest = {}                          # res_paths key -> the sequence number that owns its slot
    _inflight = set()                     # pipelined: the _PairLock keys a post thread of ours still holds
    _n_deferred = 0
    _match_streams = _threading.local()

    def _add(part, dt):
        with _tlock:
            _t[part] += dt

    # A POINTMAP HOME HOLDS (held_fwd) COUNTS AS THERE. A matching job's final sweep re-infers a pair whose
    # matches home lacks (new masks) only for its descriptors, and keeps the pointmaps home has (keep_fwd).
    # Then the pair's REVERSE, (b,a), names the same two files: missing here, it would be computed again --
    # every pair twice, the saved writes back as model time -- and so would a second
    # run on the box. So in the stats a held stem stands for a file that is there, and a pair whose corres
    # is here (or transposed) is a hit. Only for a call whose results are not read here: held_fwd comes from
    # a matching job's final sweep, whose res_paths nobody opens.
    _AT_HOME = object()

    def _held_here(path, st):
        if st is not None or held_fwd is None:
            return st
        return _AT_HOME if os.path.relpath(_write_name(path), cache_path)[:-len(CACHE_EXT)] in held_fwd else None

    def _entry_paths(img1, img2):
        """A PAIR'S NAMES, before anything is read or locked: (idx1, idx2, path1, path2, path_corres, path_corres2,
        local_corres, masks, how) -- the lookup's, and the canary's pick, which must know a pair's writers before
        the lookup (canary_pick), from the same names (one function, so the two cannot drift apart)"""
        idx1 = hash_md5(img1['instance'])
        idx2 = hash_md5(img2['instance'])

        path1 = cache_path + f'/forward/{idx1}/{idx2}{CACHE_EXT}'
        path2 = cache_path + f'/forward/{idx2}/{idx1}{CACHE_EXT}'
        path_corres = corr_root + f'/corres_conf={desc_conf}_{subsample=}/{idx1}-{idx2}{CACHE_EXT}'
        path_corres2 = corr_root + f'/corres_conf={desc_conf}_{subsample=}/{idx2}-{idx1}{CACHE_EXT}'
        local_corres = path_corres
        # hillview: the masks belong to the PAIR, not to each image -- a view's foliage is
        # excluded only against a partner that excludes its own (CORR_MASKS_BY_PAIR)
        # and each mask comes with its IDENTITY, which keys the masked matches (CORR_MASKS_BY_IDENTITY)
        masks = None
        if corr_masks is not None:
            got = tuple(corr_masks(img1['instance'], img2['instance']))
            if len(got) != 4:
                raise ValueError(f'corr_masks must return (mask1, mask2, identity1, identity2), '
                                 f'got {len(got)} item(s)')
            masks, identities = got[:2], got[2:]
            for m, ident in zip(masks, identities):
                # 'none' says nothing masks the view; a mask under it would share matches with
                # the unmasked view's. The other way round is fine: a view whose layers left no
                # pixel this time (m None) still has the identity of what it was made from
                if m is not None and ident == MASK_IDENTITY_NONE:
                    raise ValueError(f'a mask under identity {MASK_IDENTITY_NONE!r}')
            path_corres, path_corres2 = _masked_matching_paths(
                cache_path, idx1, idx2, identities, desc_conf, subsample, seed_fill, matching_kw)

        _how = 'default'
        if provenance is not None:
            (path1, path2, path_corres, path_corres2), _how = resolve_cache_paths(
                (path1, path2, path_corres, path_corres2), provenance.get('require'), provenance['vid'])
        return idx1, idx2, path1, path2, path_corres, path_corres2, local_corres, masks, _how

    def _pick_rate(img1, img2):
        """the canary rate of a pair as stored now (canary_rate_of): the records of its files that are here -- a
        reverse's when only the reverse's matches exist, since the lookup transposes those"""
        _i1, _i2, p1, p2, pc, pc2, _l, _m, _h = _entry_paths(img1, img2)
        found = [f for f, st in (_resolve_stat(p) for p in (p1, p2)) if st is not None]
        fc, stc = _resolve_stat(pc)
        if stc is None:
            fc, stc = _resolve_stat(pc2)
        return canary_rate_of([_prov_read(f) for f in found + ([fc] if stc is not None else [])])

    _canary_picked = canary_pick(pairs, _pick_rate) if CANARY_RATE > 0 and CANARY_CAP is not None else None

    def _canary_want(idx1, idx2, cname, files):
        """whether the hit named `cname` (its stored files `files`) is drawn: the cap's pick, else under CANARY_RATE
        whoever wrote it, else in the band by its writers (canary_rate_of). Not whether it was checked already"""
        if _canary_picked is not None:
            return f'{idx1}-{idx2}' in _canary_picked
        if canary_drawn(cname):
            return True
        if canary_value(cname) < canary_ceiling():
            return canary_value(cname) < canary_rate_of([_prov_read(p) for p in files])
        return False

    def _warm_before_lock(idx1, idx2, path1, path2, path_corres, path_corres2):
        """NEVER WAIT FOR THE NETWORK UNDER A PAIR'S LOCK: a lazily loaded model
        (hillview's, whose first use waits for room on the box -- host_lease.hold_phase) was loaded in _infer, with
        the pair's lock held. One run held the room (MASt3R loaded for a canary) and waited 4946 s for a pair whose
        lock another run held -- while that run waited, holding it, for that room. The host ledger promises a top-up
        waits only for another job's network phase, never for each other; a lock under the wait broke that. So a
        pair that will run the network -- a miss to compute, or a hit drawn as a canary -- has the model loaded
        HERE, lock-free, before its lock is taken. A miss that turns out a hit once locked (another run made it)
        loaded it for nothing: room, not a deadlock."""
        if model is None or getattr(model, 'loaded', True):
            return
        (p1, s1), (p2, s2), (pc, sc) = (_resolve_stat(p) for p in (path1, path2, path_corres))
        if sc is None:
            pc, sc = _resolve_stat(path_corres2)          # a reverse is transposed under the lock: a hit
        sts = [_held_here(p1, s1), _held_here(p2, s2), sc]
        if not all(sts):
            need = FORWARD_MISSING == 'compute'
        else:
            cname = '/'.join(_write_name(path_corres).split(os.sep)[-2:])
            need = (CANARY_RATE > 0 and cname not in CANARY_RECORD['done']
                    and cname not in CANARY_RECORD.setdefault('fresh', {})
                    and _canary_want(idx1, idx2, cname, [p for p, st in zip((p1, p2, pc), sts)
                                                         if isinstance(st, os.stat_result)]))
        if need:
            _ts = time.perf_counter()
            model.warm()
            _add('model_warm', time.perf_counter() - _ts)

    def _lookup(img1, img2, seq, wait=True):
        """-> the pair's state S; None for a pair that drops out (missing under refuse, or model-less);
        _BUSY, holding nothing, when `wait` is False and another holder has its lock"""
        nonlocal _n_lock_free, _n_locked, _n_waited, _t_waited, _n_skipped, _n_computed, _n_held_fwd
        _lookup_started = time.perf_counter()
        idx1, idx2, path1, path2, path_corres, path_corres2, local_corres, masks, _how = _entry_paths(img1, img2)
        # local-vs-source is decided by the KEY, whatever container holds it: a require run resolves
        # the default to its .pth, and comparing that with the .hvz name took the SHARED corres for a
        # run-local copy -- re-stored it and deleted the shared .pth. A mask or a
        # provenance variant is another key, and still gets its local copy
        _local_is_source = _write_name(path_corres) == _write_name(local_corres)
        # each name below is resolved to the container that holds it (.hvz, else the .pth of the same
        # key) WHERE IT IS STATTED, else it stays the name it is written under (while both containers exist)
        _add('lookup', time.perf_counter() - _lookup_started)

        # Complete, read-only hits need no writer lock. Atomic publication protects readers
        # from partial files; the solver already reads these paths after releasing the lock.
        # Keep writes (including the run-local masked copy) and provenance requirements on
        # the existing serialized path. A miss is checked again AFTER acquiring its lock.
        _lookup_started = time.perf_counter()
        _sts = None
        if _local_is_source and not (provenance and provenance.get('require')):
            path_corres, _corres_stat = _resolve_stat(path_corres)
            if _corres_stat is not None:
                (path1, _st1), (path2, _st2) = _resolve_stat(path1), _resolve_stat(path2)
                _sts = [_held_here(path1, _st1), _held_here(path2, _st2), _corres_stat]
        _read_only_hit = _sts is not None and all(_sts)
        _add('lookup', time.perf_counter() - _lookup_started)
        if not _read_only_hit:
            _warm_before_lock(idx1, idx2, path1, path2, path_corres, path_corres2)
        _lock = None if _read_only_hit else _PairLock(
            cache_path + f'/forward/.locks/{"-".join(sorted((idx1, idx2)))}.lock')
        _lock_started = time.perf_counter()
        # pipelined, a post thread may hold this pair's lock (its reverse, (b, a), in flight): this
        # flock then waits for it -- correct, the reverse corres is found below once it is released --
        # and the post thread never waits on this thread, so the wait ends. Not waiting (wait=False),
        # a lock someone else holds hands the pair back to the scheduler instead
        if _lock is not None and not _lock.acquire(wait):
            return _BUSY
        if _read_only_hit:
            _n_lock_free += 1
        else:
            _n_locked += 1
        try:
            if _lock is not None:
                _add('lock_wait', time.perf_counter() - _lock_started)
            _lookup_started = time.perf_counter()
            _has_reverse = False
            _zero_rev = []
            if not _read_only_hit:
                path_corres2 = _found(path_corres2)
                path_corres = _found(path_corres)
                _st_rev = _stat(path_corres2)
                if _st_rev is not None and _st_rev.st_size == 0 and _stat(path_corres) is None:
                    _zero_rev = [path_corres2]          # EMPTY (below): never transposed into an entry
                _has_reverse = not _zero_rev and _st_rev is not None and _stat(path_corres) is None
            _add('lookup', time.perf_counter() - _lookup_started)
            if _lock is not None and _lock.waited > 1.0:
                _n_waited += 1; _t_waited += _lock.waited
            if _has_reverse:
                _reverse_started = time.perf_counter()
                # hillview: mapped like every other cache read. A GPU box saves its tensors
                # where they were (cuda:0), and the relay brings that file home whole, so a CPU reader
                # without map_location dies on it -- one run did, 11.5 h into its matching, on the one
                # pair whose cached corres existed only in the other order.
                score, (xy1, xy2, confs) = _cache_load(path_corres2, 'cpu')
                _save_atomic((score, (xy2, xy1, confs)), mkdir_for(path_corres))
                # the transposed file is the same content: it carries the original's record, or none --
                # written only once the file it describes exists, so no record is ever an orphan
                _rec2 = _prov_read(path_corres2) if provenance is not None else None
                if _rec2 is not None and os.path.isfile(path_corres):
                    _prov_write(path_corres, _rec2)

                _add('reverse_corres', time.perf_counter() - _reverse_started)

            _lookup_started = time.perf_counter()
            if not _read_only_hit:
                (path1, _st1), (path2, _st2), (path_corres, _stc) = (_resolve_stat(p) for p in (path1, path2, path_corres))
                _sts = [_held_here(path1, _st1), _held_here(path2, _st2), _stc]
            if _AT_HOME in _sts[:2] and all(_sts[:2]):    # a pointmap at home that this pair does not write
                _n_held_fwd += 1
            _add('lookup', time.perf_counter() - _lookup_started)
            S = dict(seq=seq, img1=img1, img2=img2, idx1=idx1, idx2=idx2, path1=path1, path2=path2,
                     path_corres=path_corres, local_corres=local_corres, masks=masks, how=_how,
                     local_is_source=_local_is_source, lock=_lock, sts=_sts, computed=False)
            # AN EMPTY ENTRY IS NEITHER A HIT NOR A MISS (hillview): a write that never reached the disk
            # (_durable) under a complete name. Read, it dies in the decoder without saying which file; recomputed,
            # the damage would go unseen. Every such file of the call is named at its end (EmptyCacheEntry)
            _zero = _zero_rev + [p for p, st in zip((path1, path2, path_corres), _sts)
                                 if isinstance(st, os.stat_result) and st.st_size == 0]
            _src = None if SOURCE_FRESH is None else SOURCE_FRESH.get((idx1, idx2))
            if _src is None and SOURCE_READS is not None:
                # a restore: the pointmaps the SOURCE read for this pair, wherever the cache's name resolves today. The
                # reverse of a pair its canary solved on its recompute: the source read the stored forward/<idx2>/<idx1>
                # here by name (its record names the recompute), so by name; the canonical check compares it after
                _p12 = SOURCE_READS.get((idx1, idx2))
                _p21 = path2 if (idx2, idx1) in (SOURCE_FRESH or {}) else SOURCE_READS.get((idx2, idx1))
                if _p12 is None or _p21 is None:
                    raise ValueError(f"restore: the source's read record names no file for "
                                     f"{[k for k, p in (((idx1, idx2), _p12), ((idx2, idx1), _p21)) if p is None]}: "
                                     f"this run would read a pair its source did not")
                path1, path2 = _p12, _p21
                _sts = [_stat(path1), _stat(path2), _sts[2]]
                if not all(_sts):
                    raise FileNotFoundError(f"restore: pair {idx1}-{idx2}: "
                                            f"{[p for p, st in zip((path1, path2, path_corres), _sts) if st is None]} "
                                            f"not there; the source read them, and a restore computes no pair")
                _zero = _zero_rev + [p for p, st in zip((path1, path2, path_corres), _sts) if st.st_size == 0]
                S.update(path1=path1, path2=path2, how='source_read', sts=_sts)
            if _src is not None:
                # a restore: the source's canary solved this pair on its own recompute, so the source read these files and
                # never the cache's; this run reads them too, unchecked (they are the source's input, not a cache hit)
                _sts = [_stat(p) for p in _src]
                if not all(_sts):
                    raise FileNotFoundError(f"restore: the source solved {idx1}-{idx2} on its own recompute and kept "
                                            f"it in its run dir, but {[p for p, st in zip(_src, _sts) if st is None]} "
                                            f"are not there")
                _zero = [p for p, st in zip(_src, _sts) if st.st_size == 0]
                S.update(path1=_src[0], path2=_src[1], path_corres=_src[2], how='source_fresh', sts=_sts)
            elif all(_sts) and not _zero and CANARY_RATE > 0:
                # A CANARY: named by its matches file -- the masks' identities are in the name
                _cname = '/'.join(_write_name(path_corres).split(os.sep)[-2:])
                _want = _canary_want(idx1, idx2, _cname, [p for p, st in zip((path1, path2, path_corres), _sts)
                                                          if isinstance(st, os.stat_result)])
                with _tlock:
                    _fresh = CANARY_RECORD.setdefault('fresh', {}).get(_cname)
                    if _fresh is not None:
                        # ONCE SOLVED ON ITS RECOMPUTE, ALWAYS: a pair is checked once per process, and
                        # a later call (an adaptive round, the solve's own pass) would otherwise read the stored
                        # entry that did not pass. It reads this run's recompute, as the first call did
                        S.update(path1=_fresh[0], path2=_fresh[1], path_corres=_fresh[2], how='canary_fresh',
                                 sts=[_stat(p) for p in _fresh])
                    elif _cname not in CANARY_RECORD['done'] and _want:
                        CANARY_RECORD['done'].add(_cname)
                        S['canary'] = _cname
            if _zero:
                _empty.extend(_zero)
                S = None
            elif not all(_sts):
                if FORWARD_MISSING == 'refuse':
                    _missing.append((idx1, idx2))
                    S = None
                elif model is None:
                    _n_skipped += 1
                    S = None
                else:
                    S['computed'] = True
                    _n_computed += 1
                    # hillview: A PAIR RE-INFERRED ONLY FOR ITS MATCHES keeps the pointmaps it found.
                    # Masked matching needs the descriptors, which no cache holds, so a new mask identity (or
                    # matches lost on the way home) runs the network again -- but both pointmap files are
                    # already there under the same key. Rewriting them cost 22-25 % of the forward phase on
                    # one rental (RTX 3090, cache in RAM), and put other bytes (GPU inference differs by
                    # ~2e-4) under a key other readers had already read. Same key, interchangeable: keep them.
                    # Decided HERE, from the stats taken under the lock; the post stage only executes it.
                    # ...OR HOME HOLDS THEM (held_fwd): a rented box matching under masks home has
                    # no matches for wrote the pointmaps home already had -- 44.7 GB dropped by the relay on
                    # one rental, 87 % of one run's forward wall (save_fwd, the post threads' cap). _held_here
                    # put them in the stats as there, so the same test keeps them where they are
                    S['keep_fwd'] = bool(_sts[0]) and bool(_sts[1])
        except BaseException:
            if _lock is not None:
                _lock.__exit__(None, None, None)
            raise
        if S is None and _lock is not None:
            _lock.__exit__(None, None, None)
        return S

    def _infer(S):
        nonlocal _n_encoded, _enc_peak
        _ts = time.perf_counter()
        _held = len(_encodings)
        res = symmetric_inference(model, S['img1'], S['img2'], device=device, encodings=_encodings)
        _n_encoded += len(_encodings) - _held
        _enc_peak = max(_enc_peak, len(_encodings))
        S['X'] = [r['pts3d'][0] for r in res]            # X11, X21, X22, X12
        S['C'] = [r['conf'][0] for r in res]             # C11, C21, C22, C12
        S['descs'] = [r['desc'][0] for r in res]
        S['qonfs'] = [r[desc_conf][0] for r in res]
        if _pipe is not None and str(device).startswith('cuda'):
            # THE MATCHING READS THESE ON ANOTHER STREAM: an event recorded on the
            # inference stream, waited on by the match stream before it reads them; record_stream (in
            # _post) keeps the caching allocator from handing their memory back to the next inference
            # while the match kernels still read it -- silent corruption otherwise, not a crash. And the
            # MODEL TIME keeps its meaning (rental_ledger's passes per model-second): the main thread
            # waits for its own inference here, as the sequential loop did at its first host copy.
            S['event'] = torch.cuda.Event()
            S['event'].record()
            S['event'].synchronize()
        S['t_model'] = time.perf_counter() - _ts
        _add('model', S['t_model'])

    def _post(S):
        nonlocal _n_kept_fwd
        ctx = nullcontext()
        if S.get('event') is not None:
            ms = getattr(_match_streams, 's', None)
            if ms is None:
                ms = _match_streams.s = torch.cuda.Stream(device=device)
            ms.wait_event(S['event'])
            for t in S['X'] + S['C'] + S['descs'] + S['qonfs']:
                t.record_stream(ms)
            ctx = torch.cuda.stream(ms)
        X11, X21, X22, X12 = S['X']
        C11, C21, C22, C12 = S['C']
        if S.get('canary'):
            # the canary: recomputed to be COMPARED -- nothing goes into a cache. A pair that passes keeps the stored
            # paths; one that does not is solved on this recompute, from the run's own dir (_use_fresh) -- but in a
            # restore (CANARY_SOLVE_FRESH False) it keeps them too: the source read the stored files
            with torch.no_grad(), ctx:
                warned, matches = _canary_check(S, (X11, C11, X21, C21), (X22, C22, X12, C12))
                if warned and CANARY_SOLVE_FRESH:
                    _use_fresh(S, (X11, C11, X21, C21), (X22, C22, X12, C12), matches)
            for k in ('X', 'C', 'descs', 'qonfs', 'event'):
                S.pop(k, None)
            return
        # no_grad is THREAD-LOCAL: a post thread does not inherit forward_mast3r's decorator, and with
        # autograd on extract_correspondences dies on numpy() ("requires grad", a bug once met)
        with torch.no_grad(), ctx:
            _ts = time.perf_counter()
            # save -- under the CACHE_EXT names: a pair recomputed beside a .pth twin is written as .hvz
            S['path_corres'] = _write_name(S['path_corres'])
            if S['keep_fwd']:
                with _tlock:
                    _n_kept_fwd += 1
            else:
                S['path1'], S['path2'] = (_write_name(p) for p in (S['path1'], S['path2']))
                _save_atomic(to_cpu((X11, C11, X21, C21)), mkdir_for(S['path1']))
                _save_atomic(to_cpu((X22, C22, X12, C12)), mkdir_for(S['path2']))
                if provenance is not None:          # a pointmap's record names no matcher (MATCH_ONLY_KEYS)
                    _prov_write(S['path1'], forward_part(provenance['record']))
                    _prov_write(S['path2'], forward_part(provenance['record']))
            _add('save_fwd', time.perf_counter() - _ts); _ts = time.perf_counter()

            # perform reciprocal matching
            try:
                corres = extract_correspondences(
                    S['descs'], S['qonfs'], device=device, subsample=subsample, seed_fill=seed_fill,
                    seed_masks=S['masks'])
            except NNIndexOutOfRange as e:            # the pair it happened on (hillview, fast_nn._nn_checked)
                raise NNIndexOutOfRange(f"{S['img1']['instance']} - {S['img2']['instance']}: {e}") from e

            S['t_match'] = S.get('t_match', 0.0) + time.perf_counter() - _ts
            _add('match', time.perf_counter() - _ts); _ts = time.perf_counter()

            matching_score = _matching_score(S['C'], corres)
            if cache_path is not None:
                _rec = provenance['record'] if provenance is not None else None
                if _rec is not None and MATCH_QUALITY is not None:
                    # THIS entry's matches scored where they are made (MATCH_QUALITY), into its record only -- BEFORE
                    # the data is written: a scorer that fails leaves nothing behind, never an entry without its
                    # record, which a requirement would read as a miss and a quarantine could not select
                    _tq = time.perf_counter()
                    _rec = dict(_rec, match_quality=MATCH_QUALITY(*to_cpu((X11, X21)), to_cpu(corres)))
                    _add('match_quality', time.perf_counter() - _tq)
                # to_cpu: on a GPU box `corres` is on the device (extract_correspondences), and the
                # .hvz container stores CPU arrays only -- as the forward passes above always were
                _save_atomic((matching_score, to_cpu(corres)), mkdir_for(S['path_corres']))
                if _rec is not None:
                    _prov_write(S['path_corres'], _rec)
            _add('save_corres', time.perf_counter() - _ts)
        S['sts'] = [_stat(p) for p in (S['path1'], S['path2'], S['path_corres'])]
        for k in ('X', 'C', 'descs', 'qonfs', 'event'):       # the device memory goes with the pair
            S.pop(k, None)

    def _canary_rel(path):
        root = os.path.join(os.path.abspath(cache_path), '') if cache_path is not None else None
        return os.path.relpath(path, root) if root and os.path.abspath(path).startswith(root) else path

    def _canary_views(S):
        return [os.path.basename(str(S['img1']['instance'])), os.path.basename(str(S['img2']['instance']))]

    def _canary_transposed_reverse(S):
        """(a, b)'s matches -- matched and scored as their own pair would be -- transposed into (b, a)'s rows. S
        holds (b, a)'s outputs 11, 21, 22, 12; (a, b)'s are the same four reordered: 11'=22, 21'=12, 22'=11, 12'=21"""
        o = [2, 3, 0, 1]
        masks = None if S['masks'] is None else (S['masks'][1], S['masks'][0])
        c = extract_correspondences([S['descs'][i] for i in o], [S['qonfs'][i] for i in o], device=device,
                                    subsample=subsample, seed_fill=seed_fill, seed_masks=masks)
        xy1, xy2, cf = to_cpu(c)
        return _matching_score([S['C'][i] for i in o], c), (xy2, xy1, cf)

    def _canary_check(S, fwd1, fwd2):
        """The canary: a hit, recomputed, against each of its files that is here (a pointmap only home holds is
        not); the tier from each file's writer record (canary_tier). -> (whether any check did not pass, the
        fresh matches as an entry holds them)"""
        nonlocal _n_canary
        _ts = time.perf_counter()
        corres = extract_correspondences(S['descs'], S['qonfs'], device=device, subsample=subsample,
                                         seed_fill=seed_fill, seed_masks=S['masks'])
        fresh_matches = (_matching_score(S['C'], corres), to_cpu(corres))
        fresh = [(S['path_corres'], 'matches', fresh_matches)]
        for path, st, obj in ((S['path1'], S['sts'][0], fwd1), (S['path2'], S['sts'][1], fwd2)):
            if isinstance(st, os.stat_result):
                fresh.append((path, 'pointmaps', to_cpu(obj)))
        own = provenance['record'] if provenance is not None else None
        warned = False
        for path, kind, obj in fresh:
            writer = _prov_read(path)
            tier = canary_tier(writer, own)
            stored = _cache_load(path, 'cpu')
            verdict, detail = canary_compare(kind, obj, stored, tier)
            if kind == 'matches' and tier == 'bits' and verdict != 'ok':
                # A REVERSE FROM THE CACHE (every (b, a) of a 4-frame match job failed until
                # this): (b, a)'s entry is usually (a, b)'s matches transposed (_has_reverse) -- the same rows, but
                # where both decoder passes found a row, the OTHER pass's confidence. Its bits are those of
                # (a, b)'s matching, transposed, from the same four network outputs reordered: no inference
                verdict_r, detail_r = canary_compare(kind, _canary_transposed_reverse(S), stored, tier)
                detail = dict(detail_r, as_reverse=True) if verdict_r == 'ok' else dict(detail, as_reverse=detail_r)
                verdict = verdict_r if verdict_r == 'ok' else verdict
            r = dict(pair=[S['idx1'], S['idx2']], views=_canary_views(S), file=_canary_rel(path), kind=kind,
                     check='bits' if tier == 'bits' else 'divergence', tier=tier, verdict=verdict, writer=writer,
                     detail=detail)
            warned = warned or verdict != 'ok'
            with _tlock:
                CANARY_RECORD['results'].append(r)
                if CANARY_SINK is not None:
                    CANARY_SINK(r)
        with _tlock:
            _n_canary += 1
        _add('canary', time.perf_counter() - _ts)
        return warned, fresh_matches

    def _use_fresh(S, fwd1, fwd2, matches):
        """The canary: a pair whose check did not pass is solved on THIS run's recompute -- kept under
        CANARY_FRESH_ROOT, the run's own dir, never in a cache: no reader meets it, nothing is keyed for reuse. The
        pair's paths and its reads record then name these files (how 'canary_fresh', computed here)"""
        nonlocal _n_fresh
        names = (os.path.join(CANARY_FRESH_ROOT, 'forward', S['idx1'], S['idx2'] + CACHE_EXT),
                 os.path.join(CANARY_FRESH_ROOT, 'forward', S['idx2'], S['idx1'] + CACHE_EXT),
                 os.path.join(CANARY_FRESH_ROOT, 'corres', f"{S['idx1']}-{S['idx2']}{CACHE_EXT}"))
        p1, p2, pc = (_write_name(n) for n in names)
        _save_atomic(to_cpu(fwd1), mkdir_for(p1))
        _save_atomic(to_cpu(fwd2), mkdir_for(p2))
        _save_atomic(matches, mkdir_for(pc))
        if provenance is not None:
            _prov_write(p1, forward_part(provenance['record']))
            _prov_write(p2, forward_part(provenance['record']))
            _prov_write(pc, provenance['record'])
        S.update(path1=p1, path2=p2, path_corres=pc, how='canary_fresh', computed=True)
        S['sts'] = [_stat(p) for p in (p1, p2, pc)]
        with _tlock:
            _n_fresh += 1
            CANARY_RECORD.setdefault('fresh', {})[S['canary']] = (p1, p2, pc)

    def _mask_check(S):
        """The canary: the stored matches of a masked hit against the reader's own masks, eroded (CANARY_MASK_CHECK)"""
        nonlocal _n_mask_checked
        _ts = time.perf_counter()
        from scipy import ndimage
        _score, (xy1, xy2, _conf) = _cache_load(S['path_corres'], 'cpu')
        on, out_of_frame = [], []
        for xy, m in ((xy1, S['masks'][0]), (xy2, S['masks'][1])):
            if m is None:
                on.append(0); out_of_frame.append(0)
                continue
            with _tlock:
                got = _eroded.get(id(m))
            if got is None:
                mm = np.asarray(m, dtype=bool)
                # border_value=1: only the mask's INNER rim is forgiven -- a sky or a mount mask touches the frame,
                # and eroded from the frame's edge too a band of 2 x CANARY_MASK_ERODE rows would vanish
                got = (m, ndimage.binary_erosion(mm, iterations=CANARY_MASK_ERODE, border_value=1)
                       if CANARY_MASK_ERODE else mm)
                with _tlock:
                    _eroded[id(m)] = got          # the mask itself is kept, so its id is not reused
            e = got[1]
            xy = _canary_arrays(xy)[0].reshape(-1, 2).astype(np.int64)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < e.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < e.shape[0])
            out_of_frame.append(int((~inside).sum()))
            on.append(int(e[xy[inside, 1], xy[inside, 0]].sum()))
        n = len(_canary_arrays(xy1)[0])
        bad = sum(on) + sum(out_of_frame)
        rec = None
        if bad:
            rec = dict(pair=[S['idx1'], S['idx2']], views=_canary_views(S), file=_canary_rel(S['path_corres']),
                       kind='matches', check='mask', tier='mask',
                       verdict='ok' if CANARY_MASK_WARN is None or bad <= CANARY_MASK_WARN * max(n, 1) else 'warn',
                       writer=_prov_read(S['path_corres']),
                       detail=dict(on_mask=on, out_of_frame=out_of_frame, n=n, erode_px=CANARY_MASK_ERODE))
        with _tlock:
            CANARY_RECORD['mask_checked'] += 1
            _n_mask_checked += 1
            if rec is not None:
                CANARY_RECORD['mask_on'].append(rec)
                if CANARY_SINK is not None:
                    CANARY_SINK(rec)
        _add('mask_check', time.perf_counter() - _ts)

    def _complete(S):
        """The run-local copy (still under the lock), the lock released, the pair's slot filled."""
        try:
            if (CANARY_MASK_CHECK and not S['computed'] and S['masks'] is not None
                    and any(m is not None for m in S['masks'])):
                _mask_check(S)
            # NOTHING is materialized without a corres_root: the corres root is then the SHARED cache, and a
            # copy there writes a variant's (or a mask's) matches over the shared default -- every default
            # run would pair the default's forward passes with another producer's matches.
            # The solve reads the source it resolved; its reads record says which.
            if not S['local_is_source'] and corres_root is not None:
                # Always replace the local copy: it may belong to an older mask/settings variant. ONE file per
                # entry in a run dir, always the CACHE_EXT container (a stale twin in the other
                # container would shadow it): an .hvz source is copied with its digests, a .pth source is
                # re-stored; the other container's twin goes in the same step.
                _copy_started = time.perf_counter()
                import fcntl
                import shutil
                path_corres = S['path_corres']
                local_corres = _write_name(S['local_corres'])
                if path_corres.endswith(CACHE_EXT):
                    # under _save_atomic's directory lock (the converter's publish takes it too); a copy
                    # never keeps a digest of what it replaces: old sidecars go first
                    _dir = os.open(os.path.dirname(mkdir_for(local_corres)), os.O_RDONLY)
                    try:
                        fcntl.flock(_dir, fcntl.LOCK_EX)
                        for side in ('.sha256', '.content'):
                            try:
                                os.remove(local_corres + side)
                            except FileNotFoundError:
                                pass
                        for side in ('', '.sha256', '.content'):
                            if side == '' or os.path.exists(path_corres + side):
                                tmp = f'{local_corres}{side}.tmp.{os.getpid()}.{_threading.get_ident()}'
                                shutil.copyfile(path_corres + side, tmp)
                                os.replace(tmp, local_corres + side)
                    finally:
                        os.close(_dir)
                else:
                    _save_atomic(_cache_load(path_corres, 'cpu'), mkdir_for(local_corres))
                _rec = _prov_read(path_corres)              # the copy carries its source's record
                if _rec is not None:
                    _prov_write(local_corres, _rec)
                # the other container's twin of the COPY goes -- only in a run's own root: a solve never
                # deletes anything in the shared cache
                if corres_root is not None and os.path.realpath(corres_root) != os.path.realpath(cache_path):
                    # the twin is the other container IN THE SAME DIRECTORY: with read-only roots installed
                    # the candidates also name other roots, and nothing is ever deleted there
                    for twin in [t for t in _candidates(local_corres)[1:]
                                 if os.path.dirname(t) == os.path.dirname(local_corres)]:
                        for side in ('.content', '.sha256', PROV_SUFFIX, ''):
                            try:
                                os.remove(twin + side)
                            except FileNotFoundError:
                                pass
                S['path_corres'] = local_corres
                _add('local_corres_copy', time.perf_counter() - _copy_started)
        finally:
            if S['lock'] is not None:
                S['lock'].__exit__(None, None, None)

        # THE SLOT, filled in PAIR order whatever order the pairs finish in:
        # res_paths' order is the order canonical views are summed downstream, so finish order would
        # break bit-identity with the sequential loop. The key's position was taken at _lookup (_reserve);
        # a key met twice keeps its FIRST position and the LAST pair's value, as the old loop's dict did.
        key = S['img1']['instance'], S['img2']['instance']
        with _tlock:
            if _latest.get(key) == S['seq']:
                res_paths[key] = (S['path1'], S['path2']), S['path_corres']
            if provenance is not None:
                # WHAT THIS RUN READ, per pair: how the paths were chosen, whether it computed them, and
                # the pointmap file's identity (mtime, size) from the stat the hit path makes anyway --
                # two runs that read the same file instance are sharing it, which is what a twin
                # comparison needs to know
                # a pointmap home holds (held_fwd) was not read here: no file identity, as for a missing one
                _s = S['sts'][0] if isinstance(S['sts'][0], os.stat_result) else None
                provenance['reads'][S['slot']] = (S['idx1'], S['idx2'], S['how'], S['computed'],
                                                  _s.st_mtime_ns if _s else None, _s.st_size if _s else None)

    def _reserve(S):
        S['slot'] = _slot0 + S['seq']       # reserved up front, in PAIR order (the LOOKAHEAD below)

    def _infer_or_release(S):
        # the old `with _lock:` released the lock on ANY exception; the stages keep that
        try:
            _infer(S)
        except BaseException:
            if S['lock'] is not None:
                S['lock'].__exit__(None, None, None)
            raise

    def _post_and_complete(S):
        _tp = time.perf_counter()
        try:
            _post(S)
        except BaseException:
            if S['lock'] is not None:
                S['lock'].__exit__(None, None, None)
            raise
        S['t_post'] = time.perf_counter() - _tp
        _complete(S)                # releases the lock in its own finally
        rec = {'forward_s': round(S.get('t_model', 0.0), 3), 'post_s': round(S['t_post'], 3),
               'match_s': round(S['t_match'], 3) if 't_match' in S else None,
               'computed': bool(S['computed']), 'canary': bool(S.get('canary')), 'idx1': S['idx1'], 'idx2': S['idx2'],
               'post_threads': _pipe.target if _pipe is not None else 0}
        with _tlock:
            _recent.append(rec)
        if PAIR_TIMING is not None:
            PAIR_TIMING(rec)

    import collections as _collections
    _recent = _collections.deque(maxlen=50)     # the last pairs' timings, for the bar's postfix

    def _postfix():
        """the bar's postfix: medians of the last pairs' forward and matching, so the run.log line says how fast the
        DEVICE is, apart from how many pairs were cache hits (which the bar's own rate mixes in)"""
        with _tlock:
            fw = sorted(r['forward_s'] for r in _recent)
            mt = sorted(r['match_s'] for r in _recent if r['match_s'] is not None)
        if fw:
            _pbar.set_postfix_str(f"fwd {fw[len(fw) // 2]:.1f}s" + (f" match {mt[len(mt) // 2]:.1f}s" if mt else "")
                                  + f" (last {len(fw)})", refresh=False)

    # LOOKAHEAD instead of waiting on a lock. Two holders can have a pair's _PairLock:
    #  * OUR POST THREADS (pipelined): (b,a) follows (a,b) and takes the same lock, which a post thread holds
    #    while it saves (a,b). On one rental (a 3090, 2 x 3) the main thread spent 258 s of a 414 s
    #    match waiting there, the card idle 74 %. Such a pair is set aside until that lock is free.
    #  * ANOTHER RUN (both modes): arms of one span share every forward pass, and two of them on one box
    #    walk the pairs in the same order, so the second waits on nearly every pair the first is computing.
    #    On one CPU box (two arms of a church span, 20 threads each) that was 34509 s and
    #    ~67600 s of waiting: ~28 h of a worker idle. A pair whose lock another holder has is tried without
    #    waiting and set aside; the run goes on with pairs nobody holds, and comes back to it at the end,
    #    when it is mostly a cache hit. It blocks on a flock only once every pair left is held elsewhere.
    # The ORDER the solve sees stays the in-order loop's: every pair's res_paths key and provenance slot is
    # reserved here, in pair order, before anything runs (a slot is its sequence number; slots of pairs
    # that drop out -- missing, model-less -- are compacted away at the end), and _complete fills them.
    # With nobody else holding a lock, the pairs run in exactly the old order.
    _BUSY = object()
    _lock_key_of = ["-".join(sorted((hash_md5(a['instance']), hash_md5(b['instance'])))) for a, b in pairs]
    for _k, (_a, _b) in enumerate(pairs):
        key = _a['instance'], _b['instance']
        _latest[key] = _k
        res_paths.setdefault(key, None)
        if provenance is not None:
            provenance['reads'].append(None)
    _slot0 = len(provenance['reads']) - len(pairs) if provenance is not None else 0
    _dropped = set()
    _todo = list(range(len(pairs)))[::-1]   # pop() takes the next in pair order
    _aside = []                             # [seq, holder]: 'ours' (a post thread) or 'other' (another run)
    _deferred_other = set()                 # pairs ever set aside for another run
    _busy_in_a_row = 0                      # 'other' pairs found still held since the last pair that ran

    def _next_pair():
        """-> (seq, wait): the next pair to look up, and whether to block on its lock"""
        nonlocal _n_deferred
        with _tlock:
            busy = set(_inflight)
        for i, (q, holder) in enumerate(_aside):
            if holder == 'ours' and _lock_key_of[q] not in busy:
                del _aside[i]
                return q, False
        while _todo:
            q = _todo.pop()
            if _lock_key_of[q] in busy:
                _aside.append((q, 'ours')); _n_deferred += 1
            else:
                return q, False
        q, holder = _aside.pop(0)
        # nothing new to run: the oldest pair set aside. Ours: wait, as before (a post thread never waits
        # on this one). Another run's: try each in turn without waiting, and block once all were held
        return q, holder == 'ours' or _busy_in_a_row > len(_aside)

    _pbar = tqdm(total=len(pairs), desc='forward passes + matches', ascii=' .')
    try:
      while _todo or _aside:
        if GPU_YIELD is not None:
            GPU_YIELD(model, device, _pipe.quiesce if _pipe is not None else (lambda: None), _encodings)
        _seq, _wait = _next_pair()
        img1, img2 = pairs[_seq]
        # a view's encoding goes once its last pair is behind EVERY pair still to come -- the ones set aside
        # for our post threads too, so they never re-encode. Not those set aside for another run: they are
        # mostly cache hits when taken up, and one that is not re-encodes its views
        _horizon = min([_seq] + [q for q, h in _aside if h == 'ours'] + _todo[-1:])
        for _gone in [v for v in _encodings if _last_use[v] < _horizon]:
            del _encodings[_gone]
        if _pipe is None:
            S = _lookup(img1, img2, _seq, _wait)
            if S is _BUSY:
                _aside.append((_seq, 'other')); _deferred_other.add(_seq); _busy_in_a_row += 1
                continue
            _busy_in_a_row = 0
            _pbar.update(1)
            if S is None:
                _dropped.add(_seq)
                continue
            _reserve(S)
            if S['computed'] or S.get('canary'):
                _infer_or_release(S)
                _post_and_complete(S)
                _postfix()
            else:
                _complete(S)
            continue
        # PIPELINED: the knobs between pairs, a free slot BEFORE any lock is taken, then the lookup and
        # the model here and the rest on a post thread, which frees the slot when the pair is complete
        _pipe.raise_if_failed()
        _pipe.refresh()
        _pipe.acquire_slot()
        _handed = False
        S = None                    # never the previous pair's: its lock is a post thread's now
        try:
            S = _lookup(img1, img2, _seq, _wait)
            if S is _BUSY:
                _aside.append((_seq, 'other')); _deferred_other.add(_seq); _busy_in_a_row += 1
                continue
            _busy_in_a_row = 0
            _pbar.update(1)
            if S is None:
                _dropped.add(_seq)
                continue
            _reserve(S)
            if not S['computed'] and not S.get('canary'):
                _complete(S)
                continue
            _infer_or_release(S)
            _postfix()
            with _tlock:
                _inflight.add(_lock_key_of[_seq])

            def _task(S=S, lk=_lock_key_of[_seq]):
                try:
                    _post_and_complete(S)
                finally:
                    with _tlock:
                        _inflight.discard(lk)
                    _pipe.release_slot()
            _pipe.submit(_task)
            _handed = True
        finally:
            # every lock is released where the pair failed (_lookup, _infer_or_release, _complete); only
            # the slot is left to give back when the pair never reached a post thread
            if not _handed:
                _pipe.release_slot()
    except BaseException:
        if _pipe is not None:
            _pipe.abort()           # no post thread outlives the call, nor a lock it holds
        raise
    finally:
        _pbar.close()
    if _pipe is not None:
        _pipe.finish()
    # a pair skipped as missing/model-less never had a slot in the in-order loop: drop its reserved one
    if provenance is not None and _dropped:
        provenance['reads'][_slot0:] = [r for q, r in enumerate(provenance['reads'][_slot0:]) if q not in _dropped]
    for _k in [k for k, v in res_paths.items() if v is None]:
        del res_paths[_k]
    if provenance is not None and any(r is None for r in provenance['reads']):
        raise RuntimeError('forward_mast3r: a reserved provenance slot was never filled')
    if _empty:
        raise EmptyCacheEntry(f'{len(_empty)} cache file(s) are EMPTY -- a write that never reached the disk '
                              f'under a complete name (an unclean shutdown); quarantine them: ' + ', '.join(_empty))
    if _missing:
        raise ForwardCacheMissing(
            f'{len(_missing)} of {len(pairs)} directed pair(s) have no cached forward pass or '
            f'matches, and this run may not compute them (FORWARD_MISSING=refuse): '
            + ', '.join(f'{a}-{b}' for a, b in _missing[:8]) + (' ...' if len(_missing) > 8 else ''))
    _wall = time.perf_counter() - _t0
    LAST_FORWARD_PROFILE.update(wall_s=_wall, pairs=len(pairs), computed_pairs=_n_computed,
                                cached_pairs=len(pairs) - _n_computed - _n_skipped, skipped_pairs=_n_skipped,
                                kept_pointmap_pairs=_n_kept_fwd, pointmaps_at_home_pairs=_n_held_fwd,
                                encoded_views=_n_encoded, encodings_held_peak=_enc_peak,
                                lock_free_pairs=_n_lock_free, locked_pairs=_n_locked,
                                deferred_other_run_pairs=len(_deferred_other),
                                canary_pairs=_n_canary, canary_fresh_pairs=_n_fresh, mask_checked_pairs=_n_mask_checked,
                                parts_s=dict(_t), other_s=max(0.0, _wall - sum(_t.values())))
    if _pipe is not None:
        # PIPELINED, THE PARTS OVERLAP: parts_s sums the stages across threads and can exceed wall_s.
        # computed_pairs and parts_s.model mean what they always did (the model's own time, waited for).
        # post_busy_s / post_idle_s: the post threads' work and their wait for a pair; slot_wait_s: the
        # main thread's wait for a free slot (the post side is the bottleneck when it grows). gpu_idle is a
        # best effort: the share of wall the main thread was NOT inferring (the post threads' match kernels
        # run in it, so it over-counts idleness).
        LAST_FORWARD_PROFILE.update(
            parts_overlap=True,
            pipeline=dict(post_threads=_pipe.target, post_threads_max=_pipe.max_threads, depth=_pipe.depth,
                          post_busy_s=round(_pipe.busy_s, 3), post_idle_s=round(_pipe.idle_s, 3),
                          slot_wait_s=round(_pipe.slot_wait_s, 3), deferred_pairs=_n_deferred,
                          gpu_idle_frac=round(max(0.0, 1 - _t['model'] / _wall), 3) if _wall else None))
    print(f' >> forward cache profile: {LAST_FORWARD_PROFILE}')
    if _n_computed:
        _acc = sum(_t.values())
        print(f' >> forward pass time: {_wall:.0f} s wall for {_n_computed} computed pair(s) '
              f'({_wall / _n_computed:.2f} s each) -- model {_t["model"]:.0f} s '
              f'({100 * _t["model"] / _wall:.0f} %), pointmap writes {_t["save_fwd"]:.0f} s '
              f'({100 * _t["save_fwd"] / _wall:.0f} %), matching {_t["match"]:.0f} s '
              f'({100 * _t["match"] / _wall:.0f} %), correspondence writes {_t["save_corres"]:.0f} s '
              f'({100 * _t["save_corres"] / _wall:.0f} %), '
              # pipelined, the parts run at once and their sum passes the wall: no "elsewhere" (it went negative)
              + (f'elsewhere {_wall - _acc:.0f} s ({100 * (_wall - _acc) / _wall:.0f} %)' if _pipe is None
                 else f'parts overlapping on {_pipe.target} post thread(s), {_acc - _wall:.0f} s past the wall')
              + f'; the encoder ran {_n_encoded} time(s) for '
              f'{2 * _n_computed} image(s) of computed pairs')
    if _n_computed and _nn_stats['calls']:
        print(f' >> {_nn_line()}')
        _c = _nn_stats['converged_at']
        if _c:
            import collections as _co
            _h = _co.Counter(_c)
            print('    hops to convergence: '
                  + ', '.join(f'{k}:{_h[k]}' for k in sorted(_h))
                  + '  (10 = hit max_iter without converging)')
    print(f' >> forward passes: {_n_computed} of {len(pairs)} pairs computed here, '
          f'{len(pairs) - _n_computed} from the cache' + (f', {_n_waited} of them after waiting {_t_waited:.0f} s for '
                                                          f'another run computing the same pair' if _n_waited else '')
          + (f'; {len(_deferred_other)} set aside while another run held them, taken up later'
             if _deferred_other else ''))
    del model
    torch.cuda.empty_cache()

    return res_paths, cache_path


def symmetric_inference(model, img1, img2, device, encodings):
    view1, view2 = img1, img2
    shape1 = torch.from_numpy(img1['true_shape']).to(device, non_blocking=True)
    shape2 = torch.from_numpy(img2['true_shape']).to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)

    # hillview: INFERENCE_AUTOCAST (None = fp32) covers the encoder and the decoder; the heads below
    # turn it off on BOTH device types -- torch.cuda.amp.autocast(enabled=False) left a CPU autocast on
    dev_type = 'cuda' if str(device).startswith('cuda') else 'cpu'
    reduced = dict(device_type=dev_type, dtype=INFERENCE_AUTOCAST, enabled=INFERENCE_AUTOCAST is not None)

    # hillview: the encoder runs once per VIEW across a forward_mast3r call, not once per pair
    # (ENCODE EACH VIEW ONCE there); `encodings` is that call's {instance: (features, positions)}
    with torch.autocast(**reduced):
        (feat1, pos1), (feat2, pos2) = (_view_encoding(model, encodings, v, im, sh)
                                        for v, im, sh in ((view1, img1, shape1), (view2, img2, shape2)))

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
        with torch.autocast(**reduced):
            dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.autocast(device_type=dev_type, enabled=False):
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
        return res1, res2

    # decoder 1-2
    res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)
    # decoder 2-1
    res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

    return (res11, res21, res22, res12)


def _view_encoding(model, encodings, view, img, shape):
    """hillview: a view's encoder output (features, positions), made on its first computed pair and
    reused by every later one. An image is encoded ALONE here, where _encode_image_pairs batches a
    same-shaped pair: on CPU the two are bit-identical (max |diff| 0 alone vs batched, and batched
    with either of two partners, real checkpoint). dust3r's _encode_image returns
    (x, pos, None), the DUNE variant's (features, pos): both start with the two used here."""
    key = view['instance']
    if key not in encodings:
        out = model._encode_image(img, shape)
        encodings[key] = (out[0], out[1])
    return encodings[key]


def extract_correspondences(feats, qonfs, subsample=8, device=None, ptmap_key='pred_desc',
                            seed_masks=None, seed_fill=False):   # hillview: match only what is not masked
    feat11, feat21, feat22, feat12 = feats
    qonf11, qonf21, qonf22, qonf12 = qonfs
    assert feat11.shape[:2] == feat12.shape[:2] == qonf11.shape == qonf12.shape
    assert feat21.shape[:2] == feat22.shape[:2] == qonf21.shape == qonf22.shape

    if '3d' in ptmap_key:
        opt = dict(device='cpu', workers=32)
    else:
        opt = dict(device=device, dist='dot', block_size=2**13)

    # matching the two pairs
    idx1 = []
    idx2 = []
    qonf1 = []
    qonf2 = []
    # TODO add non symmetric / pixel_tol options
    # hillview: A is always a view-1 descriptor map and B a view-2 one, so one mask each
    mask1, mask2 = (seed_masks or (None, None))
    for A, B, QA, QB in [(feat11, feat21, qonf11.cpu(), qonf21.cpu()),
                         (feat12, feat22, qonf12.cpu(), qonf22.cpu())]:
        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample, ret_xy=False,
                                     seed_mask=mask1, target_mask=mask2, seed_fill=seed_fill, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample, ret_xy=False,
                                     seed_mask=mask2, target_mask=mask1, seed_fill=seed_fill, **opt)

        idx1.append(np.r_[nn1to2[0], nn2to1[1]])
        idx2.append(np.r_[nn1to2[1], nn2to1[0]])
        qonf1.append(QA.ravel()[idx1[-1]])
        qonf2.append(QB.ravel()[idx2[-1]])

    # merge corres from opposite pairs
    H1, W1 = feat11.shape[:2]
    H2, W2 = feat22.shape[:2]
    cat = np.concatenate

    xy1, xy2, idx = merge_corres(cat(idx1), cat(idx2), (H1, W1), (H2, W2), ret_xy=True, ret_index=True)
    corres = (xy1.copy(), xy2.copy(), np.sqrt(cat(qonf1)[idx] * cat(qonf2)[idx]))

    return todevice(corres, device)


def sample_at_anchors(X, C, subsample):
    """A partner's predicted pointmap X (H, W, 3) and its confidence C, kept at the ANCHOR pixels.

    hillview: the anchor grid -- the pixels the core depths
    are measured at (anchor_depth_offsets, canonical_view) -- is centred, S//2, S//2 + S, ...; these
    were sampled from the corner, X[::S, ::S], so anchor (4, 4) was handed the prediction for
    (0, 0): on a planar X(x, y) = (x, y, 1) the fallback target came out (0, 0, 1) for a point at
    (4, 4, 1). Sampled on the anchors' own grid the arrays also have the anchor grid's shape when
    a side is not a multiple of S, where the corner grid had one column or row more and every
    later row's index was off. A changed representation, so the shared cache key is canon-v2."""
    S = slice(subsample // 2, None, subsample)
    return X[S, S].reshape(-1, 3), C[S, S].ravel()


def anchor_grid_pixels(W, H, subsample, device=None, dtype=torch.float32):
    """(x, y, 1) of every anchor of a W x H view, in anchor-index order: the pixel its core depth is
    measured at (anchor_depth_offsets' grid), where the solver's point for it has offset 1."""
    cy, cx = np.mgrid[subsample // 2:H:subsample, subsample // 2:W:subsample].reshape(2, -1)
    xy = torch.from_numpy(np.stack([cx, cy], 1).astype(np.float64))
    return torch.cat([xy, torch.ones(len(xy), 1, dtype=xy.dtype)], 1).to(device=device, dtype=dtype)


def _under(path, root):
    return root is not None and os.path.abspath(str(path)).startswith(os.path.join(os.path.abspath(root), ''))


def _source_canonical(restore, vi, img, view_pairs, shared_cache, image_keys, subsample, device):
    """hillview: A RESTORE READS THE SOURCE'S CANONICAL ENTRY of view `vi`, by the name its
    scene.npz keeps, in place of this run's own. Checked against this run: the entry's owner is this view's image (its
    content key; the source's order is this run's), and its stored pair predictions are, one by one in this run's pair
    order, BIT-EQUAL to what this run reads from the same forward entries -- the same partners, the same predictions.
    Every one: the run reads what its source read (SOURCE_FRESH, CANARY_SOLVE_FRESH: the source's canary pairs from the
    source's own recompute, this run's canary never solving on its own), so a prediction that differs is an input that
    differs. Its key could not prove this: it names the entries' sidecars as well (a copy that reorders `.content` and
    its entry leaves the view unkeyable), and a canary-solved pair changes it in either run.
    -> (payload, the file found, how many of its predictions came from the source's own recompute)"""
    from mast3r.cloud_opt.canonical_cache import validate
    src = restore['source']
    name = str(restore['canonical'][vi])
    base = os.path.basename(name)
    owner = base.split('@', 1)[0] if '@' in base else None
    if owner != image_keys[img]:
        raise ValueError(f"restore {src}: this run's optimiser inputs are not the source's: view {vi} "
                         f"({os.path.basename(str(img))}, image {image_keys[img]}) is not the source's {base}"
                         + ('' if owner else ' (a run-local name, no content key)'))
    found = _resolve(os.path.join(shared_cache, name))
    if found is None:
        raise ValueError(f"restore {src}: the source's canonical entry {name} is in no cache root")
    payload = {'value': _cache_load(found, device),
               'predictions': _cache_load(found + '.predictions' + os.path.splitext(found)[1], device)}
    mine = []
    for (img1, img2), ((path1, path2), _) in view_pairs:
        for matches, partner, path in ((img == img1, img2, path1), (img == img2, img1, path2)):
            if matches:
                mine.append((partner, path))
    if len(payload['predictions']) != len(mine):
        raise ValueError(f"restore {src}: this run's optimiser inputs are not the source's: view {vi} "
                         f"({os.path.basename(str(img))}) has {len(mine)} pair prediction(s) here, "
                         f"{len(payload['predictions'])} in the source's canonical entry")
    validate(payload, len(mine))
    source_files = {os.path.abspath(str(p_)) for v_ in SOURCE_FRESH.values() for p_ in v_}
    from_source = 0
    for k, (partner, path) in enumerate(mine):
        if _under(path, CANARY_FRESH_ROOT):
            raise ValueError(f"restore {src}: view {vi}'s pair with {os.path.basename(str(partner))} was solved on "
                             f"THIS run's recompute ({path}): a restore reads the stored files, as its source did")
        from_source += os.path.abspath(str(path)) in source_files
        X, C, X2, C2 = _cache_load(path, device)
        pts, conf = sample_at_anchors(X2, C2, subsample)
        pts_s, conf_s = payload['predictions'][k]
        if not (torch.equal(pts, pts_s.to(pts)) and torch.equal(conf, conf_s.to(conf))):
            raise ValueError(f"restore {src}: this run's optimiser inputs are not the source's: view {vi} "
                             f"({os.path.basename(str(img))}): its prediction {k} (from "
                             f"{os.path.basename(str(partner))}, {path}) is not the source's")
    return payload, found, from_source


@torch.no_grad()
def prepare_canonical_data(imgs, tmp_pairs, subsample, order_imgs=False, min_conf_thr=0,
                           cache_path=None, device='cuda', shared_cache=None, image_keys=None, restore=None, **kw):
    from mast3r.cloud_opt.canonical_cache import cache_path as content_path, index_view_pairs, validate
    import pickle
    if shared_cache is not None and image_keys is None:
        raise ValueError('shared canonical cache requires explicit image_keys')
    if restore is not None:
        if shared_cache is None:
            raise ValueError("restore: the source's canonical entries are read from the shared cache, and none is set")
        if len(restore['canonical']) != len(imgs):
            raise ValueError(f"restore {restore['source']}: this run's optimiser inputs are not the source's: "
                             f"{len(imgs)} canonical views vs {len(restore['canonical'])}")
        if int(restore['subsample']) != int(subsample):
            raise ValueError(f"restore {restore['source']}: anchor grid {subsample} vs the source's {restore['subsample']}")
        if SOURCE_FRESH is None or SOURCE_READS is None or CANARY_SOLVE_FRESH:
            raise ValueError(f"restore {restore['source']}: the caller must say which pairs the source read from its "
                             f"own recompute (SOURCE_FRESH) and which files it read for every other (SOURCE_READS), "
                             f"and keep this run's canary from solving on its own (CANARY_SOLVE_FRESH False) before "
                             f"matching: otherwise its reads are not the source's")
    restore_from_source = 0
    cache_hits = cache_misses = 0
    canonical_views = {}
    pairwise_scores = torch.zeros((len(imgs), len(imgs)), device=device)
    canonical_paths = []
    preds_21 = {}

    by_view = index_view_pairs(tmp_pairs)
    for vi, img in enumerate(tqdm(imgs, desc='canonical views', ascii=' .')):
        view_pairs = by_view.get(img, ())
        cache = None
        payload = None
        shared_path = None
        n_pairs = len(view_pairs)
        # Each incidence supplies one prediction; a self-pair supplies two.
        contributions = sum(sum(img == item for item in pair) for pair, _ in view_pairs)
        if restore is not None:
            # a restore: the source's entry, checked against this run's reads (_source_canonical); read as a hit
            payload, cache, n_src = _source_canonical(restore, vi, img, view_pairs, shared_cache, image_keys,
                                                      subsample, device)
            restore_from_source += n_src
            cache_hits += 1
            (canon, canon2, cconf), focal = payload['value']
        elif shared_cache is not None:
            shared_path = content_path(shared_cache, img, tmp_pairs, image_keys, subsample, device, kw,
                                       view_pairs=view_pairs)
            cache = shared_path
            if cache is not None:
                try:
                    # the value in whichever container holds it; its predictions sit beside it in the same
                    found = _resolve(cache)
                    if found is None:
                        raise FileNotFoundError(cache)
                    payload = {'value': _cache_load(found, device),
                               'predictions': _cache_load(found + '.predictions' + os.path.splitext(found)[1], device)}
                    validate(payload, contributions)
                    if content_path(shared_cache, img, tmp_pairs, image_keys, subsample, device, kw,
                                    view_pairs=view_pairs) != shared_path:
                        payload = None
                except FileNotFoundError:
                    payload = None
                except (OSError, EOFError, ValueError, TypeError, RuntimeError, IndexError, KeyError, pickle.UnpicklingError) as e:
                    print(f' >> canonical cache miss: {type(e).__name__}: {e}')
                    payload = None
            if payload is None:
                cache_misses += 1
                canon = focal = None
            else:
                cache_hits += 1
                cache = found
                (canon, canon2, cconf), focal = payload['value']
            # Legacy forward entries cannot name a shared derived value. Still
            # write an explicit run-local artifact for dense extraction below.
            if cache is None:
                if cache_path is None:
                    raise ValueError('legacy canonical inputs require run-local cache_path')
                cache = os.path.join(cache_path, 'canon_views', hash_md5(img) + f'_{subsample=}_{kw=}{CACHE_EXT}')
        else:
            if cache_path:
                cache = _found(os.path.join(cache_path, 'canon_views', hash_md5(img) + f'_{subsample=}_{kw=}{CACHE_EXT}'))
            try:
                (canon, canon2, cconf), focal = _cache_load(cache, device)
            except IOError:
                # cache does not exist yet, we create it!
                canon = focal = None
        if cache is not None:
            canonical_paths.append(_same_root_name(cache))   # a name, resolved where it is read
        sampled_predictions = []
        prediction_index = 0

        ptmaps11 = None
        confs11 = None          # hillview: paired with ptmaps11, see the guard below
        pixels = {}
        n = 0
        for (img1, img2), ((path1, path2), path_corres) in view_pairs:
            score = None
            if img == img1:
                if payload is None:
                    X, C, X2, C2 = _cache_load(path1, device)
                    prediction = sample_at_anchors(X2, C2, subsample)
                else:
                    prediction = payload['predictions'][prediction_index]
                sampled_predictions.append(prediction)
                prediction_index += 1
                score, (xy1, xy2, confs) = load_corres(path_corres, device, min_conf_thr)
                pixels[img2] = xy1, confs
                if img not in preds_21:
                    preds_21[img] = {}
                # Subsample preds_21
                preds_21[img][img2] = prediction

            if img == img2:
                if payload is None:
                    X, C, X2, C2 = _cache_load(path2, device)
                    prediction = sample_at_anchors(X2, C2, subsample)
                else:
                    prediction = payload['predictions'][prediction_index]
                sampled_predictions.append(prediction)
                prediction_index += 1
                score, (xy1, xy2, confs) = load_corres(path_corres, device, min_conf_thr)
                pixels[img1] = xy2, confs
                if img not in preds_21:
                    preds_21[img] = {}
                preds_21[img][img1] = prediction

            if score is not None:
                i, j = imgs.index(img1), imgs.index(img2)
                # score = score[0]
                # score = np.log1p(score[2])
                score = score[2]
                pairwise_scores[i, j] = score
                pairwise_scores[j, i] = score

                if canon is not None:
                    continue
                if ptmaps11 is None:
                    H, W = C.shape
                    ptmaps11 = torch.empty((n_pairs, H, W, 3), device=device)
                    confs11 = torch.empty((n_pairs, H, W), device=device)

                ptmaps11[n] = X
                confs11[n] = C
                n += 1

        if canon is None:
            # EVERY PAIR OF THIS VIEW WAS DROPPED (hillview). Both buffers are only
            # allocated inside the loop, so a view whose pairs all failed reached canonical_view
            # with confs11 unbound and died as `UnboundLocalError: confs11` -- which names a
            # local variable and not the actual fault. A church ride's run and two queued
            # runs hit it after the pointmap-consistency filter removed 2 pairs.
            # Still a hard failure, deliberately: dropping the view instead would silently
            # change which frames a run solved. It just says what happened now.
            if ptmaps11 is None or confs11 is None:
                raise RuntimeError(
                    f"canonical view: {os.path.basename(str(img))} has no usable pair left -- "
                    f"every pair it appears in was dropped before this point (the pointmap "
                    f"consistency filter and --pair_min_matches are the usual reasons). A view "
                    f"with no pair cannot be reconstructed; drop it from the manifest, or "
                    f"loosen whatever removed its pairs.")
            canon, canon2, cconf = canonical_view(ptmaps11, confs11, subsample, **kw)
            del ptmaps11
            del confs11

        # compute focals
        H, W = canon.shape[:2]
        pp = torch.tensor([W / 2, H / 2], device=device)
        if focal is None:
            focal = estimate_focal_knowing_depth(canon[None], pp, focal_mode='weiszfeld', min_focal=0.5, max_focal=3.5)
            if cache:
                if shared_cache is not None and shared_path is not None:
                    value = {'value': ((canon, canon2, cconf), focal), 'predictions': sampled_predictions}
                    # If an input changed while we read it, publish only run-local.
                    if content_path(shared_cache, img, tmp_pairs, image_keys, subsample, device, kw,
                                    view_pairs=view_pairs) != shared_path:
                        cache = os.path.join(cache_path, 'canon_views', hash_md5(img) + f'_{subsample=}_{kw=}{CACHE_EXT}')
                        canonical_paths[-1] = cache
                        _save_atomic(to_cpu(((canon, canon2, cconf), focal)), mkdir_for(cache))
                    else:
                        _save_atomic(to_cpu(sampled_predictions), mkdir_for(cache + '.predictions' + CACHE_EXT))
                        _save_atomic(to_cpu(value['value']), mkdir_for(cache))
                else:
                    _save_atomic(to_cpu(((canon, canon2, cconf), focal)), mkdir_for(cache))

        # extract depth offsets with correspondences
        # A strided view here keeps the entire H*W*3 canonical pointmap alive.
        # The depth grid is immutable input; copy only its sampled values.
        core_depth = canon[subsample // 2::subsample, subsample // 2::subsample, 2].clone()
        idxs, offsets = anchor_depth_offsets(canon2, pixels, subsample=subsample)

        canonical_views[img] = (pp, (H, W), focal.view(1), core_depth, pixels, idxs, offsets)

    if restore is not None:
        print(f" >> canonical views: the source's {len(imgs)} entries ({restore['source']}), each this run's view; "
              f"every pair prediction bit-equal to this run's reads, {restore_from_source} of them from the "
              f"{len(SOURCE_FRESH)} pair(s) the source's canary solved on its own recompute, read as the source did; "
              f"every other pointmap from the copy the source's record names ({len(SOURCE_READS)} directed pair(s))")
    elif shared_cache is not None:
        print(f' >> canonical cache: {cache_hits}/{cache_hits + cache_misses} hits ({cache_misses} misses)')
    return tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21


def load_corres(path_corres, device, min_conf_thr):
    score, (xy1, xy2, confs) = _cache_load(path_corres, device)
    valid = confs > min_conf_thr if min_conf_thr else slice(None)
    # valid = (xy1 > 0).all(dim=1) & (xy2 > 0).all(dim=1) & (xy1 < 512).all(dim=1) & (xy2 < 512).all(dim=1)
    # print(f'keeping {valid.sum()} / {len(valid)} correspondences')
    return score, (xy1[valid], xy2[valid], confs[valid])


PairOfSlices = namedtuple(
    'ImgPair', 'img1, slice1, pix1, anchor_idxs1, img2, slice2, pix2, anchor_idxs2, confs, confs_sum')
# hillview: pickle finds a namedtuple by its DECLARED name, and this one is declared 'ImgPair' but bound
# to PairOfSlices, so a capture (--capture_solve) of the correspondences failed with "attribute lookup
# ImgPair failed". The alias makes the declared name resolve, both when saving and when loading.
ImgPair = PairOfSlices


def pair_matching_ok(confs, matching_conf_thr):
    """hillview: THE matching gate -- a pair's correspondences enter the 2D and 3D losses only when
    its best one beats matching_conf_thr; the rest is left to the pointmap fallback. A fully
    masked pair has no correspondences: it is a weak pair, not an empty tensor to reduce.
    One definition for the optimiser, the pointmap selection and the per-point `tied` column."""
    return confs.numel() > 0 and bool(confs.max() > matching_conf_thr)


PointPartners = namedtuple('PointPartners', 'img1, slice1, pix1, img2, slice2, pix2, confs, tied')


# hillview: condense_data takes `pointmap_masks`, the PAIR's correspondence masks, and
# removes the pointmap term's targets on masked pixels. The caller refuses a checkout without it.
POINTMAP_MASKS = True


# hillview: WHICH PAIRS CARRY THE POINTMAP TERM IS THE CALLER'S POLICY, passed in (dependency injection: the
# policy keeps one source, the caller's). This module holds upstream's own rule, the expression
# 'default': a pair is weak when no correspondence of it clears `matching_conf_thr` (pair_matching_ok), and the weak
# pairs carry the term (upstream's dust3r_slices). Any other expression needs `pointmap_policy`, a callable
# (expression, pairs, matched, inputs) -> one decision dict per pair, in order, each with view_i, view_j and
# selected. So a stock call imports nothing of the caller's. condense_data evaluates it once, in the calling process;
# sparse_scene_optimizer receives the decisions (`pointmap_selection`), which cross into an optimiser process as data,
# where a callable would not. The caller refuses a checkout without it.
POINTMAP_POLICY = True


def select_pointmap_terms(pairs, matched, expression, inputs=None, policy=None):
    """One decision per (view_i, view_j) of `pairs`, in order: does the pair carry the pointmap term.
    `matched` holds the pairs whose matching is trusted (pair_matching_ok). Without a policy only 'default' is known:
    upstream's rule, the pairs not matched."""
    if policy is None:
        if expression != 'default':
            raise ValueError(f'pointmap_pairs {expression!r} needs a pointmap_policy: this module knows only '
                             f"upstream's rule, 'default' (the pairs whose matching is not trusted)")
        return [dict(view_i=int(i), view_j=int(j), default=(i, j) not in matched, selected=(i, j) not in matched)
                for i, j in pairs]
    decisions = policy(expression, [(int(i), int(j)) for i, j in pairs], set(matched), inputs)
    got = [(d['view_i'], d['view_j']) for d in decisions]
    if got != [(int(i), int(j)) for i, j in pairs]:
        raise ValueError(f'pointmap_policy answered {len(got)} decision(s) not in the order of the '
                         f'{len(pairs)} pair(s) it was asked about')
    return decisions


def corres_pointmap_selection(corres, matching_conf_thr, expression, inputs=None, policy=None):
    """select_pointmap_terms over the pairs of `corres` (condense_data's), in its order: the decisions
    sparse_scene_optimizer takes as `pointmap_selection`. Matched is pair_matching_ok at `matching_conf_thr`,
    the optimiser's own gate."""
    slices = corres[2]
    matched = {(s.img1, s.img2) for s in slices if pair_matching_ok(s.confs, matching_conf_thr)}
    return select_pointmap_terms([(int(s.img1), int(s.img2)) for s in slices], matched, expression, inputs, policy)


def pointmap_mask_keep(mask, shape, subsample, idxs):
    """Which pointmap targets of a view survive its mask: one bool per entry of `idxs`.

    hillview. The pointmap term's targets for view j in pair (i, j) are one per
    correspondence of j -- with ANY partner -- at that correspondence's anchor cell (`idxs`, from
    img_anchors), predicted by the network at the cell's centre (sample_at_anchors: S//2, S//2 + S,
    ...). A correspondence never lands on a pixel its own pair masks, but j's correspondences with
    OTHER partners do enter this pair's term: a view whose foliage is kept against a partner that
    keeps its own (hillview's pair rule) brings that foliage into every pair that drops it, and a
    cell straddling a mask's edge is predicted at a centre inside the mask. `mask` is j's mask in
    THIS pair, (H, W) bool at the loaded resolution, True = masked; None keeps every target."""
    if mask is None:
        return None
    mask = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
    if tuple(mask.shape) != tuple(int(x) for x in shape):
        raise ValueError(f'pointmap mask of shape {tuple(mask.shape)} for a view of shape '
                         f'{tuple(int(x) for x in shape)}: a mask made at another resolution')
    S = slice(subsample // 2, None, subsample)
    grid = mask[S, S].reshape(-1)
    return ~grid.to(idxs.device)[idxs]


# hillview: the dense consistency term reads only the pixels its pair's matching reads.
# sparse_global_alignment takes `dense_masks`; the caller refuses a checkout without it.
DENSE_MASKS = True


def dense_pair_masks(pairs, imgs, shapes, dense_masks, device):
    """The masks the dense term's pairs are matched under, as tensors that survive the optimiser's process.

    hillview. The dense term sampled every view's whole grid: on one phone walk 29 % of its samples
    were blue sky, and movers and anonymisation boxes went
    in too -- depth guesses no correspondence would ever be made on, compared across views. `dense_masks`
    is (img_i, img_j) -> (mask_i, mask_j), the PAIR's correspondence masks, (H, W) bool at the loaded
    resolution, True = masked, None = nothing masked (hillview recon_pair_masks.PairMasks.pair: the hard
    masks per view, a view's foliage dropped exactly where the pair's matching drops it).

    Returns (views, pair_var): views[v] is a bool tensor (n_variants, H, W) of view v's distinct masks
    over all its pairs (hard and composed: at most two), pair_var[(i, j)] = (variant of i, variant of j),
    -1 for no mask. A mask made at another resolution is refused."""
    views, keys, pair_var = {}, {}, {}

    def variant(v, mask):
        if mask is None:
            return -1
        mask = np.asarray(mask, dtype=bool)
        if tuple(mask.shape) != tuple(int(x) for x in shapes[v]):
            raise ValueError(f'dense mask of shape {tuple(mask.shape)} for view {v} of shape '
                             f'{tuple(int(x) for x in shapes[v])}: a mask made at another resolution')
        if not mask.any():
            return -1
        key = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        known = keys.setdefault(v, [])
        if key not in known:
            known.append(key)
            views.setdefault(v, []).append(torch.from_numpy(mask))
        return known.index(key)

    for i, j in pairs:
        mi, mj = dense_masks(imgs[i], imgs[j])
        pair_var[(i, j)] = (variant(i, mi), variant(j, mj))
    return {v: torch.stack(ms).to(device) for v, ms in views.items()}, pair_var


def dense_source_keep(pix, views, var):
    """The grid samples of a source view that are not on its mask in the pair: indices into `pix` ((N, 2)
    x, y at pixel centres), or None for all of them (no mask)."""
    if var < 0:
        return None
    m = views[var]
    return (~m[pix[:, 1].long(), pix[:, 0].long()]).nonzero().squeeze(1)


def dense_target_masked(var, views, vi, ui):
    """Which samples land on a masked pixel of the target view in their pair. `var` is each sample's
    variant of the target's masks (-1: none), `vi`/`ui` the pixel each sample reads the target's depth at."""
    if views is None:
        return torch.zeros(len(var), dtype=torch.bool, device=var.device)
    hit = var >= 0
    out = torch.zeros(len(var), dtype=torch.bool, device=var.device)
    if bool(hit.any()):
        out[hit] = views[var[hit], vi[hit], ui[hit]]
    return out


def condense_data(imgs, tmp_paths, canonical_views, preds_21, dtype=torch.float32, *,
                  pointmap_pairs=None, pointmap_pair_inputs=None, matching_conf_thr=5.,
                  pointmap_masks=None, subsample=None, pointmap_policy=None, selection_out=None):
    # hillview: `pointmap_masks` (i, j) -> (mask_i, mask_j), the pair's correspondence masks, or None
    # for the historical targets; with it, the targets on view j's masked pixels lose their
    # confidence (pointmap_mask_keep), and `subsample` locates the anchor cells' centres
    if pointmap_masks is not None and not subsample:
        raise ValueError('condense_data: pointmap_masks needs the anchor grid\'s subsample')
    # aggregate all data properly
    set_imgs = set(imgs)

    principal_points = []
    shapes = []
    focals = []
    core_depth = []
    img_anchors = {}
    tmp_pixels = {}

    for idx1, img1 in enumerate(imgs):
        # load stuff
        pp, shape, focal, anchors, pixels_confs, idxs, offsets = canonical_views[img1]

        principal_points.append(pp)
        shapes.append(shape)
        focals.append(focal)
        core_depth.append(anchors)

        img_uv1 = []
        img_idxs = []
        img_offs = []
        cur_n = [0]

        for img2, (pixels, match_confs) in pixels_confs.items():
            if img2 not in set_imgs:
                continue
            assert len(pixels) == len(idxs[img2]) == len(offsets[img2])
            img_uv1.append(torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=-1))
            img_idxs.append(idxs[img2])
            img_offs.append(offsets[img2])
            cur_n.append(cur_n[-1] + len(pixels))
            # store the position of 3d points
            tmp_pixels[img1, img2] = pixels.to(dtype), match_confs.to(dtype), slice(*cur_n[-2:])
        img_anchors[idx1] = (torch.cat(img_uv1), torch.cat(img_idxs), torch.cat(img_offs))

    all_confs = []
    imgs_slices = []
    corres2d = {img: [] for img in range(len(imgs))}

    for img1, img2 in tmp_paths:
        try:
            pix1, confs1, slice1 = tmp_pixels[img1, img2]
            pix2, confs2, slice2 = tmp_pixels[img2, img1]
        except KeyError:
            continue
        img1 = imgs.index(img1)
        img2 = imgs.index(img2)
        confs = (confs1 * confs2).sqrt()

        # prepare for loss_3d
        all_confs.append(confs)
        anchor_idxs1 = canonical_views[imgs[img1]][5][imgs[img2]]
        anchor_idxs2 = canonical_views[imgs[img2]][5][imgs[img1]]
        imgs_slices.append(PairOfSlices(img1, slice1, pix1, anchor_idxs1,
                                        img2, slice2, pix2, anchor_idxs2,
                                        confs, float(confs.sum())))

        # prepare for loss_2d
        corres2d[img1].append((pix1, confs, img2, slice2))
        corres2d[img2].append((pix2, confs, img1, slice1))

    all_confs = torch.cat(all_confs)
    corres = (all_confs, float(all_confs.sum()), imgs_slices)

    def aggreg_matches(img1, list_matches):
        pix1, confs, img2, slice2 = zip(*list_matches)
        all_pix1 = torch.cat(pix1).to(dtype)
        all_confs = torch.cat(confs).to(dtype)
        return img1, all_pix1, all_confs, float(all_confs.sum()), [(j, sl2) for j, sl2 in zip(img2, slice2)]
    corres2d = [aggreg_matches(img, m) for img, m in corres2d.items()]

    imsizes = torch.tensor([(W, H) for H, W in shapes], device=pp.device)  # (W,H)
    principal_points = torch.stack(principal_points)
    focals = torch.cat(focals)

    # Only selected pointmaps enter either optimiser loss or pose initialisation.
    # None retains the historical all-pairs preparation API for direct callers.
    selected = None
    if pointmap_pairs is not None:
        decisions = corres_pointmap_selection(corres, matching_conf_thr, pointmap_pairs, pointmap_pair_inputs,
                                              pointmap_policy)
        selected = {(imgs[p['view_j']], imgs[p['view_i']]) for p in decisions if p['selected']}
        # the decisions as data, for sparse_scene_optimizer (POINTMAP_POLICY): evaluated once, here
        if selection_out is not None:
            selection_out[:] = decisions
    # Subsample preds_21
    subsamp_preds_21 = {}
    _pm = dict(pairs=0, touched=0, emptied=0, conf=0.0, removed=0.0)
    for imk, imv in preds_21.items():
        subsamp_preds_21[imk] = {}
        for im2k, (pred, conf) in preds_21[imk].items():
            if selected is not None and (imk, im2k) not in selected:
                continue
            idxs = img_anchors[imgs.index(im2k)][1]
            # one prediction per ANCHOR of im2k, so anchor k's target is the prediction at anchor
            # k's own pixel (sample_at_anchors); a count that differs is a grid this code did not make
            n_grid = canonical_views[im2k][3].numel()
            if len(pred) != n_grid:
                raise ValueError(f'pointmap prediction {imk} -> {im2k}: {len(pred)} samples for '
                                 f'{n_grid} anchors (a prediction sampled on another grid)')
            tgt_conf = conf[idxs]
            if pointmap_masks is not None:
                keep = pointmap_mask_keep(pointmap_masks(imk, im2k)[1], canonical_views[im2k][1],
                                          subsample, idxs)
                c_all = float(tgt_conf.sum())
                _pm['pairs'] += 1
                _pm['conf'] += c_all
                if keep is not None and not bool(keep.all()):
                    tgt_conf = tgt_conf * keep.to(tgt_conf.dtype)
                    _pm['touched'] += 1
                    _pm['removed'] += c_all - float(tgt_conf.sum())
                    _pm['emptied'] += int(not bool(keep.any()))
            subsamp_preds_21[imk][im2k] = (pred[idxs], tgt_conf)  # anchors subsample
    if pointmap_masks is not None:
        print(f' >> pointmap targets masked: {_pm["touched"]} of {_pm["pairs"]} selected pair(s) had '
              f'targets on their masked pixels; {100 * _pm["removed"] / max(_pm["conf"], 1e-12):.2f} % of '
              f'the targets\' confidence removed; {_pm["emptied"]} pair(s) left with none')

    return imsizes, principal_points, focals, core_depth, img_anchors, corres, corres2d, subsamp_preds_21


def canonical_view(ptmaps11, confs11, subsample, mode='avg-angle'):
    assert len(ptmaps11) == len(confs11) > 0, 'not a single view1 for img={i}'

    # canonical pointmap is just a weighted average
    confs11 = confs11.unsqueeze(-1) - 0.999
    canon = (confs11 * ptmaps11).sum(0) / confs11.sum(0)

    canon_depth = ptmaps11[..., 2].unsqueeze(1)
    S = slice(subsample // 2, None, subsample)
    center_depth = canon_depth[:, :, S, S]
    center_depth = torch.clip(center_depth, min=torch.finfo(center_depth.dtype).eps)

    stacked_depth = F.pixel_unshuffle(canon_depth, subsample)
    stacked_confs = F.pixel_unshuffle(confs11[:, None, :, :, 0], subsample)

    if mode == 'avg-reldepth':
        rel_depth = stacked_depth / center_depth
        stacked_canon = (stacked_confs * rel_depth).sum(dim=0) / stacked_confs.sum(dim=0)
        canon2 = F.pixel_shuffle(stacked_canon.unsqueeze(0), subsample).squeeze()

    elif mode == 'avg-angle':
        xy = ptmaps11[..., 0:2].permute(0, 3, 1, 2)
        stacked_xy = F.pixel_unshuffle(xy, subsample)
        B, _, H, W = stacked_xy.shape
        stacked_radius = (stacked_xy.view(B, 2, -1, H, W) - xy[:, :, None, S, S]).norm(dim=1)
        stacked_radius.clip_(min=1e-8)

        stacked_angle = torch.arctan((stacked_depth - center_depth) / stacked_radius)
        avg_angle = (stacked_confs * stacked_angle).sum(dim=0) / stacked_confs.sum(dim=0)

        # back to depth
        stacked_depth = stacked_radius.mean(dim=0) * torch.tan(avg_angle)

        canon2 = F.pixel_shuffle((1 + stacked_depth / canon[S, S, 2]).unsqueeze(0), subsample).squeeze()
    else:
        raise ValueError(f'bad {mode=}')

    confs = (confs11.square().sum(dim=0) / confs11.sum(dim=0)).squeeze()
    return canon, canon2, confs


def anchor_depth_offsets(canon_depth, pixels, subsample=8):
    device = canon_depth.device

    # create a 2D grid of anchor 3D points
    H1, W1 = canon_depth.shape
    yx = np.mgrid[subsample // 2:H1:subsample, subsample // 2:W1:subsample]
    H2, W2 = yx.shape[1:]
    cy, cx = yx.reshape(2, -1)
    core_depth = canon_depth[cy, cx]
    assert (core_depth > 0).all()

    # slave 3d points (attached to core 3d points)
    core_idxs = {}  # core_idxs[img2] = {corr_idx:core_idx}
    core_offs = {}  # core_offs[img2] = {corr_idx:3d_offset}

    for img2, (xy1, _confs) in pixels.items():
        px, py = xy1.long().T

        # find nearest anchor == block quantization
        core_idx = (py // subsample) * W2 + (px // subsample)
        core_idxs[img2] = core_idx.to(device)

        # compute relative depth offsets w.r.t. anchors
        ref_z = core_depth[core_idx]
        pts_z = canon_depth[py, px]
        offset = pts_z / ref_z
        core_offs[img2] = offset.detach().to(device)

    return core_idxs, core_offs


def spectral_clustering(graph, k=None, normalized_cuts=False):
    graph.fill_diagonal_(0)

    # graph laplacian
    degrees = graph.sum(dim=-1)
    laplacian = torch.diag(degrees) - graph
    if normalized_cuts:
        i_inv = torch.diag(degrees.sqrt().reciprocal())
        laplacian = i_inv @ laplacian @ i_inv

    # compute eigenvectors!
    eigval, eigvec = torch.linalg.eigh(laplacian)
    return eigval[:k], eigvec[:, :k]


def sim_func(p1, p2, gamma):
    diff = (p1 - p2).norm(dim=-1)
    avg_depth = (p1[:, :, 2] + p2[:, :, 2])
    rel_distance = diff / avg_depth
    sim = torch.exp(-gamma * rel_distance.square())
    return sim


def backproj(K, depthmap, subsample):
    H, W = depthmap.shape
    uv = np.mgrid[subsample // 2:subsample * W:subsample, subsample // 2:subsample * H:subsample].T.reshape(H, W, 2)
    xyz = depthmap.unsqueeze(-1) * geotrf(inv(K), todevice(uv, K.device), ncol=3)
    return xyz


def spectral_projection_depth(K, depthmap, subsample, k=64, cache_path='',
                              normalized_cuts=True, gamma=7, min_norm=5):
    try:
        # hillview: NOT a shared-cache entry -- a run's own <cache>/lora_depth/ projector
        # (lora_depth=0 by default), never pulled, seeded or converted (hvz_convert walks forward/,
        # canon_views/, corres_*/ and matching/ only), so it stays a torch pickle like a solve capture
        if cache_path:
            cache_path = cache_path + f'_{k=}_norm={normalized_cuts}_{gamma=}.pth'
        lora_proj = torch.load(cache_path, map_location=K.device)

    except IOError:
        # reconstruct 3d points in camera coordinates
        xyz = backproj(K, depthmap, subsample)

        # compute all distances
        xyz = xyz.reshape(-1, 3)
        graph = sim_func(xyz[:, None], xyz[None, :], gamma=gamma)
        _, lora_proj = spectral_clustering(graph, k, normalized_cuts=normalized_cuts)

        if cache_path:
            torch.save(lora_proj.cpu(), mkdir_for(cache_path))

    lora_proj, coeffs = lora_encode_normed(lora_proj, depthmap.ravel(), min_norm=min_norm)

    # depthmap ~= lora_proj @ coeffs
    return coeffs, lora_proj


def lora_encode_normed(lora_proj, x, min_norm, global_norm=False):
    # encode the pointmap
    coeffs = torch.linalg.pinv(lora_proj) @ x

    # rectify the norm of basis vector to be ~ equal
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]
    if global_norm:
        lora_proj *= coeffs[1:].norm() * min_norm / coeffs.shape[1]
    elif min_norm:
        lora_proj *= coeffs.norm(dim=1).clip(min=min_norm)
    # can have rounding errors here!
    coeffs = (torch.linalg.pinv(lora_proj.double()) @ x.double()).float()

    return lora_proj.detach(), coeffs.detach()


@torch.no_grad()
def spectral_projection_of_depthmaps(imgs, intrinsics, depthmaps, subsample, cache_path=None, **kw):
    # recover 3d points
    core_depth = []
    lora_proj = []

    for i, img in enumerate(tqdm(imgs)):
        cache = os.path.join(cache_path, 'lora_depth', hash_md5(img)) if cache_path else None
        depth, proj = spectral_projection_depth(intrinsics[i], depthmaps[i], subsample,
                                                cache_path=cache, **kw)
        core_depth.append(depth)
        lora_proj.append(proj)

    return core_depth, lora_proj


def reproj2d(Trf, pts3d):
    res = (pts3d @ Trf[:3, :3].transpose(-1, -2)) + Trf[:3, 3]
    clipped_z = res[:, 2:3].clip(min=1e-3)  # make sure we don't have nans!
    uv = res[:, 0:2] / clipped_z
    return uv.clip(min=-1000, max=2000)


def bfs(tree, start_node):
    order, predecessors = sp.csgraph.breadth_first_order(tree, start_node, directed=False)
    ranks = np.arange(len(order))
    ranks[order] = ranks.copy()
    return ranks, predecessors


def min_depth_spanning_tree(pws, log=print):
    """(hillview patch) The SHALLOWEST tree the pair graph allows, instead of the strongest one.

    Every camera's pose is composed from its parent's, so the tree's DEPTH is how many rotations
    multiply together before the far end of a span is placed, and an error at any level rides all
    the way down. A maximum spanning tree over match scores makes that depth as large as it can be
    on a walk: the strongest edges are always the consecutive ones, so the tree IS the walk --
    twelve panoramas of the church lane came out depth 12, and their rigs' up turned by 56 and then
    70 degrees at two joints while four panoramas (depth 5) held to 5 degrees.

    A window of w frames means a camera pairs with w neighbours, so a tree of depth about N/w is
    available and a breadth-first tree from a central node finds the shallowest one there is (BFS
    is optimal for depth by construction). The price is that a tree edge is then a FARTHER pair --
    fewer correspondences, a weaker initial relative pose -- so this is a trade to be measured, not
    an obvious win: shallow parameterisation against strong edges. Among the parents that give the
    same depth, the strongest pair wins, which keeps the price as low as the graph allows.

    Returns (root, edges) like compute_min_spanning_tree: edges are (parent, child), parents first.
    """
    pw = pws.detach().cpu().double() if hasattr(pws, 'detach') else torch.as_tensor(np.asarray(pws), dtype=torch.float64)
    n = pw.shape[0]
    sym = torch.maximum(pw, pw.T)
    sym.fill_diagonal_(0)
    adj = [[j for j in range(n) if sym[i, j] > 0] for i in range(n)]

    def bfs(src):
        """(depth, parent) of a breadth-first tree from src; unreached nodes keep parent None."""
        par = [None] * n
        lvl = [-1] * n
        lvl[src], order = 0, [src]
        for i in order:
            for j in sorted(adj[i], key=lambda j_: -float(sym[i, j_])):
                if lvl[j] < 0:
                    lvl[j], par[j] = lvl[i] + 1, i
                    order.append(j)
        return max(lvl), par, lvl, order

    if n == 0:
        return 0, []
    # a graph's centre by the usual double sweep: the far end of a far end, then its midpoint on
    # that path. Exact for a tree and close enough on a banded walk graph, at two BFS passes.
    _, _, lvl0, order0 = bfs(0)
    a_ = order0[-1]
    _, par_a, lvl_a, order_a = bfs(a_)
    b_ = order_a[-1]
    path = [b_]
    while par_a[path[-1]] is not None:
        path.append(par_a[path[-1]])
    root = path[len(path) // 2]
    depth, par, lvl, order = bfs(root)
    unreached = [i for i in range(n) if i != root and par[i] is None]
    if unreached:
        # A DISCONNECTED GRAPH HAS NO TREE, shallow or otherwise. This used to log "falling back to
        # the maximum spanning tree" and call it -- which raises on the same disconnected graph, so
        # the log announced a fallback that never happened (hillview). Said once, here.
        _, labels = sp.csgraph.connected_components(sp.csr_array(sym.numpy() > 0), directed=False)
        raise DisconnectedGraph(f'shallow tree: the pair graph is not connected ({len(unreached)} camera(s) '
                                f'out of {n} unreachable from the root); no spanning tree exists', labels,
                                'Connect the views (pairing, masks) or drop them (--view_unpaired drop).')
    edges = [(par[j], j) for j in order if j != root]
    log(f' >> shallow tree: depth {depth} from root {root} '
        f'(the maximum spanning tree over the same graph would be deeper on a walk)')
    return root, edges


def rig_tree_breaks(mst, rig):
    """{rig: its views split into the pieces the tree's same-rig edges join}, for every rig whose
    views a kinematic tree does NOT hold together; {} when the tree is rig-shaped.

    hillview: the optimizer fixes a rig only along tree edges
    whose two ends are slices of the same rig (rig_edges), so a rig's slices are bound together
    only through a chain of such edges. rig_spanning_tree builds trees that way; a PINNED tree
    (kinematic parents) need not be: 0 -> 1 -> 2 with 0 and 2 slices of one panorama and 1 another
    photograph left 0 and 2 free of each other, and they came out 0.34 units apart."""
    up = list(range(len(rig)))

    def find(x):
        while up[x] != x:
            up[x] = up[up[x]]
            x = up[x]
        return x
    for i, j in mst[1]:
        if rig[i] is not None and rig[j] is not None and rig[i][0] == rig[j][0]:
            up[find(i)] = find(j)
    pieces = {}
    for v, r in enumerate(rig):
        if r is not None:
            pieces.setdefault(r[0], {}).setdefault(find(v), []).append(v)
    return {g: sorted(p.values()) for g, p in pieces.items() if len(p) > 1}


def refuse_rig_breaking_tree(mst, rig):
    broken = rig_tree_breaks(mst, rig)
    if broken:
        raise ValueError('kinematic tree breaks ' + str(len(broken)) + ' rig(s) -- their slices are joined only '
                         'through other views, so their shared centre and fixed rotations would not be held: '
                         + '; '.join(f'rig {g!r}: views {pieces}' for g, pieces in broken.items())
                         + '. A pinned tree must reach each rig once and hang its other slices off that view '
                         '(as the solve that logged it built it).')


def rig_spanning_tree(pws, rig, shallow=False):
    """(hillview patch) A kinematic tree that enters every rig exactly once.

    Groups = rigs (views sharing a centre) plus singletons. The tree over groups is the
    minimum spanning tree of the best pairwise score between any two of their views; each
    group edge is realised by that best view pair, and on entering a group its remaining
    views hang directly off the entry view — those are the edges the optimizer holds fixed.
    """
    n = pws.shape[0]
    gid = [(r[0] if r is not None else ('single', i)) for i, r in enumerate(rig)]
    groups = {}
    for i, g in enumerate(gid):
        groups.setdefault(g, []).append(i)
    keys = list(groups)
    G = len(keys)
    gscore = torch.zeros((G, G), dtype=torch.float64)
    best_pair = {}
    pw = pws.detach().cpu().double()
    for a in range(G):
        for b in range(a + 1, G):
            ia, ib = groups[keys[a]], groups[keys[b]]
            block = pw[ia][:, ib]
            v, flat = block.max(), int(block.argmax())
            v2 = pw[ib][:, ia].max()
            if v2 > v:
                block2 = pw[ib][:, ia]; flat2 = int(block2.argmax())
                j, i = ib[flat2 // len(ia)], ia[flat2 % len(ia)]
                v = v2
            else:
                i, j = ia[flat // len(ib)], ib[flat % len(ib)]
            if v > 0:
                gscore[a, b] = gscore[b, a] = v
                best_pair[(a, b)] = (i, j); best_pair[(b, a)] = (j, i)
    try:
        if G == 1:
            groot, gedges = 0, []
        elif shallow:
            # the rigs themselves are the walk: their tree is where depth accumulates (the slices
            # always hang one level off their own rig's entry view)
            groot, gedges = min_depth_spanning_tree(gscore)
        else:
            groot, gedges = compute_min_spanning_tree(gscore)
    except DisconnectedGraph as e:
        # the pieces were found over GROUPS (a rig is one node): say them in views
        e.over_groups([groups[k] for k in keys])
        raise
    # the root group's entry view: its best-connected member
    ia = groups[keys[groot]]
    root = ia[int(pw[ia].sum(1).argmax())] if len(ia) > 1 else ia[0]
    edges = [(root, k) for k in ia if k != root]
    for ga, gb in gedges:
        i, j = best_pair[(ga, gb)]
        edges.append((i, j))
        edges += [(j, k) for k in groups[keys[gb]] if k != j]
    return root, edges


class DisconnectedGraph(ValueError):
    """hillview: NO KINEMATIC TREE, the pair graph is in pieces -- and WHERE it split.

    `components` holds each piece's view indices, pieces in order of their first view. The message
    names them as index ranges, and by file once sparse_global_alignment has called name_views,
    which also logs the pairs that were asked for across the cut. A 10-frame span once refused with
    "2 disconnected components", and nothing recorded which frames sat on either side, so a
    suspected blind stretch between two frames could not be confirmed. Only the refusal changes; a connected graph never builds one."""

    def __init__(self, head, labels, advice):
        super().__init__(head)
        self.head, self.advice, self.names, self.cut = head, advice, None, None
        pieces = {}
        for v, c in enumerate(labels):
            pieces.setdefault(int(c), []).append(v)
        self.components = sorted(pieces.values())

    def over_groups(self, members):
        """The pieces were found over groups (a rig's views are one node): node k stands for the
        views members[k]."""
        self.components = sorted(sorted(int(v) for k in c for v in members[k]) for c in self.components)

    def name_views(self, imgs, asked, reached, log=print):
        """Name the pieces by file, and LOG every pair asked for across the cut, unordered: none
        gave the tree an edge -- it was removed before the solve (the caller's pair gates), or it
        reached the solve with no correspondence (matches masked away or none found)."""
        self.names = [os.path.basename(str(p)) for p in imgs]
        piece = {v: k for k, c in enumerate(self.components) for v in c}
        reached = {tuple(sorted(p)) for p in reached}
        self.cut = [(i, j, (i, j) in reached)
                    for i, j in sorted({tuple(sorted((int(a), int(b)))) for a, b in asked})
                    if piece[i] != piece[j]]
        log(f' >> kinematic tree refused: {len(self.components)} pieces; {len(self.cut)} pair(s) were asked for '
            f'across the cut, and none has a correspondence:')
        for i, j, r in self.cut:
            log(f'    [{piece[i] + 1}] {i} {self.names[i]}  -  [{piece[j] + 1}] {j} {self.names[j]}: '
                + ('reached the solve with 0 correspondences' if r else 'removed before the solve'))

    def __str__(self):
        def runs(views):
            out = []
            for v in views:
                if out and v == out[-1][1] + 1:
                    out[-1][1] = v
                else:
                    out.append([v, v])
            return out

        def run(a, b):
            ids = f'{a}' if a == b else f'{a}-{b}'
            if self.names is None:
                return ids
            return f'{ids} {self.names[a]}' if a == b else f'{ids} {self.names[a]} .. {self.names[b]}'
        return (f'{self.head}: '
                + '; '.join(f'[{k + 1}] {len(c)} view(s) ' + ', '.join(run(a, b) for a, b in runs(c))
                            for k, c in enumerate(self.components))
                + '. ' + self.advice)


def compute_min_spanning_tree(pws):
    if pws.shape[0] <= 1:
        return 0, []
    sparse_graph = sp.dok_array(pws.shape)
    for i, j in pws.nonzero().cpu().tolist():
        sparse_graph[i, j] = -float(pws[i, j])
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph)
    components, labels = sp.csgraph.connected_components(msp, directed=False)
    if components != 1:
        raise DisconnectedGraph(f'kinematic tree: correspondence graph has {components} disconnected components',
                                labels, 'Zero-match pairs provide no correspondence edge. Use an explicit '
                                'chain/parent prior, or change pairing/masks to connect the views.')

    # now reorder the oriented edges, starting from the central point
    ranks1, _ = bfs(msp, 0)
    ranks2, _ = bfs(msp, ranks1.argmax())
    ranks1, _ = bfs(msp, ranks2.argmax())
    # this is the point farther from any leaf
    root = np.minimum(ranks1, ranks2).argmax()

    # find the ordered list of edges that describe the tree
    order, predecessors = sp.csgraph.breadth_first_order(msp, root, directed=False)
    order = order[1:]  # root not do not have a predecessor
    edges = [(predecessors[i], i) for i in order]

    return root, edges


def show_reconstruction(shapes_or_imgs, K, cam2w, pts3d, gt_cam2w=None, gt_K=None, cam_size=None, masks=None, **kw):
    viz = SceneViz()

    cc = cam2w[:, :3, 3]
    cs = cam_size or float(torch.cdist(cc, cc).fill_diagonal_(np.inf).min(dim=0).values.median())
    colors = 64 + np.random.randint(255 - 64, size=(len(cam2w), 3))

    if isinstance(shapes_or_imgs, np.ndarray) and shapes_or_imgs.ndim == 2:
        cam_kws = dict(imsizes=shapes_or_imgs[:, ::-1], cam_size=cs)
    else:
        imgs = shapes_or_imgs
        cam_kws = dict(images=imgs, cam_size=cs)
    if K is not None:
        viz.add_cameras(to_numpy(cam2w), to_numpy(K), colors=colors, **cam_kws)

    if gt_cam2w is not None:
        if gt_K is None:
            gt_K = K
        viz.add_cameras(to_numpy(gt_cam2w), to_numpy(gt_K), colors=colors, marker='o', **cam_kws)

    if pts3d is not None:
        for i, p in enumerate(pts3d):
            if not len(p):
                continue
            if masks is None:
                viz.add_pointcloud(to_numpy(p), color=tuple(colors[i].tolist()))
            else:
                viz.add_pointcloud(to_numpy(p), mask=masks[i], color=imgs[i])
    viz.show(**kw)
