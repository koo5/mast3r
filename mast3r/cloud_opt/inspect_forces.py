"""hillview: what pulled on each camera, and what actually moved it (sparse_scene_optimizer inspect=).

The optimiser's unknowns are not camera poses. Each camera is its parent's pose times a relative
transform (the kinematic tree), plus a per-image scale and the intrinsics; Adam then rescales every
unknown by its own history. So the pull a term exerts on a camera (its gradient on that camera's
pose) is NOT what moves the camera: a pull on camera 5 swings every camera hung below 5, and a
pull on 29 moves 29's own link and everything beyond it.

This module records both, per GROUP -- one group per image pair (its matches, or its pointmap
fallback when the pair is weak) and one per prior term (GPS, gyro, speed, ...):

  force   the group's pull on every camera's pose: minus the gradient of the group's loss on the
          camera's centre (3) and on its orientation (3, a small world-frame rotation)
  disp    how far the group's share of each Adam step moved every camera: the step is split
          exactly (Adam's first moment is linear in the gradient, the second moment is shared),
          each share is applied to the unknowns ALONE and pushed through the tree

Nothing here touches the solve: the real update runs as before, from the real backward pass; every
number is computed beside it from detached copies. Two sums are checked every iteration and stored
(`chk_grad`, `chk_step`): the groups' gradients must add up to the optimiser's own gradient on the
pose unknowns, and the groups' steps to the step it took. A term missing from the groups shows up
there, not as a plausible-looking wrong answer.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch


def _vee_skew(M):
    # d/d(omega) of L(exp(omega^) R) at 0, given M = R G^T with G = dL/dR
    return torch.stack([M[..., 1, 2] - M[..., 2, 1], M[..., 2, 0] - M[..., 0, 2], M[..., 0, 1] - M[..., 1, 0]], -1)


def _rotvec(R):
    # small-angle-safe axis-angle of a batch of rotation matrices
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    th = torch.acos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    k = torch.where(th > 1e-6, th / (2 * torch.sin(th).clamp(min=1e-12)), 0.5 + th ** 2 / 12)
    return v * k[..., None]


class ForceInspector:
    def __init__(self, out_path, bucket, n_imgs, ctx, chunk=2_000_000, debug=False):
        self.out_path, self.bucket, self.N, self.c = out_path, max(1, int(bucket)), n_imgs, ctx
        self.chunk, self.debug = int(chunk), bool(debug)
        c = ctx
        self.device, self.dtype = c['device'], c['dtype']
        N = n_imgs
        # ---- the groups. A pair group is every view pair between two PHOTOGRAPHS of one kind: its
        # trusted matches ('match'), or its pointmap fallback ('fallback'). Without a rig every view is
        # its own photograph; with one, the slices of a panorama are one photograph, so 320 views and
        # 4,878 slice pairs read as 40 photographs and a few hundred photograph pairs. Then one group
        # per prior term, and NUMERICS last.
        rig = c.get('rig')
        ph_raw = [(r[0] if (rig is not None and rig[i] is not None) else ('v', i)) for i, r in enumerate(rig or [None] * N)]
        ph_ids = {}
        self.photo_of = np.asarray([ph_ids.setdefault(x, len(ph_ids)) for x in ph_raw])
        ph = self.photo_of
        # one view per photograph carries the Jacobian: a rig's slices share its centre and turn with it
        self.rep_views = torch.as_tensor([int(np.nonzero(ph == p_)[0][0]) for p_ in range(int(ph.max()) + 1)], device=self.device)
        self.view_rep = torch.as_tensor(ph, device=self.device)
        self._tm = {}
        keys = {}

        def gid(i, j, kind):
            pa, pb = sorted((int(ph[i]), int(ph[j])))
            return keys.setdefault((pa, pb, kind), len(keys))
        lut_m = torch.full((N, N), -1, dtype=torch.long, device=self.device)
        lut_f = torch.full((N, N), -1, dtype=torch.long, device=self.device)
        for s in c['loss3d_slices']:
            lut_m[s.img1, s.img2] = lut_m[s.img2, s.img1] = gid(s.img1, s.img2, 'match')
        for s in c['dust3r_slices']:
            lut_f[s.img1, s.img2] = lut_f[s.img2, s.img1] = gid(s.img1, s.img2, 'fallback')
        order = sorted(keys, key=lambda k: (k[0], k[1], k[2]))
        remap = torch.full((max(len(keys), 1),), -1, dtype=torch.long, device=self.device)
        for new_id, k in enumerate(order):
            remap[keys[k]] = new_id
        lut_m = torch.where(lut_m >= 0, remap[lut_m.clamp(min=0)], lut_m)
        lut_f = torch.where(lut_f >= 0, remap[lut_f.clamp(min=0)], lut_f)
        self.pair_keys = [(a_, b_) for a_, b_, _ in order]
        self.pair_kind = [k_ for _, _, k_ in order]
        self.P = len(order)
        self.prior_names = [n for n, _ in c['priors']]
        # the last group is NUMERICS: whatever the optimiser's own float32 gradient holds that the
        # groups (computed in float64) do not explain. Every gauge direction -- the root's pose, the
        # scene's overall scale -- is a sum of large pulls that cancel exactly in exact arithmetic,
        # so there the float32 sum is mostly round-off, and the round-off moves cameras too.
        self.G = self.P + len(self.prior_names) + 1
        self.group_names = [f'{"m" if k == "match" else "f"}:{a_}-{b_}' for (a_, b_), k in zip(self.pair_keys, self.pair_kind)] \
            + [f'prior:{n}' for n in self.prior_names] + ['numerics']
        # ---- every correspondence side reads its camera from a COPY owned by (group, view): one
        # backward pass then yields each group's pull on each camera separately
        img_of = c['_img_of']
        sides = []
        self._l3 = self._l2 = self._d3 = None
        if c.get('_idx3d_1') is not None:
            i1, i2 = c['_idx3d_1'], c['_idx3d_2']
            g = lut_m[img_of[i1], img_of[i2]]
            assert bool((g >= 0).all())
            self._l3 = dict(i1=i1, i2=i2, conf=c['_confs3d'], pid=g)
            sides += [('_l3', 'q1', g, img_of[i1]), ('_l3', 'q2', g, img_of[i2])]
        if c.get('_l2_idx') is not None:
            idx, obs = c['_l2_idx'], c['_l2_img_of']
            g = lut_m[obs, img_of[idx]]
            assert bool((g >= 0).all()), 'a 2D correspondence on a pair the 3D loss does not know'
            self._l2 = dict(idx=idx, pix=c['_l2_pix'], conf=c['_l2_conf'], pid=g)
            sides += [('_l2', 'qs', g, img_of[idx]), ('_l2', 'qo', g, obs)]
        if c.get('_d3_src') is not None:
            src_a, cam = c['_d3_src'], c['_d3_cam']
            g = lut_f[img_of[src_a], cam]
            assert bool((g >= 0).all())
            self._d3 = dict(src=src_a, pix=c['_d3_pix'], tgt=c['_d3_tgt'], conf=c['_d3_conf'], cfsum=c['_d3_cfsum'], pid=g)
            sides += [('_d3', 'qs', g, img_of[src_a]), ('_d3', 'qc', g, cam)]
        if sides:
            allkey = torch.cat([g_ * N + v_ for _, _, g_, v_ in sides])
            uniq, inv = torch.unique(allkey, return_inverse=True)
            off = 0
            for name, field, g_, _ in sides:
                getattr(self, name)[field] = inv[off:off + len(g_)]
                off += len(g_)
            self.copy_img, self.copy_grp = uniq % N, uniq // N
        else:
            self.copy_img = self.copy_grp = torch.zeros(0, dtype=torch.long, device=self.device)
        # ---- storage, filled per bucket
        self.rec = dict(snap=[], disp=[], actual=[], force=[], gloss=[], it=[], stage=[], chk_grad=[], chk_step=[],
                        lin_resid=[], scale=[])
        self._acc = None
        self.stage = None
        self._warned = False
        self.t_spent = 0.0
        self._meta_for_partial = (c['imgs'], c['mst']) if 'imgs' in c else None

    def _tick(self, name):
        now = time.time()
        self._tm[name] = self._tm.get(name, 0.0) + now - getattr(self, '_t_last', now)
        self._t_last = now

    # ------------------------------------------------------------------ the camera interface
    def _pose_params(self):
        c = self.c
        seen, out, names = set(), [], []
        for kind in ('quats', 'trans', 'log_sizes', 'log_focals', 'pps'):
            for i, p in enumerate(c[kind]):
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p)); out.append(p); names.append(f'{kind}[{i}]')
        self._pnames = names
        return out

    def _iface(self, params_override=None):
        """(cam2w, K, B, C) from the pose unknowns, depth = B + C * u per anchor (u from core_depth)."""
        c = self.c
        q, t, ls, lf, pp = c['quats'], c['trans'], c['log_sizes'], c['log_focals'], c['pps']
        if params_override is not None:
            m = params_override
            q = [m.get(id(x), x) for x in q]; t = [m.get(id(x), x) for x in t]; ls = [m.get(id(x), x) for x in ls]
            lf = [m.get(id(x), x) for x in lf]; pp = [m.get(id(x), x) for x in pp]
        dummy = [torch.zeros(1, device=self.device, dtype=self.dtype) for _ in range(self.N)]
        # make_K_cam_depth memoises the depth it built for make_pts3d_fast; this call must not leave
        # the placeholder depth there in the solve's place
        saved = list(c['_depth_flat'])
        # the rig links batched for THESE calls only: exact in the forward pass, and only their gradient
        # summation order differs -- which matters to the solve's trajectory (sparse_ga, church6), not to
        # a split computed beside it. The solve keeps its own path; the set is restored either way.
        slow = c.get('slow_parts')
        had_rig = slow is not None and 'rig' in slow
        if had_rig:
            slow.discard('rig')
        try:
            K, (_, cam2w), _ = c['make_K_cam_depth'](lf, pp, t, q, ls, dummy)
        finally:
            c['_depth_flat'][:] = saved
            if had_rig:
                slow.add('rig')
        sizes = torch.cat(ls).exp()
        gs = 1 / sizes.min()
        focals = c['focals_of'](lf)         # the solve's own: a held known focal is not clipped
        zc = sizes * c['median_depths'] * focals / c['base_focals']
        if c['depth_mode'] == 'mul':
            B, C = torch.zeros_like(zc), gs * zc
        else:
            B, C = gs * zc, gs * c['median_depths'] * sizes
        return cam2w, K, B, C

    def _u(self):
        c = self.c
        cd = torch.cat([d.detach() for d in c['core_depth']])
        if c['exp_depth']:
            cd = cd.exp()
        return cd if c['depth_mode'] == 'mul' else cd - 1

    # ------------------------------------------------------------------ one iteration
    def before_step(self, it, niter, stage, pixel, pix_loss, lr):
        """after loss.backward(), before optimizer.step(): every group's pull, mapped to the unknowns."""
        t0 = time.time()
        self._t_last = t0
        c, N, P, G = self.c, self.N, self.P, self.G
        if stage != self.stage:
            self._flush()
            self.stage, self._mom = stage, None
            if c['loss2_3d_w'] or c['dense_consistency_w']:
                raise NotImplementedError('inspect: stage-2 3D tie / dense consistency are not split into groups yet')
        self._it, self._niter = it, niter
        params = self._pose_params()
        self._params = params
        cam2w_F, K_F, B_F, C_F = self._iface()
        with torch.no_grad():
            # the interface must reproduce the solve's own depth, or the split is of something else
            u = self._u()
            d_main = c['_depth_flat'][0]
        # ---- every group's pull on every camera it touches, as gradients on (group, view) COPIES of
        # the camera quantities: cam2w (4x4), K (3x3), and the depth scale B + C*u
        nq = len(self.copy_img)
        u = u.double()
        gl = torch.zeros(G, dtype=torch.float64, device=self.device)   # each group's loss
        acc = [torch.zeros(nq, 4, 4, dtype=torch.float64, device=self.device),
               torch.zeros(nq, 3, 3, dtype=torch.float64, device=self.device),
               torch.zeros(nq, dtype=torch.float64, device=self.device),
               torch.zeros(nq, dtype=torch.float64, device=self.device)]
        acc_t = torch.zeros(nq, 3, dtype=torch.float64, device=self.device)   # contiguous, folded in below
        acc_R = torch.zeros(nq, 9, dtype=torch.float64, device=self.device)
        acc_K = torch.zeros(nq, 3, dtype=torch.float64, device=self.device)   # f, cx, cy
        chunk = self.chunk
        Rw = cam2w_F.detach().double()[:, :3, :3]
        tw = cam2w_F.detach().double()[:, :3, 3]
        Kd = K_F.detach().double()
        Bd, Cd = B_F.detach().double(), C_F.detach().double()
        want_k = K_F.requires_grad
        img_of, didx = c['_img_of'], c['_pts_didx'][0]
        bfd = c['base_focals'].double()

        def anchor(a_, at_anchor=None):
            # the camera-frame point of anchors a_ and what it depends on, as make_pts3d_fast builds it;
            # at_anchor: the pixels of their anchors instead, offset 1 -- the pointmap fallback's
            # source (sparse_ga _grid_pts_fast)
            v = img_of[a_]
            f, cx, cy = Kd[v, 0, 0], Kd[v, 0, 2], Kd[v, 1, 2]
            o = c['_offs_all'][a_].double() if at_anchor is None else torch.ones(len(a_), dtype=torch.float64, device=self.device)
            offs = 1 + (o - 1) * bfd[v] / f
            pix = (c['_pix_all'][a_] if at_anchor is None else at_anchor).double()
            ray = torch.stack([(pix[:, 0] - cx) / f, (pix[:, 1] - cy) / f, torch.ones_like(f)], 1)
            uu = u[didx[a_]]
            raw = Bd[v] + Cd[v] * uu
            depth = raw.clamp(min=1e-3)
            live = (raw > 1e-3).double()
            w = offs[:, None] * ray
            p_ = depth[:, None] * w
            X = (Rw[v] @ p_.unsqueeze(-1)).squeeze(-1) + tw[v]
            return dict(v=v, f=f, cx=cx, cy=cy, o=o, bf=bfd[v], offs=offs, pix=pix, ray=ray, uu=uu, depth=depth,
                        live=live, w=w, p=p_, X=X)

        def put(A_, q, gX):
            # dL/d(cam2w) = [gX p^T | gX]; dL/dB, dL/dC through depth; dL/dK through offs and the ray
            acc_t.index_add_(0, q, gX)
            acc_R.index_add_(0, q, (gX[:, :, None] * A_['p'][:, None, :]).reshape(-1, 9))
            gc = torch.einsum('nji,nj->ni', Rw[A_['v']], gX)                 # in the camera frame
            gw = (gc * A_['w']).sum(1) * A_['live']
            acc[2].index_add_(0, q, gw)
            acc[3].index_add_(0, q, gw * A_['uu'])
            if want_k:
                f = A_['f']; d = A_['depth']
                dpf = d[:, None] * (-(A_['o'] - 1) * A_['bf'] / f ** 2)[:, None] * A_['ray'] \
                    + (d * A_['offs'])[:, None] * torch.stack([-(A_['pix'][:, 0] - A_['cx']) / f ** 2,
                                                              -(A_['pix'][:, 1] - A_['cy']) / f ** 2,
                                                              torch.zeros_like(f)], 1)
                acc_K.index_add_(0, q, torch.stack([(gc * dpf).sum(1), -gc[:, 0] * d * A_['offs'] / f,
                                                    -gc[:, 1] * d * A_['offs'] / f], 1))

        if pixel == 'l3d' and self._l3 is not None:
            L = self._l3
            cf = L['conf'].double(); cfs = cf.sum()
            for a0 in range(0, len(cf), chunk):
                sl = slice(a0, min(len(cf), a0 + chunk))
                A1, A2 = anchor(L['i1'][sl]), anchor(L['i2'][sl])
                x1 = A1['X'].requires_grad_(True); x2 = A2['X'].requires_grad_(True)
                per = cf[sl] * pix_loss(x1, x2) / cfs
                g1, g2 = torch.autograd.grad(per.sum(), (x1, x2))
                put(A1, L['q1'][sl], g1); put(A2, L['q2'][sl], g2)
                gl[:P] += torch.zeros(P, dtype=torch.float64, device=self.device).index_add_(0, L['pid'][sl], per.detach())
        elif pixel == 'l2d' and self._l2 is not None:
            # the reprojection term also differentiates through the OBSERVING camera's inverse and K: kept
            # on autograd, with the copies' inverses taken once per copy rather than per correspondence
            L = self._l2
            cf = L['conf'].double(); cfs = cf.sum()
            cq = cam2w_F.detach()[self.copy_img].double().requires_grad_(True)
            kq = K_F.detach()[self.copy_img].double().requires_grad_(True)
            for a0 in range(0, len(cf), chunk):
                sl = slice(a0, min(len(cf), a0 + chunk))
                A1 = anchor(L['idx'][sl])
                x = A1['X'].requires_grad_(True)
                Pq = kq @ torch.linalg.inv(cq)[:, :3]
                Pm = Pq[L['qo'][sl]]
                res = (Pm[:, :3, :3] @ x.unsqueeze(-1)).squeeze(-1) + Pm[:, :3, 3]
                uv = (res[:, 0:2] / res[:, 2:3].clip(min=1e-3)).clip(min=-1000, max=2000)
                per = cf[sl] * pix_loss(L['pix'][sl].double(), uv) / cfs
                gx, gcq, gkq = torch.autograd.grad(per.sum(), (x, cq, kq))
                put(A1, L['qs'][sl], gx)
                acc[0] += gcq; acc[1] += gkq
                gl[:P] += torch.zeros(P, dtype=torch.float64, device=self.device).index_add_(0, L['pid'][sl], per.detach())
        if self._d3 is not None and c['loss_dust3r_w']:
            L = self._d3
            cf = L['conf'].double(); cfs = float(L['cfsum'])
            for a0 in range(0, len(cf), chunk):
                sl = slice(a0, min(len(cf), a0 + chunk))
                A1 = anchor(L['src'][sl], at_anchor=L['pix'][sl])
                tl = L['tgt'][sl].double()
                vj = self.copy_img[L['qc'][sl]]
                x = A1['X'].requires_grad_(True)
                T = ((Rw[vj] @ tl.unsqueeze(-1)).squeeze(-1) + tw[vj]).requires_grad_(True)
                per = c['loss_dust3r_w'] * cf[sl] * c['lossd'](x, T) / cfs
                gx, gT = torch.autograd.grad(per.sum(), (x, T))
                put(A1, L['qs'][sl], gx)
                acc_t.index_add_(0, L['qc'][sl], gT)
                acc_R.index_add_(0, L['qc'][sl], (gT[:, :, None] * tl[:, None, :]).reshape(-1, 9))
                gl[:P] += torch.zeros(P, dtype=torch.float64, device=self.device).index_add_(0, L['pid'][sl], per.detach())
        acc[0][:, :3, 3] += acc_t
        acc[0][:, :3, :3] += acc_R.reshape(-1, 3, 3)
        acc[1][:, 0, 0] += acc_K[:, 0]; acc[1][:, 0, 2] += acc_K[:, 1]; acc[1][:, 1, 2] += acc_K[:, 2]
        self._tick('pixel groups')
        g_c = torch.zeros(G, N, 4, 4, device=self.device, dtype=self.dtype)
        g_k = torch.zeros(G, N, 3, 3, device=self.device, dtype=self.dtype)
        g_b = torch.zeros(G, N, device=self.device, dtype=self.dtype)
        g_d = torch.zeros(G, N, device=self.device, dtype=self.dtype)
        for dst, src in ((g_c, acc[0]), (g_k, acc[1]), (g_b, acc[2]), (g_d, acc[3])):
            dst.index_put_((self.copy_grp, self.copy_img), src.to(dst.dtype), accumulate=True)
        # ---- the priors, one group each, on a detached copy of all cameras
        for gi, (name, fn) in enumerate(c['priors']):
            cl = cam2w_F.detach().clone().requires_grad_(True)
            v = fn(cl)
            if not torch.is_tensor(v) or not v.requires_grad:
                continue
            gl[P + gi] = float(v.detach())
            g_c[P + gi] = torch.autograd.grad(v, cl)[0]
        self._tick('priors')
        # ---- forces in the world: minus the gradient on the centre, and on a world-frame rotation
        R = cam2w_F.detach()[:, :3, :3]
        f_t = -g_c[:, :, :3, 3]
        f_r = -_vee_skew(R.unsqueeze(0) @ g_c[:, :, :3, :3].transpose(-1, -2))
        # ---- the same pulls on the unknowns: one batched pass back through the tree
        # only the outputs the trainable unknowns reach (stage 1 freezes the intrinsics: K has no graph)
        live = [(o, g) for o, g in ((cam2w_F, g_c), (K_F, g_k), (B_F, g_b), (C_F, g_d)) if o.requires_grad]
        outs, gouts = tuple(o for o, _ in live), tuple(g for _, g in live)
        try:
            G_par = torch.autograd.grad(outs, params, grad_outputs=gouts, is_grads_batched=True,
                                        retain_graph=True, allow_unused=True)
        except RuntimeError as e:
            if not self._warned:
                print(f' >> inspect: batched pass back through the tree refused ({e}); one pass per group')
                self._warned = True
            G_par = [torch.stack(z) if z[0] is not None else None for z in zip(*[
                torch.autograd.grad(outs, params, grad_outputs=tuple(x[g] for x in gouts), retain_graph=True,
                                    allow_unused=True) for g in range(G)])]
        G_par = [torch.zeros((G,) + p.shape, device=self.device, dtype=self.dtype) if gp is None else gp.clone()
                 for gp, p in zip(G_par, params)]
        for gp, p in zip(G_par, params):
            # the numerics group (last, all zeros so far) takes what the groups do not explain
            if p.grad is not None:
                gp[-1] = p.grad - gp[:-1].double().sum(0).to(gp.dtype)
        # completeness: the groups must add up to the gradient the optimiser is about to use
        num = sum(float((gp[-1].double() ** 2).sum()) for gp, p in zip(G_par, params) if p.grad is not None)
        den = sum(float((p.grad ** 2).sum()) for p in params if p.grad is not None)
        self._chk_grad = math.sqrt(num / max(den, 1e-30))
        # per unknown: Adam normalises every unknown by its own history, so an unknown with a tiny
        # gradient split 10 % wrong takes a full-size wrong step even when the global norm reads 1e-6
        worst = (0.0, '')
        for gp, p, nm in zip(G_par, params, self._pnames):
            if p.grad is None:
                continue
            e = (gp[-1].abs() / (p.grad.abs() + 1e-12 * float(p.grad.abs().max() + 1e-30))).max()
            if float(e) > worst[0]:
                worst = (float(e), f'{nm} numerics {gp[-1].flatten()[:4].tolist()} of grad {p.grad.flatten()[:4].tolist()}')
        self._worst = worst
        # the tree's Jacobian at this iteration: camera [R|t] (12 per camera) against every trainable
        # unknown, so that a group's share of the step moves the cameras LINEARLY -- the shares then add
        # up exactly to the linearised step, and what is left over is the step's own curvature
        self._tick('pulls -> unknowns')
        nr = len(self.rep_views)
        n12 = nr * 12
        basis = torch.eye(n12, device=self.device, dtype=self.dtype).reshape(n12, nr, 3, 4)
        gout = torch.zeros(n12, self.N, 4, 4, device=self.device, dtype=self.dtype)
        gout[:, self.rep_views, :3, :] = basis
        J = torch.autograd.grad((cam2w_F,), params, grad_outputs=(gout,), is_grads_batched=True,
                                retain_graph=True, allow_unused=True)
        self._J = torch.cat([(torch.zeros((n12,) + p.shape, device=self.device, dtype=self.dtype) if j is None else j).reshape(n12, -1)
                             for j, p in zip(J, params)], 1)                     # (12 photos, n_unknowns)
        self._tick('jacobian')
        self._G_par = G_par
        self._theta0 = [p.detach().clone() for p in params]
        self._cam0 = cam2w_F.detach().clone()
        self._force = torch.cat([f_t, f_r], -1)
        self._gl = gl
        # the depth the interface reproduces, against the solve's own
        with torch.no_grad():
            img = c['_img_of']
            dd = (B_F.detach()[img] + C_F.detach()[img] * u[c['_pts_didx'][0]]).clamp(min=1e-3)
            ref = d_main[c['_pts_didx'][0]]
            self._depth_err = float(((dd - ref).abs().max() / ref.abs().max().clamp(min=1e-12)))
        self.t_spent += time.time() - t0

    def after_step(self, optimizer):
        """after optimizer.step(), before the quaternions are renormalised: split the step itself."""
        t0 = time.time()
        self._t_last = t0
        params, G_par = self._params, self._G_par
        grp = optimizer.param_groups[0]
        b1, b2 = grp['betas']
        eps, lr = grp['eps'], grp['lr']
        if self._mom is None:
            self._mom = {id(p): torch.zeros_like(g) for p, g in zip(params, G_par)}
        steps = []
        num = den = 0.0
        for p, g, th0 in zip(params, G_par, self._theta0):
            st = optimizer.state[p]
            m = self._mom.setdefault(id(p), torch.zeros_like(g))
            m.mul_(b1).add_(g, alpha=1 - b1)
            k = float(st['step'])
            bc1, bc2 = 1 - b1 ** k, 1 - b2 ** k
            denom = st['exp_avg_sq'].sqrt() / math.sqrt(bc2) + eps
            dth = -(lr / bc1) * m / denom
            steps.append(dth)
            actual = p.detach() - th0
            num += float(((dth.sum(0) - actual) ** 2).sum()); den += float((actual ** 2).sum())
        chk_step = math.sqrt(num / max(den, 1e-30))
        self._tick('adam split')
        if chk_step > 1e-2 and self.debug:
            for p, dth, th0, nm in zip(params, steps, self._theta0, self._pnames):
                a_ = p.detach() - th0
                e_ = float((dth.sum(0) - a_).norm() / a_.norm().clamp(min=1e-30))
                if e_ > 1e-2:
                    print(f' >> inspect debug it {self._it}: step split {e_:.2e} on {nm}: '
                          f'split {dth.sum(0).flatten()[:4].tolist()} vs taken {a_.flatten()[:4].tolist()}; '
                          f'worst grad split {self._worst[0]:.2e} ({self._worst[1]})')
        # each group's share through the tree's Jacobian: centre moves, and the small world-frame
        # rotation dR R^T of every camera
        G, N = self.G, self.N
        c0, R0 = self._cam0[:, :3, 3], self._cam0[:, :3, :3]
        with torch.no_grad():
            dth = torch.cat([st.reshape(G, -1) for st in steps], 1)                 # (G, n_unknowns)
            nr = len(self.rep_views)
            dXr = (dth.double() @ self._J.double().T).reshape(G, nr, 3, 4)
            Ar = dXr[..., :3] @ R0[self.rep_views].double().transpose(-1, -2).unsqueeze(0)
            # every slice moves with its photograph: the same centre shift, the same world rotation
            dX, A = dXr[:, self.view_rep], Ar[:, self.view_rep]
            om = 0.5 * torch.stack([A[..., 2, 1] - A[..., 1, 2], A[..., 0, 2] - A[..., 2, 0], A[..., 1, 0] - A[..., 0, 1]], -1)
            disp = torch.cat([dX[..., 3], om], -1)                                  # (G, N, 6)
            cam1, _, _, _ = self._iface()
            actual = torch.cat([(cam1[:, :3, 3] - c0), _rotvec(cam1[:, :3, :3] @ R0.transpose(-1, -2))], -1).double()
            lin = float((disp.sum(0) - actual).norm() / actual.norm().clamp(min=1e-30))
        self._tick('moves')
        self._accumulate(disp, actual, chk_step, lin)
        self.t_spent += time.time() - t0

    # ------------------------------------------------------------------ buckets
    def _accumulate(self, disp, actual, chk_step, lin):
        if self._acc is None:
            sizes = torch.cat([p.detach() for p in self.c['log_sizes']]).exp()
            self._acc = dict(it=self._it, snap=self._cam0.cpu().numpy(), disp=disp.clone(), actual=actual.clone(),
                             force=self._force.cpu().numpy(), gloss=self._gl.cpu().numpy(),
                             chk_grad=[self._chk_grad], chk_step=[chk_step], lin=[lin], depth_err=[self._depth_err],
                             scale=sizes.cpu().numpy())
        else:
            a = self._acc
            a['disp'] += disp; a['actual'] += actual
            a['chk_grad'].append(self._chk_grad); a['chk_step'].append(chk_step); a['lin'].append(lin)
            a['depth_err'].append(self._depth_err)
        if (self._it + 1) % self.bucket == 0 or self._it + 1 >= self._niter:
            self._flush()

    def _flush(self):
        a, self._acc = self._acc, None
        if a is None:
            return
        r = self.rec
        r['it'].append(a['it']); r['stage'].append(self.stage); r['snap'].append(a['snap'])
        r['disp'].append(a['disp'].float().cpu().numpy()); r['actual'].append(a['actual'].float().cpu().numpy())
        r['force'].append(a['force']); r['gloss'].append(a['gloss']); r['scale'].append(a['scale'])
        r['chk_grad'].append(max(a['chk_grad'])); r['chk_step'].append(max(a['chk_step']))
        r['lin_resid'].append(max(a['lin']))
        r.setdefault('depth_err', []).append(max(a['depth_err']))
        if len(r['it']) % 5 == 0 and self._meta_for_partial is not None:
            # a long recording must survive its own process: the latest snapshot stands in for the final poses
            self.save(torch.as_tensor(r['snap'][-1]), *self._meta_for_partial, partial=True)
        print(f' >> inspect: {self.stage} it {a["it"]}: grad split {max(a["chk_grad"]):.1e}, '
              f'step split {max(a["chk_step"]):.1e}, linear {max(a["lin"]):.1e}, depth {max(a["depth_err"]):.1e}, '
              f'{self.t_spent:.0f} s spent inspecting ('
              + ', '.join(f'{k} {v:.0f}' for k, v in sorted(self._tm.items(), key=lambda kv: -kv[1])) + ')')

    def save(self, final_cam2w, imgs, mst, partial=False):
        if not partial:
            self._flush()
        r = self.rec
        if not r['it']:
            return
        (np.savez if partial else np.savez_compressed)(
            self.out_path + ('.partial.npz' if partial else ''),
            it=np.asarray(r['it']), stage=np.asarray(r['stage']), snap=np.stack(r['snap']),
            disp=np.stack(r['disp']), actual=np.stack(r['actual']), force=np.stack(r['force']).astype(np.float32),
            gloss=np.stack(r['gloss']), scale=np.stack(r['scale']), chk_grad=np.asarray(r['chk_grad']),
            chk_step=np.asarray(r['chk_step']), lin_resid=np.asarray(r['lin_resid']),
            depth_err=np.asarray(r.get('depth_err', [])),
            final=final_cam2w.detach().cpu().numpy(), imgs=np.asarray([str(x) for x in imgs]),
            group_names=np.asarray(self.group_names), pair_keys=np.asarray(self.pair_keys).reshape(-1, 2),
            pair_kind=np.asarray(self.pair_kind), prior_names=np.asarray(self.prior_names), photo_of=self.photo_of,
            mst_root=np.asarray(mst[0]), mst_edges=np.asarray(mst[1]).reshape(-1, 2), bucket=np.asarray(self.bucket))
        if partial:
            return
        if os.path.exists(self.out_path + '.partial.npz'):
            os.remove(self.out_path + '.partial.npz')
        print(f' >> inspect: wrote {self.out_path} ({self.G} groups x {self.N} cameras, {len(r["it"])} buckets, '
              f'{self.t_spent:.0f} s spent inspecting)')
