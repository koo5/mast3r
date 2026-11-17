# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R Fast Nearest Neighbor
# --------------------------------------------------------
import time

import torch
import numpy as np
import math
from scipy.spatial import KDTree

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.device import to_numpy, todevice  # noqa


# hillview: WHERE THE MATCHING TIME GOES, because two readings of the same number want opposite
# fixes. Reciprocal matching is 67-75 % of a run's wall clock and the card sits at util_mean 44 %
# with util_idle_frac 0.06 -- busy but half-idle. That is either
#   (a) SYNC-BOUND: each hop calls to_numpy(tree.query(...)) twice, and the convergence mask
#       lives in numpy, so a pair pays ~80 GPU->CPU syncs and the late hops are tiny queries
#       paying full launch cost. The fix is to keep the hop bookkeeping on the device.
#   (b) BANDWIDTH-BOUND: every hop builds a |queries| x |db| similarity matrix -- at subsample 8
#       that is ~3072 seeds against ~196k descriptors -- so the kernels are large matmuls
#       re-reading the descriptor table. The fix is batching pairs, or fewer/cheaper hops.
# Guessing between them would cost a rental. These counters answer it instead, and cost one
# perf_counter pair per hop: the query timing already includes the sync that to_numpy forces,
# and the bookkeeping between the two queries is pure numpy, so the split is meaningful without
# adding a single synchronisation of its own (STRUCTURAL_TOOLS.md: make the program say).
STATS = {'calls': 0, 'hops': 0, 'converged_at': [], 't_query': 0.0, 't_book': 0.0,
         'seeds': 0, 'db_rows': 0, 'queries': 0, 'small_cpu': 0}


def stats_reset():
    STATS.update(calls=0, hops=0, converged_at=[], t_query=0.0, t_book=0.0,
                 seeds=0, db_rows=0, queries=0, small_cpu=0)


def stats_line():
    """One line for the run log, or '' when nothing was matched here."""
    if not STATS['calls']:
        return ''
    t = STATS['t_query'] + STATS['t_book']
    hops = STATS['hops'] or 1
    return (f"reciprocal NN: {STATS['calls']} call(s), {STATS['hops']} hop(s) "
            f"({STATS['hops'] / STATS['calls']:.1f} a call), "
            f"query {STATS['t_query']:.1f} s ({100 * STATS['t_query'] / t if t else 0:.0f} %), "
            f"bookkeeping {STATS['t_book']:.2f} s, "
            f"{STATS['queries'] / hops:.0f} seeds a hop against {STATS['db_rows'] / STATS['calls']:.0f} "
            f"db rows, {1000 * t / hops:.1f} ms a hop"
            + (f", {STATS['small_cpu']} small quer(ies) on the CPU (NN_SMALL_QUERY_CPU_ROWS {NN_SMALL_QUERY_CPU_ROWS})"
               if STATS['small_cpu'] else ""))


@torch.no_grad()
def bruteforce_reciprocal_nns(A, B, device='cuda', block_size=None, dist='l2'):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).to(device)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B).to(device)

    A = A.to(device)
    B = B.to(device)

    if dist == 'l2':
        dist_func = torch.cdist
        argmin = torch.min
    elif dist == 'dot':
        def dist_func(A, B):
            return A @ B.T

        def argmin(X, dim):
            sim, nn = torch.max(X, dim=dim)
            return sim.neg_(), nn
    else:
        raise ValueError(f'Unknown {dist=}')

    if block_size is None or len(A) * len(B) <= block_size**2:
        dists = dist_func(A, B)
        _, nn_A = argmin(dists, dim=1)
        _, nn_B = argmin(dists, dim=0)
    else:
        dis_A = torch.full((A.shape[0],), float('inf'), device=device, dtype=A.dtype)
        dis_B = torch.full((B.shape[0],), float('inf'), device=device, dtype=B.dtype)
        nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
        nn_B = torch.full((B.shape[0],), -1, device=device, dtype=torch.int64)
        number_of_iteration_A = math.ceil(A.shape[0] / block_size)
        number_of_iteration_B = math.ceil(B.shape[0] / block_size)

        for i in range(number_of_iteration_A):
            A_i = A[i * block_size:(i + 1) * block_size]
            for j in range(number_of_iteration_B):
                B_j = B[j * block_size:(j + 1) * block_size]
                dists_blk = dist_func(A_i, B_j)  # A, B, 1
                # dists_blk = dists[i * block_size:(i+1)*block_size, j * block_size:(j+1)*block_size]
                min_A_i, argmin_A_i = argmin(dists_blk, dim=1)
                min_B_j, argmin_B_j = argmin(dists_blk, dim=0)

                col_mask = min_A_i < dis_A[i * block_size:(i + 1) * block_size]
                line_mask = min_B_j < dis_B[j * block_size:(j + 1) * block_size]

                dis_A[i * block_size:(i + 1) * block_size][col_mask] = min_A_i[col_mask]
                dis_B[j * block_size:(j + 1) * block_size][line_mask] = min_B_j[line_mask]

                nn_A[i * block_size:(i + 1) * block_size][col_mask] = argmin_A_i[col_mask] + (j * block_size)
                nn_B[j * block_size:(j + 1) * block_size][line_mask] = argmin_B_j[line_mask] + (i * block_size)
    nn_A = nn_A.cpu().numpy()
    nn_B = nn_B.cpu().numpy()
    return nn_A, nn_B


# hillview (on CPU boxes): THE DB'S BLOCK ON A CPU, in rows (None: block_size, as before).
# A query writes its seeds x block similarities and the max reads them back; at 8192 rows that is ~36 MB a
# block, out of every cache. On one CPU box at 4 threads 512-2048-row blocks took 0.23-0.34x the time, the
# matches BIT-IDENTICAL in every case measured (masked and unmasked, 512..65536 rows: each similarity is
# the same 24-term dot whatever the block). At 16 threads on a loaded box small blocks were 4-8x SLOWER
# (each tiny matmul pays the thread pool), so the value is a fact of the BOX, measured there (hillview
# RECON_NN_CPU_DB_ROWS), never a guess. With it set, a CPU query is always blocked by it.
NN_CPU_DB_ROWS = None
# hillview: THE DB'S BLOCK ON A CARD, in rows (None: block_size, as before) -- NN_CPU_DB_ROWS's
# twin for every other device. The matmul is 92-97 % of a hop on an RX 580 (gfx803, ROCm 5.7), and rocBLAS's
# kernel for today's 8192-row block is close to its worst there: 2048 rows took a hop at 1300 seeds against 196 k db
# rows from 179 to 45 ms (3072 seeds: 417 -> 88), every answer of every block size equal to a float64 argmax on the
# one pair measured (nn_hop_bench). NOT identical by construction, unlike the CPU's: on a card another N can
# pick another GEMM kernel, so a value is set per BOX (hillview RECON_NN_GPU_DB_ROWS, recorded, never keyed: the
# default matcher's answer) only after that card's many-pairs check against float64.
NN_GPU_DB_ROWS = None


def bruteforce_nns_A(A, B, device='cuda', block_size=None, dist='l2'):
    """bruteforce_reciprocal_nns's nn_A alone -- bit-identical, at about half the work.

    hillview: cdistMatcher.query takes only nn_A, but bruteforce_reciprocal_nns also
    reduces every block along dim 0 for nn_B, and merges both through BOOLEAN-MASK writes: each
    `x[mask]` read is a nonzero(), a device->host sync, ~4 per 8192-row block, ~96 per query at
    512x384. Matching was measured at 50-81 % of the forward phase on rentals, 0.56-0.66 s a
    pair on a 380 W and a 250 W 3090 alike: latency, not compute. Here: the A side only, merged
    with torch.where (no sync until the one .cpu()). Same matmul, same reduction, the same strict
    `<` across blocks, so the same index on ties. colmap/database.py still takes both directions
    from bruteforce_reciprocal_nns."""
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    if isinstance(B, np.ndarray):
        B = torch.from_numpy(B)
    A = A.to(device)
    B = B.to(device)

    if dist == 'l2':
        dist_func = torch.cdist
        argmin = torch.min
    elif dist == 'dot':
        def dist_func(A, B):
            return A @ B.T

        def argmin(X, dim):
            sim, nn = torch.max(X, dim=dim)
            return sim.neg_(), nn
    else:
        raise ValueError(f'Unknown {dist=}')

    rows_b = block_size
    if NN_CPU_DB_ROWS is not None and torch.device(device).type == 'cpu':
        if int(NN_CPU_DB_ROWS) != NN_CPU_DB_ROWS or NN_CPU_DB_ROWS < 1:
            raise ValueError(f"fast_nn.NN_CPU_DB_ROWS {NN_CPU_DB_ROWS!r}: a positive number of rows")
        rows_b, block_size = int(NN_CPU_DB_ROWS), block_size or len(A)
    if NN_GPU_DB_ROWS is not None and torch.device(device).type != 'cpu':
        if int(NN_GPU_DB_ROWS) != NN_GPU_DB_ROWS or NN_GPU_DB_ROWS < 1:
            raise ValueError(f"fast_nn.NN_GPU_DB_ROWS {NN_GPU_DB_ROWS!r}: a positive number of rows")
        rows_b, block_size = int(NN_GPU_DB_ROWS), block_size or len(A)
    if rows_b == block_size and (block_size is None or len(A) * len(B) <= block_size**2):
        dis_A, nn_A = argmin(dist_func(A, B), dim=1)
    else:
        dis_A = torch.full((A.shape[0],), float('inf'), device=device, dtype=A.dtype)
        nn_A = torch.full((A.shape[0],), -1, device=device, dtype=torch.int64)
        for i in range(math.ceil(A.shape[0] / block_size)):
            A_i = A[i * block_size:(i + 1) * block_size]
            dis_i = dis_A[i * block_size:(i + 1) * block_size]     # views: written in place
            nn_i = nn_A[i * block_size:(i + 1) * block_size]
            for j in range(math.ceil(B.shape[0] / rows_b)):
                min_A_i, argmin_A_i = argmin(dist_func(A_i, B[j * rows_b:(j + 1) * rows_b]), dim=1)
                better = min_A_i < dis_i
                dis_i.copy_(torch.where(better, min_A_i, dis_i))
                nn_i.copy_(torch.where(better, argmin_A_i + (j * rows_b), nn_i))
    return _verified(A, B, dis_A, nn_A, dist).cpu().numpy()


NN_UNVERIFIED = -2       # an answer whose distance at its index is not the reduction's best (_verified)


def _verified(A, B, best, nn, dist):
    """nn, with every answer whose distance AT its index is not the `best` the reduction returned with it set to
    NN_UNVERIFIED -- which _nn_checked then refuses by name, on the host, with no sync added here. A CONSISTENCY
    check of each (best, index) pair against this call's rows, not an optimality one: a reduction that wrote a
    consistent but non-minimal pair would pass; stale or unwritten output does not.

    hillview: an RX 580 (ROCm) twice answered a hop with indices that were float bits: an output the
    reduction never wrote, stale allocator memory. Out of range, _nn_checked refuses
    them; but stale memory that last held another hop's INDICES is in range and plausible, and nothing else
    would notice. Recomputing each answer's distance at its index -- a gather of Q rows -- does, on every device,
    at nothing next to the matmul. The tolerance is the matmul's own precision: fp32 sums agree to ~1e-7 on unit
    descriptors, TF32's 10-bit products to ~1e-3 (a --tf32 run must not refuse every pair); a wrong
    index is off by far more. A NaN row compares False and is left as it was."""
    safe = nn.clamp(0, len(B) - 1)                      # an out-of-range one is _nn_checked's: never gathered with
    Bn = B[safe]
    if dist == 'dot':
        at = -(A * Bn).sum(dim=1)                       # argmin returned the NEGATED similarity
    else:
        at = torch.linalg.vector_norm(A - Bn, dim=1)
    off = (at - best).abs() > _verify_tolerance(A)
    return torch.where(off & (nn >= 0) & (nn < len(B)), torch.full_like(nn, NN_UNVERIFIED), nn)


def _verify_tolerance(A):
    """the matmul's own precision: TF32 on a card (--tf32), else fp32"""
    return 4e-3 if (A.is_cuda and torch.backends.cuda.matmul.allow_tf32) else 1e-4


class NNIndexOutOfRange(RuntimeError):
    """hillview: a nearest-neighbour query answered an index outside its database."""


def _nn_checked(idx, n, side, device):
    """the query's indices, refused when any is outside [0, n). An RX 580 (ROCm, a card a desktop shares)
    answered 4475493158418731731 for a db of 782 -- float bits read as int64 -- and the run
    died 57 pairs in on keep1[next1], naming nothing. Worse is what passes silently: xy2 is int32, so an int64
    out of range is truncated on assignment and may land in range; and the blocked path's -1 (a row whose
    similarities were all NaN) wraps to the db's last row. Checked on the raw answer, before either can happen.
    In-range garbage is beyond any cheap check. int64 whatever comes in: a hop with nothing left to ask (the other
    hop converged every seed) answers [], which np.asarray makes float64 -- and keep1[float array] raises where
    keep1[[]] did not."""
    idx = np.asarray(idx, dtype=np.int64)
    bad = (idx < 0) | (idx >= n)
    if bad.any():
        raise NNIndexOutOfRange(
            f'{int(bad.sum())} of {idx.size} nearest-neighbour indices into view {side}\'s {n} rows are out of range '
            f'on {device} (e.g. {idx[bad][:3].tolist()}): -1 is a row whose similarities were all NaN, '
            f'{NN_UNVERIFIED} an answer whose similarity at its index is not the best the reduction returned with it '
            f'(an output it did not write: _verified), anything else a fault in the query or on the device')
    return idx


# hillview: THE FUSED QUERY. bruteforce_nns_A writes every seed's similarity to
# every db row (1.1 k x 196 k fp32, ~870 MB a query) and torch.max reads it all back: on an RTX 4000 Ada
# the device was 92 % busy matching, 98 % of it that matmul and that max, ~10 ms a hop -- bandwidth, not
# arithmetic (24 products an element). The fused kernel never writes the matrix: a program holds a tile
# of seeds, streams a chunk of the db through in tiles, and keeps each seed's running (max, argmax).
#   * Each similarity is ONE fp32 FMA chain over the 24 channels in order, from 0 -- whatever the tile
#     sizes, so they are speed knobs only and never change an answer. TF32 never enters (cuBLAS takes
#     allow_tf32; this does not).
#   * Ties go to the FIRST db row: a strict `>` keeps each lane's earlier column, the smallest column
#     holding the best wins across lanes, and the earlier chunk across chunks -- torch.max's rule, and
#     bruteforce_nns_A's across its blocks.
#   * Bit-identical to bruteforce_nns_A only where cuBLAS sums in the same order. The first card check
#     (an RTX 3090) found it did on every query of 5 real pair cases -- but that is not identity
#     by construction (fixed 16 k blocks flipped near-ties on CUDA), so it is chosen explicitly
#     (NN_KERNEL = 'fused', set by the caller, recorded by it) and never replaces the default silently.
# CUDA only: Triton ships inside torch's CUDA wheel; a CPU caller asking for it is refused, not served
# the other kernel.
NN_KERNEL = 'torch'
# speed only: the answer is the same at any tiling. kernel: 'dot' (tl.dot, IEEE fp32) or 'fma' (one explicit
# FMA per channel) -- the same products in the same order, if Triton's FMA dot sums as it says; the card check
# (nn_fused_check.py) compares them
NN_FUSED_TILES = dict(kernel='dot', BLOCK_M=64, BLOCK_N=64, num_warps=4)
_FUSED = {}


def _fused_kernels():
    if _FUSED:
        return _FUSED
    import triton
    import triton.language as tl

    # the sizes are not specialised on (Triton would compile a variant per divisibility of each), so
    # fused_selftest compiles every variant a run uses: the two seed tiles and the reduction
    @triton.jit(do_not_specialize=['M', 'N', 'S', 'CHUNK'])
    def nn_dot_partial(QT, DBT, PV, PI, M, N, S, CHUNK, D: tl.constexpr, DP: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, DOT: tl.constexpr):
        """per (seed tile, db chunk): each seed's best similarity and its first db row in the chunk.
        QT is (D, M) and DBT (D, N), both contiguous. DOT: the tile's products by tl.dot in IEEE fp32
        (the channels padded to DP with zeros, which add nothing); else one explicit FMA per channel.

        NO REDUCTION PER TILE (the card check: an argmax across threads on every 128-column tile
        made the first version 7-8x slower than cuBLAS + max): each (seed, lane) keeps its own running
        best over the tiles, and the lanes are reduced once, at the chunk's end. A lane sees its
        columns in increasing order and keeps the first on a tie (strict >); across lanes the first is
        the smallest column holding the row's best."""
        pm = tl.program_id(0)
        ps = tl.program_id(1)
        rows = pm * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < M
        ks = tl.arange(0, DP)
        lo = ps * CHUNK
        hi = tl.minimum(lo + CHUNK, N)
        best = tl.full((BLOCK_M, BLOCK_N), float('-inf'), tl.float32)
        arg = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
        if DOT:
            q = tl.load(QT + ks[None, :] * M + rows[:, None], mask=rmask[:, None] & (ks[None, :] < D), other=0.)
        for n0 in range(lo, hi, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            cmask = cols < hi
            if DOT:
                d = tl.load(DBT + ks[:, None] * N + cols[None, :], mask=(ks[:, None] < D) & cmask[None, :], other=0.)
                acc = tl.dot(q, d, input_precision="ieee")
            else:
                acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
                for k in tl.static_range(D):
                    qk = tl.load(QT + k * M + rows, mask=rmask, other=0.)
                    dk = tl.load(DBT + k * N + cols, mask=cmask, other=0.)
                    acc = tl.fma(qk[:, None], dk[None, :], acc)
            acc = tl.where(cmask[None, :], acc, float('-inf'))
            # NaN: torch.max would return a NaN's row, `>` never takes one. The descriptors are unit
            # vectors, so a NaN similarity means the network itself failed, not a near-tie
            better = acc > best
            best = tl.where(better, acc, best)
            arg = tl.where(better, cols[None, :], arg)
        m = tl.max(best, axis=1)
        a = tl.min(tl.where(best == m[:, None], arg, 2147483647), axis=1)
        tl.store(PV + rows * S + ps, m, mask=rmask)
        tl.store(PI + rows * S + ps, a, mask=rmask)

    @triton.jit(do_not_specialize=['M', 'S'])
    def nn_dot_reduce(PV, PI, OUT, M, S, BLOCK_M: tl.constexpr):
        """the chunks of each seed in db order, the earlier kept on a tie"""
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < M
        best = tl.full((BLOCK_M,), float('-inf'), tl.float32)
        arg = tl.zeros((BLOCK_M,), tl.int32)
        for s in range(0, S):
            v = tl.load(PV + rows * S + s, mask=rmask, other=float('-inf'))
            i = tl.load(PI + rows * S + s, mask=rmask, other=0)
            better = v > best
            best = tl.where(better, v, best)
            arg = tl.where(better, i, arg)
        tl.store(OUT + rows, arg.to(tl.int64), mask=rmask)

    _FUSED.update(partial=nn_dot_partial, reduce=nn_dot_reduce, cdiv=triton.cdiv)
    return _FUSED


def fused_nns_A(A, B_T, device):
    """bruteforce_nns_A(A, B, dist='dot')'s answer in one pass over the db, the similarities never
    stored (NN_KERNEL above). B_T is the db TRANSPOSED, (D, N) contiguous: cdistMatcher makes it once."""
    dev = torch.device(device)
    if dev.type != 'cuda':
        raise ValueError(f"NN_KERNEL 'fused' is a CUDA kernel; device {device!r} has none -- set "
                         f"fast_nn.NN_KERNEL = 'torch' on this device")
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    return _fused_launch(A.to(dev), B_T).cpu().numpy()


def _fused_launch(A, B_T):
    """the two kernels on A's device (a test runs them on CPU tensors under TRITON_INTERPRET=1)"""
    dev = A.device
    if A.dtype != torch.float32 or B_T.dtype != torch.float32:
        raise TypeError(f"the fused NN query is fp32 only (got {A.dtype}, {B_T.dtype})")
    K = _fused_kernels()
    D, N = B_T.shape
    M = A.shape[0]
    if A.shape[1] != D:
        raise ValueError(f"descriptor widths differ: queries {A.shape[1]}, db {D}")
    if N == 0:
        raise ValueError("the fused NN query against an empty db: there is no row to return")
    QT = A.T.contiguous()
    if NN_FUSED_TILES['kernel'] not in ('dot', 'fma'):
        raise ValueError(f"NN_FUSED_TILES kernel {NN_FUSED_TILES['kernel']!r}: 'dot' or 'fma'")
    BLOCK_N = NN_FUSED_TILES['BLOCK_N']
    BLOCK_M = 16 if M <= 16 else NN_FUSED_TILES['BLOCK_M']
    row_blocks = K['cdiv'](M, BLOCK_M)
    # ~512 programs whatever the hop's seed count (a few per SM on the biggest cards): few seeds, short chunks
    chunks_wanted = max(1, 512 // row_blocks)
    CHUNK = max(BLOCK_N, K['cdiv'](K['cdiv'](N, chunks_wanted), BLOCK_N) * BLOCK_N)
    S = K['cdiv'](N, CHUNK)
    PV = torch.empty((M, S), device=dev, dtype=torch.float32)
    PI = torch.empty((M, S), device=dev, dtype=torch.int32)
    K['partial'][(row_blocks, S)](QT, B_T, PV, PI, M, N, S, CHUNK, D=D, DP=max(16, 1 << (D - 1).bit_length()),
                                  BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, DOT=NN_FUSED_TILES['kernel'] == 'dot',
                                  num_warps=NN_FUSED_TILES['num_warps'])
    out = torch.empty((M,), device=dev, dtype=torch.int64)
    K['reduce'][(K['cdiv'](M, 128),)](PV, PI, out, M, S, BLOCK_M=128)
    return out


def fused_selftest(device):
    """compile the fused kernels NOW, in the calling thread, and check them on queries whose answer is
    plain (each query is a db row). Triton builds its launcher with a C compiler and Python's headers
    on first use: a box without them stops here, before any pair is computed, not in a post thread."""
    g = torch.Generator().manual_seed(0)
    B = torch.nn.functional.normalize(torch.randn(5000, 24, generator=g), dim=1)
    B_T = B.T.contiguous().to(device)
    for M in (5, 300):                                  # both seed tiles
        want = torch.randperm(5000, generator=g)[:M]
        got = fused_nns_A(B[want], B_T, device)
        if not np.array_equal(got, want.numpy()):
            raise RuntimeError(f"fast_nn fused NN self-test failed on {device}: {int((got != want.numpy()).sum())} "
                               f"of {M} queries did not find themselves")


# hillview: A QUERY OF FEWER ROWS IS ANSWERED ON THE CPU (None: every query on its device).
# On an RX 580 (gfx803, ROCm 5.7 -- outside ROCm's support list) long-running match jobs answered 1-row hops
# with output the reduction never wrote: nn-verify (_verified) refused 4 of 4 jobs after its deploy, every refusal a
# "1 of 1" hop, while the same M = 1 matmul in a fresh process agrees with float64 to 6e-8. A fixed route,
# set per BOX (hillview RECON_NN_SMALL_QUERY_CPU_ROWS, recorded, never keyed: the default matcher's matches, any
# device's) -- never a retry: _verified stays on every answer on either device, so a fault in bigger hops still
# refuses the job. At 64 rows against ~196 k db rows a query is ~0.3 GFLOP; the db comes home once per matcher.
NN_SMALL_QUERY_CPU_ROWS = None


class cdistMatcher:
    def __init__(self, db_pts, device='cuda'):
        self.db_pts = db_pts.to(device)
        self.device = device
        self.small_rows = NN_SMALL_QUERY_CPU_ROWS       # hillview: read once, like the kernel below
        if self.small_rows is not None and (isinstance(self.small_rows, bool) or not isinstance(self.small_rows, int)
                                            or self.small_rows < 1):
            raise ValueError(f"fast_nn.NN_SMALL_QUERY_CPU_ROWS {self.small_rows!r}: a positive number of query rows")
        self._db_cpu = None
        self.kernel = NN_KERNEL                     # hillview: read once, so one matcher never mixes them
        if self.kernel not in ('torch', 'fused'):
            raise ValueError(f"fast_nn.NN_KERNEL {self.kernel!r}: 'torch' or 'fused'")
        if self.kernel == 'fused':
            self.db_T = self.db_pts.T.contiguous()

    def query(self, queries, k=1, **kw):
        assert k == 1
        if queries.numel() == 0:
            return None, []
        if (self.small_rows is not None and len(queries) < self.small_rows
                and torch.device(self.device).type != 'cpu'):
            if self._db_cpu is None:
                self._db_cpu = self.db_pts.cpu()
            STATS['small_cpu'] += 1
            q = queries.cpu() if torch.is_tensor(queries) else queries
            return None, bruteforce_nns_A(q, self._db_cpu, device='cpu', **kw)
        if self.kernel == 'fused':
            if kw.get('dist') != 'dot':
                raise ValueError(f"NN_KERNEL 'fused' computes dist='dot' only, not {kw.get('dist')!r}")
            return None, fused_nns_A(queries, self.db_T, self.device)
        nnA = bruteforce_nns_A(queries, self.db_pts, device=self.device, **kw)   # hillview: nnB was unused
        dis = None
        return dis, nnA


def merge_corres(idx1, idx2, shape1=None, shape2=None, ret_xy=True, ret_index=False):
    assert idx1.dtype == idx2.dtype == np.int32

    # unique and sort along idx1
    corres = np.unique(np.c_[idx2, idx1].view(np.int64), return_index=ret_index)
    if ret_index:
        corres, indices = corres
    xy2, xy1 = corres[:, None].view(np.int32).T

    if ret_xy:
        assert shape1 and shape2
        xy1 = np.unravel_index(xy1, shape1)
        xy2 = np.unravel_index(xy2, shape2)
        if ret_xy != 'y_x':
            xy1 = xy1[0].base[:, ::-1]
            xy2 = xy2[0].base[:, ::-1]

    if ret_index:
        return xy1, xy2, indices
    return xy1, xy2


def fast_reciprocal_NNs(pts1, pts2, subsample_or_initxy1=8, ret_xy=True, pixel_tol=0, ret_basin=False,
                        device='cuda', seed_mask=None, target_mask=None, seed_fill=False, **matcher_kw):
    H1, W1, DIM1 = pts1.shape
    H2, W2, DIM2 = pts2.shape
    assert DIM1 == DIM2
    # A supplied mask must describe these descriptors. Silently dropping a wrong-size mask
    # turns an exclusion request into unrestricted matching (and poisons the masked cache).
    if seed_mask is not None:
        seed_mask = np.asarray(seed_mask, dtype=bool)
        if seed_mask.shape != (H1, W1):
            raise ValueError(f'seed mask shape {seed_mask.shape} != {(H1, W1)}')
    if target_mask is not None:
        target_mask = np.asarray(target_mask, dtype=bool)
        if target_mask.shape != (H2, W2):
            raise ValueError(f'target mask shape {target_mask.shape} != {(H2, W2)}')

    def empty_matches():
        empty = np.empty(0, dtype=np.int32)
        result = merge_corres(empty, empty, (H1, W1), (H2, W2), ret_xy=ret_xy)
        return (*result, np.full(H1 * W1 + 1, -1, dtype=np.int32)) if ret_basin else result

    if ((seed_mask is not None and seed_mask.all()) or
            (target_mask is not None and target_mask.all())):
        return empty_matches()

    pts1 = pts1.reshape(-1, DIM1)
    pts2 = pts2.reshape(-1, DIM2)

    if isinstance(subsample_or_initxy1, int) and pixel_tol == 0:
        S = subsample_or_initxy1
        # hillview: the seeds may be restricted to the pixels this run still believes in
        # (sky, movers, the camera mount). This stays the STRIDE branch on purpose — seeds
        # are only starting points for the reciprocal hops, and handing the function an
        # explicit seed list silently drops max_iter from 10 to 1.
        if seed_mask is not None:
            keep_frac = float((~seed_mask).mean())
            if seed_fill and 0.02 < keep_frac < 1.0:
                # spend the grid's budget where it can land: a frame that is half sky gets
                # twice the seed density over the half that is not
                S = max(1, int(round(S * float(np.sqrt(keep_frac)))))
            y1, x1 = np.mgrid[S // 2:H1:S, S // 2:W1:S].reshape(2, -1)
            ok = ~seed_mask[np.clip(y1, 0, seed_mask.shape[0] - 1),
                            np.clip(x1, 0, seed_mask.shape[1] - 1)]
            y1, x1 = y1[ok], x1[ok]
            if len(x1) == 0:
                return empty_matches()  # never invent a seed on an excluded pixel
        else:
            y1, x1 = np.mgrid[S // 2:H1:S, S // 2:W1:S].reshape(2, -1)
        max_iter = 10
    else:
        x1, y1 = subsample_or_initxy1
        if isinstance(x1, torch.Tensor):
            x1 = x1.cpu().numpy()
        if isinstance(y1, torch.Tensor):
            y1 = y1.cpu().numpy()
        max_iter = 1

    xy1 = np.int32(np.unique(x1 + W1 * y1))  # make sure there's no doublons
    xy2 = np.full_like(xy1, -1)
    old_xy1 = xy1.copy()
    old_xy2 = xy2.copy()

    # hillview: the DATABASE side can be restricted too. Seeding only from good pixels of view 1
    # still lets a match LAND in view 2's sky or on its camera mount, and those pixels are then
    # free to win the nearest-neighbour vote; on the first test 13.7 % of matches still had to be
    # thrown away afterwards for exactly that. While the hops run, view-2 indices are into the
    # kept subset; they are mapped back to pixels the moment the loop ends.
    keep2 = None
    if target_mask is not None:
        k = np.flatnonzero(~np.asarray(target_mask).ravel())
        if len(k) < H2 * W2:            # even one allowed target must stay restricted
            # int32 throughout: merge_corres asserts both index arrays are int32, and
            # flatnonzero hands back int64, which would widen xy2 the moment it is mapped
            keep2 = k.astype(np.int32)
            pts2 = pts2[torch.as_tensor(keep2, device=pts2.device)
                        if isinstance(pts2, torch.Tensor) else keep2]

    # The reverse hop searches view 1 too. Restrict that database, not just the initial
    # seeds: otherwise an excluded source can beat the best allowed source on reciprocity.
    keep1 = None
    db1 = pts1
    if seed_mask is not None:
        k = np.flatnonzero(~seed_mask.ravel())
        if len(k) < H1 * W1:
            keep1 = k.astype(np.int32)
            db1 = pts1[torch.as_tensor(keep1, device=pts1.device)
                       if isinstance(pts1, torch.Tensor) else keep1]

    n1, n2 = len(db1), len(pts2)           # the rows each tree is built over (_nn_checked)
    if 'dist' in matcher_kw or 'block_size' in matcher_kw \
            or (isinstance(device, str) and device.startswith('cuda')) \
            or (isinstance(device, torch.device) and device.type.startswith('cuda')):
        pts1 = pts1.to(device)
        pts2 = pts2.to(device)
        tree1 = cdistMatcher(db1, device=device)
        tree2 = cdistMatcher(pts2, device=device)
    else:
        pts1, pts2 = to_numpy((pts1, pts2))
        tree1 = KDTree(to_numpy(db1))
        tree2 = KDTree(pts2)

    notyet = np.ones(len(xy1), dtype=bool)
    basin = np.full((H1 * W1 + 1,), -1, dtype=np.int32) if ret_basin else None

    niter = 0
    # n_notyet = [len(notyet)]
    STATS['calls'] += 1                                   # hillview: see STATS at the top
    STATS['seeds'] += len(xy1)
    STATS['db_rows'] += len(pts2)
    while notyet.any():
        # hillview: the two timed regions are query-plus-sync (to_numpy forces it) and the
        # numpy bookkeeping between them. Neither adds a synchronisation that was not already
        # there, so the split is honest.
        STATS['queries'] += int(notyet.sum())
        _t0 = time.perf_counter()
        _, nn2 = to_numpy(tree2.query(pts1[xy1[notyet]], **matcher_kw))
        xy2[notyet] = _nn_checked(nn2, n2, 2, device)
        STATS['t_query'] += time.perf_counter() - _t0
        _t0 = time.perf_counter()
        if not ret_basin:
            notyet &= (old_xy2 != xy2)  # remove points that have converged
        STATS['t_book'] += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        _, next1 = to_numpy(tree1.query(pts2[xy2[notyet]], **matcher_kw))
        next1 = _nn_checked(next1, n1, 1, device)
        xy1[notyet] = next1 if keep1 is None else keep1[next1]
        STATS['t_query'] += time.perf_counter() - _t0
        _t0 = time.perf_counter()
        if ret_basin:
            basin[old_xy1[notyet]] = xy1[notyet]
        notyet &= (old_xy1 != xy1)  # remove points that have converged
        STATS['t_book'] += time.perf_counter() - _t0

        # n_notyet.append(notyet.sum())
        niter += 1
        STATS['hops'] += 1
        if niter >= max_iter:
            break

        old_xy2[:] = xy2
        old_xy1[:] = xy1
    STATS['converged_at'].append(niter)

    # print('notyet_stats:', ' '.join(map(str, (n_notyet+[0]*10)[:max_iter])))

    if keep2 is not None:
        xy2 = keep2[xy2]        # subset positions -> view-2 pixel indices, as callers expect

    if pixel_tol > 0:
        # in case we only want to match some specific points
        # and still have some way of checking reciprocity
        old_yx1 = np.unravel_index(old_xy1, (H1, W1))[0].base
        new_yx1 = np.unravel_index(xy1, (H1, W1))[0].base
        dis = np.linalg.norm(old_yx1 - new_yx1, axis=-1)
        converged = dis < pixel_tol
        if not isinstance(subsample_or_initxy1, int):
            xy1 = old_xy1  # replace new points by old ones
    else:
        converged = ~notyet  # converged correspondences

    # Reciprocal hops can leave the original seed and land on an excluded source pixel.
    # Keep the post-filter in reconstruct.py too, but never return masked endpoints here.
    if seed_mask is not None:
        converged &= ~seed_mask.ravel()[xy1]
    if target_mask is not None:
        converged &= ~target_mask.ravel()[xy2]
    # keep only unique correspondences, and sort on xy1
    xy1, xy2 = merge_corres(xy1[converged], xy2[converged], (H1, W1), (H2, W2), ret_xy=ret_xy)
    if ret_basin:
        return xy1, xy2, basin
    return xy1, xy2


def extract_correspondences_nonsym(A, B, confA, confB, subsample=8, device=None, ptmap_key='pred_desc', pixel_tol=0):
    if '3d' in ptmap_key:
        opt = dict(device='cpu', workers=32)
    else:
        opt = dict(device=device, dist='dot', block_size=2**13)

    # matching the two pairs
    idx1 = []
    idx2 = []
    # merge corres from opposite pairs
    HA, WA = A.shape[:2]
    HB, WB = B.shape[:2]
    if pixel_tol == 0:
        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample, ret_xy=False, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample, ret_xy=False, **opt)
    else:
        S = subsample
        yA, xA = np.mgrid[S // 2:HA:S, S // 2:WA:S].reshape(2, -1)
        yB, xB = np.mgrid[S // 2:HB:S, S // 2:WB:S].reshape(2, -1)

        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=(xA, yA), ret_xy=False, pixel_tol=pixel_tol, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=(xB, yB), ret_xy=False, pixel_tol=pixel_tol, **opt)

    idx1 = np.r_[nn1to2[0], nn2to1[1]]
    idx2 = np.r_[nn1to2[1], nn2to1[0]]

    c1 = confA.ravel()[idx1]
    c2 = confB.ravel()[idx2]

    xy1, xy2, idx = merge_corres(idx1, idx2, (HA, WA), (HB, WB), ret_xy=True, ret_index=True)
    conf = np.minimum(c1[idx], c2[idx])
    corres = (xy1.copy(), xy2.copy(), conf)
    return todevice(corres, device)
