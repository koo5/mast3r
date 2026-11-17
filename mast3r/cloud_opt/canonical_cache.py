"""Content identity for canonical values and sampled cross-view predictions.

Correspondence decisions stay live. The owner is a single content-keyed view;
partner membership, multiplicity, forward CONTENT and reduction order are inputs.

canon-v3 (hillview): a forward entry is identified by its `<f>.content`
(sha256 over the DECODED arrays, written beside every entry) and its path WITHOUT the
data extension, not by `<f>.sha256` (the stored bytes) and the full path. The cache is about to be
stored in another container (.hvz) -- the same arrays in other bytes under another name -- and a key on
bytes would orphan every shared canonical value for nothing. Same content, same key, any container.
"""
import hashlib
import json
import os

DATA_EXTS = ('.pth', '.hvz')    # a cache entry's data file, in either container


def entry_stem(rel):
    """a cache entry's relative path without its data extension; None for anything else"""
    for ext in DATA_EXTS:
        if rel.endswith(ext):
            return rel[:-len(ext)]
    return None


def index_view_pairs(pairs):
    """Each view's ordered pair entries; self-pairs occur once and still supply two sides."""
    by_view = {}
    for item in pairs.items():
        (img1, img2), _ = item
        by_view.setdefault(img1, []).append(item)
        if img2 != img1:
            by_view.setdefault(img2, []).append(item)
    return by_view


def cache_path(root, img, pairs, image_keys, subsample, device, options, *, view_pairs=None):
    import torch
    from mast3r.cloud_opt import sparse_ga
    if sparse_ga.CACHE_ENTRY_REL is None:
        raise sparse_ga._codec_unset('CACHE_ENTRY_REL')
    owner = image_keys[img]
    contributions = []
    for (img1, img2), ((path1, path2), _) in (pairs.items() if view_pairs is None else view_pairs):
        for matches, partner, path in ((img == img1, img2, path1), (img == img2, img1, path2)):
            if not matches:
                continue
            # relative to the root that HOLDS the entry: a pair read from a read-only
            # root -- a prepared local copy, an archive -- keys exactly as the same pair at home
            stem = entry_stem(sparse_ga.CACHE_ENTRY_REL(path, root))
            if stem is None:
                return None
            try:
                stat = os.stat(path)  # a lost spill target cannot identify shared values
                with open(path + '.content') as f:
                    identity = f.read().strip()
                if os.stat(path + '.content').st_mtime_ns < stat.st_mtime_ns:
                    return None
            except (OSError, UnicodeError):
                return None
            digest = identity[len('sha256:'):] if identity.startswith('sha256:') else ''
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                return None
            contributions.append([image_keys[partner], stem, digest])
    # Reduction order matters for floating-point sums. Keep it as well as the
    # sorted partner multiset, without changing the solver's arithmetic/order.
    # canon-v2: the stored predictions are sampled at the anchor pixels (S//2 + k*S),
    # no longer at the corners (k*S): other values in an array of the same size (a multiple-of-S
    # image), so nothing but the key keeps a v1 entry from being read as v2
    # canon-v3: forward CONTENT and the extension-less path (module docstring)
    identity = ['canon-v3', owner, sorted(contributions), contributions, subsample,
                str(device), torch.__version__, options,
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                list(torch.cuda.get_device_capability(device)) if str(device).startswith('cuda') else None]
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
    # the name a value is WRITTEN under: the cache's container (sparse_ga.CACHE_EXT, .hvz);
    # a reader resolves it to the .hvz or the .pth of the same key (sparse_ga._resolve), as any entry
    if sparse_ga.CACHE_EXT is None:
        raise sparse_ga._codec_unset('CACHE_EXT')
    return os.path.join(root, 'canon_views', f'{owner}@{signature}{sparse_ga.CACHE_EXT}')


def validate(payload, contributions):
    import torch
    if not isinstance(payload, dict) or set(payload) != {'value', 'predictions'}:
        raise ValueError('invalid shared canonical payload')
    (canon, depth, conf), focal = payload['value']
    if (not all(isinstance(v, torch.Tensor) for v in (canon, depth, conf, focal))
            or canon.ndim != 3 or canon.shape[-1] != 3 or depth.shape != canon.shape[:2]
            or conf.shape != depth.shape or focal.numel() != 1):
        raise ValueError('invalid shared canonical value shapes')
    if len(payload['predictions']) != contributions:
        raise ValueError('invalid shared canonical prediction count')
    for points, confidence in payload['predictions']:
        if (not isinstance(points, torch.Tensor) or not isinstance(confidence, torch.Tensor)
                or points.ndim != 2 or points.shape[-1] != 3
                or confidence.shape != points.shape[:1]):
            raise ValueError('invalid shared canonical prediction shapes')
