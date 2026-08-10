"""
Designer wireframe -> centerline strokes -> piecewise cubic Bezier equations.

Input : Wireframe.png (pen-and-ink line drawing; ~4 px strokes, nothing filled)

Why this pipeline differs from the silhouette one
-------------------------------------------------
The drawing is made of STROKES, not filled regions. Tracing region boundaries
(cv2.RETR_TREE) would follow both edges of every pen line and return 211
contours -- each line drawn twice. The correct primitive is the CENTERLINE:
skeletonize the ink, turn the skeleton into a graph, and read off pen strokes.

Those strokes are overwhelmingly OPEN curves. A periodic Fourier series is the
wrong model for an open curve (it must wrap end->start and rings at the seam),
so the primary representation here is a piecewise cubic Bezier path fitted with
Schneider's algorithm. Closed strokes additionally get an elliptic Fourier fit,
and an equal-budget Fourier fit is computed for every stroke purely to measure
what the wrong model would cost.

Outputs
    wireframe_skeleton.png       1-px centerline mask
    wireframe_strokes.png        colour-coded stroke decomposition (diagnostic)
    wireframe.svg                cubic Bezier paths, fill:none
    wireframe_strokes.csv        stroke_id, index, x, y, t
    wireframe_curves.json        normalization + control points + coefficients
    wireframe_equations.txt      human-readable equations
    wireframe_preview.png        extracted vs fitted, stroked
    wireframe_fit_comparison.png Bezier vs equal-budget Fourier

Run: uv run python wireframe_pipeline.py
Every number reported is measured; nothing is assumed.
"""

import csv
import json
import sys

import cv2
import matplotlib
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "Wireframe.png"

SPUR_PX = 4          # dangling skeleton stub shorter than this is an artifact
MERGE_COS = -0.60    # join two branches through a node if they meet at >= ~127 deg
MIN_STROKE_PX = 4    # strokes shorter than this are dropped (reported, not silent)
SMOOTH_SIGMA = 1.5   # mild along-stroke Gaussian, removes skeleton staircasing
RESAMPLE_STEP = 2.0  # px between resampled points before fitting
BEZIER_TOL_PX = 0.75 # max allowed deviation of the fitted Bezier, in pixels

NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
REPORT = {}
sys.setrecursionlimit(10000)


# ====================================================================== #
# 1. ink mask + centerline
# ====================================================================== #
def load_ink(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr, b = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = b.astype(np.uint8)

    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    sk = skeletonize(ink.astype(bool))
    widths = 2.0 * dt[sk]
    REPORT.update(
        otsu_threshold=float(thr),
        image_size_wh=[int(gray.shape[1]), int(gray.shape[0])],
        ink_pixels=int(ink.sum()),
        ink_fraction=float(ink.mean()),
        skeleton_pixels=int(sk.sum()),
        stroke_width_px={
            "median": float(np.median(widths)),
            "p10": float(np.percentile(widths, 10)),
            "p90": float(np.percentile(widths, 90)),
        },
    )
    return ink, sk


# ====================================================================== #
# 2. skeleton -> graph -> pen strokes
# ====================================================================== #
def _nbrs(S, p):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in NB8 if (r + dr, c + dc) in S]


def _order_path(pix):
    adj = {p: _nbrs(pix, p) for p in pix}
    ends = [p for p in pix if len(adj[p]) <= 1]
    if not ends:                                    # pure cycle
        start = next(iter(pix))
        path, prev, cur = [start], None, start
        while True:
            nxt = [q for q in adj[cur] if q != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
            path.append(cur)
        return path, True
    start = ends[0]
    path, prev, cur = [start], None, start
    while True:
        nxt = [q for q in adj[cur] if q != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        path.append(cur)
    return path, False


def skeleton_edges(sk):
    """Split the skeleton at endpoints/junctions into branch polylines."""
    H, W = sk.shape
    S = set(map(tuple, np.argwhere(sk)))
    node_px = {p for p in S if len(_nbrs(S, p)) != 2}

    nm = np.zeros((H, W), np.uint8)
    for r, c in node_px:
        nm[r, c] = 1
    n_nodes, nlab = cv2.connectedComponents(nm, 8)

    bm = np.zeros((H, W), np.uint8)
    for r, c in S - node_px:
        bm[r, c] = 1
    n_br, blab = cv2.connectedComponents(bm, 8)

    def touching(p):
        r, c = p
        for dr, dc in NB8:
            q = (r + dr, c + dc)
            if q in node_px:
                return int(nlab[q])
        return None

    edges = []
    for b in range(1, n_br):
        pix = set(map(tuple, np.argwhere(blab == b)))
        path, closed = _order_path(pix)
        edges.append(
            {"u": None if closed else touching(path[0]),
             "v": None if closed else touching(path[-1]),
             "path": path, "closed": closed}
        )
    REPORT.update(skeleton_nodes=int(n_nodes - 1), skeleton_branches=int(n_br - 1))
    return edges


def prune_spurs(edges):
    removed = 0
    while True:
        incid = {}
        for i, e in enumerate(edges):
            if e is None or e["closed"]:
                continue
            for nd in (e["u"], e["v"]):
                if nd is not None:
                    incid.setdefault(nd, []).append(i)
        drop = []
        for i, e in enumerate(edges):
            if e is None or e["closed"] or len(e["path"]) > SPUR_PX:
                continue
            du = len(incid.get(e["u"], [])) if e["u"] is not None else 0
            dv = len(incid.get(e["v"], [])) if e["v"] is not None else 0
            if (du <= 1 and dv >= 3) or (dv <= 1 and du >= 3):
                drop.append(i)
        if not drop:
            break
        for i in drop:
            edges[i] = None
        removed += len(drop)
    REPORT["spurs_pruned"] = int(removed)
    return [e for e in edges if e is not None]


def _direction(path, from_start, k=8):
    p = np.array(path, float)
    if not from_start:
        p = p[::-1]
    k = min(k, len(p) - 1)
    if k <= 0:
        return np.zeros(2)
    d = p[k] - p[0]
    n = np.linalg.norm(d)
    return d / n if n > 0 else np.zeros(2)


def merge_strokes(edges):
    """Pair branches through a junction when they continue each other."""
    incid = {}
    for i, e in enumerate(edges):
        if e["closed"]:
            continue
        for side in ("u", "v"):
            if e[side] is not None:
                incid.setdefault(e[side], []).append((i, side))

    pair, cand_all = {}, 0
    for nd, lst in incid.items():
        if len(lst) < 2:
            continue
        dirs = {k: _direction(edges[k[0]]["path"], k[1] == "u") for k in lst}
        cand = []
        for a in range(len(lst)):
            for b in range(a + 1, len(lst)):
                ka, kb = lst[a], lst[b]
                cand.append((float(np.dot(dirs[ka], dirs[kb])), ka, kb))
        cand.sort(key=lambda t: t[0])
        used = set()
        for d, ka, kb in cand:
            if d > MERGE_COS:
                break
            if ka in used or kb in used:
                continue
            used.add(ka)
            used.add(kb)
            pair[(nd, ka[0])] = kb[0]
            pair[(nd, kb[0])] = ka[0]
            cand_all += 1
    REPORT["tangent_merges"] = int(cand_all)

    strokes, used_e = [], set()
    for i, e in enumerate(edges):
        if e["closed"]:
            strokes.append((np.array(e["path"], float), True))
            used_e.add(i)

    def walk(i0, side0):
        seq, i, side = [(i0, side0)], i0, side0
        while True:
            nd = edges[i][side]
            if nd is None:
                break
            j = pair.get((nd, i))
            if j is None or j in {k for k, _ in seq}:
                break
            side_j = "u" if edges[j]["u"] == nd else "v"
            out_j = "v" if side_j == "u" else "u"
            seq.append((j, out_j))
            i, side = j, out_j
        return seq

    for i, e in enumerate(edges):
        if e["closed"] or i in used_e:
            continue
        back, fwd = walk(i, "u"), walk(i, "v")
        chain = [(j, s) for j, s in reversed(back[1:])] + [(i, "v")] + fwd[1:]
        pts, seen = [], []
        for j, s in chain:
            if j in used_e:
                continue
            p = np.array(edges[j]["path"], float)
            seen.append(j)
            q = p if s == "v" else p[::-1]
            if pts and np.linalg.norm(q[0] - pts[-1][-1]) > np.linalg.norm(q[-1] - pts[-1][-1]):
                q = q[::-1]
            pts.append(q)
        for j in seen:
            used_e.add(j)
        if pts:
            strokes.append((np.vstack(pts), False))
    return strokes


# ====================================================================== #
# 3. cleaning / resampling
# ====================================================================== #
def smooth_stroke(p, closed):
    mode = "wrap" if closed else "nearest"
    return np.stack(
        [ndimage.gaussian_filter1d(p[:, k], SMOOTH_SIGMA, mode=mode) for k in (0, 1)],
        axis=1,
    )


def resample(p, closed, step):
    loop = np.vstack([p, p[:1]]) if closed else p
    seg = np.hypot(*np.diff(loop, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    if L <= 1e-9:
        return p[:1], 0.0, np.zeros(1)
    n = max(4, int(round(L / step)) + 1)
    tgt = np.linspace(0.0, L, n, endpoint=not closed)
    out = np.stack([np.interp(tgt, s, loop[:, 0]), np.interp(tgt, s, loop[:, 1])], axis=1)
    return out, L, tgt / L


# ====================================================================== #
# 4. Schneider adaptive cubic Bezier fit
# ====================================================================== #
def _bezier(ctrl, t):
    t = np.asarray(t, float)[:, None]
    mt = 1 - t
    return (mt ** 3 * ctrl[0] + 3 * mt ** 2 * t * ctrl[1]
            + 3 * mt * t ** 2 * ctrl[2] + t ** 3 * ctrl[3])


def _chord_param(p):
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(p))


def _generate(p, u, t1, t2):
    """Least-squares interior control points with fixed end tangents."""
    mt = 1 - u
    b0 = mt ** 3
    b1 = 3 * mt ** 2 * u
    b2 = 3 * mt * u ** 2
    b3 = u ** 3
    A0 = b1[:, None] * t1
    A1 = b2[:, None] * t2
    # P0 contributes through BOTH b0 and b1 (since P1 = P0 + t1*a1), and P3
    # through both b2 and b3. Subtracting only b0/b3 corrupts the normal
    # equations and makes the fitter split instead of fit.
    tmp = p - ((b0 + b1)[:, None] * p[0] + (b2 + b3)[:, None] * p[-1])
    c00 = float((A0 * A0).sum())
    c01 = float((A0 * A1).sum())
    c11 = float((A1 * A1).sum())
    x0 = float((A0 * tmp).sum())
    x1 = float((A1 * tmp).sum())
    det = c00 * c11 - c01 * c01
    seg = float(np.linalg.norm(p[-1] - p[0]))
    if abs(det) < 1e-12:
        a1 = a2 = seg / 3.0
    else:
        a1 = (x0 * c11 - x1 * c01) / det
        a2 = (c00 * x1 - c01 * x0) / det
        if a1 < 1e-6 or a2 < 1e-6:
            a1 = a2 = seg / 3.0
    return np.array([p[0], p[0] + t1 * a1, p[-1] + t2 * a2, p[-1]])


def _max_err(p, ctrl, u):
    d = np.hypot(*(_bezier(ctrl, u) - p).T)
    i = int(np.argmax(d))
    return float(d[i]), i


def _reparam(p, ctrl, u):
    q1 = 3 * (ctrl[1:] - ctrl[:-1])
    q2 = 2 * (q1[1:] - q1[:-1])
    out = u.copy()
    for k, t in enumerate(u):
        mt = 1 - t
        d = _bezier(ctrl, [t])[0] - p[k]
        d1 = mt ** 2 * q1[0] + 2 * mt * t * q1[1] + t ** 2 * q1[2]
        d2 = mt * q2[0] + t * q2[1]
        den = float((d1 * d1).sum() + (d * d2).sum())
        if abs(den) > 1e-12:
            out[k] = t - float((d * d1).sum()) / den
    return np.clip(out, 0.0, 1.0)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(2)


def fit_cubic(p, t1, t2, tol, depth=0):
    if len(p) < 2:
        return []
    if len(p) == 2:
        d = np.linalg.norm(p[1] - p[0]) / 3.0
        return [np.array([p[0], p[0] + t1 * d, p[1] + t2 * d, p[1]])]
    u = _chord_param(p)
    ctrl = _generate(p, u, t1, t2)
    err, split = _max_err(p, ctrl, u)
    if err < tol:
        return [ctrl]
    if err < tol * 4 and depth < 24:
        for _ in range(12):
            u = _reparam(p, ctrl, u)
            ctrl = _generate(p, u, t1, t2)
            err, split = _max_err(p, ctrl, u)
            if err < tol:
                return [ctrl]
    split = int(np.clip(split, 1, len(p) - 2))
    ct = _unit(p[split - 1] - p[split + 1])
    return (fit_cubic(p[: split + 1], t1, ct, tol, depth + 1)
            + fit_cubic(p[split:], -ct, t2, tol, depth + 1))


def fit_path(p, closed, tol):
    q = np.vstack([p, p[:1]]) if closed else p
    t1 = _unit(q[1] - q[0])
    # For a closed stroke force t2 = -t1 so the path is G1 across the seam.
    t2 = -t1 if closed else _unit(q[-2] - q[-1])
    return fit_cubic(q, t1, t2, tol)


def eval_bezier_path(segs, n_per=64):
    out = []
    for k, c in enumerate(segs):
        t = np.linspace(0, 1, n_per, endpoint=(k == len(segs) - 1))
        out.append(_bezier(c, t))
    return np.vstack(out)


# ====================================================================== #
# 5. equal-budget Fourier comparison (to quantify the wrong model)
# ====================================================================== #
def efd_fit(pts, N):
    M = len(pts)
    t = np.arange(M) / M
    n = np.arange(1, N + 1)[:, None]
    co, si = np.cos(2 * np.pi * n * t), np.sin(2 * np.pi * n * t)
    return (pts[:, 0].mean(), pts[:, 1].mean(),
            2 / M * co @ pts[:, 0], 2 / M * si @ pts[:, 0],
            2 / M * co @ pts[:, 1], 2 / M * si @ pts[:, 1])


def efd_eval(A0, C0, a, b, c, d, t):
    t = np.atleast_1d(np.asarray(t, float))
    n = np.arange(1, len(a) + 1)[None, :]
    ang = 2 * np.pi * n * t[:, None]
    co, si = np.cos(ang), np.sin(ang)
    return np.stack([A0 + co @ a + si @ b, C0 + co @ c + si @ d], axis=1)


def dev(orig, other):
    D = np.hypot(orig[:, None, 0] - other[None, :, 0], orig[:, None, 1] - other[None, :, 1])
    near = D.min(axis=1)
    return float(np.sqrt((near ** 2).mean())), float(near.max())


# ====================================================================== #
# driver
# ====================================================================== #
def main():
    ink, sk = load_ink(SRC)
    edges = prune_spurs(skeleton_edges(sk))
    raw_strokes = merge_strokes(edges)

    strokes, dropped = [], []
    for p, closed in raw_strokes:
        if len(p) < MIN_STROKE_PX:
            dropped.append(int(len(p)))
            continue
        p = np.stack([p[:, 1], p[:, 0]], axis=1)      # (row,col) -> (x,y_down)
        strokes.append({"px": p, "closed": bool(closed)})
    REPORT["strokes_dropped_below_min_length"] = {
        "count": len(dropped), "pixel_lengths": sorted(dropped, reverse=True)[:40],
        "total_pixels": int(sum(dropped)),
    }
    REPORT["strokes_kept"] = len(strokes)

    # ---- clean, orient to y-up, normalize with ONE shared transform ---- #
    for s in strokes:
        s["px"] = smooth_stroke(s["px"], s["closed"])
    allpts = np.vstack([s["px"] for s in strokes])
    math_all = np.stack([allpts[:, 0], -allpts[:, 1]], axis=1)
    x0, x1 = math_all[:, 0].min(), math_all[:, 0].max()
    y0, y1 = math_all[:, 1].min(), math_all[:, 1].max()
    height_px = float(y1 - y0)
    scale = 1.0 / height_px
    centroid = math_all.mean(axis=0)

    total_len = 0.0
    for s in strokes:
        m = np.stack([s["px"][:, 0], -s["px"][:, 1]], axis=1)
        s["norm"] = (m - centroid) * scale
        rs, L, tv = resample(s["norm"], s["closed"], RESAMPLE_STEP * scale)
        s.update(rs=rs, length=float(L), t=tv, n_points=int(len(rs)))
        total_len += L

    # ---- fit ------------------------------------------------------------ #
    tol = BEZIER_TOL_PX * scale
    for s in strokes:
        segs = fit_path(s["rs"], s["closed"], tol)
        s["bezier"] = segs
        s["bez_curve"] = eval_bezier_path(segs)
        # Error is measured against a densely sampled curve: at 24-64 samples
        # per segment the polyline itself deviates from the true Bezier by more
        # than the fit does, which inflates the max by ~4x.
        s["bez_rms"], s["bez_max"] = dev(s["rs"], eval_bezier_path(segs, 256))
        s["n_segments"] = len(segs)
        s["n_params_bezier"] = 2 * (3 * len(segs) + 1)

        # equal-budget periodic Fourier, to measure what the wrong model costs
        N = max(1, int(round((s["n_params_bezier"] - 2) / 4)))
        N = min(N, max(1, len(s["rs"]) // 2 - 1))
        f = efd_fit(s["rs"], N)
        fc = efd_eval(*f, np.linspace(0, 1, max(len(s["rs"]), 64), endpoint=False))
        s["fourier_N"] = int(N)
        s["fourier_curve"] = fc
        s["fou_rms"], s["fou_max"] = dev(s["rs"], fc)
        if s["closed"]:
            s["fourier_coeffs"] = {"A0": float(f[0]), "C0": float(f[1]),
                                   "a": [float(v) for v in f[2]], "b": [float(v) for v in f[3]],
                                   "c": [float(v) for v in f[4]], "d": [float(v) for v in f[5]]}

    strokes.sort(key=lambda s: -s["length"])           # structural lines first

    bez_rms = float(np.sqrt(np.mean([s["bez_rms"] ** 2 for s in strokes])))
    bez_max = float(max(s["bez_max"] for s in strokes))
    fou_rms = float(np.sqrt(np.mean([s["fou_rms"] ** 2 for s in strokes])))
    fou_max = float(max(s["fou_max"] for s in strokes))
    n_open = sum(1 for s in strokes if not s["closed"])

    # ================================================================== #
    #                             outputs                                #
    # ================================================================== #
    W, H = REPORT["image_size_wh"]
    cv2.imwrite("wireframe_skeleton.png", (255 - sk.astype(np.uint8) * 255))

    canvas = np.full((H, W, 3), 255, np.uint8)
    rng = np.random.default_rng(7)
    for s in strokes:
        col = tuple(int(v) for v in rng.integers(25, 210, 3))
        pl = np.round(s["px"]).astype(np.int32)
        cv2.polylines(canvas, [pl], s["closed"], col, 2, cv2.LINE_AA)
    cv2.imwrite("wireframe_strokes.png", canvas)

    # SVG straight from the Bezier control points, in pixel space
    def to_px(p):
        q = p / scale + centroid
        return np.stack([q[:, 0], -q[:, 1]], axis=1)

    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}">',
             f'  <rect width="{W}" height="{H}" fill="#ffffff"/>',
             '  <g fill="none" stroke="#111111" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round">']
    for k, s in enumerate(strokes):
        d = []
        for i, c in enumerate(s["bezier"]):
            q = to_px(c)
            if i == 0:
                d.append(f"M {q[0,0]:.2f},{q[0,1]:.2f}")
            d.append(f"C {q[1,0]:.2f},{q[1,1]:.2f} {q[2,0]:.2f},{q[2,1]:.2f} "
                     f"{q[3,0]:.2f},{q[3,1]:.2f}")
        if s["closed"]:
            d.append("Z")
        parts.append(f'    <path id="stroke_{k}" d="{" ".join(d)}"/>')
    parts += ["  </g>", "</svg>", ""]
    with open("wireframe.svg", "w") as f:
        f.write("\n".join(parts))

    with open("wireframe_strokes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stroke_id", "index", "x", "y", "t"])
        for k, s in enumerate(strokes):
            for i, (p, t) in enumerate(zip(s["rs"], s["t"])):
                w.writerow([k, i, f"{p[0]:.10f}", f"{p[1]:.10f}", f"{t:.10f}"])

    blocks = []
    for k, s in enumerate(strokes):
        blk = {
            "stroke_id": k,
            "closed": s["closed"],
            "arc_length_normalized": s["length"],
            "sample_points": s["n_points"],
            "bezier_segments": s["n_segments"],
            "n_parameters": s["n_params_bezier"],
            "control_points": [[[float(v) for v in pt] for pt in seg] for seg in s["bezier"]],
            "error_bezier": {"nearest_rms": s["bez_rms"], "nearest_max": s["bez_max"]},
            "equal_budget_fourier": {
                "harmonics": s["fourier_N"],
                "nearest_rms": s["fou_rms"],
                "nearest_max": s["fou_max"],
            },
        }
        if s["closed"]:
            blk["fourier_coefficients"] = s["fourier_coeffs"]
        blocks.append(blk)

    payload = {
        "source_image": SRC,
        "input_type": "designer pen-and-ink wireframe (stroke drawing, nothing filled)",
        "contour_type": "centerline pen strokes from a skeleton graph (open polylines + 1 closed loop)",
        "representation": "piecewise cubic Bezier (primary); elliptic Fourier for closed strokes",
        "why": ("strokes are open curves; a periodic Fourier series must wrap end->start "
                "and rings at the seam. Cubic Beziers are native to both SVG and Manim VMobject."),
        "rendering": "stroked thin lines, fill:none",
        "coordinate_convention": "x right, y upward",
        "preprocessing": REPORT,
        "normalization": {
            "centroid_of_all_stroke_points_px": [float(centroid[0]), float(centroid[1])],
            "scale_factor": float(scale),
            "drawing_height_before_normalization_px": height_px,
            "drawing_width_before_normalization_px": float(x1 - x0),
            "height_after_normalization": 1.0,
            "rotation": None,
            "aspect_ratio_preserved": True,
            "shared_by_all_strokes": True,
        },
        "num_strokes": len(strokes),
        "num_open_strokes": n_open,
        "num_closed_strokes": len(strokes) - n_open,
        "total_pen_length_normalized": float(total_len),
        "total_bezier_segments": int(sum(s["n_segments"] for s in strokes)),
        "total_parameters": int(sum(s["n_params_bezier"] for s in strokes)),
        "bezier_tolerance_px": BEZIER_TOL_PX,
        "fit_quality": {
            "bezier_nearest_rms": bez_rms,
            "bezier_nearest_max": bez_max,
            "equal_budget_fourier_nearest_rms": fou_rms,
            "equal_budget_fourier_nearest_max": fou_max,
            "units": "fraction of normalized drawing height (height = 1)",
        },
        "strokes": blocks,
        "generated_files": [
            "wireframe_skeleton.png", "wireframe_strokes.png", "wireframe.svg",
            "wireframe_strokes.csv", "wireframe_curves.json",
            "wireframe_equations.txt", "wireframe_preview.png",
            "wireframe_fit_comparison.png", "wireframe_manim.py",
        ],
    }
    with open("wireframe_curves.json", "w") as f:
        json.dump(payload, f, indent=2)

    with open("wireframe_equations.txt", "w") as f:
        f.write("DESIGNER WIREFRAME - PIECEWISE CUBIC BEZIER STROKE EQUATIONS\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"source                 : {SRC}\n")
        f.write("primitive              : centerline pen strokes (skeleton graph)\n")
        f.write("representation         : piecewise cubic Bezier, G1 at joins\n")
        f.write("coordinate convention  : x right, y upward\n")
        f.write(f"shared centroid        : ({centroid[0]:.6f}, {centroid[1]:.6f}) px\n")
        f.write(f"shared scale factor    : {scale:.10f}   (drawing height = 1)\n")
        f.write("rotation applied       : none\n")
        f.write(f"strokes                : {len(strokes)}  ({n_open} open, "
                f"{len(strokes)-n_open} closed)\n")
        f.write(f"bezier segments        : {sum(s['n_segments'] for s in strokes)}\n")
        f.write(f"total parameters       : {sum(s['n_params_bezier'] for s in strokes)}\n")
        f.write(f"total pen length       : {total_len:.6f}\n")
        f.write(f"fit nearest RMS        : {bez_rms:.6e}   max {bez_max:.6e}\n\n")
        f.write("Each segment j of stroke k is\n")
        f.write("  B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3,  0 <= t <= 1\n")
        f.write("with P3 of segment j equal to P0 of segment j+1.\n\n")
        for k, s in enumerate(strokes):
            f.write("-" * 78 + "\n")
            f.write(f"STROKE {k}  {'closed' if s['closed'] else 'open'}  "
                    f"len={s['length']:.5f}  segments={s['n_segments']}  "
                    f"params={s['n_params_bezier']}\n")
            f.write(f"  fit nearest RMS={s['bez_rms']:.4e}  max={s['bez_max']:.4e}   |   "
                    f"equal-budget Fourier N={s['fourier_N']}: "
                    f"RMS={s['fou_rms']:.4e}  max={s['fou_max']:.4e}\n")
            for j, c in enumerate(s["bezier"]):
                f.write(f"    seg{j:>3}: P0=({c[0,0]:+.8f},{c[0,1]:+.8f}) "
                        f"P1=({c[1,0]:+.8f},{c[1,1]:+.8f}) "
                        f"P2=({c[2,0]:+.8f},{c[2,1]:+.8f}) "
                        f"P3=({c[3,0]:+.8f},{c[3,1]:+.8f})\n")
            f.write("\n")

    # preview: extracted vs fitted, stroked only
    fig, axs = plt.subplots(1, 3, figsize=(17, 8.4))
    for ax, mode in zip(axs, ("orig", "fit", "over")):
        for s in strokes:
            o, b = s["rs"], s["bez_curve"]
            if s["closed"]:
                o, b = np.vstack([o, o[:1]]), np.vstack([b, b[:1]])
            if mode in ("orig", "over"):
                ax.plot(o[:, 0], o[:, 1], "-", lw=2.4 if mode == "over" else 1.2,
                        color="#b9c0cc" if mode == "over" else "#14181d",
                        solid_capstyle="round")
            if mode in ("fit", "over"):
                ax.plot(b[:, 0], b[:, 1], "-", lw=1.2,
                        color="#d1495b" if mode == "over" else "#14181d",
                        solid_capstyle="round")
        ax.set_aspect("equal")
        ax.axis("off")
    axs[0].set_title("extracted centerline strokes", fontsize=12, pad=12)
    axs[1].set_title("reconstructed from Bezier equations", fontsize=12, pad=12)
    axs[2].set_title("overlay (grey = extracted, red = fitted)", fontsize=12, pad=12)
    fig.suptitle(f"Designer wireframe — {len(strokes)} pen strokes, "
                 f"{sum(s['n_segments'] for s in strokes)} cubic Bezier segments "
                 f"(nearest RMS {bez_rms:.2e}, height = 1)", fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("wireframe_preview.png", dpi=150)
    plt.close(fig)

    # fit comparison: Bezier vs equal-budget Fourier
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.4))
    lens = np.array([s["length"] for s in strokes])
    # A few dead-straight strokes are fitted exactly (~1e-16); clamp so the
    # axis shows the range everything else lives in.
    FLOOR = 1e-6
    a1.scatter(lens, np.maximum([s["bez_rms"] for s in strokes], FLOOR), s=18, alpha=.8,
               label="piecewise cubic Bezier", color="#2a628f")
    a1.scatter(lens, np.maximum([s["fou_rms"] for s in strokes], FLOOR), s=18, alpha=.8,
               label="periodic Fourier, equal budget", color="#d1495b")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_ylim(FLOOR * 0.7, None)
    a1.set_xlabel("stroke arc length (height = 1)")
    a1.set_ylabel("nearest-point RMS")
    a1.set_title("Per-stroke accuracy at equal parameter count")
    a1.grid(alpha=.25, which="both", ls=":"); a1.legend(fontsize=9)

    # Worst absolute failure of the periodic model. Ranking by ratio instead
    # would pick a dead-straight stroke whose Bezier error is ~1e-16.
    worst = max(strokes, key=lambda s: s["fou_rms"])
    o = worst["rs"]
    a2.plot(o[:, 0], o[:, 1], "-", lw=4, color="#c9ced6", label="extracted stroke")
    a2.plot(worst["bez_curve"][:, 0], worst["bez_curve"][:, 1], "-", lw=1.8,
            color="#2a628f", label=f"Bezier ({worst['n_segments']} seg)")
    a2.plot(worst["fourier_curve"][:, 0], worst["fourier_curve"][:, 1], "-", lw=1.4,
            color="#d1495b", label=f"Fourier N={worst['fourier_N']}")
    a2.set_aspect("equal"); a2.axis("off"); a2.legend(fontsize=9, loc="best")
    a2.set_title(f"Worst absolute case for the periodic model\n"
                 f"(open stroke it must force closed): "
                 f"RMS {worst['fou_rms']:.1e} vs {worst['bez_rms']:.1e}", fontsize=11)
    fig.tight_layout()
    fig.savefig("wireframe_fit_comparison.png", dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------- console
    print(json.dumps(REPORT, indent=2))
    print(f"\nstrokes kept        : {len(strokes)}  ({n_open} open, {len(strokes)-n_open} closed)")
    print(f"shared centroid px  : ({centroid[0]:.3f}, {centroid[1]:.3f})")
    print(f"drawing height px   : {height_px:.3f}   scale {scale:.10f}")
    print(f"total pen length    : {total_len:.6f}   sample points "
          f"{sum(s['n_points'] for s in strokes)}")
    print(f"bezier segments     : {sum(s['n_segments'] for s in strokes)}   "
          f"parameters {sum(s['n_params_bezier'] for s in strokes)}")
    print(f"bezier   nearest RMS {bez_rms:.6e}  max {bez_max:.6e}")
    print(f"fourier  nearest RMS {fou_rms:.6e}  max {fou_max:.6e}  (equal budget)")
    print(f"fourier is worse by  {fou_rms/bez_rms:.1f}x in RMS\n")
    print(f"{'id':>3} {'len':>8} {'pts':>5} {'seg':>4} {'par':>5} {'bezRMS':>11} "
          f"{'bezMAX':>11} {'N':>4} {'fouRMS':>11}  closed")
    for k, s in enumerate(strokes[:18]):
        print(f"{k:>3} {s['length']:>8.4f} {s['n_points']:>5} {s['n_segments']:>4} "
              f"{s['n_params_bezier']:>5} {s['bez_rms']:>11.4e} {s['bez_max']:>11.4e} "
              f"{s['fourier_N']:>4} {s['fou_rms']:>11.4e}  {s['closed']}")
    print(f"... {len(strokes)-18} more strokes")


if __name__ == "__main__":
    main()
