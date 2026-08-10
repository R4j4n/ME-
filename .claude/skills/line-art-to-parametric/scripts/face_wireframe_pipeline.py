"""
Wireframe face-outline pipeline.

Input : rajan.png (designer line art: filled ink masses + closed outline strokes)
Goal  : a STROKED wireframe -- every ink/paper boundary kept as its own closed
        path, each fitted with its own parametric equation. Nothing is filled.

Representation chain:
    binary ink mask -> contour paths (RETR_TREE, all nesting levels)
                    -> elliptic Fourier descriptors, per contour
                    -> periodic cubic B-spline, per contour (compared, not used)

Outputs
    face_wireframe_mask.png            thin-line contour mask (black on white)
    face_wireframe.svg                 stroked paths, fill:none
    face_wireframe_contours.csv        contour_id, index, x, y, t
    face_wireframe_coefficients.json   normalization + per-contour coefficients
    face_wireframe_equations.txt       human-readable equations
    face_wireframe_preview.png         original vs fitted, thin strokes
    face_wireframe_fit_comparison.png  harmonic sweep + Fourier-vs-spline

Run:  uv run python face_wireframe_pipeline.py
Every number reported is measured from the image; nothing is assumed.
"""

import csv
import json

import cv2
import matplotlib
import numpy as np
from scipy import ndimage
from scipy.interpolate import splev, splprep

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "rajan.png"

MIN_PERIMETER_PX = 40.0      # below this a contour is pixel noise, not line work
SMOOTH_SIGMA_PX = 1.8        # mild periodic Gaussian along each contour
BASE_POINTS = 500            # resample density is set from the outer contour
MIN_PTS, MAX_PTS = 96, 700
N_GRID = [5, 8, 12, 16, 20, 25, 30, 40, 50, 60, 80]
RMS_TOL = 0.0035             # fractions of the normalized head height
MAX_TOL = 0.0120

REPORT = {}


# ------------------------------------------------------------------ Part 1
def ink_mask(path):
    """Flatten alpha onto white, grayscale, Otsu -> 1 = ink, 0 = paper."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SystemExit(f"cannot read {path}")
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        img = (raw[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)).astype(
            np.uint8
        )
    elif raw.ndim == 2:
        img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    else:
        img = raw

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr, binar = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    REPORT["otsu_threshold"] = float(thr)
    REPORT["image_size_wh"] = [int(gray.shape[1]), int(gray.shape[0])]
    # No morphological opening: the face/neck outline is drawn with 1-2 px
    # strokes and even a 2x2 opening severs them. Noise is removed by dropping
    # short contours instead, which cannot break a stroke.
    return binar.astype(np.uint8)


# ------------------------------------------------------------------ Part 2
def all_boundaries(mask):
    """Every ink/paper boundary at every nesting level -> list of open polylines.

    RETR_TREE (not RETR_EXTERNAL) is the whole point: the external contour alone
    is the filled-silhouette answer. The nested contours are the line work --
    hairline, glasses, nose/lip, jaw, ear.
    """
    cnts, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    kept, dropped = [], []
    for i, c in enumerate(cnts):
        perim = cv2.arcLength(c, True)
        rec = {
            "cv_index": int(i),
            "parent": int(hier[0][i][3]),
            "depth": _depth(hier[0], i),
            "perimeter_px": float(perim),
            "area_px": float(abs(cv2.contourArea(c))),
            "raw_points": int(len(c)),
            "points": c[:, 0, :].astype(np.float64),
        }
        (kept if perim >= MIN_PERIMETER_PX else dropped).append(rec)

    REPORT["contours_found"] = len(cnts)
    REPORT["contours_kept"] = len(kept)
    REPORT["contours_dropped_as_noise"] = [
        {"cv_index": d["cv_index"], "perimeter_px": round(d["perimeter_px"], 2),
         "raw_points": d["raw_points"]}
        for d in dropped
    ]
    # Top-level boundaries first, then by size. Sorting on perimeter alone would
    # put the (longer, wigglier) mid-face region ahead of the outer head outline
    # and normalization would key off the wrong contour.
    kept.sort(key=lambda r: (r["depth"], -r["perimeter_px"]))
    return kept


def _depth(hier, i):
    d = 0
    while hier[i][3] != -1:
        i = hier[i][3]
        d += 1
    return d


def dedupe(pts, min_step=0.5):
    keep = [pts[0]]
    for p in pts[1:]:
        if np.hypot(*(p - keep[-1])) >= min_step:
            keep.append(p)
    out = np.asarray(keep)
    while len(out) > 3 and np.hypot(*(out[0] - out[-1])) < min_step:
        out = out[:-1]
    return out


def smooth_closed(pts, sigma):
    return np.stack(
        [ndimage.gaussian_filter1d(pts[:, k], sigma, mode="wrap") for k in (0, 1)],
        axis=1,
    )


def polygon_centroid_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2.0
    if abs(A) < 1e-12:
        return pts.mean(axis=0), 0.0
    return np.array([((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)]), A


def _cross(p1, p2, p3, p4):
    def o(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    return o(p3, p4, p1) != o(p3, p4, p2) and o(p1, p2, p3) != o(p1, p2, p4)


def self_intersections(pts, cap=900):
    """Crossing count between non-adjacent edges (subsampled for speed)."""
    q = pts[:: max(1, len(pts) // cap)]
    n = len(q)
    b = np.roll(q, -1, axis=0)
    hits = 0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _cross(q[i], b[i], q[j], b[j]):
                hits += 1
    return hits


# ------------------------------------------------------------------ Part 3/4
def resample_closed(pts, m):
    loop = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(loop, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    P = s[-1]
    tgt = np.linspace(0.0, P, m, endpoint=False)
    return (
        np.stack([np.interp(tgt, s, loop[:, 0]), np.interp(tgt, s, loop[:, 1])], axis=1),
        P,
        tgt / P,
    )


def efd_fit(pts, N):
    M = len(pts)
    t = np.arange(M) / M
    n = np.arange(1, N + 1)[:, None]
    co, si = np.cos(2 * np.pi * n * t), np.sin(2 * np.pi * n * t)
    return (
        pts[:, 0].mean(),
        pts[:, 1].mean(),
        2 / M * co @ pts[:, 0],
        2 / M * si @ pts[:, 0],
        2 / M * co @ pts[:, 1],
        2 / M * si @ pts[:, 1],
    )


def efd_eval(A0, C0, a, b, c, d, t):
    t = np.atleast_1d(np.asarray(t, float))
    n = np.arange(1, len(a) + 1)[None, :]
    ang = 2 * np.pi * n * t[:, None]
    co, si = np.cos(ang), np.sin(ang)
    return np.stack([A0 + co @ a + si @ b, C0 + co @ c + si @ d], axis=1)


def curve_errors(orig, recon):
    """Pointwise (same t) and geometric (nearest point) deviations."""
    dp = np.hypot(*(recon - orig).T)
    D = np.hypot(orig[:, None, 0] - recon[None, :, 0], orig[:, None, 1] - recon[None, :, 1])
    near = D.min(axis=1)
    return {
        "pointwise_rms": float(np.sqrt((dp ** 2).mean())),
        "pointwise_max": float(dp.max()),
        "nearest_rms": float(np.sqrt((near ** 2).mean())),
        "nearest_max": float(near.max()),
    }


def spline_fit(pts, smoothing):
    """Periodic cubic B-spline; returns (knots, coeffs, resampled curve, n_params)."""
    loop = np.vstack([pts, pts[:1]])
    u = np.linspace(0.0, 1.0, len(loop))
    tck, _ = splprep([loop[:, 0], loop[:, 1]], u=u, s=smoothing, per=True, k=3)
    ev = np.asarray(splev(np.linspace(0, 1, len(pts), endpoint=False), tck)).T
    n_ctrl = len(tck[1][0])
    return tck, ev, 2 * n_ctrl


# ------------------------------------------------------------------ driver
def main():
    mask = ink_mask(SRC)
    raw_contours = all_boundaries(mask)

    # ---- clean every contour ------------------------------------------ #
    cleaned = []
    for rec in raw_contours:
        pts = dedupe(rec["points"], 0.5)
        pts = smooth_closed(pts, SMOOTH_SIGMA_PX)
        pts = dedupe(pts, 0.5)
        rec["clean_px"] = pts
        rec["clean_points"] = int(len(pts))
        cleaned.append(rec)

    # ---- ONE global normalization, driven by the outer head contour ---- #
    tops = [r for r in cleaned if r["depth"] == 0]
    outer = max(tops, key=lambda r: r["area_px"])
    if outer is not cleaned[0]:
        raise SystemExit("outer boundary is not first after sorting")
    outer_math = np.stack([outer["clean_px"][:, 0], -outer["clean_px"][:, 1]], axis=1)
    centroid, area = polygon_centroid_area(outer_math)
    height_px = outer_math[:, 1].max() - outer_math[:, 1].min()
    scale = 1.0 / height_px

    for rec in cleaned:
        m = np.stack([rec["clean_px"][:, 0], -rec["clean_px"][:, 1]], axis=1)
        _, a_signed = polygon_centroid_area(m)
        if a_signed < 0:                      # counter-clockwise in x-right/y-up
            m = m[::-1]
            rec["clean_px"] = rec["clean_px"][::-1]
        rec["norm"] = (m - centroid) * scale  # same transform for every contour

    # ---- resample, density set by the outer contour -------------------- #
    _, outer_perim, _ = resample_closed(outer["norm"], BASE_POINTS)
    density = BASE_POINTS / outer_perim
    total_pts = 0
    for rec in cleaned:
        _, per, _ = resample_closed(rec["norm"], MIN_PTS)
        n = int(np.clip(round(per * density), MIN_PTS, MAX_PTS))
        rs, per, tv = resample_closed(rec["norm"], n)
        rec.update(rs=rs, perimeter=float(per), tvals=tv, n_resampled=int(n))
        rec["self_intersections_cleaned"] = self_intersections(rs)
        total_pts += n

    # ---- per-contour Fourier fit: smallest N meeting the tolerance ----- #
    for rec in cleaned:
        sweep = {}
        chosen = None
        for N in N_GRID:
            if 2 * N >= rec["n_resampled"]:   # Nyquist guard
                break
            fit = efd_fit(rec["rs"], N)
            err = curve_errors(rec["rs"], efd_eval(*fit, rec["tvals"]))
            sweep[N] = err
            if chosen is None and err["nearest_rms"] <= RMS_TOL and err["nearest_max"] <= MAX_TOL:
                chosen = N
        rec["tolerance_met"] = chosen is not None
        if chosen is None:
            chosen = max(sweep)               # best available, reported honestly
        fit = efd_fit(rec["rs"], chosen)
        rec.update(
            N=chosen,
            A0=float(fit[0]), C0=float(fit[1]),
            a=fit[2], b=fit[3], c=fit[4], d=fit[5],
            sweep=sweep,
            recon=efd_eval(*fit, rec["tvals"]),
        )
        rec["err"] = curve_errors(rec["rs"], rec["recon"])
        rec["self_intersections_recon"] = self_intersections(rec["recon"])
        rec["n_params_fourier"] = 4 * chosen + 2

    # ---- spline comparison at a matched parameter budget --------------- #
    for rec in cleaned:
        budget = rec["n_params_fourier"]
        best = None
        for s in np.logspace(-6, 0.5, 40):
            try:
                tck, ev, npar = spline_fit(rec["rs"], float(s))
            except Exception:
                continue
            if npar > budget * 1.15:
                continue
            e = curve_errors(rec["rs"], ev)
            if best is None or npar > best["n_params"]:
                best = {"s": float(s), "n_params": int(npar),
                        "n_control_points": int(npar // 2), **e}
        rec["spline"] = best

    # ================================================================== #
    #                              outputs                               #
    # ================================================================== #
    W, H = REPORT["image_size_wh"]

    # 1. thin-line contour mask (NOT a filled silhouette)
    canvas = np.full((H, W), 255, np.uint8)
    for rec in cleaned:
        poly = np.round(rec["clean_px"]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], True, 0, 2, cv2.LINE_AA)
    cv2.imwrite("face_wireframe_mask.png", canvas)

    # 2. stroked SVG -- fill:none, one path per contour
    parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        f'  <rect width="{W}" height="{H}" fill="#ffffff"/>',
        '  <g fill="none" stroke="#111111" stroke-width="2.2" '
        'stroke-linejoin="round" stroke-linecap="round">',
    ]
    for k, rec in enumerate(cleaned):
        d = "M " + " L ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in rec["clean_px"]) + " Z"
        parts.append(f'    <path id="contour_{k}" d="{d}"/>')
    parts += ["  </g>", "</svg>", ""]
    with open("face_wireframe.svg", "w") as f:
        f.write("\n".join(parts))

    # 3. ordered contour points
    with open("face_wireframe_contours.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["contour_id", "index", "x", "y", "t"])
        for k, rec in enumerate(cleaned):
            for i, (p, t) in enumerate(zip(rec["rs"], rec["tvals"])):
                w.writerow([k, i, f"{p[0]:.10f}", f"{p[1]:.10f}", f"{t:.10f}"])

    # 4. coefficients + normalization
    def contour_block(k, rec):
        return {
            "contour_id": k,
            "role_hint": ROLE_HINTS.get(rec["cv_index"], "unlabeled boundary"),
            "cv_index": rec["cv_index"],
            "nesting_depth": rec["depth"],
            "raw_points": rec["raw_points"],
            "cleaned_points": rec["clean_points"],
            "resampled_points": rec["n_resampled"],
            "perimeter_normalized": rec["perimeter"],
            "self_intersections_cleaned": rec["self_intersections_cleaned"],
            "self_intersections_reconstructed": rec["self_intersections_recon"],
            "harmonics": rec["N"],
            "harmonics_tolerance_met": rec["tolerance_met"],
            "n_parameters": rec["n_params_fourier"],
            "coefficients": {
                "A0": rec["A0"], "C0": rec["C0"],
                "a": [float(v) for v in rec["a"]],
                "b": [float(v) for v in rec["b"]],
                "c": [float(v) for v in rec["c"]],
                "d": [float(v) for v in rec["d"]],
            },
            "coefficient_vector": [
                float(v) for n in range(rec["N"])
                for v in (rec["a"][n], rec["b"][n], rec["c"][n], rec["d"][n])
            ],
            "reconstruction_error": rec["err"],
            "harmonic_sweep": {str(n): e for n, e in rec["sweep"].items()},
            "spline_at_matched_budget": rec["spline"],
        }

    blocks = [contour_block(k, r) for k, r in enumerate(cleaned)]
    agg_f = float(np.sqrt(np.mean([b["reconstruction_error"]["nearest_rms"] ** 2 for b in blocks])))
    agg_s = float(np.sqrt(np.mean([b["spline_at_matched_budget"]["nearest_rms"] ** 2
                                   for b in blocks if b["spline_at_matched_budget"]])))
    payload = {
        "input_type": "designer line art (stencil portrait), used directly; no person identification",
        "source_image": SRC,
        "contour_type": "multi-contour closed wireframe paths (RETR_TREE, all nesting levels)",
        "representation": "contour paths -> per-contour elliptic Fourier descriptors",
        "rendering": "stroked thin lines, fill:none -- not a filled silhouette",
        "coordinate_convention": "x right, y upward",
        "preprocessing": REPORT,
        "normalization": {
            "driven_by": "outer head contour (contour_id 0)",
            "applied_to": "all contours with one shared transform, so relative geometry is preserved",
            "centroid_pixel_math_coords": [float(centroid[0]), float(centroid[1])],
            "scale_factor": float(scale),
            "height_before_normalization_px": float(height_px),
            "height_after_normalization": 1.0,
            "rotation": None,
            "aspect_ratio_preserved": True,
        },
        "num_contours": len(blocks),
        "num_points_total_resampled": int(total_pts),
        "aggregate_error": {
            "fourier_nearest_rms": agg_f,
            "spline_nearest_rms_at_matched_budget": agg_s,
            "units": "fraction of normalized head height (height = 1)",
        },
        "contours": blocks,
        "generated_files": [
            "face_wireframe_mask.png",
            "face_wireframe.svg",
            "face_wireframe_contours.csv",
            "face_wireframe_coefficients.json",
            "face_wireframe_equations.txt",
            "face_wireframe_preview.png",
            "face_wireframe_fit_comparison.png",
            "face_wireframe_manim.py",
        ],
    }
    with open("face_wireframe_coefficients.json", "w") as f:
        json.dump(payload, f, indent=2)

    # 5. human-readable equations
    with open("face_wireframe_equations.txt", "w") as f:
        f.write("WIREFRAME FACE OUTLINE - PER-CONTOUR PARAMETRIC EQUATIONS\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"source                : {SRC}\n")
        f.write("representation        : closed contour paths + elliptic Fourier descriptors\n")
        f.write("rendering             : stroked wireframe (no fill)\n")
        f.write("coordinate convention : x right, y upward\n")
        f.write(f"shared centroid       : ({centroid[0]:.6f}, {centroid[1]:.6f}) px\n")
        f.write(f"shared scale factor   : {scale:.10f}   (head height = 1)\n")
        f.write("rotation applied      : none\n")
        f.write(f"contours              : {len(blocks)}\n")
        f.write(f"total resampled points: {total_pts}\n")
        f.write(f"aggregate nearest RMS : {agg_f:.6e}\n\n")
        f.write("x_k(t) = A0 + sum_{n=1..N_k} [ a_n cos(2 pi n t) + b_n sin(2 pi n t) ]\n")
        f.write("y_k(t) = C0 + sum_{n=1..N_k} [ c_n cos(2 pi n t) + d_n sin(2 pi n t) ]\n")
        f.write("0 <= t < 1, one independent closed curve per contour k\n\n")
        for k, (blk, rec) in enumerate(zip(blocks, cleaned)):
            f.write("-" * 78 + "\n")
            f.write(f"CONTOUR {k}  ({blk['role_hint']})\n")
            f.write(f"  depth={blk['nesting_depth']}  points {blk['raw_points']} raw -> "
                    f"{blk['cleaned_points']} cleaned -> {blk['resampled_points']} resampled\n")
            f.write(f"  perimeter={blk['perimeter_normalized']:.6f}  N={blk['harmonics']}  "
                    f"params={blk['n_parameters']}\n")
            e = blk["reconstruction_error"]
            f.write(f"  nearest RMS={e['nearest_rms']:.6e}  nearest max={e['nearest_max']:.6e}"
                    f"  (tolerance met: {blk['harmonics_tolerance_met']})\n")
            f.write(f"  A0={rec['A0']:.12f}   C0={rec['C0']:.12f}\n")
            f.write(f"  {'n':>3} {'a_n':>16} {'b_n':>16} {'c_n':>16} {'d_n':>16}\n")
            for n in range(rec["N"]):
                f.write(f"  {n+1:>3} {rec['a'][n]:>16.12f} {rec['b'][n]:>16.12f} "
                        f"{rec['c'][n]:>16.12f} {rec['d'][n]:>16.12f}\n")
            f.write("  F = [" + ", ".join(f"{v:.10f}" for v in blk["coefficient_vector"]) + "]\n\n")

    # 6. wireframe preview: original vs fitted, strokes only
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 8.0))
    for ax, mode in zip(axs, ("orig", "fit", "over")):
        for k, rec in enumerate(cleaned):
            lo = np.vstack([rec["rs"], rec["rs"][:1]])
            lr = np.vstack([rec["recon"], rec["recon"][:1]])
            if mode in ("orig", "over"):
                ax.plot(lo[:, 0], lo[:, 1], "-", lw=2.6 if mode == "over" else 1.5,
                        color="#b9c0cc" if mode == "over" else "#1b1f24",
                        label="extracted contour" if (mode == "over" and k == 0) else None)
            if mode in ("fit", "over"):
                ax.plot(lr[:, 0], lr[:, 1], "-", lw=1.5,
                        color="#d1495b" if mode == "over" else "#1b1f24",
                        label="Fourier fit" if (mode == "over" and k == 0) else None)
        ax.set_aspect("equal")
        ax.axis("off")
    axs[0].set_title("extracted wireframe contours", fontsize=12, pad=14)
    axs[1].set_title("reconstructed from the equations", fontsize=12, pad=14)
    axs[2].set_title("overlay", fontsize=12, pad=14)
    axs[2].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        f"Wireframe face outline — {len(blocks)} closed contours, "
        f"per-contour elliptic Fourier fit (aggregate nearest RMS {agg_f:.2e}, height = 1)",
        fontsize=13,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("face_wireframe_preview.png", dpi=150)
    plt.close(fig)

    # 7. fit comparison: harmonic sweep + Fourier vs spline
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(15, 6.2))
    for k, rec in enumerate(cleaned):
        ns = sorted(rec["sweep"])
        axl.plot(ns, [rec["sweep"][n]["nearest_rms"] for n in ns], "o-", ms=3.5,
                 lw=1.4, label=f"c{k} {ROLE_HINTS.get(rec['cv_index'],'')[:22]}")
        axl.plot([rec["N"]], [rec["err"]["nearest_rms"]], "*", ms=15, color="k", zorder=5)
    axl.axhline(RMS_TOL, color="#d1495b", ls="--", lw=1.2, label=f"tolerance {RMS_TOL}")
    axl.set_xscale("log"); axl.set_yscale("log")
    axl.set_xlabel("harmonics N"); axl.set_ylabel("nearest-point RMS (height = 1)")
    axl.set_title("Harmonic sweep per contour (★ = selected)")
    axl.grid(alpha=0.25, which="both", ls=":")
    axl.legend(fontsize=7, ncol=2)

    idx = np.arange(len(cleaned))
    fvals = [r["err"]["nearest_rms"] for r in cleaned]
    svals = [r["spline"]["nearest_rms"] if r["spline"] else np.nan for r in cleaned]
    axr.bar(idx - 0.2, fvals, 0.4, label="elliptic Fourier", color="#2a628f")
    axr.bar(idx + 0.2, svals, 0.4, label="periodic cubic B-spline", color="#d1495b")
    axr.set_yscale("log")
    axr.set_xticks(idx)
    axr.set_xticklabels([f"c{k}\n{r['n_params_fourier']}p" for k, r in enumerate(cleaned)],
                        fontsize=8)
    axr.set_ylabel("nearest-point RMS (height = 1)")
    axr.set_title("Fourier vs spline at a matched parameter budget")
    axr.grid(alpha=0.25, axis="y", ls=":")
    axr.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("face_wireframe_fit_comparison.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- console
    print(json.dumps(REPORT, indent=2))
    print(f"\nshared centroid (px, y-up): ({centroid[0]:.4f}, {centroid[1]:.4f})")
    print(f"head height (px)          : {height_px:.4f}   scale = {scale:.10f}")
    print(f"contours kept             : {len(blocks)}   total points {total_pts}")
    print(f"\n{'id':>3} {'depth':>5} {'raw':>6} {'clean':>6} {'resamp':>7} {'perim':>8} "
          f"{'N':>4} {'par':>5} {'nearRMS':>11} {'nearMAX':>11} {'ok':>4} {'xi':>3}  role")
    for k, (blk, rec) in enumerate(zip(blocks, cleaned)):
        e = blk["reconstruction_error"]
        print(f"{k:>3} {blk['nesting_depth']:>5} {blk['raw_points']:>6} "
              f"{blk['cleaned_points']:>6} {blk['resampled_points']:>7} "
              f"{blk['perimeter_normalized']:>8.4f} {blk['harmonics']:>4} "
              f"{blk['n_parameters']:>5} {e['nearest_rms']:>11.4e} {e['nearest_max']:>11.4e} "
              f"{str(blk['harmonics_tolerance_met']):>4} {rec['self_intersections_recon']:>3}  "
              f"{blk['role_hint']}")
    print(f"\naggregate nearest RMS  Fourier {agg_f:.6e}   spline {agg_s:.6e}")
    for k, rec in enumerate(cleaned):
        sp = rec["spline"]
        if sp:
            print(f"  c{k}: fourier {rec['n_params_fourier']:>4}p -> {rec['err']['nearest_rms']:.4e} | "
                  f"spline {sp['n_params']:>4}p ({sp['n_control_points']} ctrl) -> {sp['nearest_rms']:.4e}")


# Descriptive hints, keyed by the OpenCV contour index and derived from each
# contour's measured bounding box within the drawing. They label regions of the
# artwork, not anatomy of a person.
ROLE_HINTS = {
    0: "outer head + neck boundary",
    5: "mid-face paper region (glasses lower edge, nose, lip, cheek, ear)",
    1: "lower face / neck paper region (beard edge, jaw, throat)",
    8: "forehead paper region (hairline, glasses upper edge)",
    4: "mustache gap detail",
    7: "brow gap detail",
    2: "throat mark detail",
    6: "lip / beard detail",
}

if __name__ == "__main__":
    main()
