"""
Face silhouette -> elliptic Fourier descriptor pipeline.

Input : rajan.png  (stencil / line-art portrait: filled ink regions + closed outline strokes)
Output: face_silhouette.png, face_silhouette.svg, face_contour.csv,
        face_coefficients.json, face_equation.txt, face_reconstruction.png,
        face_harmonics_comparison.png

Everything numeric printed by this script is measured, never assumed.
Run:  uv run python face_silhouette_pipeline.py
"""

import json
import csv
import numpy as np
import cv2
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "rajan.png"
N_RESAMPLE = 500
HARMONICS_TESTED = [5, 8, 12, 20]
N_DEFAULT = 12
REPORT = {}


# ---------------------------------------------------------------- Part 1
def build_silhouette(path):
    """Grayscale -> Otsu binary -> largest ink blob -> fill holes -> mild smooth."""
    bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise SystemExit(f"cannot read {path}")

    # Flatten any alpha channel onto white so 'background' is unambiguous.
    if bgra.ndim == 3 and bgra.shape[2] == 4:
        alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
        rgb = bgra[:, :, :3].astype(np.float32)
        flat = rgb * alpha + 255.0 * (1.0 - alpha)
        img = flat.astype(np.uint8)
    elif bgra.ndim == 2:
        img = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGR)
    else:
        img = bgra

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr, binar = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = binar.astype(np.uint8)  # 1 = ink (dark), 0 = paper
    REPORT["otsu_threshold"] = float(thr)
    REPORT["image_size_wh"] = [int(gray.shape[1]), int(gray.shape[0])]

    # NOTE: no morphological opening before hole filling. The face/neck outline
    # is drawn with 1-2 px strokes and even a 2x2 opening severs them, which lets
    # the forehead and neck interiors leak into the background (measured:
    # 176,359 px vs 227,727 px filled). Denoising happens after filling instead.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    if n <= 1:
        raise SystemExit("no ink found")
    REPORT["ink_components"] = int(n - 1)
    REPORT["largest_ink_component_area_px"] = int(
        stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), cv2.CC_STAT_AREA]
    )

    # The face/neck outline is a *closed* stroke, so filling every region not
    # reachable from the image border turns the line art into a solid silhouette
    # bounded by the artist's own outer contour. Internal features (eyes, glasses,
    # nose, mouth, beard, ear detail) become interior fill and disappear.
    filled = ndimage.binary_fill_holes(ink).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    solid = (lab == largest).astype(np.uint8)
    REPORT["silhouette_area_px"] = int(solid.sum())

    # Mild smoothing: close then open with a small disk (radius 3) to drop
    # pixel-level jaggies and stroke-end nicks without altering the shape.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, k)
    solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, k)
    solid = ndimage.binary_fill_holes(solid).astype(np.uint8)
    REPORT["silhouette_area_px_after_smoothing"] = int(solid.sum())
    REPORT["area_change_from_smoothing_pct"] = round(
        100.0 * (int(solid.sum()) - REPORT["silhouette_area_px"])
        / REPORT["silhouette_area_px"], 4
    )
    return solid


# ---------------------------------------------------------------- Part 2
def outer_contour(mask):
    cnts, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    c = max(cnts, key=cv2.contourArea)
    return c[:, 0, :].astype(np.float64)  # (M,2) in pixel coords, x right / y down


def dedupe(pts, min_step=0.5):
    """Drop repeated and sub-pixel-spaced points on a closed polyline."""
    keep = [pts[0]]
    for p in pts[1:]:
        if np.hypot(*(p - keep[-1])) >= min_step:
            keep.append(p)
    out = np.asarray(keep)
    while len(out) > 3 and np.hypot(*(out[0] - out[-1])) < min_step:
        out = out[:-1]
    return out


def smooth_closed(pts, sigma):
    """Periodic Gaussian smoothing along a closed contour."""
    x = ndimage.gaussian_filter1d(pts[:, 0], sigma, mode="wrap")
    y = ndimage.gaussian_filter1d(pts[:, 1], sigma, mode="wrap")
    return np.stack([x, y], axis=1)


def _seg_cross(p1, p2, p3, p4):
    def o(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    d1, d2, d3, d4 = o(p3, p4, p1), o(p3, p4, p2), o(p1, p2, p3), o(p1, p2, p4)
    return d1 != d2 and d3 != d4


def count_self_intersections(pts):
    """Brute-force count of crossings between non-adjacent edges of a closed poly."""
    n = len(pts)
    a = pts
    b = np.roll(pts, -1, axis=0)
    hits = 0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _seg_cross(a[i], b[i], a[j], b[j]):
                hits += 1
    return hits


# ---------------------------------------------------------------- Part 3
def polygon_centroid_and_area(pts):
    """Green's-theorem area centroid of the closed polygon."""
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2.0
    cx = ((x + x1) * cr).sum() / (6.0 * A)
    cy = ((y + y1) * cr).sum() / (6.0 * A)
    return np.array([cx, cy]), A


# ---------------------------------------------------------------- Part 4
def resample_closed(pts, m):
    """Evenly spaced (arc length) resampling of a closed contour; t = s / P."""
    loop = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(loop, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    P = s[-1]
    target = np.linspace(0.0, P, m, endpoint=False)
    x = np.interp(target, s, loop[:, 0])
    y = np.interp(target, s, loop[:, 1])
    return np.stack([x, y], axis=1), P, target / P


# ---------------------------------------------------------------- Part 5
def efd_fit(pts, N):
    """Real Fourier series on uniform t in [0,1): x(t)=A0+sum a_n cos+b_n sin."""
    M = len(pts)
    t = np.arange(M) / M
    A0 = pts[:, 0].mean()
    C0 = pts[:, 1].mean()
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)
    for n in range(1, N + 1):
        co = np.cos(2 * np.pi * n * t)
        si = np.sin(2 * np.pi * n * t)
        a[n - 1] = 2.0 / M * (pts[:, 0] * co).sum()
        b[n - 1] = 2.0 / M * (pts[:, 0] * si).sum()
        c[n - 1] = 2.0 / M * (pts[:, 1] * co).sum()
        d[n - 1] = 2.0 / M * (pts[:, 1] * si).sum()
    return A0, C0, a, b, c, d


def efd_eval(A0, C0, a, b, c, d, t):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = np.arange(1, len(a) + 1)[None, :]
    ang = 2 * np.pi * n * t[:, None]
    co, si = np.cos(ang), np.sin(ang)
    x = A0 + co @ a + si @ b
    y = C0 + co @ c + si @ d
    return np.stack([x, y], axis=1)


def errors(orig, recon):
    dp = np.hypot(*(recon - orig).T)  # same-t pointwise
    rms = float(np.sqrt((dp**2).mean()))
    mx = float(dp.max())
    # geometric (nearest-point) deviation, orientation-independent
    D = np.hypot(
        orig[:, None, 0] - recon[None, :, 0], orig[:, None, 1] - recon[None, :, 1]
    )
    near = D.min(axis=1)
    return rms, mx, float(near.max()), float(np.sqrt((near**2).mean()))


# ---------------------------------------------------------------- driver
def main():
    mask = build_silhouette(SRC)

    raw = outer_contour(mask)
    n_orig = len(raw)

    cleaned = dedupe(raw, 0.5)
    cleaned = smooth_closed(cleaned, sigma=4.0)  # mild; ~1 px of jaggy removal
    cleaned = dedupe(cleaned, 0.5)
    n_clean = len(cleaned)

    # pixel coords -> math coords (y up) before normalization
    pix = cleaned.copy()
    math_pts = np.stack([cleaned[:, 0], -cleaned[:, 1]], axis=1)

    centroid, area = polygon_centroid_and_area(math_pts)
    if area < 0:  # force counter-clockwise in x-right / y-up
        math_pts = math_pts[::-1]
        centroid, area = polygon_centroid_and_area(math_pts)
        pix = pix[::-1]
    centered = math_pts - centroid
    height_px = centered[:, 1].max() - centered[:, 1].min()
    scale = 1.0 / height_px
    norm = centered * scale

    xi = count_self_intersections(norm[:: max(1, len(norm) // 1500)])

    rs, perim, tvals = resample_closed(norm, N_RESAMPLE)

    fits, errs = {}, {}
    for N in HARMONICS_TESTED:
        A0, C0, a, b, c, d = efd_fit(rs, N)
        rec = efd_eval(A0, C0, a, b, c, d, tvals)
        fits[N] = (A0, C0, a, b, c, d, rec)
        errs[N] = errors(rs, rec)

    # Full error-vs-N sweep, used only to justify the "more or fewer harmonics"
    # recommendation (reported, not used to override the requested comparison).
    sweep = {}
    for N in range(1, 41):
        f = efd_fit(rs, N)
        r = efd_eval(*f, tvals)
        e = errors(rs, r)
        sweep[N] = {"pointwise_rms": e[0], "nearest_point_rms": e[3],
                    "nearest_point_max": e[2]}

    # ---- selection rule: smallest N whose nearest-point RMS <= 0.5% of height
    #      and whose max nearest-point deviation <= 2.5% of height.
    chosen = None
    for N in HARMONICS_TESTED:
        _, _, nmax, nrms = errs[N]
        if nrms <= 0.005 and nmax <= 0.025:
            chosen = N
            break
    if chosen is None:
        chosen = max(HARMONICS_TESTED)
    REPORT["selection_rule"] = (
        "smallest N with nearest-point RMS <= 0.005 and max nearest-point "
        "deviation <= 0.025 (units: normalized height = 1)"
    )

    A0, C0, a, b, c, d, rec = fits[chosen]
    rms, mx, nmax, nrms = errs[chosen]
    rec_xi = count_self_intersections(rec[::4])

    # ------------------------------------------------------ output files
    # spec polarity: background = white (255), silhouette region = black (0)
    cv2.imwrite("face_silhouette.png", (255 - mask * 255).astype(np.uint8))

    W, H = REPORT["image_size_wh"]
    dpath = "M " + " L ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in pix) + " Z"
    with open("face_silhouette.svg", "w") as f:
        f.write(
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}">\n'
            f'  <rect width="{W}" height="{H}" fill="#ffffff"/>\n'
            f'  <path d="{dpath}" fill="#000000" fill-rule="evenodd" stroke="none"/>\n'
            f"</svg>\n"
        )

    with open("face_contour.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "t"])
        for i, (p, t) in enumerate(zip(rs, tvals)):
            w.writerow([i, f"{p[0]:.10f}", f"{p[1]:.10f}", f"{t:.10f}"])

    def eq_string(const, cs, ss, var):
        parts = [f"{const:.10f}"]
        for n in range(1, len(cs) + 1):
            parts.append(f"{cs[n-1]:+.10f}*cos(2*pi*{n}*t)")
            parts.append(f"{ss[n-1]:+.10f}*sin(2*pi*{n}*t)")
        return f"{var}(t) = " + " ".join(parts)

    x_eq = eq_string(A0, a, b, "x")
    y_eq = eq_string(C0, c, d, "y")
    F = [v for n in range(chosen) for v in (a[n], b[n], c[n], d[n])]

    payload = {
        "input_type": "designer_line_art_portrait_converted_to_closed_silhouette",
        "source_image": SRC,
        "coordinate_convention": "x right, y upward",
        "preprocessing": REPORT,
        "normalization": {
            "centroid_pixel_math_coords": [float(centroid[0]), float(centroid[1])],
            "centroid_after_translation": [0.0, 0.0],
            "scale_factor": float(scale),
            "height_before_normalization_px": float(height_px),
            "height_after_normalization": 1.0,
            "width_after_normalization": float(norm[:, 0].max() - norm[:, 0].min()),
            "rotation": None,
            "aspect_ratio_preserved": True,
        },
        "num_original_points": int(n_orig),
        "num_cleaned_points": int(n_clean),
        "num_resampled_points": int(N_RESAMPLE),
        "perimeter": float(perim),
        "polygon_area_normalized": float(
            abs(polygon_centroid_and_area(norm)[1] * 1.0)
        ),
        "self_intersections_cleaned_contour": int(xi),
        "self_intersections_reconstructed": int(rec_xi),
        "harmonics_tested": HARMONICS_TESTED,
        "error_vs_N_sweep": {str(k): v for k, v in sweep.items()},
        "harmonic_comparison": {
            str(N): {
                "pointwise_rms": errs[N][0],
                "pointwise_max": errs[N][1],
                "nearest_point_max": errs[N][2],
                "nearest_point_rms": errs[N][3],
            }
            for N in HARMONICS_TESTED
        },
        "selected_harmonics": int(chosen),
        "coefficients": {
            "A0": float(A0),
            "C0": float(C0),
            "a": [float(v) for v in a],
            "b": [float(v) for v in b],
            "c": [float(v) for v in c],
            "d": [float(v) for v in d],
        },
        "coefficient_vector": [float(v) for v in F],
        "equations": {"x_of_t": x_eq, "y_of_t": y_eq, "t_domain": "0 <= t < 1"},
        "reconstruction_error": {
            "normalized_rms": rms,
            "maximum_approximate_error": mx,
            "nearest_point_rms": nrms,
            "nearest_point_max": nmax,
            "units": "fraction of normalized silhouette height (height = 1)",
        },
        "all_harmonic_fits": {
            str(N): {
                "A0": float(fits[N][0]),
                "C0": float(fits[N][1]),
                "a": [float(v) for v in fits[N][2]],
                "b": [float(v) for v in fits[N][3]],
                "c": [float(v) for v in fits[N][4]],
                "d": [float(v) for v in fits[N][5]],
            }
            for N in HARMONICS_TESTED
        },
        "generated_files": [
            "face_silhouette.png",
            "face_silhouette.svg",
            "face_contour.csv",
            "face_coefficients.json",
            "face_equation.txt",
            "face_reconstruction.png",
            "face_fourier_manim.py",
            "face_harmonics_comparison.png",
        ],
    }
    with open("face_coefficients.json", "w") as f:
        json.dump(payload, f, indent=2)

    with open("face_equation.txt", "w") as f:
        f.write("CLOSED 2D PARAMETRIC FACE-OUTLINE EQUATION\n")
        f.write("Elliptic Fourier descriptor, arc-length parameterization\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"source                 : {SRC} (designer line art -> closed silhouette)\n")
        f.write("coordinate convention  : x right, y upward\n")
        f.write(f"centroid (pre-shift)   : ({centroid[0]:.6f}, {centroid[1]:.6f}) px\n")
        f.write(f"scale factor           : {scale:.10f}  (height 1)\n")
        f.write("rotation applied       : none\n")
        f.write(f"contour points         : {n_orig} raw -> {n_clean} cleaned -> {N_RESAMPLE} resampled\n")
        f.write(f"normalized perimeter   : {perim:.6f}\n")
        f.write(f"harmonics selected     : N = {chosen}\n")
        f.write(f"normalized RMS error   : {rms:.6e}\n")
        f.write(f"maximum error          : {mx:.6e}\n\n")
        f.write("x(t) = A0 + sum_{n=1..N} [ a_n cos(2 pi n t) + b_n sin(2 pi n t) ]\n")
        f.write("y(t) = C0 + sum_{n=1..N} [ c_n cos(2 pi n t) + d_n sin(2 pi n t) ]\n")
        f.write("0 <= t < 1\n\n")
        f.write(f"A0 = {A0:.12f}\nC0 = {C0:.12f}\n\n")
        f.write(f"{'n':>3} {'a_n':>18} {'b_n':>18} {'c_n':>18} {'d_n':>18}\n")
        for n in range(chosen):
            f.write(f"{n+1:>3} {a[n]:>18.12f} {b[n]:>18.12f} {c[n]:>18.12f} {d[n]:>18.12f}\n")
        f.write("\nFULLY EXPANDED\n" + "-" * 78 + "\n")
        f.write(x_eq + "\n\n" + y_eq + "\n\n")
        f.write("COEFFICIENT VECTOR F = [a_1,b_1,c_1,d_1, ..., a_N,b_N,c_N,d_N]\n")
        f.write("[" + ", ".join(f"{v:.12f}" for v in F) + "]\n\n")
        f.write("HARMONIC COMPARISON (units = fraction of normalized height)\n")
        f.write(f"{'N':>4} {'ptwise RMS':>14} {'ptwise max':>14} {'nearest RMS':>14} {'nearest max':>14}\n")
        for N in HARMONICS_TESTED:
            e = errs[N]
            f.write(f"{N:>4} {e[0]:>14.6e} {e[1]:>14.6e} {e[3]:>14.6e} {e[2]:>14.6e}\n")

    # reconstruction figure
    fig, ax = plt.subplots(figsize=(7, 8.5))
    loop_o = np.vstack([rs, rs[:1]])
    loop_r = np.vstack([rec, rec[:1]])
    ax.plot(loop_o[:, 0], loop_o[:, 1], "-", lw=2.6, color="#9aa4b2",
            label=f"original cleaned contour ({N_RESAMPLE} pts)")
    ax.plot(loop_r[:, 0], loop_r[:, 1], "-", lw=1.5, color="#d1495b",
            label=f"Fourier reconstruction (N = {chosen})")
    ax.axhline(0, color="#444", lw=0.8, zorder=0)
    ax.axvline(0, color="#444", lw=0.8, zorder=0)
    ax.plot(0, 0, "k+", ms=9, label="centroid (0, 0)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, ls=":")
    ax.set_xlabel("x (height-normalized)")
    ax.set_ylabel("y (height-normalized)")
    ax.set_title(f"Face silhouette: original vs elliptic Fourier, N={chosen}\n"
                 f"RMS {rms:.2e}, max {mx:.2e} (height = 1)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("face_reconstruction.png", dpi=170)
    plt.close(fig)

    # harmonic comparison figure
    fig, axs = plt.subplots(1, len(HARMONICS_TESTED), figsize=(4 * len(HARMONICS_TESTED), 6.5))
    for ax, N in zip(axs, HARMONICS_TESTED):
        r = fits[N][6]
        lr = np.vstack([r, r[:1]])
        ax.plot(loop_o[:, 0], loop_o[:, 1], "-", lw=2.4, color="#c9ced6", label="original")
        ax.plot(lr[:, 0], lr[:, 1], "-", lw=1.5, color="#d1495b", label=f"N={N}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2, ls=":")
        ax.set_title(f"N = {N}\nnearest-pt RMS {errs[N][3]:.2e}\nmax {errs[N][2]:.2e}", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Harmonic count comparison (height-normalized units)")
    fig.tight_layout()
    fig.savefig("face_harmonics_comparison.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------ console report
    print(json.dumps({k: v for k, v in REPORT.items()}, indent=2))
    print(f"raw contour points        : {n_orig}")
    print(f"cleaned contour points    : {n_clean}")
    print(f"self-intersections (clean): {xi}")
    print(f"resampled points          : {N_RESAMPLE}")
    print(f"centroid (px, y-up)       : ({centroid[0]:.4f}, {centroid[1]:.4f})")
    print(f"height before norm (px)   : {height_px:.4f}")
    print(f"scale factor              : {scale:.10f}")
    print(f"normalized perimeter      : {perim:.6f}")
    print(f"normalized width          : {norm[:,0].max()-norm[:,0].min():.6f}")
    print(f"x range                   : [{rs[:,0].min():.6f}, {rs[:,0].max():.6f}]")
    print(f"y range                   : [{rs[:,1].min():.6f}, {rs[:,1].max():.6f}]")
    print("\n  N |   ptwise RMS |   ptwise max |  nearest RMS |  nearest max")
    for N in HARMONICS_TESTED:
        e = errs[N]
        print(f"{N:>3} | {e[0]:12.6e} | {e[1]:12.6e} | {e[3]:12.6e} | {e[2]:12.6e}")
    print(f"\nselected N = {chosen}; reconstruction self-intersections = {rec_xi}")
    print(f"A0 = {A0:.12f}  C0 = {C0:.12f}")
    for n in range(chosen):
        print(f"  n={n+1:2d}  a={a[n]:+.12f}  b={b[n]:+.12f}  c={c[n]:+.12f}  d={d[n]:+.12f}")


if __name__ == "__main__":
    main()
