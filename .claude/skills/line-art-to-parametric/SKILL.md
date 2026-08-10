---
name: line-art-to-parametric
description: Convert a face photo, silhouette, line-art illustration, or pen-and-ink wireframe into parametric curve equations (elliptic Fourier descriptors or piecewise cubic Bézier paths) and animate them being drawn in Manim. Use when asked to turn an image into a contour equation, fit Fourier/spline/Bézier curves to a traced drawing, build a "signature being written" or wireframe-trace animation, or extract centerlines/contours from artwork.
---

# Line art → parametric curves → Manim

The single decision that determines whether the output is good: **pick the primitive
from what the image is made of.** Getting this wrong produces a technically correct
but useless result (a filled blob, or every line traced twice).

## Step 1: classify the input before writing any pipeline

Always measure first. Never assume from the thumbnail.

```python
ink_fraction   = ink.mean()                       # ink = Otsu-thresholded dark pixels
n_tree         = len(cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0])
sk             = skimage.morphology.skeletonize(ink.astype(bool))
stroke_width   = np.median(2 * cv2.distanceTransform(ink, cv2.DIST_L2, 5)[sk])
```

| Measurement | Input regime | Primitive | Fit with |
|---|---|---|---|
| ink ≳ 10%, few contours, large solid masses | photo / silhouette / stencil | **outer region boundary** (`RETR_EXTERNAL` + `binary_fill_holes`) | elliptic Fourier, one closed curve |
| ink ≳ 10%, solid masses **and** thin outline strokes | stencil line art | **all region boundaries** (`RETR_TREE`, every nesting level) | elliptic Fourier, one per contour |
| ink ≲ 5%, `n_tree` ≫ visible line count, stroke width 2–6 px | pen-and-ink wireframe | **centerlines** (skeleton graph) | piecewise cubic Bézier |

**The tell for a stroke drawing:** `n_tree` is roughly *twice* the number of lines you
can count by eye, because region tracing follows both edges of every pen stroke. On a
1296×1213 wireframe this was 211 contours vs 140 actual strokes. If you see that,
stop and switch to skeletonization.

**Second tell:** count open vs closed curves after extraction. If most curves are
**open**, a periodic Fourier series is the wrong model — it must wrap end→start, so it
throws an arc across empty space to close the curve and rings (Gibbs) along the rest.
Measured on a real wireframe: at equal parameter count Fourier was **4.4× worse in RMS
and 15× worse in max error** than cubic Bézier. Reserve Fourier for genuinely closed loops.

## Step 2: run the matching pipeline

`scripts/` holds three working, self-contained pipelines. Copy the one that matches
and adjust constants at the top; do not rewrite from scratch.

- `face_silhouette_pipeline.py` — silhouette → one closed contour → elliptic Fourier,
  with an N = 5/8/12/20 harmonic sweep and a selection rule.
- `face_wireframe_pipeline.py` — stencil line art → `RETR_TREE` multi-contour →
  per-contour Fourier with independently chosen N, one shared normalization.
- `wireframe_pipeline.py` — pen-and-ink wireframe → skeleton graph → pruned,
  tangent-merged pen strokes → adaptive cubic Bézier (Schneider). **Best starting point
  for any real line drawing.**

Matching Manim scenes: `face_fourier_manim.py`, `face_wireframe_manim.py`,
`wireframe_manim.py`.

## Step 3: the traps that actually cost time

Each of these produced a wrong-looking result before being found.

### Image processing

**Never morphologically open before hole-filling.** Even a 2×2 opening severs 1-px
outline strokes, and the enclosed regions then leak into the background. Measured:
176,359 px filled instead of 227,727 — the forehead and neck silently escaped.
Denoise *after* filling, or by dropping short contours, which cannot break a stroke.

**Sort contours by `(depth, -perimeter)`, not perimeter alone.** A wiggly interior
region can be longer than the outer boundary (2628 px vs 2028 px here), so a plain
perimeter sort anchors normalization to the wrong contour and the reported height is
nonsense. Assert that the chosen outer contour has `depth == 0`.

**Apply one shared normalization to every curve.** Per-curve centroid/scale destroys
the relative geometry. Anchor to the outer boundary (or the union bbox) and reuse that
transform everywhere.

### Bézier fitting (Schneider)

**The least-squares residual must subtract `(B0+B1)·P0 + (B2+B3)·P3`**, not just
`B0·P0 + B3·P3`. Because `P1 = P0 + t₁a₁`, the endpoints contribute through the middle
basis functions too. Getting this wrong corrupts the normal equations, so the fitter
*splits* instead of fitting: **2,034 segments before the fix, 354 after**, at the same
accuracy. If segment count looks absurd, suspect this before touching the tolerance.

**Force `t2 = -t1` on closed strokes** so the path is G1 across the seam.

**Measure error against a densely sampled curve.** At 24–64 samples/segment the
evaluation polyline deviates more than the fit does and inflates max error ~4×
(3.7 px reported vs 0.79 px actual). Use ≥256 samples/segment for error metrics only.

### Skeleton → strokes

Raw skeleton branches are far too fragmented (464 branches, median length 5 px).
Three passes fix it, in order:
1. **Prune spurs**: drop dangling branches ≤ 4 px whose far node has degree ≥ 3.
2. **Merge through junctions by tangent continuity**: pair branches whose outgoing
   unit directions satisfy `dot ≤ -0.60` (meeting at ≥ ~127°). This is what turns
   fragments into natural pen strokes and made the head outline a single 950-px curve.
3. **Drop strokes below a minimum length**, and *report the count and total pixels* —
   never truncate silently. (100 strokes / 126 px = 1.25% of the skeleton was fine.)

### Manim

**`always_redraw(f)` + `become` does NOT clip a curve.** The point array is correctly
truncated, but `become` realigns the arrays and the full loop reappears — the whole
shape shows up at t = 0.03. Use an **in-place** updater instead:

```python
def upd(mob):
    fr = float(np.clip(tracker.get_value(), 1e-4, 1.0))
    mob.pointwise_become_partial(reference, 0.0, fr)   # reference is never added
mob.add_updater(upd)
```

**Bake layout offsets into the coordinate transform, not `move_to()`.** The updater
above rewrites points from `reference` every frame, so any `move_to()`/`shift()` on the
drawn mobject is silently undone. Put the shift inside the shared `to_scene()` function
that references, strokes, ghost and pen dot all go through.

**`MathTex` and `DecimalNumber` shell out to `latex`**, which pip does not install.
Detect and degrade:

```python
HAVE_LATEX = shutil.which("latex") is not None
NUMBER_MOB = MathTex if HAVE_LATEX else Text      # DecimalNumber(..., mob_class=NUMBER_MOB)
```

**A `VMobject` *is* a cubic Bézier path.** Feed fitted control points straight in with
`start_new_path(P0)` + `add_cubic_bezier_curve_to(P1, P2, P3)` — zero resampling loss
between fit and render. This alone justifies Bézier over Fourier for Manim work.

**Uniform scale only.** One factor on both axes preserves the face shape; per-axis
scaling or `Axes` with mismatched ranges distorts it.

## Step 4: animation that reads well

- Trace at **constant pen speed in arc length**, not constant time per stroke: map one
  global `ValueTracker` through cumulative stroke lengths. Use `rate_func=linear` so the
  stroke, pen dot and any readout stay in lockstep.
- Order strokes **longest first**. Structural lines land before detail, and the hatching
  dashes in at the end — it reads like an artist working.
- A faint full-drawing "ghost" (opacity ≈ 0.15) fading in first is a cheap, effective
  reveal. Keep captions at `GREY_B` or lighter; `GREY_D` on black is unreadable.
- Keep everything stroked: `set_fill(opacity=0)` explicitly.

## Reporting

State which representation was used (edge mask / contour path / Fourier descriptor /
spline / Bézier) and **why**, backed by measured numbers. Always report: point counts
before and after cleaning, normalization (centroid, scale, rotation), fit RMS **and**
max error in normalized units *and* pixels, parameter count, and anything dropped.
Never invent a coefficient or an error value.

Sub-pixel is the bar: on a 4-px-wide pen stroke, a good Bézier fit lands at
RMS ≈ 0.2 px, max ≈ 0.8 px.
