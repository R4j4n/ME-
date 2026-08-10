"""
wireframe_manim.py — Manim Community Edition

Draws the designer wireframe purely from the piecewise cubic Bezier equations in
wireframe_curves.json. Every stroke is a thin stroked path; nothing is filled and
the source image is never shown.

Each stroke k is a chain of cubic Bezier segments
    B_j(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3,   0 <= t <= 1
with P3 of segment j equal to P0 of segment j+1.

Those control points are fed straight into VMobject.add_cubic_bezier_curve_to --
a Manim VMobject IS a cubic Bezier path, so there is no resampling loss anywhere
between the fit and the render.

Strokes are traced one after another at constant pen speed measured in arc
length, longest first, so structural lines land before the hatching -- it reads
like an artist working, not like every line growing at once.

Render
------
    manim -pqh wireframe_manim.py WireframeFace         # 1080p60
    manim -pql wireframe_manim.py WireframeFace         # fast 480p15 preview
    manim -pqk wireframe_manim.py WireframeFace         # 4K
    manim -s -pqh wireframe_manim.py WireframeFace      # final frame only (PNG)
    manim -pqh wireframe_manim.py WireframeStrokeOrder  # colour = drawing order

    # transparent background, for compositing:
    manim -qh --format=mov -t wireframe_manim.py WireframeFace

Requires: pip install "manim>=0.18"  (plus system cairo / pango / ffmpeg)
Run from the directory holding wireframe_curves.json, or set FACE_JSON.
"""

import json
import os

import numpy as np
from manim import (
    BLUE_B,
    DOWN,
    GREY_B,
    TEAL_A,
    Dot,
    FadeIn,
    FadeOut,
    Scene,
    Text,
    VGroup,
    VMobject,
    ValueTracker,
    Write,
    color_gradient,
    linear,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
FACE_JSON = os.environ.get("FACE_JSON", "wireframe_curves.json")

DRAW_TIME = 18.0          # seconds to trace the whole drawing
TARGET_HEIGHT = 6.6       # on-screen height of the drawing, in Manim units
STROKE_WIDTH = 2.2        # thin line: this is a wireframe, not a fill
LAYOUT_SHIFT = np.array([0.0, 0.42, 0.0])

SHOW_GHOST = True         # subtle reveal: faint full drawing under the strokes
GHOST_OPACITY = 0.14
SHOW_PEN = True
SHOW_CAPTION = True

INK = BLUE_B
GHOST = GREY_B
PEN = TEAL_A


# --------------------------------------------------------------------------- #
# Load the fitted curves
# --------------------------------------------------------------------------- #
def load_curves(path=FACE_JSON):
    with open(path, "r") as fh:
        data = json.load(fh)
    strokes = []
    for blk in data["strokes"]:
        segs = np.asarray(blk["control_points"], dtype=float)   # (n_seg, 4, 2)
        strokes.append(
            {
                "id": int(blk["stroke_id"]),
                "closed": bool(blk["closed"]),
                "length": float(blk["arc_length_normalized"]),
                "segments": segs,
            }
        )
    return data, strokes


DATA, STROKES = load_curves()

# One uniform scale for every stroke -- the same factor on x and y is what
# preserves the face shape. Bounds come from all control points.
_ALL = np.concatenate([s["segments"].reshape(-1, 2) for s in STROKES], axis=0)
_X0, _Y0 = _ALL.min(axis=0)
_X1, _Y1 = _ALL.max(axis=0)
SCALE = TARGET_HEIGHT / (_Y1 - _Y0)
_CENTER = np.array([0.5 * (_X0 + _X1), 0.5 * (_Y0 + _Y1)])


def to_scene(p):
    """Normalized curve coords -> scene coords.

    Everything (references, visible strokes, ghost, pen) goes through this, so
    they stay registered even though the stroke updater rewrites points from the
    references every frame -- a move_to() on the strokes would be undone.
    """
    q = (np.asarray(p, dtype=float) - _CENTER) * SCALE
    return np.array([q[0], q[1], 0.0]) + LAYOUT_SHIFT


def stroke_mobject(st, color, width, opacity=1.0):
    """Build a VMobject directly from the fitted cubic Bezier control points."""
    vm = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    vm.set_fill(opacity=0.0)              # explicit: wireframe, never filled
    for j, seg in enumerate(st["segments"]):
        p0, p1, p2, p3 = (to_scene(p) for p in seg)
        if j == 0:
            vm.start_new_path(p0)
        vm.add_cubic_bezier_curve_to(p1, p2, p3)
    return vm


# --------------------------------------------------------------------------- #
# Main scene
# --------------------------------------------------------------------------- #
class WireframeFace(Scene):
    """Traces the whole wireframe stroke by stroke at constant pen speed."""

    def construct(self):
        order = sorted(range(len(STROKES)), key=lambda i: -STROKES[i]["length"])
        lengths = np.array([STROKES[i]["length"] for i in order], dtype=float)
        total = float(lengths.sum())
        starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])

        refs = [stroke_mobject(STROKES[i], INK, STROKE_WIDTH) for i in order]
        drawn = [r.copy() for r in refs]
        ghost = VGroup(
            *[stroke_mobject(STROKES[i], GHOST, STROKE_WIDTH, GHOST_OPACITY)
              for i in order]
        )

        u = ValueTracker(0.0)

        def frac_of(k):
            pen = u.get_value() * total
            return float(np.clip((pen - starts[k]) / lengths[k], 0.0, 1.0))

        def make_updater(k):
            def upd(mob):
                fr = frac_of(k)
                if fr <= 1e-6:
                    mob.set_stroke(opacity=0.0)
                    return
                mob.set_stroke(opacity=1.0)
                # In-place partial update. Rebuilding the mobject each frame and
                # handing it to always_redraw/become does NOT clip the path.
                mob.pointwise_become_partial(refs[k], 0.0, max(fr, 1e-4))
            return upd

        for k, d in enumerate(drawn):
            upd = make_updater(k)
            upd(d)
            d.add_updater(upd)

        cum = np.cumsum(lengths)

        def update_pen(mob):
            pen = u.get_value() * total
            k = min(int(np.searchsorted(cum, pen, side="right")), len(refs) - 1)
            mob.move_to(refs[k].point_from_proportion(frac_of(k)))

        pen_dot = Dot(radius=0.05, color=PEN)
        pen_halo = Dot(radius=0.15, color=PEN, fill_opacity=0.20)
        for m in (pen_dot, pen_halo):
            update_pen(m)
            m.add_updater(update_pen)

        caption = VGroup(
            Text("Face wireframe traced from its Bézier equations",
                 font_size=25, color=GREY_B),
            Text(f"{DATA['num_strokes']} pen strokes  ·  "
                 f"{DATA['total_bezier_segments']} cubic Bézier segments  ·  "
                 f"fit RMS {DATA['fit_quality']['bezier_nearest_rms']:.1e}",
                 font_size=17, color=GREY_B),
        ).arrange(DOWN, buff=0.16).to_edge(DOWN, buff=0.28)

        # ---- animate ---------------------------------------------------- #
        if SHOW_CAPTION:
            self.play(Write(caption), run_time=1.2)
        if SHOW_GHOST:
            self.play(FadeIn(ghost), run_time=1.4)      # subtle reveal

        self.add(*drawn)
        if SHOW_PEN:
            self.add(pen_halo, pen_dot)

        # linear keeps the pen speed constant in arc length
        self.play(u.animate.set_value(1.0), run_time=DRAW_TIME, rate_func=linear)

        for d in drawn:
            d.clear_updaters()
        pen_dot.clear_updaters()
        pen_halo.clear_updaters()

        if SHOW_PEN:
            self.play(FadeOut(pen_halo), FadeOut(pen_dot), run_time=0.5)
        if SHOW_GHOST:
            self.play(FadeOut(ghost), run_time=0.8)
        self.wait(2.0)


# --------------------------------------------------------------------------- #
# Bonus scene: colour by drawing order
# --------------------------------------------------------------------------- #
class WireframeStrokeOrder(Scene):
    """Same wireframe, coloured by the order the strokes are drawn in."""

    def construct(self):
        order = sorted(range(len(STROKES)), key=lambda i: -STROKES[i]["length"])
        cols = color_gradient(["#4C86D6", TEAL_A, "#F2C14E"], max(len(order), 2))
        group = VGroup(
            *[stroke_mobject(STROKES[i], cols[k], STROKE_WIDTH + 0.5)
              for k, i in enumerate(order)]
        )
        caption = Text(
            f"{len(order)} strokes, coloured by drawing order "
            f"(blue = first / longest → yellow = last / shortest)",
            font_size=18, color=GREY_B,
        ).to_edge(DOWN, buff=0.3)

        self.play(FadeIn(caption), run_time=0.8)
        self.play(*[Write(m) for m in group], run_time=6.0)
        self.wait(2.0)
