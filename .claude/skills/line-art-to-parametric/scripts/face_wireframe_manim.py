"""
face_wireframe_manim.py — Manim Community Edition

Draws a WIREFRAME face outline purely from the per-contour Fourier equations in
face_wireframe_coefficients.json. Every stroke is a thin stroked path; nothing
is ever filled, and the source image is never shown.

For each contour k:
    x_k(t) = A0 + sum_{n=1..N_k} [ a_n cos(2 pi n t) + b_n sin(2 pi n t) ]
    y_k(t) = C0 + sum_{n=1..N_k} [ c_n cos(2 pi n t) + d_n sin(2 pi n t) ]
    0 <= t < 1   (t = 1 reproduces t = 0, so every loop closes exactly)

The contours are drawn one after another, ordered outer boundary first, at a
constant pen speed measured in arc length -- so it reads like a signature being
written rather than every line growing at once.

Render
------
    manim -pqh face_wireframe_manim.py WireframeFace       # 1080p60
    manim -pql face_wireframe_manim.py WireframeFace       # fast 480p15 preview
    manim -pqk face_wireframe_manim.py WireframeFace       # 4K
    manim -s -pqh face_wireframe_manim.py WireframeFace    # final frame only (PNG)
    manim -pqh face_wireframe_manim.py WireframeContourMap # colour-coded contours

    # transparent background (for compositing over other footage):
    manim -qh --format=mov -t face_wireframe_manim.py WireframeFace

Requires: pip install "manim>=0.18"   (plus system cairo / pango / ffmpeg)
Run from the directory holding face_wireframe_coefficients.json, or set
the FACE_JSON environment variable.
"""

import json
import os

import numpy as np
from manim import (
    BLUE_B,
    DOWN,
    GREY_B,
    LEFT,
    TEAL_A,
    UP,
    VGroup,
    ValueTracker,
    Dot,
    FadeIn,
    FadeOut,
    ParametricFunction,
    Scene,
    Text,
    Write,
    color_gradient,
    config,
    linear,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
FACE_JSON = os.environ.get("FACE_JSON", "face_wireframe_coefficients.json")

DRAW_TIME = 12.0          # seconds to trace the whole wireframe
TARGET_HEIGHT = 6.4       # on-screen height of the head, in Manim units
STROKE_WIDTH = 2.6        # thin line -- this is a wireframe, not a fill
SAMPLES_PER_UNIT = 420    # polyline resolution per unit of normalized arc length
MIN_SAMPLES = 160

SHOW_GHOST = True         # subtle reveal: faint full wireframe under the strokes
GHOST_OPACITY = 0.16
SHOW_MOVING_DOT = True
SHOW_TITLE = True
LAYOUT_SHIFT = np.array([0.0, 0.45, 0.0])   # nudge the art clear of the caption

INK = BLUE_B
GHOST = GREY_B
PEN = TEAL_A
CAPTION = GREY_B


# --------------------------------------------------------------------------- #
# Load the equations
# --------------------------------------------------------------------------- #
def load_wireframe(path=FACE_JSON):
    with open(path, "r") as fh:
        data = json.load(fh)

    contours = []
    for blk in data["contours"]:
        co = blk["coefficients"]
        contours.append(
            {
                "id": blk["contour_id"],
                "role": blk.get("role_hint", ""),
                "A0": float(co["A0"]),
                "C0": float(co["C0"]),
                "a": np.asarray(co["a"], dtype=float),
                "b": np.asarray(co["b"], dtype=float),
                "c": np.asarray(co["c"], dtype=float),
                "d": np.asarray(co["d"], dtype=float),
                "N": int(blk["harmonics"]),
                "perimeter": float(blk["perimeter_normalized"]),
            }
        )
    return data, contours


DATA, CONTOURS = load_wireframe()


def make_eval(ct):
    """Return f(t) -> (x, y) in normalized units, for one contour."""
    orders = np.arange(1, ct["N"] + 1, dtype=float)
    a, b, c, d, A0, C0 = ct["a"], ct["b"], ct["c"], ct["d"], ct["A0"], ct["C0"]

    def f(t):
        ang = 2.0 * np.pi * orders * float(t)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        return A0 + float(a @ cos_a) + float(b @ sin_a), C0 + float(c @ cos_a) + float(d @ sin_a)

    return f


def wireframe_bounds(n=800):
    """Bounding box over every contour, so one uniform scale fits them all."""
    xs, ys = [], []
    for ct in CONTOURS:
        orders = np.arange(1, ct["N"] + 1, dtype=float)
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        ang = 2 * np.pi * np.outer(ts, orders)
        xs.append(ct["A0"] + np.cos(ang) @ ct["a"] + np.sin(ang) @ ct["b"])
        ys.append(ct["C0"] + np.cos(ang) @ ct["c"] + np.sin(ang) @ ct["d"])
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    return xs.min(), xs.max(), ys.min(), ys.max()


# One uniform scale + translation for every contour. Using the SAME factor on x
# and y is what preserves the face shape; anything per-axis would distort it.
_X0, _X1, _Y0, _Y1 = wireframe_bounds()
SCALE = TARGET_HEIGHT / (_Y1 - _Y0)
_CENTER = np.array([0.5 * (_X0 + _X1), 0.5 * (_Y0 + _Y1)])


def to_scene(xy):
    """Normalized contour coords -> scene coords.

    One uniform SCALE on both axes, plus a fixed layout shift. Every consumer
    (reference curves, visible strokes, ghost, pen dot) goes through this, so
    they stay registered even though the stroke updater rewrites points from
    the references every frame -- a move_to() on the strokes would be undone.
    """
    p = (np.asarray(xy, dtype=float) - _CENTER) * SCALE
    return np.array([p[0], p[1], 0.0]) + LAYOUT_SHIFT


def contour_mobject(ct, color, width, opacity=1.0):
    """One closed contour as a stroked ParametricFunction (never filled)."""
    f = make_eval(ct)
    n = max(MIN_SAMPLES, int(SAMPLES_PER_UNIT * ct["perimeter"]))
    mob = ParametricFunction(
        lambda t: to_scene(f(t)),
        t_range=[0.0, 1.0, 1.0 / n],
        stroke_color=color,
        stroke_width=width,
        stroke_opacity=opacity,
    )
    mob.set_fill(opacity=0.0)   # explicit: wireframe, no fill
    return mob


# --------------------------------------------------------------------------- #
# Main scene
# --------------------------------------------------------------------------- #
class WireframeFace(Scene):
    """Traces the whole wireframe, contour by contour, at constant pen speed."""

    def construct(self):
        lengths = np.array([ct["perimeter"] for ct in CONTOURS], dtype=float)
        total = float(lengths.sum())
        starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])

        # Reference geometry (never added to the scene) + the visible strokes.
        refs = [contour_mobject(ct, INK, STROKE_WIDTH) for ct in CONTOURS]
        strokes = [r.copy() for r in refs]
        evals = [make_eval(ct) for ct in CONTOURS]

        ghost = VGroup(
            *[contour_mobject(ct, GHOST, STROKE_WIDTH, GHOST_OPACITY) for ct in CONTOURS]
        )

        u = ValueTracker(0.0)

        def frac_of(i):
            """How much of contour i is drawn, from the global pen position."""
            pen = u.get_value() * total
            return float(np.clip((pen - starts[i]) / lengths[i], 0.0, 1.0))

        def make_updater(i):
            def upd(mob):
                fr = frac_of(i)
                if fr <= 1e-6:
                    mob.set_stroke(opacity=0.0)
                    return
                mob.set_stroke(opacity=1.0)
                # In-place partial update. Rebuilding a mobject each frame and
                # handing it to always_redraw/become does NOT clip the curve --
                # become realigns the point arrays and the full loop reappears.
                mob.pointwise_become_partial(refs[i], 0.0, max(fr, 1e-4))
            return upd

        for i, st in enumerate(strokes):
            upd = make_updater(i)
            upd(st)
            st.add_updater(upd)

        # Pen dot: sits on whichever contour is currently being traced.
        def update_pen(mob):
            pen = u.get_value() * total
            i = int(np.searchsorted(np.cumsum(lengths), pen, side="right"))
            i = min(i, len(CONTOURS) - 1)
            mob.move_to(to_scene(evals[i](frac_of(i))))

        pen_dot = Dot(radius=0.055, color=PEN)
        pen_halo = Dot(radius=0.16, color=PEN, fill_opacity=0.22)
        update_pen(pen_dot)
        update_pen(pen_halo)
        pen_dot.add_updater(update_pen)
        pen_halo.add_updater(update_pen)

        art = VGroup(*strokes)
        title = Text(
            "Face wireframe traced from its Fourier equations",
            font_size=26,
            color=CAPTION,
        )
        sub = Text(
            f"{len(CONTOURS)} closed contours   ·   "
            f"N = {min(c['N'] for c in CONTOURS)}–{max(c['N'] for c in CONTOURS)} harmonics"
            f"   ·   0 ≤ t < 1",
            font_size=18,
            color=CAPTION,
        )
        caption = VGroup(title, sub).arrange(DOWN, buff=0.18)
        caption.to_edge(DOWN, buff=0.3)

        # ---- animate ---------------------------------------------------- #
        if SHOW_TITLE:
            self.play(Write(caption), run_time=1.2)
        if SHOW_GHOST:
            self.play(FadeIn(ghost), run_time=1.4)   # subtle reveal

        self.add(art)
        if SHOW_MOVING_DOT:
            self.add(pen_halo, pen_dot)

        # linear keeps the pen speed constant in arc length
        self.play(u.animate.set_value(1.0), run_time=DRAW_TIME, rate_func=linear)

        for st in strokes:
            st.clear_updaters()
        pen_dot.clear_updaters()
        pen_halo.clear_updaters()

        if SHOW_MOVING_DOT:
            self.play(FadeOut(pen_halo), FadeOut(pen_dot), run_time=0.5)
        if SHOW_GHOST:
            self.play(FadeOut(ghost), run_time=0.8)
        self.wait(1.8)


# --------------------------------------------------------------------------- #
# Bonus scene: which equation draws which part
# --------------------------------------------------------------------------- #
class WireframeContourMap(Scene):
    """Same wireframe, one colour per contour, drawn all at once."""

    def construct(self):
        colors = color_gradient(["#4C86D6", TEAL_A, "#F2C14E"], max(len(CONTOURS), 2))
        strokes = VGroup(
            *[contour_mobject(ct, colors[i], STROKE_WIDTH + 0.4)
              for i, ct in enumerate(CONTOURS)]
        )
        # No updaters in this scene, so transforming the group is safe here.
        strokes.scale(0.80).to_edge(UP, buff=0.25)
        legend = VGroup(
            *[
                Text(f"c{ct['id']}  N={ct['N']}  {ct['role'][:52]}",
                     font_size=12, color=colors[i])
                for i, ct in enumerate(CONTOURS)
            ]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.09)
        legend.scale_to_fit_width(min(5.6, config.frame_width * 0.40))
        legend.to_corner(DOWN + LEFT, buff=0.3)

        self.play(FadeIn(legend), run_time=0.8)
        self.play(*[Write(s) for s in strokes], run_time=5.0)
        self.wait(2.0)
