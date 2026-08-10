"""
face_fourier_manim.py — Manim Community Edition

Draws the face outline *purely* from the elliptic Fourier equation stored in
face_coefficients.json. The source photograph / line art is never displayed.

    x(t) = A0 + sum_{n=1..N} [ a_n cos(2 pi n t) + b_n sin(2 pi n t) ]
    y(t) = C0 + sum_{n=1..N} [ c_n cos(2 pi n t) + d_n sin(2 pi n t) ]
    0 <= t < 1     (t = 1 reproduces t = 0, so the loop closes exactly)

Render
------
    manim -pqh face_fourier_manim.py FourierFace       # 1080p60, opens when done
    manim -pql face_fourier_manim.py FourierFace       # fast 480p15 preview
    manim -pqk face_fourier_manim.py FourierFace       # 4K
    manim -s  -pqh face_fourier_manim.py FourierFace   # final frame only (PNG)
    manim -pqh face_fourier_manim.py FourierFaceHarmonics   # N = 5/8/12/20 panel

    # transparent-background MOV:
    manim -qh --format=mov -t face_fourier_manim.py FourierFace

Requires:  pip install "manim>=0.18"   (plus system cairo / pango / ffmpeg)
Run from the directory that holds face_coefficients.json, or set FACE_JSON.
"""

import json
import os
import shutil

import numpy as np
from manim import (
    BLUE_B,
    DOWN,
    GREY_B,
    LEFT,
    RIGHT,
    UP,
    YELLOW,
    Axes,
    Create,
    DecimalNumber,
    Dot,
    FadeIn,
    FadeOut,
    Group,
    MathTex,
    ParametricFunction,
    Scene,
    Text,
    VGroup,
    ValueTracker,
    Write,
    config,
    linear,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
FACE_JSON = os.environ.get("FACE_JSON", "face_coefficients.json")

# None -> use the harmonic count selected by the fitting pipeline.
# Set to 5, 8, 12 or 20 to force one of the pre-computed alternative fits.
FORCE_HARMONICS = None

DRAW_TIME = 8.0          # seconds for the outline to draw
SHOW_MOVING_DOT = True   # requirement 8 (optional)
SHOW_T_READOUT = True    # requirement 9 (optional)
CURVE_COLOR = BLUE_B
SAMPLES = 900            # polyline samples used to render the parametric curve


# --------------------------------------------------------------------------- #
# Coefficient loading + the equation itself
# --------------------------------------------------------------------------- #
def load_fourier(path=FACE_JSON, force_n=FORCE_HARMONICS):
    """Return (N, A0, C0, a, b, c, d) from the pipeline's JSON."""
    with open(path, "r") as fh:
        data = json.load(fh)

    if force_n is not None and str(force_n) in data.get("all_harmonic_fits", {}):
        co = data["all_harmonic_fits"][str(force_n)]
        n_harm = int(force_n)
    else:
        co = data["coefficients"]
        n_harm = int(data["selected_harmonics"])

    a = np.asarray(co["a"], dtype=float)[:n_harm]
    b = np.asarray(co["b"], dtype=float)[:n_harm]
    c = np.asarray(co["c"], dtype=float)[:n_harm]
    d = np.asarray(co["d"], dtype=float)[:n_harm]
    return n_harm, float(co["A0"]), float(co["C0"]), a, b, c, d, data


N_HARM, A0, C0, A_N, B_N, C_N, D_N, DATA = load_fourier()
_ORDERS = np.arange(1, N_HARM + 1, dtype=float)


def face_xy(t):
    """Evaluate the closed parametric face outline at scalar t in [0, 1]."""
    ang = 2.0 * np.pi * _ORDERS * float(t)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    x = A0 + float(A_N @ cos_a) + float(B_N @ sin_a)
    y = C0 + float(C_N @ cos_a) + float(D_N @ sin_a)
    return x, y


def face_point(t):
    """3D point for Manim (z = 0). Kept in normalized contour units."""
    x, y = face_xy(t)
    return np.array([x, y, 0.0])


def contour_bounds(samples=2000):
    """Bounding box of the reconstructed curve, used to auto-fit the axes."""
    ts = np.linspace(0.0, 1.0, samples, endpoint=False)
    ang = 2.0 * np.pi * np.outer(ts, _ORDERS)
    xs = A0 + np.cos(ang) @ A_N + np.sin(ang) @ B_N
    ys = C0 + np.cos(ang) @ C_N + np.sin(ang) @ D_N
    return xs.min(), xs.max(), ys.min(), ys.max()


# MathTex / DecimalNumber shell out to `latex`, which pip does not install.
# Fall back to Pango text so the scene renders on a machine without TeX.
HAVE_LATEX = shutil.which("latex") is not None
NUMBER_MOB = MathTex if HAVE_LATEX else Text


def tex_or_text(tex, plain, **kwargs):
    """MathTex when a LaTeX install is available, plain Text otherwise."""
    if HAVE_LATEX:
        try:
            return MathTex(tex, **kwargs)
        except Exception:
            pass
    return Text(plain, **kwargs)


def build_axes(pad=0.10, tick=0.1, max_h=6.0, max_w=9.0):
    """Axes whose x and y units are the SAME length -> no distortion."""
    x0, x1, y0, y1 = contour_bounds()
    span = max(x1 - x0, y1 - y0)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    # Symmetric, equal-span ranges snapped outward to whole ticks.
    half = 0.5 * span * (1.0 + 2.0 * pad)
    half = tick * np.ceil(half / tick)
    xr = [cx - half, cx + half, tick]
    yr = [cy - half, cy + half, tick]

    # Equal spans + equal lengths => equal scale on both axes.
    side = min(max_h, max_w)
    return Axes(
        x_range=xr,
        y_range=yr,
        x_length=side,
        y_length=side,
        tips=False,
        axis_config={
            "include_numbers": False,
            "stroke_width": 2,
            "color": GREY_B,
            "include_ticks": True,
        },
    )


# --------------------------------------------------------------------------- #
# Main scene
# --------------------------------------------------------------------------- #
class FourierFace(Scene):
    """Progressively draws the face outline from the Fourier equation."""

    def construct(self):
        axes = build_axes()
        axes.to_edge(LEFT, buff=0.9)

        # ---- title + equation card ------------------------------------- #
        title = Text("Face outline from an elliptic Fourier series", font_size=30)
        title.to_edge(UP, buff=0.35)

        eq = tex_or_text(
            r"\begin{aligned}"
            r"x(t) &= A_0 + \sum_{n=1}^{%d}\left[a_n\cos(2\pi n t)"
            r"+ b_n\sin(2\pi n t)\right]\\"
            r"y(t) &= C_0 + \sum_{n=1}^{%d}\left[c_n\cos(2\pi n t)"
            r"+ d_n\sin(2\pi n t)\right]"
            r"\end{aligned}" % (N_HARM, N_HARM),
            f"x(t) = A0 + Σ  aₙ cos(2πnt) + bₙ sin(2πnt)\n"
            f"y(t) = C0 + Σ  cₙ cos(2πnt) + dₙ sin(2πnt)\n"
            f"n = 1 … {N_HARM}",
            font_size=26,
        )
        sub = Text(
            f"N = {N_HARM} harmonics    0 ≤ t < 1    height-normalized",
            font_size=22,
            color=GREY_B,
        )
        card = VGroup(eq, sub).arrange(DOWN, buff=0.45)
        # Cap the card width so long equations never run off the frame.
        max_card_w = config.frame_width - axes.width - 2.0
        if card.width > max_card_w:
            card.scale_to_fit_width(max_card_w)
        card.next_to(axes, RIGHT, buff=0.6).shift(UP * 0.5)

        # ---- the curve, driven by a single parameter tracker ------------ #
        t_tracker = ValueTracker(0.0)
        step = 1.0 / SAMPLES

        # Reference outline, evaluated from the equation only (never traced
        # from an image). ParametricFunction samples t inclusive of t = 1, and
        # x(1) = x(0) / y(1) = y(0) exactly, so the loop is already closed.
        full_curve = ParametricFunction(
            face_point,
            t_range=[0.0, 1.0, step],
            color=CURVE_COLOR,
            stroke_width=5,
        )
        full_curve.apply_function(lambda p: axes.c2p(p[0], p[1]))

        # The visible stroke. Updating it with pointwise_become_partial in place
        # is what actually clips the curve -- rebuilding a fresh mobject each
        # frame and handing it to always_redraw/become does NOT (become realigns
        # the point arrays and the full outline reappears).
        drawn = full_curve.copy()

        def update_drawn(mob):
            frac = float(np.clip(t_tracker.get_value(), 1e-4, 1.0))
            mob.pointwise_become_partial(full_curve, 0.0, frac)

        update_drawn(drawn)
        drawn.add_updater(update_drawn)

        # ---- optional moving dot (requirement 8) ------------------------ #
        def update_dot(mob):
            x, y = face_xy(t_tracker.get_value())
            mob.move_to(axes.c2p(x, y))

        moving_dot = Dot(radius=0.075, color=YELLOW)
        update_dot(moving_dot)
        moving_dot.add_updater(update_dot)

        # ---- optional live t readout (requirement 9) -------------------- #
        t_label = VGroup(
            tex_or_text("t =", "t =", font_size=34),
            DecimalNumber(
                0.0,
                num_decimal_places=3,
                font_size=34,
                color=YELLOW,
                mob_class=NUMBER_MOB,
            ),
        ).arrange(RIGHT, buff=0.18)
        t_label[1].add_updater(lambda m: m.set_value(t_tracker.get_value()))
        t_label.next_to(card, DOWN, buff=0.8)

        # ---- animate ---------------------------------------------------- #
        self.play(Create(axes), run_time=1.2)
        self.play(Write(title), run_time=1.0)
        self.play(FadeIn(card, shift=UP * 0.2), run_time=1.0)

        self.add(drawn)
        if SHOW_MOVING_DOT:
            self.add(moving_dot)
        if SHOW_T_READOUT:
            self.play(FadeIn(t_label), run_time=0.5)

        # linear rate_func keeps the stroke, the dot and the readout in lockstep
        self.play(t_tracker.animate.set_value(1.0), run_time=DRAW_TIME, rate_func=linear)

        # Drop the updaters and show the exact closed loop.
        drawn.clear_updaters()
        drawn.become(full_curve)
        moving_dot.clear_updaters()
        t_label[1].clear_updaters()
        self.wait(0.6)

        if SHOW_MOVING_DOT:
            self.play(FadeOut(moving_dot), run_time=0.4)
        self.wait(1.6)


# --------------------------------------------------------------------------- #
# Bonus scene: side-by-side harmonic comparison
# --------------------------------------------------------------------------- #
class FourierFaceHarmonics(Scene):
    """Static panel comparing N = 5, 8, 12, 20 reconstructions."""

    def construct(self):
        with open(FACE_JSON, "r") as fh:
            data = json.load(fh)

        title = Text("Effect of the harmonic count N", font_size=32).to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=1.0)

        panels = Group()
        for n_harm in data["harmonics_tested"]:
            co = data["all_harmonic_fits"][str(n_harm)]
            a = np.asarray(co["a"])
            b = np.asarray(co["b"])
            c = np.asarray(co["c"])
            d = np.asarray(co["d"])
            orders = np.arange(1, n_harm + 1, dtype=float)

            def pt(t, a=a, b=b, c=c, d=d, o=orders, co=co):
                ang = 2 * np.pi * o * float(t)
                cs, sn = np.cos(ang), np.sin(ang)
                return np.array(
                    [co["A0"] + a @ cs + b @ sn, co["C0"] + c @ cs + d @ sn, 0.0]
                )

            curve = ParametricFunction(
                pt, t_range=[0, 1, 1 / 600], color=CURVE_COLOR, stroke_width=4
            )
            curve.scale(4.2)  # uniform scale -> aspect ratio preserved
            err = data["harmonic_comparison"][str(n_harm)]["nearest_point_rms"]
            cap = VGroup(
                Text(f"N = {n_harm}", font_size=26),
                Text(f"RMS {err:.2e}", font_size=18, color=GREY_B),
            ).arrange(DOWN, buff=0.12)
            panels.add(VGroup(curve, cap).arrange(DOWN, buff=0.3))

        panels.arrange(RIGHT, buff=0.6).scale_to_fit_width(config.frame_width - 1.2)
        panels.next_to(title, DOWN, buff=0.5)
        for panel in panels:
            self.play(Create(panel), run_time=1.4)
        self.wait(2.0)
