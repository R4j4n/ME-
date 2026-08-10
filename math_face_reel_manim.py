"""
math_face_reel_manim.py -- 9:16 Instagram reel, ~60 s

    "Turn the world into equations -- and equations into possibilities."

How a face gets approximated by mathematics, shown honestly: a real digitised
pen stroke (352 sampled points from wireframe_strokes.csv) is fitted with 1, 2,
4, 8 and finally 19 cubic Beziers, and the measured error is on screen at every
step -- 84 px, 50 px, 8.8 px, 1.7 px, 0.33 px. Then the same idea runs over the
whole drawing.

Nothing is traced over a hidden outline: there is no ghost layer anywhere in
this file. Every line appears only as it is computed.

Beat sheet
----------
    0. hook        the line                                  ~6 s
    1. points      one pen stroke, digitised                 ~8 s
    2. one curve   the cubic Bezier, and how wrong it is     ~8 s
    3. refine      2 -> 4 -> 8 -> 19 curves, error collapses ~13 s
    4. face        the same trick, 140 strokes               ~14 s
    5. possible    zoom forever, restyle instantly           ~8 s
    6. payoff      the line again                            ~4 s

Text uses Manim's own default font (no font= override anywhere).

Render
------
    manim -pqh math_face_reel_manim.py MathFaceReel      # 1080x1920 @ 60 fps
    manim -s -qh math_face_reel_manim.py MathFaceFrame   # cover still
"""

import csv
import json
import os

import numpy as np
from manim import (
    BOLD,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Scene,
    Text,
    Transform,
    VGroup,
    VMobject,
    ValueTracker,
    config,
    linear,
    rate_functions,
    there_and_back,
)

# --------------------------------------------------------------------------- #
# Vertical canvas
# --------------------------------------------------------------------------- #
config.pixel_width = 1080
config.pixel_height = 1920
config.frame_height = 8.0
config.frame_width = 8.0 * 1080 / 1920          # 4.5
config.frame_rate = 60
config.background_color = "#05070C"

FRAME_W = config.frame_width
SAFE_TOP = 3.05
SAFE_BOTTOM = -2.85

INK = "#6FD3FF"
PEN = "#FFC857"
ANCHOR = "#FF6B9D"
ERR = "#FF5C7A"
DIM = "#8FA3B8"
WHITE_ = "#EAF2FF"
DATA_PT = "#9AA7B8"

JSON_PATH = os.environ.get("FACE_JSON", "wireframe_curves.json")
CSV_PATH = os.environ.get("FACE_CSV", "wireframe_strokes.csv")

HERO = 2                 # the top rim + temple: 352 points, 19 fitted curves
STAGES = (1, 2, 4, 8)    # ...then the real fit

HOOK_TIME = 5.0
POINTS_TIME = 7.0
ONE_TIME = 7.2
STAGE_TIME = 2.9         # per refinement step
TRACE_TIME = 11.0
FACE_HEIGHT = 4.7
STROKE_W = 2.0
DEMO_W = 3.4


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_curves(path=JSON_PATH):
    with open(path) as fh:
        data = json.load(fh)
    strokes = [
        {
            "closed": bool(b["closed"]),
            "length": float(b["arc_length_normalized"]),
            "segments": np.asarray(b["control_points"], dtype=float),
        }
        for b in data["strokes"]
    ]
    return data, strokes


def load_samples(path=CSV_PATH):
    """The resampled pen-stroke centrelines -- the data the fit was made from."""
    pts = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            pts.setdefault(int(r["stroke_id"]), []).append(
                (float(r["x"]), float(r["y"])))
    return {k: np.asarray(v) for k, v in pts.items()}


DATA, STROKES = load_curves()
SAMPLES = load_samples()
PX = 1.0 / float(DATA["normalization"]["scale_factor"])   # normalized -> pixels

_ALL = np.concatenate([s["segments"].reshape(-1, 2) for s in STROKES], axis=0)
_LO, _HI = _ALL.min(axis=0), _ALL.max(axis=0)
_SCALE = FACE_HEIGHT / (_HI[1] - _LO[1])
_CENTER = 0.5 * (_LO + _HI)
LAYOUT_SHIFT = np.array([0.0, 0.05, 0.0])


def to_face(p):
    q = (np.asarray(p, dtype=float) - _CENTER) * _SCALE
    return np.array([q[0], q[1], 0.0]) + LAYOUT_SHIFT


def framer(points, box_w, box_h, focus):
    xy = np.array([to_face(p)[:2] for p in np.asarray(points).reshape(-1, 2)])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    k = float(min(box_w / span[0], box_h / span[1]))
    mid = np.append(0.5 * (lo + hi), 0.0)

    def xf(p):
        return (to_face(p) - mid) * k + focus
    return xf


# --------------------------------------------------------------------------- #
# Least-squares cubic fitting (Schneider), so every error on screen is measured
# --------------------------------------------------------------------------- #
def chord_param(p):
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(d)])
    return u / u[-1] if u[-1] > 0 else u


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def generate(p, u, t1, t2):
    """One cubic through p[0], p[-1] with tangents t1, t2, least squares.

    The residual subtracts (B0+B1)P0 + (B2+B3)P3, not just B0*P0 + B3*P3:
    because P1 = P0 + a1*t1, the endpoints feed through the middle basis
    functions too. Getting that wrong makes the fitter split instead of fit.
    """
    P0, P3 = p[0], p[-1]
    B0 = (1 - u) ** 3
    B1 = 3 * u * (1 - u) ** 2
    B2 = 3 * u ** 2 * (1 - u)
    B3 = u ** 3
    A1 = B1[:, None] * t1[None, :]
    A2 = B2[:, None] * t2[None, :]
    R = p - ((B0 + B1)[:, None] * P0 + (B2 + B3)[:, None] * P3)
    c11, c12, c22 = (A1 * A1).sum(), (A1 * A2).sum(), (A2 * A2).sum()
    x1, x2 = (A1 * R).sum(), (A2 * R).sum()
    det = c11 * c22 - c12 * c12
    third = np.linalg.norm(P3 - P0) / 3.0
    if abs(det) < 1e-12:
        a1 = a2 = third
    else:
        a1 = (x1 * c22 - c12 * x2) / det
        a2 = (c11 * x2 - x1 * c12) / det
        if a1 < third * 3e-3 or a2 < third * 3e-3:
            a1 = a2 = third
    return np.array([P0, P0 + a1 * t1, P3 + a2 * t2, P3])


def fit_k(p, k):
    """Split the samples into k arcs, fit one cubic to each."""
    idx = np.linspace(0, len(p) - 1, k + 1).astype(int)
    segs = []
    for j in range(k):
        a, b = idx[j], idx[j + 1]
        sub = p[a:b + 1]
        t1 = unit(sub[min(3, len(sub) - 1)] - sub[0])
        t2 = unit(sub[max(-4, -len(sub))] - sub[-1])
        segs.append(generate(sub, chord_param(sub), t1, t2))
    return np.array(segs)


def dense(segs, n=256):
    t = np.linspace(0, 1, n)[:, None]
    return np.vstack([
        (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1
        + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3 for P0, P1, P2, P3 in segs])


def residuals(p, segs):
    """Nearest distance from every sample to the curve -- measured, not assumed."""
    c = dense(segs)
    d = np.linalg.norm(p[:, None, :] - c[None, :, :], axis=2)
    j = d.argmin(axis=1)
    return d[np.arange(len(p)), j], c[j]


# ---- precompute every stage of the approximation --------------------------- #
HERO_PTS = SAMPLES[HERO]
HERO_REAL = STROKES[HERO]["segments"]

FITS = []
for k in STAGES:
    segs = fit_k(HERO_PTS, k)
    d, near = residuals(HERO_PTS, segs)
    FITS.append({"k": k, "segs": segs, "rms": float(np.sqrt((d ** 2).mean())),
                 "max": float(d.max()), "near": near})
_d, _near = residuals(HERO_PTS, HERO_REAL)
FITS.append({"k": len(HERO_REAL), "segs": HERO_REAL,
             "rms": float(np.sqrt((_d ** 2).mean())), "max": float(_d.max()),
             "near": _near})

to_demo = framer(HERO_PTS, DEMO_W, 2.1, np.array([0.0, 1.35, 0.0]))


# --------------------------------------------------------------------------- #
# Mobject helpers  (Manim's default font -- no font= anywhere)
# --------------------------------------------------------------------------- #
def txt(s, size=26, color=WHITE_, weight=None, **kw):
    return Text(s, font_size=size, color=color, weight=weight or "NORMAL", **kw)


def fit_w(mob, max_w=FRAME_W - 0.55, max_h=None):
    k = min(max_w / mob.width, (max_h / mob.height) if max_h else 1e9, 1.0)
    if k < 1.0:
        mob.scale(k)
    return mob


def path_mobject(segments, color, width, opacity=1.0, xf=to_face):
    """A VMobject IS a cubic Bezier path: the fitted control points go straight
    in, so nothing is resampled between the fit and the render."""
    vm = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    vm.set_fill(opacity=0.0)
    for j, seg in enumerate(segments):
        p0, p1, p2, p3 = (xf(p) for p in seg)
        if j == 0:
            vm.start_new_path(p0)
        vm.add_cubic_bezier_curve_to(p1, p2, p3)
    return vm


def whiskers(fit, every=6, xf=to_demo):
    """One short line per sample, from the point to the curve. This is the
    approximation error, drawn at true length -- nothing is exaggerated."""
    g = VGroup()
    for i in range(0, len(HERO_PTS), every):
        a, b = xf(HERO_PTS[i]), xf(fit["near"][i])
        if np.linalg.norm(a - b) < 1e-4:
            continue
        g.add(Line(a, b, color=ERR, stroke_width=2.0, stroke_opacity=0.85))
    return g


def scoreboard(fit):
    n = fit["k"]
    return VGroup(
        txt(f"{n} curve" + ("" if n == 1 else "s"), 30, INK, BOLD),
        txt(f"average error   {fit['rms'] * PX:.2f} px", 23, ERR),
        txt(f"worst error   {fit['max'] * PX:.2f} px", 19, DIM),
    ).arrange(DOWN, buff=0.16)


def live_text(getter, anchor, size=22, color=DIM, aligned_edge=ORIGIN,
              weight=None):
    """Re-renders only when the string changes -- one Pango layout per value,
    not one per frame."""
    holder = VGroup(txt(getter(), size, color, weight)).move_to(anchor,
                                                               aligned_edge)
    state = {"s": getter()}

    def upd(m):
        s = getter()
        if s != state["s"]:
            state["s"] = s
            m.become(VGroup(txt(s, size, color, weight)))
            m.move_to(anchor, aligned_edge)
    holder.add_updater(upd)
    return holder


# --------------------------------------------------------------------------- #
class MathFaceReel(Scene):
    def construct(self):
        self.beat_hook()
        pts = self.beat_points()
        curve, board = self.beat_one_curve(pts)
        curve, board = self.beat_refine(curve, board, pts)
        self.beat_face(curve, board)
        self.beat_possibilities()
        self.beat_payoff()

    # -- 0. the line -------------------------------------------------------- #
    def beat_hook(self):
        l1 = txt("Turn the world", 38, WHITE_, BOLD)
        l2 = txt("into equations —", 38, INK, BOLD)
        l3 = txt("and equations", 32, WHITE_, BOLD)
        l4 = txt("into possibilities.", 32, PEN, BOLD)
        top = VGroup(l1, l2).arrange(DOWN, buff=0.16)
        bot = VGroup(l3, l4).arrange(DOWN, buff=0.16)
        group = VGroup(top, bot).arrange(DOWN, buff=0.75)
        fit_w(group).move_to([0, 0.35, 0])

        self.play(FadeIn(l1, shift=UP * 0.2), run_time=0.6)
        self.play(FadeIn(l2, shift=UP * 0.2), run_time=0.6)
        self.wait(0.5)
        self.play(FadeIn(l3, shift=UP * 0.2), run_time=0.55)
        self.play(FadeIn(l4, shift=UP * 0.2), run_time=0.55)
        self.wait(max(HOOK_TIME - 3.6, 0.1))
        self.play(FadeOut(group), run_time=0.6)

    # -- 1. one pen stroke, digitised --------------------------------------- #
    def beat_points(self):
        cap = fit_w(VGroup(
            txt("one pen stroke,", 26, DIM),
            txt("digitised", 30, WHITE_, BOLD),
        ).arrange(DOWN, buff=0.12), FRAME_W - 0.9).move_to([0, 2.75, 0])

        dots = VGroup(*[
            Dot(to_demo(p), radius=0.022, color=DATA_PT, fill_opacity=0.9)
            for p in HERO_PTS
        ])
        note = fit_w(VGroup(
            txt(f"{len(HERO_PTS)} points", 30, PEN, BOLD),
            txt(f"{2 * len(HERO_PTS)} numbers, and still just dots —", 20, DIM),
            txt("you cannot scale it, bend it, or redraw it.", 20, DIM),
        ).arrange(DOWN, buff=0.16), FRAME_W - 0.6).move_to([0, -1.35, 0])

        self.play(FadeIn(cap), run_time=0.5)
        self.play(FadeIn(dots, lag_ratio=0.004), run_time=2.0)
        self.play(FadeIn(note, shift=UP * 0.12), run_time=0.7)
        self.wait(max(POINTS_TIME - 4.7, 0.1))

        ask = fit_w(txt("what if one equation held all of them?", 24, INK),
                    FRAME_W - 0.5)
        ask.move_to([0, -1.35, 0])
        self.play(FadeOut(note), run_time=0.35)
        self.play(FadeIn(ask), run_time=0.6)
        self.wait(1.1)
        self.play(FadeOut(ask), FadeOut(cap), run_time=0.4)
        return dots

    # -- 2. one curve, and how wrong it is ---------------------------------- #
    def beat_one_curve(self, dots):
        c = {"P₀": ANCHOR, "P₁": PEN, "P₂": PEN, "P₃": ANCHOR, "B(t)": INK}
        eq = VGroup(
            txt("B(t) = (1−t)³ P₀  +  3(1−t)² t P₁", 24, WHITE_, t2c=c),
            txt("+  3(1−t) t² P₂  +  t³ P₃", 24, WHITE_, t2c=c),
        ).arrange(DOWN, buff=0.2)
        eq[1].align_to(eq[0], RIGHT)
        fit_w(eq).move_to([0, 2.7, 0])

        self.play(FadeIn(eq[0]), run_time=0.5)
        self.play(FadeIn(eq[1]), run_time=0.5)
        self.wait(0.6)

        fit0 = FITS[0]
        curve = path_mobject(fit0["segs"], INK, 4.0, xf=to_demo)
        board = scoreboard(fit0)
        fit_w(board, FRAME_W - 0.6).move_to([0, -1.35, 0])
        wh = whiskers(fit0)

        self.play(Create(curve), run_time=1.3)
        self.play(FadeIn(wh, lag_ratio=0.02), run_time=0.9)
        self.play(FadeIn(board, shift=UP * 0.1), run_time=0.6)
        verdict = fit_w(txt("one curve is not enough.", 24, ERR, BOLD),
                        FRAME_W - 0.8).move_to([0, -2.45, 0])
        self.play(FadeIn(verdict), run_time=0.5)
        self.wait(max(ONE_TIME - 6.2, 0.1))
        self.play(FadeOut(verdict), FadeOut(eq), run_time=0.5)
        self.whisk = wh
        return curve, board

    # -- 3. more curves, less error ----------------------------------------- #
    def beat_refine(self, curve, board, dots):
        cap = fit_w(txt("so cut it, and fit again", 24, DIM),
                    FRAME_W - 0.8).move_to([0, 2.7, 0])
        self.play(FadeIn(cap), run_time=0.4)

        for fit in FITS[1:]:
            new_curve = path_mobject(fit["segs"], INK, 4.0, xf=to_demo)
            new_board = fit_w(scoreboard(fit), FRAME_W - 0.6)
            new_board.move_to([0, -1.35, 0])
            new_wh = whiskers(fit)

            self.play(
                Transform(curve, new_curve),
                Transform(self.whisk, new_wh),
                run_time=STAGE_TIME * 0.5,
                rate_func=rate_functions.ease_in_out_sine,
            )
            # hard swap, not a cross-fade: two scoreboards sharing one slot
            # overlap into an unreadable smear
            self.remove(board)
            self.play(FadeIn(new_board, scale=0.96), run_time=0.3)
            board = new_board
            self.wait(max(STAGE_TIME * 0.5 - 0.3, 0.1))

        self.play(FadeOut(self.whisk), FadeOut(dots), FadeOut(cap), run_time=0.6)
        punch = fit_w(VGroup(
            txt(f"{FITS[-1]['rms'] * PX:.2f} px", 34, INK, BOLD),
            txt("thinner than the pen that drew it", 21, DIM),
        ).arrange(DOWN, buff=0.14), FRAME_W - 0.7).move_to([0, -2.35, 0])
        self.play(FadeIn(punch), run_time=0.5)
        self.wait(1.0)
        self.play(FadeOut(punch), run_time=0.4)
        return curve, board

    # -- 4. now do it 140 times --------------------------------------------- #
    def beat_face(self, curve, board):
        cap = fit_w(txt("now do that for every stroke", 24, DIM),
                    FRAME_W - 0.8).move_to([0, SAFE_TOP - 0.15, 0])
        hero_small = path_mobject(HERO_REAL, INK, STROKE_W)
        self.play(FadeOut(board), FadeIn(cap), run_time=0.5)
        self.play(Transform(curve, hero_small), run_time=1.2,
                  rate_func=rate_functions.ease_in_out_sine)
        self.remove(curve)
        self.add(hero_small)

        order = sorted(range(len(STROKES)), key=lambda i: -STROKES[i]["length"])
        lengths = np.array([STROKES[i]["length"] for i in order])
        total = float(lengths.sum())
        starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
        cum = np.cumsum(lengths)
        seg_cum = np.cumsum([len(STROKES[i]["segments"]) for i in order])

        refs = [path_mobject(STROKES[i]["segments"], INK, STROKE_W)
                for i in order]
        drawn = [r.copy() for r in refs]
        u = ValueTracker(0.0)

        def frac_of(k):
            return float(np.clip((u.get_value() * total - starts[k]) / lengths[k],
                                 0.0, 1.0))

        for k, d in enumerate(drawn):
            def upd(m, k=k):
                fr = frac_of(k)
                if fr <= 1e-6:
                    m.set_stroke(opacity=0.0)
                    return
                m.set_stroke(opacity=1.0)
                # in-place partial: always_redraw + become does NOT clip a path
                m.pointwise_become_partial(refs[k], 0.0, max(fr, 1e-4))
            upd(d)
            d.add_updater(upd)

        def current_k():
            return min(int(np.searchsorted(cum, u.get_value() * total, "right")),
                       len(refs) - 1)

        pen = Dot(radius=0.045, color=PEN)
        halo = Dot(radius=0.14, color=PEN, fill_opacity=0.22)
        for m in (pen, halo):
            def up_pen(mob):
                k = current_k()
                mob.move_to(refs[k].point_from_proportion(frac_of(k)))
            up_pen(m)
            m.add_updater(up_pen)

        counter = live_text(
            lambda: f"{int(seg_cum[current_k()])} / "
                    f"{DATA['total_bezier_segments']} curves",
            np.array([0.0, SAFE_BOTTOM + 0.3, 0.0]), 24, PEN, ORIGIN, BOLD)

        self.remove(hero_small)
        self.add(*drawn, halo, pen, counter)
        # linear = constant pen speed in arc length, so the counter stays honest
        self.play(u.animate.set_value(1.0), run_time=TRACE_TIME,
                  rate_func=linear)
        for m in (*drawn, pen, halo, counter):
            m.clear_updaters()
        self.play(FadeOut(halo), FadeOut(pen), FadeOut(cap), FadeOut(counter),
                  run_time=0.6)

        stats = fit_w(VGroup(
            txt(f"{DATA['num_strokes']} strokes", 24, WHITE_, BOLD),
            txt(f"{DATA['total_bezier_segments']} equations", 24, INK, BOLD),
            txt(f"{DATA['total_parameters']:,} numbers", 24, PEN, BOLD),
        ).arrange(DOWN, buff=0.14), FRAME_W - 0.8).move_to([0, SAFE_BOTTOM + 0.5, 0])
        self.face = VGroup(*drawn)
        # remember where the spectacles ended up, whatever we do to the group
        self.lens = drawn[order.index(7)] if 7 in order else drawn[0]
        self.play(self.face.animate.scale(0.88).move_to([0, 1.05, 0]),
                  run_time=0.8)
        self.play(FadeIn(stats, shift=UP * 0.1), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(stats), run_time=0.5)

    # -- 5. ...and equations into possibilities ------------------------------ #
    def beat_possibilities(self):
        face = self.face
        # ask the mobject where it is rather than re-deriving the transform
        eye = self.lens.get_center()
        wide = face.copy()                      # the state to come back to

        cap = fit_w(txt("no pixels. just numbers.", 24, INK, BOLD),
                    FRAME_W - 0.8).move_to([0, SAFE_BOTTOM + 0.45, 0])
        self.play(FadeIn(cap), run_time=0.4)
        # .scale() leaves stroke width alone, so the lines stay hairline however
        # far in you go -- which is the whole point: there is nothing to pixelate
        self.play(face.animate.scale(5.0, about_point=eye)
                  .shift(np.array([0, 0.9, 0]) - eye),
                  run_time=1.5, rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.7)
        self.play(Transform(face, wide), run_time=1.2,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeOut(cap), run_time=0.3)

        cap2 = fit_w(txt("same equations. any style.", 24, PEN, BOLD),
                     FRAME_W - 0.8).move_to([0, SAFE_BOTTOM + 0.45, 0])
        self.play(FadeIn(cap2), run_time=0.4)
        self.play(face.animate.set_color_by_gradient(PEN, ANCHOR, INK),
                  run_time=0.9)
        self.play(face.animate.set_stroke(width=STROKE_W + 2.2),
                  run_time=0.5, rate_func=there_and_back)
        self.play(face.animate.set_stroke(color=INK), run_time=0.7)
        self.play(FadeOut(cap2), run_time=0.4)

    # -- 6. payoff ----------------------------------------------------------- #
    def beat_payoff(self):
        end = fit_w(VGroup(
            txt("Turn the world into equations", 25, WHITE_, BOLD),
            txt("and equations into possibilities.", 25, INK, BOLD),
        ).arrange(DOWN, buff=0.18), FRAME_W - 0.4)
        end.move_to([0, SAFE_BOTTOM + 0.6, 0])
        self.play(FadeIn(end, scale=0.95), run_time=0.9)
        self.wait(1.9)


class MathFaceFrame(Scene):
    """Cover still.  manim -s -qh math_face_reel_manim.py MathFaceFrame"""

    def construct(self):
        order = sorted(range(len(STROKES)), key=lambda i: -STROKES[i]["length"])
        face = VGroup(*[path_mobject(STROKES[i]["segments"], INK, STROKE_W)
                        for i in order]).scale(0.88).move_to([0, 1.05, 0])
        title = fit_w(VGroup(
            txt("Turn the world into equations —", 25, WHITE_, BOLD),
            txt("and equations into possibilities.", 25, INK, BOLD),
        ).arrange(DOWN, buff=0.18), FRAME_W - 0.4)
        title.move_to([0, SAFE_BOTTOM + 0.6, 0])
        self.add(face, title)
