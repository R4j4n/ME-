"""
art_reel_manim.py -- 9:16 Instagram story / reel

    "This is how I create art"

Beat sheet
----------
    0.  hook          "this is how I create art"      ~4 s
    1.  equation      the cubic Bezier formula        ~7 s
    2.  anatomy       4 control points -> one curve   ~13 s   (de Casteljau)
    3.  chain         segments chain into a stroke,
                      then drop into place in the face ~6 s
    4.  trace         all 140 strokes draw themselves ~17 s
    5.  payoff        the numbers, then the tagline   ~7 s

Everything on screen is generated from wireframe_curves.json -- the piecewise
cubic Bezier fit of the pen strokes. No image is ever shown; the face you see at
the end is literally the equations being evaluated.

Render
------
    manim -p art_reel_manim.py ArtReel                # 2160x3840 @ 60  <- post this
    RES=1080 manim -p art_reel_manim.py ArtReel       # 1080x1920, ~4x faster
    RES=540  manim -p art_reel_manim.py ArtReel       # 540x960 preview

The 9:16 geometry is set in this file, so no -r flag is needed -- and manim's
-q flags assume 16:9, so this file overrides them. Use RES instead.

LaTeX IS required -- every formula, control-point label and coordinate is real
typeset math (MathTex); only the prose is Text.
"""

import json
import os

import numpy as np
from manim import (
    BOLD,
    DEGREES,
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    MathTex,
    Scene,
    Text,
    Transform,
    VGroup,
    VMobject,
    ValueTracker,
    config,
    linear,
    rate_functions,
    smooth,
    there_and_back,
)

# --------------------------------------------------------------------------- #
# Vertical canvas
# --------------------------------------------------------------------------- #
# RES is the short edge; the long edge follows at 16/9. 2160 -> 2160x3840 (4K),
# 1080 -> Instagram native, 540 -> quick preview. Only the pixel grid changes:
# strokes, dots and type are all sized in scene units and scale with it, so a
# preview is a faithful, cheaper version of the 4K master -- nothing to retune.
RES = int(os.environ.get("RES", 2160))
config.pixel_width = RES
config.pixel_height = RES * 16 // 9
config.frame_height = 8.0
config.frame_width = 8.0 * 9 / 16               # 4.5 -- portrait frame
config.frame_rate = 60
config.background_color = "#0A0C12"

FRAME_W = config.frame_width
FRAME_H = config.frame_height

# Instagram overlays the top ~10% and bottom ~18%. Keep content inside this.
SAFE_TOP = 3.05
SAFE_BOTTOM = -2.85

FONT = "DejaVu Sans"

# Palette -- graphite ground, seafoam drawing, warm amber for anything live.
# One cool hue carries the artwork; the three accents are reserved for meaning,
# so nothing on screen is coloured just for decoration.
INK = "#5FD9C8"          # seafoam  -- the drawing
INK_ALT = "#6BA6F2"      # azure    -- alternating segments in the chain beat
GLOW = "#C6F7EE"         # pale seafoam -- the payoff flash
GHOST = "#1E3038"        # faint full drawing under the trace
PEN = "#F5B94E"          # amber    -- pen dot and live numbers
ANCHOR = "#FF7A93"       # rose     -- P0, P3: the curve passes through these
HANDLE = "#A78BFA"       # violet   -- P1, P2: the curve only leans toward these
SCAFFOLD = "#6E7F96"     # slate    -- de Casteljau's first lerp level
DIM = "#8A9AB0"          # muted slate -- supporting copy
WHITE_ = "#EDF2FA"       # off-white

FACE_JSON = os.environ.get("FACE_JSON", "wireframe_curves.json")

# Beat durations (seconds). Trim TRACE_TIME first if you need a shorter cut.
HOOK_TIME = 3.6
EQ_TIME = 6.2
SWEEP_TIME = 6.0
CHAIN_TIME = 4.0
TRACE_TIME = 15.0

TARGET_HEIGHT = 4.7      # on-screen height of the face, in Manim units
STROKE_WIDTH = 2.0
GHOST_OPACITY = 0.30

EQ_PARK = 2.98           # where the small formula sits, clear of the head
CLOSEUP_FOCUS = np.array([0.0, 0.5, 0.0])


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
def load_curves(path=FACE_JSON):
    with open(path, "r") as fh:
        data = json.load(fh)
    strokes = [
        {
            "id": int(b["stroke_id"]),
            "closed": bool(b["closed"]),
            "length": float(b["arc_length_normalized"]),
            "segments": np.asarray(b["control_points"], dtype=float),  # (n,4,2)
        }
        for b in data["strokes"]
    ]
    return data, strokes


DATA, STROKES = load_curves()

# One uniform scale for every stroke -- same factor on x and y is what keeps the
# face from shearing. Bounds come from all control points.
_ALL = np.concatenate([s["segments"].reshape(-1, 2) for s in STROKES], axis=0)
_LO, _HI = _ALL.min(axis=0), _ALL.max(axis=0)
SCALE = TARGET_HEIGHT / (_HI[1] - _LO[1])
_CENTER = 0.5 * (_LO + _HI)
LAYOUT_SHIFT = np.array([0.0, 0.05, 0.0])


def to_scene(p):
    """Normalized curve coords -> scene coords (the one shared transform).

    The stroke updaters rewrite points from their reference every frame, so a
    move_to() on a drawn stroke would be silently undone -- the layout offset
    has to live in here instead.
    """
    q = (np.asarray(p, dtype=float) - _CENTER) * SCALE
    return np.array([q[0], q[1], 0.0]) + LAYOUT_SHIFT


def path_mobject(segments, color, width, opacity=1.0, xf=to_scene):
    """A VMobject IS a cubic Bezier path -- feed the fitted control points in
    directly, so nothing is resampled between the fit and the render."""
    vm = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    vm.set_fill(opacity=0.0)
    for j, seg in enumerate(segments):
        p0, p1, p2, p3 = (xf(p) for p in seg)
        if j == 0:
            vm.start_new_path(p0)
        vm.add_cubic_bezier_curve_to(p1, p2, p3)
    return vm


def txt(s, size=28, color=WHITE_, weight=None, **kw):
    """Prose. Sentence case everywhere except the opening title card."""
    return Text(s, font=FONT, font_size=size, color=color,
                weight=weight or "NORMAL", **kw)


def mtx(s, size=30, color=WHITE_, **kw):
    """Mathematics -- always typeset, never faked with Unicode lookalikes."""
    return MathTex(s, font_size=size, color=color, **kw)


def fit(mob, max_w=FRAME_W - 0.55):
    if mob.width > max_w:
        mob.scale(max_w / mob.width)
    return mob


def partial_updater(mob, ref, tracker, lo=0.0, hi=1.0):
    """Clip `mob` to a fraction of `ref` in place.

    always_redraw + become does NOT clip a Bezier path -- become realigns the
    point arrays and the whole curve pops back. pointwise_become_partial on the
    live mobject is the thing that works.
    """
    def upd(m):
        fr = (tracker.get_value() - lo) / (hi - lo)
        fr = float(np.clip(fr, 0.0, 1.0))
        if fr <= 1e-6:
            m.set_stroke(opacity=0.0)
            return
        m.set_stroke(opacity=1.0)
        m.pointwise_become_partial(ref, 0.0, max(fr, 1e-4))
    mob.add_updater(upd)
    upd(mob)
    return mob


def live_text(getter, anchor, size=22, color=DIM, aligned_edge=ORIGIN, weight=None,
              builder=None):
    """A label that re-renders only when its string actually changes.

    Rebuilding it every frame means one Pango layout -- or one LaTeX compile --
    per frame; this keeps it to one per distinct value. `builder` swaps in mtx
    for readouts that are mathematics rather than prose.
    """
    make = builder or (lambda s: txt(s, size, color, weight))
    holder = VGroup(make(getter())).move_to(anchor, aligned_edge)
    state = {"s": getter()}

    def upd(m):
        s = getter()
        if s != state["s"]:
            state["s"] = s
            m.become(VGroup(make(s)))
            m.move_to(anchor, aligned_edge)
    holder.add_updater(upd)
    return holder


# --- the hero: one closed, instantly recognisable stroke -------------------- #
# Stroke 7 is the right spectacle lens: closed, 7 segments, sits centrally in
# the face, so the closeup pays off when it drops back into place.
HERO_STROKE = 7 if len(STROKES) > 7 and STROKES[7]["closed"] else max(
    range(len(STROKES)), key=lambda i: -abs(len(STROKES[i]["segments"]) - 7))


def bendiest_segment(segs):
    """The segment whose handles pull furthest off the chord -- the one that
    actually looks like a Bezier when blown up."""
    best, arg = -1.0, 0
    for j, (p0, p1, p2, p3) in enumerate(segs):
        chord = np.linalg.norm(p3 - p0)
        if chord < 1e-9:
            continue
        n = np.array([-(p3 - p0)[1], (p3 - p0)[0]]) / chord
        bend = (abs(n @ (p1 - p0)) + abs(n @ (p2 - p0))) / chord
        score = bend * min(chord / 0.04, 1.0)
        if score > best:
            best, arg = score, j
    return arg


HERO_SEG = bendiest_segment(STROKES[HERO_STROKE]["segments"])


def framer(points, box_w, box_h, focus=CLOSEUP_FOCUS):
    """Build a normalized-coords -> scene-coords map that fits `points` (in
    normalized curve coords) inside box_w x box_h, centred on `focus`."""
    xy = np.array([to_scene(p)[:2] for p in np.asarray(points).reshape(-1, 2)])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    k = float(min(box_w / span[0], box_h / span[1]))
    mid = np.append(0.5 * (lo + hi), 0.0)

    def xf(p):
        return (to_scene(p) - mid) * k + focus
    return xf


# --------------------------------------------------------------------------- #
class ArtReel(Scene):
    def construct(self):
        self.beat_hook()
        self.beat_equation()
        hero_group = self.beat_anatomy()
        self.beat_chain(hero_group)
        self.beat_trace()
        self.beat_payoff()

    # -- 0. hook ----------------------------------------------------------- #
    def beat_hook(self):
        l1 = txt("THIS IS HOW", 44, DIM, BOLD)
        l2 = txt("I CREATE ART", 44, INK, BOLD)
        title = fit(VGroup(l1, l2).arrange(DOWN, buff=0.22))
        sub = fit(txt("No brush.  Just equations.", 24, DIM), FRAME_W - 1.0)
        sub.next_to(title, DOWN, buff=0.5)
        VGroup(title, sub).move_to([0, 0.3, 0])

        self.play(FadeIn(l1, shift=UP * 0.25), run_time=0.7)
        self.play(FadeIn(l2, shift=UP * 0.25), run_time=0.7)
        self.play(FadeIn(sub), run_time=0.6)
        self.wait(max(HOOK_TIME - 2.6, 0.1))
        self.play(FadeOut(VGroup(title, sub)), run_time=0.5)

    # -- 1. the equation --------------------------------------------------- #
    def beat_equation(self):
        lead = fit(txt("Every line in the drawing", 26, DIM), FRAME_W - 0.9)
        lead2 = fit(txt("is this one equation", 30, WHITE_, BOLD), FRAME_W - 0.9)
        VGroup(lead, lead2).arrange(DOWN, buff=0.18).move_to([0, 2.0, 0])

        # Each term is its own tex arg, so a control point can be tinted without
        # dragging the operators with it -- the legend below then reads as a key
        # to the equation rather than a restatement of it.
        e1 = MathTex(r"B(t)", "=", r"(1-t)^3", "P_0", "+", r"3(1-t)^2 t", "P_1",
                     font_size=34, color=WHITE_)
        e2 = MathTex("+", r"3(1-t) t^2", "P_2", "+", "t^3", "P_3",
                     font_size=34, color=WHITE_)
        e1[0].set_color(INK)
        e1[3].set_color(ANCHOR)
        e1[6].set_color(HANDLE)
        e2[2].set_color(HANDLE)
        e2[5].set_color(ANCHOR)
        eq = VGroup(e1, e2).arrange(DOWN, buff=0.26)
        e2.align_to(e1, RIGHT)
        fit(eq).move_to([0, 0.35, 0])

        def key(sym, words, col):
            s = mtx(sym, 30, col)
            return VGroup(s, txt(words, 21, DIM).next_to(s, RIGHT, buff=0.26))

        legend = VGroup(
            key(r"t \in [0,1]", "sweeps along the curve", PEN),
            key(r"P_0,\, P_3", "points it passes through", ANCHOR),
            key(r"P_1,\, P_2", "points it only leans toward", HANDLE),
        ).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        # symbols are not all the same width -- pull the prose into one column
        gutter = max(row[1].get_left()[0] for row in legend)
        for row in legend:
            row[1].shift(RIGHT * (gutter - row[1].get_left()[0]))
        fit(legend, FRAME_W - 1.0).next_to(eq, DOWN, buff=0.6)

        self.play(FadeIn(lead, shift=UP * 0.2), run_time=0.5)
        self.play(FadeIn(lead2, shift=UP * 0.2), run_time=0.5)
        self.play(FadeIn(e1, shift=UP * 0.15), run_time=0.7)
        self.play(FadeIn(e2, shift=UP * 0.15), run_time=0.7)
        for row in legend:
            self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.35)
        self.wait(max(EQ_TIME - 3.45, 0.1))
        self.eq_small = eq
        # park the formula and clear the copy in one move -- no dead frame
        self.play(
            FadeOut(VGroup(lead, lead2, legend)),
            eq.animate.scale(0.60).move_to([0, EQ_PARK, 0]).set_opacity(0.7),
            run_time=0.8,
        )

    # -- 2. anatomy: four points make one curve ---------------------------- #
    def beat_anatomy(self):
        seg = STROKES[HERO_STROKE]["segments"][HERO_SEG]
        big = framer(seg, 2.2, 1.9)               # one segment, blown up

        P = [big(p) for p in seg]
        centroid = np.mean(P, axis=0)
        cols = [ANCHOR, HANDLE, HANDLE, ANCHOR]
        names = [rf"P_{i}" for i in range(4)]

        dots = VGroup(*[Dot(P[i], radius=0.075, color=cols[i]) for i in range(4)])
        labels = VGroup(*[
            # push each label radially outward, so it never lands on a
            # neighbouring point or on the curve
            mtx(rf"\mathbf{{{names[i]}}}", 34, cols[i]).move_to(
                P[i] + 0.34 * (P[i] - centroid)
                / max(np.linalg.norm(P[i] - centroid), 1e-6))
            for i in range(4)
        ])
        coords = VGroup(*[
            mtx(rf"{names[i]} = ({seg[i][0]:+.3f},\; {seg[i][1]:+.3f})", 24, cols[i])
            for i in range(4)
        ]).arrange(DOWN, buff=0.13, aligned_edge=LEFT)
        fit(coords, FRAME_W - 1.4).move_to([0, -1.95, 0])

        hull = VGroup(*[
            DashedLine(P[i], P[i + 1], color=DIM, stroke_width=1.6,
                       stroke_opacity=0.55, dash_length=0.06)
            for i in range(3)
        ])

        cap = fit(txt("Four points → one curve", 24, DIM),
                  FRAME_W - 1.0).move_to([0, 2.25, 0])

        self.play(FadeIn(cap), run_time=0.4)
        for i in range(4):
            self.play(FadeIn(dots[i], scale=0.4), FadeIn(labels[i]),
                      FadeIn(coords[i], shift=RIGHT * 0.15), run_time=0.35)
        self.play(Create(hull), run_time=0.8)

        # ---- de Casteljau: the equation being evaluated, live ------------- #
        t = ValueTracker(0.0)

        def lerp(a, b, u):
            return a + (b - a) * u

        def level(pts, u):
            return [lerp(pts[i], pts[i + 1], u) for i in range(len(pts) - 1)]

        def bracket():
            u = t.get_value()
            a = level(P, u)
            b = level(a, u)
            c = level(b, u)[0]
            g = VGroup()
            for pair, col, w in ((a, SCAFFOLD, 1.6), (b, PEN, 2.0)):
                for i in range(len(pair) - 1):
                    g.add(Line(pair[i], pair[i + 1], color=col, stroke_width=w,
                               stroke_opacity=0.85))
            for p in a:
                g.add(Dot(p, radius=0.04, color=SCAFFOLD))
            for p in b:
                g.add(Dot(p, radius=0.05, color=PEN))
            g.add(Dot(c, radius=0.085, color=WHITE_))
            return g

        scaffold = bracket()
        scaffold.add_updater(lambda m: m.become(bracket()))

        ref = path_mobject([seg], INK, 4.2, xf=big)
        curve = ref.copy()
        partial_updater(curve, ref, t)

        readout = live_text(lambda: f"{t.get_value():.2f}",
                            np.array([-1.6, 2.25, 0.0]), aligned_edge=LEFT,
                            builder=lambda s: mtx(rf"\mathbf{{t}} = \mathbf{{{s}}}",
                                                  40, PEN))

        self.play(FadeOut(cap), run_time=0.3)
        self.add(scaffold, curve, readout)
        self.play(t.animate.set_value(1.0), run_time=SWEEP_TIME - 1.2,
                  rate_func=linear)
        scaffold.clear_updaters()
        curve.clear_updaters()
        readout.clear_updaters()
        self.play(FadeOut(scaffold), FadeOut(readout), FadeOut(hull),
                  FadeOut(labels), FadeOut(dots), FadeOut(coords), run_time=0.6)

        return curve

    # -- 3. one segment -> one stroke -> its place in the face ------------- #
    def beat_chain(self, curve):
        segs = STROKES[HERO_STROKE]["segments"]
        mid = framer(segs, 2.5, 2.3)              # the whole hero stroke

        cap = fit(txt("Chain them end to end", 24, DIM),
                  FRAME_W - 0.8).move_to([0, 2.25, 0])

        # pull back from one segment to the room the whole stroke needs
        self.play(
            Transform(curve, path_mobject([segs[HERO_SEG]], INK, 3.4, xf=mid)),
            FadeIn(cap), run_time=0.8,
            rate_func=rate_functions.ease_in_out_sine,
        )

        # alternating shades so you can see where one segment ends
        rest = [
            path_mobject([segs[j]], INK if j % 2 else INK_ALT, 3.4, xf=mid)
            for j in range(len(segs)) if j != HERO_SEG
        ]
        step = max((CHAIN_TIME - 2.4) / max(len(rest), 1), 0.12)
        for m in rest:
            self.play(Create(m), run_time=step, rate_func=linear)

        n_seg = len(segs)
        tag = fit(txt(f"{n_seg} segments  →  1 pen stroke", 22, PEN, BOLD),
                  FRAME_W - 0.9).move_to([0, -1.85, 0])
        self.play(FadeIn(tag), run_time=0.4)
        self.wait(0.5)

        # ...and the stroke drops into the place it came from
        whole_big = VGroup(curve, *rest)
        whole_small = path_mobject(segs, INK, STROKE_WIDTH)
        self.play(FadeOut(cap), FadeOut(tag), run_time=0.3)
        self.play(Transform(whole_big, whole_small), run_time=1.3,
                  rate_func=rate_functions.ease_in_out_sine)
        self.remove(whole_big)
        self.add(whole_small)
        self.hero_small = whole_small

    # -- 4. the whole drawing traces itself -------------------------------- #
    def beat_trace(self):
        order = sorted(range(len(STROKES)), key=lambda i: -STROKES[i]["length"])
        lengths = np.array([STROKES[i]["length"] for i in order])
        total = float(lengths.sum())
        starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
        cum = np.cumsum(lengths)
        seg_counts = np.array([len(STROKES[i]["segments"]) for i in order])
        seg_cum = np.cumsum(seg_counts)

        refs = [path_mobject(STROKES[i]["segments"], INK, STROKE_WIDTH)
                for i in order]
        drawn = [r.copy() for r in refs]
        ghost = VGroup(*[path_mobject(STROKES[i]["segments"], GHOST,
                                      STROKE_WIDTH, GHOST_OPACITY)
                         for i in order])

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
            lambda: f"{int(seg_cum[current_k()])} / {DATA['total_bezier_segments']} "
                    f"Bézier segments",
            np.array([0.0, SAFE_BOTTOM + 0.42, 0.0]), 22, PEN, ORIGIN, BOLD)
        cap = fit(txt(f"{DATA['num_strokes']} strokes, longest first", 20, DIM),
                  FRAME_W - 0.8).move_to([0, SAFE_BOTTOM + 0.06, 0])

        self.play(FadeIn(ghost), FadeIn(cap), run_time=0.9)
        if hasattr(self, "hero_small"):
            self.remove(self.hero_small)
        self.add(*drawn, halo, pen, counter)
        # linear: constant pen speed in arc length, so the counter stays honest
        self.play(u.animate.set_value(1.0), run_time=TRACE_TIME, rate_func=linear)

        for m in (*drawn, pen, halo, counter):
            m.clear_updaters()
        self.play(FadeOut(halo), FadeOut(pen), FadeOut(ghost), FadeOut(cap),
                  FadeOut(counter), run_time=0.7)
        self.face = VGroup(*drawn)

    # -- 5. payoff --------------------------------------------------------- #
    def beat_payoff(self):
        face = self.face
        self.play(face.animate.set_stroke(width=STROKE_WIDTH + 1.6, color=GLOW),
                  run_time=0.45, rate_func=there_and_back)
        self.play(FadeOut(self.eq_small), run_time=0.4)

        q = DATA["fit_quality"]
        stats = VGroup(
            txt(f"{DATA['num_strokes']} pen strokes", 24, WHITE_, BOLD),
            txt(f"{DATA['total_bezier_segments']} cubic Bézier curves", 24, INK, BOLD),
            txt(f"{DATA['total_parameters']:,} numbers", 24, PEN, BOLD),
            txt(f"Error {q['bezier_nearest_rms'] * 100:.2f}% of face height", 19, DIM),
        ).arrange(DOWN, buff=0.16)
        fit(stats, FRAME_W - 0.9).move_to([0, SAFE_BOTTOM + 0.75, 0])

        self.play(face.animate.scale(0.88).move_to([0, 1.05, 0]), run_time=0.9)
        for row in stats:
            self.play(FadeIn(row, shift=UP * 0.12), run_time=0.28)
        self.wait(1.5)

        # no end card: the stats clear and the finished drawing recentres, so
        # the last frame is the artwork on its own
        self.play(FadeOut(stats, shift=DOWN * 0.15),
                  face.animate.move_to([0, 0.2, 0]), run_time=0.9)
        self.wait(2.0)


class ArtReelFrame(Scene):
    """Final composition only -- for a thumbnail / cover.  manim -s -qh ..."""

    def construct(self):
        order = sorted(range(len(STROKES)), key=lambda i: -STROKES[i]["length"])
        face = VGroup(*[path_mobject(STROKES[i]["segments"], INK, STROKE_WIDTH)
                        for i in order]).scale(0.88).move_to([0, 0.2, 0])
        self.add(face)
