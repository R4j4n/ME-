"""
glasses_reel_manim.py -- 9:16 Instagram story / reel, part 2

    "42 equations. one pair of glasses."

Same flow as art_reel_manim.py, zoomed all the way in: the spectacles are built
one cubic Bezier at a time, and every segment's *actual* control points are on
screen as it is drawn. Then the camera pulls back and the glasses drop into the
face they came from.

Beat sheet
----------
    0.  hook        "42 equations / one pair of glasses"        ~4 s
    1.  form        the general cubic Bezier                    ~5 s
    2.  build       42 segments, one equation each              ~21 s
    3.  numbers     every coefficient, all at once              ~6 s
    4.  reveal      zoom out into the finished face             ~7 s

Text uses Manim's own default font (no font= override anywhere).

Render
------
    manim -pqh glasses_reel_manim.py GlassesReel     # 1080x1920 @ 60fps
    manim -s -qh glasses_reel_manim.py GlassesFrame  # cover still

Vertical geometry is set in this file, so no -r flag is needed.
"""

import json
import os

import numpy as np
from manim import (
    BOLD,
    DOWN,
    LEFT,
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
GHOST = "#2B3A4A"
PEN = "#FFC857"
ANCHOR = "#FF6B9D"
HANDLE = "#FFC857"
DIM = "#8FA3B8"
WHITE_ = "#EAF2FF"

FACE_JSON = os.environ.get("FACE_JSON", "wireframe_curves.json")

# The four pen strokes that make up the spectacles, in the order they get built.
# (Everything else in the drawing is hair, brows, jaw -- see art_reel_manim.py.)
PARTS = [
    (2, "top rim  →  temple"),
    (3, "lower rim  →  bridge"),
    (7, "right lens"),
    (11, "temple tip"),
]

HOOK_TIME = 4.0
FORM_TIME = 5.0
BUILD_TIME = 21.0        # spread across all 42 segments

GLASS_BOX = (3.5, 1.75)  # the closeup box the glasses are fitted into
GLASS_FOCUS = np.array([0.0, 1.45, 0.0])
FACE_HEIGHT = 4.7        # for the pull-back at the end
STROKE_W = 2.0
CLOSE_W = 3.6            # thicker while zoomed in
CARD_LEFT = -1.92        # equations are left-aligned to this x, so they don't
CARD_Y = -1.25           # jitter horizontally as the digits change


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
            "segments": np.asarray(b["control_points"], dtype=float),
        }
        for b in data["strokes"]
    ]
    return data, strokes


DATA, STROKES = load_curves()

_ALL = np.concatenate([s["segments"].reshape(-1, 2) for s in STROKES], axis=0)
_LO, _HI = _ALL.min(axis=0), _ALL.max(axis=0)
_SCALE = FACE_HEIGHT / (_HI[1] - _LO[1])
_CENTER = 0.5 * (_LO + _HI)
LAYOUT_SHIFT = np.array([0.0, 0.05, 0.0])


def to_face(p):
    """Normalized curve coords -> scene coords, whole face in frame."""
    q = (np.asarray(p, dtype=float) - _CENTER) * _SCALE
    return np.array([q[0], q[1], 0.0]) + LAYOUT_SHIFT


def framer(points, box_w, box_h, focus):
    """A second coord map: fit `points` into box_w x box_h centred on `focus`.

    Both maps are affine and share to_face(), so a mobject built with one can be
    Transformed into the other and stay registered.
    """
    xy = np.array([to_face(p)[:2] for p in np.asarray(points).reshape(-1, 2)])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    k = float(min(box_w / span[0], box_h / span[1]))
    mid = np.append(0.5 * (lo + hi), 0.0)

    def xf(p):
        return (to_face(p) - mid) * k + focus
    return xf


GLASS_SEGS = np.concatenate(
    [STROKES[i]["segments"].reshape(-1, 2) for i, _ in PARTS], axis=0)
to_glass = framer(GLASS_SEGS, *GLASS_BOX, GLASS_FOCUS)

N_SEG = sum(len(STROKES[i]["segments"]) for i, _ in PARTS)
# shared endpoints are counted once: 3n+1 per open stroke, 3n per closed one
N_PTS = sum(3 * len(STROKES[i]["segments"]) + (0 if STROKES[i]["closed"] else 1)
            for i, _ in PARTS)


def path_mobject(segments, color, width, opacity=1.0, xf=to_face):
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


# Manim's own default font -- no font= is passed anywhere in this file.
def txt(s, size=28, color=WHITE_, weight=None, **kw):
    return Text(s, font_size=size, color=color, weight=weight or "NORMAL", **kw)


def fit(mob, max_w=FRAME_W - 0.55, max_h=None):
    k = min(max_w / mob.width, (max_h / mob.height) if max_h else 1e9, 1.0)
    if k < 1.0:
        mob.scale(k)
    return mob


def equation_card(k, seg, part_name):
    """One segment's equation, with its real coefficients substituted in."""
    def vec(p):
        return f"({p[0]:+.3f}, {p[1]:+.3f})"

    head = txt(f"segment {k + 1} / {N_SEG}   ·   {part_name}", 19, DIM)
    lines = VGroup(
        txt(f"B(t) =  (1−t)³ {vec(seg[0])}", 20, ANCHOR),
        txt(f"      + 3(1−t)² t {vec(seg[1])}", 20, HANDLE),
        txt(f"      + 3(1−t) t² {vec(seg[2])}", 20, HANDLE),
        txt(f"      + t³ {vec(seg[3])}", 20, ANCHOR),
    ).arrange(DOWN, buff=0.13, aligned_edge=LEFT)
    card = VGroup(head, lines).arrange(DOWN, buff=0.26, aligned_edge=LEFT)
    fit(card, FRAME_W - 0.5)
    card.move_to([CARD_LEFT + card.width / 2, CARD_Y, 0])
    return card


# --------------------------------------------------------------------------- #
class GlassesReel(Scene):
    def construct(self):
        self.beat_hook()
        self.beat_form()
        built = self.beat_build()
        self.beat_numbers()
        self.beat_reveal(built)

    # -- 0. hook ----------------------------------------------------------- #
    def beat_hook(self):
        big = txt(f"{N_SEG} EQUATIONS", 46, INK, BOLD)
        sub = txt("ONE PAIR OF GLASSES", 30, WHITE_, BOLD)
        tail = txt("drawn one at a time", 22, DIM)
        group = VGroup(big, sub, tail).arrange(DOWN, buff=0.34)
        fit(group).move_to([0, 0.3, 0])

        self.play(FadeIn(big, scale=0.9), run_time=0.8)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.6)
        self.play(FadeIn(tail), run_time=0.5)
        self.wait(max(HOOK_TIME - 2.6, 0.1))
        self.play(FadeOut(group), run_time=0.5)

    # -- 1. the general form ----------------------------------------------- #
    def beat_form(self):
        lead = fit(txt("one curve, four points", 26, DIM), FRAME_W - 0.9)
        c = {"P₀": ANCHOR, "P₁": HANDLE, "P₂": HANDLE, "P₃": ANCHOR, "B(t)": INK}
        eq = VGroup(
            txt("B(t) = (1−t)³ P₀  +  3(1−t)² t P₁", 25, WHITE_, t2c=c),
            txt("+  3(1−t) t² P₂  +  t³ P₃", 25, WHITE_, t2c=c),
        ).arrange(DOWN, buff=0.22)
        eq[1].align_to(eq[0], RIGHT)
        fit(eq)
        note = fit(txt("swap in real numbers → a real line", 21, DIM),
                   FRAME_W - 0.9)
        VGroup(lead, eq, note).arrange(DOWN, buff=0.55).move_to([0, 0.4, 0])

        self.play(FadeIn(lead, shift=UP * 0.2), run_time=0.5)
        self.play(FadeIn(eq[0], shift=UP * 0.15), run_time=0.6)
        self.play(FadeIn(eq[1], shift=UP * 0.15), run_time=0.6)
        self.play(FadeIn(note), run_time=0.5)
        self.wait(max(FORM_TIME - 2.7, 0.1))
        self.play(FadeOut(VGroup(lead, eq, note)), run_time=0.5)

    # -- 2. build the glasses, one equation per segment --------------------- #
    def beat_build(self):
        ghost = VGroup(*[
            path_mobject(STROKES[i]["segments"], GHOST, CLOSE_W, 0.35,
                         xf=to_glass)
            for i, _ in PARTS
        ])

        bar_l, bar_r = -1.55, 1.55
        bar_y = SAFE_BOTTOM + 0.05
        bar_bg = Line([bar_l, bar_y, 0], [bar_r, bar_y, 0], color=DIM,
                      stroke_width=3, stroke_opacity=0.25)
        bar = Line([bar_l, bar_y, 0], [bar_l + 0.01, bar_y, 0], color=PEN,
                   stroke_width=3)

        self.play(FadeIn(ghost), FadeIn(bar_bg), run_time=0.7)
        self.add(bar)

        step = BUILD_TIME / N_SEG
        built, card, k = [], None, 0
        for sid, name in PARTS:
            segs = STROKES[sid]["segments"]
            for seg in segs:
                new_card = equation_card(k, seg, name)
                mob = path_mobject([seg], PEN, CLOSE_W + 0.6, xf=to_glass)
                end = Dot(to_glass(seg[3]), radius=0.045, color=PEN)
                frac = (k + 1) / N_SEG

                anims = [
                    FadeIn(new_card, shift=UP * 0.06),
                    Create(mob),
                    bar.animate.put_start_and_end_on(
                        [bar_l, bar_y, 0],
                        [bar_l + (bar_r - bar_l) * frac, bar_y, 0]),
                ]
                if card is not None:
                    anims.append(FadeOut(card, shift=UP * 0.06))
                self.play(*anims, run_time=step, rate_func=linear)
                self.add(end)

                # the fresh segment cools from pen-yellow to ink
                mob.set_stroke(color=INK, width=CLOSE_W)
                end.set_fill(color=INK, opacity=0.55).scale(0.7)
                built.append(mob)
                card = new_card
                k += 1

        tail = VGroup(*[d for d in self.mobjects if isinstance(d, Dot)])
        self.play(FadeOut(card), FadeOut(ghost), FadeOut(bar), FadeOut(bar_bg),
                  FadeOut(tail), run_time=0.6)
        return VGroup(*built)

    # -- 3. every coefficient at once --------------------------------------- #
    def beat_numbers(self):
        pts = []
        for sid, _ in PARTS:
            segs = STROKES[sid]["segments"]
            if not STROKES[sid]["closed"]:
                pts.append(segs[0][0])          # on a closed stroke the last P3
            for s in segs:                      # IS this point -- don't count it
                pts.extend([s[1], s[2], s[3]])  # twice.  P3 of j == P0 of j+1
        rows = [pts[i:i + 4] for i in range(0, len(pts), 4)]
        wall = VGroup(*[
            txt("  ".join(f"({p[0]:+.3f},{p[1]:+.3f})" for p in r), 12,
                INK if ri % 2 else DIM)
            for ri, r in enumerate(rows)
        ]).arrange(DOWN, buff=0.055)
        # sits under the glasses, never on top of them
        fit(wall, FRAME_W - 0.4, 2.5).move_to([0, -0.75, 0])

        cap = fit(VGroup(
            txt(f"{N_SEG} equations   ·   {len(pts)} control points", 22, WHITE_,
                BOLD),
            txt(f"{2 * len(pts)} numbers, and that is the whole object", 19, DIM),
        ).arrange(DOWN, buff=0.16), FRAME_W - 0.6)
        cap.move_to([0, SAFE_BOTTOM + 0.4, 0])

        self.play(FadeIn(wall, scale=0.96), run_time=1.0)
        self.play(FadeIn(cap, shift=UP * 0.1), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(wall), FadeOut(cap), run_time=0.7)

    # -- 4. pull back into the face ----------------------------------------- #
    def beat_reveal(self, built):
        glasses_small = VGroup(*[
            path_mobject([seg], INK, STROKE_W)
            for sid, _ in PARTS for seg in STROKES[sid]["segments"]
        ])
        rest = VGroup(*[
            path_mobject(STROKES[i]["segments"], INK, STROKE_W, 0.0)
            for i in range(len(STROKES)) if i not in [p[0] for p in PARTS]
        ])

        self.play(built.animate.set_stroke(color="#B6ECFF", width=CLOSE_W + 1.4),
                  run_time=0.4, rate_func=there_and_back)
        self.add(rest)
        self.play(
            Transform(built, glasses_small),
            rest.animate.set_stroke(opacity=1.0),
            run_time=1.8, rate_func=rate_functions.ease_in_out_sine,
        )

        cap = fit(txt("…and it was one part of a face", 22, DIM),
                  FRAME_W - 0.7).move_to([0, SAFE_BOTTOM + 0.15, 0])
        self.play(FadeIn(cap), run_time=0.6)
        self.wait(1.2)

        face = VGroup(built, rest)
        end = fit(VGroup(
            txt("THIS IS HOW", 40, DIM, BOLD),
            txt("I CREATE ART", 40, INK, BOLD),
        ).arrange(DOWN, buff=0.2)).move_to([0, SAFE_BOTTOM + 0.7, 0])
        self.play(FadeOut(cap), face.animate.scale(0.88).move_to([0, 1.05, 0]),
                  run_time=0.9)
        self.play(FadeIn(end, scale=0.94), run_time=0.8)
        self.wait(1.8)


class GlassesFrame(Scene):
    """Cover still: the glasses alone, closeup.  manim -s -qh ..."""

    def construct(self):
        glasses = VGroup(*[
            path_mobject(STROKES[i]["segments"], INK, CLOSE_W, xf=to_glass)
            for i, _ in PARTS
        ])
        title = fit(VGroup(
            txt(f"{N_SEG} EQUATIONS", 44, INK, BOLD),
            txt("ONE PAIR OF GLASSES", 28, DIM, BOLD),
        ).arrange(DOWN, buff=0.24)).move_to([0, -0.6, 0])
        self.add(glasses, title)
