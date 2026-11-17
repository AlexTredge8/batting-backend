#!/usr/bin/env python3
"""
Procedural cartoon batter silhouettes for the fault cards ("Not this" / "This").

A simple, friendly, gender-neutral cartoon cricketer drawn in flat filled shapes: rounded limbs,
a rounded body, a round helmet with a visor band that shows which way it faces (no face, no skin
tone, no hair), pads, gloves, a bat and, where it matters, a ball, the stumps and the crease. Every
figure is a parameter set (a few joint angles, see Pose), never a hand drawing, so a fault picture
is the correct pose with one or two parameters pushed too far.

Design spec: docs/phase6/annotation_benchmark.md (sections 5 and 6): slate figure, the one body
part that matters filled in the panel's accent colour (vermilion for "Not this", blue for "This"),
far limbs one step lighter, dashed ink cue lines, arrows only on the "This" panel, no numbers or
angle arcs, no text inside the single panels (so the app can mirror them for left-handers and put
the words in as app text). Colour is never the only cue: the pair adds a cross / tick badge and the
words "Not this" / "This".

Views (from docs/phase6/annotation_benchmark.md, section 6):
  front  - as our camera sees it, from behind the bowler; off side on the left for a right-hander;
  side   - as if from point (the off side), the bowler to the right;
  top    - from above (F15 footprints);
  pitch  - a side-on pitch strip (F19 length).

Output is SVG text (no external files, no fonts embedded; system sans-serif). PNGs are made with
cairosvg when it is installed (build step only; the app reads the SVGs).

Build the static assets:  python silhouettes.py  [--out assets/silhouettes] [--no-png]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ASSET_DIR = SCRIPT_DIR / "assets" / "silhouettes"
RUBRIC_PATH = SCRIPT_DIR / "docs" / "phase1" / "rubric_v0_3.json"
SILHOUETTE_VERSION = "silhouettes_v1"

# --- palette (docs/phase6/annotation_benchmark.md, 5.7; Okabe-Ito accents) -------------------
BG = "#F5F4F0"          # card background, warm off-white (white pads still show on it)
BODY = "#3D4450"        # figure (near limbs)
FAR = "#6B7380"         # far limbs, one step lighter
HELMET = "#2B303A"
VISOR = "#C9CED6"
PAD = "#FFFFFF"
FAR_PAD = "#E2E5E9"     # far pad, one step darker than the near pad
BAT = "#C8A165"         # willow
BAT_EDGE = "#7A5F33"
BALL = "#7A1F1F"        # dark cherry, white seam
INK = "#1A1A1A"         # cue lines
GROUND = "#9AA0A6"      # ground line, stumps
PITCH = "#E9E4D6"       # the pitch under the batter
CREASE = "#FFFFFF"
VERMILION = "#D55E00"   # "Not this"
BLUE = "#0072B2"        # "This"
TEXT = "#1D2330"
ACCENT = {"fault": VERMILION, "correct": BLUE}
ACCENT_TINT = {"fault": "#F6D3BC", "correct": "#BFD9EC"}

# --- sizes (viewBox units) --------------------------------------------------------------------
PANEL_W, PANEL_H = 300, 420      # one picture panel (about 156 x 220 dp on a phone)
GROUND_Y = 372                   # ground line under the batter
OUTLINE = 3.6                    # figure outline (about 2 dp at phone size)
CUE_W = 4.0                      # cue line width
DASH = "12 8"

# Body proportions in units of the figure height H (standing, ankle to top of the helmet is about 1)
SHIN, THIGH, TORSO, NECK, HEAD_R = 0.255, 0.25, 0.29, 0.055, 0.078
UPPER_ARM, FOREARM = 0.165, 0.16
ANKLE_H = 0.035
BAT_ABOVE, HANDLE, BAT_LEN = 0.035, 0.12, 0.47   # handle top above the top hand; handle; hands to toe
W_THIGH, W_SHIN, W_PAD, W_UPPER, W_FORE, W_NECK, W_TORSO, W_FOOT = 0.088, 0.07, 0.098, 0.064, 0.056, 0.05, 0.12, 0.052
GLOVE_R = 0.036


def _dir(a: float) -> tuple[float, float]:
    """Unit vector for a direction in degrees: 0 = up, 90 = right, 180 = down (y up)."""
    r = math.radians(a)
    return math.sin(r), math.cos(r)


def _add(p, q, k: float = 1.0):
    return p[0] + k * q[0], p[1] + k * q[1]


def _sub(p, q):
    return p[0] - q[0], p[1] - q[1]


def _len(v) -> float:
    return math.hypot(v[0], v[1])


def _ik(root, target, a: float, b: float, bend: int):
    """Two-bone joint (elbow) from root to target with bone lengths a, b; bend = +1 / -1 picks the side."""
    d = _sub(target, root)
    dist = _len(d)
    if dist < 1e-9:
        return _add(root, (0, -a))
    dist_c = min(dist, a + b - 1e-6)
    cos_a = max(-1.0, min(1.0, (a * a + dist_c * dist_c - b * b) / (2 * a * dist_c)))
    ang = math.acos(cos_a) * bend
    ux, uy = d[0] / dist, d[1] / dist
    rx = ux * math.cos(ang) - uy * math.sin(ang)
    ry = ux * math.sin(ang) + uy * math.cos(ang)
    return root[0] + a * rx, root[1] + a * ry


# --- the pose ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pose:
    """A right-handed batter. Angles in degrees (0 = up, 90 = towards picture right, 180 = down).
    Positions in figure heights. The front ankle is the anchor (on the ground unless lifted)."""
    view: str = "side"                  # "side" (from point, bowler right) or "front" (from behind the bowler)
    ff_x: float = 0.12                  # front ankle x (figure heights from the panel's ground point)
    ff_lift: float = 0.0                # front ankle height above its resting height
    ff_depth: float = 0.0               # front-on: the front foot nearer the camera sits lower in the picture
    f_shin: float = -5.0                # front ankle -> knee
    f_thigh: float = -10.0              # front knee -> hip
    b_thigh: float = 186.0              # back hip -> knee
    b_shin: float = 182.0               # back knee -> ankle
    f_shin_k: float = 1.0               # length factors: a limb pointing at or away from the camera looks shorter
    f_thigh_k: float = 1.0
    b_thigh_k: float = 1.0
    b_shin_k: float = 1.0
    hip_w: float = 0.13
    hip_tilt: float = 0.0               # + = the right-hand hip higher
    torso: float = 0.0                  # pelvis -> neck
    torso_k: float = 1.0                # torso length factor (a bow towards the camera looks shorter)
    sh_w: float = 0.2                   # visible shoulder width
    sh_tilt: float = 0.0
    head: float = 0.0                   # neck -> head centre
    head_tilt: float = 0.0              # helmet roll (+ = clockwise, top to the right)
    face: str = "right"                 # "right", "left" or "front": which way the visor faces
    hands: tuple = (-0.03, -0.30)       # top hand relative to the neck (figure heights, y up)
    fe_bend: int = 1                    # front (top-hand) elbow side
    be_bend: int = -1                   # back (bottom-hand) elbow side
    bat: float = 190.0                  # hands -> toe of the bat
    bat_face: float = 1.0               # blade width factor (edge-on < 1)
    f_foot: tuple = (90.0, 0.10)        # front foot direction and length (heel -> toe)
    b_foot: tuple = (90.0, 0.10)
    side: int = 1                       # +1: the front hip and shoulder are on the picture's right


def joints(p: Pose) -> dict[str, tuple[float, float]]:
    """Forward kinematics: joint positions in figure heights, origin at the panel's ground point, y up."""
    j: dict[str, tuple[float, float]] = {}
    j["front_ankle"] = (p.ff_x, ANKLE_H + p.ff_lift - p.ff_depth)
    j["front_knee"] = _add(j["front_ankle"], _dir(p.f_shin), SHIN * p.f_shin_k)
    j["front_hip"] = _add(j["front_knee"], _dir(p.f_thigh), THIGH * p.f_thigh_k)
    hv = (p.hip_w * math.cos(math.radians(p.hip_tilt)), p.hip_w * math.sin(math.radians(p.hip_tilt)))
    j["pelvis"] = _add(j["front_hip"], hv, -0.5 * p.side)
    j["back_hip"] = _add(j["pelvis"], hv, -0.5 * p.side)
    j["back_knee"] = _add(j["back_hip"], _dir(p.b_thigh), THIGH * p.b_thigh_k)
    j["back_ankle"] = _add(j["back_knee"], _dir(p.b_shin), SHIN * p.b_shin_k)
    j["neck"] = _add(j["pelvis"], _dir(p.torso), TORSO * p.torso_k)
    sv = (p.sh_w * math.cos(math.radians(p.sh_tilt)), p.sh_w * math.sin(math.radians(p.sh_tilt)))
    j["front_shoulder"] = _add(j["neck"], sv, 0.5 * p.side)
    j["back_shoulder"] = _add(j["neck"], sv, -0.5 * p.side)
    j["head"] = _add(j["neck"], _dir(p.head), NECK + HEAD_R)
    j["hands"] = _add(j["neck"], p.hands)
    j["front_elbow"] = _ik(j["front_shoulder"], j["hands"], UPPER_ARM, FOREARM, p.fe_bend)
    bottom = _add(j["hands"], _dir(p.bat), 2 * GLOVE_R * 0.9)
    j["bottom_hand"] = bottom
    j["back_elbow"] = _ik(j["back_shoulder"], bottom, UPPER_ARM, FOREARM, p.be_bend)
    j["bat_top"] = _add(j["hands"], _dir(p.bat), -BAT_ABOVE)
    j["blade_top"] = _add(j["hands"], _dir(p.bat), HANDLE)
    j["bat_toe"] = _add(j["hands"], _dir(p.bat), BAT_LEN)
    j["sweet_spot"] = _add(j["hands"], _dir(p.bat), BAT_LEN - 0.1)
    j["front_toe"] = _add(j["front_ankle"], _dir(p.f_foot[0]), p.f_foot[1])
    j["back_toe"] = _add(j["back_ankle"], _dir(p.b_foot[0]), p.b_foot[1])
    return j


# --- SVG helpers ------------------------------------------------------------------------------

def _f(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else str(int(round(v)))


class Canvas:
    """Collects SVG elements. to_px maps figure units (y up) to panel pixels."""

    def __init__(self, ox: float = 0.0, oy: float = 0.0, scale: float = 250.0, gx: float = PANEL_W / 2,
                 gy: float = GROUND_Y):
        self.parts: list[str] = []
        self.ox, self.oy, self.s, self.gx, self.gy = ox, oy, scale, gx, gy

    def px(self, p) -> tuple[float, float]:
        return self.ox + self.gx + p[0] * self.s, self.oy + self.gy - p[1] * self.s

    def add(self, el: str) -> None:
        self.parts.append(el)

    # primitives in pixel space -----------------------------------------------------------
    def path(self, d: str, fill: str = "none", stroke: str = "none", width: float = 0.0, cap: str = "round",
             join: str = "round", dash: str | None = None, opacity: float | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' opacity="{opacity}"' if opacity is not None else ""
        sw = f' stroke-width="{_f(width)}" stroke-linecap="{cap}" stroke-linejoin="{join}"' if stroke != "none" else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}"{sw}{extra}/>')

    def circle(self, c, r: float, fill: str, stroke: str = "none", width: float = 0.0, dash: str | None = None,
               opacity: float | None = None) -> None:
        extra = f' stroke="{stroke}" stroke-width="{_f(width)}"' if stroke != "none" else ""
        extra += f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' opacity="{opacity}"' if opacity is not None else ""
        self.add(f'<circle cx="{_f(c[0])}" cy="{_f(c[1])}" r="{_f(r)}" fill="{fill}"{extra}/>')

    def line(self, a, b, stroke: str, width: float, dash: str | None = None, cap: str = "round",
             opacity: float | None = None) -> None:
        self.path(f"M{_f(a[0])} {_f(a[1])}L{_f(b[0])} {_f(b[1])}", stroke=stroke, width=width, dash=dash, cap=cap,
                  opacity=opacity)

    def text(self, xy, s: str, size: float, fill: str = TEXT, weight: int = 700, anchor: str = "middle") -> None:
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{_f(xy[0])}" y="{_f(xy[1])}" font-family="Inter, \'Helvetica Neue\', Arial, sans-serif" '
                 f'font-size="{_f(size)}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')

    def svg(self, w: float, h: float, title: str = "", desc: str = "") -> str:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(w)} {_f(h)}" width="{_f(w)}" '
                f'height="{_f(h)}" role="img">')
        meta = ""
        if title:
            meta += f"<title>{_esc(title)}</title>"
        if desc:
            meta += f"<desc>{_esc(desc)}</desc>"
        return head + meta + "".join(self.parts) + "</svg>\n"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _cubic_split(pts, t: float):
    """The part of a cubic Bezier after parameter t (de Casteljau)."""
    p0, p1, p2, p3 = pts

    def lerp(a, b):
        return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
    a, b, c = lerp(p0, p1), lerp(p1, p2), lerp(p2, p3)
    d, e = lerp(a, b), lerp(b, c)
    f = lerp(d, e)
    return f, e, c, p3


def _poly(pts) -> str:
    return "M" + "L".join(f"{_f(x)} {_f(y)}" for x, y in pts)


# --- the figure -------------------------------------------------------------------------------

def draw_batter(cv: Canvas, p: Pose, highlight: set[str] | frozenset = frozenset(), accent: str = BLUE,
                tint: str | None = None, ghost: bool = False) -> dict[str, tuple[float, float]]:
    """Draw the batter; returns the joints in pixels. highlight: parts filled in the accent colour
    ("head", "hands", "front_leg", "front_knee", "back_leg", "back_foot", "front_foot", "torso", "bat")."""
    j = joints(p)
    P = {k: cv.px(v) for k, v in j.items()}
    s = cv.s
    hl = set(highlight)
    tint = tint or accent
    body = FAR if ghost else BODY
    far = FAR

    def colour(part: str, base: str) -> str:
        return accent if part in hl else base

    def limb_group(items):
        """items: (points_px, width_px, fill). Outline pass then fill pass, so one group reads as one shape."""
        for pts, w, _ in items:
            cv.path(_poly(pts), stroke=HELMET, width=w + 2 * OUTLINE)
        for pts, w, fill in items:
            cv.path(_poly(pts), stroke=fill, width=w)

    def leg(which: str, base: str):
        hip, knee, ankle, toe = P[f"{which}_hip"], P[f"{which}_knee"], P[f"{which}_ankle"], P[f"{which}_toe"]
        part = f"{which}_leg"
        limb_group([([hip, knee], W_THIGH * s, colour(part, base))])
        # shoe, then the pad over the shin and knee
        heel = _add(ankle, _sub(ankle, toe), 0.25)
        foot_fill = accent if (f"{which}_foot" in hl) else PAD
        limb_group([([heel, toe], W_FOOT * s, foot_fill)])
        ku = _sub(hip, knee)
        kl = max(_len(ku), 1e-6)
        pad_top = _add(knee, ku, 0.07 * s / kl)
        pad_fill = tint if (part in hl or f"{which}_knee" in hl) else (PAD if which == "front" else FAR_PAD)
        limb_group([([ankle, knee, pad_top], W_PAD * s, pad_fill)])
        # pad canes: two thin lines along the shin
        d = _sub(knee, ankle)
        dl = max(_len(d), 1e-6)
        n = (-d[1] / dl, d[0] / dl)
        for k in (-0.22, 0.22):
            a = _add(_add(ankle, n, k * W_PAD * s), d, 0.12)
            b = _add(_add(knee, n, k * W_PAD * s), d, -0.12)
            cv.line(a, b, "#B9BEC6" if pad_fill in (PAD, FAR_PAD) else HELMET, 1.6,
                    opacity=None if pad_fill in (PAD, FAR_PAD) else 0.6)
        knee_fill = accent if f"{which}_knee" in hl else pad_fill
        cv.circle(knee, 0.048 * s, knee_fill, HELMET, OUTLINE * 0.7)

    def torso():
        quad = [P["back_hip"], P["front_hip"], P["front_shoulder"], P["back_shoulder"]]
        fill = colour("torso", body)
        cv.path(_poly(quad) + "Z", fill=HELMET, stroke=HELMET, width=W_TORSO * s + 2 * OUTLINE)
        cv.path(_poly(quad) + "Z", fill=fill, stroke=fill, width=W_TORSO * s)
        # neck
        limb_group([([P["neck"], _add(P["neck"], _sub(P["head"], P["neck"]), 0.5)], W_NECK * s, fill)])

    def head():
        c, r = P["head"], HEAD_R * s
        fill = colour("head", HELMET)
        cv.circle(c, r, fill, HELMET, OUTLINE * 0.6)
        rot = f' transform="rotate({_f(p.head_tilt)} {_f(c[0])} {_f(c[1])})"'
        if p.face == "front":
            # visor band across the face, a peak above it, grille bars
            band = (f'<rect x="{_f(c[0] - 0.72 * r)}" y="{_f(c[1] - 0.05 * r)}" width="{_f(1.44 * r)}" '
                    f'height="{_f(0.52 * r)}" rx="{_f(0.2 * r)}" fill="{VISOR}"/>')
            peak = (f'<rect x="{_f(c[0] - 1.05 * r)}" y="{_f(c[1] - 0.32 * r)}" width="{_f(2.1 * r)}" '
                    f'height="{_f(0.2 * r)}" rx="{_f(0.1 * r)}" fill="{fill}" stroke="{HELMET}" stroke-width="2"/>')
            bars = "".join(f'<line x1="{_f(c[0] + k * r)}" y1="{_f(c[1] - 0.05 * r)}" x2="{_f(c[0] + k * r)}" '
                           f'y2="{_f(c[1] + 0.47 * r)}" stroke="{HELMET}" stroke-width="1.6"/>' for k in (-0.3, 0.0, 0.3))
            cv.add(f"<g{rot}>{band}{bars}{peak}</g>")
        else:
            sx = 1 if p.face == "right" else -1
            x0 = c[0] + sx * 0.05 * r
            x1 = c[0] + sx * 1.0 * r
            band = (f'<path d="M{_f(x0)} {_f(c[1] - 0.12 * r)}L{_f(x1)} {_f(c[1] - 0.12 * r)}'
                    f'Q{_f(c[0] + sx * 1.02 * r)} {_f(c[1] + 0.55 * r)} {_f(c[0] + sx * 0.55 * r)} {_f(c[1] + 0.78 * r)}'
                    f'L{_f(x0)} {_f(c[1] + 0.55 * r)}Z" fill="{VISOR}"/>')
            peak = (f'<path d="M{_f(c[0] + sx * 0.2 * r)} {_f(c[1] - 0.38 * r)}L{_f(c[0] + sx * 1.42 * r)} '
                    f'{_f(c[1] - 0.2 * r)}L{_f(c[0] + sx * 1.38 * r)} {_f(c[1] - 0.05 * r)}L{_f(c[0] + sx * 0.2 * r)} '
                    f'{_f(c[1] - 0.15 * r)}Z" fill="{fill}" stroke="{HELMET}" stroke-width="2" stroke-linejoin="round"/>')
            bars = "".join(f'<line x1="{_f(c[0] + sx * k * r)}" y1="{_f(c[1] - 0.1 * r)}" x2="{_f(c[0] + sx * k * r)}" '
                           f'y2="{_f(c[1] + 0.62 * r)}" stroke="{HELMET}" stroke-width="1.6"/>' for k in (0.45, 0.8))
            ear = (f'<circle cx="{_f(c[0] - sx * 0.18 * r)}" cy="{_f(c[1] + 0.12 * r)}" r="{_f(0.22 * r)}" '
                   f'fill="none" stroke="{VISOR}" stroke-width="2"/>')
            cv.add(f"<g{rot}>{band}{bars}{peak}{ear}</g>")

    def arm(which: str, base: str):
        sh = P[f"{which}_shoulder"]
        el = P[f"{which}_elbow"]
        hand = P["hands"] if which == "front" else P["bottom_hand"]
        limb_group([([sh, el], W_UPPER * s, base), ([el, hand], W_FORE * s, base)])

    def bat():
        top, bt, toe = P["bat_top"], P["blade_top"], P["bat_toe"]
        d = _sub(toe, bt)
        dl = max(_len(d), 1e-6)
        n = (-d[1] / dl, d[0] / dl)
        w0, w1 = 0.05 * s * p.bat_face, 0.062 * s * p.bat_face
        fill = colour("bat", BAT)
        blade = [_add(bt, n, w0 / 2), _add(toe, n, w1 / 2), _add(toe, n, -w1 / 2), _add(bt, n, -w0 / 2)]
        cv.line(top, bt, HELMET, 0.024 * s + 2)
        cv.line(_add(top, d, 0.02 / dl * s), _add(bt, d, -0.01 / dl * s), "#8E949C", 0.016 * s)
        cv.path(_poly(blade) + "Z", fill=fill, stroke=BAT_EDGE, width=OUTLINE * 0.8)
        if p.bat_face > 0.7:  # the spine of the blade, only when the face is towards the camera
            cv.line(_add(bt, d, 0.12), _add(toe, d, -0.06), BAT_EDGE, 1.4, opacity=0.5)

    def gloves():
        fill = colour("hands", PAD)
        for c in (P["bottom_hand"], P["hands"]):
            cv.circle(c, GLOVE_R * s, fill, HELMET, OUTLINE * 0.75)

    if p.view == "front":
        order = ["back_leg", "back_arm", "torso", "front_leg", "head", "bat", "front_arm", "gloves"]
    else:
        order = ["back_leg", "torso", "front_leg", "head", "back_arm", "bat", "front_arm", "gloves"]
    for part in order:
        if part == "back_leg":
            leg("back", colour("back_leg", far))
        elif part == "front_leg":
            leg("front", colour("front_leg", body))
        elif part == "torso":
            torso()
        elif part == "head":
            head()
        elif part == "back_arm":
            arm("back", far)
        elif part == "front_arm":
            arm("front", body)
        elif part == "bat":
            bat()
        elif part == "gloves":
            gloves()
    return P


# --- scenery and cue marks ------------------------------------------------------------------------

def draw_ground(cv: Canvas, view: str) -> None:
    gy = cv.oy + cv.gy
    x0, x1 = cv.ox, cv.ox + PANEL_W
    if view == "front":
        # the pitch strip running away from the camera, the popping crease across it
        top_y = gy - 70
        cv.path(_poly([(x0 + 30, gy + 48), (x1 - 30, gy + 48), (x1 - 75, top_y), (x0 + 75, top_y)]) + "Z", fill=PITCH)
        cv.line((x0 + 44, gy + 6), (x1 - 44, gy + 6), CREASE, 5, cap="butt")
    else:
        cv.path(f"M{_f(x0)} {_f(gy + 4)}H{_f(x1)}V{_f(gy + 48)}H{_f(x0)}Z", fill=PITCH)
        cv.line((x0, gy + 4), (x1, gy + 4), GROUND, 3, cap="butt")


def stumps_front(cv: Canvas, x_off: float, base_y: float, s: float, pick_off: bool = True) -> float:
    """Three stumps seen from the bowler's end (off stump on the left for a right-hander). x_off is the
    off stump's x in pixels. Returns the off stump's x."""
    h, gap, w = 0.42 * s, 0.065 * s, 0.024 * s
    for i in range(3):
        x = x_off + i * gap
        col = INK if (i == 0 and pick_off) else GROUND
        cv.line((x, base_y), (x, base_y - h), col, w, cap="butt")
    cv.line((x_off - 0.01 * s, base_y - h - 2), (x_off + 2 * gap + 0.01 * s, base_y - h - 2), GROUND, 4)
    return x_off


def stumps_side(cv: Canvas, x: float, base_y: float, s: float) -> None:
    h = 0.42 * s
    cv.line((x, base_y), (x, base_y - h), GROUND, 0.03 * s, cap="butt")
    cv.line((x - 0.03 * s, base_y - h - 3), (x + 0.03 * s, base_y - h - 3), GROUND, 4)


def to_bowler_mark(cv: Canvas) -> None:
    """A small chevron at the right edge, at ground level: the bowler is this way (side-on pictures)."""
    gy = cv.oy + cv.gy + 26
    x = cv.ox + PANEL_W - 18
    cv.path(f"M{_f(x - 12)} {_f(gy - 10)}L{_f(x)} {_f(gy)}L{_f(x - 12)} {_f(gy + 10)}", stroke=GROUND, width=4)
    cv.path(f"M{_f(x - 26)} {_f(gy - 10)}L{_f(x - 14)} {_f(gy)}L{_f(x - 26)} {_f(gy + 10)}", stroke=GROUND, width=4,
            opacity=0.55)


def crease_side(cv: Canvas, x: float) -> None:
    gy = cv.oy + cv.gy
    cv.line((x, gy + 6), (x - 16, gy + 40), CREASE, 5, cap="butt")


def draw_ball(cv: Canvas, c, r: float = 9.0, ghost: bool = False) -> None:
    cv.circle(c, r, BALL, "#3A0E0E", 1.5, opacity=0.45 if ghost else None)
    if not ghost:
        cv.path(f"M{_f(c[0] - r * 0.75)} {_f(c[1] - r * 0.3)}Q{_f(c[0])} {_f(c[1] + r * 0.25)} "
                f"{_f(c[0] + r * 0.75)} {_f(c[1] - r * 0.3)}", stroke="#FFFFFF", width=1.5)


def cue_line(cv: Canvas, a, b) -> None:
    """The dashed ink reference line, with a light halo so it shows over the figure."""
    cv.line(a, b, BG, CUE_W + 4, dash=DASH, cap="butt", opacity=0.9)
    cv.line(a, b, INK, CUE_W, dash=DASH, cap="butt")


def cue_dot(cv: Canvas, c, accent: str, r: float = 8.0) -> None:
    cv.circle(c, r + 2.5, "#FFFFFF")
    cv.circle(c, r, accent)


def arrow(cv: Canvas, a, b, colour: str = BLUE, width: float = 10.0, head: float = 28.0, bend: float = 0.0) -> None:
    """One thick arrow (the "This" panel only). bend: sideways bow of the shaft, as a fraction of its length."""
    d = _sub(b, a)
    L = max(_len(d), 1e-6)
    u = (d[0] / L, d[1] / L)
    n = (-u[1], u[0])
    mid = _add(_add(a, d, 0.5), n, bend * L)
    # the arrowhead points along the curve's end tangent
    t = _sub(b, mid)
    tl = max(_len(t), 1e-6)
    tu = (t[0] / tl, t[1] / tl)
    tn = (-tu[1], tu[0])
    base = _add(b, tu, -head * 0.9)
    shaft_end = _add(b, tu, -head * 0.6)
    d_path = f"M{_f(a[0])} {_f(a[1])}Q{_f(mid[0])} {_f(mid[1])} {_f(shaft_end[0])} {_f(shaft_end[1])}"
    cv.path(d_path, stroke="#FFFFFF", width=width + 5)
    cv.path(d_path, stroke=colour, width=width, cap="round")
    tri = [b, _add(base, tn, head / 2), _add(base, tn, -head / 2)]
    cv.path(_poly(tri) + "Z", fill=colour, stroke="#FFFFFF", width=2.5)


def trail(cv: Canvas, pts, colour: str = VERMILION, width: float = 5.0) -> None:
    """A dotted trail (the "Not this" panel's way of showing a wrong movement; never an arrow)."""
    d = "M" + "L".join(f"{_f(x)} {_f(y)}" for x, y in pts)
    cv.path(d, stroke=colour, width=width, dash="1 11", cap="round")


def smooth_trail(cv: Canvas, a, ctrl, b, colour: str, width: float = 5.0, dotted: bool = True) -> None:
    d = f"M{_f(a[0])} {_f(a[1])}Q{_f(ctrl[0])} {_f(ctrl[1])} {_f(b[0])} {_f(b[1])}"
    cv.path(d, stroke=colour, width=width, dash="1 11" if dotted else None, cap="round")


def pause_mark(cv: Canvas, c, colour: str = VERMILION, size: float = 26.0) -> None:
    cv.circle(c, size * 0.85, "#FFFFFF", colour, 3)
    for dx in (-0.22, 0.22):
        x = c[0] + dx * size
        cv.line((x, c[1] - 0.35 * size), (x, c[1] + 0.35 * size), colour, 0.2 * size, cap="butt")


def badge(cv: Canvas, c, kind: str, r: float = 23.0) -> None:
    """A cross (fault) or a tick (correct) in a filled circle."""
    col = ACCENT[kind]
    cv.circle(c, r, col, "#FFFFFF", 3)
    if kind == "fault":
        k = 0.42 * r
        cv.line((c[0] - k, c[1] - k), (c[0] + k, c[1] + k), "#FFFFFF", 0.22 * r)
        cv.line((c[0] - k, c[1] + k), (c[0] + k, c[1] - k), "#FFFFFF", 0.22 * r)
    else:
        cv.path(f"M{_f(c[0] - 0.48 * r)} {_f(c[1] + 0.02 * r)}L{_f(c[0] - 0.12 * r)} {_f(c[1] + 0.38 * r)}"
                f"L{_f(c[0] + 0.5 * r)} {_f(c[1] - 0.36 * r)}", stroke="#FFFFFF", width=0.22 * r)


def panel_open(cv: Canvas, kind: str, clip_id: str) -> None:
    """Panel background, then a clip to its rounded rectangle (closed by panel_close)."""
    x, y = cv.ox, cv.oy
    cv.add(f'<clipPath id="{clip_id}"><rect x="{_f(x)}" y="{_f(y)}" width="{PANEL_W}" height="{PANEL_H}" rx="18"/>'
           f'</clipPath>')
    cv.add(f'<rect x="{_f(x)}" y="{_f(y)}" width="{PANEL_W}" height="{PANEL_H}" rx="18" fill="{BG}"/>')
    cv.add(f'<g clip-path="url(#{clip_id})">')


def panel_close(cv: Canvas, kind: str) -> None:
    x, y = cv.ox, cv.oy
    cv.add("</g>")
    cv.add(f'<rect x="{_f(x + 1.5)}" y="{_f(y + 1.5)}" width="{_f(PANEL_W - 3)}" height="{_f(PANEL_H - 3)}" rx="17" '
           f'fill="none" stroke="{ACCENT[kind]}" stroke-width="3"/>')


# --- base poses ---------------------------------------------------------------------------------
# Side-on (from point, bowler right). Front-on (from behind the bowler, off side left).

SIDE_SETUP = Pose(view="side", ff_x=0.13, f_shin=-4, f_thigh=-12, b_thigh=187, b_shin=181, hip_w=0.12, torso=4,
                  sh_w=0.19, head=8, face="right", hands=(-0.04, -0.29), bat=196, bat_face=0.75,
                  f_foot=(100, 0.06), b_foot=(95, 0.06), fe_bend=1, be_bend=-1)
SIDE_TOP = replace(SIDE_SETUP, hands=(-0.15, 0.0), bat=-38, fe_bend=-1, be_bend=-1, ff_lift=0.025, f_shin=10,
                   f_thigh=-22, torso=6, head=12, b_thigh=190)
SIDE_CONTACT = Pose(view="side", ff_x=0.25, f_shin=18, f_thigh=-48, b_thigh=228, b_shin=236, hip_w=0.11, torso=32,
                    sh_w=0.15, head=40, head_tilt=18, face="right", hands=(0.08, -0.225), bat=188, bat_face=1.0,
                    f_foot=(105, 0.1), b_foot=(160, 0.1), fe_bend=1, be_bend=-1)
SIDE_FINISH = replace(SIDE_CONTACT, hands=(0.1, 0.13), bat=-58, fe_bend=-1, be_bend=1, torso=24, head=28, head_tilt=10,
                      b_thigh=222, b_shin=238)

FRONT_SETUP = Pose(view="front", ff_x=-0.05, ff_depth=0.02, f_shin=1, f_thigh=4, b_thigh=176, b_shin=181, hip_w=0.1,
                   torso=-3, sh_w=0.13, head=-4, face="front", hands=(-0.12, -0.27), bat=184, bat_face=0.55,
                   f_foot=(-100, 0.05), b_foot=(-95, 0.05), fe_bend=1, be_bend=1, b_shin_k=0.95)
FRONT_TOP = replace(FRONT_SETUP, hands=(0.1, -0.02), bat=25, fe_bend=-1, be_bend=-1, bat_face=0.6)
FRONT_FFD = Pose(view="front", ff_x=-0.08, ff_depth=0.05, f_shin=4, f_shin_k=0.9, f_thigh=14, f_thigh_k=0.75,
                 b_thigh=168, b_thigh_k=0.8, b_shin=180, b_shin_k=0.85, hip_w=0.1, torso=-6, torso_k=0.92, sh_w=0.12,
                 head=-12, face="front", hands=(0.08, 0.0), bat=20, bat_face=0.6, f_foot=(-150, 0.06),
                 b_foot=(-95, 0.05), fe_bend=-1, be_bend=-1)
FRONT_CONTACT = replace(FRONT_FFD, torso=-9, torso_k=0.85, head=-16, head_tilt=-5, hands=(-0.06, -0.24), bat=182,
                        bat_face=1.0, fe_bend=1, be_bend=1)
FRONT_FINISH = replace(FRONT_CONTACT, hands=(0.06, 0.12), bat=-22, fe_bend=-1, be_bend=-1, bat_face=0.6, head=-10,
                       head_tilt=-3, torso=-6, torso_k=0.9)


# --- what each panel shows --------------------------------------------------------------------

@dataclass
class Panel:
    pose: Pose | None
    highlight: frozenset = frozenset()
    draw: Any = None          # callable(cv, P, kind) adding the cue and scenery-specific marks after the figure
    under: Any = None         # callable(cv, kind) drawn before the figure (stumps, lines on the ground)
    scene: str | None = None  # "top" or "pitch": a non-figure picture drawn by draw()


@dataclass
class Spec:
    fault: str
    view: str                 # "front", "side", "top", "pitch"
    moment: str | None        # pose_compare.FAULT_TABLE moment (None for F19)
    drawn_moment: str         # what the picture shows
    caption_fault: str | None
    caption_correct: str
    cue: dict
    fault_panel: Panel | None
    correct_panel: Panel
    alt: str = ""
    extra_panel: Panel | None = None    # F00: a second "This" panel
    caption_extra: str | None = None
    note: str = ""
    top_margin: float = 0.0             # room kept above the figures (px), e.g. for an inset
    shift_x: float = 0.0                # nudge the figures sideways (figure heights)


def _front_stumps(off_x_units: float):
    def under(cv: Canvas, kind: str) -> None:
        x = cv.px((off_x_units, 0))[0]
        stumps_front(cv, x, cv.oy + cv.gy - 26, cv.s)
    return under


def _ball_line_front(x_units: float):
    """The ball's line on the ground, running from near the camera towards the stumps."""
    def under(cv: Canvas, kind: str) -> None:
        near = cv.px((x_units - 0.05, -0.17))
        far = cv.px((x_units + 0.025, 0.3))
        cue_line(cv, near, far)
    return under


def _plumb_from(joint: str, to_ground: bool = True):
    def draw(cv: Canvas, P, kind: str) -> None:
        h = P[joint]
        cue_line(cv, (h[0], h[1] - 0.12 * cv.s), (h[0], cv.oy + cv.gy + 2))
        cue_dot(cv, h, ACCENT[kind], 7)
    return draw


def build_specs() -> dict[str, Spec]:
    S: dict[str, Spec] = {}

    # F01 head still at set-up (front-on): eye-level line; a motion trail of earlier heads on "Not this"
    f01_bad = replace(FRONT_SETUP, head=-14, head_tilt=-24, torso=-6)

    def f01_draw(cv, P, kind):
        h = P["head"]
        r = HEAD_R * cv.s
        if kind == "fault":
            for k, op in ((1.0, 0.18), (2.0, 0.1)):
                cv.circle((h[0] + 0.55 * k * r, h[1] - 0.12 * k * r), r, FAR, opacity=op)
            # the eye line on a slant, following the tilted head
            a = math.radians(-24)
            dx, dy = math.cos(a) * 2.4 * r, math.sin(a) * 2.4 * r
            cue_line(cv, (h[0] - dx, h[1] + 0.1 * r - dy), (h[0] + dx, h[1] + 0.1 * r + dy))
        else:
            cue_line(cv, (h[0] - 2.4 * r, h[1] + 0.1 * r), (h[0] + 2.4 * r, h[1] + 0.1 * r))
    S["F01"] = Spec("F01", "front", "setup", "set-up", "Head moving, eyes tilted", "Head still, eyes level",
                    {"type": "line", "what": "eye-level line across the helmet visor",
                     "fault": "the line slants and earlier head positions trail behind",
                     "correct": "the line is level"},
                    Panel(f01_bad, frozenset({"head"}), f01_draw, _front_stumps(-0.13)),
                    Panel(FRONT_SETUP, frozenset({"head"}), f01_draw, _front_stumps(-0.13)),
                    alt="Two cartoon batters at set-up, seen from the bowler's end. Not this: the head tilted and "
                        "moving, eyes on a slant. This: the head still and the eyes level.")

    # F02 unbalanced stance (front-on): plumb line from the head
    f02_bad = replace(FRONT_SETUP, f_shin=-10, f_thigh=-12, b_thigh=178, torso=38, head=52, head_tilt=18,
                      hands=(-0.14, -0.22), bat=172)
    S["F02"] = Spec("F02", "front", "setup", "set-up", "Leaning, head outside feet", "Weight even, knees soft",
                    {"type": "line", "what": "plumb line straight down from the head",
                     "fault": "the line lands outside the feet", "correct": "the line lands between the feet"},
                    Panel(f02_bad, frozenset({"head"}), _plumb_from("head"), _front_stumps(-0.13)),
                    Panel(FRONT_SETUP, frozenset({"head"}), _plumb_from("head"), _front_stumps(-0.13)),
                    alt="Two cartoon batters at set-up, seen from the bowler's end. Not this: leaning to one side "
                        "with the head outside the feet. This: weight even, head between the feet.")

    # F17 head, hands and feet over off stump (front-on): the line up from off stump
    off_x = -0.115
    f17_good = replace(FRONT_SETUP, ff_x=off_x + 0.005, hands=(-0.05, -0.3), head=-2, torso=-2, bat=182)
    f17_bad = replace(FRONT_SETUP, ff_x=0.07, f_shin=-10, f_thigh=-18, b_thigh=172, torso=-26, head=-34,
                      head_tilt=-18, hands=(-0.25, -0.18), bat=196)

    def f17_draw(cv, P, kind):
        x = cv.px((off_x, 0))[0]
        cue_line(cv, (x, cv.oy + cv.gy - 26), (x, P["head"][1] - 0.16 * cv.s))
        for k in ("head", "hands", "front_ankle"):
            cue_dot(cv, P[k], ACCENT[kind], 6)
    S["F17"] = Spec("F17", "front", "setup", "set-up, at release", "Head, hands, feet spread",
                    "Head, hands, feet over off stump",
                    {"type": "line", "what": "vertical line up from off stump",
                     "fault": "head, hands and feet sit apart, the head beyond off stump",
                     "correct": "head, hands and feet stacked on the line"},
                    Panel(f17_bad, frozenset({"head", "hands", "front_foot"}), f17_draw, _front_stumps(off_x)),
                    Panel(f17_good, frozenset({"head", "hands", "front_foot"}), f17_draw, _front_stumps(off_x)),
                    alt="Two cartoon batters at set-up, seen from the bowler's end, with the stumps behind. "
                        "Not this: head, hands and feet spread apart, the head leaning beyond off stump. "
                        "This: head, hands and feet lined up over off stump.")

    # F13 hands late or static (side-on): a small bowler-at-release inset in both panels
    f13_bad = SIDE_SETUP
    f13_good = replace(SIDE_SETUP, hands=(-0.13, -0.13), bat=-28, fe_bend=-1, be_bend=-1, bat_face=0.8)

    def bowler_inset(cv, P, kind):
        # a small grey bowler at release in a framed box, top right: the ball just leaving the raised hand
        x0, y0, w, h = cv.ox + PANEL_W - 98, cv.oy + 14, 84, 104
        cv.add(f'<rect x="{_f(x0)}" y="{_f(y0)}" width="{w}" height="{h}" rx="12" fill="#FFFFFF" '
               f'stroke="{GROUND}" stroke-width="2.5"/>')
        g = "#8A919B"
        gy = y0 + h - 12
        cv.line((x0 + 8, gy + 2), (x0 + w - 8, gy + 2), GROUND, 2)
        hip = (x0 + 42, y0 + 62)
        neck = (x0 + 38, y0 + 36)
        cv.line(hip, (x0 + 26, gy - 3), g, 9)          # back leg
        cv.line(hip, (x0 + 56, y0 + 76), g, 9)          # front leg, braced
        cv.line((x0 + 56, y0 + 76), (x0 + 60, gy - 3), g, 9)
        cv.line(hip, neck, g, 13)                       # body
        cv.circle((x0 + 36, y0 + 25), 8, g)             # head
        cv.line(neck, (x0 + 54, y0 + 18), g, 7)          # bowling arm, high
        cv.line((x0 + 54, y0 + 18), (x0 + 60, y0 + 10), g, 7)
        cv.line(neck, (x0 + 20, y0 + 50), g, 7)          # front arm pulled down
        draw_ball(cv, (x0 + 66, y0 + 13), 6)
    S["F13"] = Spec("F13", "side", "setup", "at release", "Bat still grounded", "Bat already lifting",
                    {"type": "timing", "what": "small bowler-at-release picture in the corner of both panels",
                     "fault": "the ball is leaving the bowler's hand and the bat is still on the ground",
                     "correct": "the ball is leaving the bowler's hand and the bat is already rising"},
                    Panel(f13_bad, frozenset({"bat"}), bowler_inset, _side_under()),
                    Panel(f13_good, frozenset({"bat"}), bowler_inset, _side_under()),
                    alt="Two cartoon batters seen from the side, each with a small bowler releasing the ball. "
                        "Not this: the bat still resting on the ground. This: the bat already lifting.",
                    top_margin=70, shift_x=-0.12)

    # F03 hands out away from the body in the backlift (front-on)
    f03_good = replace(FRONT_TOP, hands=(0.07, -0.03), bat=12)
    f03_bad = replace(FRONT_TOP, hands=(-0.25, 0.02), bat=-10, fe_bend=1, be_bend=1, bat_face=0.9)

    def f03_draw(cv, P, kind):
        a = P["hands"]
        b = P["back_hip"] if kind == "fault" else P["back_hip"]
        cue_line(cv, a, b)
    S["F03"] = Spec("F03", "front", "top_of_backlift", "top of the backlift", "Hands push out wide",
                    "Hands up, close to body",
                    {"type": "line", "what": "short line from the hands to the back hip",
                     "fault": "a long line: the hands are far out from the body",
                     "correct": "a short line: the hands are up and back, close to the body"},
                    Panel(f03_bad, frozenset({"hands"}), f03_draw, _front_stumps(-0.13)),
                    Panel(f03_good, frozenset({"hands"}), f03_draw, _front_stumps(-0.13)),
                    alt="Two cartoon batters at the top of the backlift, seen from the bowler's end. Not this: the "
                        "hands pushed far out from the body. This: the hands up and back, close to the body.")

    # F19 too short to drive (pitch diagram, side-on)
    S["F19"] = Spec("F19", "pitch", None, "length (shot choice)", "Driving a short ball",
                    "Drive the pitched-up ball",
                    {"type": "zones", "what": "the pitch from the side with a full zone near the batter and a "
                                              "short zone further away",
                     "fault": "the ball bounces in the short zone and arrives high, and the batter drives",
                     "correct": "the ball bounces in the full zone, close to the batter, and the batter drives"},
                    Panel(None, scene="pitch", draw=lambda cv, P, kind: pitch_scene(cv, "fault")),
                    Panel(None, scene="pitch", draw=lambda cv, P, kind: pitch_scene(cv, "correct")),
                    alt="Two pitch diagrams seen from the side. Not this: the ball bounces short and arrives high, "
                        "and the batter still drives. This: the ball bounces full, near the batter, so it is "
                        "there to drive.",
                    note="Shot choice, not body position: the card has no measurement bar.")

    # F15 moves across, not out (from above: footprints)
    S["F15"] = Spec("F15", "top", "front_foot_down", "front foot down", "Steps across the crease",
                    "Steps out to the ball",
                    {"type": "footprints", "what": "footprints from above with the ball's line on the pitch",
                     "fault": "the front foot steps sideways across the crease (dotted trail)",
                     "correct": "the front foot steps out along the ball's line (arrow)"},
                    Panel(None, scene="top", draw=lambda cv, P, kind: top_scene(cv, "fault")),
                    Panel(None, scene="top", draw=lambda cv, P, kind: top_scene(cv, "correct")),
                    alt="Two diagrams from above showing footprints at the crease. Not this: the front foot steps "
                        "sideways across the crease. This: the front foot steps out towards the ball, along its line.")

    # F04 head and shoulder lead (front-on): arrow from the head to the ball's line on "This"
    f04_good = replace(FRONT_FFD, ff_x=-0.2, f_shin=2, f_thigh=8, torso=-12, head=-16, head_tilt=-8)
    f04_bad = replace(f04_good, torso=18, head=24, head_tilt=8, b_thigh=172)
    f04_line = joints(f04_good)["head"][0]   # the ball's line runs under the correct figure's head

    def f04_draw(cv, P, kind):
        x = cv.px((f04_line, 0))[0]
        cue_line(cv, (x, cv.oy + cv.gy + 30), (x, P["head"][1] - 0.14 * cv.s))
        draw_ball(cv, (x, cv.oy + cv.gy + 36), 9)
        if kind == "correct":
            h = P["head"]
            arrow(cv, (h[0] + 0.3 * cv.s, h[1] - 0.14 * cv.s), (h[0] + 0.11 * cv.s, h[1] - 0.03 * cv.s), BLUE, 9, 24, 0.12)
    S["F04"] = Spec("F04", "front", "front_foot_down", "front foot down", "Head and shoulder stay back",
                    "Head and shoulder lead",
                    {"type": "arrow", "what": "the ball's line on the ground; an arrow from the head towards it "
                                              "on the correct panel",
                     "fault": "the foot steps out while the head and front shoulder stay upright, back",
                     "correct": "head and front shoulder lean towards the ball's line; the foot follows"},
                    Panel(f04_bad, frozenset({"head", "torso"}), f04_draw, None),
                    Panel(f04_good, frozenset({"head", "torso"}), f04_draw, None),
                    alt="Two cartoon batters stepping forward, seen from the bowler's end. Not this: the foot steps "
                        "out but the head and front shoulder stay back. This: the head and front shoulder lean "
                        "towards the line of the ball first.")

    # F05 front foot to the line of the ball (front-on)
    ball_x = -0.2
    f05_good = replace(FRONT_FFD, ff_x=-0.17, f_shin=-4, f_thigh=-2, torso=-14, head=-20)
    f05_bad = replace(FRONT_FFD, ff_x=0.1, f_shin=14, f_thigh=24, b_thigh=170, torso=-24, head=-30, head_tilt=-12,
                      sh_w=0.15, hip_w=0.1, f_foot=(-80, 0.07))

    def f05_draw(cv, P, kind):
        draw_ball(cv, cv.px((ball_x - 0.05, -0.12)), 9)
        cue_dot(cv, P["front_ankle"], ACCENT[kind], 7)
    S["F05"] = Spec("F05", "front", "front_foot_down", "front foot down", "Foot lands off the line",
                    "Foot on the ball's line",
                    {"type": "line", "what": "the ball's line dashed on the ground, from the ball towards the stumps",
                     "fault": "the front foot lands well to the leg side of the line",
                     "correct": "the front foot lands on the line, or just inside it"},
                    Panel(f05_bad, frozenset({"front_foot", "front_leg"}), f05_draw, _ball_line_front(ball_x)),
                    Panel(f05_good, frozenset({"front_foot", "front_leg"}), f05_draw, _ball_line_front(ball_x)),
                    alt="Two cartoon batters stepping forward, seen from the bowler's end, with the line of the "
                        "ball on the ground. Not this: the front foot lands well away from the line. This: the front "
                        "foot lands on the line of the ball.")

    # F06 front leg stiff (side-on): highlight the front knee
    f06_bad = replace(SIDE_CONTACT, ff_x=0.3, f_shin=-28, f_thigh=-30, b_thigh=208, b_shin=200, torso=8, head=14,
                      head_tilt=6, hands=(0.06, -0.3), bat=170, b_foot=(100, 0.09))
    S["F06"] = Spec("F06", "side", "contact", "contact", "Stiff front leg, weight back", "Front knee bent, weight on it",
                    {"type": "highlight", "what": "the front leg and knee filled in the accent colour",
                     "fault": "the front leg is locked straight and the body leans back",
                     "correct": "the front knee is bent over the front foot, taking the weight"},
                    Panel(f06_bad, frozenset({"front_leg", "front_knee"}), _side_extras(False), _side_under()),
                    Panel(SIDE_CONTACT, frozenset({"front_leg", "front_knee"}), _side_extras(False), _side_under()),
                    alt="Two cartoon batters at contact, seen from the side. Not this: the front leg locked straight, "
                        "weight back. This: the front knee bent, the weight over it.")

    # F18 back foot stays put (side-on): arrow along the back leg towards the bowler on "This"
    f18_bad = replace(SIDE_CONTACT, b_thigh=210, b_shin=186, torso=20, head=30, head_tilt=14, b_foot=(95, 0.1),
                      f_thigh=-55)

    def f18_draw(cv, P, kind):
        _side_extras(False)(cv, P, kind)
        if kind == "correct":
            a = _add(P["back_ankle"], (-0.02 * cv.s, 0.02 * cv.s))
            arrow(cv, (a[0] - 0.02 * cv.s, a[1] - 0.03 * cv.s), _add(P["back_knee"], (0.1 * cv.s, -0.1 * cv.s)), BLUE,
                  9, 24, 0.12)
    S["F18"] = Spec("F18", "side", "contact", "contact", "Back foot stuck flat", "Push off the back foot",
                    {"type": "arrow", "what": "an arrow along the back leg towards the bowler on the correct panel",
                     "fault": "the back heel is flat and the back knee bent, weight sitting back",
                     "correct": "the back heel is up and the back leg pushes the weight forward"},
                    Panel(f18_bad, frozenset({"back_foot", "back_leg"}), f18_draw, _side_under()),
                    Panel(SIDE_CONTACT, frozenset({"back_foot", "back_leg"}), f18_draw, _side_under()),
                    alt="Two cartoon batters at contact, seen from the side. Not this: the back foot flat and the "
                        "weight back. This: the back heel up, pushing the weight forward.")

    # F16 centre of mass stays back (side-on): plumb line from the head
    f16_bad = replace(SIDE_CONTACT, f_shin=10, f_thigh=-48, torso=-6, head=0, head_tilt=0, b_thigh=212, b_shin=212,
                      hands=(0.16, -0.24), bat=150, b_foot=(100, 0.09))

    def f16_draw(cv, P, kind):
        _side_extras(False)(cv, P, kind)
        _plumb_from("head")(cv, P, kind)
        cue_dot(cv, P["front_ankle"], "#FFFFFF", 0)
    S["F16"] = Spec("F16", "side", "contact", "contact", "Weight stays back", "Head and chest over front foot",
                    {"type": "line", "what": "plumb line straight down from the head",
                     "fault": "the line lands well behind the front foot",
                     "correct": "the line lands on the front foot"},
                    Panel(f16_bad, frozenset({"head", "torso"}), f16_draw, _side_under()),
                    Panel(SIDE_CONTACT, frozenset({"head", "torso"}), f16_draw, _side_under()),
                    alt="Two cartoon batters at contact, seen from the side. Not this: the head and chest stay back, "
                        "well behind the front foot. This: the head and chest are over the front foot.")

    # F07 front shoulder opens early (front-on): the shoulder line
    f07_good = replace(FRONT_CONTACT, hands=(0.05, -0.12), bat=30, bat_face=0.7, fe_bend=-1, be_bend=-1, sh_w=0.06,
                       head=-14)
    f07_bad = replace(f07_good, sh_w=0.34, hip_w=0.16, torso=4, head=0, head_tilt=0, hands=(-0.16, -0.16), bat=-30,
                      fe_bend=1, be_bend=1, bat_face=0.9)

    def f07_draw(cv, P, kind):
        a, b = P["back_shoulder"], P["front_shoulder"]
        d = _sub(b, a)
        L = max(_len(d), 1e-6)
        ext = 0.07 * cv.s / L
        cue_line(cv, _add(a, d, -ext), _add(b, d, ext))
        for k in (a, b):
            cue_dot(cv, k, ACCENT[kind], 6)
    S["F07"] = Spec("F07", "front", "before_contact", "just before contact", "Front shoulder opens early",
                    "Shoulder points down the line",
                    {"type": "line", "what": "the shoulder line as a dashed bar",
                     "fault": "a long bar: the chest has turned to face the bowler",
                     "correct": "a short bar: the front shoulder still points at the bowler"},
                    Panel(f07_bad, frozenset({"torso"}), f07_draw, _front_stumps(-0.13)),
                    Panel(f07_good, frozenset({"torso"}), f07_draw, _front_stumps(-0.13)),
                    alt="Two cartoon batters just before contact, seen from the bowler's end. Not this: the chest "
                        "turned square to the bowler, shoulders wide. This: the front shoulder still pointing down "
                        "the pitch, shoulders narrow.")

    # F08 bat across the line (front-on): ball's line; dotted trail across vs one arrow along the line
    l8 = -0.17
    f08_good = replace(FRONT_CONTACT, ff_x=-0.12, hands=(-0.06, -0.3), bat=180)
    f08_bad = replace(FRONT_CONTACT, ff_x=-0.12, hands=(0.02, -0.3), bat=232, sh_w=0.2, torso=-2, head=-10,
                      bat_face=0.8, be_bend=-1)

    def f08_draw(cv, P, kind):
        draw_ball(cv, cv.px((l8 - 0.02, 0.06)), 9)
        if kind == "fault":
            t = P["bat_toe"]
            trail(cv, [(t[0] + 0.32 * cv.s, t[1] - 0.42 * cv.s), (t[0] + 0.2 * cv.s, t[1] - 0.16 * cv.s),
                       (t[0], t[1]), (t[0] - 0.12 * cv.s, t[1] + 0.05 * cv.s)])
        else:
            b = P["blade_top"]
            x = b[0] - 0.09 * cv.s
            arrow(cv, (x, b[1] - 0.12 * cv.s), (x, P["bat_toe"][1] - 0.02 * cv.s), BLUE, 9, 24)
    S["F08"] = Spec("F08", "front", "contact", "contact", "Bat swings across the line", "Straight bat down the line",
                    {"type": "arrow", "what": "the ball's line on the ground; the bat's path",
                     "fault": "a dotted trail shows the bat sweeping across the line",
                     "correct": "one arrow shows the bat coming straight down the line"},
                    Panel(f08_bad, frozenset({"bat"}), f08_draw, _ball_line_front(l8)),
                    Panel(f08_good, frozenset({"bat"}), f08_draw, _ball_line_front(l8)),
                    alt="Two cartoon batters at contact, seen from the bowler's end. Not this: the bat swinging "
                        "across the line of the ball. This: the bat coming straight down the line.")

    # F09 pause or hitch (side-on): the swing arc
    def f09_draw(cv, P, kind):
        _side_extras(False)(cv, P, kind)
        s_ = cv.s
        toe = P["bat_toe"]
        fa = P["front_ankle"]
        c1 = (toe[0] - 0.12 * s_, toe[1] + 0.3 * s_)
        c2 = (fa[0] - 0.42 * s_, fa[1] + 0.02 * s_)
        end = (fa[0] + 0.12 * s_, fa[1] - 0.08 * s_)
        d = (f"M{_f(toe[0])} {_f(toe[1])}C{_f(c1[0])} {_f(c1[1])} {_f(c2[0])} {_f(c2[1])} {_f(end[0])} {_f(end[1])}")
        if kind == "fault":
            # the same arc, broken just below the top, with a pause mark in the gap
            q = _cubic_split((toe, c1, c2, end), 0.2)
            cv.add(f'<path d="M{_f(q[0][0])} {_f(q[0][1])}C{_f(q[1][0])} {_f(q[1][1])} {_f(q[2][0])} {_f(q[2][1])} '
                   f'{_f(q[3][0])} {_f(q[3][1])}" fill="none" stroke="{VERMILION}" stroke-width="5" '
                   f'stroke-linecap="round" stroke-dasharray="1 11"/>')
            pause_mark(cv, (toe[0] + 0.05 * s_, toe[1] + 0.13 * s_), VERMILION, 24)
        else:
            cv.add(f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="5" stroke-linecap="round" '
                   f'stroke-dasharray="1 11"/>')
            arrow(cv, (end[0] - 0.13 * s_, end[1] + 0.035 * s_), (end[0] + 0.03 * s_, end[1] - 0.01 * s_), BLUE, 8, 22)
    S["F09"] = Spec("F09", "side", "top_of_backlift", "top of the backlift", "Stops at the top", "One smooth swing",
                    {"type": "trail", "what": "the swing arc from the top of the backlift to contact",
                     "fault": "the arc is broken, with a pause mark at the top",
                     "correct": "one unbroken arc from backlift to contact"},
                    Panel(SIDE_TOP, frozenset({"hands"}), f09_draw, _side_under()),
                    Panel(SIDE_TOP, frozenset({"hands"}), f09_draw, _side_under()),
                    alt="Two cartoon batters at the top of the backlift, seen from the side. Not this: the swing "
                        "stops at the top. This: one smooth swing from the top down to the ball.")

    # F10 head falls away at contact (front-on): the line from the head down to the ball
    f10_good = replace(FRONT_CONTACT)
    f10_bad = replace(FRONT_CONTACT, torso=14, head=26, head_tilt=20, hands=(-0.24, -0.24), bat=200,
                      fe_bend=1, be_bend=1, f_thigh=6)

    def f10_draw(cv, P, kind):
        ball = _add(P["sweet_spot"], (-0.04 * cv.s, 0.0)) if kind == "fault" else _add(P["sweet_spot"], (-0.035 * cv.s, 0))
        draw_ball(cv, ball, 9)
        h = P["head"]
        cue_line(cv, (h[0], h[1] - 0.12 * cv.s), (h[0], cv.oy + cv.gy + 2))
        cue_dot(cv, h, ACCENT[kind], 7)
    S["F10"] = Spec("F10", "front", "contact", "contact", "Head falls away", "Head over the ball",
                    {"type": "line", "what": "vertical line from the head down to the ground",
                     "fault": "the line misses the ball: the head has fallen to the leg side and the ball is met "
                              "away from the pad",
                     "correct": "the line runs through the ball, met next to the front pad"},
                    Panel(f10_bad, frozenset({"head"}), f10_draw, None),
                    Panel(f10_good, frozenset({"head"}), f10_draw, None),
                    alt="Two cartoon batters at contact, seen from the bowler's end. Not this: the head falls away "
                        "and the ball is met away from the pad. This: the head is over the ball, next to the front pad.")

    # F11 off balance at the finish (front-on): plumb line from the head
    f11_bad = replace(FRONT_FINISH, torso=-30, head=-42, head_tilt=-26, f_shin=-6, f_thigh=-4, b_thigh=150,
                      b_shin=130, b_foot=(-60, 0.08), hands=(-0.02, 0.14))
    S["F11"] = Spec("F11", "front", "follow_through", "the finish", "Falling out of the shot", "Hold the finish",
                    {"type": "line", "what": "plumb line straight down from the head",
                     "fault": "the line lands outside the front foot: the body topples",
                     "correct": "the line lands on the front foot: balanced, held finish"},
                    Panel(f11_bad, frozenset({"head"}), _plumb_from("head"), _front_stumps(-0.13)),
                    Panel(FRONT_FINISH, frozenset({"head"}), _plumb_from("head"), _front_stumps(-0.13)),
                    alt="Two cartoon batters at the finish, seen from the bowler's end. Not this: toppling to one "
                        "side, the head outside the feet. This: a balanced finish, held, the head over the front knee.")

    # F12 hands stop at contact (side-on): arrow from contact to the finish on "This"
    f12_bad = replace(SIDE_CONTACT, hands=(0.12, -0.24), bat=118, fe_bend=1, be_bend=-1)

    def f12_draw(cv, P, kind):
        _side_extras(False)(cv, P, kind)
        if kind == "fault":
            h = P["hands"]
            cv.line((h[0] + 0.07 * cv.s, h[1] - 0.06 * cv.s), (h[0] + 0.07 * cv.s, h[1] + 0.06 * cv.s), VERMILION, 7)
        else:
            a = cv.px((0.3, 0.38))
            b = _add(P["hands"], (0.06 * cv.s, -0.02 * cv.s))
            arrow(cv, a, b, BLUE, 9, 24, 0.25)
    S["F12"] = Spec("F12", "side", "follow_through", "the finish", "Hands stop at the ball", "Hands go on to target",
                    {"type": "arrow", "what": "an arrow from contact up to the finish, towards the target, on the "
                                              "correct panel",
                     "fault": "the hands stop at waist height and the bat is checked",
                     "correct": "the hands carry on high towards the target"},
                    Panel(f12_bad, frozenset({"hands"}), f12_draw, _side_under()),
                    Panel(SIDE_FINISH, frozenset({"hands"}), f12_draw, _side_under()),
                    alt="Two cartoon batters seen from the side. Not this: the hands stop at the ball. This: the "
                        "hands carry on high towards the target.")

    # F00 keep it grooved (front-on): two "This" panels, contact and finish
    def f00_draw(cv, P, kind):
        if P.get("sweet_spot") and P["bat_toe"][1] > P["hands"][1]:
            draw_ball(cv, _add(P["sweet_spot"], (-0.035 * cv.s, 0)), 9)
    S["F00"] = Spec("F00", "front", "contact", "contact and the finish", None, "Keep repeating this",
                    {"type": "none", "what": "no cue: the clean figure at contact and at the finish",
                     "fault": None, "correct": "head over the ball at contact; a balanced, held finish"},
                    None, Panel(FRONT_CONTACT, frozenset(), f00_draw, _front_stumps(-0.13)),
                    extra_panel=Panel(FRONT_FINISH, frozenset(), None, _front_stumps(-0.13)),
                    caption_extra="Hold the finish",
                    alt="Two cartoon batters seen from the bowler's end, both with a tick: head over the ball at "
                        "contact, and a balanced, held finish.")
    return S


def _side_under():
    def under(cv: Canvas, kind: str) -> None:
        stumps_side(cv, cv.px((-0.36, 0))[0], cv.oy + cv.gy + 2, cv.s)
        crease_side(cv, cv.px((-0.22, 0))[0])
    return under


def _side_extras(ball: bool):
    def draw(cv: Canvas, P, kind: str) -> None:
        to_bowler_mark(cv)
        if ball:
            draw_ball(cv, _add(P["sweet_spot"], (0.03 * cv.s, 0)), 9)
    return draw


def _py(self, y_units: float) -> float:
    return self.oy + self.gy - y_units * self.s


Canvas.py = _py  # type: ignore[attr-defined]


# --- the two non-figure scenes --------------------------------------------------------------------

def pitch_scene(cv: Canvas, kind: str) -> None:
    """F19: the pitch from the side. Batter on the left, the bowler's end off to the right. A full zone
    (blue hatch) near the batter and a short zone (vermilion hatch) further away; the ball's path."""
    x0, gy = cv.ox, cv.oy + 300
    cv.path(f"M{_f(x0)} {_f(gy)}H{_f(x0 + PANEL_W)}V{_f(gy + 60)}H{_f(x0)}Z", fill=PITCH)
    cv.line((x0, gy), (x0 + PANEL_W, gy), GROUND, 3, cap="butt")
    full = (x0 + 70, x0 + 150)
    short = (x0 + 190, x0 + 285)
    cv.add(f'<rect x="{_f(full[0])}" y="{_f(gy + 2)}" width="{_f(full[1] - full[0])}" height="22" fill="{BLUE}" '
           f'opacity="0.35"/>')
    for x in range(int(short[0]), int(short[1]), 12):
        cv.line((x, gy + 24), (x + 12, gy + 2), VERMILION, 3, cap="butt", opacity=0.8)
    cv.add(f'<rect x="{_f(short[0])}" y="{_f(gy + 2)}" width="{_f(short[1] - short[0])}" height="22" fill="none" '
           f'stroke="{VERMILION}" stroke-width="2.5"/>')
    to_bowler_mark(Canvas(ox=cv.ox, oy=cv.oy - 46, gy=gy - cv.oy + 46))
    # a small batter driving, at the left
    small = Canvas(ox=cv.ox, oy=cv.oy, scale=170, gx=70, gy=gy - cv.oy)
    small.parts = cv.parts
    stumps_side(cv, small.px((-0.36, 0))[0], gy, 170)
    pose = replace(SIDE_CONTACT, ff_x=0.12)
    if kind == "fault":
        pose = replace(pose, hands=(0.1, -0.12), bat=120, torso=20, head=26)
    P = draw_batter(small, pose, frozenset())
    contact = _add(P["sweet_spot"], (0.03 * 170, 0))
    if kind == "fault":
        bounce = (x0 + 238, gy - 7)
        start = (x0 + PANEL_W + 10, gy - 150)
        cv.path(f"M{_f(start[0])} {_f(start[1])}Q{_f(x0 + 270)} {_f(gy - 60)} {_f(bounce[0])} {_f(bounce[1])}"
                f"Q{_f(x0 + 160)} {_f(gy - 170)} {_f(contact[0] + 10)} {_f(contact[1] - 4)}",
                stroke=BALL, width=3.5, dash="2 9")
        draw_ball(cv, bounce, 8)
        draw_ball(cv, (contact[0] + 8, contact[1] - 4), 8)
    else:
        bounce = (x0 + 108, gy - 7)
        start = (x0 + PANEL_W + 10, gy - 150)
        cv.path(f"M{_f(start[0])} {_f(start[1])}Q{_f(x0 + 170)} {_f(gy - 90)} {_f(bounce[0])} {_f(bounce[1])}"
                f"Q{_f(x0 + 92)} {_f(gy - 22)} {_f(contact[0] + 4)} {_f(contact[1])}",
                stroke=BALL, width=3.5, dash="2 9")
        draw_ball(cv, bounce, 8)
        draw_ball(cv, (contact[0] + 4, contact[1]), 8)


def top_scene(cv: Canvas, kind: str) -> None:
    """F15: the batter's end from above. Stumps at the top (three dots), the popping crease, the ball's
    line coming down the pitch from the bowler (bottom), the back footprint, and the front footprint."""
    x0, y0 = cv.ox, cv.oy
    cv.add(f'<rect x="{_f(x0 + 40)}" y="{_f(y0 + 30)}" width="{_f(PANEL_W - 80)}" height="{_f(PANEL_H - 60)}" rx="8" '
           f'fill="{PITCH}"/>')
    # popping crease and the stumps (bowling crease) behind it
    cv.line((x0 + 30, y0 + 140), (x0 + PANEL_W - 30, y0 + 140), CREASE, 6, cap="butt")
    cv.line((x0 + 30, y0 + 70), (x0 + PANEL_W - 30, y0 + 70), CREASE, 4, cap="butt")
    sx = x0 + 130
    for i in range(3):
        cv.circle((sx + i * 18, y0 + 70), 6, INK if i == 0 else GROUND)
    # the ball's line, from the bowler (bottom) towards off stump
    cue_line(cv, (sx - 8, y0 + PANEL_H - 40), (sx, y0 + 80))
    draw_ball(cv, (sx - 7, y0 + PANEL_H - 70), 9)

    def footprint(c, ang, col, ghost=False):
        """A shoe print from above: a round heel and a wider forefoot, toe pointing along ang (0 = up)."""
        a = math.radians(ang)
        u = (math.sin(a), -math.cos(a))
        heel = _add(c, u, -17)
        fore = _add(c, u, 9)
        op = 0.45 if ghost else None
        for col_, extra in ((HELMET, OUTLINE), (col, 0.0)):
            cv.line(heel, fore, col_, 20 + 2 * extra, opacity=op)
            cv.circle(heel, 11 + extra, col_, opacity=op)
            cv.circle(fore, 14.5 + extra, col_, opacity=op)
        cv.circle(_add(c, u, 27), 5.5, col, opacity=op)   # toe

    back = (sx + 70, y0 + 118)
    front_start = (sx + 52, y0 + 160)
    footprint(back, -90, FAR)
    if kind == "fault":
        land = (sx - 40, y0 + 175)
        footprint(front_start, -95, BODY, ghost=True)
        trail(cv, [front_start, (sx + 6, y0 + 168), land], VERMILION, 6)
        footprint(land, -100, VERMILION)
    else:
        land = (sx + 6, y0 + 265)
        footprint(front_start, -95, BODY, ghost=True)
        footprint(land, -150, BLUE)
        arrow(cv, (front_start[0] - 6, front_start[1] + 26), (land[0] + 10, land[1] - 34), BLUE, 9, 24, -0.1)


# --- composing panels, pairs and the contact sheet -------------------------------------------------

def render_panel(cv: Canvas, panel: Panel, kind: str, view: str, clip_id: str = "clip") -> None:
    """One picture panel at (cv.ox, cv.oy): background, scenery, figure, cue, frame. No text."""
    panel_open(cv, kind, clip_id)
    if panel.scene == "pitch":
        pitch_scene(cv, kind)
    elif panel.scene == "top":
        top_scene(cv, kind)
    else:
        draw_ground(cv, view)
        if panel.under:
            panel.under(cv, kind)
        P = draw_batter(cv, panel.pose, panel.highlight, ACCENT[kind], ACCENT_TINT[kind])
        if panel.draw:
            panel.draw(cv, P, kind)
        elif view == "side":
            to_bowler_mark(cv)
    panel_close(cv, kind)


GROUND_AT = {"front": 352.0, "side": 368.0}
MAX_SCALE = {"front": 300.0, "side": 290.0}


def _pose_extent(p: Pose) -> tuple[float, float, float, float]:
    j = joints(p)
    xs = [v[0] for v in j.values()] + [j["head"][0] - HEAD_R * 1.4, j["head"][0] + HEAD_R * 1.4]
    ys = [v[1] for v in j.values()] + [j["head"][1] + HEAD_R * 1.2, 0.0]
    return min(xs) - 0.05, max(xs) + 0.05, min(ys), max(ys) + 0.02


def fit(spec: "Spec") -> tuple[float, float, float]:
    """(scale, ground x, ground y) shared by every panel of a pair: the figures as large as fits,
    on the same ground point in both panels."""
    if spec.view not in GROUND_AT:
        return 1.0, PANEL_W / 2, GROUND_Y
    poses = [pn.pose for pn in (spec.fault_panel, spec.correct_panel, spec.extra_panel) if pn is not None and pn.pose]
    ext = [_pose_extent(p) for p in poses]
    x0, x1 = min(e[0] for e in ext), max(e[1] for e in ext)
    y1 = max(e[3] for e in ext)
    gy = GROUND_AT[spec.view]
    top_room = gy - 24 - spec.top_margin
    s = min(MAX_SCALE[spec.view], (PANEL_W - 24) / max(x1 - x0, 1e-3), top_room / max(y1, 1e-3))
    gx = PANEL_W / 2 - (x0 + x1) / 2 * s + spec.shift_x * s
    return s, gx, gy


def _canvas(ox: float, oy: float, spec: "Spec") -> Canvas:
    s, gx, gy = fit(spec)
    return Canvas(ox=ox, oy=oy, scale=s, gx=gx, gy=gy)


def panel_svg(spec: Spec, kind: str, which: str = "main") -> str:
    """A single text-free panel (mirror-safe), for the app to place with its own labels."""
    panel = {"fault": spec.fault_panel, "correct": spec.correct_panel}[kind] if which == "main" else spec.extra_panel
    cv = _canvas(0, 0, spec)
    render_panel(cv, panel, kind, spec.view, f"clip-{spec.fault}-{kind}-{which}")
    cap = {"fault": spec.caption_fault, "correct": spec.caption_correct}[kind] if which == "main" else spec.caption_extra
    return cv.svg(PANEL_W, PANEL_H, title=f"{'Not this' if kind == 'fault' else 'This'}: {cap}", desc=spec.alt)


def wrap(text: str, size: float, width: float) -> list[str]:
    """Greedy word wrap with an average bold glyph width (Inter / Helvetica bold, about 0.56 em)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if len(trial) * 0.56 * size > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


PAIR_GAP = 24
PAIR_PAD = 14
LABEL_H = 58
CAPTION_H = 96
PAIR_W = 2 * PANEL_W + PAIR_GAP + 2 * PAIR_PAD
PAIR_H = LABEL_H + PANEL_H + CAPTION_H + PAIR_PAD


def pair_group(cv: Canvas, spec: Spec, ox: float, oy: float) -> None:
    """Badge + label row, the two panels, the captions under each, drawn at (ox, oy) on cv."""
    panels = []
    if spec.fault_panel is not None:
        panels.append(("fault", spec.fault_panel, spec.caption_fault, "Not this"))
        panels.append(("correct", spec.correct_panel, spec.caption_correct, "This"))
    else:
        panels.append(("correct", spec.correct_panel, spec.caption_correct, "This"))
        panels.append(("correct", spec.extra_panel, spec.caption_extra, "This"))
    for i, (kind, panel, caption, label) in enumerate(panels):
        px0 = ox + PAIR_PAD + i * (PANEL_W + PAIR_GAP)
        oy0 = oy + PAIR_PAD
        badge(cv, (px0 + 26, oy0 + 25), kind)
        cv.text((px0 + 58, oy0 + 36), label, 30, ACCENT[kind], 800, "start")
        sub = _canvas(px0, oy0 + LABEL_H, spec)
        sub.parts = cv.parts
        render_panel(sub, panel, kind, spec.view, f"clip-pair-{spec.fault}-{i}")
        lines = wrap(caption or "", 28, PANEL_W - 12)[:2]
        y = oy0 + LABEL_H + PANEL_H + 36
        for ln in lines:
            cv.text((px0 + PANEL_W / 2, y), ln, 28, TEXT, 700)
            y += 34


def pair_svg(spec: Spec) -> str:
    cv = Canvas()
    cv.add(f'<rect width="{PAIR_W}" height="{PAIR_H}" rx="22" fill="#FFFFFF"/>')
    pair_group(cv, spec, 0, 0)
    title = (f"Not this: {spec.caption_fault}. This: {spec.caption_correct}." if spec.caption_fault
             else f"This: {spec.caption_correct}. {spec.caption_extra}.")
    return cv.svg(PAIR_W, PAIR_H, title=title, desc=spec.alt)


def contact_sheet_svg(specs: dict[str, Spec], order: list[str], texts: dict[str, str]) -> str:
    cols, scale = 4, 0.5
    cell_w, cell_h = PAIR_W * scale + 26, PAIR_H * scale + 78
    rows = math.ceil(len(order) / cols)
    W, Hh = cols * cell_w + 34, rows * cell_h + 120
    cv = Canvas()
    cv.add(f'<rect width="{_f(W)}" height="{_f(Hh)}" fill="#E9ECEF"/>')
    cv.text((30, 52), f"Batting IQ fault pictures, {SILHOUETTE_VERSION}: draft for Alex to review", 34, TEXT, 800, "start")
    cv.text((30, 88), "Left: Not this (cross, vermilion). Right: This (tick, blue). Front-on = from behind the bowler "
                      "(our camera). Side-on = from point, bowler to the right.", 19, "#4A5260", 500, "start")
    for i, fid in enumerate(order):
        r, c = divmod(i, cols)
        x, y = 30 + c * cell_w, 116 + r * cell_h
        sp = specs[fid]
        view_word = {"front": "front-on", "side": "side-on", "top": "from above", "pitch": "pitch, side-on"}[sp.view]
        cv.text((x + 4, y + 24), f"{fid}: {view_word}, {sp.drawn_moment}", 18, TEXT, 800, "start")
        name_lines = wrap(texts.get(fid, ""), 15, PAIR_W * scale - 10)
        if len(name_lines) > 2:
            name_lines = name_lines[:2]
            name_lines[1] = name_lines[1].rstrip(".,;:") + "..."
        for k, ln in enumerate(name_lines):
            cv.text((x + 4, y + 46 + 18 * k), ln, 15, "#4A5260", 500, "start")
        inner = Canvas()
        inner.add(f'<rect width="{PAIR_W}" height="{PAIR_H}" rx="22" fill="#FFFFFF"/>')
        pair_group(inner, sp, 0, 0)
        cv.add(f'<g transform="translate({_f(x)} {_f(y + 76)}) scale({scale})">' + "".join(inner.parts) + "</g>")
    return cv.svg(W, Hh, title="Batting IQ fault pictures, contact sheet")


# --- build ------------------------------------------------------------------------------------------

def fault_order() -> list[str]:
    try:
        r = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
        ids = [f["id"] for f in r.get("faults_in_chain_order", [])]
    except (OSError, ValueError):
        ids = []
    return (ids or ["F01", "F02", "F17", "F13", "F03", "F19", "F15", "F04", "F05", "F06", "F18", "F16", "F07", "F08",
                    "F09", "F10", "F11", "F12"]) + ["F00"]


def fault_texts() -> dict[str, str]:
    try:
        r = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"F00": "Nothing major to fix: keep it grooved"}
    out = {f["id"]: f["text"] for f in r.get("faults_in_chain_order", [])}
    out["F00"] = "Nothing major to fix: keep it grooved"
    return out


def _png(svg_text: str, path: Path, width: int | None = None, colours: int = 96) -> bool:
    """SVG -> PNG with cairosvg (build step only), then a palette PNG (flat colours, so it stays small)."""
    try:
        import cairosvg  # build-time only
    except Exception:  # noqa: BLE001 - not installed or no cairo library
        return False
    import io
    kw = {"output_width": width} if width else {}
    raw = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), **kw)
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        im.quantize(colors=colours, method=Image.Quantize.FASTOCTREE).save(path, "PNG", optimize=True)
    except Exception:  # noqa: BLE001 - Pillow missing: keep the full-colour PNG
        path.write_bytes(raw)
    return True


PNG_WIDTH = {"pair": 1000, "panel": 480}   # about 3x the size each is shown at on a phone


def build(out_dir: Path = ASSET_DIR, png: bool = True, sheet_copy: Path | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs()
    order = fault_order()
    texts = fault_texts()
    index: dict[str, Any] = {}
    made_png = False
    for fid in order:
        sp = specs[fid]
        files: dict[str, str | None] = {}
        if sp.fault_panel is not None:
            files["fault"] = f"{fid}_fault.svg"
            (out_dir / files["fault"]).write_text(panel_svg(sp, "fault"), encoding="utf-8")
        else:
            files["fault"] = None
        files["correct"] = f"{fid}_correct.svg"
        (out_dir / files["correct"]).write_text(panel_svg(sp, "correct"), encoding="utf-8")
        if sp.extra_panel is not None:
            files["correct_finish"] = f"{fid}_correct_finish.svg"
            (out_dir / files["correct_finish"]).write_text(panel_svg(sp, "correct", "extra"), encoding="utf-8")
        files["pair"] = f"{fid}_pair.svg"
        pair = pair_svg(sp)
        (out_dir / files["pair"]).write_text(pair, encoding="utf-8")
        if png:
            for key in [k for k in list(files) if files[k]]:
                name = files[key]
                ok = _png((out_dir / name).read_text(encoding="utf-8"), out_dir / name.replace(".svg", ".png"),
                          width=PNG_WIDTH["pair" if key == "pair" else "panel"])
                if ok:
                    files[f"{key}_png"] = name.replace(".svg", ".png")
                    made_png = True
        index[fid] = {
            "view": {"front": "front_on", "side": "side_on", "top": "top_down", "pitch": "pitch_side"}[sp.view],
            "moment": sp.moment, "drawn_moment": sp.drawn_moment,
            "caption_fault": sp.caption_fault, "caption_correct": sp.caption_correct,
            **({"caption_correct_finish": sp.caption_extra} if sp.caption_extra else {}),
            "label_fault": "Not this" if sp.caption_fault else None, "label_correct": "This",
            "cue": sp.cue, "alt": sp.alt, "files": files,
            **({"note": sp.note} if sp.note else {}),
        }
    index["F14"] = {"same_as": "F07", "note": "F14 was retired in rubric v0.3 and is now part of F07."}
    meta = {"version": SILHOUETTE_VERSION, "status": "DRAFT for Alex to sign off (docs/phase6/annotation_benchmark.md)",
            "handedness": "drawn for a right-hander; mirror the single panels (scaleX(-1)) for a left-hander; the "
                          "*_pair files carry text and must not be mirrored",
            "views": {"front_on": "from behind the bowler (our camera); off side on the left for a right-hander",
                      "side_on": "from point; the bowler to the right (chevron at the ground's right edge)",
                      "top_down": "from above", "pitch_side": "the pitch from the side"},
            "panel_size": [PANEL_W, PANEL_H], "pair_size": [PAIR_W, PAIR_H],
            "palette": {"not_this": VERMILION, "this": BLUE, "figure": BODY, "far_limbs": FAR, "background": BG,
                        "cue_line": INK}}
    doc = {"_meta": meta, **index}
    (out_dir / "index.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sheet = contact_sheet_svg(specs, order, texts)
    (out_dir / "contact_sheet.svg").write_text(sheet, encoding="utf-8")
    if png and _png(sheet, out_dir / "contact_sheet.png", colours=160):
        made_png = True
        if sheet_copy:
            shutil.copyfile(out_dir / "contact_sheet.png", sheet_copy)
    return {"faults": order, "png": made_png, "out_dir": str(out_dir)}


def load_index(path: Path | None = None) -> dict:
    try:
        return json.loads((path or ASSET_DIR / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the fault silhouette pictures (assets/silhouettes).")
    ap.add_argument("--out", default=str(ASSET_DIR))
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--sheet-copy", default=None, help="also copy contact_sheet.png here")
    a = ap.parse_args()
    res = build(Path(a.out), png=not a.no_png, sheet_copy=Path(a.sheet_copy) if a.sheet_copy else None)
    print(f"built {len(res['faults'])} faults into {res['out_dir']} (png: {res['png']})")


if __name__ == "__main__":
    main()
