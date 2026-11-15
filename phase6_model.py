#!/usr/bin/env python3
"""
Phase 6 prototype: score and diagnosis from the Phase 5 checkpoint measurements. OFFLINE ONLY.

Nothing here is imported by the live app (scorer.py, run_analysis.py, api.py). It reads the
Phase 5 measurement table and the coaches' judgements and writes a report.

Score (order first)
  A linear Bradley-Terry model with features: P(clip i beats clip j) = sigmoid(w . (z_i - z_j)).
  z are a few Phase 5 measurements, oriented so that higher = better (the direction registered
  on 14 Nov 2026), standardised with the training clips' median and IQR, clipped at +-3.
  Training rows are pairs: every within-rater pair implied by Alex's scores and by Cam's
  quick-check scores (ties skipped), plus both coaches' swipe pairs. L2-regularised logistic
  loss with no intercept; optionally w >= 0 (sign constraint: every feature is oriented so that
  higher = better, so a negative weight would contradict the registered direction).

Features (chosen before any fit, from the automatic-key-frame test only)
  Candidates: measurements that pass test 3 (automatic key frames) and test 0 (coverage) in
  baselines/phase5_2026-11-15_v3/entry_tests.csv. Primary set: the best one per pillar by the
  test-3 Spearman. Extended set: then the next best overall, at most two per pillar, up to 6.
  Agreement with the coaches plays no part in the choice.

Evaluation
  Leave one PLAYER out (clip_sheet.csv player_id): a held-out player's clips appear in no
  training pair and in no scaling statistic. Spearman of the out-of-fold strength against Alex's
  and Cam's overall scores; accuracy on held-out pairs; 95% intervals from a bootstrap over
  players (the evaluation is resampled, the model is not refitted). Benchmarks: the four-rule
  scorer (Phase 0, mode=auto) and a constant scorer; the model-as-fourth-coach for context.

Calibration
  Isotonic map from strength to the average of the two coaches' z-scored overall scores,
  rescaled to Alex's mean and SD. Reported, but the order metrics are what matter.

Diagnosis
  Priority fix = the earliest failed checkpoint in rubric_v0_2.json faults_in_chain_order,
  among checkpoints whose measurement passes test 3 (and coverage). A checkpoint fails when its
  measurement is worse than the 75th percentile of the Good Club + Elite clips (PROVISIONAL).
  Nothing fails -> F00 "nothing major to fix". Compared with Cam's main fault per clip.

  python phase6_model.py [--out baselines/phase6_prototype_<date>] [--reps 2000]
                         [--measurements <phase5 dir>/checkpoint_measurements.csv --entry-tests <dir>/entry_tests.csv]
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
PHASE5 = ROOT / "baselines" / "phase5_2026-11-15_v3"
PHASE2 = ROOT / "baselines" / "phase2_preliminary_2026-11-14"
PHASE0 = ROOT / "baselines" / "phase0_2026-11-14"

PILLARS = ("access", "tracking", "stability", "flow")
Z_CLIP = 3.0              # standardised values are clipped at +-3 (limits one wild pose frame)
LAMBDA = 0.1              # L2 penalty on the mean pairwise log-loss; fixed before any fit
THRESHOLD_Q = 0.75        # a checkpoint fails when worse than this share of Good Club + Elite clips
REFERENCE_TIERS = ("Good Club", "Elite")
F00 = "F00"


# ----------------------------------------------------------------------------- small helpers

def clip_key(filename: str) -> str:
    """'Drive_Average 2.MOV' and 'Drive_Average 2' -> 'drive_average 2'."""
    name = filename.strip()
    stem = Path(name).stem if Path(name).suffix.lower() in (".mov", ".mp4", ".m4v", ".avi") else name
    return stem.casefold()


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as h:
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(h)]


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    fields = list(fields or (rows[0].keys() if rows else []))
    with Path(path).open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})


def _fmt(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.4g}"
    return v


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(xa) | np.isnan(ya))
    xa, ya = xa[keep], ya[keep]
    if len(xa) < 3:
        return float("nan")
    rx, ry = rankdata(xa), rankdata(ya)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ----------------------------------------------------------------------------- measurements

def orient(value: float, direction: str) -> float:
    """Registered direction -> 'higher = better'."""
    if direction == "higher_worse":
        return -value
    if direction == "higher_better":
        return value
    raise ValueError(f"unknown direction {direction!r}")


def load_measurements(path: Path, anchor_set: str = "auto") -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
    """-> ({clip_key: {measurement: raw value or nan}}, {measurement: {direction, checkpoint, fault, pillar, unit}})."""
    values: dict[str, dict[str, float]] = {}
    meta: dict[str, dict] = {}
    for r in read_csv(path):
        if r.get("anchor_set") != anchor_set:
            continue
        m = r["measurement"]
        values.setdefault(clip_key(r["filename"]), {})[m] = _num(r["value"])
        meta.setdefault(m, {k: r[k] for k in ("direction", "checkpoint", "fault", "pillar", "unit")})
    return values, meta


def select_features(entry_rows: Sequence[dict], n_features: int = 4, per_pillar_max: int = 2) -> list[str]:
    """Pre-registered rule: pass test 3 and test 0; best per pillar by test-3 rho; then fill to n."""
    cands = [r for r in entry_rows if r.get("test3_auto_anchors") == "pass" and r.get("test0_coverage") == "pass"]

    def rank_key(r):
        iqr = _num(r.get("iqr"))
        gap = _num(r.get("median_abs_auto_diff")) / iqr if iqr and iqr > 0 else float("inf")
        return (-_num(r.get("rho_auto")), gap, r["measurement"])

    cands.sort(key=rank_key)
    chosen: list[str] = []
    for p in PILLARS:
        best = [r for r in cands if r["pillar"] == p]
        if best and len(chosen) < n_features:
            chosen.append(best[0]["measurement"])
    pillar_of = {r["measurement"]: r["pillar"] for r in cands}
    for r in cands:
        if len(chosen) >= n_features:
            break
        m = r["measurement"]
        if m not in chosen and sum(pillar_of[c] == r["pillar"] for c in chosen) < per_pillar_max:
            chosen.append(m)
    return chosen


def fit_scaler(values: dict[str, dict[str, float]], clips: Iterable[str], feats: Sequence[str],
               meta: dict[str, dict]) -> dict[str, tuple[float, float]]:
    """Median and IQR of the ORIENTED values on the given (training) clips only."""
    out = {}
    clips = list(clips)
    for m in feats:
        v = np.array([orient(values[c].get(m, np.nan), meta[m]["direction"]) for c in clips], float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            out[m] = (0.0, 1.0)
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        out[m] = (float(med), float(iqr) if iqr > 0 else 1.0)
    return out


def standardise(values: dict[str, dict[str, float]], clips: Sequence[str], feats: Sequence[str],
                meta: dict[str, dict], scaler: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    """{clip: z vector}, oriented higher = better, clipped at +-Z_CLIP; missing stays nan."""
    out = {}
    for c in clips:
        z = []
        for m in feats:
            raw = values.get(c, {}).get(m, np.nan)
            if np.isnan(raw):
                z.append(np.nan)
                continue
            med, iqr = scaler[m]
            z.append(float(np.clip((orient(raw, meta[m]["direction"]) - med) / iqr, -Z_CLIP, Z_CLIP)))
        out[c] = np.array(z, float)
    return out


# ----------------------------------------------------------------------------- pairs

def pairs_from_scores(scores: dict[str, float], source: str) -> list[dict]:
    """Every within-rater pair implied by one rater's scores; ties and missing scores skipped."""
    keys = sorted(k for k, v in scores.items() if not np.isnan(v))
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if scores[a] == scores[b]:
                continue
            w, l = (a, b) if scores[a] > scores[b] else (b, a)
            out.append({"source": source, "winner": w, "loser": l})
    return out


def pairs_from_swipes(rows: Iterable[dict], source: str) -> list[dict]:
    out = []
    for r in rows:
        pick = (r.get("better") or "").strip().upper()
        if pick not in ("L", "R"):
            continue
        a, b = clip_key(r["left"]), clip_key(r["right"])
        w, l = (a, b) if pick == "L" else (b, a)
        out.append({"source": source, "winner": w, "loser": l})
    return out


def pair_design(pairs: Sequence[dict], z: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Both orientations of every pair: (z_w - z_l, 1) and (z_l - z_w, 0). Missing on either side -> 0 difference."""
    rows, ys = [], []
    for p in pairs:
        d = z[p["winner"]] - z[p["loser"]]
        d = np.where(np.isnan(d), 0.0, d)
        rows += [d, -d]
        ys += [1.0, 0.0]
    if not rows:
        return np.zeros((0, 0)), np.zeros(0)
    return np.vstack(rows), np.array(ys)


def strength(zc: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(np.where(np.isnan(zc), 0.0, zc), w))


# ----------------------------------------------------------------------------- the model

def _sigmoid(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def fit_pairwise(X: np.ndarray, y: np.ndarray, lam: float = LAMBDA, nonneg: bool = True,
                 use_sklearn: bool = True, iters: int = 20000, tol: float = 1e-10) -> np.ndarray:
    """min mean log-loss(sigmoid(X w), y) + lam/2 |w|^2, no intercept, optionally w >= 0.

    Unconstrained with scikit-learn installed: LogisticRegression (same objective, C = 1/(n lam)).
    Otherwise: accelerated projected gradient (FISTA) in numpy, step 1/L with
    L = max eig(X'X)/(4n) + lam (the loss is L-smooth), so it converges for any data.
    """
    n, d = X.shape
    if n == 0:
        return np.zeros(d)
    if not nonneg and use_sklearn:
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0 / (n * lam), fit_intercept=False, tol=1e-10, max_iter=10000)
            clf.fit(X, y.astype(int))
            return clf.coef_.ravel().astype(float)
        except ImportError:
            pass
    L = float(np.linalg.eigvalsh(X.T @ X).max()) / (4 * n) + lam
    step = 1.0 / L
    w = np.zeros(d)
    v = w.copy()
    t = 1.0
    for _ in range(iters):
        g = X.T @ (_sigmoid(X @ v) - y) / n + lam * v
        w_new = v - step * g
        if nonneg:
            w_new = np.maximum(w_new, 0.0)
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        v = w_new + ((t - 1) / t_new) * (w_new - w)
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w, t = w_new, t_new
    return w


# ----------------------------------------------------------------------------- folds

def lopo_folds(clips: Sequence[str], player_of: dict[str, str]) -> list[tuple[str, list[str], list[str]]]:
    """[(player, test clips, train clips)]: one fold per player."""
    players = sorted({player_of[c] for c in clips})
    folds = []
    for p in players:
        test = [c for c in clips if player_of[c] == p]
        train = [c for c in clips if player_of[c] != p]
        folds.append((p, test, train))
    return folds


def training_pairs(pairs: Sequence[dict], train: Iterable[str]) -> list[dict]:
    tr = set(train)
    return [p for p in pairs if p["winner"] in tr and p["loser"] in tr]


def assert_no_leak(train_pairs: Sequence[dict], test_clips: Iterable[str], scaler_clips: Iterable[str],
                   player_of: dict[str, str]) -> None:
    held = {player_of[c] for c in test_clips}
    for p in train_pairs:
        if player_of[p["winner"]] in held or player_of[p["loser"]] in held:
            raise AssertionError(f"held-out player in a training pair: {p}")
    for c in scaler_clips:
        if player_of[c] in held:
            raise AssertionError(f"held-out player in the scaling statistics: {c}")


def run_lopo(values, meta, feats, clips, player_of, pairs, lam=LAMBDA, nonneg=True) -> dict:
    """-> {'oof': {clip: strength}, 'records': [pair evaluations], 'weights': {player: w}, 'fold_models': {...}}."""
    oof: dict[str, float] = {}
    records: list[dict] = []
    weights: dict[str, np.ndarray] = {}
    fold_strengths: dict[str, dict[str, float]] = {}
    clipset = set(clips)
    usable = [p for p in pairs if p["winner"] in clipset and p["loser"] in clipset]
    for player, test, train in lopo_folds(clips, player_of):
        scaler = fit_scaler(values, train, feats, meta)
        tr_pairs = training_pairs(usable, train)
        assert_no_leak(tr_pairs, test, train, player_of)
        z = standardise(values, clips, feats, meta, scaler)
        X, y = pair_design(tr_pairs, z)
        w = fit_pairwise(X, y, lam=lam, nonneg=nonneg)
        weights[player] = w
        s = {c: strength(z[c], w) for c in clips}
        fold_strengths[player] = s
        for c in test:
            oof[c] = s[c]
        tset = set(test)
        for p in usable:
            if p["winner"] in tset or p["loser"] in tset:
                records.append({"fold": player, "source": p["source"], "winner": p["winner"], "loser": p["loser"],
                                "s_winner": s[p["winner"]], "s_loser": s[p["loser"]]})
    return {"oof": oof, "records": records, "weights": weights, "fold_strengths": fold_strengths}


# ----------------------------------------------------------------------------- metrics and bootstrap

def pair_correct(s_w: float, s_l: float) -> float:
    if np.isnan(s_w) or np.isnan(s_l):
        return float("nan")
    return 1.0 if s_w > s_l else (0.5 if s_w == s_l else 0.0)


def player_resamples(players: Sequence[str], reps: int, seed: int = 0) -> list[dict[str, int]]:
    rng = np.random.default_rng(seed)
    players = list(players)
    out = []
    for _ in range(reps):
        draw = rng.integers(0, len(players), len(players))
        m: dict[str, int] = {}
        for i in draw:
            m[players[i]] = m.get(players[i], 0) + 1
        out.append(m)
    return out


def boot_spearman(clips: Sequence[str], x: dict[str, float], y: dict[str, float], player_of: dict[str, str],
                  resamples: Sequence[dict[str, int]]) -> tuple[float, float, float, int]:
    cl = [c for c in clips if c in x and c in y and not np.isnan(x[c]) and not np.isnan(y[c])]
    est = spearman([x[c] for c in cl], [y[c] for c in cl])
    boots = []
    for m in resamples:
        idx = [c for c in cl for _ in range(m.get(player_of[c], 0))]
        r = spearman([x[c] for c in idx], [y[c] for c in idx])
        if not np.isnan(r):
            boots.append(r)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return est, float(lo), float(hi), len(cl)


def boot_diff(clips, x1, x2, y, player_of, resamples) -> tuple[float, float, float]:
    """Spearman(x1, y) - Spearman(x2, y) on the clips both cover, with a player bootstrap."""
    cl = [c for c in clips if all(c in d and not np.isnan(d[c]) for d in (x1, x2, y))]
    est = spearman([x1[c] for c in cl], [y[c] for c in cl]) - spearman([x2[c] for c in cl], [y[c] for c in cl])
    boots = []
    for m in resamples:
        idx = [c for c in cl for _ in range(m.get(player_of[c], 0))]
        d = spearman([x1[c] for c in idx], [y[c] for c in idx]) - spearman([x2[c] for c in idx], [y[c] for c in idx])
        if not np.isnan(d):
            boots.append(d)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return est, float(lo), float(hi)


def boot_accuracy(records: Sequence[dict], correct_key: str, player_of, resamples) -> tuple[float, float, float, int]:
    recs = [r for r in records if not np.isnan(r[correct_key])]
    if not recs:
        return float("nan"), float("nan"), float("nan"), 0
    c = np.array([r[correct_key] for r in recs])
    pw = [player_of[r["winner"]] for r in recs]
    pl = [player_of[r["loser"]] for r in recs]
    est = float(c.mean())
    boots = []
    for m in resamples:
        wt = np.array([m.get(a, 0) * m.get(b, 0) for a, b in zip(pw, pl)], float)
        if wt.sum() > 0:
            boots.append(float((wt * c).sum() / wt.sum()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi), len(recs)


# ----------------------------------------------------------------------------- calibration

def isotonic_fit(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators (non-decreasing). -> (sorted unique x, fitted y) for np.interp."""
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(xa, kind="mergesort")
    xs, ys = xa[order], ya[order]
    ux, inv = np.unique(xs, return_inverse=True)
    sums = np.bincount(inv, weights=ys)
    cnts = np.bincount(inv).astype(float)
    blocks = [[s, c, 1] for s, c in zip(sums, cnts)]  # [sum, weight, number of x values]
    merged: list[list[float]] = []
    for b in blocks:
        merged.append(b)
        while len(merged) > 1 and merged[-2][0] / merged[-2][1] > merged[-1][0] / merged[-1][1]:
            s, c, k = merged.pop()
            merged[-1][0] += s
            merged[-1][1] += c
            merged[-1][2] += k
    fitted = np.concatenate([np.full(int(k), s / c) for s, c, k in merged])
    return ux, fitted


def isotonic_predict(model: tuple[np.ndarray, np.ndarray], x: float) -> float:
    ux, fy = model
    return float(np.interp(x, ux, fy))


def calibration_target(alex: dict[str, float], cam: dict[str, float], clips: Sequence[str]) -> dict[str, float]:
    """Mean of the two coaches' z-scores (whichever exist), rescaled to Alex's mean and SD."""
    def zs(d):
        cl = [c for c in clips if c in d and not np.isnan(d[c])]
        v = np.array([d[c] for c in cl])
        return {c: (d[c] - v.mean()) / v.std(ddof=1) for c in cl}
    za, zc = zs(alex), zs(cam)
    av = np.array([alex[c] for c in clips if c in alex and not np.isnan(alex[c])])
    out = {}
    for c in clips:
        parts = [d[c] for d in (za, zc) if c in d]
        if parts:
            out[c] = float(av.mean() + av.std(ddof=1) * np.mean(parts))
    return out


# ----------------------------------------------------------------------------- diagnosis

def diagnosis_checkpoints(chain: Sequence[dict], entry_rows: Sequence[dict]) -> list[dict]:
    """Chain order restricted to checkpoints with a measurement passing test 3 and coverage.
    -> [{fault, checkpoint, measurement}] in chain order (one measurement per checkpoint: best test-3 rho)."""
    ok: dict[str, dict] = {}
    for r in entry_rows:
        if r.get("test3_auto_anchors") == "pass" and r.get("test0_coverage") == "pass":
            cur = ok.get(r["checkpoint"])
            if cur is None or _num(r["rho_auto"]) > _num(cur["rho_auto"]):
                ok[r["checkpoint"]] = r
    return [{"fault": f["id"], "checkpoint": f["checkpoint"], "measurement": ok[f["checkpoint"]]["measurement"]}
            for f in chain if f["checkpoint"] in ok]


def fit_thresholds(values: dict[str, dict[str, float]], ref_clips: Iterable[str], checkpoints: Sequence[dict],
                   meta: dict[str, dict], q: float = THRESHOLD_Q) -> dict[str, float]:
    """{measurement: threshold on the ORIENTED (higher = better) scale}: a clip fails when it is
    below the (1 - q) quantile of the reference clips, i.e. worse than a share q of them."""
    ref = list(ref_clips)
    out = {}
    for cp in checkpoints:
        m = cp["measurement"]
        v = np.array([orient(values[c].get(m, np.nan), meta[m]["direction"]) for c in ref if c in values], float)
        v = v[~np.isnan(v)]
        out[m] = float(np.percentile(v, 100 * (1 - q))) if len(v) else float("nan")
    return out


def diagnose(clip_values: dict[str, float], thresholds: dict[str, float], checkpoints: Sequence[dict],
             meta: dict[str, dict]) -> dict:
    """-> {'priority': fault id or F00, 'failed': [fault ids in chain order], 'not_measured': [...]}."""
    failed, missing = [], []
    for cp in checkpoints:
        m = cp["measurement"]
        raw = clip_values.get(m, np.nan)
        thr = thresholds.get(m, np.nan)
        if np.isnan(raw) or np.isnan(thr):
            missing.append(cp["fault"])
            continue
        if orient(raw, meta[m]["direction"]) < thr:
            failed.append(cp["fault"])
    return {"priority": failed[0] if failed else F00, "failed": failed, "not_measured": missing}


# ----------------------------------------------------------------------------- data loading

def load_inputs(root: Path = ROOT, measurements: Path | None = None, entry_tests: Path | None = None) -> dict:
    measurements = measurements or root / "baselines/phase5_2026-11-15_v3/checkpoint_measurements.csv"
    entry_tests = entry_tests or root / "baselines/phase5_2026-11-15_v3/entry_tests.csv"
    values, meta = load_measurements(measurements, "auto")
    entry = read_csv(entry_tests)
    rubric = json.loads((root / "docs/phase1/rubric_v0_2.json").read_text(encoding="utf-8"))
    tips = json.loads((root / "docs/phase6/fault_tips_draft.json").read_text(encoding="utf-8"))["tips"]
    gt = read_csv(root / "coach_ground_truth_from_screenshot.csv")
    sheet = read_csv(root / "clip_sheet.csv")
    cam = read_csv(root / "baselines/phase2_preliminary_2026-11-14/cam_quick_check.csv")
    base = read_csv(root / "baselines/phase0_2026-11-14/baseline_table.csv")
    fourth = read_csv(root / "baselines/phase2_preliminary_2026-11-14/model_vs_alex.csv")
    swipes_a = read_csv(root / "baselines/phase2_preliminary_2026-11-14/alex_swipe_pairs.csv")
    swipes_c = read_csv(root / "baselines/phase2_preliminary_2026-11-14/cam_swipe_pairs.csv")
    names = {clip_key(r["filename"]): r["filename"] for r in gt}
    return {
        "values": values, "meta": meta, "entry": entry, "chain": rubric["faults_in_chain_order"], "tips": tips,
        "names": names,
        "tier": {clip_key(r["filename"]): r["tier"] for r in gt},
        "alex": {clip_key(r["filename"]): _num(r["overall_score"]) for r in gt},
        "cam": {clip_key(r["filename"]): _num(r["overall"]) for r in cam},
        "cam_fault": {clip_key(r["filename"]): r["main_fault"] for r in cam},
        "cam_second": {clip_key(r["filename"]): r.get("second_fault", "") for r in cam},
        "player": {clip_key(r["filename"]): r["player_id"] for r in sheet},
        "venue": {clip_key(r["filename"]): r["venue"] for r in sheet},
        "four_rule": {clip_key(r["filename"]): _num(r["sys_overall"]) for r in base if r["mode"] == "auto"},
        "fourth_coach": {clip_key(r["filename"]): _num(r["model_overall"]) for r in fourth},
        "fourth_fault": {clip_key(r["filename"]): r["model_priority_fault"] for r in fourth},
        "swipes_alex": swipes_a, "swipes_cam": swipes_c, "measurements_path": measurements,
    }


def all_pairs(d: dict) -> list[dict]:
    return (pairs_from_scores(d["alex"], "alex_scores") + pairs_from_scores(d["cam"], "cam_scores")
            + pairs_from_swipes(d["swipes_alex"], "alex_swipes") + pairs_from_swipes(d["swipes_cam"], "cam_swipes"))


# ----------------------------------------------------------------------------- the run

SOURCES = ("alex_scores", "cam_scores", "alex_swipes", "cam_swipes")


def evaluate(d: dict, feats: Sequence[str], clips: Sequence[str], pairs, resamples, lam=LAMBDA, nonneg=True,
             full: bool = True) -> dict:
    res = run_lopo(d["values"], d["meta"], feats, clips, d["player"], pairs, lam=lam, nonneg=nonneg)
    oof = res["oof"]
    out = {"res": res, "rows": []}
    for target in ("alex", "cam"):
        est, lo, hi, n = boot_spearman(clips, oof, d[target], d["player"], resamples if full else resamples[:200])
        out[f"rho_{target}"] = (est, lo, hi, n)
    for r in res["records"]:
        r["model"] = pair_correct(r["s_winner"], r["s_loser"])
    acc = boot_accuracy(res["records"], "model", d["player"], resamples if full else resamples[:200])
    out["acc"] = acc
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measurements", type=Path, default=None,
                    help="Phase 5 checkpoint_measurements.csv (default: baselines/phase5_2026-11-15_v3)")
    ap.add_argument("--entry-tests", type=Path, default=None, help="the matching entry_tests.csv")
    args = ap.parse_args(argv)
    out = args.out or ROOT / "baselines" / f"phase6_prototype_{_dt.date.today().isoformat()}"
    out.mkdir(parents=True, exist_ok=True)

    d = load_inputs(measurements=args.measurements, entry_tests=args.entry_tests)
    values, meta = d["values"], d["meta"]
    clips = sorted(c for c in values if c in d["player"] and c in d["alex"])
    excluded = sorted(set(d["alex"]) - set(clips))
    players = sorted({d["player"][c] for c in clips})
    resamples = player_resamples(players, args.reps, args.seed)
    pairs = all_pairs(d)
    clipset = set(clips)
    usable = [p for p in pairs if p["winner"] in clipset and p["loser"] in clipset]

    feats4 = select_features(d["entry"], 4)
    feats6 = select_features(d["entry"], 6)

    # ---- primary model and variants
    primary = evaluate(d, feats4, clips, pairs, resamples)
    oof = primary["res"]["oof"]
    variants = [("primary: 4 features, w >= 0, lambda 0.1", feats4, LAMBDA, True)]
    variants += [("4 features, unconstrained", feats4, LAMBDA, False),
                 ("4 features, w >= 0, lambda 0.01", feats4, 0.01, True),
                 ("4 features, w >= 0, lambda 1", feats4, 1.0, True),
                 ("6 features, w >= 0, lambda 0.1", feats6, LAMBDA, True),
                 ("6 features, unconstrained", feats6, LAMBDA, False)]
    values_validated, _ = load_measurements(d["measurements_path"], "validated")
    d_validated = {**d, "values": values_validated}
    variants.append(("primary, but validated key frames (not what the product sees)", feats4, LAMBDA, True))
    variant_rows = []
    for name, f, lam, nn in variants:
        dv = d_validated if "validated" in name else d
        if name.startswith("primary:"):
            ev = primary
        else:
            ev = evaluate(dv, f, clips, pairs, resamples, lam, nn)
        wfull = fit_pairwise(*pair_design(usable, standardise(dv["values"], clips, f, meta,
                                                              fit_scaler(dv["values"], clips, f, meta))),
                             lam=lam, nonneg=nn)
        variant_rows.append({
            "variant": name, "features": " ".join(f),
            "rho_alex": ev["rho_alex"][0], "rho_alex_lo": ev["rho_alex"][1], "rho_alex_hi": ev["rho_alex"][2],
            "rho_cam": ev["rho_cam"][0], "rho_cam_lo": ev["rho_cam"][1], "rho_cam_hi": ev["rho_cam"][2],
            "pair_acc": ev["acc"][0], "pair_acc_lo": ev["acc"][1], "pair_acc_hi": ev["acc"][2],
            "full_data_weights": " ".join(f"{m}={w:.3f}" for m, w in zip(f, wfull)),
        })
    write_csv(out / "variants.csv", variant_rows)

    # ---- descriptive, after the fact: which single measurements order the clips as the coaches do
    signal_rows = []
    for mm in sorted(meta):
        row = {"measurement": mm, "checkpoint": meta[mm]["checkpoint"], "pillar": meta[mm]["pillar"],
               "direction": meta[mm]["direction"]}
        for tag, vals in (("auto", values), ("validated", values_validated)):
            o = {c: orient(vals[c].get(mm, np.nan), meta[mm]["direction"]) for c in clips}
            for t in ("alex", "cam"):
                est, lo, hi, n = boot_spearman(clips, o, d[t], d["player"], resamples)
                row.update({f"rho_{t}_{tag}": est, f"rho_{t}_{tag}_lo": lo, f"rho_{t}_{tag}_hi": hi})
        signal_rows.append(row)
    write_csv(out / "measurement_signal.csv", signal_rows)

    # ---- full-data fit (what would be shipped) and per-fold weights
    scaler_full = fit_scaler(values, clips, feats4, meta)
    z_full = standardise(values, clips, feats4, meta, scaler_full)
    w_full = fit_pairwise(*pair_design(usable, z_full), lam=LAMBDA, nonneg=True)
    coef_rows = [{"fit": "all clips", **{m: w for m, w in zip(feats4, w_full)}}]
    for p, w in primary["res"]["weights"].items():
        coef_rows.append({"fit": f"without {p}", **{m: x for m, x in zip(feats4, w)}})
    write_csv(out / "coefficients.csv", coef_rows, ["fit", *feats4])
    scaler_rows = [{"measurement": m, "checkpoint": meta[m]["checkpoint"], "pillar": meta[m]["pillar"],
                    "direction": meta[m]["direction"], "median_oriented": scaler_full[m][0],
                    "iqr": scaler_full[m][1], "weight": w} for m, w in zip(feats4, w_full)]
    write_csv(out / "features.csv", scaler_rows)

    # ---- benchmarks on the same clips and the same held-out pair records
    fr = d["four_rule"]
    venue = {c: (1.0 if d["venue"].get(c) == "indoor net lane" else 0.0) for c in clips}
    records = primary["res"]["records"]
    for r in records:
        r["four_rule"] = pair_correct(fr.get(r["winner"], np.nan), fr.get(r["loser"], np.nan))
        r["constant"] = 0.5
        r["venue_only"] = pair_correct(venue[r["winner"]], venue[r["loser"]])
        r["fourth_coach"] = pair_correct(d["fourth_coach"].get(r["winner"], np.nan),
                                         d["fourth_coach"].get(r["loser"], np.nan))
    scorers = {"model (out of fold)": oof, "four-rule scorer (Phase 0, auto)": fr, "venue only (confound check)": venue,
               "model as fourth coach (Claude, 14 clips)": d["fourth_coach"]}
    acc_keys = {"model (out of fold)": "model", "four-rule scorer (Phase 0, auto)": "four_rule",
                "venue only (confound check)": "venue_only", "model as fourth coach (Claude, 14 clips)": "fourth_coach"}
    metric_rows = []
    for name, sc in scorers.items():
        row = {"scorer": name}
        for t in ("alex", "cam"):
            est, lo, hi, n = boot_spearman(clips, sc, d[t], d["player"], resamples)
            row.update({f"rho_{t}": est, f"rho_{t}_lo": lo, f"rho_{t}_hi": hi, f"n_{t}": n})
        est, lo, hi, n = boot_accuracy(records, acc_keys[name], d["player"], resamples)
        row.update({"pair_acc": est, "pair_acc_lo": lo, "pair_acc_hi": hi, "n_pair_evals": n})
        for src in SOURCES:
            sub = [r for r in records if r["source"] == src and not np.isnan(r[acc_keys[name]])]
            row[f"acc_{src}"] = float(np.mean([r[acc_keys[name]] for r in sub])) if sub else float("nan")
        metric_rows.append(row)
    metric_rows.append({"scorer": "constant", "rho_alex": float("nan"), "rho_cam": float("nan"),
                        "pair_acc": 0.5, "pair_acc_lo": 0.5, "pair_acc_hi": 0.5, "n_pair_evals": len(records),
                        **{f"acc_{s}": 0.5 for s in SOURCES}})
    coach_rho = boot_spearman(clips, d["alex"], d["cam"], d["player"], resamples)
    metric_rows.append({"scorer": "Cam vs Alex (ceiling)", "rho_alex": 1.0, "rho_cam": coach_rho[0],
                        "rho_cam_lo": coach_rho[1], "rho_cam_hi": coach_rho[2], "n_cam": coach_rho[3]})
    write_csv(out / "metrics.csv", metric_rows, ["scorer", "rho_alex", "rho_alex_lo", "rho_alex_hi", "n_alex",
                                                 "rho_cam", "rho_cam_lo", "rho_cam_hi", "n_cam", "pair_acc",
                                                 "pair_acc_lo", "pair_acc_hi", "n_pair_evals",
                                                 *[f"acc_{s}" for s in SOURCES]])

    diffs = {}
    for t in ("alex", "cam"):
        diffs[t] = boot_diff(clips, oof, fr, d[t], d["player"], resamples)
    # paired accuracy difference, model minus four-rule, same bootstrap
    for r in records:
        r["diff_four_rule"] = r["model"] - r["four_rule"] + 0.5  # shift so boot_accuracy's mean works; undo below
    da = boot_accuracy(records, "diff_four_rule", d["player"], resamples)
    diffs["acc"] = (da[0] - 0.5, da[1] - 0.5, da[2] - 0.5)
    # fourth coach on its own 14 clips, with the model restricted to the same clips
    fc_clips = [c for c in clips if c in d["fourth_coach"]]
    same14 = {t: boot_spearman(fc_clips, oof, d[t], d["player"], resamples) for t in ("alex", "cam")}

    # ---- calibration
    target = calibration_target(d["alex"], d["cam"], clips)
    iso_full = isotonic_fit([oof[c] for c in clips], [target[c] for c in clips])
    lopo_cal: dict[str, float] = {}
    const_cal: dict[str, float] = {}
    for p, test, train in lopo_folds(clips, d["player"]):
        s = primary["res"]["fold_strengths"][p]
        iso = isotonic_fit([s[c] for c in train], [target[c] for c in train])
        for c in test:
            lopo_cal[c] = isotonic_predict(iso, s[c])
            const_cal[c] = float(np.mean([target[c2] for c2 in train]))
    cal_rows = []
    for name, sc in (("model, calibrated (leave one player out)", lopo_cal),
                     ("model, calibrated (in-sample map)", {c: isotonic_predict(iso_full, oof[c]) for c in clips}),
                     ("four-rule scorer (Phase 0, auto)", fr),
                     ("constant (training mean of target)", const_cal)):
        row = {"scorer": name}
        for t, ref in (("target", target), ("alex", d["alex"]), ("cam", d["cam"])):
            cl = [c for c in clips if c in ref and not np.isnan(ref[c]) and c in sc]
            row[f"mae_{t}"] = float(np.mean([abs(sc[c] - ref[c]) for c in cl]))
            row[f"n_{t}"] = len(cl)
        cal_rows.append(row)
    write_csv(out / "calibration.csv", cal_rows)

    # ---- diagnosis
    cps = diagnosis_checkpoints(d["chain"], d["entry"])
    ref_all = [c for c in clips if d["tier"].get(c) in REFERENCE_TIERS]
    thr_full = fit_thresholds(values, ref_all, cps, meta)
    thr_rows = [{"order": i + 1, "fault": cp["fault"], "checkpoint": cp["checkpoint"], "measurement": cp["measurement"],
                 "direction": meta[cp["measurement"]]["direction"], "unit": meta[cp["measurement"]]["unit"],
                 "fails_when": ("above " if meta[cp["measurement"]]["direction"] == "higher_worse" else "below ")
                 + f"{orient(thr_full[cp['measurement']], meta[cp['measurement']]['direction']):.3g}",
                 "reference_clips": len(ref_all)} for i, cp in enumerate(cps)]
    write_csv(out / "thresholds.csv", thr_rows)

    def lopo_diagnoses(q):
        res = {}
        for p, test, train in lopo_folds(clips, d["player"]):
            ref = [c for c in train if d["tier"].get(c) in REFERENCE_TIERS]
            thr = fit_thresholds(values, ref, cps, meta, q)
            for c in test:
                res[c] = diagnose(values[c], thr, cps, meta)
        return res

    diag = lopo_diagnoses(THRESHOLD_Q)
    diag_sens = {q: lopo_diagnoses(q) for q in (0.9, 1.0)}
    measurable = {cp["fault"] for cp in cps}

    def agreement(dg):
        cl = [c for c in clips if d["cam_fault"].get(c)]
        exact = [dg[c]["priority"] == d["cam_fault"][c] for c in cl]
        cam_real = [c for c in cl if d["cam_fault"][c] != F00]
        in_failed = [d["cam_fault"][c] in dg[c]["failed"] for c in cam_real]
        cam_meas = [c for c in cam_real if d["cam_fault"][c] in measurable]
        in_failed_meas = [d["cam_fault"][c] in dg[c]["failed"] for c in cam_meas]
        f00_cam = [c for c in cl if d["cam_fault"][c] == F00]
        f00_hit = [dg[c]["priority"] == F00 for c in f00_cam]
        # chance: exact match if the model's choices were independent of Cam's
        pm: dict[str, float] = {}
        pc: dict[str, float] = {}
        for c in cl:
            pm[dg[c]["priority"]] = pm.get(dg[c]["priority"], 0) + 1 / len(cl)
            pc[d["cam_fault"][c]] = pc.get(d["cam_fault"][c], 0) + 1 / len(cl)
        chance = sum(pm.get(f, 0) * pc[f] for f in pc)
        mean_failed = float(np.mean([len(dg[c]["failed"]) for c in cl]))
        return {"clips": len(cl), "exact": sum(exact), "chance_exact": chance * len(cl),
                "cam_real": len(cam_real), "in_failed": sum(in_failed), "cam_measurable": len(cam_meas),
                "in_failed_measurable": sum(in_failed_meas), "cam_f00": len(f00_cam), "f00_hit": sum(f00_hit),
                "model_f00": sum(dg[c]["priority"] == F00 for c in cl), "mean_failed": mean_failed}

    agr = {THRESHOLD_Q: agreement(diag), **{q: agreement(v) for q, v in diag_sens.items()}}
    write_csv(out / "diagnosis_agreement.csv",
              [{"threshold": f"worse than {int(q * 100)}% of Good Club + Elite", **a} for q, a in agr.items()])
    fourth_exact = sum(d["fourth_fault"].get(c) == d["cam_fault"].get(c) for c in clips
                       if c in d["fourth_fault"] and d["cam_fault"].get(c))
    fourth_n = sum(1 for c in clips if c in d["fourth_fault"] and d["cam_fault"].get(c))

    counts: dict[str, int] = {}
    fail_counts: dict[str, int] = {}
    for c in clips:
        counts[diag[c]["priority"]] = counts.get(diag[c]["priority"], 0) + 1
        for f in diag[c]["failed"]:
            fail_counts[f] = fail_counts.get(f, 0) + 1
    order = [cp["fault"] for cp in cps] + [F00]
    write_csv(out / "fault_counts.csv",
              [{"fault": f, "chosen_as_priority": counts.get(f, 0), "failed_on_clips": fail_counts.get(f, ""),
                "cam_main_fault": sum(1 for c in clips if d["cam_fault"].get(c) == f)} for f in order])

    # ---- per-clip table
    clip_rows = []
    for c in sorted(clips, key=lambda c: -oof[c]):
        clip_rows.append({
            "filename": d["names"].get(c, c), "player_id": d["player"][c], "tier": d["tier"].get(c, ""),
            "alex_overall": d["alex"].get(c, np.nan), "cam_overall": d["cam"].get(c, np.nan),
            "four_rule_auto": fr.get(c, np.nan), "strength_oof": oof[c],
            "score_calibrated_oof": lopo_cal[c], "calibration_target": target[c],
            **{f"z_{m}": z_full[c][i] for i, m in enumerate(feats4)},
            "priority_fault": diag[c]["priority"], "failed_checkpoints": " ".join(diag[c]["failed"]),
            "not_measured": " ".join(diag[c]["not_measured"]),
            "cam_main_fault": d["cam_fault"].get(c, ""), "cam_second_fault": d["cam_second"].get(c, ""),
            "exact_match": ("" if not d["cam_fault"].get(c) else int(diag[c]["priority"] == d["cam_fault"][c])),
            "tip": d["tips"].get(diag[c]["priority"], ""),
        })
    write_csv(out / "clip_predictions.csv", clip_rows)
    write_csv(out / "pairs.csv", [{"source": p["source"], "winner": d["names"].get(p["winner"], p["winner"]),
                                   "loser": d["names"].get(p["loser"], p["loser"]),
                                   **{f"d_{m}": x for m, x in zip(feats4, pair_design([p], z_full)[0][0])}}
                                  for p in usable])
    write_csv(out / "heldout_pair_records.csv", records,
              ["fold", "source", "winner", "loser", "s_winner", "s_loser", "model", "four_rule", "venue_only",
               "fourth_coach"])

    report = build_report(d=d, clips=clips, excluded=excluded, players=players, pairs=pairs, usable=usable,
                          feats4=feats4, feats6=feats6, meta=meta, w_full=w_full, scaler_full=scaler_full,
                          metric_rows=metric_rows, diffs=diffs, same14=same14, variant_rows=variant_rows,
                          cal_rows=cal_rows, cps=cps, thr_rows=thr_rows, agr=agr, counts=counts,
                          fail_counts=fail_counts, order=order, clip_rows=clip_rows, reps=args.reps,
                          fourth_exact=(fourth_exact, fourth_n), coef_rows=coef_rows, signal_rows=signal_rows)
    (out / "report.md").write_text(report, encoding="utf-8")
    print(f"wrote {out}")
    return 0


# ----------------------------------------------------------------------------- report

def _rel(path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _ci(est, lo, hi, digits=2) -> str:
    if est is None or (isinstance(est, float) and math.isnan(est)):
        return "n/a"
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return f"{est:.{digits}f}"
    return f"{est:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})"


def _pct(est, lo, hi) -> str:
    if est is None or (isinstance(est, float) and math.isnan(est)):
        return "n/a"
    return f"{100 * est:.0f}% ({100 * lo:.0f} to {100 * hi:.0f})"


def build_report(**k) -> str:
    d, meta = k["d"], k["meta"]
    m = {r["scorer"]: r for r in k["metric_rows"]}
    model = m["model (out of fold)"]
    four = m["four-rule scorer (Phase 0, auto)"]
    venue = m["venue only (confound check)"]
    fourth = m["model as fourth coach (Claude, 14 clips)"]
    ceiling = m["Cam vs Alex (ceiling)"]
    src_n = {s: sum(1 for p in k["usable"] if p["source"] == s) for s in SOURCES}
    src_all = {s: sum(1 for p in k["pairs"] if p["source"] == s) for s in SOURCES}
    a = k["agr"][THRESHOLD_Q]
    L = []
    L.append("# Phase 6 prototype: score and diagnosis (offline)\n")
    L.append(f"Generated by `phase6_model.py` on {_dt.date.today().isoformat()}. Offline only: nothing here changes "
             f"the live scorer. Inputs: `{_rel(d['measurements_path'])}` with automatic key frames (`anchor_set = auto`, as "
             "the product would run), Alex's scores, Cam's quick check, both coaches' swipe pairs.\n")
    L.append(f"**Read this first.** {len(k['clips'])} clips from {len(k['players'])} players. This is a feasibility "
             "check, not proof: the intervals below are wide, the strong clips were nearly all filmed in one indoor "
             "net lane and the weak ones outdoors or in a sports hall (the venue/camera confound), and the coaches' "
             "scores were known when the plan was written. The question it answers is \"is this worth building on "
             "more clips?\", not \"does it work?\".\n")
    if k["excluded"]:
        L.append(f"Left out: {', '.join(d['names'].get(c, c) for c in k['excluded'])} (no Phase 5 measurements; "
                 "filmed_correctly = N).\n")

    L.append("## What was built\n")
    L.append("- **Score:** a linear Bradley-Terry model with features (pairwise logistic regression, L2 penalty "
             f"lambda = {LAMBDA} on the mean log-loss, no intercept, weights constrained to be >= 0 because every "
             "feature is oriented so that higher = better). Penalty, constraint and features were fixed before any fit.")
    L.append(f"- **Features** (rule: pass test 3 and coverage; best automatic-key-frame Spearman in each pillar): "
             + ", ".join(f"{f} ({meta[f]['pillar']}, {meta[f]['checkpoint']}, {meta[f]['direction'].replace('_', ' ')})"
                         for f in k["feats4"]) + ". Standardised with the training clips' median and IQR, "
             "clipped at +-3; a missing value counts as no difference.")
    L.append(f"- **Training pairs:** Alex's scores imply {src_n['alex_scores']} pairs, Cam's {src_n['cam_scores']} "
             f"(ties skipped), plus {src_n['alex_swipes']} + {src_n['cam_swipes']} swipe pairs "
             f"({sum(src_n.values())} in all; {sum(src_all.values()) - sum(src_n.values())} more involve the clip "
             "with no measurements).")
    L.append("- **Evaluation:** leave one player out (13 folds; Cam's four clips leave together, as do Alex's two). "
             "A held-out player is in no training pair and no scaling statistic (checked in code on every fold). "
             f"Intervals: 95% bootstrap over players ({k['reps']} resamples); they resample the evaluation, not the fit, "
             "so they are if anything too narrow. With 13 players (one of them holding 4 of the 8 strong clips) the "
             "bootstrap is lumpy and the intervals are skewed; read them as rough.\n")

    L.append("## Order: does the model rank the drives as the coaches do?\n")
    L.append("Spearman of out-of-fold strength with each coach's overall score; pairwise accuracy on held-out pairs "
             "(every coach pair with at least one clip from the held-out player, scored by that fold's model; "
             "a tie counts as half).\n")
    L.append("| Scorer | Spearman vs Alex | Spearman vs Cam | Held-out pairs correct |")
    L.append("|---|---|---|---|")
    for r in (model, four, k["metric_rows"][-2], venue, fourth):
        name = r["scorer"]
        cells = []
        for t in ("alex", "cam"):
            txt = _ci(r.get(f"rho_{t}"), r.get(f"rho_{t}_lo"), r.get(f"rho_{t}_hi"))
            cells.append(txt + (f" (n={r[f'n_{t}']})" if r.get(f"n_{t}") else ""))
        L.append(f"| {name} | {cells[0]} | {cells[1]} | {_pct(r['pair_acc'], r['pair_acc_lo'], r['pair_acc_hi'])} |")
    L.append(f"| Cam vs Alex (the ceiling) | 1 | {_ci(ceiling['rho_cam'], ceiling['rho_cam_lo'], ceiling['rho_cam_hi'])} | |")
    L.append("")
    L.append("The constant scorer gives every clip the same score, so it has no rank correlation and gets every pair "
             "half right. The venue-only row is not a candidate: it scores 1 for the indoor net lane and 0 elsewhere, "
             "to show how much of the order on these clips the filming conditions alone explain. The fourth-coach row "
             "is scored on its own 14 clips and the held-out pairs among them.\n")
    da, dc, dacc = k["diffs"]["alex"], k["diffs"]["cam"], k["diffs"]["acc"]
    L.append(f"Model minus four-rule scorer, same clips and same bootstrap: Spearman vs Alex {_ci(*da)}, vs Cam "
             f"{_ci(*dc)}; held-out pairs {100 * dacc[0]:+.0f} points ({100 * dacc[1]:+.0f} to {100 * dacc[2]:+.0f}).\n")
    s14a, s14c = k["same14"]["alex"], k["same14"]["cam"]
    L.append(f"On the 14 clips the model-as-fourth-coach scored, this model's Spearman is {_ci(*s14a[:3])} vs Alex and "
             f"{_ci(*s14c[:3])} vs Cam (fourth coach: {fourth['rho_alex']:.2f} and {fourth['rho_cam']:.2f}).\n")
    L.append("Held-out pair accuracy by source (model / four-rule / constant):\n")
    L.append("| Source | Model | Four-rule | Constant |")
    L.append("|---|---|---|---|")
    for s in SOURCES:
        L.append(f"| {s.replace('_', ' ')} | {100 * model[f'acc_{s}']:.0f}% | {100 * four[f'acc_{s}']:.0f}% | 50% |")
    L.append("")
    L.append("Variants (reported to show sensitivity; the primary model was declared before running):\n")
    L.append("| Variant | Spearman vs Alex | Spearman vs Cam | Held-out pairs | Weights (all clips) |")
    L.append("|---|---|---|---|---|")
    for r in k["variant_rows"]:
        L.append(f"| {r['variant']} | {_ci(r['rho_alex'], r['rho_alex_lo'], r['rho_alex_hi'])} | "
                 f"{_ci(r['rho_cam'], r['rho_cam_lo'], r['rho_cam_hi'])} | "
                 f"{_pct(r['pair_acc'], r['pair_acc_lo'], r['pair_acc_hi'])} | {r['full_data_weights']} |")
    L.append("")
    ws = k["coef_rows"]
    L.append("Weights of the primary model on all clips, and their range across the 13 folds:\n")
    L.append("| Feature | Pillar | Weight (all clips) | Fold range | Folds at 0 |")
    L.append("|---|---|---|---|---|")
    for i, f in enumerate(k["feats4"]):
        fv = [r[f] for r in ws[1:]]
        L.append(f"| {f} | {meta[f]['pillar']} | {ws[0][f]:.3f} | {min(fv):.3f} to {max(fv):.3f} | "
                 f"{sum(1 for v in fv if v <= 1e-9)} of {len(fv)} |")
    L.append("")

    L.append("### Which single measurements order the clips as the coaches do? (descriptive, after the fact)\n")
    L.append("Spearman of each measurement, turned so that higher = better (registered direction), with the "
             "coaches' overall scores, on automatic and on validated key frames. Not used to choose anything; "
             "17 clips, so any one of these can be noise or the venue.\n")
    L.append("| Meas. | Pillar | vs Alex, auto | vs Cam, auto | vs Alex, validated | vs Cam, validated |")
    L.append("|---|---|---|---|---|---|")
    for r in k["signal_rows"]:
        L.append(f"| {r['measurement']} | {r['pillar']} | "
                 + " | ".join(_ci(r[f'rho_{t}_{g}'], r[f'rho_{t}_{g}_lo'], r[f'rho_{t}_{g}_hi'])
                              for g in ("auto", "validated") for t in ("alex", "cam")) + " |")
    L.append("")

    L.append("## Calibration to 0-100 (secondary)\n")
    L.append("Target: the mean of the two coaches' z-scored overall scores, rescaled to Alex's mean and SD (the coaches "
             "agree on order but Cam spreads weak drives much lower). Isotonic map from strength to target; in the "
             "leave-one-player-out row the map is fitted on the training players only.\n")
    L.append("| Scorer | Mean abs error vs target | vs Alex | vs Cam |")
    L.append("|---|---|---|---|")
    for r in k["cal_rows"]:
        L.append(f"| {r['scorer']} | {r['mae_target']:.1f} | {r['mae_alex']:.1f} | {r['mae_cam']:.1f} |")
    L.append("")
    L.append("The number is only as good as the order underneath it; with 17 clips the isotonic map has few steps "
             "and the in-sample row is optimistic. Judge the order metrics above, not this table.\n")

    L.append("## Diagnosis: one priority fix per clip\n")
    L.append("Rule: walk `faults_in_chain_order` (rubric v0.2) and name the first checkpoint that fails; F00 "
             "\"nothing major to fix\" when none fails. Only checkpoints with a measurement passing test 3 and "
             "coverage take part: " + ", ".join(f"{cp['fault']} ({cp['measurement']})" for cp in k["cps"]) +
             ". Not assessable from these measurements: F13, F15, F14, F16 (no measurement yet), F09 (M09 fails test 3). "
             "A checkpoint fails when its measurement is worse than 75% of the Good Club + Elite clips "
             "(**provisional** threshold; in the evaluation the held-out player's clips are left out of the reference).\n")
    L.append("Thresholds on all reference clips:\n")
    L.append("| Order | Fault | Measurement | Fails when | Unit |")
    L.append("|---|---|---|---|---|")
    for r in k["thr_rows"]:
        L.append(f"| {r['order']} | {r['fault']} | {r['measurement']} | {r['fails_when']} | {r['unit']} |")
    L.append("")
    L.append("Agreement with Cam's main fault:\n")
    L.append("| Threshold | Exact match | Chance (same marginals) | Cam's fault among failed checkpoints | "
             "...where Cam's fault is measurable | Cam says F00 and model says F00 | Model says F00 | Failed checkpoints per clip |")
    L.append("|---|---|---|---|---|---|---|---|")
    for q, g in k["agr"].items():
        L.append(f"| worse than {int(q * 100)}% of good clips | {g['exact']} of {g['clips']} | {g['chance_exact']:.1f} | "
                 f"{g['in_failed']} of {g['cam_real']} | {g['in_failed_measurable']} of {g['cam_measurable']} | "
                 f"{g['f00_hit']} of {g['cam_f00']} | {g['model_f00']} | {g['mean_failed']:.1f} |")
    L.append("")
    fe, fn = k["fourth_exact"]
    L.append(f"For context, the model-as-fourth-coach (Claude on stills) matched Cam's main fault on {fe} of {fn} clips.\n")
    L.append("How often each fault is chosen (primary threshold):\n")
    L.append("| Fault | Chosen as priority | Failed on clips | Cam's main fault |")
    L.append("|---|---|---|---|")
    for f in k["order"]:
        L.append(f"| {f} | {k['counts'].get(f, 0)} | {k['fail_counts'].get(f, 0) if f != F00 else ''} | "
                 f"{sum(1 for c in k['clips'] if d['cam_fault'].get(c) == f)} |")
    other = sorted({d["cam_fault"][c] for c in k["clips"] if d["cam_fault"].get(c)} - set(k["order"]))
    if other:
        L.append(f"| other (Cam only: {', '.join(other)}) | | | "
                 f"{sum(1 for c in k['clips'] if d['cam_fault'].get(c) in other)} |")
    L.append("")

    L.append("## Per clip\n")
    L.append("| Clip | Tier | Alex | Cam | Four-rule | Strength (out of fold) | Calibrated | Priority fix | Failed | Cam's fault |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in k["clip_rows"]:
        cam = r["cam_overall"]
        L.append(f"| {r['filename']} | {r['tier']} | {r['alex_overall']:.0f} | {'' if np.isnan(cam) else f'{cam:.0f}'} | "
                 f"{r['four_rule_auto']:.0f} | {r['strength_oof']:.2f} | {r['score_calibrated_oof']:.0f} | "
                 f"{r['priority_fault']} | {r['failed_checkpoints']} | {r['cam_main_fault']} |")
    L.append("")
    L.append(CONCLUSIONS_MARKER)
    L.append(conclusions(k))
    L.append("\nFiles: `clip_predictions.csv` (per clip, with the tip text), `metrics.csv`, `variants.csv`, "
             "`coefficients.csv`, `features.csv`, `calibration.csv`, `thresholds.csv`, `diagnosis_agreement.csv`, "
             "`fault_counts.csv`, `pairs.csv` (training pairs with feature differences, all clips), "
             "`heldout_pair_records.csv` (every held-out pair evaluation).\n")
    return "\n".join(L)


CONCLUSIONS_MARKER = "## Conclusions\n"


def conclusions(k) -> str:
    m = {r["scorer"]: r for r in k["metric_rows"]}
    model, four, venue = m["model (out of fold)"], m["four-rule scorer (Phase 0, auto)"], m["venue only (confound check)"]
    da, dc = k["diffs"]["alex"], k["diffs"]["cam"]
    a = k["agr"][THRESHOLD_Q]
    out = []
    if da[1] > 0 and dc[1] > 0:
        verdict = "The interval on the difference excludes zero for both coaches: better than the four-rule scorer. "
    elif da[2] < 0 and dc[2] < 0:
        verdict = "The interval on the difference excludes zero for both coaches: worse than the four-rule scorer. "
    else:
        lead = "the model" if (da[0] + dc[0]) > 0 else "the four-rule scorer"
        verdict = (f"The point estimates favour {lead}, but the intervals on the difference include zero, so on "
                   "17 clips neither is shown to be better. ")
    out.append(f"1. **Order.** Out of fold, the model's Spearman is {model['rho_alex']:.2f} with Alex and "
               f"{model['rho_cam']:.2f} with Cam; the four-rule scorer's is {four['rho_alex']:.2f} and "
               f"{four['rho_cam']:.2f}; a constant scorer has none. " + verdict
               + f"Held-out pairs correct: model {100 * model['pair_acc']:.0f}%, four-rule "
               f"{100 * four['pair_acc']:.0f}%, constant 50%.")
    out.append(f"2. **Confound.** A scorer that only knows the venue (indoor net lane = 1, else 0) reaches "
               f"{venue['rho_alex']:.2f} with Alex and {venue['rho_cam']:.2f} with Cam. Any number in the order "
               "table has to beat that to mean much on these clips; the new clips must mix venues within tier.")
    out.append(f"3. **Diagnosis.** At the provisional threshold the model names Cam's main fault on {a['exact']} of "
               f"{a['clips']} clips (chance with the same marginals: {a['chance_exact']:.1f}); Cam's fault is among the "
               f"model's failed checkpoints on {a['in_failed']} of {a['cam_real']} clips; it says F00 on "
               f"{a['f00_hit']} of the {a['cam_f00']} clips where Cam does. With eleven checkpoints each failing about "
               f"a quarter of good clips by construction, a clip fails {a['mean_failed']:.1f} on average, so the "
               "earliest-in-chain rule mostly picks an early checkpoint. The thresholds need coach checkpoint ratings, "
               "not tier percentiles.")
    vrow = {r["variant"]: r for r in k["variant_rows"]}
    val = next(r for n, r in vrow.items() if "validated" in n)
    prim = next(r for n, r in vrow.items() if n.startswith("primary:"))
    against = [r["measurement"] for r in k["signal_rows"]
               if r["rho_alex_auto"] < -0.2 and r["rho_cam_auto"] < -0.2]
    out.append(f"4. **Where the weakness is.** With validated key frames in place of automatic ones the same model "
               f"reaches {val['rho_alex']:.2f} / {val['rho_cam']:.2f} (automatic: {prim['rho_alex']:.2f} / "
               f"{prim['rho_cam']:.2f}), so key-frame error is not the main limit. The measurements themselves carry "
               "little of what the coaches see in the overall score: no single one reaches 0.4 with either coach, and "
               + (f"{', '.join(against)} run against their registered direction with both coaches "
                  "(as the Phase 5 log already noted for v1). They are not flipped here: flipping them on 17 clips "
                  "would fit noise and the venue." if against else "none runs clearly against its registered direction."))
    cal = {r["scorer"]: r for r in k["cal_rows"]}
    cm = cal["model, calibrated (leave one player out)"]
    out.append(f"5. **Calibration.** Leave one player out, the calibrated number is off the coaches' average by "
               f"{cm['mae_target']:.1f} points on average, against {cal['four-rule scorer (Phase 0, auto)']['mae_target']:.1f} "
               f"for the four-rule scorer and {cal['constant (training mean of target)']['mae_target']:.1f} for a "
               "constant. It inherits every error in the order, and with 17 clips the map is coarse.")
    out.append("6. **Status.** Feasibility only, and on this evidence the prototype is not ready to replace the "
               "four-rule scorer or to name a priority fix. The machinery (pairs, grouped evaluation, sign "
               "constraints, chain-order diagnosis with F00) works and is tested, so it can be rerun unchanged as "
               "clips and checkpoint ratings arrive.")
    out.append("")
    out.append("## What would make it better\n")
    out.append("- **More players, mixed venues within tier.** 13 players cannot separate technique from filming "
               "conditions; the venue alone orders these clips better than any measurement model. About 60 clips from "
               "30+ players, with good and weak batters in each venue, is the plan's next step (at 60 clips a Spearman "
               "of 0.7 has an interval of about 0.54 to 0.81).")
    out.append("- **Coach checkpoint ratings (2 / 1 / 0 / X).** They would (a) decide test 1 for each measurement, "
               "so only measurements that track the coaches enter the model; (b) replace the tier-percentile "
               "thresholds with thresholds fitted to the coaches' own fail/pass calls; (c) let the diagnosis be "
               "checked checkpoint by checkpoint, not only against one main fault.")
    out.append("- **Measurements for what coaches name but this run cannot measure:** F13 (trigger), F14 (top half "
               "closes), F15 (moves across, not out), F16 (weight over the ball), and a flow measurement to replace "
               "M09. Phase 5 v5 has registered M14 (front-foot landing against the start of the downswing, the "
               "research note's best-supported marker), M15, M16 and M17 but not yet run them on the clips; once "
               "they pass the entry tests, rerun this script on the v5 table (the feature rule picks them up "
               "automatically).")
    out.append("- **Pose quality on small, distant batters** (RTMPose in place of MediaPipe; see methods notes): "
               "several beginner clips have values set by a few bad frames, and Ollie P (two figures in shot) gets "
               "the lowest strength despite a Good Club score.")
    out.append("- **More swipe pairs, chosen actively**, so the training signal is not dominated by pairs implied "
               "by two sets of scores on the same 17 clips.")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
