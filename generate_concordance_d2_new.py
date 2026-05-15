#!/usr/bin/env python3
"""
D2-new: Per-pillar concordance between system scores and manual scores.
Reads from D1-new auto and validated detailed CSVs plus heldout_split.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "calibration_output" / "d1_new_baseline"
PILLARS = ("access", "tracking", "stability", "flow")
TIER_ORDER = ("Beginner", "Average", "Good Club", "Elite")


def _clean(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _fmt(v: Any, p: int = 2) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{float(v):.{p}f}"


def _load_heldout_split(path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            fn = _clean(row.get("filename", ""))
            val = _clean(row.get("split", "tuning")) or "tuning"
            if fn:
                lookup[fn.casefold()] = val
                lookup[Path(fn).stem.casefold()] = val
    return lookup


def _resolve_split(filename: str, lookup: dict[str, str]) -> str:
    return (
        lookup.get(Path(filename).name.casefold())
        or lookup.get(Path(filename).stem.casefold())
        or "tuning"
    )


def _load_detailed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    numeric = [
        "manual_overall", "local_overall",
        "access_manual", "access_local",
        "tracking_manual", "tracking_local",
        "stability_manual", "stability_local",
        "flow_manual", "flow_local",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_concordance(auto_df: pd.DataFrame, val_df: pd.DataFrame,
                       split_lookup: dict[str, str]) -> pd.DataFrame:
    # Store rows as plain dicts for easy None checks
    auto_map: dict[str, dict] = {_clean(r["filename"]): r.to_dict() for _, r in auto_df.iterrows()}
    val_map:  dict[str, dict] = {_clean(r["filename"]): r.to_dict() for _, r in val_df.iterrows()}
    filenames = sorted(set(auto_map) | set(val_map))

    rows: list[dict[str, Any]] = []
    for fn in filenames:
        a = auto_map.get(fn)
        v = val_map.get(fn)
        ref = a if a is not None else v
        tier  = _clean((ref or {}).get("tier", ""))
        split = _resolve_split(fn, split_lookup)

        # Manual scores (same in both modes — just manual ground truth)
        man_overall  = pd.to_numeric((ref or {}).get("manual_overall"), errors="coerce")
        man = {p: pd.to_numeric((ref or {}).get(f"{p}_manual"), errors="coerce") for p in PILLARS}

        # Auto system scores
        auto_overall = pd.to_numeric(a.get("local_overall"), errors="coerce") if a is not None else float("nan")
        auto = {p: pd.to_numeric(a.get(f"{p}_local"), errors="coerce") if a is not None else float("nan") for p in PILLARS}

        # Validated system scores
        val_overall  = pd.to_numeric(v.get("local_overall"), errors="coerce") if v is not None else float("nan")
        val = {p: pd.to_numeric(v.get(f"{p}_local"), errors="coerce") if v is not None else float("nan") for p in PILLARS}

        def _abs_err(sys_val: Any, man_val: Any) -> Any:
            s, m = pd.to_numeric(sys_val, errors="coerce"), pd.to_numeric(man_val, errors="coerce")
            if pd.isna(s) or pd.isna(m):
                return float("nan")
            return abs(s - m)

        row: dict[str, Any] = {
            "filename": fn,
            "tier": tier,
            "split": split,
            "manual_overall": man_overall,
            **{f"manual_{p}": man[p] for p in PILLARS},
            "auto_overall": auto_overall,
            **{f"auto_{p}": auto[p] for p in PILLARS},
            "validated_overall": val_overall,
            **{f"validated_{p}": val[p] for p in PILLARS},
            "auto_overall_abs_err": _abs_err(auto_overall, man_overall),
            **{f"auto_{p}_abs_err": _abs_err(auto[p], man[p]) for p in PILLARS},
            "validated_overall_abs_err": _abs_err(val_overall, man_overall),
            **{f"validated_{p}_abs_err": _abs_err(val[p], man[p]) for p in PILLARS},
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str]] = []
    for tier in list(TIER_ORDER) + sorted(set(df["tier"].dropna()) - set(TIER_ORDER)):
        for split in ("tuning", "heldout", "all"):
            if split == "all":
                subset = df[df["tier"] == tier]
            else:
                subset = df[(df["tier"] == tier) & (df["split"] == split)]
            if subset.empty:
                continue
            r: dict[str, Any] = {"tier": tier, "split": split, "n": len(subset)}
            for mode in ("auto", "validated"):
                r[f"mean_{mode}_overall_abs_err"] = subset[f"{mode}_overall_abs_err"].mean()
                for p in PILLARS:
                    r[f"mean_{mode}_{p}_abs_err"] = subset[f"{mode}_{p}_abs_err"].mean()
            rows.append(r)
    # All-tier rows
    for split in ("tuning", "heldout", "all"):
        if split == "all":
            subset = df
        else:
            subset = df[df["split"] == split]
        if subset.empty:
            continue
        r = {"tier": "ALL", "split": split, "n": len(subset)}
        for mode in ("auto", "validated"):
            r[f"mean_{mode}_overall_abs_err"] = subset[f"{mode}_overall_abs_err"].mean()
            for p in PILLARS:
                r[f"mean_{mode}_{p}_abs_err"] = subset[f"{mode}_{p}_abs_err"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def _build_bias(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        r: dict[str, Any] = {
            "filename": row["filename"],
            "tier": row["tier"],
            "split": row["split"],
        }
        for mode_label, mode_col in (("auto", "auto"), ("validated", "validated")):
            r[f"{mode_label}_overall_signed_err"] = (
                row[f"{mode_col}_overall"] - row["manual_overall"]
                if pd.notna(row[f"{mode_col}_overall"]) and pd.notna(row["manual_overall"])
                else float("nan")
            )
            for p in PILLARS:
                r[f"{mode_label}_{p}_signed_err"] = (
                    row[f"{mode_col}_{p}"] - row[f"manual_{p}"]
                    if pd.notna(row[f"{mode_col}_{p}"]) and pd.notna(row[f"manual_{p}"])
                    else float("nan")
                )
        rows.append(r)

    # Append tier mean rows
    detail = pd.DataFrame(rows)
    for tier in list(TIER_ORDER) + sorted(set(df["tier"].dropna()) - set(TIER_ORDER)):
        t = detail[detail["tier"] == tier]
        if t.empty:
            continue
        r = {"filename": f"__tier_mean__{tier}", "tier": tier, "split": ""}
        for mode in ("auto", "validated"):
            r[f"{mode}_overall_signed_err"] = t[f"{mode}_overall_signed_err"].mean()
            for p in PILLARS:
                r[f"{mode}_{p}_signed_err"] = t[f"{mode}_{p}_signed_err"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def _build_ranking(df: pd.DataFrame) -> str:
    lines: list[str] = []

    for mode, label in (("auto", "AUTO MODE"), ("validated", "VALIDATED MODE")):
        lines.append(f"── Top 5 worst video×pillar combinations ({label}) ──")
        combos: list[tuple[float, str, str, str]] = []
        for _, row in df.iterrows():
            fn = row["filename"]
            tier = row["tier"]
            for dim in ["overall"] + list(PILLARS):
                err = row[f"{mode}_{dim}_abs_err"]
                if pd.notna(err):
                    combos.append((float(err), fn, tier, dim))
        combos.sort(reverse=True)
        for err, fn, tier, dim in combos[:5]:
            lines.append(f"  {fn} [{tier}]  {dim}  abs_err={err:.1f}")
        lines.append("")

    # Within ±5 counts
    lines.append("── Within ±5 overall ──")
    for mode in ("auto", "validated"):
        col = f"{mode}_overall_abs_err"
        for split in ("tuning", "heldout"):
            sub = df[df["split"] == split]
            ok = int((sub[col] <= 5).sum()) if not sub.empty else 0
            lines.append(f"  {mode:<12}  {split:<8}  {ok}/{len(sub)}")
    lines.append("")

    lines.append("── Within ±5 on ALL FOUR pillars simultaneously ──")
    for mode in ("auto", "validated"):
        cols = [f"{mode}_{p}_abs_err" for p in PILLARS]
        for split in ("tuning", "heldout"):
            sub = df[df["split"] == split]
            if sub.empty:
                ok = 0
            else:
                ok = int((sub[cols].le(5).all(axis=1)).sum())
            lines.append(f"  {mode:<12}  {split:<8}  {ok}/{len(sub)}")
    lines.append("")

    # Per-tier worst pillar in validated mode
    lines.append("── Per-tier worst pillar in VALIDATED mode (pure rule failure, detection noise removed) ──")
    for tier in TIER_ORDER:
        t = df[df["tier"] == tier]
        if t.empty:
            continue
        pillar_maes = {p: t[f"validated_{p}_abs_err"].mean() for p in PILLARS}
        worst = max(pillar_maes, key=lambda k: pillar_maes[k] if pd.notna(pillar_maes[k]) else -1)
        lines.append(
            f"  {tier:<12}  worst={worst:<12}  mae={_fmt(pillar_maes[worst])}"
            f"  (all: " + "  ".join(f"{p}={_fmt(pillar_maes[p])}" for p in PILLARS) + ")"
        )
    lines.append("")

    # Highlight any validated worse than auto
    lines.append("── HIGHLIGHT: pillars where validated mode is WORSE than auto (detection noise accidentally helps) ──")
    found_any = False
    for tier in TIER_ORDER:
        t = df[df["tier"] == tier]
        if t.empty:
            continue
        for dim in ["overall"] + list(PILLARS):
            a_mae = t[f"auto_{dim}_abs_err"].mean()
            v_mae = t[f"validated_{dim}_abs_err"].mean()
            if pd.notna(a_mae) and pd.notna(v_mae) and v_mae > a_mae + 1.0:
                lines.append(
                    f"  {tier:<12}  {dim:<12}  auto_mae={_fmt(a_mae)}  validated_mae={_fmt(v_mae)}"
                    f"  delta=+{_fmt(v_mae - a_mae)}"
                )
                found_any = True
    if not found_any:
        lines.append("  None — validated mode is better (or equal) across all tiers and pillars.")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=str(BASE))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--heldout-split", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.base_dir).expanduser()
    if not base.is_absolute():
        base = (SCRIPT_DIR / base).resolve()
    out = Path(args.output_dir).expanduser() if args.output_dir else base
    if not out.is_absolute():
        out = (SCRIPT_DIR / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    split_path = (
        Path(args.heldout_split).expanduser()
        if args.heldout_split
        else SCRIPT_DIR / "heldout_split.csv"
    )
    split_lookup = _load_heldout_split(split_path)

    auto_df = _load_detailed(base / "auto" / "battingiq_calibration_comparison_detailed.csv")
    val_df  = _load_detailed(base / "validated" / "battingiq_calibration_comparison_detailed.csv")

    conc_df = _build_concordance(auto_df, val_df, split_lookup)
    summ_df = _build_summary(conc_df)
    bias_df = _build_bias(conc_df)
    ranking_text = _build_ranking(conc_df)

    # Define output columns
    conc_cols = (
        ["filename", "tier", "split", "manual_overall"]
        + [f"manual_{p}" for p in PILLARS]
        + ["auto_overall"] + [f"auto_{p}" for p in PILLARS]
        + ["validated_overall"] + [f"validated_{p}" for p in PILLARS]
        + ["auto_overall_abs_err"] + [f"auto_{p}_abs_err" for p in PILLARS]
        + ["validated_overall_abs_err"] + [f"validated_{p}_abs_err" for p in PILLARS]
    )
    summ_cols = (
        ["tier", "split", "n",
         "mean_auto_overall_abs_err"] + [f"mean_auto_{p}_abs_err" for p in PILLARS]
        + ["mean_validated_overall_abs_err"] + [f"mean_validated_{p}_abs_err" for p in PILLARS]
    )
    bias_cols = (
        ["filename", "tier", "split",
         "auto_overall_signed_err"] + [f"auto_{p}_signed_err" for p in PILLARS]
        + ["validated_overall_signed_err"] + [f"validated_{p}_signed_err" for p in PILLARS]
    )

    conc_df.reindex(columns=conc_cols).to_csv(out / "concordance_report.csv", index=False)
    summ_df.reindex(columns=summ_cols).to_csv(out / "concordance_summary.csv", index=False)
    bias_df.reindex(columns=bias_cols).to_csv(out / "concordance_bias.csv", index=False)
    (out / "concordance_ranking.txt").write_text(ranking_text, encoding="utf-8")

    print(f"concordance_report.csv:  {out / 'concordance_report.csv'}")
    print(f"concordance_summary.csv: {out / 'concordance_summary.csv'}")
    print(f"concordance_ranking.txt: {out / 'concordance_ranking.txt'}")
    print(f"concordance_bias.csv:    {out / 'concordance_bias.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
