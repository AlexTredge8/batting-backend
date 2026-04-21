#!/usr/bin/env python3
"""
Run BattingIQ analysis against coach ground-truth scores and export a calibration CSV.

The script matches coach rows to local videos by filename stem (case-insensitive,
extension-agnostic), runs the local BattingIQ pipeline, and writes a side-by-side
comparison file for score calibration.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable
from urllib import error as urlerror
from urllib import request as urlrequest
from xml.etree import ElementTree as ET
from zipfile import ZipFile


# The analysis stack uses Python 3.10+ syntax in its imported modules.
if sys.version_info < (3, 10):
    raise SystemExit(
        f"batch_compare.py requires Python 3.10+; current version is "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

# Keep matplotlib from trying to write under a non-writable home cache directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "battingiq-mplconfig"))

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv"}
REQUIRED_INPUT_COLUMNS = [
    "filename",
    "tier",
    "access_score",
    "tracking_score",
    "stability_score",
    "flow_score",
    "overall_score",
    "filmed_correctly",
    "coach_notes",
]
REVIEWER_EXPORT_COLUMNS = {
    "ai_overall_score",
    "ai_access_score",
    "ai_tracking_score",
    "ai_stability_score",
    "ai_flow_score",
    "reviewer_1_name",
}
OUTPUT_COLUMNS = [
    "filename",
    "tier",
    "filmed_correctly",
    "contact_detector_version",
    "anchor_detector_version",
    "coach_access",
    "coach_tracking",
    "coach_stability",
    "coach_flow",
    "coach_overall",
    "model_access",
    "model_tracking",
    "model_stability",
    "model_flow",
    "model_overall",
    "gap_access",
    "gap_tracking",
    "gap_stability",
    "gap_flow",
    "gap_overall",
]
COACH_TO_OUTPUT = {
    "access_score": "coach_access",
    "tracking_score": "coach_tracking",
    "stability_score": "coach_stability",
    "flow_score": "coach_flow",
    "overall_score": "coach_overall",
}
MODEL_FIELDS = {
    "model_access": ("pillars", "access", "score"),
    "model_tracking": ("pillars", "tracking", "score"),
    "model_stability": ("pillars", "stability", "score"),
    "model_flow": ("pillars", "flow", "score"),
    "model_overall": ("battingiq_score",),
}
GAP_FIELDS = {
    "gap_access": ("coach_access", "model_access"),
    "gap_tracking": ("coach_tracking", "model_tracking"),
    "gap_stability": ("coach_stability", "model_stability"),
    "gap_flow": ("coach_flow", "model_flow"),
    "gap_overall": ("coach_overall", "model_overall"),
}


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def _normalise_header_row(headers: Iterable[object]) -> list[str]:
    return [str(cell).strip() if cell is not None else "" for cell in headers]


def _ensure_expected_columns(headers: list[str], source: Path) -> None:
    if headers == REQUIRED_INPUT_COLUMNS:
        return

    if REVIEWER_EXPORT_COLUMNS.intersection(headers):
        raise SystemExit(
            f"{source} looks like the reviewer export, not the coach ground-truth table. "
            f"Expected columns exactly: {', '.join(REQUIRED_INPUT_COLUMNS)}"
        )

    raise SystemExit(
        f"{source} has unexpected columns.\n"
        f"Expected exactly: {', '.join(REQUIRED_INPUT_COLUMNS)}\n"
        f"Found: {', '.join(headers)}"
    )


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(round(float(text)))
    except ValueError:
        return None


def _normalise_filename(value: object) -> str:
    return Path(str(value).strip()).stem.casefold()


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return max(index - 1, 0)


def _xml_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext())


def _load_shared_strings(zf: ZipFile) -> list[str]:
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ET.fromstring(raw)
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    return [_xml_text(si) for si in root.findall("x:si", ns)]


def _first_sheet_path(zf: ZipFile) -> str:
    ns_main = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    ns_rel = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    sheets = workbook.find("x:sheets", ns_main)
    if sheets is None or not list(sheets):
        raise SystemExit("Workbook does not contain any sheets.")

    first_sheet = list(sheets)[0]
    relation_id = first_sheet.attrib.get(f"{{{ns_rel['r']}}}id")
    if not relation_id:
        raise SystemExit("Workbook sheet relationship is missing.")

    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_ns = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
    for rel in rels.findall("rel:Relationship", rel_ns):
        if rel.attrib.get("Id") == relation_id:
            target = rel.attrib["Target"].lstrip("/")
            return f"xl/{target}" if not target.startswith("xl/") else target

    raise SystemExit("Could not resolve the first worksheet path from the workbook.")


def _parse_xlsx_rows(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as zf:
        shared_strings = _load_shared_strings(zf)
        sheet_path = _first_sheet_path(zf)
        sheet_root = ET.fromstring(zf.read(sheet_path))

    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    parsed_rows: list[list[str]] = []

    for row in sheet_root.findall(".//x:sheetData/x:row", ns):
        cells: dict[int, str] = {}
        for cell in row.findall("x:c", ns):
            cell_ref = cell.attrib.get("r", "")
            idx = _column_index(cell_ref)
            cell_type = cell.attrib.get("t")
            value = ""

            if cell_type == "inlineStr":
                value = _xml_text(cell.find("x:is", ns))
            else:
                raw_value = cell.find("x:v", ns)
                raw_text = raw_value.text if raw_value is not None else ""
                if cell_type == "s" and raw_text:
                    shared_index = int(raw_text)
                    value = shared_strings[shared_index] if shared_index < len(shared_strings) else ""
                elif cell_type == "b":
                    value = "TRUE" if raw_text == "1" else "FALSE"
                else:
                    value = raw_text or ""

            cells[idx] = value

        if not cells:
            continue

        max_idx = max(cells)
        row_values = [cells.get(i, "") for i in range(max_idx + 1)]
        if any(str(cell).strip() for cell in row_values):
            parsed_rows.append(row_values)

    if not parsed_rows:
        raise SystemExit(f"{path} does not contain any visible worksheet rows.")

    headers = _normalise_header_row(parsed_rows[0])
    _ensure_expected_columns(headers, path)

    records: list[dict[str, str]] = []
    for row_values in parsed_rows[1:]:
        padded = row_values + [""] * max(0, len(headers) - len(row_values))
        record = {header: str(padded[idx]).strip() for idx, header in enumerate(headers)}
        if any(record.values()):
            records.append(record)

    return records


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = _normalise_header_row(reader.fieldnames or [])
        _ensure_expected_columns(headers, path)

        rows: list[dict[str, str]] = []
        for row in reader:
            cleaned = {key: (value.strip() if isinstance(value, str) else "") for key, value in row.items()}
            if any(cleaned.values()):
                rows.append(cleaned)
    return rows


def load_coach_rows(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        rows = _load_csv_rows(path)
    elif suffix == ".xlsx":
        rows = _parse_xlsx_rows(path)
    else:
        raise SystemExit(f"Unsupported coach file type: {path.suffix}. Use .csv or .xlsx")

    if not rows:
        raise SystemExit(f"No coach rows found in {path}")

    return rows


def discover_videos(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_SUFFIXES
    )


def build_video_lookup(videos: list[Path]) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = defaultdict(list)
    for video in videos:
        lookup[_normalise_filename(video.name)].append(video)
    return lookup


def match_video(filename: str, lookup: dict[str, list[Path]]) -> Path | None:
    candidates = lookup.get(_normalise_filename(filename), [])
    if not candidates:
        return None

    desired_name = Path(filename).name.casefold()
    for candidate in candidates:
        if candidate.name.casefold() == desired_name:
            return candidate

    if len(candidates) > 1:
        _warn(
            f"Multiple videos match '{filename}' by stem; using '{candidates[0].name}' "
            f"from {candidates[0].parent}"
        )
    return candidates[0]


def discover_cached_reports(root: Path) -> list[Path]:
    reports: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        json_reports = sorted(path.glob("*_battingiq.json"))
        if json_reports:
            reports.append(json_reports[0])
    return reports


def build_cached_lookup(reports: list[Path]) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = defaultdict(list)
    for report in reports:
        lookup[_normalise_filename(report.parent.name)].append(report)
    return lookup


def match_cached_report(filename: str, lookup: dict[str, list[Path]]) -> Path | None:
    candidates = lookup.get(_normalise_filename(filename), [])
    if not candidates:
        return None

    desired_name = Path(filename).name.casefold()
    for candidate in candidates:
        if candidate.parent.name.casefold() == desired_name:
            return candidate

    if len(candidates) > 1:
        _warn(
            f"Multiple cached reports match '{filename}' by stem; using '{candidates[0].name}' "
            f"from {candidates[0].parent}"
        )
    return candidates[0]


def _nested_get(data: dict, path: tuple[str, ...]) -> object:
    current: object = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_model_scores(report: dict) -> dict[str, int | None]:
    return {field: _coerce_int(_nested_get(report, path)) for field, path in MODEL_FIELDS.items()}


def extract_model_provenance(report: dict) -> dict[str, str]:
    metadata = report.get("metadata", {}) or {}
    contact = report.get("phases", {}).get("contact", {}) or {}
    return {
        "contact_detector_version": str(
            contact.get("detector_version")
            or metadata.get("contact_detector_version")
            or metadata.get("detector_version")
            or ""
        ),
        "anchor_detector_version": str(
            metadata.get("anchor_detector_version")
            or metadata.get("detector_version")
            or ""
        ),
    }


def build_output_row(
    coach_row: dict[str, str],
    model_scores: dict[str, int | None] | None,
    model_provenance: dict[str, str] | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "filename": coach_row["filename"],
        "tier": coach_row["tier"],
        "filmed_correctly": coach_row["filmed_correctly"],
    }

    provenance = model_provenance or {}
    row["contact_detector_version"] = provenance.get("contact_detector_version", "")
    row["anchor_detector_version"] = provenance.get("anchor_detector_version", "")

    for source_field, output_field in COACH_TO_OUTPUT.items():
        row[output_field] = _coerce_int(coach_row.get(source_field))

    if model_scores is None:
        for field in MODEL_FIELDS:
            row[field] = None
    else:
        row.update(model_scores)

    for gap_field, (coach_field, model_field) in GAP_FIELDS.items():
        coach_value = row.get(coach_field)
        model_value = row.get(model_field)
        row[gap_field] = None if coach_value is None or model_value is None else int(coach_value) - int(model_value)

    return row


def write_output_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key)) for key in OUTPUT_COLUMNS})


def _format_mean(values: list[int]) -> str:
    return "" if not values else f"{mean(values):.2f}"


def print_summary(rows: list[dict[str, object]]) -> None:
    groups: dict[str, list[dict[str, object]]] = {"ALL": rows}
    for row in rows:
        groups.setdefault(str(row["tier"]), []).append(row)

    headers = ["tier", "gap_access", "gap_tracking", "gap_stability", "gap_flow", "gap_overall"]
    summary_rows: list[list[str]] = []
    for tier, tier_rows in groups.items():
        summary_rows.append([
            tier,
            _format_mean([int(row["gap_access"]) for row in tier_rows if row.get("gap_access") is not None]),
            _format_mean([int(row["gap_tracking"]) for row in tier_rows if row.get("gap_tracking") is not None]),
            _format_mean([int(row["gap_stability"]) for row in tier_rows if row.get("gap_stability") is not None]),
            _format_mean([int(row["gap_flow"]) for row in tier_rows if row.get("gap_flow") is not None]),
            _format_mean([int(row["gap_overall"]) for row in tier_rows if row.get("gap_overall") is not None]),
        ])

    widths = [len(header) for header in headers]
    for row in summary_rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    print("\nMean gap summary (coach - model):")
    print("  " + "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    for row in summary_rows:
        print("  " + "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare coach scores against local BattingIQ model scores.")
    parser.add_argument("--video-dir", required=True, help="Directory containing batting videos")
    parser.add_argument("--coach-csv", required=True, help="Coach source file (.csv or .xlsx)")
    parser.add_argument(
        "--mode",
        choices=("local", "live", "cached"),
        default="local",
        help="Run the local pipeline, upload each clip to a live BattingIQ API endpoint, or rebuild from cached JSON reports",
    )
    parser.add_argument(
        "--api-base",
        default="https://web-production-e9c26.up.railway.app",
        help="Base URL for live mode, e.g. https://battingiq-api.example.com",
    )
    return parser.parse_args()


def run_local_analysis(video_path: Path, output_dir: Path) -> dict:
    # Force the local, full-quality processing path only when the local runner is used.
    os.environ["LOCAL_MODE"] = "1"

    import pose_extractor
    from run_analysis import run_full_analysis

    pose_extractor.LOCAL_MODEL_COMPLEXITY = 2
    return run_full_analysis(str(video_path), output_dir=str(output_dir))


def load_cached_report(report_path: Path) -> dict:
    import json

    with report_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _build_multipart_request(video_path: Path) -> tuple[bytes, dict[str, str]]:
    boundary = f"----BattingIQBoundary{uuid.uuid4().hex}"
    file_bytes = video_path.read_bytes()
    filename = video_path.name
    suffix = video_path.suffix.casefold()
    content_type = {
        ".mov": "video/quicktime",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".m4v": "video/x-m4v",
        ".webm": "video/webm",
    }.get(suffix, "application/octet-stream")

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
        "User-Agent": "BattingIQ batch_compare/1.0",
    }
    return bytes(body), headers


def run_live_analysis(video_path: Path, api_base: str) -> dict:
    body, headers = _build_multipart_request(video_path)
    endpoint = f"{api_base.rstrip('/')}/analyse"
    request = urlrequest.Request(endpoint, data=body, headers=headers, method="POST")

    try:
        with urlrequest.urlopen(request, timeout=300) as response:
            payload = response.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Live API returned HTTP {exc.code}: {error_body}") from exc
    except urlerror.URLError as exc:
        raise RuntimeError(f"Could not reach live API at {endpoint}: {exc.reason}") from exc

    import json

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Live API returned invalid JSON: {payload[:300]}") from exc


def main() -> int:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    output_path = project_root / "calibration_comparison.csv"
    debug_root = project_root / "calibration_batch_output"

    video_dir = Path(args.video_dir).expanduser()
    coach_path = Path(args.coach_csv).expanduser()

    if not video_dir.exists() or not video_dir.is_dir():
        raise SystemExit(f"Video directory not found: {video_dir}")
    if not coach_path.exists() or not coach_path.is_file():
        raise SystemExit(f"Coach file not found: {coach_path}")

    coach_rows = load_coach_rows(coach_path)
    debug_root.mkdir(parents=True, exist_ok=True)

    video_lookup: dict[str, list[Path]] | None = None
    cached_lookup: dict[str, list[Path]] | None = None

    if args.mode == "cached":
        cached_reports = discover_cached_reports(debug_root)
        if not cached_reports:
            raise SystemExit(f"No cached JSON reports found in {debug_root}")
        cached_lookup = build_cached_lookup(cached_reports)
    else:
        videos = discover_videos(video_dir)
        if not videos:
            raise SystemExit(f"No supported video files found in {video_dir}")
        video_lookup = build_video_lookup(videos)

    output_rows: list[dict[str, object]] = []
    failures = 0
    successes = 0

    for index, coach_row in enumerate(coach_rows, start=1):
        filename = coach_row["filename"]
        print(f"Processing {index}/{len(coach_rows)}: {filename}")
        source_path: Path | None = None

        try:
            if args.mode == "cached":
                assert cached_lookup is not None
                matched_report = match_cached_report(filename, cached_lookup)
                if matched_report is None:
                    failures += 1
                    _warn(f"Cached report not found for '{filename}' under {debug_root}")
                    output_rows.append(build_output_row(coach_row, model_scores=None, model_provenance=None))
                    continue
                source_path = matched_report
                report = load_cached_report(matched_report)
            else:
                assert video_lookup is not None
                matched_video = match_video(filename, video_lookup)
                if matched_video is None:
                    failures += 1
                    _warn(f"Video not found for '{filename}' under {video_dir}")
                    output_rows.append(build_output_row(coach_row, model_scores=None, model_provenance=None))
                    continue
                source_path = matched_video

                if args.mode == "live":
                    report = run_live_analysis(matched_video, args.api_base)
                else:
                    output_dir = debug_root / matched_video.name
                    report = run_local_analysis(matched_video, output_dir)

            output_rows.append(
                build_output_row(
                    coach_row,
                    extract_model_scores(report),
                    extract_model_provenance(report),
                )
            )
            successes += 1
        except Exception as exc:  # pragma: no cover - batch path should continue on failures
            failures += 1
            _warn(f"Analysis failed for '{filename}' ({source_path}): {exc}")
            output_rows.append(build_output_row(coach_row, model_scores=None, model_provenance=None))
            continue

    if successes == 0:
        raise SystemExit(
            "All analyses failed. Existing calibration_comparison.csv was left untouched "
            "so a valid comparison file is not replaced with blanks."
        )

    write_output_csv(output_rows, output_path)

    print(f"\nSaved calibration comparison CSV to: {output_path}")
    print(f"Rows written: {len(output_rows)}")
    print(f"Successful analyses: {successes}")
    print(f"Rows with missing video or failed analysis: {failures}")
    print_summary(output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
