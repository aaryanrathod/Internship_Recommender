"""
CLI script to build the FAISS internship index from a CSV or JSON dataset.

Reads internship listings, validates each against the
:class:`~api.schemas.InternshipListing` Pydantic model, embeds them using
:class:`~engine.embedder.Embedder`, and persists the resulting FAISS index
to disk.

Usage::

    python scripts/build_index.py --input data/internships_batch1.csv
    python scripts/build_index.py --input data/internships.json --output data/faiss.index
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

from loguru import logger
from pydantic import ValidationError

# Ensure project root is on sys.path so that package imports work when
# running the script directly with ``python scripts/build_index.py``.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api.schemas import InternshipListing
from engine.embedder import get_embedder
from engine.indexer import InternshipIndex


# ── Required CSV columns ─────────────────────────────────────────────────────

_REQUIRED_COLUMNS = {"internship_id", "title", "company"}

_EXPECTED_COLUMNS = (
    _REQUIRED_COLUMNS
    | {
        "location",
        "country",
        "location_type",
        "description",
        "required_skills",
        "preferred_skills",
        "domain",
        "duration_months",
        "stipend_usd",
        "experience_level",
    }
)


# ── Data loaders ─────────────────────────────────────────────────────────────


def _load_csv(path: Path) -> list[dict]:
    """Read rows from a CSV file and return a list of dicts.

    Args:
        path: Absolute path to the CSV file.

    Returns:
        List of row dictionaries (keys = column headers).
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)

        # ── Column validation ────────────────────────────────────────────
        if reader.fieldnames is None:
            logger.error("CSV file has no header row.")
            sys.exit(1)

        actual = set(reader.fieldnames)
        missing = _REQUIRED_COLUMNS - actual
        if missing:
            logger.error(
                "CSV is missing REQUIRED columns: {}. Found columns: {}",
                sorted(missing),
                sorted(actual),
            )
            sys.exit(1)

        extra_expected = _EXPECTED_COLUMNS - actual
        if extra_expected:
            logger.info(
                "CSV does not contain optional columns (will use defaults): {}",
                sorted(extra_expected),
            )

        return list(reader)


def _load_json(path: Path) -> list[dict]:
    """Read an array of objects from a JSON file.

    Args:
        path: Absolute path to the JSON file.

    Returns:
        List of row dictionaries.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a top-level array of objects.")
    return data


def _load_data(path: Path) -> list[dict]:
    """Dispatch to the correct loader based on file extension.

    Args:
        path: Path to the input data file (.csv or .json).

    Returns:
        List of raw row dictionaries.

    Raises:
        ValueError: If the extension is unsupported.
    """
    ext = path.suffix.lower()
    if ext == ".csv":
        return _load_csv(path)
    elif ext == ".json":
        return _load_json(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .csv or .json.")


# ── Validation ───────────────────────────────────────────────────────────────


def _parse_skills_field(raw: str | list) -> list[str]:
    """Normalise a skills field from raw input.

    Handles a Python list (from JSON), pipe-separated strings (new CSV
    format, e.g. ``"Python|SQL|TensorFlow"``), and comma-separated strings
    (legacy CSV format).

    Args:
        raw: The raw skills value.

    Returns:
        List of skill strings.
    """
    if isinstance(raw, list):
        return [s.strip() for s in raw if s and str(s).strip()]
    if isinstance(raw, str) and raw.strip():
        # Pipe-separated takes precedence (new format)
        sep = "|" if "|" in raw else ","
        return [s.strip() for s in raw.split(sep) if s.strip()]
    return []


def _validate_rows(rows: list[dict]) -> list[InternshipListing]:
    """Validate each raw row into an :class:`InternshipListing`.

    Malformed rows are logged as warnings and skipped.

    Args:
        rows: List of raw dictionaries from the data file.

    Returns:
        List of validated listings.
    """
    valid: list[InternshipListing] = []
    skipped = 0

    for i, row in enumerate(rows, start=1):
        try:
            # Normalise skills fields if present
            if "required_skills" in row:
                row["required_skills"] = _parse_skills_field(row["required_skills"])
            if "preferred_skills" in row:
                row["preferred_skills"] = _parse_skills_field(row["preferred_skills"])

            # Convert numeric fields from CSV strings
            if "duration_months" in row and isinstance(row["duration_months"], str):
                row["duration_months"] = int(row["duration_months"]) if row["duration_months"].strip() else None
            if "stipend_usd" in row and isinstance(row["stipend_usd"], str):
                row["stipend_usd"] = int(row["stipend_usd"]) if row["stipend_usd"].strip() else None

            listing = InternshipListing(**row)
            valid.append(listing)
        except (ValidationError, TypeError, ValueError) as exc:
            logger.warning("Row {}: skipped — {}", i, exc)
            skipped += 1

    logger.info(
        "Validation complete: {} valid, {} skipped out of {} total rows.",
        len(valid),
        skipped,
        len(rows),
    )
    return valid


# ── Summary helpers ──────────────────────────────────────────────────────────


def _print_breakdown(listings: list[InternshipListing]) -> None:
    """Print dataset breakdowns by domain and location_type."""
    # Domain breakdown
    domain_counts = Counter(
        getattr(l, "domain", "Unknown") or "Unknown" for l in listings
    )
    loc_type_counts = Counter(
        getattr(l, "location_type", "Unknown") or "Unknown" for l in listings
    )
    exp_counts = Counter(
        getattr(l, "experience_level", "Unknown") or "Unknown" for l in listings
    )

    print()
    print("  Domain breakdown:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        bar = "#" * count
        print(f"    {domain:<25s} {count:>3d}  {bar}")

    print()
    print("  Location type breakdown:")
    for lt, count in sorted(loc_type_counts.items(), key=lambda x: -x[1]):
        pct = count / len(listings) * 100
        print(f"    {lt:<12s} {count:>3d}  ({pct:.0f}%)")

    print()
    print("  Experience level breakdown:")
    for el, count in sorted(exp_counts.items(), key=lambda x: -x[1]):
        pct = count / len(listings) * 100
        print(f"    {el:<15s} {count:>3d}  ({pct:.0f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────


def main(input_path: str, output_path: str | None = None) -> None:
    """End-to-end index build pipeline.

    Args:
        input_path: Path to the internship data file (CSV or JSON).
        output_path: Destination for the FAISS index.  Defaults to
            ``Settings.FAISS_INDEX_PATH``.
    """
    t_start = time.perf_counter()

    # ── Resolve output path ──────────────────────────────────────────────
    if output_path is None:
        try:
            from config import settings
            output_path = settings.FAISS_INDEX_PATH
        except Exception:
            output_path = "data/faiss.index"
            logger.warning(
                "Could not read FAISS_INDEX_PATH from Settings; "
                "defaulting to '{}'.",
                output_path,
            )

    # ── Validate input path ──────────────────────────────────────────────
    src = Path(input_path).resolve()
    if not src.exists():
        logger.error("Input file not found: {}", src)
        sys.exit(1)
    if not src.is_file():
        logger.error("Input path is not a file: {}", src)
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────
    logger.info("Loading data from '{}' …", src)
    rows = _load_data(src)
    logger.info("Read {} raw rows.", len(rows))

    # ── Validate ─────────────────────────────────────────────────────────
    listings = _validate_rows(rows)
    if not listings:
        logger.error("No valid internship listings to index. Aborting.")
        sys.exit(1)

    # ── Embed & build index ──────────────────────────────────────────────
    logger.info("Initialising embedder …")
    embedder = get_embedder()

    index = InternshipIndex()
    index.build(listings, embedder)

    # ── Save ─────────────────────────────────────────────────────────────
    index.save(output_path)

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    idx_size_kb = Path(output_path).stat().st_size / 1024

    print()
    print("=" * 60)
    print("  FAISS Index Build — Summary")
    print("=" * 60)
    print(f"  Input file       : {src}")
    print(f"  Internships      : {len(listings)}")
    print(f"  Index dimensions : {embedder.embedding_dim}")
    print(f"  Index file       : {output_path}")
    print(f"  Index size       : {idx_size_kb:.1f} KB")
    print(f"  Total time       : {elapsed:.2f}s")
    sw_status = "YES (built & saved)" if index.skill_weighter else "NO (not available)"
    print(f"  SkillWeighter    : {sw_status}")
    print("=" * 60)

    _print_breakdown(listings)

    print()


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the FAISS internship index from a CSV or JSON file.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the internship data file (.csv or .json).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Destination path for the FAISS index. "
            "Defaults to Settings.FAISS_INDEX_PATH."
        ),
    )

    args = parser.parse_args()
    main(input_path=args.input, output_path=args.output)
