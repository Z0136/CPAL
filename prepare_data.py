"""Download or import the seven CAPL datasets, validate them, and prepare CSVs.

Only pandas and numpy are required. Run with --help for offline and online options.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))


def verify_source(path, spec):
    """Reject a changed benchmark, incomplete download, or HTML error page."""
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != spec["source_sha256"]:
        raise ValueError(
            f"Source SHA-256 mismatch: {path}\n"
            f"Expected {spec['source_sha256']}, got {actual}. "
            "Use the download linked in README.md; source CSVs must be unmodified."
        )


def validate_frame(frame, spec):
    """Validate the original benchmark layout without repairing or resampling it."""
    if list(frame.columns) != spec["columns"]:
        raise ValueError(f"Unexpected columns; expected {spec['columns']}")
    if len(frame) != spec["rows"]:
        raise ValueError(f"Expected {spec['rows']} rows, found {len(frame)}")
    if frame.isna().any().any():
        raise ValueError("Missing values found; no imputation is performed")
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numeric values found")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if str(dates.iloc[0]) != spec["start"] or str(dates.iloc[-1]) != spec["end"]:
        raise ValueError("Timestamp range differs from the reference benchmark")
    if not dates.is_monotonic_increasing:
        raise ValueError("Timestamps must remain in their original chronological order")
    intervals = {
        str(int(delta.total_seconds())): int(count)
        for delta, count in dates.diff().value_counts().items()
    }
    if intervals != spec["interval_counts_seconds"]:
        raise ValueError("Timestamp intervals differ from the reference benchmark")


def validate_output(path, name):
    spec = MANIFEST[name]
    if name != "Appliances":
        verify_source(path, spec)
    frame = pd.read_csv(path)
    validate_frame(frame, spec)
    return frame


def locate_source(source_dir, spec):
    preferred = source_dir / spec["source_path"]
    if preferred.is_file():
        return preferred
    # Accept a flat downloads directory or an extracted benchmark archive.
    matches = sorted(source_dir.rglob(Path(spec["source_path"]).name))
    matches = [path for path in matches if path.is_file()]
    if len(matches) > 1:
        raise ValueError(f"Multiple source files found: {matches}. Use a narrower --source_dir.")
    return matches[0] if matches else None


def download_source(source_dir, name, timeout):
    spec = MANIFEST[name]
    destination = source_dir / spec["source_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name}: {spec['url']}", flush=True)
    # Stage on the destination filesystem; only publish complete, verified files.
    with tempfile.TemporaryDirectory(prefix=".capl-download-", dir=destination.parent) as tmp:
        payload = Path(tmp) / "download"
        request = Request(spec["url"], headers={"User-Agent": "CAPL-data-preparation/1.0"})
        with urlopen(request, timeout=timeout) as response, payload.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        source = payload
        if name == "Appliances":
            source = Path(tmp) / "energydata_complete.csv"
            with ZipFile(payload) as archive:
                # Read only the known member; never extract arbitrary archive paths.
                with archive.open("energydata_complete.csv") as member, source.open("wb") as stream:
                    shutil.copyfileobj(member, stream)
        verify_source(source, spec)
        source.replace(destination)
    return destination


def prepare_appliances(source, output):
    """Remove only rv1/rv2, preserving all observations and the 10-minute grid."""
    spec = MANIFEST["Appliances"]
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Appliances source and output paths must be different")
    verify_source(source, spec)
    frame = pd.read_csv(source)
    if list(frame.columns) != spec["columns"] + ["rv1", "rv2"]:
        raise ValueError("Expected the original UCI schema, including rv1 and rv2")
    frame = frame.drop(columns=["rv1", "rv2"])
    validate_frame(frame, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".capl-prepare-", dir=output.parent) as tmp:
        staged = Path(tmp) / output.name
        frame.to_csv(staged, index=False, lineterminator="\n")
        validate_output(staged, "Appliances")
        staged.replace(output)
    return frame


def prepare_dataset(name, source_dir, data_dir, download=False, force=False, timeout=60):
    spec = MANIFEST[name]
    output = data_dir / spec["output_path"]
    if output.exists() and not force:
        validate_output(output, name)
        print(f"Verified existing {name}: {output}")
        return
    source = locate_source(source_dir, spec)
    if source is None:
        if not download:
            raise FileNotFoundError(
                f"Missing source for {name}: {source_dir / spec['source_path']}. "
                "Use --download or --source_dir with files from README.md."
            )
        source = download_source(source_dir, name, timeout)
    if name == "Appliances":
        prepare_appliances(source, output)
    else:
        validate_output(source, name)
        output.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != output.resolve():
            with tempfile.TemporaryDirectory(prefix=".capl-prepare-", dir=output.parent) as tmp:
                staged = Path(tmp) / output.name
                shutil.copyfile(source, staged)
                staged.replace(output)
    print(f"Prepared {name}: {spec['rows']} rows, {len(spec['columns']) - 1} channels -> {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *MANIFEST], default="all")
    parser.add_argument("--source_dir", type=Path, default=ROOT / "raw_data",
                        help="Directory containing manually downloaded or cached source files")
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data",
                        help="Output directory consumed by run.py --data_dir")
    parser.add_argument("--download", action="store_true", help="Download missing source files")
    parser.add_argument("--check", action="store_true", help="Validate prepared CSVs without writing or downloading")
    parser.add_argument("--force", action="store_true", help="Regenerate existing output CSVs from validated sources")
    parser.add_argument("--timeout", type=int, default=60, help="Network socket timeout in seconds")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.check and (args.download or args.force):
        parser.error("--check cannot be combined with --download or --force")
    names = list(MANIFEST) if args.dataset == "all" else [args.dataset]
    try:
        for name in names:
            if args.check:
                output = args.data_dir.resolve() / MANIFEST[name]["output_path"]
                validate_output(output, name)
                print(f"Verified {name}: {output}")
            else:
                prepare_dataset(name, args.source_dir.resolve(), args.data_dir.resolve(),
                                args.download, args.force, args.timeout)
    except (OSError, ValueError, URLError, BadZipFile, KeyError) as error:
        parser.exit(1, f"Data preparation failed for {name}: {error}\n"
                    "See README.md for manual downloads. Existing outputs are reused after validation; "
                    "use --force to regenerate them.\n")


if __name__ == "__main__":
    main()
