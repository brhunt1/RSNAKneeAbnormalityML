"""DICOM loading and series selection.

The competition mixes transfer syntaxes: uncompressed Explicit VR Little Endian,
JPEG Lossless, JPEG 2000, and Implicit VR Little Endian. pydicom alone cannot decode
the compressed ones. Install the handlers listed in requirements.txt or a portion of
the data will fail to load, quietly, and you will not notice until your CV is bad.

Run `check_decoders()` once before doing anything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import SERIES_COL


def check_decoders() -> dict[str, bool]:
    """Report which optional DICOM decoders are importable."""
    status = {}
    for mod in ("pylibjpeg", "libjpeg", "openjpeg", "gdcm"):
        try:
            __import__(mod)
            status[mod] = True
        except ImportError:
            status[mod] = False
    return status


def audit_decoding(series_dir: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Attempt to decode every slice in a directory tree. Returns failures.

    Point this at a random sample of train_series/ in Week 1 and confirm the failure
    count is zero before building anything on top of the loader.
    """
    import pydicom

    series_dir = Path(series_dir)
    files = sorted(series_dir.rglob("*.dcm"))
    if limit:
        files = files[:limit]

    rows = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f))
            syntax = str(getattr(ds.file_meta, "TransferSyntaxUID", "unknown"))
            _ = ds.pixel_array
            rows.append({"path": str(f), "syntax": syntax, "ok": True, "error": ""})
        except Exception as exc:  # noqa: BLE001 - we want every failure mode
            rows.append(
                {"path": str(f), "syntax": "unknown", "ok": False, "error": repr(exc)}
            )

    return pd.DataFrame(rows)


def load_slice(path: str | Path) -> np.ndarray:
    """Load one DICOM slice as a float32 array, rescale applied if present."""
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept

    return arr


def load_series(series_dir: str | Path) -> np.ndarray:
    """Load a full series as (n_slices, H, W), ordered by position where possible.

    Slice ordering is by ImagePositionPatient along the slice axis when available,
    falling back to InstanceNumber, then filename. Ordering matters for 2.5D input:
    a shuffled stack destroys the spatial context you are trying to give the model.
    """
    import pydicom

    series_dir = Path(series_dir)
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no .dcm files in {series_dir}")

    entries = []
    for f in files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        ipp = getattr(ds, "ImagePositionPatient", None)
        inst = getattr(ds, "InstanceNumber", None)
        key = float(ipp[2]) if ipp is not None else (float(inst) if inst is not None else 0.0)
        entries.append((key, f))

    entries.sort(key=lambda t: t[0])
    slices = [load_slice(f) for _, f in entries]

    shapes = {s.shape for s in slices}
    if len(shapes) > 1:
        raise ValueError(f"inconsistent slice shapes in {series_dir}: {shapes}")

    return np.stack(slices)


def normalize_series(vol: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Per-series percentile normalization to [0, 1].

    Per-series, not per-slice and not global. Intensities vary wildly across scanners
    and protocols in this dataset, and a global normalization would let scanner
    identity leak into the model as a shortcut feature.
    """
    lo, hi = np.percentile(vol, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    out = (vol - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def select_series(
    series_df: pd.DataFrame,
    study_id: str,
    plane: str,
    fluid_sensitive: int | None = None,
    fat_suppression: int | None = None,
) -> list[str]:
    """Return SeriesInstanceUIDs for a study matching the given criteria.

    Series selection rule v1. Uses only the metadata in train_series.csv. Refine this
    once EDA tells you what combinations actually exist per study.
    """
    from src.constants import ID_COL

    m = (series_df[ID_COL] == study_id) & (series_df["Anatomical_Plane"] == plane)
    if fluid_sensitive is not None:
        m &= series_df["Fluid_Sensitive"] == fluid_sensitive
    if fat_suppression is not None:
        m &= series_df["Fat_Suppression"] == fat_suppression

    return series_df.loc[m, SERIES_COL].tolist()
