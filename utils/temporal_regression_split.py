"""Immutable, leakage-safe preparation for TDC BindingDB_Patent regression."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


STANDARD_COLUMNS = ("SMILES", "Protein", "Y", "drug_id", "pr_id", "Year")
HISTORICAL_MAX_YEAR = 2018
TEMPORAL_MIN_YEAR = 2019


@dataclass(frozen=True)
class TemporalRegressionPaths:
    root: Path
    train: Path
    validation: Path
    test: Path

    @property
    def feature_csvs(self):
        """Feature extraction may see sequences but never labels from test rows."""
        return (self.train, self.validation, self.test)


def temporal_regression_paths(root: Path) -> TemporalRegressionPaths:
    root = Path(root)
    paths = TemporalRegressionPaths(
        root=root,
        train=root / "train.csv",
        validation=root / "validation.csv",
        test=root / "test.csv",
    )
    missing = [str(path) for path in (paths.train, paths.validation, paths.test) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Temporal regression split is incomplete: " + ", ".join(missing))
    return paths


def _sha256(prefix: str, value: object) -> str:
    normalized = "".join(str(value).strip().split())
    if not normalized:
        raise ValueError("TDC identifier fallback cannot be empty.")
    return prefix + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _standardize(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    required = {"Drug", "Target", "Y", "Year"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("%s is missing required TDC columns: %s" % (source_name, missing))
    result = pd.DataFrame({
        "SMILES": frame["Drug"].astype(str).str.strip(),
        "Protein": frame["Target"].astype(str).str.strip(),
        "Y": pd.to_numeric(frame["Y"], errors="raise").astype(float),
        "Year": pd.to_numeric(frame["Year"], errors="raise").astype(int),
    })
    if (result["SMILES"] == "").any() or (result["Protein"] == "").any():
        raise ValueError("%s contains empty SMILES or protein sequence." % source_name)
    if not np.isfinite(result["Y"].to_numpy(dtype=np.float64)).all():
        raise ValueError("%s contains non-finite affinity labels." % source_name)

    drug_ids = frame["Drug_ID"] if "Drug_ID" in frame.columns else pd.Series([None] * len(frame))
    target_ids = frame["Target_ID"] if "Target_ID" in frame.columns else pd.Series([None] * len(frame))
    result["drug_id"] = [
        str(value).strip() if pd.notna(value) and str(value).strip() else _sha256("TDC_DRUG_", smiles)
        for value, smiles in zip(drug_ids, result["SMILES"])
    ]
    result["pr_id"] = [
        str(value).strip() if pd.notna(value) and str(value).strip() else _sha256("TDC_PROTEIN_", protein)
        for value, protein in zip(target_ids, result["Protein"])
    ]
    return result.loc[:, STANDARD_COLUMNS]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: Dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_tdc_bindingdb_patent(
    raw_root: Path,
    output_root: Path,
    seed: int = 6,
    validation_fraction: float = 0.2,
) -> Tuple[TemporalRegressionPaths, Dict[str, object]]:
    """Create deterministic train/validation/test CSVs without reading test for selection."""
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    historical_path = raw_root / "train_val_official.csv"
    test_path = raw_root / "test_official.csv"
    if not historical_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Expected train_val_official.csv and test_official.csv in %s" % raw_root)

    historical = _standardize(pd.read_csv(historical_path), "historical official CSV")
    temporal_test = _standardize(pd.read_csv(test_path), "temporal test official CSV")
    if (historical["Year"] > HISTORICAL_MAX_YEAR).any():
        raise ValueError("Historical source contains rows after %d" % HISTORICAL_MAX_YEAR)
    if (temporal_test["Year"] < TEMPORAL_MIN_YEAR).any():
        raise ValueError("Temporal test source contains rows before %d" % TEMPORAL_MIN_YEAR)

    order = np.random.RandomState(int(seed)).permutation(len(historical))
    validation_count = max(1, int(round(len(historical) * float(validation_fraction))))
    validation_indices = set(order[:validation_count].tolist())
    validation = historical.loc[historical.index.isin(validation_indices)].reset_index(drop=True)
    train = historical.loc[~historical.index.isin(validation_indices)].reset_index(drop=True)
    if train.empty or validation.empty or temporal_test.empty:
        raise ValueError("Temporal regression split must have non-empty train, validation, and test sets.")

    output_root.mkdir(parents=True, exist_ok=True)
    paths = TemporalRegressionPaths(output_root, output_root / "train.csv", output_root / "validation.csv", output_root / "test.csv")
    _atomic_csv(train, paths.train)
    _atomic_csv(validation, paths.validation)
    _atomic_csv(temporal_test, paths.test)
    audit = {
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "historical_rows": int(len(historical)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(temporal_test)),
        "historical_year_min": int(historical["Year"].min()),
        "historical_year_max": int(historical["Year"].max()),
        "test_year_min": int(temporal_test["Year"].min()),
        "test_year_max": int(temporal_test["Year"].max()),
        "target_test_used_for_split": False,
        "raw_historical_sha256": _file_sha256(historical_path),
        "raw_test_sha256": _file_sha256(test_path),
    }
    _atomic_json(audit, output_root / "audit.json")
    return temporal_regression_paths(output_root), audit
