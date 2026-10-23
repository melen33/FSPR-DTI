"""Preparation and validation for the BindingDB temporal binary benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


STANDARD_COLUMNS = ("SMILES", "Protein", "Y", "drug_id", "pr_id")
PRIMARY_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class TemporalBinaryPaths:
    root: Path
    train: Path
    validation: Path
    test: Path
    prospective_2026: Path | None = None

    @property
    def feature_csvs(self):
        """Sequences from all evaluation splits are available at inference time."""
        primary = (self.train, self.validation, self.test)
        # Frozen ESM embeddings are inference-time representations, so include
        # the held-out 2026 panel when it exists. This prevents a later
        # prospective evaluation from failing merely because its proteins were
        # not cached during the main temporal split preparation.
        if self.prospective_2026 is not None and self.prospective_2026.is_file():
            return primary + (self.prospective_2026,)
        return primary


def temporal_binary_paths(root: Path) -> TemporalBinaryPaths:
    root = Path(root)
    paths = TemporalBinaryPaths(
        root=root,
        train=root / "train.csv",
        validation=root / "validation.csv",
        test=root / "test.csv",
        prospective_2026=(root / "prospective_2026.csv"),
    )
    missing = [str(path) for path in paths.feature_csvs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Temporal binary split is incomplete: " + ", ".join(missing))
    return paths


def _stable_id(prefix: str, value: object) -> str:
    normalized = "".join(str(value).strip().split())
    if not normalized:
        raise ValueError("Temporal binary identifiers cannot be empty.")
    return prefix + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _standardize(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    required = {"SMILES", "Protein", "Y"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("%s is missing required columns: %s" % (source_name, missing))
    result = pd.DataFrame({
        "SMILES": frame["SMILES"].astype(str).str.strip(),
        "Protein": frame["Protein"].astype(str).str.replace(r"\s+", "", regex=True),
        "Y": pd.to_numeric(frame["Y"], errors="raise").astype(int),
    })
    if (result["SMILES"] == "").any() or (result["Protein"] == "").any():
        raise ValueError("%s contains empty SMILES or protein sequences." % source_name)
    if not result["Y"].isin((0, 1)).all():
        raise ValueError("%s must contain binary labels {0, 1}." % source_name)
    result["drug_id"] = [_stable_id("TEMPORAL_BINARY_DRUG_", value) for value in result["SMILES"]]
    result["pr_id"] = [_stable_id("TEMPORAL_BINARY_PROTEIN_", value) for value in result["Protein"]]
    return result.loc[:, STANDARD_COLUMNS]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: Dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def materialize_temporal_binary_assets(
    raw_root: Path, output_root: Path
) -> Tuple[TemporalBinaryPaths, Dict[str, object]]:
    """Standardize existing temporal CSVs without altering source split membership."""
    raw_root, output_root = Path(raw_root), Path(output_root)
    raw_paths = {name: raw_root / (name + ".csv") for name in PRIMARY_SPLITS}
    missing = [str(path) for path in raw_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Temporal binary source split is incomplete: " + ", ".join(missing))
    split_audit = {}
    for name, path in raw_paths.items():
        frame = _standardize(pd.read_csv(path), name)
        _atomic_csv(frame, output_root / (name + ".csv"))
        split_audit[name] = {
            "rows": int(len(frame)),
            "positive": int(frame.Y.sum()),
            "negative": int((frame.Y == 0).sum()),
            "unique_proteins": int(frame.pr_id.nunique()),
            "unique_drugs": int(frame.drug_id.nunique()),
        }
        del frame
    prospective_source = raw_root / "prospective_2026.csv"
    if prospective_source.is_file():
        prospective = _standardize(pd.read_csv(prospective_source), "prospective_2026")
        _atomic_csv(prospective, output_root / "prospective_2026.csv")
    paths = temporal_binary_paths(output_root)
    audit = {
        "protocol": "bindingdb_temporal_binary_core_v1",
        "source_root": str(raw_root),
        "splits": split_audit,
    }
    _atomic_json(audit, output_root / "asset_audit.json")
    return paths, audit
