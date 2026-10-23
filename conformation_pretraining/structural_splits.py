"""Deterministic protein-group partitions for structural supervision."""

from __future__ import annotations

import hashlib

import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["sample_id"].isna().any() or frame["protein_id"].isna().any():
        raise ValueError("sample_id and protein_id must be non-null")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id must be unique")


def _ordered_proteins(proteins, seed: int, namespace: str) -> list[str]:
    return sorted(
        (str(protein) for protein in proteins),
        key=lambda protein: hashlib.sha256(
            f"{namespace}|{int(seed)}|{protein}".encode("utf-8")
        ).hexdigest(),
    )


def select_structural_source_rows(
    frame: pd.DataFrame, max_real_residues: int
) -> pd.DataFrame:
    """Keep only precomputed feature rows usable by TAPB's protein window."""
    _require_columns(
        frame,
        {"sample_id", "protein_id", "feature_eligible", "original_length"},
    )
    if int(max_real_residues) <= 0:
        raise ValueError("max_real_residues must be positive")
    length = pd.to_numeric(frame["original_length"], errors="raise")
    if (length <= 0).any():
        raise ValueError("original_length must be positive")
    eligible = frame["feature_eligible"].map(
        lambda value: value is True or str(value).strip().lower() in {"1", "true", "yes"}
    )
    return frame.loc[eligible & (length <= int(max_real_residues))].copy().reset_index(drop=True)


def build_protein_group_folds(
    frame: pd.DataFrame, n_splits: int, seed: int
) -> pd.DataFrame:
    """Assign every protein, and all of its samples, to one 1-based fold."""
    _require_columns(frame, {"sample_id", "protein_id"})
    if int(n_splits) < 2:
        raise ValueError("n_splits must be at least 2")
    proteins = _ordered_proteins(frame["protein_id"].unique(), seed, "fold")
    if len(proteins) < int(n_splits):
        raise ValueError("n_splits cannot exceed the number of proteins")
    mapping = {
        protein: index % int(n_splits) + 1
        for index, protein in enumerate(proteins)
    }
    result = frame.copy()
    result["protein_id"] = result["protein_id"].astype(str)
    result["structural_fold"] = result["protein_id"].map(mapping).astype(int)
    return result


def build_pilot_partitions(
    folded: pd.DataFrame,
    heldout_fold: int,
    validation_fraction: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Return complete, protein-disjoint train/valid/test frames from folds."""
    _require_columns(folded, {"sample_id", "protein_id", "structural_fold"})
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    folds = pd.to_numeric(folded["structural_fold"], errors="raise")
    if not (folds > 0).all() or not (folds == folds.astype(int)).all():
        raise ValueError("structural_fold must contain positive integers")
    if int(heldout_fold) not in set(folds.astype(int)):
        raise ValueError("heldout_fold is absent from structural_fold")

    result = folded.copy()
    result["protein_id"] = result["protein_id"].astype(str)
    result["structural_fold"] = folds.astype(int)
    test = result.loc[result["structural_fold"] == int(heldout_fold)].copy()
    remaining = result.loc[result["structural_fold"] != int(heldout_fold)].copy()
    proteins = _ordered_proteins(remaining["protein_id"].unique(), seed, "validation")
    if len(proteins) < 2:
        raise ValueError("At least two non-test proteins are required")
    valid_count = int(round(len(proteins) * float(validation_fraction)))
    valid_count = min(max(valid_count, 1), len(proteins) - 1)
    valid_proteins = set(proteins[:valid_count])
    valid = remaining.loc[remaining["protein_id"].isin(valid_proteins)].copy()
    train = remaining.loc[~remaining["protein_id"].isin(valid_proteins)].copy()
    return {
        "train": train.reset_index(drop=True),
        "valid": valid.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
