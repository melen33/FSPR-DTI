"""Exact downstream-entity exclusion and TAPB-window eligibility checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem


@dataclass(frozen=True)
class EntityUnion:
    protein_hashes: frozenset[str]
    ligand_inchikeys: frozenset[str]


@dataclass(frozen=True)
class FilterResult:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    all_rows: pd.DataFrame
    audit: dict


def _clean_sequence(sequence: str) -> str:
    return "".join(str(sequence).split()).upper()


def protein_sha256(sequence: str) -> str:
    clean = _clean_sequence(sequence)
    if not clean:
        raise ValueError("Empty protein sequence")
    try:
        encoded = clean.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Protein sequence must contain ASCII characters") from exc
    return hashlib.sha256(encoded).hexdigest()


def canonical_inchikey(smiles: str) -> str:
    text = str(smiles).strip()
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"Invalid downstream SMILES: {smiles}")
    return Chem.MolToInchiKey(mol)


def build_downstream_entity_union(paths: Sequence[Path]) -> EntityUnion:
    protein_hashes: set[str] = set()
    ligand_inchikeys: set[str] = set()
    if not paths:
        raise ValueError("At least one downstream CSV is required")
    for raw_path in paths:
        path = Path(raw_path)
        frame = pd.read_csv(path)
        missing = {"Protein", "SMILES"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for index, value in frame["Protein"].items():
            try:
                protein_hashes.add(protein_sha256(value))
            except ValueError as exc:
                raise ValueError(f"Invalid protein in {path} row {index}: {exc}") from exc
        for index, value in frame["SMILES"].items():
            try:
                ligand_inchikeys.add(canonical_inchikey(value))
            except ValueError as exc:
                raise ValueError(f"Invalid downstream SMILES in {path} row {index}: {value}") from exc
    return EntityUnion(frozenset(protein_hashes), frozenset(ligand_inchikeys))


def _is_true(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _validate_label(path: Path, expected_length: int, max_real_residues: int) -> tuple[int, str]:
    try:
        with np.load(path, allow_pickle=False) as labels:
            required = ("ca_displacement", "valid_mask", "pocket_mask")
            missing = [name for name in required if name not in labels.files]
            if missing:
                return 0, "LABEL_MISSING_ARRAY"
            lengths = {int(np.asarray(labels[name]).shape[0]) for name in required}
            for name in labels.files:
                array = np.asarray(labels[name])
                if array.ndim > 0:
                    lengths.add(int(array.shape[0]))
            if lengths != {expected_length}:
                return 0, "LABEL_LENGTH_MISMATCH"
            valid = np.asarray(labels["valid_mask"], dtype=bool)
            pocket = np.asarray(labels["pocket_mask"], dtype=bool)
            count = int(np.logical_and(valid, pocket)[:max_real_residues].sum())
            return count, "" if count >= 5 else "INSUFFICIENT_WINDOWED_POCKET"
    except (OSError, ValueError, KeyError):
        return 0, "LABEL_LOAD_ERROR"


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[0:0].copy().reset_index(drop=True)


def filter_apo_manifest(
    apo_manifest_path: Path,
    downstream_paths: Sequence[Path],
    apo_root: Path,
    max_real_residues: int = 510,
) -> FilterResult:
    if max_real_residues <= 0:
        raise ValueError("max_real_residues must be positive")
    entity_union = build_downstream_entity_union(downstream_paths)
    manifest = pd.read_csv(apo_manifest_path)
    required = {
        "sample_id", "official_split", "protein_sequence", "ligand_smiles",
        "label_path", "num_sequence_residues", "quality_pass",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Apo manifest is missing columns: {sorted(missing)}")

    decisions: list[dict] = []
    for _, source in manifest.iterrows():
        row = source.to_dict()
        sequence = _clean_sequence(row["protein_sequence"])
        row["Protein"] = sequence
        row["SMILES"] = str(row["ligand_smiles"]).strip()
        row["protein_id"] = protein_sha256(sequence)
        row["drug_id"] = str(row.get("ligand_inchikey", "") or "").strip()
        if not row["drug_id"] or row["drug_id"].lower() == "nan":
            row["drug_id"] = canonical_inchikey(row["SMILES"])
        row["original_length"] = len(sequence)
        row["window_length"] = min(len(sequence), max_real_residues)
        row["protein_overlap"] = row["protein_id"] in entity_union.protein_hashes
        row["drug_overlap"] = row["drug_id"] in entity_union.ligand_inchikeys
        row["window_valid_pocket_count"] = 0
        reason = ""
        if not _is_true(row["quality_pass"]):
            reason = "QUALITY_FAILED"
        elif row["protein_overlap"] and row["drug_overlap"]:
            reason = "DOWNSTREAM_EXACT_BOTH"
        elif row["protein_overlap"]:
            reason = "DOWNSTREAM_EXACT_PROTEIN"
        elif row["drug_overlap"]:
            reason = "DOWNSTREAM_EXACT_DRUG"
        elif int(row["num_sequence_residues"]) != len(sequence):
            reason = "MANIFEST_LENGTH_MISMATCH"
        else:
            label_path = Path(row["label_path"])
            if not label_path.is_absolute():
                label_path = Path(apo_root) / label_path
            count, reason = _validate_label(label_path, len(sequence), max_real_residues)
            row["window_valid_pocket_count"] = count
        row["eligibility_reason"] = reason
        row["eligible"] = reason == ""
        decisions.append(row)

    all_rows = pd.DataFrame(decisions)
    quality = all_rows["quality_pass"].map(_is_true)
    protein_hit = quality & all_rows["protein_overlap"]
    drug_hit = quality & all_rows["drug_overlap"]
    both_hit = protein_hit & drug_hit
    union_hit = protein_hit | drug_hit
    pre_window = quality & ~union_hit
    eligible = all_rows["eligible"]

    split_order = ("train", "valid", "test")
    pre_counts = {split: int((pre_window & all_rows["official_split"].eq(split)).sum()) for split in split_order}
    post_counts = {split: int((eligible & all_rows["official_split"].eq(split)).sum()) for split in split_order}
    audit = {
        "manifest_rows": int(len(all_rows)),
        "quality_pass_rows": int(quality.sum()),
        "downstream_unique_proteins": len(entity_union.protein_hashes),
        "downstream_unique_drugs": len(entity_union.ligand_inchikeys),
        "protein_hit_rows": int(protein_hit.sum()),
        "drug_hit_rows": int(drug_hit.sum()),
        "both_hit_rows": int(both_hit.sum()),
        "union_removed_rows": int(union_hit.sum()),
        "pre_window_counts": pre_counts,
        "post_window_counts": post_counts,
        "window_removed_rows": int(pre_window.sum() - eligible.sum()),
        "label_failure_rows": int(all_rows["eligibility_reason"].isin(
            ["LABEL_MISSING_ARRAY", "LABEL_LENGTH_MISMATCH", "LABEL_LOAD_ERROR", "MANIFEST_LENGTH_MISMATCH"]
        ).sum()),
        "retained_exact_protein_overlaps": int((eligible & all_rows["protein_overlap"]).sum()),
        "retained_exact_drug_overlaps": int((eligible & all_rows["drug_overlap"]).sum()),
        "max_real_residues": int(max_real_residues),
    }

    retained = all_rows.loc[eligible].copy().reset_index(drop=True)
    outputs = {}
    for split in split_order:
        outputs[split] = retained.loc[retained["official_split"].eq(split)].reset_index(drop=True)
    unknown = sorted(set(retained["official_split"]).difference(split_order))
    if unknown:
        raise ValueError(f"Unexpected official splits: {unknown}")
    template = retained if len(retained) else all_rows
    return FilterResult(
        train=outputs.get("train", _empty_like(template)),
        valid=outputs.get("valid", _empty_like(template)),
        test=outputs.get("test", _empty_like(template)),
        all_rows=all_rows,
        audit=audit,
    )
