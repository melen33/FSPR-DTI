"""Feature preparation and common-cohort auditing."""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


def build_unique_entity_tables(
    split_paths: Sequence[Path],
    output_root: Path | None = None,
    max_real_residues: int = 510,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(path) for path in map(Path, split_paths)]
    if not frames:
        raise ValueError("At least one filtered split CSV is required")
    combined = pd.concat(frames, ignore_index=True)
    required = {"protein_id", "Protein", "drug_id", "SMILES"}
    missing = required.difference(combined.columns)
    if missing:
        raise ValueError(f"Filtered CSVs are missing columns: {sorted(missing)}")

    protein_pairs = combined[["protein_id", "Protein"]].drop_duplicates()
    if protein_pairs["protein_id"].duplicated().any():
        examples = protein_pairs.loc[protein_pairs["protein_id"].duplicated(False), "protein_id"].head().tolist()
        raise ValueError(f"protein_id maps to multiple protein sequences: {examples}")
    proteins = protein_pairs.sort_values("protein_id").reset_index(drop=True)
    proteins["protein_id"] = proteins["protein_id"].astype(str)
    proteins["Protein"] = proteins["Protein"].astype(str)
    proteins["full_length"] = proteins["Protein"].str.len()
    proteins["window_length"] = proteins["full_length"].clip(upper=max_real_residues)

    drug_pairs = combined[["drug_id", "SMILES"]].drop_duplicates().copy()
    def canonical_smiles(value):
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            raise ValueError(f"Invalid Apo2Mol SMILES: {value}")
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    drug_pairs["SMILES"] = drug_pairs["SMILES"].map(canonical_smiles)
    # InChIKey is the declared entity key. Apo2Mol occasionally contains
    # equivalent/underspecified stereochemical SMILES for one InChIKey; select
    # one canonical representation deterministically for tokenization/nTABS.
    drugs = drug_pairs.sort_values(["drug_id", "SMILES"]).drop_duplicates("drug_id").reset_index(drop=True)
    drugs["drug_id"] = drugs["drug_id"].astype(str)
    drugs["SMILES"] = drugs["SMILES"].astype(str)

    if output_root is not None:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        proteins.to_csv(root / "filtered_proteins.csv", index=False)
        drugs.to_csv(root / "filtered_drugs.csv", index=False)
        drug_dir = root / "drug_entities"
        drug_dir.mkdir(parents=True, exist_ok=True)
        drugs.to_csv(drug_dir / "filtered_drugs.csv", index=False)
    return proteins, drugs


def _move_inputs(inputs, device):
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _atomic_pickle(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output_path)


def generate_esm2_cache(
    proteins_csv: Path,
    model_path: str | Path,
    output_path: Path,
    device: str = "cpu",
    tokenizer=None,
    model=None,
    limit: int | None = None,
    save_every: int = 100,
    resume: bool = False,
) -> dict[str, torch.Tensor]:
    proteins = pd.read_csv(proteins_csv)
    required = {"protein_id", "Protein"}
    missing = required.difference(proteins.columns)
    if missing:
        raise ValueError(f"Protein table is missing columns: {sorted(missing)}")
    if limit is not None:
        proteins = proteins.head(limit)
    if tokenizer is None or model is None:
        from transformers import AutoTokenizer, EsmModel
        tokenizer = AutoTokenizer.from_pretrained(str(model_path)) if tokenizer is None else tokenizer
        model = EsmModel.from_pretrained(str(model_path)) if model is None else model
    hidden_size = int(getattr(model.config, "hidden_size", -1))
    if hidden_size != 1280:
        raise ValueError(f"ESM2 hidden size must be 1280, got hidden size {hidden_size}")
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output_path = Path(output_path)
    cache: dict[str, torch.Tensor] = {}
    if resume and output_path.exists():
        with output_path.open("rb") as handle:
            cache = pickle.load(handle)
    processed = 0
    with torch.no_grad():
        for row in proteins.itertuples(index=False):
            protein_id = str(row.protein_id)
            sequence = str(row.Protein)
            expected_length = min(len(sequence), 510) + 2
            if protein_id in cache:
                tensor = cache[protein_id]
                if tuple(tensor.shape) != (expected_length, 1280):
                    raise ValueError(f"Existing ESM cache shape mismatch for {protein_id}: {tuple(tensor.shape)}")
                continue
            inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**_move_inputs(inputs, device))
            tensor = outputs.last_hidden_state.squeeze(0).detach().to(device="cpu", dtype=torch.float32).contiguous()
            if tuple(tensor.shape) != (expected_length, 1280):
                raise ValueError(
                    f"ESM alignment mismatch for {protein_id}: expected {(expected_length, 1280)}, got {tuple(tensor.shape)}"
                )
            cache[protein_id] = tensor
            processed += 1
            if save_every > 0 and processed % save_every == 0:
                _atomic_pickle(cache, output_path)
                print(f"Cached {len(cache)}/{len(proteins)} proteins", flush=True)
    _atomic_pickle(cache, output_path)
    return cache


def _load_dynamine_status(path: Path, protein_sequences: dict[str, str]) -> dict[str, str]:
    frame = pd.read_csv(path)
    required = {"protein_id", "residue_index", "residue", "protein_flex", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"DynaMine table is missing columns: {sorted(missing)}")
    statuses = {protein_id: "PROTEIN_FLEX_MISSING" for protein_id in protein_sequences}
    protein_keys = frame["protein_id"].astype(str)
    for protein_id, group in frame.groupby(protein_keys, sort=False):
        if protein_id not in protein_sequences:
            continue
        sequence = protein_sequences[protein_id]
        group = group.sort_values("residue_index")
        if len(group) != len(sequence) or group["residue_index"].tolist() != list(range(len(sequence))):
            statuses[protein_id] = "PROTEIN_FLEX_LENGTH_MISMATCH"
        elif "".join(group["residue"].astype(str)) != sequence:
            statuses[protein_id] = "PROTEIN_FLEX_RESIDUE_MISMATCH"
        elif not group["status"].astype(str).eq("ok").all():
            statuses[protein_id] = "PROTEIN_FLEX_INVALID"
        elif not np.isfinite(pd.to_numeric(group["protein_flex"], errors="coerce")).all():
            statuses[protein_id] = "PROTEIN_FLEX_INVALID"
        else:
            statuses[protein_id] = "ok"
    return statuses


def audit_feature_intersection(
    filtered_root: Path,
    esm_cache_path: Path,
    dynamine_path: Path,
    ntabs_path: Path,
    output_root: Path,
) -> dict:
    filtered_root = Path(filtered_root)
    split_frames = {}
    for split in ("train", "valid", "test"):
        path = filtered_root / f"{split}.csv"
        split_frames[split] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    nonempty = [frame for frame in split_frames.values() if not frame.empty]
    if not nonempty:
        raise ValueError("No filtered samples found")
    combined = pd.concat(nonempty, ignore_index=True)
    with Path(esm_cache_path).open("rb") as handle:
        esm_cache = pickle.load(handle)
    protein_sequences = combined.drop_duplicates("protein_id").set_index("protein_id")["Protein"].astype(str).to_dict()
    protein_sequences = {str(key): value for key, value in protein_sequences.items()}
    dynamine_status = _load_dynamine_status(Path(dynamine_path), protein_sequences)

    ntabs = pd.read_csv(ntabs_path)
    required = {"drug_id", "nTABS", "status"}
    missing = required.difference(ntabs.columns)
    if missing:
        raise ValueError(f"nTABS table is missing columns: {sorted(missing)}")
    drug_status = {}
    for row in ntabs.itertuples(index=False):
        valid = str(row.status) == "ok" and math.isfinite(float(row.nTABS)) and float(row.nTABS) >= 0
        drug_status[str(row.drug_id)] = "ok" if valid else "DRUG_FLEX_INVALID"

    decisions = []
    for row in combined.to_dict("records"):
        protein_id = str(row["protein_id"])
        drug_id = str(row["drug_id"])
        expected = min(len(str(row["Protein"])), 510) + 2
        if protein_id not in esm_cache:
            status = "ESM_MISSING"
        elif tuple(esm_cache[protein_id].shape) != (expected, 1280):
            status = "ESM_SHAPE_MISMATCH"
        elif dynamine_status.get(protein_id, "PROTEIN_FLEX_MISSING") != "ok":
            status = dynamine_status.get(protein_id, "PROTEIN_FLEX_MISSING")
        elif drug_status.get(drug_id, "DRUG_FLEX_MISSING") != "ok":
            status = drug_status.get(drug_id, "DRUG_FLEX_MISSING")
        else:
            status = "ok"
        row["feature_status"] = status
        row["feature_eligible"] = status == "ok"
        decisions.append(row)
    manifest = pd.DataFrame(decisions)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_root / "feature_manifest.csv", index=False)
    counts = {}
    for split in ("train", "valid", "test"):
        output = manifest.loc[manifest["official_split"].eq(split) & manifest["feature_eligible"]].copy()
        output.to_csv(output_root / f"{split}.csv", index=False)
        counts[split] = int(len(output))
    reason_counts = {str(k): int(v) for k, v in manifest["feature_status"].value_counts().sort_index().items()}
    audit = {"counts": counts, "feature_status_counts": reason_counts, "samples_total": int(len(manifest))}
    temp = output_root / "feature_audit.json.tmp"
    temp.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, output_root / "feature_audit.json")
    return audit
