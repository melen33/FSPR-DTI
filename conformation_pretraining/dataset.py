"""Strictly aligned Apo2Mol dataset and TAPB-compatible collation."""

from __future__ import annotations

import hashlib
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ConformationSample:
    sample_id: str
    protein_id: str
    drug_id: str
    smiles: str
    window_sequence: str
    protein_features: torch.Tensor
    protein_flex: torch.Tensor
    drug_flex: torch.Tensor
    protein_mask: torch.Tensor
    log_displacement: torch.Tensor
    log_gearnet_response: torch.Tensor
    gearnet_change: torch.Tensor
    global_gearnet_change: torch.Tensor
    contact_distance: torch.Tensor
    loss_mask: torch.Tensor
    pocket_mask: torch.Tensor
    pocket_rmsd: torch.Tensor
    log_global_targets: torch.Tensor
    site_id: str


@dataclass(frozen=True)
class ConformationBatch:
    model_inputs: dict
    targets: dict
    sample_ids: tuple[str, ...]

    def to(self, device):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value
        return ConformationBatch(move(self.model_inputs), move(self.targets), self.sample_ids)


def infer_gearnet_embedding_dim(gearnet_root: Path) -> int:
    """Read and validate the frozen GearNet width before TAPB is constructed."""
    root = Path(gearnet_root)
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No GearNet embedding NPZ files were found in: {root}")
    with np.load(paths[0], allow_pickle=False) as archive:
        if "apo_embedding" not in archive.files or "holo_embedding" not in archive.files:
            raise ValueError(f"GearNet embedding is missing apo/holo arrays: {paths[0]}")
        apo = np.asarray(archive["apo_embedding"])
        holo = np.asarray(archive["holo_embedding"])
    if apo.ndim != 2 or apo.shape != holo.shape or apo.shape[0] < 1 or apo.shape[1] < 1:
        raise ValueError(f"GearNet apo/holo embedding shape mismatch: {paths[0]}")
    return int(apo.shape[1])


def derive_pocket_site_id(protein_id, valid_mask, pocket_mask):
    """Return a deterministic identifier for one protein's supervised pocket.

    Site identity is defined only by real residues that are both structurally
    valid and annotated as pocket residues; it never uses a structural target
    value or enters the model forward pass.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    pocket = np.asarray(pocket_mask, dtype=bool)
    if valid.ndim != 1 or pocket.shape != valid.shape:
        raise ValueError("valid_mask and pocket_mask must be same-length vectors")
    indices = np.flatnonzero(valid & pocket).astype("<i4", copy=False)
    if indices.size == 0:
        raise ValueError("A pocket site requires at least one valid pocket residue")
    digest = hashlib.sha256(indices.tobytes()).hexdigest()[:20]
    return f"{str(protein_id)}:{digest}"


def build_global_random_target_indices(site_group_ids, seed: int):
    """Build a stable, within-site random derangement of structural targets.

    Groups containing one structural example retain their own target because no
    same-protein/same-pocket alternative exists.  Every larger group is mapped
    to a different member of that group, once per seed, for the full run.
    """
    groups = defaultdict(list)
    for index, group_id in enumerate(site_group_ids):
        groups[str(group_id)].append(index)
    mapping = list(range(len(site_group_ids)))
    for group_id, positions in groups.items():
        if len(positions) < 2:
            continue
        digest = hashlib.sha256(f"{int(seed)}:{group_id}".encode("utf-8")).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
        shuffled = [positions[item] for item in torch.randperm(len(positions), generator=generator).tolist()]
        for source, target in zip(shuffled, shuffled[1:] + shuffled[:1]):
            mapping[source] = target
    return tuple(mapping)


def _load_pickle(path_or_mapping):
    if isinstance(path_or_mapping, Mapping):
        return path_or_mapping
    with Path(path_or_mapping).open("rb") as handle:
        return pickle.load(handle)


class ConformationDataset(Dataset):
    def __init__(
        self,
        split_csv: Path,
        apo_root: Path,
        esm_cache,
        dynamine_path: Path,
        ntabs_path: Path,
        max_real_residues: int = 510,
        protein_feature_dim: int = 1280,
        gearnet_root: Optional[Path] = None,
        contact_data_root: Optional[Path] = None,
        structural_target_permutation: str = "none",
        structural_target_permutation_seed: int = 0,
    ):
        self.rows = pd.read_csv(split_csv).reset_index(drop=True)
        self.apo_root = Path(apo_root)
        self.esm_cache = _load_pickle(esm_cache)
        self.max_real_residues = int(max_real_residues)
        self.protein_feature_dim = int(protein_feature_dim)
        self.gearnet_root = None if gearnet_root is None else Path(gearnet_root)
        self.contact_data_root = None if contact_data_root is None else Path(contact_data_root)
        if self.gearnet_root is not None and not self.gearnet_root.is_dir():
            raise FileNotFoundError(f"GearNet embedding directory was not found: {self.gearnet_root}")
        if self.protein_feature_dim < 1:
            raise ValueError("protein_feature_dim must be positive")
        required = {
            "sample_id", "protein_id", "drug_id", "Protein", "SMILES", "label_path",
            "global_mean_ca_displacement", "global_max_ca_displacement",
        }
        missing = required.difference(self.rows.columns)
        if missing:
            raise ValueError(f"Split CSV is missing columns: {sorted(missing)}")
        self.protein_flex = self._load_dynamine(dynamine_path)
        self.drug_flex = self._load_ntabs(ntabs_path)
        self._site_group_ids = None
        self.structural_target_permutation = str(structural_target_permutation)
        if self.structural_target_permutation not in {"none", "global_random_derangement"}:
            raise ValueError("structural_target_permutation must be none or global_random_derangement")
        self.structural_target_permutation_seed = int(structural_target_permutation_seed)
        self._global_target_indices = None
        self.gearnet_embedding_dim = 0
        self._contact_distances = {}

    def _site_id_for_row(self, row):
        full_sequence = str(row["Protein"])
        window_length = min(len(full_sequence), self.max_real_residues)
        label_path = Path(str(row["label_path"]))
        if not label_path.is_absolute():
            label_path = self.apo_root / label_path
        with np.load(label_path, allow_pickle=False) as labels:
            required = ("ca_displacement", "valid_mask", "pocket_mask")
            missing = [name for name in required if name not in labels.files]
            if missing:
                raise ValueError(f"Missing label arrays for {row['sample_id']}: {missing}")
            displacement = np.asarray(labels["ca_displacement"])
            valid = np.asarray(labels["valid_mask"], dtype=bool)
            pocket = np.asarray(labels["pocket_mask"], dtype=bool)
        if {displacement.shape[0], valid.shape[0], pocket.shape[0]} != {len(full_sequence)}:
            raise ValueError(f"Apo label length mismatch for {row['sample_id']}")
        return derive_pocket_site_id(
            row["protein_id"],
            valid[:window_length] & np.isfinite(displacement[:window_length]),
            pocket[:window_length],
        )

    @property
    def site_group_ids(self):
        if self._site_group_ids is None:
            self._site_group_ids = tuple(
                self._site_id_for_row(row) for _, row in self.rows.iterrows()
            )
        return self._site_group_ids

    @property
    def global_target_indices(self):
        if self._global_target_indices is None:
            if self.structural_target_permutation == "none":
                self._global_target_indices = tuple(range(len(self.rows)))
            else:
                self._global_target_indices = build_global_random_target_indices(
                    self.site_group_ids, self.structural_target_permutation_seed
                )
        return self._global_target_indices

    def _load_dynamine(self, path):
        frame = pd.read_csv(path)
        required = {"protein_id", "residue_index", "residue", "protein_flex", "status"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"DynaMine table is missing columns: {sorted(missing)}")
        wanted_series = self.rows.drop_duplicates("protein_id").set_index("protein_id")["Protein"].astype(str)
        wanted = {str(protein_id): sequence for protein_id, sequence in wanted_series.items()}
        result = {}
        protein_keys = frame["protein_id"].astype(str)
        for protein_id, group in frame.groupby(protein_keys, sort=False):
            if protein_id not in wanted:
                continue
            sequence = wanted[protein_id]
            group = group.sort_values("residue_index")
            if len(group) != len(sequence):
                raise ValueError(f"DynaMine length mismatch for {protein_id}: {len(group)} vs {len(sequence)}")
            if group["residue_index"].tolist() != list(range(len(sequence))):
                raise ValueError(f"DynaMine residue indices are not contiguous for {protein_id}")
            observed = "".join(group["residue"].astype(str))
            if observed != sequence:
                raise ValueError(f"DynaMine residue mismatch for {protein_id}")
            if not group["status"].astype(str).eq("ok").all():
                raise ValueError(f"DynaMine status is not ok for {protein_id}")
            values = pd.to_numeric(group["protein_flex"], errors="coerce").to_numpy(np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite DynaMine values for {protein_id}")
            result[protein_id] = torch.from_numpy(values.copy())
        missing_ids = sorted(set(wanted).difference(result))
        if missing_ids:
            raise ValueError(f"Missing DynaMine values for protein IDs: {missing_ids[:5]}")
        return result

    def _load_ntabs(self, path):
        frame = pd.read_csv(path)
        required = {"drug_id", "nTABS", "status"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"nTABS table is missing columns: {sorted(missing)}")
        result = {}
        for row in frame.itertuples(index=False):
            value = float(row.nTABS)
            if str(row.status) == "ok" and np.isfinite(value) and value >= 0:
                result[str(row.drug_id)] = value
        missing_ids = sorted(set(self.rows["drug_id"].astype(str)).difference(result))
        if missing_ids:
            raise ValueError(f"Missing valid nTABS values for drug IDs: {missing_ids[:5]}")
        return result

    def _load_gearnet_targets(self, sample_id, window_length, real_valid, pocket):
        if self.gearnet_root is None:
            return None, None, None
        embedding_path = self.gearnet_root / (str(sample_id) + ".npz")
        if not embedding_path.is_file():
            raise FileNotFoundError(f"GearNet embedding was not found for {sample_id}: {embedding_path}")
        with np.load(embedding_path, allow_pickle=False) as archive:
            required = ("apo_embedding", "holo_embedding", "residue_index")
            missing = [name for name in required if name not in archive.files]
            if missing:
                raise ValueError(f"GearNet embedding is missing arrays for {sample_id}: {missing}")
            apo = np.asarray(archive["apo_embedding"], dtype=np.float32)
            holo = np.asarray(archive["holo_embedding"], dtype=np.float32)
            residue_index = np.asarray(archive["residue_index"])
        if apo.ndim != 2 or apo.shape != holo.shape or apo.shape[0] == 0:
            raise ValueError(f"GearNet apo/holo embedding shape mismatch for {sample_id}")
        if not np.isfinite(apo).all() or not np.isfinite(holo).all():
            raise ValueError(f"GearNet embeddings are non-finite for {sample_id}")
        teacher_dim = int(apo.shape[1])
        if self.gearnet_embedding_dim not in {0, teacher_dim}:
            raise ValueError(
                f"GearNet embedding width mismatch for {sample_id}: "
                f"{teacher_dim} vs {self.gearnet_embedding_dim}"
            )
        self.gearnet_embedding_dim = teacher_dim
        if not np.issubdtype(residue_index.dtype, np.integer) or residue_index.shape != (apo.shape[0],):
            raise ValueError(f"GearNet residue indices are invalid for {sample_id}")
        residue_index = residue_index.astype(np.int64, copy=False)
        if (residue_index < 0).any() or (residue_index >= window_length).any() or np.unique(residue_index).size != residue_index.size:
            raise ValueError(f"GearNet residue indices are not unique/window-aligned for {sample_id}")
        lookup = np.full(window_length, -1, dtype=np.int64)
        lookup[residue_index] = np.arange(residue_index.size, dtype=np.int64)
        selected = np.flatnonzero(np.logical_and(np.logical_and(real_valid, pocket), lookup >= 0))
        if selected.size == 0:
            raise ValueError(f"No valid pocket GearNet targets for {sample_id}")
        change = holo - apo
        response = np.linalg.norm(change, axis=1)
        target = torch.full((window_length + 2,), float("nan"), dtype=torch.float32)
        target[torch.from_numpy(selected + 1)] = torch.from_numpy(
            np.log1p(response[lookup[selected]]).astype(np.float32)
        )
        aligned_change = torch.zeros((window_length + 2, teacher_dim), dtype=torch.float32)
        aligned_change[torch.from_numpy(selected + 1)] = torch.from_numpy(
            change[lookup[selected]].astype(np.float32, copy=False)
        )
        global_selected = np.flatnonzero(np.logical_and(real_valid, lookup >= 0))
        if global_selected.size == 0:
            raise ValueError(f"No valid full-protein GearNet targets for {sample_id}")
        global_change = torch.from_numpy(
            change[lookup[global_selected]].mean(axis=0).astype(np.float32, copy=False)
        )
        return target, aligned_change, global_change

    def _load_contact_distance(self, sample_id, label_path, window_length, real_valid, pocket):
        """Load target-only holo ligand distances, caching each structural sample."""
        cached = self._contact_distances.get(str(sample_id))
        expected_shape = (window_length + 2,)
        if cached is not None:
            if tuple(cached.shape) != expected_shape:
                raise ValueError(f"Contact-distance shape mismatch for {sample_id}")
            return cached
        target = torch.full(expected_shape, float("nan"), dtype=torch.float32)
        if self.contact_data_root is None:
            return target
        from structural_validation.contact_response_analysis import sample_contact_rows

        frame, _ = sample_contact_rows(sample_id, label_path, self.contact_data_root)
        for row in frame.itertuples(index=False):
            residue_index = int(row.residue_index)
            if residue_index >= window_length or not (real_valid[residue_index] and pocket[residue_index]):
                continue
            target[residue_index + 1] = float(row.distance_angstrom)
        expected = torch.from_numpy(np.logical_and(real_valid, pocket))
        observed = torch.isfinite(target[1:-1])
        if not torch.equal(observed, expected):
            raise ValueError(f"Contact-distance residues do not match valid pocket residues for {sample_id}")
        self._contact_distances[str(sample_id)] = target
        return target

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        target_row = self.rows.iloc[self.global_target_indices[index]]
        sample_id = str(row["sample_id"])
        target_sample_id = str(target_row["sample_id"])
        protein_id = str(row["protein_id"])
        drug_id = str(row["drug_id"])
        full_sequence = str(row["Protein"])
        window_sequence = full_sequence[: self.max_real_residues]
        window_length = len(window_sequence)
        if protein_id not in self.esm_cache:
            raise ValueError(f"Missing ESM features for {protein_id}")
        protein_features = torch.as_tensor(self.esm_cache[protein_id], dtype=torch.float32)
        expected_shape = (window_length + 2, self.protein_feature_dim)
        if tuple(protein_features.shape) != expected_shape:
            raise ValueError(f"ESM shape mismatch for {sample_id}: {tuple(protein_features.shape)} vs {expected_shape}")

        if str(target_row["protein_id"]) != protein_id or str(target_row["Protein"]) != full_sequence:
            raise ValueError("Global structural target permutation must preserve protein identity and sequence")
        label_path = Path(str(target_row["label_path"]))
        if not label_path.is_absolute():
            label_path = self.apo_root / label_path
        with np.load(label_path, allow_pickle=False) as labels:
            required = ("ca_displacement", "valid_mask", "pocket_mask")
            missing = [name for name in required if name not in labels.files]
            if missing:
                raise ValueError(f"Missing label arrays for {target_sample_id}: {missing}")
            arrays = {name: np.asarray(labels[name]) for name in required}
            lengths = {array.shape[0] for array in arrays.values()}
            if lengths != {len(full_sequence)}:
                raise ValueError(f"Apo label length mismatch for {target_sample_id}: {sorted(lengths)} vs {len(full_sequence)}")
            displacement = arrays["ca_displacement"][:window_length].astype(np.float32)
            valid = arrays["valid_mask"][:window_length].astype(bool)
            pocket = arrays["pocket_mask"][:window_length].astype(bool)
            if "residue_type" in labels.files:
                residue_type = "".join(np.asarray(labels["residue_type"]).astype(str).tolist())
                if len(residue_type) != len(full_sequence):
                    raise ValueError(f"Apo residue label length mismatch for {target_sample_id}")

        log_displacement = torch.full((window_length + 2,), float("nan"), dtype=torch.float32)
        finite = np.isfinite(displacement)
        real_valid = np.logical_and(valid, finite)
        log_displacement[1:-1] = torch.from_numpy(np.log1p(np.maximum(displacement, 0.0)).astype(np.float32))
        loss_mask = torch.zeros(window_length + 2, dtype=torch.bool)
        loss_mask[1:-1] = torch.from_numpy(real_valid)
        pocket_mask = torch.zeros(window_length + 2, dtype=torch.bool)
        pocket_mask[1:-1] = torch.from_numpy(np.logical_and(real_valid, pocket))
        contact_distance = self._load_contact_distance(
            target_sample_id, label_path, window_length, real_valid, pocket
        )
        log_gearnet_response, gearnet_change, global_gearnet_change = self._load_gearnet_targets(
            target_sample_id, window_length, real_valid, pocket
        )
        if log_gearnet_response is None:
            log_gearnet_response = torch.full((window_length + 2,), float("nan"), dtype=torch.float32)
            gearnet_change = torch.zeros((window_length + 2, 0), dtype=torch.float32)
            global_gearnet_change = torch.zeros((0,), dtype=torch.float32)

        flex = self.protein_flex[protein_id][:window_length]
        protein_flex = torch.zeros(window_length + 2, 1, dtype=torch.float32)
        protein_flex[1:-1, 0] = flex
        # The ESM tensor retains BOS/EOS, but interaction modulation and global
        # pooling operate only on real residues.
        protein_mask = torch.zeros(window_length + 2, dtype=torch.bool)
        protein_mask[1:-1] = True
        drug_flex = torch.tensor([np.log1p(self.drug_flex[drug_id])], dtype=torch.float32)
        pocket_valid = np.logical_and(real_valid, pocket)
        if not pocket_valid.any():
            raise ValueError(f"No finite pocket displacement positions for {sample_id}")
        pocket_rmsd = torch.tensor(float(np.sqrt(np.mean(np.square(displacement[pocket_valid])))), dtype=torch.float32)
        site_id = derive_pocket_site_id(protein_id, real_valid, pocket)
        global_targets = torch.tensor([
            np.log1p(max(float(target_row["global_mean_ca_displacement"]), 0.0)),
            np.log1p(max(float(target_row["global_max_ca_displacement"]), 0.0)),
        ], dtype=torch.float32)
        if not torch.isfinite(global_targets).all():
            raise ValueError(f"Non-finite global targets for {sample_id}")
        return ConformationSample(
            sample_id=sample_id,
            protein_id=protein_id,
            drug_id=drug_id,
            smiles=str(row["SMILES"]),
            window_sequence=window_sequence,
            protein_features=protein_features,
            protein_flex=protein_flex,
            drug_flex=drug_flex,
            protein_mask=protein_mask,
            log_displacement=log_displacement,
            log_gearnet_response=log_gearnet_response,
            gearnet_change=gearnet_change,
            global_gearnet_change=global_gearnet_change,
            contact_distance=contact_distance,
            loss_mask=loss_mask,
            pocket_mask=pocket_mask,
            pocket_rmsd=pocket_rmsd,
            log_global_targets=global_targets,
            site_id=site_id,
        )


def make_conformation_collate(tokenizer, max_drug_tokens: int = 200) -> Callable:
    def collate(samples):
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        tokenized = tokenizer(
            [sample.smiles for sample in samples],
            padding=True,
            truncation=True,
            max_length=max_drug_tokens,
            return_tensors="pt",
        )
        model_inputs = {
            "input_drugs": dict(tokenized),
            "input_proteins": pad_sequence([s.protein_features for s in samples], batch_first=True),
            "protein_mask": pad_sequence([s.protein_mask for s in samples], batch_first=True, padding_value=False),
            "protein_flex": pad_sequence([s.protein_flex for s in samples], batch_first=True),
            "drug_flex": torch.stack([s.drug_flex for s in samples]),
        }
        targets = {
            "log_displacement": pad_sequence(
                [s.log_displacement for s in samples], batch_first=True, padding_value=float("nan")
            ),
            "valid_mask": pad_sequence([s.loss_mask for s in samples], batch_first=True, padding_value=False),
            "log_gearnet_response": pad_sequence(
                [s.log_gearnet_response for s in samples], batch_first=True, padding_value=float("nan")
            ),
            "gearnet_change": pad_sequence(
                [s.gearnet_change for s in samples], batch_first=True
            ),
            "global_gearnet_change": torch.stack([s.global_gearnet_change for s in samples]),
            "contact_distance": pad_sequence(
                [s.contact_distance for s in samples], batch_first=True, padding_value=float("nan")
            ),
            "pocket_mask": pad_sequence([s.pocket_mask for s in samples], batch_first=True, padding_value=False),
            "log_global_targets": torch.stack([s.log_global_targets for s in samples]),
            "pocket_rmsd": torch.stack([s.pocket_rmsd for s in samples]),
            "protein_ids": tuple(s.protein_id for s in samples),
            "site_ids": tuple(s.site_id for s in samples),
            "drug_ids": tuple(s.drug_id for s in samples),
        }
        return ConformationBatch(model_inputs, targets, tuple(s.sample_id for s in samples))
    return collate
