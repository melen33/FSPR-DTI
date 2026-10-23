"""Cohort checks and latent-response extraction for external validation."""

import argparse
from collections.abc import Mapping
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import platform
import tempfile
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from conformation_pretraining.dataset import (
    ConformationDataset,
    make_conformation_collate,
)
from structural_validation.io_utils import (
    output_set_transaction,
    serialize_json_value,
)
from models.tapb import TAPB


REQUIRED_COHORT_COLUMNS = {
    "sample_id",
    "feature_eligible",
    "protein_overlap",
    "drug_overlap",
    "original_length",
    "recomputed_pocket_ca_rmsd",
}


ALLOWED_MODEL_INPUT_KEYS = {
    "input_drugs",
    "input_proteins",
    "protein_mask",
    "protein_flex",
    "drug_flex",
}



def validate_flex_only_config(config):
    """Reject configurations that differ from the confirmed flex-only model."""
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint model_config must be a mapping.")

    boolean_expectations = {
        "use_drug_flex": False,
        "use_protein_flex": False,
        "use_protein_induced_gate": False,
        "use_drug_induced_gate": False,
        "use_flex_attention_bias": False,
        "use_flex_calibration": False,
        "use_flex_film": True,
        "film_use_protein_flex": True,
        "film_use_drug_flex": False,
        "relation_alignment_enabled": False,
        "conformation_pretrained_loaded": False,
    }
    enum_expectations = {
        "film_side": "protein",
        "film_mode": "shift_only",
        "film_flex_encoding": "raw",
        "film_protein_flex_perturbation": "none",
        "alignment_target": "none",
        "conformation_pretrained_checkpoint": "",
    }
    required = set(boolean_expectations).union(enum_expectations)
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Checkpoint model_config is missing required keys: {missing}")

    invalid_boolean_fields = [
        name
        for name in boolean_expectations
        if not isinstance(config[name], (bool, np.bool_))
    ]
    if invalid_boolean_fields:
        raise ValueError(
            "Checkpoint model_config boolean fields must be actual booleans: "
            f"{invalid_boolean_fields}"
        )

    pretraining = str(config["conformation_pretrained_checkpoint"] or "")
    if pretraining:
        raise ValueError(
            "Apo2Mol-pretrained checkpoint is forbidden for external validation."
        )

    failed = [
        f"{name}={expected}"
        for name, expected in boolean_expectations.items()
        if config[name] != expected
    ]
    failed.extend(
        f"{name}={expected}"
        for name, expected in enum_expectations.items()
        if config[name] != expected
    )
    if failed:
        raise ValueError(f"Checkpoint is not the confirmed flex-only model: {failed}")



def validate_structure_supervised_config(config):
    """Reject structure-supervised checkpoints outside the approved FiLM family."""
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint model_config must be a mapping.")
    boolean_expectations = {
        "use_drug_flex": False,
        "use_protein_flex": False,
        "use_protein_induced_gate": False,
        "use_drug_induced_gate": False,
        "use_flex_attention_bias": False,
        "use_flex_calibration": False,
        "use_flex_film": True,
        "film_use_protein_flex": True,
        "film_use_drug_flex": False,
        "relation_alignment_enabled": False,
        "conformation_pretrained_loaded": False,
    }
    enum_expectations = {
        "film_side": "protein",
        "film_mode": "shift_only",
        "film_flex_encoding": "raw",
        "film_protein_flex_perturbation": "none",
        "alignment_target": "none",
        "conformation_pretrained_checkpoint": "",
    }
    required = set(boolean_expectations).union(enum_expectations)
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Checkpoint model_config is missing required keys: {missing}")
    invalid_boolean_fields = [
        name for name in boolean_expectations
        if not isinstance(config[name], (bool, np.bool_))
    ]
    if invalid_boolean_fields:
        raise ValueError(
            "Checkpoint model_config boolean fields must be actual booleans: "
            f"{invalid_boolean_fields}"
        )
    failed = [
        f"{name}={expected}"
        for name, expected in boolean_expectations.items()
        if config[name] != expected
    ]
    failed.extend(
        f"{name}={expected}"
        for name, expected in enum_expectations.items()
        if config[name] != expected
    )
    if failed:
        raise ValueError(
            "Checkpoint is not the approved structure-supervised model: "
            f"{failed}"
        )


def validate_systematic_ablation_config(config):
    """Accept only the formal V1--V9 native-encoder ablation family."""
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint model_config must be a mapping.")

    boolean_expectations = {
        "use_drug_flex": False,
        "use_protein_flex": False,
        "use_protein_induced_gate": False,
        "use_drug_induced_gate": False,
        "use_flex_attention_bias": False,
        "use_flex_calibration": False,
        "relation_alignment_enabled": False,
        "conformation_pretrained_loaded": False,
    }
    enum_expectations = {
        "drug_encoder_backend": "native",
        "protein_encoder_backend": "esm2",
        "film_mode": "shift_only",
        "film_flex_encoding": "raw",
        "film_protein_flex_perturbation": "none",
        "alignment_target": "none",
        "conformation_pretrained_checkpoint": "",
    }
    required = set(boolean_expectations).union(enum_expectations).union(
        {"use_flex_film", "film_side"}
    )
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(
            "Systematic-ablation checkpoint model_config is missing required keys: "
            f"{missing}"
        )
    for name in set(boolean_expectations).union({"use_flex_film"}):
        if not isinstance(config[name], (bool, np.bool_)):
            raise ValueError(
                "Systematic-ablation checkpoint booleans must be actual booleans: "
                f"{name}"
            )
    failed = [
        f"{name}={expected}"
        for name, expected in boolean_expectations.items()
        if config[name] != expected
    ]
    failed.extend(
        f"{name}={expected}"
        for name, expected in enum_expectations.items()
        # `film_mode` is inactive when FiLM is disabled. Historical S1
        # checkpoints may persist `scale_shift` as this constructor field.
        if config[name] != expected
        and (name != "film_mode" or config["use_flex_film"])
    )
    if failed:
        raise ValueError(
            "Checkpoint is outside the formal systematic-ablation contract: "
            f"{failed}"
        )

    film_side = config["film_side"]
    if film_side not in {"protein", "drug", "both"}:
        raise ValueError(
            "Checkpoint is outside the formal systematic-ablation contract: "
            f"film_side={film_side!r}"
        )
    side_fields = (
        "protein_film_use_protein_flex",
        "protein_film_use_drug_flex",
        "drug_film_use_drug_flex",
        "drug_film_use_protein_flex",
    )
    if not config["use_flex_film"]:
        invalid_present = [
            name for name in side_fields
            if name in config and not isinstance(config[name], (bool, np.bool_))
        ]
        if invalid_present:
            raise ValueError(
                "Systematic-ablation checkpoint booleans must be actual booleans: "
                f"{invalid_present}"
            )
        enabled_present = [
            name for name in side_fields if name in config and bool(config[name])
        ]
        if enabled_present:
            raise ValueError(
                "Checkpoint is outside the formal systematic-ablation contract: "
                f"{enabled_present} must be False when use_flex_film=False"
            )
        if config["film_mode"] not in {"shift_only", "scale_shift"}:
            raise ValueError(
                "Checkpoint is outside the formal systematic-ablation contract: "
                f"film_mode={config['film_mode']!r}"
            )
        return

    # Protein-only historical strict-ablation checkpoints were saved before
    # inactive drug-side flags were persisted. Their absence is unambiguous:
    # a protein-only FiLM has no drug-side module, hence both flags are False.
    # Do not apply this compatibility rule to drug/bidirectional FiLM, where a
    # missing field would conceal a material configuration difference.
    if film_side == "protein":
        for name in ("drug_film_use_drug_flex", "drug_film_use_protein_flex"):
            config.setdefault(name, False)

    missing_side_fields = sorted(set(side_fields).difference(config))
    if missing_side_fields:
        raise ValueError(
            "Systematic-ablation FiLM checkpoint is missing side-specific keys: "
            f"{missing_side_fields}"
        )
    invalid_side_fields = [
        name for name in side_fields
        if not isinstance(config[name], (bool, np.bool_))
    ]
    if invalid_side_fields:
        raise ValueError(
            "Systematic-ablation checkpoint booleans must be actual booleans: "
            f"{invalid_side_fields}"
        )
    expected_side_fields = {
        "protein_film_use_protein_flex": film_side in {"protein", "both"},
        "protein_film_use_drug_flex": False,
        "drug_film_use_drug_flex": film_side in {"drug", "both"},
        "drug_film_use_protein_flex": False,
    }
    mismatched = [
        f"{name}={expected}"
        for name, expected in expected_side_fields.items()
        if config[name] != expected
    ]
    if mismatched:
        raise ValueError(
            "Checkpoint is outside the formal systematic-ablation contract: "
            f"{mismatched}"
        )


def filter_validation_cohort(frame, max_length=510):
    """Return strictly eligible cohort rows plus per-sample exclusion audit rows."""
    missing = REQUIRED_COHORT_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature manifest is missing columns: {sorted(missing)}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Feature manifest contains duplicate sample_id values.")

    reasons = []
    boolean_columns = ("feature_eligible", "protein_overlap", "drug_overlap")
    for row in frame.itertuples(index=False):
        values = {name: getattr(row, name) for name in boolean_columns}
        invalid = [
            name
            for name, value in values.items()
            if not isinstance(value, (bool, np.bool_))
        ]
        if invalid:
            raise ValueError(
                "Feature manifest boolean columns must contain actual booleans; "
                f"invalid values in {invalid}."
            )

        reason = None
        if not values["feature_eligible"]:
            reason = "feature_ineligible"
        elif values["protein_overlap"]:
            reason = "protein_overlap"
        elif values["drug_overlap"]:
            reason = "drug_overlap"
        elif int(row.original_length) > int(max_length):
            reason = f"protein_length_gt_{int(max_length)}"
        elif not np.isfinite(float(row.recomputed_pocket_ca_rmsd)):
            reason = "nonfinite_recomputed_pocket_ca_rmsd"
        reasons.append(reason)

    annotated = frame.copy()
    annotated["exclusion_reason"] = reasons
    eligible = annotated[annotated["exclusion_reason"].isna()].copy()
    audit = annotated[annotated["exclusion_reason"].notna()][
        ["sample_id", "exclusion_reason"]
    ].copy()
    return eligible.drop(columns=["exclusion_reason"]), audit




def filter_gearnet_cached_samples(frame, gearnet_root):
    """Retain only rows whose per-sample GearNet archive exists, with audit."""
    root = Path(gearnet_root)
    if not root.is_dir():
        raise FileNotFoundError(f"GearNet embedding directory was not found: {root}")
    identifiers = frame["sample_id"].astype(str)
    present = identifiers.map(lambda sample_id: (root / (sample_id + ".npz")).is_file())
    retained = frame.loc[present].copy()
    missing = pd.DataFrame({
        "sample_id": identifiers.loc[~present].tolist(),
        "exclusion_reason": "missing_gearnet_embedding",
    })
    return retained, missing

def validate_selected_sequence_lengths(frame):
    """Return actual sequence lengths after exact manifest-length validation."""
    required = {"sample_id", "Protein", "original_length"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Selected cohort is missing sequence columns: {missing}")
    if frame["Protein"].isna().any():
        bad_ids = frame.loc[frame["Protein"].isna(), "sample_id"].astype(str).tolist()
        raise ValueError(f"Samples {bad_ids} have missing Protein sequences.")

    sequences = frame["Protein"].astype(str)
    actual = sequences.str.len().astype(int)
    original = pd.to_numeric(frame["original_length"], errors="coerce")
    mismatch = (~np.isfinite(original)) | (original != actual) | (actual <= 0)
    if mismatch.any():
        bad_ids = frame.loc[mismatch, "sample_id"].astype(str).tolist()
        raise ValueError(
            f"Samples {bad_ids} have original_length values that do not match "
            "the actual Protein sequence length."
        )
    return dict(zip(frame["sample_id"].astype(str), actual.tolist()))


def configure_deterministic_inference(seed):
    """Configure deterministic random seeds and cuDNN inference behavior."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_inference_device(
    requested_device,
    cuda_available=None,
    cuda_device_count=None,
    cuda_device_name=None,
):
    """Resolve CPU/CUDA inference requests with auditable fallback metadata."""
    requested_text = str(requested_device)
    try:
        requested = torch.device(requested_text)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid inference device {requested_text!r}: {error}") from None
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("Inference device must be CPU or CUDA.")

    available = bool(
        torch.cuda.is_available() if cuda_available is None else cuda_available
    )
    count = (
        torch.cuda.device_count() if cuda_device_count is None else cuda_device_count
    )
    name_lookup = torch.cuda.get_device_name if cuda_device_name is None else cuda_device_name
    selected_index = None
    selected_name = None
    fallback_reason = None
    if requested.type == "cpu":
        resolved = torch.device("cpu")
    elif not available:
        resolved = torch.device("cpu")
        fallback_reason = "cuda_unavailable"
    else:
        selected_index = 0 if requested.index is None else requested.index
        if selected_index < 0 or selected_index >= count:
            raise ValueError(
                f"CUDA device index {selected_index} is invalid for {count} visible devices."
            )
        resolved = torch.device(f"cuda:{selected_index}")
        selected_name = str(name_lookup(selected_index))

    return resolved, {
        "requested_device": requested_text,
        "resolved_device": str(resolved),
        "cuda_available": available,
        "visible_cuda_device_count": int(count),
        "selected_cuda_index": selected_index,
        "selected_gpu_name": selected_name,
        "fallback_reason": fallback_reason,
    }


def dti_inference_mask_from_residue_mask(residue_mask):
    """Include BOS/EOS for DTI inference while preserving padding positions."""
    if not torch.is_tensor(residue_mask) or residue_mask.dtype != torch.bool:
        raise ValueError("Real-residue mask must be a boolean torch tensor.")
    if residue_mask.ndim != 2 or residue_mask.numel() == 0:
        raise ValueError("Real-residue mask must be a nonempty two-dimensional batch.")
    if residue_mask.size(1) < 3:
        raise ValueError("Real-residue mask must have room for BOS, residues, and EOS.")

    inference_mask = torch.zeros_like(residue_mask)
    for row_index, row in enumerate(residue_mask):
        positions = torch.nonzero(row, as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError("Each real-residue mask must contain at least one residue.")
        first = int(positions[0].item())
        last = int(positions[-1].item())
        if first != 1 or last + 1 >= row.numel():
            raise ValueError(
                "Real-residue mask must follow BOS/residues/EOS token bounds."
            )
        expected = torch.zeros_like(row)
        expected[1:last + 1] = True
        if not torch.equal(row, expected):
            raise ValueError("Real-residue mask residues must be contiguous.")
        inference_mask[row_index, :last + 2] = True
    return inference_mask


def compute_latent_whole(before, after, residue_mask, expected_lengths):
    """Mean per-real-residue L2 change between token-aligned TAPB latents."""
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError(
            f"H_before/H_after shape mismatch: {before.shape} vs {after.shape}"
        )
    if before.size(-1) != 256:
        raise ValueError(f"Expected hidden dimension 256, got {before.size(-1)}")
    if not torch.is_tensor(residue_mask) or residue_mask.dtype != torch.bool:
        raise ValueError("Residue mask must be a boolean torch tensor.")
    if residue_mask.shape != before.shape[:2]:
        raise ValueError("Residue mask shape does not match latent tensors.")

    try:
        if isinstance(expected_lengths, torch.Tensor):
            if expected_lengths.ndim != 1:
                raise ValueError
            expected_values = expected_lengths.detach().cpu().tolist()
        elif isinstance(expected_lengths, np.ndarray):
            if expected_lengths.ndim != 1:
                raise ValueError
            expected_values = expected_lengths.tolist()
        else:
            expected_values = list(expected_lengths)
    except (TypeError, ValueError):
        raise ValueError(
            "expected_lengths must be a one-dimensional batch-sized sequence of "
            "nonnegative integers."
        ) from None

    if len(expected_values) != before.size(0):
        raise ValueError(
            "expected_lengths must be a one-dimensional batch-sized sequence of "
            "nonnegative integers."
        )
    if any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
        for value in expected_values
    ) or any(value < 0 for value in expected_values):
        raise ValueError(
            "expected_lengths must be a one-dimensional batch-sized sequence of "
            "nonnegative integers."
        )

    mask = residue_mask
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Each sample must contain at least one real residue.")
    expected = torch.tensor(expected_values, device=counts.device, dtype=counts.dtype)
    if not torch.equal(counts, expected):
        raise ValueError(
            f"Real-residue count mismatch: {counts.tolist()} vs {expected.tolist()}"
        )
    per_residue = torch.linalg.vector_norm(after - before, ord=2, dim=-1)
    values = (per_residue * mask).sum(dim=1) / counts
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite latent response encountered.")
    return values.detach().cpu().numpy()



def compute_pocket_residue_records(
    before, after, residue_mask, valid_mask, pocket_mask, log_displacement,
    sample_ids, protein_ids,
):
    """Return one raw response/C-alpha-displacement row per valid pocket residue."""
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("before and after must share shape [B, L, D].")
    shape = before.shape[:2]
    masks = {
        "residue_mask": residue_mask,
        "valid_mask": valid_mask,
        "pocket_mask": pocket_mask,
    }
    for name, mask in masks.items():
        if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != shape:
            raise ValueError(f"{name} must be a boolean tensor with shape {tuple(shape)}.")
    if not torch.is_tensor(log_displacement) or log_displacement.shape != shape:
        raise ValueError("log_displacement must be a tensor with shape [B, L].")
    try:
        sample_values, protein_values = list(sample_ids), list(protein_ids)
    except TypeError:
        raise ValueError("sample_ids and protein_ids must be batch-sized sequences.") from None
    if len(sample_values) != before.size(0) or len(protein_values) != before.size(0):
        raise ValueError("sample_ids and protein_ids must be batch-sized sequences.")
    selected = residue_mask & valid_mask & pocket_mask
    if not bool(selected.any(dim=1).all()):
        raise ValueError("Each sample must contain at least one valid pocket residue.")
    response = torch.linalg.vector_norm(after - before, ord=2, dim=-1)
    target = torch.expm1(log_displacement.to(device=response.device, dtype=response.dtype))
    if not bool(torch.isfinite(response[selected]).all()) or not bool(torch.isfinite(target[selected]).all()):
        raise ValueError("Pocket response and C-alpha displacement must be finite.")
    if bool((target[selected] < 0).any()):
        raise ValueError("C-alpha displacement must be non-negative.")
    records = []
    for batch_index in range(before.size(0)):
        positions = torch.nonzero(selected[batch_index], as_tuple=False).flatten().tolist()
        for token_index in positions:
            records.append({
                "sample_id": str(sample_values[batch_index]),
                "protein_id": str(protein_values[batch_index]),
                "residue_index": int(token_index - 1),
                "latent_residue_response": float(response[batch_index, token_index].detach().cpu()),
                "structural_ca_displacement": float(target[batch_index, token_index].detach().cpu()),
            })
    return records



def build_high_dimensional_response_cache_records(
    output, structural_targets, sample_ids, protein_ids, pocket_rmsd,
):
    """Build aligned valid-pocket high-dimensional response arrays for caching."""
    before = output.get("protein_before")
    after = output.get("protein_after")
    residue_mask = output.get("protein_residue_mask")
    raw_change = after - before
    semantic = output.get("protein_semantic_change")
    if semantic is None:
        semantic = after - before
    if not torch.is_tensor(semantic) or semantic.ndim != 3:
        raise ValueError("protein_semantic_change must have shape [B, L, D].")
    if not torch.is_tensor(residue_mask) or residue_mask.dtype != torch.bool:
        raise ValueError("protein_residue_mask must be a boolean tensor.")
    shape = semantic.shape[:2]
    if not torch.is_tensor(raw_change) or raw_change.ndim != 3:
        raise ValueError("protein_raw_change must have shape [B, L, D].")
    if tuple(raw_change.shape[:2]) != tuple(shape):
        raise ValueError("protein_raw_change must align to protein_semantic_change.")
    if tuple(residue_mask.shape) != tuple(shape):
        raise ValueError("protein_residue_mask must align to protein_semantic_change.")
    required = ("valid_mask", "pocket_mask", "log_displacement")
    missing = [name for name in required if name not in structural_targets]
    if missing:
        raise ValueError(f"Structural targets are missing: {missing}")
    valid = structural_targets["valid_mask"]
    pocket = structural_targets["pocket_mask"]
    target = structural_targets["log_displacement"]
    if any(not torch.is_tensor(value) or tuple(value.shape) != tuple(shape)
           for value in (valid, pocket, target)):
        raise ValueError("Structural target masks must align to protein_semantic_change.")
    if valid.dtype != torch.bool or pocket.dtype != torch.bool:
        raise ValueError("Structural target masks must be boolean tensors.")
    sample_values, protein_values = list(sample_ids), list(protein_ids)
    if len(sample_values) != semantic.size(0) or len(protein_values) != semantic.size(0):
        raise ValueError("sample_ids and protein_ids must be batch-sized sequences.")
    if not torch.is_tensor(pocket_rmsd) or pocket_rmsd.shape != (semantic.size(0),):
        raise ValueError("pocket_rmsd must be a batch-sized tensor.")
    selected = residue_mask & valid & pocket
    if not bool(selected.any(dim=1).all()):
        raise ValueError("Each sample must contain at least one valid pocket residue.")
    displacement = torch.expm1(target.to(dtype=semantic.dtype, device=semantic.device))
    if not bool(torch.isfinite(raw_change[selected]).all()):
        raise ValueError("High-dimensional raw changes must be finite.")
    if not bool(torch.isfinite(semantic[selected]).all()):
        raise ValueError("High-dimensional semantic changes must be finite.")
    if not bool(torch.isfinite(displacement[selected]).all()) or bool((displacement[selected] < 0).any()):
        raise ValueError("C-alpha displacement must be finite and non-negative.")
    if not bool(torch.isfinite(pocket_rmsd).all()):
        raise ValueError("pocket_rmsd must be finite.")
    gearnet = structural_targets.get("gearnet_change")
    if gearnet is None:
        gearnet = semantic.new_zeros((*semantic.shape[:2], 0))
    if not torch.is_tensor(gearnet) or gearnet.ndim != 3 or tuple(gearnet.shape[:2]) != tuple(shape):
        raise ValueError("gearnet_change must have aligned shape [B, L, D].")
    if gearnet.size(-1) and not bool(torch.isfinite(gearnet[selected]).all()):
        raise ValueError("GearNet changes must be finite.")
    rows = torch.nonzero(selected, as_tuple=False)
    batch_index, token_index = rows[:, 0], rows[:, 1]
    return {
        "sample_id": np.asarray([str(sample_values[i]) for i in batch_index.tolist()]),
        "protein_id": np.asarray([str(protein_values[i]) for i in batch_index.tolist()]),
        "residue_index": (token_index - 1).detach().cpu().numpy().astype(np.int64),
        "protein_raw_change": raw_change[batch_index, token_index].detach().cpu().numpy().astype(np.float32),
        "protein_semantic_change": semantic[batch_index, token_index].detach().cpu().numpy().astype(np.float32),
        "gearnet_change": gearnet[batch_index, token_index].detach().cpu().numpy().astype(np.float32),
        "ca_displacement": displacement[batch_index, token_index].detach().cpu().numpy().astype(np.float32),
        "pocket_rmsd": pocket_rmsd[batch_index].detach().cpu().numpy().astype(np.float32),
    }


def _concat_high_dimensional_response_cache(chunks):
    if not chunks:
        raise ValueError("High-dimensional response cache has no chunks.")
    keys = tuple(chunks[0])
    if any(tuple(chunk) != keys for chunk in chunks):
        raise ValueError("High-dimensional response cache chunks have inconsistent keys.")
    cache = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}
    order = np.lexsort((cache["residue_index"], cache["sample_id"]))
    cache = {key: value[order] for key, value in cache.items()}
    keys_array = np.stack((cache["sample_id"].astype(str), cache["residue_index"].astype(str)), axis=1)
    if len(np.unique(keys_array, axis=0)) != len(keys_array):
        raise ValueError("High-dimensional response cache has duplicate sample/residue keys.")
    return cache

def extract_model_batch(model, model_inputs, expected_lengths, return_intermediates=False):
    """Extract latent responses without forwarding structure-derived targets."""
    unexpected = set(model_inputs).difference(ALLOWED_MODEL_INPUT_KEYS)
    if unexpected:
        raise ValueError(
            f"Structure-derived or unknown model inputs: {sorted(unexpected)}"
        )
    residue_mask = model_inputs["protein_mask"]
    inference_mask = dti_inference_mask_from_residue_mask(residue_mask)
    output = model(
        input_drugs=model_inputs["input_drugs"],
        input_proteins=model_inputs["input_proteins"],
        pr_mask=inference_mask,
        protein_flex=model_inputs["protein_flex"],
        drug_flex=model_inputs["drug_flex"],
        protein_residue_mask=residue_mask,
        return_intermediates=True,
    )
    values = compute_latent_whole(
        output["protein_before"],
        output["protein_after"],
        output["protein_residue_mask"],
        expected_lengths,
    )
    return (values, output) if return_intermediates else values



CHECKPOINT_REQUIRED_FIELDS = {
    "format_version",
    "model_state_dict",
    "model_config",
    "optimization_provenance",
    "best_epoch",
    "best_validation_auroc",
}


OPTIMIZATION_PROVENANCE_FIELDS = {
    "configured_batch_size",
    "configured_gradient_accumulation_steps",
    "gradient_accumulation_implemented",
    "optimizer_step_interval_batches",
    "actual_effective_batch_size",
}


def validate_checkpoint_payload(payload, checkpoint_profile="flex_only"):
    """Validate the complete extraction checkpoint before model construction."""
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping.")
    missing = CHECKPOINT_REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")

    format_version = payload["format_version"]
    if (
        not isinstance(format_version, Integral)
        or isinstance(format_version, (bool, np.bool_))
        or format_version != 1
    ):
        raise ValueError(f"Unsupported checkpoint format: {format_version}")

    if not isinstance(payload["model_state_dict"], Mapping):
        raise ValueError("Checkpoint model_state_dict must be a mapping.")

    model_config = payload["model_config"]
    if checkpoint_profile == "flex_only":
        validate_flex_only_config(model_config)
    elif checkpoint_profile == "structure_supervised":
        validate_structure_supervised_config(model_config)
    elif checkpoint_profile == "systematic_ablation":
        validate_systematic_ablation_config(model_config)
    else:
        raise ValueError(
            "checkpoint_profile must be flex_only, structure_supervised, or "
            "systematic_ablation."
        )

    provenance = payload["optimization_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("Checkpoint optimization_provenance must be a mapping.")
    missing_provenance = OPTIMIZATION_PROVENANCE_FIELDS.difference(provenance)
    extra_provenance = set(provenance).difference(OPTIMIZATION_PROVENANCE_FIELDS)
    if missing_provenance or extra_provenance:
        raise ValueError(
            "Checkpoint optimization_provenance must contain exactly the required "
            f"fields; missing={sorted(missing_provenance)}, "
            f"extra={sorted(extra_provenance)}."
        )

    for field in (
        "configured_batch_size",
        "configured_gradient_accumulation_steps",
    ):
        value = provenance[field]
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"Checkpoint optimization_provenance {field} must be a positive "
                "built-in integer."
            )
    if provenance["gradient_accumulation_implemented"] is not False:
        raise ValueError(
            "Checkpoint optimization_provenance "
            "gradient_accumulation_implemented must be exactly False."
        )
    step_interval = provenance["optimizer_step_interval_batches"]
    if type(step_interval) is not int or step_interval != 1:
        raise ValueError(
            "Checkpoint optimization_provenance optimizer_step_interval_batches "
            "must be the built-in integer 1."
        )
    effective_batch = provenance["actual_effective_batch_size"]
    if (
        type(effective_batch) is not int
        or effective_batch != provenance["configured_batch_size"]
    ):
        raise ValueError(
            "Checkpoint optimization_provenance actual_effective_batch_size must "
            "be a built-in integer equal to configured_batch_size."
        )

    best_epoch = payload["best_epoch"]
    if type(best_epoch) is not int or best_epoch <= 0:
        raise ValueError("Checkpoint best_epoch must be a positive built-in integer.")
    best_auroc = payload["best_validation_auroc"]
    if (
        not isinstance(best_auroc, Real)
        or isinstance(best_auroc, (bool, np.bool_))
        or not np.isfinite(best_auroc)
    ):
        raise ValueError("Checkpoint best_validation_auroc must be finite numeric.")
    return payload


def load_flex_only_checkpoint(path, device):
    """Strictly load an approved flex-only checkpoint in evaluation mode."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload, checkpoint_profile="flex_only")
    model_config = payload["model_config"]
    model = TAPB(model_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload



def _config_with_semantic_projection_dimension(payload):
    """Recover optional semantic-projection width from a persisted state dict."""
    config = dict(payload["model_config"])
    weight = payload["model_state_dict"].get("protein_semantic_projection.weight")
    if weight is None:
        return config
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] <= 0:
        raise ValueError("Checkpoint semantic projection weight must be a non-empty matrix.")
    inferred_dim = int(weight.shape[0])
    configured_dim = config.get("structure_gearnet_semantic_dim", 0)
    if configured_dim not in (0, inferred_dim):
        raise ValueError(
            "Checkpoint semantic projection dimension conflicts with model_config."
        )
    config["structure_gearnet_semantic_dim"] = inferred_dim
    return config


def load_structure_supervised_checkpoint(path, device):
    """Strictly load an approved structure-supervised checkpoint in evaluation mode."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload, checkpoint_profile="structure_supervised")
    model = TAPB(_config_with_semantic_projection_dimension(payload))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def load_systematic_ablation_checkpoint(path, device):
    """Strictly load one formal V1--V9 systematic-ablation checkpoint."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload, checkpoint_profile="systematic_ablation")
    model = TAPB(_config_with_semantic_projection_dimension(payload))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def sha256_file(path):
    """Return a hexadecimal SHA-256 digest without loading the file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def sha256_path(path):
    """Hash a file or a directory tree deterministically by relative content."""
    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.is_file():
        return sha256_file(resolved)
    if not resolved.is_dir():
        raise ValueError(f"Input artifact is neither a file nor directory: {resolved}")

    digest = hashlib.sha256()
    files = sorted(
        (entry for entry in resolved.rglob("*") if entry.is_file()),
        key=lambda entry: entry.relative_to(resolved).as_posix(),
    )
    for entry in files:
        relative = entry.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(entry)))
    return digest.hexdigest()


def _artifact_metadata(path):
    resolved = Path(path).expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "kind": "directory" if resolved.is_dir() else "file",
        "sha256": sha256_path(resolved),
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-profile", choices=("flex_only", "structure_supervised", "systematic_ablation"),
        default="flex_only",
    )
    parser.add_argument("--include-pocket-residue-responses", action="store_true")
    parser.add_argument("--save-high-dimensional-response-cache", action="store_true")
    parser.add_argument("--gearnet-root", default=None)
    parser.add_argument(
        "--feature-manifest",
        default="datasets/apo_holo/features/feature_manifest.csv",
    )
    parser.add_argument("--apo-root", default="datasets/apo_holo/processed")
    parser.add_argument(
        "--esm-cache", default="datasets/apo_holo/features/esm2_features.pkl"
    )
    parser.add_argument(
        "--dynamine",
        default="datasets/apo_holo/features/protein_flexibility/dynamine.csv",
    )
    parser.add_argument(
        "--ntabs",
        default="datasets/apo_holo/features/drug_flexibility/ntabs.csv",
    )
    parser.add_argument("--tokenizer", default="models/drug/molformer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6)
    return parser



RESULT_COLUMNS = [
    "sample_id",
    "protein_id",
    "ligand_id",
    "smiles",
    "protein_length",
    "valid_residue_count",
    "pocket_size",
    "structural_rmsd_recomputed",
    "structural_rmsd_official",
    "latent_whole",
]



def run_extraction(args):
    """Run deterministic, leakage-guarded latent extraction and write its audit."""
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.limit is not None and args.limit < 0:
        raise ValueError("limit must be nonnegative.")

    configure_deterministic_inference(args.seed)
    device, device_resolution = resolve_inference_device(args.device)

    manifest = pd.read_csv(args.feature_manifest)
    eligible, audit = filter_validation_cohort(manifest, max_length=510)
    eligible = eligible.sort_values("sample_id", kind="stable")
    if getattr(args, "gearnet_root", None):
        eligible, gearnet_audit = filter_gearnet_cached_samples(
            eligible, args.gearnet_root
        )
        audit = pd.concat([audit, gearnet_audit], ignore_index=True)
    if eligible.empty:
        raise ValueError("No eligible samples remain after GearNet-cache filtering.")
    total_eligible = len(eligible)
    if args.limit is not None:
        eligible = eligible.head(int(args.limit)).copy()
    sequence_lengths = validate_selected_sequence_lengths(eligible)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_profile = getattr(args, "checkpoint_profile", "flex_only")
    if checkpoint_profile == "flex_only":
        model, checkpoint = load_flex_only_checkpoint(args.checkpoint, device)
    elif checkpoint_profile == "structure_supervised":
        model, checkpoint = load_structure_supervised_checkpoint(args.checkpoint, device)
    else:
        model, checkpoint = load_systematic_ablation_checkpoint(
            args.checkpoint, device
        )
    include_pocket_residue_responses = bool(
        getattr(args, "include_pocket_residue_responses", False)
    )
    save_high_dimensional_response_cache = bool(
        getattr(args, "save_high_dimensional_response_cache", False)
    )
    provenance_fields = {"model_config", "optimization_provenance"}
    missing_provenance = sorted(provenance_fields.difference(checkpoint))
    if missing_provenance:
        raise ValueError(
            f"Checkpoint is missing extraction provenance: {missing_provenance}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )
    row_by_id = eligible.set_index(
        eligible["sample_id"].astype(str), drop=False
    )
    records = []
    pocket_records = []
    cache_workspace = tempfile.TemporaryDirectory() if save_high_dimensional_response_cache else None
    cache_chunks = []
    with tempfile.TemporaryDirectory() as temporary:
        split_csv = Path(temporary) / "eligible.csv"
        eligible.to_csv(split_csv, index=False)
        dataset = ConformationDataset(
            split_csv=split_csv,
            apo_root=Path(args.apo_root),
            esm_cache=Path(args.esm_cache),
            dynamine_path=Path(args.dynamine),
            ntabs_path=Path(args.ntabs),
            max_real_residues=510,
            gearnet_root=None if getattr(args, "gearnet_root", None) is None else Path(args.gearnet_root),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=make_conformation_collate(tokenizer),
        )
        with torch.no_grad():
            for batch in loader:
                moved = batch.to(device)
                lengths = [sequence_lengths[str(sid)] for sid in batch.sample_ids]
                extracted = extract_model_batch(
                    model, moved.model_inputs, lengths,
                    return_intermediates=(include_pocket_residue_responses or save_high_dimensional_response_cache),
                )
                if include_pocket_residue_responses or save_high_dimensional_response_cache:
                    values, output = extracted
                    required_targets = {"valid_mask", "pocket_mask", "log_displacement"}
                    missing_targets = required_targets.difference(moved.targets)
                    if missing_targets:
                        raise RuntimeError(
                            f"Structural batch is missing residue targets: {sorted(missing_targets)}"
                        )
                    protein_ids = [

                        str(row_by_id.loc[str(sample_id), "protein_id"])
                        for sample_id in batch.sample_ids
                    ]
                    if save_high_dimensional_response_cache:
                        cache = build_high_dimensional_response_cache_records(
                            output, moved.targets, batch.sample_ids, protein_ids,
                            moved.targets["pocket_rmsd"],
                        )
                        chunk_path = Path(cache_workspace.name) / (
                            f"high_dimensional_response_cache_{len(cache_chunks):05d}.npz"
                        )
                        np.savez_compressed(chunk_path, **cache)
                        cache_chunks.append(chunk_path)
                    pocket_records.extend(compute_pocket_residue_records(
                        output["protein_before"], output["protein_after"],
                        output["protein_residue_mask"], moved.targets["valid_mask"],
                        moved.targets["pocket_mask"], moved.targets["log_displacement"],
                        batch.sample_ids, protein_ids,
                    ))
                else:
                    values = extracted
                for sample_id, latent_whole in zip(batch.sample_ids, values):
                    row = row_by_id.loc[str(sample_id)]
                    actual_length = sequence_lengths[str(sample_id)]
                    records.append(
                        {
                            "sample_id": str(sample_id),
                            "protein_id": str(row["protein_id"]),
                            "ligand_id": str(row["drug_id"]),
                            "smiles": str(row["SMILES"]),
                            "protein_length": actual_length,
                            "valid_residue_count": actual_length,
                            "pocket_size": int(row["window_valid_pocket_count"]),
                            "structural_rmsd_recomputed": float(
                                row["recomputed_pocket_ca_rmsd"]
                            ),
                            "structural_rmsd_official": float(
                                row["official_pocket_rmsd"]
                            ),
                            "latent_whole": float(latent_whole),
                        }
                    )

    result = pd.DataFrame.from_records(records, columns=RESULT_COLUMNS)
    expected_sample_ids = eligible["sample_id"].astype(str).tolist()
    actual_sample_ids = result["sample_id"].tolist()
    if actual_sample_ids != expected_sample_ids:
        raise RuntimeError(
            "Extracted output ordered sample IDs do not exactly match the "
            "eligible cohort."
        )
    primary = result[["structural_rmsd_recomputed", "latent_whole"]].to_numpy()
    if not np.isfinite(primary).all():
        raise RuntimeError("Primary output contains non-finite values.")
    pocket_result = None
    if include_pocket_residue_responses:
        pocket_columns = [
            "sample_id", "protein_id", "residue_index",
            "latent_residue_response", "structural_ca_displacement",
        ]
        pocket_result = pd.DataFrame.from_records(pocket_records, columns=pocket_columns)
        if pocket_result.empty:
            raise RuntimeError("Pocket-residue extraction produced no supervised rows.")
        if pocket_result.duplicated(["sample_id", "residue_index"]).any():
            raise RuntimeError("Pocket-residue output contains duplicate sample/residue rows.")
        numeric = pocket_result[["latent_residue_response", "structural_ca_displacement"]].to_numpy(float)
        if not np.isfinite(numeric).all() or bool((numeric[:, 1] < 0).any()):
            raise RuntimeError("Pocket-residue output contains invalid response or displacement values.")

    cache_metadata = None
    if save_high_dimensional_response_cache:
        if not cache_chunks:
            raise RuntimeError("High-dimensional response cache produced no chunks.")
        cache_metadata = {
            "format_version": 1,
            "cache_kind": "valid_pocket_residue_high_dimensional_response",
            "arrays": ["sample_id", "protein_id", "residue_index", "protein_raw_change", "protein_semantic_change", "gearnet_change", "ca_displacement", "pocket_rmsd"],
            "chunks": [
                {"filename": path.name, "sha256": sha256_file(path)}
                for path in cache_chunks
            ],
        }
        with np.load(cache_chunks[0], allow_pickle=False) as first_chunk:
            cache_metadata["raw_change_dim"] = int(first_chunk["protein_raw_change"].shape[1])
            cache_metadata["sequence_change_dim"] = int(first_chunk["protein_semantic_change"].shape[1])
            cache_metadata["gearnet_change_dim"] = int(first_chunk["gearnet_change"].shape[1])
        cache_metadata["n_valid_pocket_rows"] = int(sum(
            np.load(path, allow_pickle=False)["residue_index"].shape[0]
            for path in cache_chunks
        ))
    artifact_paths = {
        "checkpoint": args.checkpoint,
        "feature_manifest": args.feature_manifest,
        "apo_root": args.apo_root,
        "esm_cache": args.esm_cache,
        "dynamine": args.dynamine,
        "ntabs": args.ntabs,
        "tokenizer": args.tokenizer,
        **({"gearnet_root": args.gearnet_root} if getattr(args, "gearnet_root", None) else {}),
    }
    input_artifacts = {
        name: _artifact_metadata(path) for name, path in artifact_paths.items()
    }
    cuda_available = bool(torch.cuda.is_available())
    gpu_names = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ] if cuda_available else []

    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "checkpoint_format_version": checkpoint["format_version"],
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_auroc": checkpoint["best_validation_auroc"],
        "model_config": checkpoint["model_config"],
        "optimization_provenance": checkpoint["optimization_provenance"],
        "checkpoint_profile": checkpoint_profile,
        "include_pocket_residue_responses": include_pocket_residue_responses,
        "save_high_dimensional_response_cache": save_high_dimensional_response_cache,
        "high_dimensional_response_cache": cache_metadata,
        "pocket_residue_rows": 0 if pocket_result is None else len(pocket_result),
        "input_artifacts": input_artifacts,
        "feature_manifest_sha256": input_artifacts["feature_manifest"]["sha256"],
        "seed": args.seed,
        "device": str(device),
        "requested_device": device_resolution["requested_device"],
        "resolved_device": device_resolution["resolved_device"],
        "device_resolution": device_resolution,
        "batch_size": args.batch_size,
        "max_protein_length": 510,
        "total_eligible": total_eligible,
        "processed": len(result),
        "limit": args.limit,
        "filtering_rules": {
            "feature_eligible": True,
            "protein_overlap": False,
            "drug_overlap": False,
            "max_original_length": 510,
            "require_finite_recomputed_pocket_ca_rmsd": True,
            "require_original_length_match": True,
            "sort_key": "sample_id",
            "limit_applied_after_sort": True,
        },
        "runtime_environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "cuda_available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": len(gpu_names),
            "gpu_names": gpu_names,
        },
    }
    metadata_json = serialize_json_value(metadata)
    output_filenames = [
        "exclusion_audit.csv",
        "apo2mol_latent_response.csv",
        "run_metadata.json",
    ]
    if pocket_result is not None:
        output_filenames.append("apo2mol_pocket_residue_response.csv")
    if cache_metadata is not None:
        output_filenames.append("high_dimensional_response_cache_metadata.json")
        output_filenames.extend(chunk["filename"] for chunk in cache_metadata["chunks"])
    with output_set_transaction(output_dir, output_filenames) as staged:
        audit.to_csv(staged["exclusion_audit.csv"], index=False)
        result.to_csv(staged["apo2mol_latent_response.csv"], index=False)
        if pocket_result is not None:
            pocket_result.to_csv(staged["apo2mol_pocket_residue_response.csv"], index=False)
        if cache_metadata is not None:
            sidecar = dict(cache_metadata)
            sidecar.update({
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "feature_manifest_sha256": metadata["feature_manifest_sha256"],
                "checkpoint_profile": checkpoint_profile,
                "best_epoch": metadata["best_epoch"],
            })
            for source, chunk in zip(cache_chunks, cache_metadata["chunks"]):
                shutil.copyfile(source, staged[chunk["filename"]])
            staged["high_dimensional_response_cache_metadata.json"].write_text(
                serialize_json_value(sidecar), encoding="utf-8"
            )
        staged["run_metadata.json"].write_text(metadata_json, encoding="utf-8")
    if cache_workspace is not None:
        cache_workspace.cleanup()
    return result



def main():
    run_extraction(build_parser().parse_args())


if __name__ == "__main__":
    main()
