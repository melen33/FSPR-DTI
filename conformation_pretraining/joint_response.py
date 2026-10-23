"""Leakage-safe structural response batch adapter for TAPB."""

from __future__ import annotations

import random

import torch

from structural_validation.extract_latent_response import dti_inference_mask_from_residue_mask

from .dataset import ConformationBatch
from .response_objectives import (
    ResponseStructureLoss,
    compute_ca_replacement_loss,
    compute_pocket_gearnet_response_loss,
    compute_global_gearnet_response_loss,
    compute_pocket_gearnet_direct_distillation_loss,
    compute_pocket_gearnet_pair_distillation_loss,
    compute_pocket_gearnet_semantic_loss,
    compute_pocket_gearnet_relational_contrast_loss,
    compute_pocket_residue_response_loss,
    compute_pocket_residue_rank_loss,
    compute_pocket_site_ranking_loss,
    compute_response_structure_loss,
)


def _apply_dti_style_token_dropout(batch, drop_prob: float):
    """Apply the exact shared-index deletion rule used by DTI collation."""
    if not 0.0 <= float(drop_prob) < 1.0:
        raise ValueError("target_random_deletion_ratio must be in [0, 1)")
    if float(drop_prob) == 0.0:
        return batch
    inputs = dict(batch.model_inputs)
    proteins = inputs["input_proteins"]
    residue_mask = inputs["protein_mask"]
    full_attention_mask = dti_inference_mask_from_residue_mask(residue_mask)
    sequence_length = proteins.size(1)
    retained_count = int(max(1, sequence_length * (1.0 - float(drop_prob))))
    retained = sorted(random.sample(range(1, sequence_length), retained_count))
    retained.insert(0, 0)
    index = torch.tensor(retained, dtype=torch.long, device=proteins.device)
    inputs["input_proteins"] = proteins.index_select(1, index)
    inputs["protein_mask"] = residue_mask.index_select(1, index)
    inputs["protein_flex"] = inputs["protein_flex"].index_select(1, index)
    inputs["protein_attention_mask"] = full_attention_mask.index_select(1, index)
    targets = dict(batch.targets)
    for name in ("log_displacement", "log_gearnet_response", "contact_distance", "valid_mask", "pocket_mask"):
        if name in targets:
            targets[name] = targets[name].index_select(1, index)
    return ConformationBatch(inputs, targets, batch.sample_ids)

def _perturb_structural_targets(targets, mode: str, seed: int):
    """Swap structural labels within sampler-guaranteed same-protein pairs."""
    if mode in {"none", "global_random_derangement"}:
        return targets
    if mode != "batch_shuffle":
        raise ValueError(
            "structure_supervision_label_perturbation must be 'none', "
            "'batch_shuffle', or 'global_random_derangement'"
        )
    protein_ids = tuple(targets.get("protein_ids", ()))
    if not protein_ids:
        raise KeyError("batch_shuffle requires structural target protein_ids")
    site_ids = tuple(targets.get("site_ids", ()))
    index = torch.arange(len(protein_ids))
    groups = {}
    for position, protein_id in enumerate(protein_ids):
        key = (
            str(protein_id), str(site_ids[position])
        ) if len(site_ids) == len(protein_ids) else str(protein_id)
        groups.setdefault(key, []).append(position)
    for positions in groups.values():
        if len(positions) >= 2:
            # A cyclic derangement preserves protein/site membership while ensuring
            # every candidate receives a different ligand teacher target.
            index[positions] = torch.tensor(
                positions[1:] + positions[:1], dtype=torch.long
            )
    shuffled = dict(targets)
    for name in ("log_displacement", "log_gearnet_response", "contact_distance", "gearnet_change", "global_gearnet_change", "valid_mask", "pocket_mask", "log_global_targets", "pocket_rmsd"):
        if name in shuffled:
            shuffled[name] = shuffled[name].index_select(0, index.to(shuffled[name].device))
    return shuffled


def structural_response_loss_from_batch(
    model,
    batch,
    device: torch.device,
    margin: float = 0.1,
    ranking_weight: float = 1.0,
    target_random_deletion_ratio: float = 0.0,
    mode: str = "scalar_rmsd",
    gearnet_weight: float = 0.0,
    gearnet_semantic_weight: float = 0.0,
    gearnet_global_weight: float = 0.0,
    gearnet_relational_weight: float = 0.0,
    gearnet_relational_temperature: float = 0.1,
    gearnet_direct_weight: float = 0.0,
    gearnet_pair_weight: float = 0.0,
    gearnet_pair_temperature: float = 0.5,
    site_ranking_weight: float = 0.0,
    site_ranking_min_gap: float = 0.1,
    residue_ranking_weight: float = 0.0,
    residue_ranking_min_gap: float = 0.10,
    residue_ranking_temperature: float = 0.20,
    label_perturbation: str = "none",
    label_perturbation_seed: int = 0,
    contact_weight_mode: str = "none",
    contact_weight_alpha: float = 1.0,
    contact_weight_sigma: float = 4.5,
    contact_weight_shuffle_seed: int = 0,
):
    """Forward allowed sequence inputs only, then apply target-only structural loss."""
    if mode not in {"scalar_rmsd", "pocket_residue", "gearnet_only", "full_protein_residue", "pocket_rms"}:
        raise ValueError("mode must be one of: scalar_rmsd, pocket_residue, gearnet_only")
    if float(gearnet_weight) < 0.0:
        raise ValueError("gearnet_weight must be non-negative")
    if float(gearnet_semantic_weight) < 0.0:
        raise ValueError("gearnet_semantic_weight must be non-negative")
    if float(gearnet_global_weight) < 0.0:
        raise ValueError("gearnet_global_weight must be non-negative")
    if float(gearnet_relational_weight) < 0.0:
        raise ValueError("gearnet_relational_weight must be non-negative")
    if float(gearnet_relational_temperature) <= 0.0:
        raise ValueError("gearnet_relational_temperature must be positive")
    if float(gearnet_direct_weight) < 0.0:
        raise ValueError("gearnet_direct_weight must be non-negative")
    if float(gearnet_pair_weight) < 0.0:
        raise ValueError("gearnet_pair_weight must be non-negative")
    if float(gearnet_pair_temperature) <= 0.0:
        raise ValueError("gearnet_pair_temperature must be positive")
    if float(gearnet_semantic_weight) > 0.0 and float(gearnet_relational_weight) > 0.0:
        raise ValueError("direct and relational GearNet semantic weights are mutually exclusive")
    if float(site_ranking_weight) < 0.0:
        raise ValueError("site_ranking_weight must be non-negative")
    if float(site_ranking_min_gap) < 0.0:
        raise ValueError("site_ranking_min_gap must be non-negative")
    if float(residue_ranking_weight) < 0.0:
        raise ValueError("residue_ranking_weight must be non-negative")
    if float(residue_ranking_min_gap) < 0.0:
        raise ValueError("residue_ranking_min_gap must be non-negative")
    if float(residue_ranking_temperature) <= 0.0:
        raise ValueError("residue_ranking_temperature must be positive")
    if float(site_ranking_weight) > 0.0 and mode != "pocket_residue":
        raise ValueError("site_ranking_weight requires pocket_residue structural supervision")
    if float(gearnet_weight) > 0.0 and mode not in {"pocket_residue", "gearnet_only"}:
        raise ValueError("gearnet_weight requires pocket_residue or gearnet_only structural supervision")
    if float(gearnet_semantic_weight) > 0.0 and mode != "pocket_residue":
        raise ValueError("gearnet_semantic_weight requires pocket_residue structural supervision")
    if float(gearnet_relational_weight) > 0.0 and mode != "pocket_residue":
        raise ValueError("gearnet_relational_weight requires pocket_residue structural supervision")
    if (float(gearnet_direct_weight) > 0.0 or float(gearnet_pair_weight) > 0.0) and mode != "pocket_residue":
        raise ValueError("response-bound GearNet distillation requires pocket_residue structural supervision")
    if mode == "gearnet_only" and float(gearnet_weight) <= 0.0:
        raise ValueError("gearnet_only requires a positive gearnet_weight")
    # Per-residue targets must remain aligned with their input residues. The DTI
    # batch retains its normal deletion regularization; only the structural step
    # disables deletion in pocket-residue mode.
    effective_drop_ratio = 0.0 if mode in {"pocket_residue", "gearnet_only", "full_protein_residue", "pocket_rms"} else target_random_deletion_ratio
    moved = _apply_dti_style_token_dropout(
        batch.to(device), effective_drop_ratio
    )
    inputs = moved.model_inputs
    residue_mask = inputs["protein_mask"]
    pr_mask = inputs.get("protein_attention_mask")
    if pr_mask is None:
        pr_mask = dti_inference_mask_from_residue_mask(residue_mask)
    output = model(
        input_drugs=inputs["input_drugs"],
        input_proteins=inputs["input_proteins"],
        pr_mask=pr_mask,
        drug_flex=inputs["drug_flex"],
        protein_flex=inputs["protein_flex"],
        return_intermediates=True,
        protein_residue_mask=residue_mask,
        protein_ids=tuple(moved.targets["protein_ids"]),
    )
    required = {"protein_before", "protein_after", "protein_residue_mask"}
    missing = required.difference(output)
    if missing:
        raise KeyError(f"TAPB structural response output is missing: {sorted(missing)}")
    if not torch.equal(output["protein_residue_mask"], residue_mask):
        raise ValueError("TAPB returned a residue mask that differs from the batch mask")
    structural_targets = _perturb_structural_targets(
        moved.targets, label_perturbation, label_perturbation_seed
    )
    if mode in {"full_protein_residue", "pocket_rms"}:
        return compute_ca_replacement_loss(
            output["protein_before"], output["protein_after"],
            structural_targets["valid_mask"], structural_targets["pocket_mask"],
            structural_targets["log_displacement"], output["protein_residue_mask"], mode,
        )
    if mode == "scalar_rmsd":
        return compute_response_structure_loss(
            output["protein_before"], output["protein_after"], output["protein_residue_mask"],
            structural_targets["pocket_rmsd"], structural_targets["protein_ids"],
            margin=margin, ranking_weight=ranking_weight,
        )
    if mode == "gearnet_only":
        required_targets = {"log_gearnet_response", "valid_mask", "pocket_mask"}
        missing_targets = required_targets.difference(structural_targets)
        if missing_targets:
            raise KeyError(f"Structural batch is missing GearNet targets: {sorted(missing_targets)}")
        teacher_loss = compute_pocket_gearnet_response_loss(
            output["protein_before"], output["protein_after"],
            structural_targets["valid_mask"], structural_targets["pocket_mask"],
            structural_targets["log_gearnet_response"], output["protein_residue_mask"],
        )
        zero = teacher_loss * 0.0
        return ResponseStructureLoss(
            total=float(gearnet_weight) * teacher_loss,
            magnitude=zero,
            ranking=zero,
            response_score=zero.expand(output["protein_before"].shape[0]),
            ranking_pair_count=0,
            semantic=zero,
        )
    if mode == "pocket_residue":
        required_targets = {"log_displacement", "valid_mask", "pocket_mask"}
        missing_targets = required_targets.difference(structural_targets)
        if missing_targets:
            raise KeyError(f"Structural batch is missing residue targets: {sorted(missing_targets)}")
        geometry_loss = compute_pocket_residue_response_loss(
            output["protein_before"], output["protein_after"],
            structural_targets["valid_mask"], structural_targets["pocket_mask"],
            structural_targets["log_displacement"], output["protein_residue_mask"],
            contact_distance=structural_targets.get("contact_distance"),
            contact_weight_mode=contact_weight_mode,
            contact_weight_alpha=contact_weight_alpha,
            contact_weight_sigma=contact_weight_sigma,
            contact_weight_shuffle_seed=contact_weight_shuffle_seed,
            contact_weight_sample_ids=moved.sample_ids,
        )
        ranking = geometry_loss.ranking
        response_score = geometry_loss.response_score
        ranking_pair_count = geometry_loss.ranking_pair_count
        total = geometry_loss.total
        residue_ranking = geometry_loss.magnitude * 0.0
        residue_ranking_pair_count = 0
        residue_ranking_skipped_sample_count = 0
        if float(residue_ranking_weight) > 0.0:
            residue_ranking, residue_ranking_pair_count, residue_ranking_skipped_sample_count = (
                compute_pocket_residue_rank_loss(
                    output["protein_before"], output["protein_after"],
                    structural_targets["valid_mask"], structural_targets["pocket_mask"],
                    structural_targets["log_displacement"], output["protein_residue_mask"],
                    min_gap=residue_ranking_min_gap,
                    temperature=residue_ranking_temperature,
                )
            )
            total = total + float(residue_ranking_weight) * residue_ranking
        if float(site_ranking_weight) > 0.0:
            required_rank_targets = {"pocket_rmsd", "protein_ids", "site_ids", "drug_ids"}
            missing_rank_targets = required_rank_targets.difference(structural_targets)
            if missing_rank_targets:
                raise KeyError(
                    f"Structural batch is missing site-ranking targets: {sorted(missing_rank_targets)}"
                )
            ranking, response_score, ranking_pair_count = compute_pocket_site_ranking_loss(
                output["protein_before"], output["protein_after"],
                structural_targets["valid_mask"], structural_targets["pocket_mask"],
                structural_targets["pocket_rmsd"], structural_targets["protein_ids"],
                structural_targets["site_ids"], structural_targets["drug_ids"],
                min_gap=site_ranking_min_gap,
                residue_mask=output["protein_residue_mask"],
            )
            total = total + float(site_ranking_weight) * ranking
        if float(gearnet_weight) > 0.0:
            if "log_gearnet_response" not in structural_targets:
                raise KeyError("Structural batch is missing GearNet response targets")
            teacher_loss = compute_pocket_gearnet_response_loss(
                output["protein_before"], output["protein_after"],
                structural_targets["valid_mask"], structural_targets["pocket_mask"],
                structural_targets["log_gearnet_response"], output["protein_residue_mask"],
            )
            total = total + float(gearnet_weight) * teacher_loss
        semantic = geometry_loss.total * 0.0
        if float(gearnet_global_weight) > 0.0:
            if "global_gearnet_change" not in structural_targets:
                raise KeyError("Structural batch is missing global GearNet change targets")
            if "protein_semantic_change" not in output:
                raise KeyError("TAPB structural response output is missing protein_semantic_change")
            semantic = semantic + float(gearnet_global_weight) * compute_global_gearnet_response_loss(
                output["protein_semantic_change"], structural_targets["global_gearnet_change"],
                output["protein_residue_mask"],
            )
        if float(gearnet_semantic_weight) > 0.0:
            if "gearnet_change" not in structural_targets:
                raise KeyError("Structural batch is missing GearNet semantic-change targets")
            if "protein_semantic_change" not in output:
                raise KeyError("TAPB structural response output is missing protein_semantic_change")
            semantic = compute_pocket_gearnet_semantic_loss(
                output["protein_semantic_change"], structural_targets["gearnet_change"],
                structural_targets["valid_mask"], structural_targets["pocket_mask"],
                output["protein_residue_mask"],
            )
            total = total + float(gearnet_semantic_weight) * semantic
        if float(gearnet_relational_weight) > 0.0:
            required_relation_targets = {"gearnet_change", "protein_ids", "site_ids"}
            missing_relation_targets = required_relation_targets.difference(structural_targets)
            if missing_relation_targets:
                raise KeyError(
                    f"Structural batch is missing relational GearNet targets: {sorted(missing_relation_targets)}"
                )
            semantic, _ = compute_pocket_gearnet_relational_contrast_loss(
                output["protein_after"] - output["protein_before"],
                structural_targets["gearnet_change"],
                structural_targets["valid_mask"], structural_targets["pocket_mask"],
                structural_targets["protein_ids"], structural_targets["site_ids"],
                residue_mask=output["protein_residue_mask"],
                temperature=gearnet_relational_temperature,
            )
            total = total + float(gearnet_relational_weight) * semantic
        response_semantic = geometry_loss.total * 0.0
        if float(gearnet_direct_weight) > 0.0 or float(gearnet_pair_weight) > 0.0:
            if "gearnet_change" not in structural_targets:
                raise KeyError("Structural batch is missing GearNet semantic-change targets")
            if "protein_gearnet_response_change" not in output:
                raise KeyError(
                    "TAPB structural response output is missing protein_gearnet_response_change"
                )
            student_change = output["protein_gearnet_response_change"]
            if float(gearnet_direct_weight) > 0.0:
                response_semantic = response_semantic + float(gearnet_direct_weight) * (
                    compute_pocket_gearnet_direct_distillation_loss(
                        student_change, structural_targets["gearnet_change"],
                        structural_targets["valid_mask"], structural_targets["pocket_mask"],
                        output["protein_residue_mask"],
                    )
                )
            if float(gearnet_pair_weight) > 0.0:
                response_semantic = response_semantic + float(gearnet_pair_weight) * (
                    compute_pocket_gearnet_pair_distillation_loss(
                        student_change, structural_targets["gearnet_change"],
                        structural_targets["valid_mask"], structural_targets["pocket_mask"],
                        output["protein_residue_mask"], gearnet_pair_temperature,
                    )
                )
        total = total + response_semantic
        semantic = semantic + response_semantic
        return ResponseStructureLoss(
            total=total,
            magnitude=geometry_loss.magnitude,
            ranking=ranking,
            response_score=response_score,
            ranking_pair_count=ranking_pair_count,
            semantic=semantic,
            residue_ranking=residue_ranking,
            residue_ranking_pair_count=residue_ranking_pair_count,
            residue_ranking_skipped_sample_count=residue_ranking_skipped_sample_count,
        )
