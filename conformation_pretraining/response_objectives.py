"""Direct structural calibration losses for TAPB protein response tensors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class ResponseStructureLoss:
    total: torch.Tensor
    magnitude: torch.Tensor
    ranking: torch.Tensor
    response_score: torch.Tensor
    ranking_pair_count: int
    semantic: torch.Tensor
    residue_ranking: torch.Tensor | None = None
    residue_ranking_pair_count: int = 0
    residue_ranking_skipped_sample_count: int = 0


def masked_token_response_score(
    protein_before: torch.Tensor,
    protein_after: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean per-residue L2 interaction displacement for each sample."""
    if protein_before.shape != protein_after.shape or protein_before.dim() != 3:
        raise ValueError("protein_before and protein_after must share shape [B, L, D]")
    if residue_mask.shape != protein_before.shape[:2]:
        raise ValueError("residue_mask must have shape [B, L]")
    mask = residue_mask.to(device=protein_before.device).bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every sample requires at least one real residue")
    token_norm = torch.linalg.vector_norm(protein_after - protein_before, dim=-1)
    weights = mask.to(dtype=token_norm.dtype)
    return (token_norm * weights).sum(dim=1) / weights.sum(dim=1)


def compute_pocket_residue_response_loss(
    protein_before, protein_after, valid_mask, pocket_mask, log_displacement,
    residue_mask=None, contact_distance=None, contact_weight_mode="none",
    contact_weight_alpha=1.0, contact_weight_sigma=4.5,
    contact_weight_shuffle_seed=0, contact_weight_sample_ids=None,
):
    """Align valid pocket response magnitudes to C-alpha displacement.

    Contact distances are target-only structural metadata.  They weight the
    unchanged per-residue C-alpha loss and never enter the TAPB forward path.
    """
    if protein_before.shape != protein_after.shape or protein_before.dim() != 3:
        raise ValueError("protein_before and protein_after must share shape [B, L, D]")
    shape = protein_before.shape[:2]
    if valid_mask.shape != shape or pocket_mask.shape != shape or log_displacement.shape != shape:
        raise ValueError("structural masks and log_displacement must have shape [B, L]")
    supervised = valid_mask.to(protein_before.device).bool() & pocket_mask.to(protein_before.device).bool()
    if residue_mask is not None:
        if residue_mask.shape != shape:
            raise ValueError("residue_mask must have shape [B, L]")
        supervised = supervised & residue_mask.to(protein_before.device).bool()
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every structural sample requires at least one valid pocket residue")
    target = log_displacement.to(device=protein_before.device, dtype=protein_before.dtype)
    if not bool(torch.isfinite(target[supervised]).all()):
        raise ValueError("log_displacement must be finite at valid pocket residues")
    mode = str(contact_weight_mode)
    if mode not in {"none", "contact", "residue_shuffle"}:
        raise ValueError("contact_weight_mode must be none, contact, or residue_shuffle")
    if float(contact_weight_alpha) < 0.0:
        raise ValueError("contact_weight_alpha must be non-negative")
    if float(contact_weight_sigma) <= 0.0:
        raise ValueError("contact_weight_sigma must be positive")
    if mode == "none":
        weights = supervised.to(dtype=protein_before.dtype)
    else:
        if contact_distance is None:
            raise ValueError("contact_distance is required when contact weighting is enabled")
        if contact_distance.shape != shape:
            raise ValueError("contact_distance must have shape [B, L]")
        distance = contact_distance.to(device=protein_before.device, dtype=protein_before.dtype)
        if not bool(torch.isfinite(distance[supervised]).all()) or bool((distance[supervised] < 0).any()):
            raise ValueError("contact_distance must be finite and non-negative at valid pocket residues")
        raw_weights = 1.0 + float(contact_weight_alpha) * torch.exp(
            -distance.square() / (2.0 * float(contact_weight_sigma) ** 2)
        )
        raw_weights = torch.where(supervised, raw_weights, torch.zeros_like(raw_weights))
        if mode == "residue_shuffle":
            sample_ids = contact_weight_sample_ids
            if sample_ids is None or len(sample_ids) != shape[0]:
                raise ValueError("contact_weight_sample_ids must match batch size for residue_shuffle")
            shuffled = raw_weights.clone()
            for batch_index, sample_id in enumerate(sample_ids):
                positions = torch.nonzero(supervised[batch_index], as_tuple=False).flatten()
                if int(positions.numel()) < 2:
                    continue
                digest = hashlib.sha256(
                    f"{int(contact_weight_shuffle_seed)}:{sample_id}".encode("utf-8")
                ).digest()
                # A seed-derived cyclic shift is a deterministic derangement:
                # it retains the exact weight multiset but cannot leave any
                # pocket residue at its own location.
                shift = 1 + (int.from_bytes(digest[8:16], "little") % (int(positions.numel()) - 1))
                source = torch.roll(positions, shifts=int(shift))
                shuffled[batch_index, positions] = raw_weights[batch_index, source]
            raw_weights = shuffled
        counts = supervised.sum(dim=1, keepdim=True).to(dtype=raw_weights.dtype)
        mean_weight = raw_weights.sum(dim=1, keepdim=True) / counts
        weights = raw_weights / mean_weight.clamp_min(torch.finfo(raw_weights.dtype).eps)
    response = torch.linalg.vector_norm(protein_after - protein_before, dim=-1)
    # Normalize within each pair before averaging across the batch: proteins with
    # larger annotated pockets must not receive disproportionate loss weight.
    target_safe = torch.where(supervised, target, torch.zeros_like(target))
    token_loss = functional.smooth_l1_loss(
        torch.log1p(response), target_safe, reduction="none"
    )
    weights = weights.to(dtype=token_loss.dtype)
    magnitude = ((token_loss * weights).sum(dim=1) / weights.sum(dim=1)).mean()
    response_mask = residue_mask if residue_mask is not None else valid_mask
    response_score = masked_token_response_score(protein_before, protein_after, response_mask)
    zero = magnitude * 0.0
    return ResponseStructureLoss(
        total=magnitude, magnitude=magnitude, ranking=zero, response_score=response_score, ranking_pair_count=0, semantic=zero,
    )


def compute_ca_replacement_loss(before, after, valid_mask, pocket_mask, log_displacement,
                                residue_mask, mode):
    """Compare all valid residues, or the RMS response of the annotated pocket."""
    if mode == "full_protein_residue":
        return compute_pocket_residue_response_loss(
            before, after, valid_mask, torch.ones_like(pocket_mask),
            log_displacement, residue_mask,
        )
    if mode != "pocket_rms":
        raise ValueError("Unknown C-alpha replacement mode: %s" % mode)
    mask = valid_mask.bool() & pocket_mask.bool() & residue_mask.bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every sample requires valid pocket residues")
    target = log_displacement.to(before)
    if not bool(torch.isfinite(target[mask]).all()):
        raise ValueError("Non-finite pocket displacement")
    target = torch.expm1(torch.where(mask, target, torch.zeros_like(target)))
    delta = torch.where(mask.unsqueeze(-1), after - before, torch.zeros_like(before))
    counts = mask.sum(dim=1).to(before.dtype)
    # vector_norm has a finite zero subgradient, including exactly zero response.
    response = torch.linalg.vector_norm(delta.flatten(1), dim=1) / counts.sqrt()
    true_rms = torch.linalg.vector_norm(target, dim=1) / counts.sqrt()
    magnitude = functional.smooth_l1_loss(torch.log1p(response), torch.log1p(true_rms))
    zero = magnitude * 0.0
    return ResponseStructureLoss(magnitude, magnitude, zero, response, 0, zero)


def compute_pocket_gearnet_response_loss(
    protein_before, protein_after, valid_mask, pocket_mask, log_gearnet_response,
    residue_mask=None,
):
    """Distil frozen GearNet Apo/holo response magnitudes at valid pocket residues."""
    if protein_before.shape != protein_after.shape or protein_before.dim() != 3:
        raise ValueError("protein_before and protein_after must share shape [B, L, D]")
    shape = protein_before.shape[:2]
    if valid_mask.shape != shape or pocket_mask.shape != shape or log_gearnet_response.shape != shape:
        raise ValueError("structural masks and log_gearnet_response must have shape [B, L]")
    supervised = valid_mask.to(protein_before.device).bool() & pocket_mask.to(protein_before.device).bool()
    if residue_mask is not None:
        if residue_mask.shape != shape:
            raise ValueError("residue_mask must have shape [B, L]")
        supervised = supervised & residue_mask.to(protein_before.device).bool()
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every structural sample requires at least one valid pocket GearNet target")
    target = log_gearnet_response.to(device=protein_before.device, dtype=protein_before.dtype)
    if not bool(torch.isfinite(target[supervised]).all()):
        raise ValueError("log_gearnet_response must be finite at valid pocket residues")
    response = torch.linalg.vector_norm(protein_after - protein_before, dim=-1)
    token_loss = functional.smooth_l1_loss(
        torch.log1p(response), torch.where(supervised, target, torch.zeros_like(target)), reduction="none"
    )
    weights = supervised.to(dtype=token_loss.dtype)
    return ((token_loss * weights).sum(dim=1) / weights.sum(dim=1)).mean()


def compute_global_gearnet_response_loss(
    student_change: torch.Tensor,
    teacher_change: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    """Align pooled full-protein latent response with pooled GearNet Apo/Holo change."""
    if student_change.dim() != 3:
        raise ValueError("student_change must have shape [B, L, D]")
    batch_size, sequence_length, _ = student_change.shape
    if teacher_change.dim() != 2 or teacher_change.shape[0] != batch_size:
        raise ValueError("teacher_change must have shape [B, D]")
    if residue_mask.shape != (batch_size, sequence_length):
        raise ValueError("residue_mask must have shape [B, L]")
    mask = residue_mask.to(device=student_change.device).bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every sample requires at least one real residue")
    weights = mask.to(dtype=student_change.dtype).unsqueeze(-1)
    student = (student_change * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
    teacher = teacher_change.detach().to(device=student.device, dtype=student.dtype)
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("global GearNet response vectors must be finite")
    student = functional.layer_norm(student, (student.shape[-1],))
    teacher = functional.layer_norm(teacher, (teacher.shape[-1],))
    return functional.mse_loss(student, teacher) + (
        1.0 - functional.cosine_similarity(student, teacher, dim=-1)
    ).mean()



def _valid_pocket_supervision_mask(
    valid_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    residue_mask: torch.Tensor | None,
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if valid_mask.shape != shape or pocket_mask.shape != shape:
        raise ValueError("valid_mask and pocket_mask must have shape [B, L]")
    supervised = valid_mask.to(device=device).bool() & pocket_mask.to(device=device).bool()
    if residue_mask is not None:
        if residue_mask.shape != shape:
            raise ValueError("residue_mask must have shape [B, L]")
        supervised &= residue_mask.to(device=device).bool()
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every structural sample requires at least one valid pocket residue")
    return supervised


def compute_pocket_residue_rank_loss(
    protein_before: torch.Tensor,
    protein_after: torch.Tensor,
    valid_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    log_displacement: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    min_gap: float = 0.10,
    temperature: float = 0.20,
) -> tuple[torch.Tensor, int, int]:
    """Rank valid pocket responses by C-alpha displacement within each sample."""
    if float(min_gap) < 0.0:
        raise ValueError("min_gap must be non-negative")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if protein_before.shape != protein_after.shape or protein_before.dim() != 3:
        raise ValueError("protein_before and protein_after must share shape [B, L, D]")
    shape = protein_before.shape[:2]
    if (
        valid_mask.shape != shape
        or pocket_mask.shape != shape
        or log_displacement.shape != shape
    ):
        raise ValueError("structural masks and log_displacement must have shape [B, L]")
    supervised = _valid_pocket_supervision_mask(
        valid_mask, pocket_mask, residue_mask, shape, protein_before.device
    )
    target = log_displacement.to(
        device=protein_before.device, dtype=protein_before.dtype
    )
    if not bool(torch.isfinite(target[supervised]).all()):
        raise ValueError("log_displacement must be finite at valid pocket residues")
    score = torch.log1p(torch.linalg.vector_norm(protein_after - protein_before, dim=-1))
    losses: list[torch.Tensor] = []
    retained_pair_count = 0
    skipped_sample_count = 0
    for batch_index in range(score.shape[0]):
        current_score = score[batch_index, supervised[batch_index]]
        current_target = target[batch_index, supervised[batch_index]]
        target_delta = current_target[:, None] - current_target[None, :]
        upper_triangle = torch.triu(
            torch.ones_like(target_delta, dtype=torch.bool), diagonal=1
        )
        selected = upper_triangle & (target_delta.abs() >= float(min_gap))
        pair_count = int(selected.sum().item())
        if pair_count == 0:
            skipped_sample_count += 1
            continue
        score_delta = current_score[:, None] - current_score[None, :]
        losses.append(functional.softplus(
            -target_delta[selected].sign() * score_delta[selected] / float(temperature)
        ).mean())
        retained_pair_count += pair_count
    zero = score.sum() * 0.0
    return (
        torch.stack(losses).mean() if losses else zero,
        retained_pair_count,
        skipped_sample_count,
    )


def _normalized_gearnet_changes(
    student_change: torch.Tensor,
    teacher_change: torch.Tensor,
    valid_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    residue_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if student_change.shape != teacher_change.shape or student_change.dim() != 3:
        raise ValueError("student_change and teacher_change must share shape [B, L, D]")
    supervised = _valid_pocket_supervision_mask(
        valid_mask, pocket_mask, residue_mask, student_change.shape[:2], student_change.device
    )
    teacher = teacher_change.detach().to(
        device=student_change.device, dtype=student_change.dtype
    )
    if not bool(torch.isfinite(student_change[supervised]).all()):
        raise ValueError("student_change must be finite at valid pocket residues")
    if not bool(torch.isfinite(teacher[supervised]).all()):
        raise ValueError("teacher_change must be finite at valid pocket residues")
    return student_change, teacher, supervised


def compute_pocket_gearnet_direct_distillation_loss(
    student_change: torch.Tensor,
    teacher_change: torch.Tensor,
    valid_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Directly distil frozen GearNet residue changes into FiLM responses."""
    student_change, teacher, supervised = _normalized_gearnet_changes(
        student_change, teacher_change, valid_mask, pocket_mask, residue_mask
    )
    student = functional.layer_norm(
        student_change[supervised], (student_change.shape[-1],)
    )
    target = functional.layer_norm(teacher[supervised], (teacher.shape[-1],))
    return functional.mse_loss(student, target) + (
        1.0 - functional.cosine_similarity(student, target, dim=-1)
    ).mean()


def compute_pocket_gearnet_pair_distillation_loss(
    student_change: torch.Tensor,
    teacher_change: torch.Tensor,
    valid_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Distil teacher-weighted ligand-specific pocket responses."""
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    student_change, teacher, supervised = _normalized_gearnet_changes(
        student_change, teacher_change, valid_mask, pocket_mask, residue_mask
    )
    losses = []
    for batch_index in range(student_change.shape[0]):
        mask = supervised[batch_index]
        student = functional.layer_norm(
            student_change[batch_index, mask], (student_change.shape[-1],)
        )
        target = functional.layer_norm(
            teacher[batch_index, mask], (teacher.shape[-1],)
        )
        weights = functional.softmax(
            torch.linalg.vector_norm(target, dim=-1).detach() / float(temperature), dim=0
        ).detach()
        student_pair = (weights.unsqueeze(-1) * student).sum(dim=0)
        teacher_pair = (weights.unsqueeze(-1) * target).sum(dim=0)
        mse = functional.mse_loss(student_pair, teacher_pair)
        student_norm = torch.linalg.vector_norm(student_pair)
        teacher_norm = torch.linalg.vector_norm(teacher_pair)
        if bool((student_norm <= 1e-8) & (teacher_norm <= 1e-8)):
            cosine_distance = student_pair.sum() * 0.0
        else:
            cosine_distance = 1.0 - functional.cosine_similarity(
                student_pair.unsqueeze(0), teacher_pair.unsqueeze(0), dim=-1
            ).mean()
        losses.append(mse + cosine_distance)
    return torch.stack(losses).mean()
def compute_pocket_gearnet_semantic_loss(
    student_change: torch.Tensor,
    teacher_change: torch.Tensor,
    valid_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align projected sequence changes to frozen GearNet apo-to-holo changes.

    The loss is the mean cosine distance over valid pocket residues only.
    GearNet targets are detached so gradients flow solely into the sequence model.
    """
    if (
        student_change.shape != teacher_change.shape
        or student_change.dim() != 3
    ):
        raise ValueError(
            "student_change and teacher_change must share shape [B, L, D]"
        )
    shape = student_change.shape[:2]
    if valid_mask.shape != shape or pocket_mask.shape != shape:
        raise ValueError(
            "valid_mask and pocket_mask must have shape [B, L]"
        )
    supervised = (
        valid_mask.to(device=student_change.device).bool()
        & pocket_mask.to(device=student_change.device).bool()
    )
    if residue_mask is not None:
        if residue_mask.shape != shape:
            raise ValueError("residue_mask must have shape [B, L]")
        supervised &= residue_mask.to(device=student_change.device).bool()
    if not bool(supervised.any(dim=1).all()):
        raise ValueError(
            "Every structural sample requires at least one valid pocket GearNet target"
        )
    teacher = teacher_change.detach().to(
        device=student_change.device, dtype=student_change.dtype
    )
    if not bool(torch.isfinite(student_change[supervised]).all()):
        raise ValueError("student_change must be finite at valid pocket residues")
    if not bool(torch.isfinite(teacher[supervised]).all()):
        raise ValueError("teacher_change must be finite at valid pocket residues")
    student_norm = torch.linalg.vector_norm(student_change[supervised], dim=-1)
    teacher_norm = torch.linalg.vector_norm(teacher[supervised], dim=-1)
    if bool((student_norm <= 0).any()) or bool((teacher_norm <= 0).any()):
        raise ValueError("semantic change vectors must be nonzero at valid pocket residues")
    return (1.0 - functional.cosine_similarity(
        student_change[supervised], teacher[supervised], dim=-1
    )).mean()


def compute_pocket_gearnet_relational_contrast_loss(
    student_change: torch.Tensor,
    teacher_change: torch.Tensor,
    valid_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    protein_ids: Sequence[str],
    site_ids: Sequence[str],
    residue_mask: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, int]:
    """Contrast pocket response relations against matched GearNet teachers.

    Each same-protein, same-pocket-site candidate set contributes one multiclass
    InfoNCE term: its matched GearNet relation graph is positive and every
    other ligand graph in that set is a negative. The two encoders may use
    different feature widths.
    """
    if (
        student_change.dim() != 3
        or teacher_change.dim() != 3
        or student_change.shape[:2] != teacher_change.shape[:2]
    ):
        raise ValueError("student_change and teacher_change must be [B, L, D] with matching B and L")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    batch_size, sequence_length = student_change.shape[:2]
    if valid_mask.shape != (batch_size, sequence_length) or pocket_mask.shape != (batch_size, sequence_length):
        raise ValueError("valid_mask and pocket_mask must have shape [B, L]")
    if len(protein_ids) != batch_size or len(site_ids) != batch_size:
        raise ValueError("protein_ids and site_ids must each match batch size")
    supervised = (
        valid_mask.to(device=student_change.device).bool()
        & pocket_mask.to(device=student_change.device).bool()
    )
    if residue_mask is not None:
        if residue_mask.shape != (batch_size, sequence_length):
            raise ValueError("residue_mask must have shape [B, L]")
        supervised &= residue_mask.to(device=student_change.device).bool()
    teacher = teacher_change.detach().to(
        device=student_change.device, dtype=student_change.dtype
    )
    groups = {}
    for index in range(batch_size):
        groups.setdefault((str(protein_ids[index]), str(site_ids[index])), []).append(index)

    losses = []
    group_count = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        common = supervised[indices].all(dim=0)
        residue_count = int(common.sum().item())
        if residue_count < 3:
            continue
        upper = torch.triu_indices(
            residue_count, residue_count, offset=1, device=student_change.device
        )

        def signature(changes):
            normalized = functional.normalize(changes, dim=-1)
            relation = normalized @ normalized.transpose(0, 1)
            return relation[upper[0], upper[1]]

        student_relations = [
            signature(student_change[index, common]) for index in indices
        ]
        teacher_relations = [
            signature(teacher[index, common]) for index in indices
        ]
        if not all(torch.isfinite(value).all() for value in student_relations + teacher_relations):
            raise ValueError("relation-contrast changes must be finite at common valid pocket residues")
        logits = torch.stack([
            torch.stack([
                functional.cosine_similarity(student_relation, teacher_relation, dim=0)
                for teacher_relation in teacher_relations
            ])
            for student_relation in student_relations
        ]) / float(temperature)
        target = torch.arange(len(indices), device=student_change.device, dtype=torch.long)
        losses.append(functional.cross_entropy(logits, target))
        group_count += 1
    if not losses:
        return student_change.sum() * 0.0, 0
    return torch.stack(losses).mean(), group_count


def _within_protein_ranking_loss(
    response_score: torch.Tensor,
    target_rmsd: torch.Tensor,
    protein_ids: Sequence[str],
    margin: float,
) -> tuple[torch.Tensor, int]:
    if len(protein_ids) != int(response_score.shape[0]):
        raise ValueError("protein_ids length must match batch size")
    losses = []
    pair_count = 0
    for left in range(len(protein_ids)):
        for right in range(left + 1, len(protein_ids)):
            if str(protein_ids[left]) != str(protein_ids[right]):
                continue
            difference = target_rmsd[left] - target_rmsd[right]
            if float(difference.detach().cpu()) == 0.0:
                continue
            direction = torch.sign(difference)
            losses.append(functional.relu(float(margin) - direction * (response_score[left] - response_score[right])))
            pair_count += 1
    if not losses:
        return response_score.sum() * 0.0, 0
    return torch.stack(losses).mean(), pair_count


def compute_pocket_site_ranking_loss(
    protein_before: torch.Tensor,
    protein_after: torch.Tensor,
    valid_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    pocket_rmsd: torch.Tensor,
    protein_ids: Sequence[str],
    site_ids: Sequence[str],
    drug_ids: Sequence[str],
    min_gap: float = 0.1,
    residue_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Rank different ligands only within an identical protein-pocket site.

    The score is the mean L2 sequence response over valid pocket residues.
    A pair contributes only when its geometric pocket-RMSD targets differ by at
    least ``min_gap``; the softplus form avoids a non-differentiable hinge.
    """
    if float(min_gap) < 0.0:
        raise ValueError("min_gap must be non-negative")
    if protein_before.shape != protein_after.shape or protein_before.dim() != 3:
        raise ValueError("protein_before and protein_after must share shape [B, L, D]")
    shape = protein_before.shape[:2]
    if valid_mask.shape != shape or pocket_mask.shape != shape:
        raise ValueError("valid_mask and pocket_mask must have shape [B, L]")
    batch_size = int(shape[0])
    if not (len(protein_ids) == len(site_ids) == len(drug_ids) == batch_size):
        raise ValueError("protein_ids, site_ids, and drug_ids must each match batch size")
    supervised = valid_mask.to(device=protein_before.device).bool() & pocket_mask.to(device=protein_before.device).bool()
    if residue_mask is not None:
        if residue_mask.shape != shape:
            raise ValueError("residue_mask must have shape [B, L]")
        supervised &= residue_mask.to(device=protein_before.device).bool()
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("Every structural sample requires at least one valid pocket residue")
    target = pocket_rmsd.to(device=protein_before.device, dtype=protein_before.dtype).reshape(-1)
    if target.shape != (batch_size,) or not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
        raise ValueError("pocket_rmsd must be finite, non-negative, and shape [B]")
    response_score = masked_token_response_score(protein_before, protein_after, supervised)
    groups = {}
    for index in range(batch_size):
        key = (str(protein_ids[index]), str(site_ids[index]), str(drug_ids[index]))
        groups.setdefault(key, []).append(index)
    grouped = []
    for (protein_id, site_id, drug_id), indices in groups.items():
        index = torch.as_tensor(indices, device=response_score.device, dtype=torch.long)
        grouped.append((
            protein_id, site_id, drug_id,
            response_score.index_select(0, index).mean(),
            target.index_select(0, index).median(),
        ))
    losses = []
    for left in range(len(grouped)):
        for right in range(left + 1, len(grouped)):
            protein_left, site_left, _, response_left, target_left = grouped[left]
            protein_right, site_right, _, response_right, target_right = grouped[right]
            if protein_left != protein_right or site_left != site_right:
                continue
            target_gap = target_left - target_right
            if bool(torch.abs(target_gap) < float(min_gap)):
                continue
            direction = torch.sign(target_gap)
            losses.append(functional.softplus(-direction * (response_left - response_right)))
    if not losses:
        return response_score.sum() * 0.0, response_score, 0
    return torch.stack(losses).mean(), response_score, len(losses)


def compute_response_structure_loss(
    protein_before: torch.Tensor,
    protein_after: torch.Tensor,
    residue_mask: torch.Tensor,
    pocket_rmsd: torch.Tensor,
    protein_ids: Sequence[str],
    margin: float = 0.1,
    ranking_weight: float = 1.0,
) -> ResponseStructureLoss:
    """Calibrate response magnitude to its own pair's pocket RMSD target."""
    if float(margin) < 0.0 or float(ranking_weight) < 0.0:
        raise ValueError("margin and ranking_weight must be non-negative")
    response_score = masked_token_response_score(protein_before, protein_after, residue_mask)
    target = pocket_rmsd.to(device=response_score.device, dtype=response_score.dtype).reshape(-1)
    if target.shape != response_score.shape or not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
        raise ValueError("pocket_rmsd must be finite, non-negative, and shape [B]")
    magnitude = functional.smooth_l1_loss(torch.log1p(response_score), torch.log1p(target))
    ranking, pair_count = _within_protein_ranking_loss(response_score, target, protein_ids, margin)
    zero = magnitude * 0.0
    return ResponseStructureLoss(
        total=magnitude + float(ranking_weight) * ranking,
        magnitude=magnitude,
        ranking=ranking,
        response_score=response_score,
        ranking_pair_count=pair_count,
        semantic=zero,
    )
