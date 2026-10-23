
import torch
import torch.nn.functional as F


def _safe_l2_normalize(x, eps=1e-8):
    return F.normalize(x.float(), p=2, dim=-1, eps=eps)


def _mask_diagonal(matrix, fill_value=-float('inf')):
    if matrix.dim() != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError(f"Expected square [N, N] matrix, got {tuple(matrix.shape)}.")
    out = matrix.clone()
    diag = torch.eye(out.size(0), device=out.device, dtype=torch.bool)
    out = out.masked_fill(diag, fill_value)
    return out


def _offdiag_mean(matrix):
    if matrix.size(0) < 2:
        return matrix.new_tensor(0.0)
    mask = ~torch.eye(matrix.size(0), device=matrix.device, dtype=torch.bool)
    values = matrix[mask]
    if values.numel() == 0:
        return matrix.new_tensor(0.0)
    return values.mean()


def compute_relation_alignment_loss(
    drug_response,
    protein_response,
    drug_teacher,
    protein_teacher,
    labels,
    teacher_temperature=0.1,
    student_temperature=0.1,
    positive_only=True,
    min_positive=3,
    alignment_side='both',
):
    """KL-align pairwise teacher neighborhood distribution and response distribution.

    Teacher similarity uses elementwise product:
        joint_sim = drug_sim * protein_sim
    Teacher branch is detached by construction.
    """
    if teacher_temperature <= 0 or student_temperature <= 0:
        raise ValueError("teacher_temperature and student_temperature must be positive.")
    if alignment_side not in {'both', 'drug', 'protein'}:
        raise ValueError("alignment_side must be one of 'both', 'drug', or 'protein'.")
    device = drug_response.device
    dtype = drug_response.dtype
    labels = labels.to(device=device)
    if positive_only:
        keep = labels == 1
    else:
        keep = torch.ones_like(labels, dtype=torch.bool, device=device)
    positive_count = keep.sum()
    zero = drug_response.sum() * 0.0 + protein_response.sum() * 0.0
    if int(positive_count.item()) < int(min_positive):
        return {
            'relation_loss': zero,
            'drug_relation_loss': zero.detach() + zero,
            'protein_relation_loss': zero.detach() + zero,
            'positive_count': positive_count.detach(),
            'valid_relation_batch': False,
            'drug_response_offdiag_cos': zero.detach(),
            'protein_response_offdiag_cos': zero.detach(),
            'response_mean_std': zero.detach(),
            'joint_similarity_unmasked': torch.empty(0, 0, device=device, dtype=dtype),
            'joint_similarity_masked': torch.empty(0, 0, device=device, dtype=dtype),
        }

    rd = _safe_l2_normalize(drug_response[keep])
    rp = _safe_l2_normalize(protein_response[keep])
    td = _safe_l2_normalize(drug_teacher[keep].detach())
    tp = _safe_l2_normalize(protein_teacher[keep].detach())

    drug_teacher_sim = (td @ td.T + 1.0) * 0.5
    protein_teacher_sim = (tp @ tp.T + 1.0) * 0.5
    joint_sim = drug_teacher_sim * protein_teacher_sim
    joint_sim_masked = _mask_diagonal(joint_sim)
    teacher_prob = F.softmax(joint_sim_masked / float(teacher_temperature), dim=-1).detach()

    # Use a finite large negative value for the student distribution: KL with
    # target_prob=0 and input_log_prob=-inf can otherwise produce NaN in PyTorch.
    drug_student_sim = _mask_diagonal(rd @ rd.T, fill_value=-1e9)
    protein_student_sim = _mask_diagonal(rp @ rp.T, fill_value=-1e9)
    drug_log_prob = F.log_softmax(drug_student_sim / float(student_temperature), dim=-1)
    protein_log_prob = F.log_softmax(protein_student_sim / float(student_temperature), dim=-1)

    zero = drug_response.sum() * 0.0 + protein_response.sum() * 0.0
    drug_loss = F.kl_div(drug_log_prob, teacher_prob, reduction='batchmean') if alignment_side in {'both', 'drug'} else zero.detach()
    protein_loss = F.kl_div(protein_log_prob, teacher_prob, reduction='batchmean') if alignment_side in {'both', 'protein'} else zero.detach()
    selected_losses = []
    if alignment_side in {'both', 'drug'}:
        selected_losses.append(drug_loss)
    if alignment_side in {'both', 'protein'}:
        selected_losses.append(protein_loss)
    relation_loss = torch.stack(selected_losses).mean()

    if not torch.isfinite(relation_loss):
        raise FloatingPointError("Non-finite relation alignment loss.")

    response_std = torch.cat([rd, rp], dim=-1).std(dim=0, unbiased=False).mean()
    return {
        'relation_loss': relation_loss,
        'drug_relation_loss': drug_loss,
        'protein_relation_loss': protein_loss,
        'positive_count': positive_count.detach(),
        'valid_relation_batch': True,
        'drug_response_offdiag_cos': _offdiag_mean(rd @ rd.T).detach(),
        'protein_response_offdiag_cos': _offdiag_mean(rp @ rp.T).detach(),
        'response_mean_std': response_std.detach(),
        'joint_similarity_unmasked': joint_sim.detach(),
        'joint_similarity_masked': joint_sim_masked.detach(),
    }
