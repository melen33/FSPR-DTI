"""Masked conformation losses and Angstrom-space evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr


@dataclass(frozen=True)
class ConformationLosses:
    total: torch.Tensor
    all_residue: torch.Tensor
    pocket: torch.Tensor
    global_loss: torch.Tensor


def _masked_smooth_l1(prediction, target, mask):
    mask = mask.to(device=prediction.device).bool()
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    mask = mask & torch.isfinite(target) & torch.isfinite(prediction)
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[mask], target[mask], reduction="sum") / mask.sum().clamp_min(1)


def compute_conformation_loss(output, targets, pocket_weight=1.0, global_weight=0.2):
    valid_mask = targets["valid_mask"].bool()
    pocket_mask = targets["pocket_mask"].bool() & valid_mask
    all_residue = _masked_smooth_l1(
        output.log_displacement, targets["log_displacement"], valid_mask
    )
    pocket = _masked_smooth_l1(
        output.log_displacement, targets["log_displacement"], pocket_mask
    )
    global_target = targets["log_global_targets"]
    global_mask = torch.isfinite(global_target)
    global_loss = _masked_smooth_l1(
        output.log_global_targets, global_target, global_mask
    )
    total = all_residue + float(pocket_weight) * pocket + float(global_weight) * global_loss
    return ConformationLosses(total, all_residue, pocket, global_loss)


def _safe_correlations(prediction, target):
    count = int(len(prediction))
    if count < 2 or np.ptp(prediction) == 0 or np.ptp(target) == 0:
        return float("nan"), 0, float("nan"), 0
    pearson = float(pearsonr(prediction, target).statistic)
    spearman = float(spearmanr(prediction, target).statistic)
    return pearson, count if np.isfinite(pearson) else 0, spearman, count if np.isfinite(spearman) else 0


def evaluate_conformation_predictions(rows):
    """Evaluate scalar prediction records grouped by their ``scope`` field.

    Each record contains ``scope``, ``log_prediction``, and ``log_target``.
    Conversion back to Angstrom space happens before every reported metric.
    """
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["scope"]), [[], []])
        grouped[str(row["scope"])][0].append(float(row["log_prediction"]))
        grouped[str(row["scope"])][1].append(float(row["log_target"]))
    metrics = {}
    for scope, (raw_prediction, raw_target) in sorted(grouped.items()):
        log_prediction = np.asarray(raw_prediction, dtype=np.float64)
        log_target = np.asarray(raw_target, dtype=np.float64)
        finite = np.isfinite(log_prediction) & np.isfinite(log_target)
        prediction = np.maximum(np.expm1(log_prediction[finite]), 0.0)
        target = np.maximum(np.expm1(log_target[finite]), 0.0)
        if len(prediction) == 0:
            metrics[scope] = {
                "count": 0,
                "mae_angstrom": float("nan"),
                "rmse_angstrom": float("nan"),
                "pearson": float("nan"),
                "pearson_defined_count": 0,
                "spearman": float("nan"),
                "spearman_defined_count": 0,
            }
            continue
        errors = prediction - target
        pearson, pearson_count, spearman, spearman_count = _safe_correlations(prediction, target)
        metrics[scope] = {
            "count": int(len(prediction)),
            "mae_angstrom": float(np.mean(np.abs(errors))),
            "rmse_angstrom": float(np.sqrt(np.mean(np.square(errors)))),
            "pearson": pearson,
            "pearson_defined_count": pearson_count,
            "spearman": spearman,
            "spearman_defined_count": spearman_count,
        }
    return metrics
