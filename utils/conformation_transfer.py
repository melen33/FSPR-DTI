"""Strict, explicit transfer from conformation pretraining into TAPB."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


TRANSFER_PREFIXES = (
    "drug_encoder.",
    "pr_linear.",
    "aggregator_protein.",
    "flex_film.",
)


@dataclass(frozen=True)
class TransferReport:
    loaded_prefixes: tuple[str, ...]
    tensor_count: int
    parameter_count: int
    parameter_names: frozenset[str]


def load_conformation_pretrained_modules(model, checkpoint_path):
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if int(checkpoint.get("format_version", -1)) != 1:
        raise ValueError("Unsupported conformation checkpoint format")
    source = checkpoint.get("model_state_dict")
    if not isinstance(source, dict):
        raise ValueError("Conformation checkpoint has no model_state_dict")
    target = model.state_dict()
    expected = {key for key in target if key.startswith(TRANSFER_PREFIXES)}
    if not expected:
        raise ValueError("Target model has no transferable conformation parameters")
    missing = sorted(expected.difference(source))
    if missing:
        raise ValueError(f"Conformation checkpoint missing expected keys: {missing[:10]}")
    mismatches = []
    for key in sorted(expected):
        if tuple(source[key].shape) != tuple(target[key].shape):
            mismatches.append(f"{key}: checkpoint {tuple(source[key].shape)} vs model {tuple(target[key].shape)}")
    if mismatches:
        raise ValueError("Conformation checkpoint shape mismatch: " + "; ".join(mismatches[:10]))
    merged = dict(target)
    for key in expected:
        merged[key] = source[key].to(dtype=target[key].dtype)
    model.load_state_dict(merged, strict=True)
    named_parameters = dict(model.named_parameters())
    parameter_names = frozenset(key for key in expected if key in named_parameters)
    report = TransferReport(
        loaded_prefixes=tuple(prefix[:-1] for prefix in TRANSFER_PREFIXES),
        tensor_count=len(expected),
        parameter_count=sum(named_parameters[name].numel() for name in parameter_names),
        parameter_names=parameter_names,
    )
    print(
        "Loaded conformation-pretrained modules: "
        f"prefixes={report.loaded_prefixes}, tensors={report.tensor_count}, "
        f"parameters={report.parameter_count}"
    )
    return report


def build_dti_optimizer(
    model,
    base_lr,
    weight_decay,
    pretrained_lr_scale,
    pretrained_parameter_names,
):
    base_lr = float(base_lr)
    scale = float(pretrained_lr_scale)
    if base_lr <= 0 or scale <= 0:
        raise ValueError("Learning rates and pretrained_lr_scale must be positive")
    pretrained_names = set(pretrained_parameter_names)
    transferred = []
    remaining = []
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"Trainable parameter appears more than once: {name}")
        seen.add(id(parameter))
        (transferred if name in pretrained_names else remaining).append(parameter)
    unknown = pretrained_names.difference(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if unknown:
        raise ValueError(f"Unknown or frozen pretrained parameter names: {sorted(unknown)[:10]}")
    groups = []
    if transferred:
        groups.append({"params": transferred, "lr": base_lr * scale, "group_name": "conformation_pretrained"})
    if remaining:
        groups.append({"params": remaining, "lr": base_lr, "group_name": "base"})
    if not groups:
        raise ValueError("Model has no trainable parameters")
    return torch.optim.AdamW(groups, lr=base_lr, weight_decay=float(weight_decay))
