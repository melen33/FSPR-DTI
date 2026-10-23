"""Local loading utilities for the frozen pretrained MoLFormer drug encoder."""

from pathlib import Path
from typing import Mapping, Tuple

from torch import nn


DEFAULT_DRUG_ENCODER_BACKEND = "native"
DEFAULT_PRETRAINED_MOLFORMER_PATH = "models/drug/molformer"


def resolve_drug_encoder_config(model_config: Mapping) -> Tuple[str, str]:
    """Return the configured backend and local pretrained-model directory."""
    return (
        model_config.get("drug_encoder_backend", DEFAULT_DRUG_ENCODER_BACKEND),
        model_config.get("pretrained_molformer_path", DEFAULT_PRETRAINED_MOLFORMER_PATH),
    )


def load_frozen_pretrained_molformer(path: str) -> Tuple[nn.Module, int]:
    """Load local MoLFormer weights and freeze every parameter."""
    model_path = Path(path)
    required_assets = (model_path / "config.json", model_path / "pytorch_model.bin")
    missing_assets = [str(asset) for asset in required_assets if not asset.is_file()]
    if missing_assets:
        raise FileNotFoundError(
            "Missing required local MoLFormer assets: " + ", ".join(missing_assets)
        )

    from models.drug.molformer.modeling_molformer import MolformerModel

    model = MolformerModel.from_pretrained(str(model_path))
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, model.config.hidden_size
