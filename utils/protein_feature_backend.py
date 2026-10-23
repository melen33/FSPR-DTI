"""Offline protein-language-model feature backend configuration and validation."""

from dataclasses import dataclass
from pathlib import Path
import os
import pickle
from typing import Mapping, Optional

import torch


SUPPORTED_PROTEIN_ENCODERS = {"esm2", "esmc_600m"}
MAX_REAL_PROTEIN_RESIDUES = 510


@dataclass(frozen=True)
class ProteinFeatureSpec:
    backend: str
    cache_path: Path
    hidden_dim: int
    model_name: Optional[str]


def _get_value(container, key, default=None):
    if isinstance(container, Mapping):
        return container.get(key, default)
    return getattr(container, key, default)


def _resolve_path(raw_path, data_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or (path.parts and path.parts[0] == "datasets"):
        return path
    return data_root / path


def resolve_protein_feature_spec(
    train_config,
    model_config,
    data_root: Path,
    backend_override: Optional[str] = None,
    feature_path_override: Optional[str] = None,
) -> ProteinFeatureSpec:
    """Resolve one explicit cache and its required residue-feature width."""
    backend = backend_override or _get_value(
        model_config, "protein_encoder_backend", "esm2"
    )
    if backend not in SUPPORTED_PROTEIN_ENCODERS:
        raise ValueError(f"Unsupported protein encoder backend: {backend}")

    train_section = _get_value(train_config, "TRAIN")
    if train_section is None:
        raise ValueError("Training configuration is missing TRAIN settings.")

    if backend == "esm2":
        cache_name = feature_path_override or _get_value(train_section, "PR_PATH")
        if not cache_name:
            raise ValueError("ESM2 protein feature path is not configured.")
        return ProteinFeatureSpec(
            backend="esm2",
            cache_path=_resolve_path(cache_name, Path(data_root)),
            hidden_dim=1280,
            model_name=None,
        )

    cache_name = feature_path_override or _get_value(
        train_section, "ESMC_600M_PR_PATH"
    )
    if not cache_name:
        raise ValueError("ESMC-600M protein feature path is not configured.")
    hidden_dim = int(_get_value(model_config, "esmc_600m_hidden_dim", 1152))
    model_name = _get_value(model_config, "esmc_600m_model_name", "esmc_600m")
    return ProteinFeatureSpec(
        backend="esmc_600m",
        cache_path=_resolve_path(cache_name, Path(data_root)),
        hidden_dim=hidden_dim,
        model_name=str(model_name),
    )


def load_validated_protein_feature_cache(spec: ProteinFeatureSpec):
    """Load a token-feature cache and reject incompatible tensors early."""
    if not spec.cache_path.is_file():
        raise FileNotFoundError(
            f"Protein feature cache is missing for {spec.backend}: {spec.cache_path}"
        )
    with spec.cache_path.open("rb") as handle:
        features = pickle.load(handle)
    if isinstance(features, dict):
        if not features:
            raise ValueError("Protein feature cache dictionary must be non-empty.")
        feature_items = features.items()
    elif spec.backend == "esm2" and isinstance(features, list):
        if not features:
            raise ValueError("Legacy ESM2 protein feature cache list must be non-empty.")
        feature_items = enumerate(features)
    else:
        raise ValueError(
            "Protein feature cache must be a non-empty dictionary; "
            "legacy ESM2 caches may also be non-empty lists."
        )
    for protein_id, feature in feature_items:
        if not isinstance(feature, torch.Tensor) or feature.dim() != 2:
            raise ValueError(
                f"Protein feature for {protein_id!r} must be a 2-D torch.Tensor."
            )
        if feature.size(1) != spec.hidden_dim:
            raise ValueError(
                f"Protein feature for {protein_id!r} has width {feature.size(1)}; "
                f"expected width {spec.hidden_dim} for {spec.backend}."
            )
    return features


def load_esmc_600m_runtime(model_name: str, device: torch.device):
    """Load a locally cached ESMC runtime and its sequence API."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError as exc:
        raise RuntimeError(
            "ESMC-600M requires a modern runtime exposing esm.models.esmc.ESMC. "
            "Install the EvolutionaryScale ESMC package before generating features."
        ) from exc
    try:
        model = ESMC.from_pretrained(model_name, device=device).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load locally cached ESMC-600M model {model_name!r}. "
            "Ensure its assets are present in the local Hugging Face cache."
        ) from exc
    return model, ESMProtein, LogitsConfig


def generate_esmc_600m_feature_cache(sequences, spec: ProteinFeatureSpec, device):
    """Generate an atomic CPU cache of BOS/residue/EOS ESMC embeddings."""
    if spec.backend != "esmc_600m":
        raise ValueError("ESMC cache generation requires backend='esmc_600m'.")
    if not sequences:
        raise ValueError("Cannot generate an ESMC cache from an empty protein mapping.")
    if not spec.model_name:
        raise ValueError("ESMC-600M model name is not configured.")

    model, protein_factory, logits_config_factory = load_esmc_600m_runtime(
        spec.model_name, device
    )
    model.eval()
    features = {}
    with torch.no_grad():
        for protein_id, sequence in sequences.items():
            sequence = str(sequence)[:MAX_REAL_PROTEIN_RESIDUES]
            protein = protein_factory(sequence=sequence)
            tokenized = model.encode(protein)
            output = model.logits(
                tokenized, logits_config_factory(return_embeddings=True)
            )
            embedding = output.embeddings
            if not isinstance(embedding, torch.Tensor):
                raise ValueError(f"ESMC embedding for {protein_id!r} is not a tensor.")
            if embedding.dim() == 3:
                if embedding.size(0) != 1:
                    raise ValueError(
                        f"ESMC embedding for {protein_id!r} must have batch size 1."
                    )
                embedding = embedding.squeeze(0)
            if embedding.dim() != 2:
                raise ValueError(
                    f"ESMC embedding for {protein_id!r} must have shape [L, D]."
                )
            expected_tokens = len(sequence) + 2
            if embedding.size(0) != expected_tokens:
                raise ValueError(
                    f"ESMC embedding for {protein_id!r} has {embedding.size(0)} tokens; "
                    f"expected BOS/residues/EOS length {expected_tokens}."
                )
            if embedding.size(1) != spec.hidden_dim:
                raise ValueError(
                    f"ESMC embedding for {protein_id!r} has width {embedding.size(1)}; "
                    f"expected width {spec.hidden_dim}."
                )
            features[str(protein_id)] = embedding.detach().cpu()

    spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = spec.cache_path.with_suffix(spec.cache_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(features, handle)
    temporary_path.replace(spec.cache_path)
    return spec.cache_path


def validate_structure_supervision_feature_cache(backend: str, esm_cache_path: Path):
    """Prevent an ESMC run from silently consuming legacy ESM2 Apo-holo features."""
    legacy_cache = Path("datasets/apo_holo/features/esm2_features.pkl").resolve()
    normalized = Path(esm_cache_path).resolve()
    if backend == "esmc_600m" and normalized == legacy_cache:
        raise ValueError(
            "ESMC-600M structure supervision requires an explicit ESMC Apo-holo "
            "cache; the legacy ESM2 cache is not compatible."
        )
