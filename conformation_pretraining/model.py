"""TAPB-compatible protein-response conformation pretrainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.tapb import FlexibilityConditionedFiLM, precompute_freqs_cis
from models.transformer import TransformerDecoder, TransformerEncoder


@dataclass(frozen=True)
class ConformationOutput:
    log_displacement: torch.Tensor
    log_global_targets: torch.Tensor


class ConformationResponsePretrainer(nn.Module):
    """Predict apo-to-holo movement with TAPB's protein-query interaction path."""

    def __init__(self, model_config):
        super().__init__()
        drug_config = model_config["DrugEncoder"]
        decoder_config = model_config["TransformerDeocder"]
        d_model = int(drug_config["d_model"])
        if int(decoder_config["d_model"]) != d_model:
            raise ValueError("Drug encoder and decoder d_model must match")
        self.d_model = d_model
        self.encoder_n_heads = int(drug_config["n_head"])
        self.fusion_n_heads = int(decoder_config["n_head"])

        self.drug_encoder = TransformerEncoder(config=drug_config)
        self.pr_linear = nn.Linear(1280, d_model)
        self.aggregator_protein = TransformerDecoder(config=decoder_config)
        self.flex_film = FlexibilityConditionedFiLM(
            d_model=d_model,
            film_hidden_dim=int(model_config.get("film_hidden_dim", 128)),
            film_dropout=float(model_config.get("film_dropout", 0.1)),
            gamma_scale=float(model_config.get("film_gamma_scale", 0.5)),
            beta_scale=float(model_config.get("film_beta_scale", 0.1)),
            film_use_protein_flex=True,
            film_use_drug_flex=True,
            film_mode="shift_only",
            film_flex_encoding=model_config.get("film_flex_encoding", "raw"),
            film_flex_embed_dim=int(model_config.get("film_flex_embed_dim", 16)),
        )
        head_hidden = int(model_config.get("conformation_head_hidden_dim", d_model))
        self.residue_head = nn.Sequential(
            nn.Linear(3 * d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        self.global_head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 2),
        )
        freqs = precompute_freqs_cis(d_model // self.encoder_n_heads, 4000)
        self.register_buffer("precompute_freqs_cis", freqs, persistent=False)

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        input_drugs,
        input_proteins,
        protein_mask,
        protein_flex,
        drug_flex,
    ):
        drug_ids = input_drugs["input_ids"]
        drug_valid_mask = input_drugs["attention_mask"].to(device=drug_ids.device).bool()
        drug_padding_mask = ~drug_valid_mask
        batch_size, drug_length = drug_ids.shape
        encoder_mask = drug_padding_mask.unsqueeze(1).unsqueeze(1).expand(
            -1, self.encoder_n_heads, drug_length, -1
        )
        drug_tokens = self.drug_encoder(
            drug_ids,
            encoder_mask,
            self.precompute_freqs_cis[:drug_length].to(drug_ids.device),
        )

        protein_pre = self.pr_linear(input_proteins)
        protein_mask = protein_mask.to(device=protein_pre.device).bool()
        protein_length = protein_pre.size(1)
        protein_padding_mask = ~protein_mask
        self_attention_mask = protein_padding_mask.unsqueeze(1).unsqueeze(1).expand(
            -1, self.fusion_n_heads, protein_length, -1
        )
        cross_attention_mask = drug_padding_mask.unsqueeze(1).unsqueeze(1).expand(
            -1, self.fusion_n_heads, protein_length, -1
        )
        protein_cross, _ = self.aggregator_protein(
            src=drug_tokens,
            tgt=protein_pre,
            self_attn_mask=self_attention_mask,
            cross_attn_mask=cross_attention_mask,
        )
        protein_film, _ = self.flex_film(
            protein_pre=protein_pre,
            protein_ca=protein_cross,
            protein_flex=protein_flex,
            protein_mask=protein_mask,
            drug_flex=drug_flex,
        )
        response = protein_film - protein_pre
        residue_features = torch.cat([protein_pre, protein_film, response], dim=-1)
        log_displacement = self.residue_head(residue_features).squeeze(-1)
        pooled_response = self._masked_mean(response, protein_mask)
        log_global_targets = self.global_head(pooled_response)
        return ConformationOutput(log_displacement, log_global_targets)
