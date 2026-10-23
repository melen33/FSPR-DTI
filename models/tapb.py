import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import TransformerEncoder, TransformerDecoder
from models.pretrained_molformer import (
    load_frozen_pretrained_molformer,
    resolve_drug_encoder_config,
)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

class MLMHead(nn.Module):
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_embd)
        self.activation = nn.GELU()
        self.ln = nn.LayerNorm(n_embd)
        self.linear2 = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        return self.linear2(self.ln(self.activation(self.linear1(x))))


class PairSpecificFlexibilityCalibration(nn.Module):
    """Protein-side pair-specific flexibility calibration.

    This module calibrates the existing protein-side cross-attention/decoder
    output instead of adding a new predictor branch.

    h_p_pre:      [B, L_p, D], protein representation before protein-side decoder
    h_p_ca:       [B, L_p, D], current protein-side decoder output
    protein_flex: [B, L_p, 1], residue-level F_p = 1 - S^2
    protein_mask: [B, L_p], valid residue/token mask
    drug_flex:    [B, 1], optional molecule-level F_d condition
    """

    def __init__(
        self,
        d_model,
        hidden_dim=128,
        dropout=0.1,
        lambda_flex=0.5,
        use_drug_flex_condition=True,
        eps=1e-8,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.lambda_flex = float(lambda_flex)
        self.use_drug_flex_condition = bool(use_drug_flex_condition)
        self.eps = float(eps)

        # C_p = concat(H_p_pre, H_p_ca, Delta_p_ca, DeltaNorm_p, F_p, F_d)
        # last dim = 3D + 3.
        context_dim = 3 * self.d_model + 3
        self.adapt_mlp = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.scale_mlp = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 1),
        )
        # Zero-init makes Gamma_p=tanh(0)=0, Scale_p=1, and therefore
        # H_p_calibrated == H_p_ca at initialization.
        nn.init.zeros_(self.scale_mlp[-1].weight)
        nn.init.zeros_(self.scale_mlp[-1].bias)

    def forward(
        self,
        h_p_pre,
        h_p_ca,
        protein_flex,
        protein_mask,
        drug_flex=None,
        return_stats=False,
    ):
        if h_p_pre.dim() != 3 or h_p_ca.dim() != 3:
            raise ValueError("h_p_pre and h_p_ca must have shape [B, L_p, D].")
        if h_p_pre.shape != h_p_ca.shape:
            raise ValueError(
                "h_p_pre and h_p_ca shape mismatch: "
                f"{tuple(h_p_pre.shape)} vs {tuple(h_p_ca.shape)}."
            )
        if h_p_ca.size(-1) != self.d_model:
            raise ValueError(
                f"Expected hidden dim {self.d_model}, got {h_p_ca.size(-1)}."
            )
        if protein_flex is None:
            raise ValueError(
                "protein_flex is required for PairSpecificFlexibilityCalibration."
            )
        if protein_flex.dim() != 3 or protein_flex.size(-1) != 1:
            raise ValueError(
                "protein_flex must have shape [B, L_p, 1], got "
                f"{tuple(protein_flex.shape)}."
            )
        if protein_flex.shape[:2] != h_p_ca.shape[:2]:
            raise ValueError(
                "protein_flex and protein representation shape mismatch: "
                f"{tuple(protein_flex.shape)} vs {tuple(h_p_ca.shape)}."
            )
        if protein_mask.dim() != 2 or protein_mask.shape != h_p_ca.shape[:2]:
            raise ValueError(
                "protein_mask must have shape [B, L_p], got "
                f"{tuple(protein_mask.shape)} for h_p_ca {tuple(h_p_ca.shape)}."
            )

        batch_size, protein_len, _ = h_p_ca.shape
        dtype = h_p_ca.dtype
        device = h_p_ca.device
        mask = protein_mask.to(device=device).bool().unsqueeze(-1)
        mask_float = mask.to(dtype=dtype)

        f_p = protein_flex.to(device=device, dtype=dtype) * mask_float  # [B, L_p, 1]

        if self.use_drug_flex_condition:
            if drug_flex is None:
                raise ValueError(
                    "drug_flex is required when use_drug_flex_condition=True."
                )
            if drug_flex.dim() == 1:
                drug_flex = drug_flex.unsqueeze(-1)
            if drug_flex.dim() != 2 or drug_flex.size(0) != batch_size:
                raise ValueError(
                    "drug_flex must have shape [B, drug_flex_dim], got "
                    f"{tuple(drug_flex.shape)}."
                )
            # Minimal stage uses one molecule-level scalar F_d.
            f_d = drug_flex[:, :1].to(device=device, dtype=dtype)
        else:
            f_d = h_p_ca.new_zeros(batch_size, 1)
        f_d_broadcast = f_d.unsqueeze(1).expand(-1, protein_len, -1) * mask_float

        # Delta_p_ca is the protein-side decoder candidate shift. In this codebase
        # h_p_ca already includes TransformerDecoder residual/LN/FFN, so this is
        # not a pure raw attention message; it is the net token-aligned update.
        delta_p_ca = h_p_ca - h_p_pre  # [B, L_p, D]
        delta_norm = torch.log1p(
            torch.norm(delta_p_ca, p=2, dim=-1, keepdim=True).clamp_min(0.0)
        )  # [B, L_p, 1]

        context = torch.cat(
            [h_p_pre, h_p_ca, delta_p_ca, delta_norm, f_p, f_d_broadcast],
            dim=-1,
        )  # [B, L_p, 3D + 3]
        adaptation_map = self.adapt_mlp(context) * mask_float  # A_p: [B, L_p, 1]
        gamma_p = torch.tanh(self.scale_mlp(context)) * mask_float  # Gamma_p
        calibration_scale = 1.0 + self.lambda_flex * adaptation_map * gamma_p
        calibration_scale = torch.where(
            mask,
            calibration_scale,
            torch.ones_like(calibration_scale),
        )  # padding scale = 1

        h_p_calibrated = h_p_ca + (calibration_scale - 1.0) * delta_p_ca

        if not return_stats:
            return h_p_calibrated, {}

        candidate_shift_norm = torch.norm(
            delta_p_ca, p=2, dim=-1, keepdim=True
        ) * mask_float
        calibrated_shift_norm = torch.norm(
            calibration_scale * delta_p_ca, p=2, dim=-1, keepdim=True
        ) * mask_float
        denom = mask_float.sum(dim=1).clamp_min(self.eps)
        pair_adaptation_score = (f_p * adaptation_map).sum(dim=1) / denom
        return h_p_calibrated, {
            'adaptation_map': adaptation_map,
            'calibration_scale': calibration_scale,
            'candidate_shift_norm': candidate_shift_norm,
            'calibrated_shift_norm': calibrated_shift_norm,
            'pair_adaptation_score': pair_adaptation_score,
        }


class FlexibilityConditionedFiLM(nn.Module):
    """Protein-side FiLM calibration for an existing candidate interaction shift.

    protein_pre:  [B, L_p, D], protein representation before protein-side decoder
    protein_ca:   [B, L_p, D], existing protein-side decoder output
    protein_flex: [B, L_p, 1], residue-level F_p = 1 - S^2
    protein_mask: [B, L_p], 1 for valid protein tokens, 0 for padding/special tokens
    drug_flex:    [B, 1], molecule-level F_d = log(1 + nTABS), optional by config
    """

    def __init__(
        self,
        d_model,
        film_hidden_dim=128,
        film_dropout=0.1,
        gamma_scale=0.5,
        beta_scale=0.1,
        film_use_protein_flex=True,
        film_use_drug_flex=True,
        film_mode='scale_shift',
        film_flex_encoding='raw',
        film_flex_embed_dim=16,
        eps=1e-8,
    ):
        super().__init__()
        if film_mode not in {'scale_shift', 'scale_only', 'shift_only'}:
            raise ValueError("film_mode must be one of 'scale_shift', 'scale_only', or 'shift_only'.")
        if film_flex_encoding not in {'raw', 'mlp'}:
            raise ValueError("film_flex_encoding must be either 'raw' or 'mlp'.")
        self.d_model = int(d_model)
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.film_use_protein_flex = bool(film_use_protein_flex)
        self.film_use_drug_flex = bool(film_use_drug_flex)
        self.film_mode = film_mode
        self.film_flex_encoding = film_flex_encoding
        self.film_flex_embed_dim = int(film_flex_embed_dim)
        self.eps = float(eps)

        if self.film_flex_encoding == 'mlp':
            self.protein_flex_encoder = nn.Sequential(
                nn.Linear(1, self.film_flex_embed_dim),
                nn.GELU(),
                nn.Linear(self.film_flex_embed_dim, self.film_flex_embed_dim),
            )
            self.drug_flex_encoder = nn.Sequential(
                nn.Linear(1, self.film_flex_embed_dim),
                nn.GELU(),
                nn.Linear(self.film_flex_embed_dim, self.film_flex_embed_dim),
            )
            # protein_pre/protein_ca/candidate_shift + shift_norm + E_p + E_d
            condition_dim = 3 * self.d_model + 1 + 2 * self.film_flex_embed_dim
        else:
            self.protein_flex_encoder = None
            self.drug_flex_encoder = None
            # protein_pre/protein_ca/candidate_shift + shift_norm + scalar F_p + scalar F_d
            condition_dim = 3 * self.d_model + 3

        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, int(film_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(film_dropout)),
        )
        self.film_output = nn.Linear(int(film_hidden_dim), 2 * self.d_model)
        nn.init.zeros_(self.film_output.weight)
        nn.init.zeros_(self.film_output.bias)

    def forward(
        self,
        protein_pre,
        protein_ca,
        protein_flex,
        protein_mask,
        drug_flex=None,
        return_stats=False,
    ):
        if protein_pre.dim() != 3 or protein_ca.dim() != 3:
            raise ValueError("protein_pre and protein_ca must have shape [B, L_p, D].")
        if protein_pre.shape != protein_ca.shape:
            raise ValueError(
                "protein_pre and protein_ca shape mismatch: "
                f"{tuple(protein_pre.shape)} vs {tuple(protein_ca.shape)}."
            )
        if protein_ca.size(-1) != self.d_model:
            raise ValueError(f"Expected hidden dim {self.d_model}, got {protein_ca.size(-1)}.")
        if self.film_use_protein_flex:
            if protein_flex is None:
                raise ValueError("protein_flex is required when film_use_protein_flex=True.")
            if protein_flex.dim() != 3 or protein_flex.size(-1) != 1:
                raise ValueError(
                    "protein_flex must have shape [B, L_p, 1], got "
                    f"{tuple(protein_flex.shape)}."
                )
            if protein_flex.shape[:2] != protein_ca.shape[:2]:
                raise ValueError(
                    "protein_flex and protein representation shape mismatch: "
                    f"{tuple(protein_flex.shape)} vs {tuple(protein_ca.shape)}."
                )
        elif protein_flex is not None:
            if protein_flex.dim() != 3 or protein_flex.size(-1) != 1:
                raise ValueError(
                    "protein_flex must have shape [B, L_p, 1] when provided, got "
                    f"{tuple(protein_flex.shape)}."
                )
            if protein_flex.shape[:2] != protein_ca.shape[:2]:
                raise ValueError(
                    "protein_flex and protein representation shape mismatch: "
                    f"{tuple(protein_flex.shape)} vs {tuple(protein_ca.shape)}."
                )
        if protein_mask.dim() != 2 or protein_mask.shape != protein_ca.shape[:2]:
            raise ValueError(
                "protein_mask must have shape [B, L_p], got "
                f"{tuple(protein_mask.shape)} for protein_ca {tuple(protein_ca.shape)}."
            )

        batch_size, protein_len, _ = protein_ca.shape
        device = protein_ca.device
        dtype = protein_ca.dtype
        mask = protein_mask.to(device=device).bool().unsqueeze(-1)
        mask_float = mask.to(dtype=dtype)

        if self.film_use_protein_flex:
            f_p = protein_flex.to(device=device, dtype=dtype) * mask_float
        else:
            f_p = protein_ca.new_zeros(batch_size, protein_len, 1)

        if self.film_use_drug_flex:
            if drug_flex is None:
                raise ValueError("drug_flex is required when film_use_drug_flex=True.")
            if drug_flex.dim() == 1:
                drug_flex = drug_flex.unsqueeze(-1)
            if drug_flex.dim() != 2 or drug_flex.size(0) != batch_size:
                raise ValueError(
                    "drug_flex must have shape [B, drug_flex_dim], got "
                    f"{tuple(drug_flex.shape)}."
                )
            f_d = drug_flex[:, :1].to(device=device, dtype=dtype)
        else:
            f_d = protein_ca.new_zeros(batch_size, 1)
        f_d_broadcast = f_d.unsqueeze(1).expand(-1, protein_len, -1) * mask_float

        # Candidate shift is the existing protein-side decoder update, not an
        # induced-fit label. The decoder output already includes residual/LN/FFN.
        candidate_shift = protein_ca - protein_pre
        candidate_shift_norm = torch.norm(candidate_shift, p=2, dim=-1, keepdim=True) * mask_float
        shift_norm_feature = torch.log1p(candidate_shift_norm.clamp_min(0.0))

        if self.film_flex_encoding == 'mlp':
            protein_flex_feature = self.protein_flex_encoder(f_p) * mask_float
            drug_flex_feature = self.drug_flex_encoder(f_d_broadcast) * mask_float
        else:
            protein_flex_feature = f_p
            drug_flex_feature = f_d_broadcast

        film_condition = torch.cat(
            [
                protein_pre,
                protein_ca,
                candidate_shift,
                shift_norm_feature,
                protein_flex_feature,
                drug_flex_feature,
            ],
            dim=-1,
        )
        film_params = self.film_output(self.condition_encoder(film_condition))
        gamma_raw, beta_raw = film_params.chunk(2, dim=-1)
        gamma = self.gamma_scale * torch.tanh(gamma_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)
        if self.film_mode == 'scale_only':
            beta = torch.zeros_like(beta)
        elif self.film_mode == 'shift_only':
            gamma = torch.zeros_like(gamma)
        gamma = gamma * mask_float
        beta = beta * mask_float

        film_correction = (gamma * candidate_shift + beta) * mask_float
        protein_film = protein_ca + film_correction

        if not return_stats:
            return protein_film, {}

        adaptation_map = torch.norm(film_correction, p=2, dim=-1, keepdim=True) * mask_float
        protein_ca_norm = torch.norm(protein_ca, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        relative_adaptation_map = (adaptation_map / protein_ca_norm) * mask_float
        denom = mask_float.sum(dim=1).clamp_min(self.eps)
        pair_adaptation_score = (relative_adaptation_map * mask_float).sum(dim=1) / denom
        weighted_denom = (f_p * mask_float).sum(dim=1).clamp_min(self.eps)
        weighted_pair_adaptation_score = (
            relative_adaptation_map * f_p * mask_float
        ).sum(dim=1) / weighted_denom

        return protein_film, {
            'gamma': gamma,
            'beta': beta,
            'candidate_shift': candidate_shift,
            'candidate_shift_norm': candidate_shift_norm,
            'film_correction': film_correction,
            'adaptation_map': adaptation_map,
            'relative_adaptation_map': relative_adaptation_map,
            'pair_adaptation_score': pair_adaptation_score,
            'weighted_pair_adaptation_score': weighted_pair_adaptation_score,
        }


class DrugFlexibilityConditionedFiLM(nn.Module):
    """Drug-side FiLM calibration for an existing drug candidate interaction shift.

    drug_pre:             [B, L_d, D], drug representation before drug-side decoder
    drug_ca:              [B, L_d, D], existing drug-side decoder output
    drug_flex:            [B, 1], molecule-level F_d = log(1 + nTABS)
    protein_flex_global:  [B, 3], pooled protein F_p = [mean, max, std]
    drug_mask:            [B, L_d], 1 for valid drug tokens, 0 for padding
    """

    def __init__(
        self,
        d_model,
        film_hidden_dim=128,
        film_dropout=0.1,
        gamma_scale=0.5,
        beta_scale=0.1,
        film_use_drug_flex=True,
        film_use_protein_flex=True,
        film_mode='shift_only',
        film_flex_encoding='raw',
        film_flex_embed_dim=16,
        eps=1e-8,
    ):
        super().__init__()
        if film_mode not in {'scale_shift', 'scale_only', 'shift_only'}:
            raise ValueError("film_mode must be one of 'scale_shift', 'scale_only', or 'shift_only'.")
        if film_flex_encoding not in {'raw', 'mlp'}:
            raise ValueError("film_flex_encoding must be either 'raw' or 'mlp'.")
        self.d_model = int(d_model)
        self.gamma_scale = float(gamma_scale)
        self.beta_scale = float(beta_scale)
        self.film_use_drug_flex = bool(film_use_drug_flex)
        self.film_use_protein_flex = bool(film_use_protein_flex)
        self.film_mode = film_mode
        self.film_flex_encoding = film_flex_encoding
        self.film_flex_embed_dim = int(film_flex_embed_dim)
        self.eps = float(eps)

        if self.film_flex_encoding == 'mlp':
            self.drug_flex_encoder = nn.Sequential(
                nn.Linear(1, self.film_flex_embed_dim),
                nn.GELU(),
                nn.Linear(self.film_flex_embed_dim, self.film_flex_embed_dim),
            )
            self.protein_flex_global_encoder = nn.Sequential(
                nn.Linear(3, self.film_flex_embed_dim),
                nn.GELU(),
                nn.Linear(self.film_flex_embed_dim, self.film_flex_embed_dim),
            )
            condition_dim = 3 * self.d_model + 1 + 2 * self.film_flex_embed_dim
        else:
            self.drug_flex_encoder = None
            self.protein_flex_global_encoder = None
            # drug_pre/drug_ca/candidate_shift + shift_norm + scalar F_d + pooled F_p_global[3]
            condition_dim = 3 * self.d_model + 5

        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, int(film_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(film_dropout)),
        )
        self.film_output = nn.Linear(int(film_hidden_dim), 2 * self.d_model)
        nn.init.zeros_(self.film_output.weight)
        nn.init.zeros_(self.film_output.bias)

    def forward(
        self,
        drug_pre,
        drug_ca,
        drug_flex,
        protein_flex_global,
        drug_mask,
        return_stats=False,
    ):
        if drug_pre.dim() != 3 or drug_ca.dim() != 3:
            raise ValueError("drug_pre and drug_ca must have shape [B, L_d, D].")
        if drug_pre.shape != drug_ca.shape:
            raise ValueError(
                "drug_pre and drug_ca shape mismatch: "
                f"{tuple(drug_pre.shape)} vs {tuple(drug_ca.shape)}."
            )
        if drug_ca.size(-1) != self.d_model:
            raise ValueError(f"Expected hidden dim {self.d_model}, got {drug_ca.size(-1)}.")
        if drug_mask.dim() != 2 or drug_mask.shape != drug_ca.shape[:2]:
            raise ValueError(
                "drug_mask must have shape [B, L_d], got "
                f"{tuple(drug_mask.shape)} for drug_ca {tuple(drug_ca.shape)}."
            )

        batch_size, drug_len, _ = drug_ca.shape
        device = drug_ca.device
        dtype = drug_ca.dtype
        mask = drug_mask.to(device=device).bool().unsqueeze(-1)
        mask_float = mask.to(dtype=dtype)

        if self.film_use_drug_flex:
            if drug_flex is None:
                raise ValueError("drug_flex is required when film_use_drug_flex=True.")
            if drug_flex.dim() == 1:
                drug_flex = drug_flex.unsqueeze(-1)
            if drug_flex.dim() != 2 or drug_flex.size(0) != batch_size:
                raise ValueError(
                    "drug_flex must have shape [B, drug_flex_dim], got "
                    f"{tuple(drug_flex.shape)}."
                )
            f_d = drug_flex[:, :1].to(device=device, dtype=dtype)
        else:
            f_d = drug_ca.new_zeros(batch_size, 1)
        f_d_broadcast = f_d.unsqueeze(1).expand(-1, drug_len, -1) * mask_float

        if self.film_use_protein_flex:
            if protein_flex_global is None:
                raise ValueError("protein_flex_global is required when film_use_protein_flex=True.")
            if protein_flex_global.dim() != 2 or protein_flex_global.shape != (batch_size, 3):
                raise ValueError(
                    "protein_flex_global must have shape [B, 3], got "
                    f"{tuple(protein_flex_global.shape)}."
                )
            f_p_global = protein_flex_global.to(device=device, dtype=dtype)
        else:
            f_p_global = drug_ca.new_zeros(batch_size, 3)
        f_p_global_broadcast = f_p_global.unsqueeze(1).expand(-1, drug_len, -1) * mask_float

        candidate_shift = drug_ca - drug_pre
        candidate_shift_norm = torch.norm(candidate_shift, p=2, dim=-1, keepdim=True) * mask_float
        shift_norm_feature = torch.log1p(candidate_shift_norm.clamp_min(0.0))

        if self.film_flex_encoding == 'mlp':
            drug_flex_feature = self.drug_flex_encoder(f_d_broadcast) * mask_float
            protein_flex_feature = self.protein_flex_global_encoder(f_p_global_broadcast) * mask_float
        else:
            drug_flex_feature = f_d_broadcast
            protein_flex_feature = f_p_global_broadcast

        film_condition = torch.cat(
            [
                drug_pre,
                drug_ca,
                candidate_shift,
                shift_norm_feature,
                drug_flex_feature,
                protein_flex_feature,
            ],
            dim=-1,
        )
        film_params = self.film_output(self.condition_encoder(film_condition))
        gamma_raw, beta_raw = film_params.chunk(2, dim=-1)
        gamma = self.gamma_scale * torch.tanh(gamma_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)
        if self.film_mode == 'scale_only':
            beta = torch.zeros_like(beta)
        elif self.film_mode == 'shift_only':
            gamma = torch.zeros_like(gamma)
        gamma = gamma * mask_float
        beta = beta * mask_float

        film_correction = (gamma * candidate_shift + beta) * mask_float
        drug_film = drug_ca + film_correction

        if not return_stats:
            return drug_film, {}

        adaptation_map = torch.norm(film_correction, p=2, dim=-1, keepdim=True) * mask_float
        drug_ca_norm = torch.norm(drug_ca, p=2, dim=-1, keepdim=True).clamp_min(self.eps)
        relative_adaptation_map = (adaptation_map / drug_ca_norm) * mask_float
        denom = mask_float.sum(dim=1).clamp_min(self.eps)
        pair_adaptation_score = (relative_adaptation_map * mask_float).sum(dim=1) / denom
        weighted_denom = (f_d_broadcast * mask_float).sum(dim=1).clamp_min(self.eps)
        weighted_pair_adaptation_score = (
            relative_adaptation_map * f_d_broadcast * mask_float
        ).sum(dim=1) / weighted_denom

        return drug_film, {
            'gamma': gamma,
            'beta': beta,
            'candidate_shift': candidate_shift,
            'candidate_shift_norm': candidate_shift_norm,
            'film_correction': film_correction,
            'adaptation_map': adaptation_map,
            'relative_adaptation_map': relative_adaptation_map,
            'pair_adaptation_score': pair_adaptation_score,
            'weighted_pair_adaptation_score': weighted_pair_adaptation_score,
        }


class ProteinInducedFlexGate(nn.Module):
    """Minimal protein-side flexibility-guided induced-fit gate.

    H_p: [B, L_p, D]
    M_p_d: [B, L_p, D]
    F_p: [B, L_p, 1]
    protein_mask: [B, L_p]
    """

    def __init__(self, d_model, gate_hidden_dim=128, beta_init=0.0):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 1, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.message_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, h_p, m_p_d, protein_flex, protein_mask):
        if h_p.dim() != 3 or m_p_d.dim() != 3:
            raise ValueError("h_p and m_p_d must have shape [B, L_p, D].")
        if h_p.shape != m_p_d.shape:
            raise ValueError(
                f"h_p and m_p_d shape mismatch: {tuple(h_p.shape)} vs {tuple(m_p_d.shape)}."
            )
        if protein_flex.dim() != 3 or protein_flex.size(-1) != 1:
            raise ValueError(
                "protein_flex must have shape [B, L_p, 1], got "
                f"{tuple(protein_flex.shape)}."
            )
        if protein_flex.shape[:2] != h_p.shape[:2]:
            raise ValueError(
                "protein_flex and protein representation length mismatch: "
                f"{tuple(protein_flex.shape)} vs {tuple(h_p.shape)}."
            )
        if protein_mask.dim() != 2 or protein_mask.shape != h_p.shape[:2]:
            raise ValueError(
                "protein_mask must have shape [B, L_p], got "
                f"{tuple(protein_mask.shape)} for h_p {tuple(h_p.shape)}."
            )

        mask = protein_mask.bool().unsqueeze(-1)
        mask_float = mask.to(dtype=h_p.dtype)
        f_p = protein_flex.to(device=h_p.device, dtype=h_p.dtype) * mask_float

        # X_p = concat(H_p, M_p_d, F_p): [B, L_p, 2D + 1]
        gate_input = torch.cat([h_p, m_p_d, f_p], dim=-1)
        g_p_d = torch.sigmoid(self.gate_mlp(gate_input)) * mask_float  # [B, L_p, 1]
        if_p_d = f_p * g_p_d  # [B, L_p, 1]
        delta_h_p_d = if_p_d * self.message_projection(m_p_d)  # [B, L_p, D]
        delta_h_p_d = delta_h_p_d * mask_float
        # m_p_d is already the protein-side TransformerDecoder output and thus
        # already contains the original protein residual/cross-attention/FFN path.
        # The induced branch only adds a flexibility-guided shift on top of it.
        h_p_tilde = self.layer_norm(m_p_d + self.beta * delta_h_p_d)
        h_p_tilde = h_p_tilde * mask_float

        debug_shapes = {
            'H_p': tuple(h_p.shape),
            'M_p_d': tuple(m_p_d.shape),
            'F_p': tuple(f_p.shape),
            'G_p_d': tuple(g_p_d.shape),
            'IF_p_d': tuple(if_p_d.shape),
            'Delta_H_p_d': tuple(delta_h_p_d.shape),
            'H_p_tilde': tuple(h_p_tilde.shape),
        }
        return h_p_tilde, {
            'protein_induced_gate': g_p_d,
            'protein_induced_if': if_p_d,
            'protein_induced_delta': delta_h_p_d,
            'protein_induced_beta': self.beta,
            'protein_induced_debug_shapes': debug_shapes,
        }


class DrugInducedFlexGate(nn.Module):
    """Minimal drug-side flexibility-guided induced-fit gate.

    H_d: [B, L_d, D]
    M_d_p: [B, L_d, D]
    F_d: [B, drug_flex_dim], currently expected to include one scalar column.
    F_p_global: [B, 3]
    drug_mask: [B, L_d]
    """

    def __init__(
        self,
        d_model,
        drug_flex_dim=1,
        protein_flex_global_dim=3,
        gate_hidden_dim=128,
        beta_init=0.0,
    ):
        super().__init__()
        self.drug_flex_dim = int(drug_flex_dim)
        self.protein_flex_global_dim = int(protein_flex_global_dim)
        gate_input_dim = 3 * d_model + self.drug_flex_dim + self.protein_flex_global_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
        )
        self.message_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, h_d, m_d_p, drug_flex, protein_flex_global, drug_mask):
        if h_d.dim() != 3 or m_d_p.dim() != 3:
            raise ValueError("h_d and m_d_p must have shape [B, L_d, D].")
        if h_d.shape != m_d_p.shape:
            raise ValueError(
                f"h_d and m_d_p shape mismatch: {tuple(h_d.shape)} vs {tuple(m_d_p.shape)}."
            )
        if drug_flex is None:
            raise ValueError("drug_flex is required for DrugInducedFlexGate.")
        if drug_flex.dim() == 1:
            drug_flex = drug_flex.unsqueeze(-1)
        if drug_flex.dim() != 2 or drug_flex.size(0) != h_d.size(0):
            raise ValueError(
                "drug_flex must have shape [B, drug_flex_dim], got "
                f"{tuple(drug_flex.shape)}."
            )
        if protein_flex_global is None or protein_flex_global.shape != (h_d.size(0), self.protein_flex_global_dim):
            raise ValueError(
                "protein_flex_global must have shape [B, 3], got "
                f"{None if protein_flex_global is None else tuple(protein_flex_global.shape)}."
            )
        if drug_mask.dim() != 2 or drug_mask.shape != h_d.shape[:2]:
            raise ValueError(
                "drug_mask must have shape [B, L_d], got "
                f"{tuple(drug_mask.shape)} for h_d {tuple(h_d.shape)}."
            )

        mask = drug_mask.bool().unsqueeze(-1)
        mask_float = mask.to(dtype=h_d.dtype)
        f_d = drug_flex.to(device=h_d.device, dtype=h_d.dtype)
        f_p_global = protein_flex_global.to(device=h_d.device, dtype=h_d.dtype)
        f_d_broadcast = f_d.unsqueeze(1).expand(-1, h_d.size(1), -1) * mask_float
        f_p_global_broadcast = f_p_global.unsqueeze(1).expand(-1, h_d.size(1), -1) * mask_float

        # X_d = concat(H_d, M_d_p, H_d * M_d_p, F_d, F_p_global)
        gate_input = torch.cat([
            h_d,
            m_d_p,
            h_d * m_d_p,
            f_d_broadcast,
            f_p_global_broadcast,
        ], dim=-1)
        g_d_p = torch.sigmoid(self.gate_mlp(gate_input)) * mask_float  # [B, L_d, 1]
        if_d_p = f_d_broadcast[:, :, :1] * g_d_p  # [B, L_d, 1]
        delta_h_d_p = if_d_p * self.message_projection(m_d_p)  # [B, L_d, D]
        delta_h_d_p = delta_h_d_p * mask_float
        # m_d_p is already the drug-side TransformerDecoder output and thus
        # already contains the original drug residual/cross-attention/FFN path.
        # The induced branch only adds a flexibility-guided shift on top of it.
        h_d_tilde = self.layer_norm(m_d_p + self.beta * delta_h_d_p)
        h_d_tilde = h_d_tilde * mask_float

        debug_shapes = {
            'H_d': tuple(h_d.shape),
            'M_d_p': tuple(m_d_p.shape),
            'F_d_broadcast': tuple(f_d_broadcast.shape),
            'F_p_global_broadcast': tuple(f_p_global_broadcast.shape),
            'G_d_p': tuple(g_d_p.shape),
            'IF_d_p': tuple(if_d_p.shape),
            'Delta_H_d_p': tuple(delta_h_d_p.shape),
            'H_d_tilde': tuple(h_d_tilde.shape),
        }
        return h_d_tilde, {
            'drug_induced_gate': g_d_p,
            'drug_induced_if': if_d_p,
            'drug_induced_delta': delta_h_d_p,
            'drug_induced_beta': self.beta,
            'drug_induced_debug_shapes': debug_shapes,
        }


class ProteinInducedFlexFiLM(nn.Module):
    """Protein-side FiLM modulation conditioned on pair interaction and F_p.

    This keeps the same external outputs as ProteinInducedFlexGate so training,
    IF-score export, and ablation code can reuse the existing path.
    """

    def __init__(self, d_model, gate_hidden_dim=128, beta_init=0.0):
        super().__init__()
        self.film_mlp = nn.Sequential(
            nn.Linear(2 * d_model + 1, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 2 * d_model + 1),
        )
        self.message_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, h_p, m_p_d, protein_flex, protein_mask):
        if h_p.dim() != 3 or m_p_d.dim() != 3:
            raise ValueError("h_p and m_p_d must have shape [B, L_p, D].")
        if h_p.shape != m_p_d.shape:
            raise ValueError(
                f"h_p and m_p_d shape mismatch: {tuple(h_p.shape)} vs {tuple(m_p_d.shape)}."
            )
        if protein_flex.dim() != 3 or protein_flex.size(-1) != 1:
            raise ValueError(
                "protein_flex must have shape [B, L_p, 1], got "
                f"{tuple(protein_flex.shape)}."
            )
        if protein_flex.shape[:2] != h_p.shape[:2]:
            raise ValueError(
                "protein_flex and protein representation length mismatch: "
                f"{tuple(protein_flex.shape)} vs {tuple(h_p.shape)}."
            )
        if protein_mask.dim() != 2 or protein_mask.shape != h_p.shape[:2]:
            raise ValueError(
                "protein_mask must have shape [B, L_p], got "
                f"{tuple(protein_mask.shape)} for h_p {tuple(h_p.shape)}."
            )

        mask = protein_mask.bool().unsqueeze(-1)
        mask_float = mask.to(dtype=h_p.dtype)
        f_p = protein_flex.to(device=h_p.device, dtype=h_p.dtype) * mask_float
        film_input = torch.cat([h_p, m_p_d, f_p], dim=-1)
        film_raw = self.film_mlp(film_input)
        gamma_raw, shift, strength_raw = torch.split(
            film_raw,
            [h_p.size(-1), h_p.size(-1), 1],
            dim=-1,
        )
        gamma = torch.tanh(gamma_raw) * mask_float
        shift = shift * mask_float
        strength = torch.sigmoid(strength_raw) * mask_float
        if_p_d = f_p * strength
        modulated_message = gamma * self.message_projection(m_p_d) + shift
        delta_h_p_d = if_p_d * modulated_message
        delta_h_p_d = delta_h_p_d * mask_float
        h_p_tilde = self.layer_norm(m_p_d + self.beta * delta_h_p_d)
        h_p_tilde = h_p_tilde * mask_float

        debug_shapes = {
            'H_p': tuple(h_p.shape),
            'M_p_d': tuple(m_p_d.shape),
            'F_p': tuple(f_p.shape),
            'FiLM_gamma_p': tuple(gamma.shape),
            'FiLM_shift_p': tuple(shift.shape),
            'IF_p_d': tuple(if_p_d.shape),
            'Delta_H_p_d': tuple(delta_h_p_d.shape),
            'H_p_tilde': tuple(h_p_tilde.shape),
        }
        return h_p_tilde, {
            'protein_induced_gate': strength,
            'protein_induced_if': if_p_d,
            'protein_induced_delta': delta_h_p_d,
            'protein_induced_beta': self.beta,
            'protein_induced_film_gamma': gamma,
            'protein_induced_film_shift': shift,
            'protein_induced_debug_shapes': debug_shapes,
        }


class DrugInducedFlexFiLM(nn.Module):
    """Drug-side FiLM modulation conditioned on pair interaction, F_d and F_p_global."""

    def __init__(
        self,
        d_model,
        drug_flex_dim=1,
        protein_flex_global_dim=3,
        gate_hidden_dim=128,
        beta_init=0.0,
    ):
        super().__init__()
        self.drug_flex_dim = int(drug_flex_dim)
        self.protein_flex_global_dim = int(protein_flex_global_dim)
        input_dim = 3 * d_model + self.drug_flex_dim + self.protein_flex_global_dim
        self.film_mlp = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 2 * d_model + 1),
        )
        self.message_projection = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, h_d, m_d_p, drug_flex, protein_flex_global, drug_mask):
        if h_d.dim() != 3 or m_d_p.dim() != 3:
            raise ValueError("h_d and m_d_p must have shape [B, L_d, D].")
        if h_d.shape != m_d_p.shape:
            raise ValueError(
                f"h_d and m_d_p shape mismatch: {tuple(h_d.shape)} vs {tuple(m_d_p.shape)}."
            )
        if drug_flex is None:
            raise ValueError("drug_flex is required for DrugInducedFlexFiLM.")
        if drug_flex.dim() == 1:
            drug_flex = drug_flex.unsqueeze(-1)
        if drug_flex.dim() != 2 or drug_flex.size(0) != h_d.size(0):
            raise ValueError(
                "drug_flex must have shape [B, drug_flex_dim], got "
                f"{tuple(drug_flex.shape)}."
            )
        if protein_flex_global is None or protein_flex_global.shape != (h_d.size(0), self.protein_flex_global_dim):
            raise ValueError(
                "protein_flex_global must have shape [B, 3], got "
                f"{None if protein_flex_global is None else tuple(protein_flex_global.shape)}."
            )
        if drug_mask.dim() != 2 or drug_mask.shape != h_d.shape[:2]:
            raise ValueError(
                "drug_mask must have shape [B, L_d], got "
                f"{tuple(drug_mask.shape)} for h_d {tuple(h_d.shape)}."
            )

        mask = drug_mask.bool().unsqueeze(-1)
        mask_float = mask.to(dtype=h_d.dtype)
        f_d = drug_flex.to(device=h_d.device, dtype=h_d.dtype)
        f_p_global = protein_flex_global.to(device=h_d.device, dtype=h_d.dtype)
        f_d_broadcast = f_d.unsqueeze(1).expand(-1, h_d.size(1), -1) * mask_float
        f_p_global_broadcast = f_p_global.unsqueeze(1).expand(-1, h_d.size(1), -1) * mask_float
        film_input = torch.cat([
            h_d,
            m_d_p,
            h_d * m_d_p,
            f_d_broadcast,
            f_p_global_broadcast,
        ], dim=-1)
        film_raw = self.film_mlp(film_input)
        gamma_raw, shift, strength_raw = torch.split(
            film_raw,
            [h_d.size(-1), h_d.size(-1), 1],
            dim=-1,
        )
        gamma = torch.tanh(gamma_raw) * mask_float
        shift = shift * mask_float
        strength = torch.sigmoid(strength_raw) * mask_float
        if_d_p = f_d_broadcast[:, :, :1] * strength
        modulated_message = gamma * self.message_projection(m_d_p) + shift
        delta_h_d_p = if_d_p * modulated_message
        delta_h_d_p = delta_h_d_p * mask_float
        h_d_tilde = self.layer_norm(m_d_p + self.beta * delta_h_d_p)
        h_d_tilde = h_d_tilde * mask_float

        debug_shapes = {
            'H_d': tuple(h_d.shape),
            'M_d_p': tuple(m_d_p.shape),
            'F_d_broadcast': tuple(f_d_broadcast.shape),
            'F_p_global_broadcast': tuple(f_p_global_broadcast.shape),
            'FiLM_gamma_d': tuple(gamma.shape),
            'FiLM_shift_d': tuple(shift.shape),
            'IF_d_p': tuple(if_d_p.shape),
            'Delta_H_d_p': tuple(delta_h_d_p.shape),
            'H_d_tilde': tuple(h_d_tilde.shape),
        }
        return h_d_tilde, {
            'drug_induced_gate': strength,
            'drug_induced_if': if_d_p,
            'drug_induced_delta': delta_h_d_p,
            'drug_induced_beta': self.beta,
            'drug_induced_film_gamma': gamma,
            'drug_induced_film_shift': shift,
            'drug_induced_debug_shapes': debug_shapes,
        }


class FlexAttentionBias(nn.Module):
    """Pair-specific flexibility bias added to cross-attention logits.

    This module changes *where* the flexibility prior acts:
    instead of modulating the cross-attention output after attention, it adds a
    learnable scalar bias to every query-key attention score before softmax.

    drug_to_protein output: [B, 1, L_d, L_p] for drug queries attending protein keys.
    protein_to_drug output: [B, 1, L_p, L_d] for protein queries attending drug keys.
    """

    def __init__(
        self,
        drug_flex_dim=1,
        protein_flex_global_dim=3,
        hidden_dim=64,
        bias_init=0.0,
    ):
        super().__init__()
        self.drug_flex_dim = int(drug_flex_dim)
        self.protein_flex_global_dim = int(protein_flex_global_dim)
        input_dim = self.drug_flex_dim + 1 + self.protein_flex_global_dim
        self.bias_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialized to zero, so the first forward pass is exactly no-bias.
        self.bias_scale = nn.Parameter(torch.tensor(float(bias_init)))

    def _check_inputs(self, drug_flex, protein_flex, protein_flex_global):
        if drug_flex is None:
            raise ValueError("drug_flex is required for FlexAttentionBias.")
        if protein_flex is None:
            raise ValueError("protein_flex is required for FlexAttentionBias.")
        if drug_flex.dim() == 1:
            drug_flex = drug_flex.unsqueeze(-1)
        if drug_flex.dim() != 2 or drug_flex.size(-1) != self.drug_flex_dim:
            raise ValueError(
                "drug_flex must have shape [B, drug_flex_dim], got "
                f"{tuple(drug_flex.shape)}."
            )
        if protein_flex.dim() != 3 or protein_flex.size(-1) != 1:
            raise ValueError(
                "protein_flex must have shape [B, L_p, 1], got "
                f"{tuple(protein_flex.shape)}."
            )
        expected_global = (drug_flex.size(0), self.protein_flex_global_dim)
        if protein_flex_global is None or protein_flex_global.shape != expected_global:
            raise ValueError(
                "protein_flex_global must have shape [B, 3], got "
                f"{None if protein_flex_global is None else tuple(protein_flex_global.shape)}."
            )
        if protein_flex.size(0) != drug_flex.size(0):
            raise ValueError(
                "Batch size mismatch between drug_flex and protein_flex: "
                f"{drug_flex.size(0)} vs {protein_flex.size(0)}."
            )
        return drug_flex, protein_flex, protein_flex_global

    def drug_to_protein(self, drug_flex, protein_flex, protein_flex_global, drug_len):
        """Build bias for M_d_p = CrossAttn(Q=H_d, K=H_p, V=H_p).

        F_d: [B, drug_flex_dim] is broadcast over drug tokens and protein keys.
        F_p: [B, L_p, 1] varies by protein key residue.
        F_p_global: [B, 3] gives pair-level protein flexibility context.
        """
        drug_flex, protein_flex, protein_flex_global = self._check_inputs(
            drug_flex, protein_flex, protein_flex_global
        )
        B, L_p, _ = protein_flex.shape
        device = protein_flex.device
        dtype = protein_flex.dtype
        f_d = drug_flex.to(device=device, dtype=dtype)
        f_p_global = protein_flex_global.to(device=device, dtype=dtype)

        f_d_pair = f_d.unsqueeze(1).unsqueeze(2).expand(B, drug_len, L_p, -1)
        f_p_pair = protein_flex.unsqueeze(1).expand(B, drug_len, L_p, 1)
        f_pg_pair = f_p_global.unsqueeze(1).unsqueeze(2).expand(B, drug_len, L_p, -1)
        pair_feature = torch.cat([f_d_pair, f_p_pair, f_pg_pair], dim=-1)
        raw_bias = self.bias_mlp(pair_feature).squeeze(-1)  # [B, L_d, L_p]
        return (self.bias_scale * torch.tanh(raw_bias)).unsqueeze(1)

    def protein_to_drug(self, drug_flex, protein_flex, protein_flex_global, drug_len):
        """Build bias for M_p_d = CrossAttn(Q=H_p, K=H_d, V=H_d).

        Here residue-level F_p varies by protein query position, while the
        drug-level F_d is broadcast over all drug keys.
        """
        drug_flex, protein_flex, protein_flex_global = self._check_inputs(
            drug_flex, protein_flex, protein_flex_global
        )
        B, L_p, _ = protein_flex.shape
        device = protein_flex.device
        dtype = protein_flex.dtype
        f_d = drug_flex.to(device=device, dtype=dtype)
        f_p_global = protein_flex_global.to(device=device, dtype=dtype)

        f_d_pair = f_d.unsqueeze(1).unsqueeze(2).expand(B, L_p, drug_len, -1)
        f_p_pair = protein_flex.unsqueeze(2).expand(B, L_p, drug_len, 1)
        f_pg_pair = f_p_global.unsqueeze(1).unsqueeze(2).expand(B, L_p, drug_len, -1)
        pair_feature = torch.cat([f_d_pair, f_p_pair, f_pg_pair], dim=-1)
        raw_bias = self.bias_mlp(pair_feature).squeeze(-1)  # [B, L_p, L_d]
        return (self.bias_scale * torch.tanh(raw_bias)).unsqueeze(1)


class TAPB(nn.Module):
    def __init__(self, model_configs):
        super().__init__()
        # Params
        d_model = model_configs['DrugEncoder']['d_model']
        n_heads = model_configs['DrugEncoder']['n_head']
        fusion_n_heads = model_configs['TransformerDeocder']['n_head']
        vocab_size = model_configs['DrugEncoder']['vocab_size']
        self.d_model = d_model
        self.structure_gearnet_semantic_dim = int(
            model_configs.get('structure_gearnet_semantic_dim', 0)
        )
        if self.structure_gearnet_semantic_dim < 0:
            raise ValueError('structure_gearnet_semantic_dim must be non-negative.')
        self.protein_semantic_projection = (
            nn.Linear(d_model, self.structure_gearnet_semantic_dim, bias=False)
            if self.structure_gearnet_semantic_dim > 0 else None
        )
        self.structure_gearnet_response_distillation_dim = int(
            model_configs.get('structure_gearnet_response_distillation_dim', 0)
        )
        if self.structure_gearnet_response_distillation_dim < 0:
            raise ValueError(
                'structure_gearnet_response_distillation_dim must be non-negative.'
            )
        self.protein_response_distillation_projection = (
            nn.Linear(
                d_model, self.structure_gearnet_response_distillation_dim, bias=False
            ) if self.structure_gearnet_response_distillation_dim > 0 else None
        )
        self.encoder_n_heads = n_heads
        self.fusion_n_heads = fusion_n_heads
        d_esm = int(model_configs.get(
            'protein_feature_dim', model_configs['PrEncoder']['d_model']
        ))

        self.use_mlm = model_configs.get('use_mlm', True)
        self.use_drug_flex = model_configs.get('use_drug_flex', False)
        self.drug_flex_dim = int(model_configs.get('drug_flex_dim', 1))
        self.use_protein_flex = model_configs.get('use_protein_flex', False)
        self.use_protein_induced_gate = model_configs.get('use_protein_induced_gate', False)
        self.use_drug_induced_gate = model_configs.get('use_drug_induced_gate', False)
        self.use_flex_attention_bias = bool(model_configs.get('use_flex_attention_bias', False))
        self.flex_attention_bias_hidden_dim = int(model_configs.get('flex_attention_bias_hidden_dim', 64))
        self.flex_attention_bias_init = float(model_configs.get('flex_attention_bias_init', 0.0))
        self.flex_attention_bias_side = model_configs.get('flex_attention_bias_side', 'both')
        if self.flex_attention_bias_side not in {'both', 'protein', 'drug'}:
            raise ValueError("flex_attention_bias_side must be one of 'both', 'protein', or 'drug'.")
        self.use_flex_calibration = bool(model_configs.get('use_flex_calibration', False))
        self.flex_calibration_side = model_configs.get('flex_calibration_side', 'protein')
        if self.flex_calibration_side != 'protein':
            raise ValueError("Only flex_calibration_side='protein' is supported in this minimal version.")
        self.flex_hidden_dim = int(model_configs.get('flex_hidden_dim', 128))
        self.flex_dropout = float(model_configs.get('flex_dropout', 0.1))
        self.flex_lambda = float(model_configs.get('flex_lambda', 0.5))
        self.return_flex_stats = bool(model_configs.get('return_flex_stats', False))
        self.use_drug_flex_condition = bool(model_configs.get('use_drug_flex_condition', True))
        self.protein_flex_perturbation = model_configs.get('protein_flex_perturbation', 'none')
        if self.protein_flex_perturbation not in {'none', 'shuffle', 'random', 'residue_shuffle'}:
            raise ValueError("protein_flex_perturbation must be one of 'none', 'shuffle', 'random', or 'residue_shuffle'.")
        self.use_flex_film = bool(model_configs.get('use_flex_film', False))
        self.film_side = model_configs.get('film_side', 'protein')
        if self.film_side not in {'protein', 'drug', 'both'}:
            raise ValueError("film_side must be one of 'protein', 'drug', or 'both'.")
        self.film_hidden_dim = int(model_configs.get('film_hidden_dim', 128))
        self.film_dropout = float(model_configs.get('film_dropout', 0.1))
        self.film_gamma_scale = float(model_configs.get('film_gamma_scale', 0.5))
        self.film_beta_scale = float(model_configs.get('film_beta_scale', 0.1))
        self.film_use_protein_flex = bool(model_configs.get('film_use_protein_flex', True))
        self.film_use_drug_flex = bool(model_configs.get('film_use_drug_flex', True))
        self.protein_film_use_protein_flex = bool(model_configs.get(
            'protein_film_use_protein_flex', self.film_use_protein_flex
        ))
        self.protein_film_use_drug_flex = bool(model_configs.get(
            'protein_film_use_drug_flex', self.film_use_drug_flex
        ))
        self.drug_film_use_drug_flex = bool(model_configs.get(
            'drug_film_use_drug_flex', self.film_use_drug_flex
        ))
        self.drug_film_use_protein_flex = bool(model_configs.get(
            'drug_film_use_protein_flex', self.film_use_protein_flex
        ))
        self.film_mode = model_configs.get('film_mode', 'scale_shift')
        if self.film_mode not in {'scale_shift', 'scale_only', 'shift_only'}:
            raise ValueError("film_mode must be one of 'scale_shift', 'scale_only', or 'shift_only'.")
        self.film_flex_encoding = model_configs.get('film_flex_encoding', 'raw')
        if self.film_flex_encoding not in {'raw', 'mlp'}:
            raise ValueError("film_flex_encoding must be either 'raw' or 'mlp'.")
        self.film_flex_embed_dim = int(model_configs.get('film_flex_embed_dim', 16))
        self.return_film_stats = bool(model_configs.get('return_film_stats', False))
        self.film_protein_flex_perturbation = model_configs.get('film_protein_flex_perturbation', 'none')
        self.film_protein_flex_perturbation_seed = int(model_configs.get('film_protein_flex_perturbation_seed', 6))
        if self.film_protein_flex_perturbation not in {'none', 'shuffle', 'random', 'residue_shuffle'}:
            raise ValueError("film_protein_flex_perturbation must be one of 'none', 'shuffle', 'random', or 'residue_shuffle'.")
        self.relation_alignment_enabled = bool(model_configs.get('relation_alignment_enabled', False))
        self.alignment_target = model_configs.get('alignment_target', 'none')
        if self.alignment_target not in {'none', 'pre', 'post', 'response'}:
            raise ValueError("alignment_target must be one of 'none', 'pre', 'post', or 'response'.")
        if not self.relation_alignment_enabled:
            self.alignment_target = 'none'
        self.relation_alignment_side = model_configs.get('relation_alignment_side', 'both')
        if self.relation_alignment_side not in {'both', 'drug', 'protein'}:
            raise ValueError("relation_alignment_side must be one of 'both', 'drug', or 'protein'.")
        self.response_dim = int(model_configs.get('response_dim', 128))
        self.response_source = model_configs.get('response_source', 'cross_attention_residual')
        if self.response_source != 'cross_attention_residual':
            raise ValueError("Only response_source='cross_attention_residual' is supported.")
        self.induced_modulation_type = model_configs.get('induced_modulation_type', 'gate')
        if self.induced_modulation_type not in {'gate', 'film'}:
            raise ValueError("induced_modulation_type must be either 'gate' or 'film'.")
        self.induced_beta_init = float(model_configs.get('induced_beta_init', 0.0))
        self.induced_gate_hidden_dim = int(model_configs.get('induced_gate_hidden_dim', 128))
        self.drug_induced_beta_init = float(model_configs.get('drug_induced_beta_init', self.induced_beta_init))
        self.drug_induced_gate_hidden_dim = int(model_configs.get('drug_induced_gate_hidden_dim', self.induced_gate_hidden_dim))
        self.predictor_include_induced_delta_pools = bool(model_configs.get('predictor_include_induced_delta_pools', False))
        self.task_mode = str(model_configs.get('task_mode', 'binary'))
        if self.task_mode not in {'binary', 'regression'}:
            raise ValueError("task_mode must be either 'binary' or 'regression'.")
        self.predictor_output_dim = 1 if self.task_mode == 'regression' else 2
        self.predictor_pooling = model_configs.get('predictor_pooling', 'mean')
        if self.predictor_pooling not in {'mean', 'protein_attention'}:
            raise ValueError("predictor_pooling must be either 'mean' or 'protein_attention'.")
        self.predictor_head = model_configs.get('predictor_head', 'linear')
        if self.predictor_head not in {'linear', 'interaction_mlp'}:
            raise ValueError("predictor_head must be either 'linear' or 'interaction_mlp'.")
        self.predictor_dropout = float(model_configs.get('predictor_dropout', 0.1))
        if not 0.0 <= self.predictor_dropout < 1.0:
            raise ValueError('predictor_dropout must be in [0, 1).')
        self.protein_flex_pooling = model_configs.get('protein_flex_pooling', 'mean_max_std')
        if self.protein_flex_pooling != 'mean_max_std':
            raise ValueError("Only protein_flex_pooling='mean_max_std' is supported now.")

        self.drug_encoder_backend, pretrained_molformer_path = (
            resolve_drug_encoder_config(model_configs)
        )
        if (
            self.drug_encoder_backend == "pretrained_molformer"
            and self.relation_alignment_enabled
        ):
            raise ValueError(
                "Pretrained MoLFormer backend does not support relation alignment."
            )
        self.pretrained_drug_encoder = None
        self.pretrained_drug_projection = None

        # Drug encoder
        self.precompute_freqs_cis = precompute_freqs_cis(d_model // n_heads, 4000)
        if self.drug_encoder_backend == "native":
            self.drug_encoder = TransformerEncoder(config=model_configs['DrugEncoder'])
            self.mlm_head = MLMHead(d_model, vocab_size) if self.use_mlm else None

            # Fixed fallback drug teacher. This preserves a non-trainable token-level
            # teacher when offline MolFormer global embeddings are not present in the
            # batch. If offline embeddings are later added to the dataloader, replace
            # this teacher source without changing the relation loss.
            self.drug_teacher_embedding = nn.Embedding(vocab_size, d_model)
            self.drug_teacher_embedding.weight.data.copy_(self.drug_encoder.Embeddings.weight.data)
            self.drug_teacher_embedding.weight.requires_grad_(False)
        else:
            self.pretrained_drug_encoder, hidden_size = (
                load_frozen_pretrained_molformer(pretrained_molformer_path)
            )
            self.pretrained_drug_projection = nn.Linear(hidden_size, d_model)
            self.drug_encoder = None
            self.mlm_head = MLMHead(d_model, vocab_size) if self.use_mlm else None
            self.drug_teacher_embedding = None

        # Fusion
        self.aggregator = TransformerDecoder(config=model_configs['TransformerDeocder'])
        self.aggregator_protein = TransformerDecoder(config=model_configs['TransformerDeocder'])
        protein_induced_cls = (
            ProteinInducedFlexFiLM
            if self.induced_modulation_type == 'film'
            else ProteinInducedFlexGate
        )
        drug_induced_cls = (
            DrugInducedFlexFiLM
            if self.induced_modulation_type == 'film'
            else DrugInducedFlexGate
        )
        self.protein_induced_gate = (
            protein_induced_cls(
                d_model=self.d_model,
                gate_hidden_dim=self.induced_gate_hidden_dim,
                beta_init=self.induced_beta_init,
            )
            if self.use_protein_induced_gate else None
        )
        self.drug_induced_gate = (
            drug_induced_cls(
                d_model=self.d_model,
                drug_flex_dim=self.drug_flex_dim,
                protein_flex_global_dim=3,
                gate_hidden_dim=self.drug_induced_gate_hidden_dim,
                beta_init=self.drug_induced_beta_init,
            )
            if self.use_drug_induced_gate else None
        )
        self.flex_attention_bias = (
            FlexAttentionBias(
                drug_flex_dim=self.drug_flex_dim,
                protein_flex_global_dim=3,
                hidden_dim=self.flex_attention_bias_hidden_dim,
                bias_init=self.flex_attention_bias_init,
            )
            if self.use_flex_attention_bias else None
        )
        self.flex_calibration = None
        self.flex_film = None
        self._printed_flex_calibration_debug = False
        self._printed_flex_film_debug = False
        self._printed_protein_induced_gate_debug = False
        self._printed_drug_induced_gate_debug = False
        self._printed_flex_attention_bias_debug = False
        self.protein_predictor_attention = (
            nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Tanh(), nn.Linear(self.d_model, 1, bias=False))
            if self.predictor_pooling == 'protein_attention' else None
        )
        predictor_dim = 4 * self.d_model if self.predictor_head == 'interaction_mlp' else 2 * self.d_model
        if self.use_drug_flex:
            predictor_dim += self.drug_flex_dim
        if self.use_protein_flex:
            # pooled protein flexibility: [mean_flex, max_flex, std_flex]
            predictor_dim += 3
        if self.predictor_include_induced_delta_pools:
            if self.use_drug_induced_gate:
                predictor_dim += self.d_model
            if self.use_protein_induced_gate:
                predictor_dim += self.d_model
        self.classifier = (
            nn.Sequential(nn.Linear(predictor_dim, self.d_model), nn.GELU(), nn.Dropout(self.predictor_dropout), nn.Linear(self.d_model, self.predictor_output_dim))
            if self.predictor_head == 'interaction_mlp' else nn.Linear(predictor_dim, self.predictor_output_dim)
        )
        self.drug_response_head = (
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.response_dim),
            )
            if self.alignment_target != 'none' else None
        )
        self.protein_response_head = (
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.response_dim),
            )
            if self.alignment_target != 'none' else None
        )

        # Protein feature projection: ESM embedding dimension -> TAPB hidden dimension.
        self.pr_linear = nn.Linear(d_esm, d_model)

        


        # Instantiate the new module after all baseline modules so enabling it
        # does not change the random initialization order of the original TAPB.
        self.flex_calibration = (
            PairSpecificFlexibilityCalibration(
                d_model=self.d_model,
                hidden_dim=self.flex_hidden_dim,
                dropout=self.flex_dropout,
                lambda_flex=self.flex_lambda,
                use_drug_flex_condition=self.use_drug_flex_condition,
            )
            if self.use_flex_calibration else None
        )
        self.flex_film = (
            FlexibilityConditionedFiLM(
                d_model=self.d_model,
                film_hidden_dim=self.film_hidden_dim,
                film_dropout=self.film_dropout,
                gamma_scale=self.film_gamma_scale,
                beta_scale=self.film_beta_scale,
                film_use_protein_flex=self.protein_film_use_protein_flex,
                film_use_drug_flex=self.protein_film_use_drug_flex,
                film_mode=self.film_mode,
                film_flex_encoding=self.film_flex_encoding,
                film_flex_embed_dim=self.film_flex_embed_dim,
            )
            if self.use_flex_film and self.film_side in {'protein', 'both'} else None
        )
        self.drug_flex_film = (
            DrugFlexibilityConditionedFiLM(
                d_model=self.d_model,
                film_hidden_dim=self.film_hidden_dim,
                film_dropout=self.film_dropout,
                gamma_scale=self.film_gamma_scale,
                beta_scale=self.film_beta_scale,
                film_use_drug_flex=self.drug_film_use_drug_flex,
                film_use_protein_flex=self.drug_film_use_protein_flex,
                film_mode=self.film_mode,
                film_flex_encoding=self.film_flex_encoding,
                film_flex_embed_dim=self.film_flex_embed_dim,
            )
            if self.use_flex_film and self.film_side in {'drug', 'both'} else None
        )


    def encode_drug(self, input_drugs, freqs_cis):
        if self.drug_encoder_backend == "pretrained_molformer":
            self.pretrained_drug_encoder.eval()
            with torch.no_grad():
                outputs = self.pretrained_drug_encoder(
                    input_ids=input_drugs['input_ids'],
                    attention_mask=input_drugs['attention_mask'].bool(),
                )
                hidden_states = outputs.last_hidden_state
            return self.pretrained_drug_projection(hidden_states)

        drug_id = input_drugs['input_ids']

        drug_padding_mask = ~input_drugs['attention_mask'].bool()
        bz, len_d = drug_id.size()
        drug_attn_mask = drug_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.encoder_n_heads, len_d, -1)
        drug_f = self.drug_encoder(drug_id, drug_attn_mask, freqs_cis[:len_d, :].to(drug_id.device))
        return drug_f

    def encode_protein(self, pr_f, pr_mask):
        pr_f = self.pr_linear(pr_f)
        return pr_f, pr_mask

    def fusion(self, drug_f, pr_f, drug_padding_mask, protein_padding_mask, cross_attn_bias=None, return_cross_update=False):
        bz, len_d, _ = drug_f.size()
        drug_attn_mask = ~drug_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.fusion_n_heads, len_d, -1)
        cross_attn_mask = ~protein_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.fusion_n_heads, len_d,
                                                                                        -1)
        if return_cross_update:
            fusion_f, attention_map, cross_update = self.aggregator(
                src=pr_f,
                tgt=drug_f,
                self_attn_mask=drug_attn_mask,
                cross_attn_mask=cross_attn_mask,
                cross_attn_bias=cross_attn_bias,
                return_cross_update=True,
            )
            return fusion_f, attention_map, cross_update
        fusion_f, attention_map = self.aggregator(
            src=pr_f,
            tgt=drug_f,
            self_attn_mask=drug_attn_mask,
            cross_attn_mask=cross_attn_mask,
            cross_attn_bias=cross_attn_bias
        )
        return fusion_f, attention_map

    def fusion_protein(self, pr_f, drug_f, protein_padding_mask, drug_padding_mask, cross_attn_bias=None, return_cross_update=False):
        bz, len_p, _ = pr_f.size()
        protein_attn_mask = ~protein_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.fusion_n_heads, len_p, -1)
        cross_attn_mask = ~drug_padding_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, self.fusion_n_heads, len_p, -1)
        if return_cross_update:
            fusion_f, attention_map, cross_update = self.aggregator_protein(
                src=drug_f,
                tgt=pr_f,
                self_attn_mask=protein_attn_mask,
                cross_attn_mask=cross_attn_mask,
                cross_attn_bias=cross_attn_bias,
                return_cross_update=True,
            )
            return fusion_f, attention_map, cross_update
        fusion_f, attention_map = self.aggregator_protein(
            src=drug_f,
            tgt=pr_f,
            self_attn_mask=protein_attn_mask,
            cross_attn_mask=cross_attn_mask,
            cross_attn_bias=cross_attn_bias
        )
        return fusion_f, attention_map

    def _relation_drug_teacher(self, input_drugs, drug_mask):
        token_ids = input_drugs['input_ids'].to(self.drug_teacher_embedding.weight.device)
        teacher_tokens = self.drug_teacher_embedding(token_ids).to(device=drug_mask.device).detach()
        return self.masked_mean(teacher_tokens, drug_mask).detach()

    def _relation_protein_teacher(self, input_proteins, protein_mask):
        mask = protein_mask.bool().clone()
        # Exclude existing BOS/EOS-style boundary tokens when present. If the
        # dataset contains plain residue features, this only removes boundaries;
        # relation alignment remains a regularizer and can be disabled.
        for row in range(mask.size(0)):
            valid = torch.nonzero(mask[row], as_tuple=False).flatten()
            if valid.numel() >= 3:
                mask[row, valid[0]] = False
                mask[row, valid[-1]] = False
        return self.masked_mean(input_proteins.detach(), mask.to(input_proteins.device)).detach()

    def _project_relation_response(self, pooled, head):
        return F.normalize(head(pooled), p=2, dim=-1, eps=1e-8)

    def _perturb_protein_flex(self, protein_flex, protein_mask, perturbation_name, protein_ids=None):
        if perturbation_name == "none":
            return protein_flex
        if protein_flex is None:
            return None
        mask = protein_mask.to(device=protein_flex.device).bool().unsqueeze(-1)
        mask_float = mask.to(dtype=protein_flex.dtype)
        if perturbation_name == "shuffle":
            perturbed = protein_flex if protein_flex.size(0) <= 1 else protein_flex[torch.randperm(protein_flex.size(0), device=protein_flex.device)]
        elif perturbation_name == "random":
            perturbed = torch.rand_like(protein_flex)
        elif perturbation_name == "residue_shuffle":
            if protein_ids is None or len(protein_ids) != protein_flex.size(0):
                raise ValueError("residue_shuffle requires one stable protein_id per batch sample.")
            perturbed = torch.zeros_like(protein_flex)
            for row, protein_id in enumerate(protein_ids):
                positions = torch.nonzero(mask[row, :, 0], as_tuple=False).flatten()
                if positions.numel() <= 1:
                    perturbed[row, positions] = protein_flex[row, positions]
                    continue
                digest = hashlib.sha256(f"{self.film_protein_flex_perturbation_seed}:{protein_id}".encode("utf-8")).digest()
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
                permutation = torch.randperm(positions.numel(), generator=generator).to(protein_flex.device)
                perturbed[row, positions] = protein_flex[row, positions.index_select(0, permutation)]
        else:
            raise ValueError("protein flexibility perturbation must be one of none, shuffle, random, or residue_shuffle.")
        return perturbed * mask_float

    def _prepare_calibration_protein_flex(self, protein_flex, protein_mask, protein_ids=None):
        return self._perturb_protein_flex(protein_flex, protein_mask, self.protein_flex_perturbation, protein_ids)

    def _prepare_film_protein_flex(self, protein_flex, protein_mask, protein_ids=None):
        return self._perturb_protein_flex(protein_flex, protein_mask, self.film_protein_flex_perturbation, protein_ids)



    @staticmethod
    def _real_residue_mask(token_mask):
        mask = token_mask.bool().clone()
        for row in range(mask.size(0)):
            positions = torch.nonzero(mask[row], as_tuple=False).flatten()
            if positions.numel() >= 3:
                mask[row, positions[0]] = False
                mask[row, positions[-1]] = False
        return mask

    @staticmethod
    def masked_mean(x, mask):
        # x: [batch_size, seq_len, hidden_dim], mask: [batch_size, seq_len]
        return (x * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).clamp(min=1).float()

    def masked_protein_attention_pool(self, x, mask):
        """Pool protein tokens with learned weights while excluding invalid tokens."""
        if self.protein_predictor_attention is None:
            raise RuntimeError('protein attention pooling was not configured.')
        valid = mask.bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError('Every protein sample requires at least one valid token for attention pooling.')
        scores = self.protein_predictor_attention(x).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * valid.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=torch.finfo(weights.dtype).eps)
        return (x * weights.unsqueeze(-1)).sum(dim=1), weights

    @staticmethod
    def pool_protein_flex(protein_flex, protein_mask):
        """Pool residue-level F_p into global [mean, max, std].

        protein_flex: [B, L_p, 1], already aligned with protein_mask.
        protein_mask: [B, L_p], 1 for valid protein tokens, 0 for padding.
        return: [B, 3]
        """
        if protein_flex.dim() != 3 or protein_flex.size(-1) != 1:
            raise ValueError(
                "protein_flex must have shape [B, L_p, 1], got "
                f"{tuple(protein_flex.shape)}."
            )
        if protein_mask.dim() != 2:
            raise ValueError(
                "protein_mask must have shape [B, L_p], got "
                f"{tuple(protein_mask.shape)}."
            )
        if protein_flex.shape[:2] != protein_mask.shape:
            raise ValueError(
                "protein_flex and protein_mask length mismatch: "
                f"{tuple(protein_flex.shape)} vs {tuple(protein_mask.shape)}."
            )

        mask = protein_mask.bool().unsqueeze(-1)
        mask_float = mask.float()
        denom = mask_float.sum(dim=1).clamp(min=1.0)
        flex = protein_flex.float()

        mean_flex = (flex * mask_float).sum(dim=1) / denom
        max_source = flex.masked_fill(~mask, torch.finfo(flex.dtype).min)
        max_flex = max_source.max(dim=1).values
        max_flex = torch.where(torch.isfinite(max_flex), max_flex, torch.zeros_like(max_flex))
        var_flex = (((flex - mean_flex.unsqueeze(1)) ** 2) * mask_float).sum(dim=1) / denom
        std_flex = torch.sqrt(var_flex.clamp(min=0.0) + 1e-12)
        return torch.cat([mean_flex, max_flex, std_flex], dim=-1)


    def forward(
        self,
        input_drugs,
        input_proteins,
        pr_mask=None,
        masked_drugs=None,
        drug_flex=None,
        protein_flex=None,
        return_intermediates=False,
        protein_residue_mask=None,
        protein_ids=None,
    ):
        # 1. 基础编码
        drug_f = self.encode_drug(input_drugs, self.precompute_freqs_cis)  # [B, Ld, 256]
        pr_f, pr_mask = self.encode_protein(input_proteins, pr_mask)  # [B, Lp, 256]
        drug_attn_mask = input_drugs['attention_mask']


        protein_before = pr_f
        pr_mask_extended = pr_mask
        induced_gate_out = {}
        protein_flex_global = None
        needs_protein_flex = (
            self.use_protein_flex
            or self.use_protein_induced_gate
            or self.use_drug_induced_gate
            or self.use_flex_attention_bias
            or self.use_flex_calibration
            or (
                self.use_flex_film
                and (
                    (self.film_side in {'protein', 'both'} and self.protein_film_use_protein_flex)
                    or (self.film_side in {'drug', 'both'} and self.drug_film_use_protein_flex)
                )
            )
        )
        if needs_protein_flex:
            if protein_flex is None:
                raise ValueError(
                    'Protein flexibility is required but protein_flex is None. Provide '
                    'batch_protein_flex with shape [B, L_p, 1].'
                )
            protein_flex = protein_flex.to(device=pr_f.device, dtype=pr_f.dtype)
            protein_flex_global = self.pool_protein_flex(
                protein_flex,
                pr_mask_extended,
            ).to(dtype=pr_f.dtype)

        drug_to_protein_bias = None
        protein_to_drug_bias = None
        if self.use_flex_attention_bias:
            if drug_flex is None:
                raise ValueError(
                    'use_flex_attention_bias=True but drug_flex is None. Provide '
                    'batch_drug_flex with shape [B, drug_flex_dim].'
                )
            drug_flex_for_bias = drug_flex.to(device=pr_f.device, dtype=pr_f.dtype)
            if self.flex_attention_bias_side in {'both', 'drug'}:
                # Drug-side message: M_d_p = CrossAttn(Q=H_d, K=H_p, V=H_p)
                drug_to_protein_bias = self.flex_attention_bias.drug_to_protein(
                    drug_flex_for_bias,
                    protein_flex,
                    protein_flex_global,
                    drug_f.size(1),
                )
            if self.flex_attention_bias_side in {'both', 'protein'}:
                # Protein-side message: M_p_d = CrossAttn(Q=H_p, K=H_d, V=H_d)
                protein_to_drug_bias = self.flex_attention_bias.protein_to_drug(
                    drug_flex_for_bias,
                    protein_flex,
                    protein_flex_global,
                    drug_f.size(1),
                )
            induced_gate_out.update({
                'flex_attention_bias_drug_to_protein': drug_to_protein_bias,
                'flex_attention_bias_protein_to_drug': protein_to_drug_bias,
                'flex_attention_bias_scale': self.flex_attention_bias.bias_scale,
            })
            if not self._printed_flex_attention_bias_debug:
                print('FlexAttentionBias debug shapes:', {
                    'side': self.flex_attention_bias_side,
                    'drug_to_protein_bias': None if drug_to_protein_bias is None else tuple(drug_to_protein_bias.shape),
                    'protein_to_drug_bias': None if protein_to_drug_bias is None else tuple(protein_to_drug_bias.shape),
                    'bias_scale': float(self.flex_attention_bias.bias_scale.detach().cpu()),
                })
                self._printed_flex_attention_bias_debug = True

        # 3. 双向 Cross-Attention
        needs_relation_outputs = self.alignment_target != 'none'
        needs_cross_updates = needs_relation_outputs or return_intermediates
        if needs_cross_updates:
            fusion_drug, attn_map, drug_update = self.fusion(
                drug_f, pr_f, drug_attn_mask, pr_mask_extended,
                cross_attn_bias=drug_to_protein_bias,
                return_cross_update=True,
            )
            fusion_protein, attn_map_protein, protein_update = self.fusion_protein(
                pr_f, drug_f, pr_mask_extended, drug_attn_mask,
                cross_attn_bias=protein_to_drug_bias,
                return_cross_update=True,
            )
        else:
            fusion_drug, attn_map = self.fusion(
                drug_f, pr_f, drug_attn_mask, pr_mask_extended,
                cross_attn_bias=drug_to_protein_bias,
            )
            fusion_protein, attn_map_protein = self.fusion_protein(
                pr_f, drug_f, pr_mask_extended, drug_attn_mask,
                cross_attn_bias=protein_to_drug_bias,
            )
            drug_update = None
            protein_update = None

        relation_drug_pre = drug_f
        relation_protein_pre = pr_f
        relation_drug_post = fusion_drug
        relation_protein_post = fusion_protein

        if self.use_flex_calibration:
            if protein_flex is None:
                raise ValueError(
                    'use_flex_calibration=True but protein_flex is None. Provide '
                    'batch_protein_flex with shape [B, L_p, 1].'
                )
            if self.use_drug_flex_condition and drug_flex is None:
                raise ValueError(
                    'use_flex_calibration=True and use_drug_flex_condition=True but '
                    'drug_flex is None. Provide batch_drug_flex with shape [B, drug_flex_dim], '
                    'or pass --no-drug-flex-condition.'
                )
            calibration_protein_flex = self._prepare_calibration_protein_flex(
                protein_flex,
                pr_mask_extended,
            )
            fusion_protein, flex_calibration_out = self.flex_calibration(
                h_p_pre=pr_f,
                h_p_ca=fusion_protein,
                protein_flex=calibration_protein_flex,
                protein_mask=pr_mask_extended,
                drug_flex=drug_flex,
                return_stats=self.return_flex_stats,
            )
            if self.return_flex_stats:
                induced_gate_out.update({
                    f'flex_calibration_{key}': value
                    for key, value in flex_calibration_out.items()
                })
            if not self._printed_flex_calibration_debug:
                print('PairSpecificFlexibilityCalibration debug:', {
                    'H_p_pre': tuple(pr_f.shape),
                    'H_p_ca': tuple(fusion_protein.shape),
                    'F_p': tuple(calibration_protein_flex.shape),
                    'F_d': None if drug_flex is None else tuple(drug_flex.shape),
                    'protein_mask': tuple(pr_mask_extended.shape),
                    'return_flex_stats': self.return_flex_stats,
                    'protein_flex_perturbation': self.protein_flex_perturbation,
                })
                self._printed_flex_calibration_debug = True

        if self.use_flex_film:
            film_needs_protein_flex = (
                (self.film_side in {'protein', 'both'} and self.protein_film_use_protein_flex)
                or (self.film_side in {'drug', 'both'} and self.drug_film_use_protein_flex)
            )
            film_needs_drug_flex = (
                (self.film_side in {'protein', 'both'} and self.protein_film_use_drug_flex)
                or (self.film_side in {'drug', 'both'} and self.drug_film_use_drug_flex)
            )
            if film_needs_protein_flex and protein_flex is None:
                raise ValueError(
                    'An enabled FiLM side requires protein flexibility but protein_flex is None. '
                    'Provide batch_protein_flex with shape [B, L_p, 1].'
                )
            if film_needs_drug_flex and drug_flex is None:
                raise ValueError(
                    'An enabled FiLM side requires drug flexibility but drug_flex is None. '
                    'Provide batch_drug_flex with shape [B, drug_flex_dim].'
                )
            film_protein_flex = None
            film_flex_mask = (
                protein_residue_mask.bool()
                if protein_residue_mask is not None
                else self._real_residue_mask(pr_mask_extended)
            )
            if film_needs_protein_flex:
                film_protein_flex = self._prepare_film_protein_flex(
                    protein_flex,
                    film_flex_mask,
                    protein_ids=protein_ids,
                )
            if self.film_side in {'protein', 'both'}:
                fusion_protein, flex_film_out = self.flex_film(
                    protein_pre=pr_f,
                    protein_ca=fusion_protein,
                    protein_flex=film_protein_flex,
                    protein_mask=pr_mask_extended,
                    drug_flex=drug_flex,
                    return_stats=self.return_film_stats,
                )
                if self.return_film_stats:
                    induced_gate_out.update({
                        f'flex_film_protein_{key}': value
                        for key, value in flex_film_out.items()
                    })
                    # Backward-compatible aliases for older analysis code.
                    induced_gate_out.update({
                        f'flex_film_{key}': value
                        for key, value in flex_film_out.items()
                    })
            if self.film_side in {'drug', 'both'}:
                fusion_drug, drug_flex_film_out = self.drug_flex_film(
                    drug_pre=drug_f,
                    drug_ca=fusion_drug,
                    drug_flex=drug_flex,
                    protein_flex_global=protein_flex_global,
                    drug_mask=drug_attn_mask,
                    return_stats=self.return_film_stats,
                )
                if self.return_film_stats:
                    induced_gate_out.update({
                        f'flex_film_drug_{key}': value
                        for key, value in drug_flex_film_out.items()
                    })
            if not self._printed_flex_film_debug:
                print('FlexibilityConditionedFiLM debug:', {
                    'film_side': self.film_side,
                    'drug_pre': tuple(drug_f.shape),
                    'drug_ca': tuple(fusion_drug.shape),
                    'protein_pre': tuple(pr_f.shape),
                    'protein_ca': tuple(fusion_protein.shape),
                    'protein_flex': None if film_protein_flex is None else tuple(film_protein_flex.shape),
                    'protein_flex_global': None if protein_flex_global is None else tuple(protein_flex_global.shape),
                    'drug_flex': None if drug_flex is None else tuple(drug_flex.shape),
                    'drug_mask': tuple(drug_attn_mask.shape),
                    'protein_mask': tuple(pr_mask_extended.shape),
                    'film_mode': self.film_mode,
                    'film_flex_encoding': self.film_flex_encoding,
                    'film_flex_embed_dim': self.film_flex_embed_dim,
                    'protein_film_use_protein_flex': self.protein_film_use_protein_flex,
                    'protein_film_use_drug_flex': self.protein_film_use_drug_flex,
                    'drug_film_use_drug_flex': self.drug_film_use_drug_flex,
                    'drug_film_use_protein_flex': self.drug_film_use_protein_flex,
                    'return_film_stats': self.return_film_stats,
                    'film_protein_flex_perturbation': self.film_protein_flex_perturbation,
                })
                self._printed_flex_film_debug = True

        if self.use_protein_induced_gate:
            # Stage-3 protein gate: H_p=pr_f, M_p_d=fusion_protein, F_p=protein_flex.
            fusion_protein, protein_gate_out = self.protein_induced_gate(
                pr_f,
                fusion_protein,
                protein_flex,
                pr_mask_extended,
            )
            induced_gate_out.update(protein_gate_out)
            if not self._printed_protein_induced_gate_debug:
                print(f'{self.protein_induced_gate.__class__.__name__} debug shapes:', protein_gate_out['protein_induced_debug_shapes'])
                self._printed_protein_induced_gate_debug = True

        if self.use_drug_induced_gate:
            if drug_flex is None:
                raise ValueError(
                    'use_drug_induced_gate=True but drug_flex is None. Provide '
                    'batch_drug_flex with shape [B, drug_flex_dim].'
                )
            # Stage-4 drug gate: H_d=drug_f, M_d_p=fusion_drug, F_d=drug_flex, F_p_global=pooled F_p.
            fusion_drug, drug_gate_out = self.drug_induced_gate(
                drug_f,
                fusion_drug,
                drug_flex,
                protein_flex_global,
                drug_attn_mask,
            )
            induced_gate_out.update(drug_gate_out)
            if not self._printed_drug_induced_gate_debug:
                print(f'{self.drug_induced_gate.__class__.__name__} debug shapes:', drug_gate_out['drug_induced_debug_shapes'])
                self._printed_drug_induced_gate_debug = True

        fusion_f = torch.cat([fusion_drug, fusion_protein], dim=1)


        drug_mlm_logits = None
        if self.use_mlm and masked_drugs is not None:
            drug_f_mlm = self.encode_drug(masked_drugs, self.precompute_freqs_cis)
            drug_mlm_logits = self.mlm_head(drug_f_mlm).permute(0, 2, 1)

        # ------------------- 5. Classification -------------------
        pooled_drug = self.masked_mean(fusion_drug, drug_attn_mask)
        predictor_protein_attention = None
        if self.predictor_pooling == 'protein_attention':
            pooled_protein, predictor_protein_attention = self.masked_protein_attention_pool(
                fusion_protein, pr_mask_extended
            )
        else:
            pooled_protein = self.masked_mean(fusion_protein, pr_mask_extended)
        if self.predictor_head == 'interaction_mlp':
            predictor_features = [pooled_drug, pooled_protein, pooled_drug * pooled_protein, torch.abs(pooled_drug - pooled_protein)]
        else:
            predictor_features = [pooled_drug, pooled_protein]
        if self.predictor_include_induced_delta_pools:
            if self.use_drug_induced_gate:
                predictor_features.append(self.masked_mean(induced_gate_out['drug_induced_delta'], drug_attn_mask))
            if self.use_protein_induced_gate:
                predictor_features.append(self.masked_mean(induced_gate_out['protein_induced_delta'], pr_mask_extended))
        fusion_final = torch.cat(predictor_features, dim=-1)
        if self.use_drug_flex:
            if drug_flex is None:
                raise ValueError(
                    'use_drug_flex=True but drug_flex is None. Provide '
                    'batch_drug_flex with shape [B, drug_flex_dim].'
                )
            if drug_flex.dim() == 1:
                drug_flex = drug_flex.unsqueeze(-1)
            if drug_flex.shape != (fusion_final.size(0), self.drug_flex_dim):
                raise ValueError(
                    'drug_flex must have shape [B, drug_flex_dim], got '
                    f'{tuple(drug_flex.shape)} expected '
                    f'({fusion_final.size(0)}, {self.drug_flex_dim}).'
                )
            # Z = concat(original_predictor_features, F_d)
            # original_predictor_features: [B, 2*d_model], F_d: [B, drug_flex_dim]
            fusion_final = torch.cat([
                fusion_final,
                drug_flex.to(device=fusion_final.device, dtype=fusion_final.dtype),
            ], dim=-1)
        if self.use_protein_flex:
            if protein_flex_global is None:
                raise ValueError('use_protein_flex=True but protein_flex_global was not computed.')
            # Z = concat(original_predictor_features, F_d, F_p_global)
            # F_p_global: [B, 3] = [mean_flex, max_flex, std_flex]
            fusion_final = torch.cat([
                fusion_final,
                protein_flex_global.to(device=fusion_final.device, dtype=fusion_final.dtype),
            ], dim=-1)
        logits = self.classifier(fusion_final)

        output = {
            'fusion_f': fusion_f,
            'attn_map': attn_map,
            'attn_map_protein': attn_map_protein,
            'drug_mlm_logits': drug_mlm_logits,
            'drug_cond': fusion_drug,
            'protein_cond': fusion_protein
        }
        if self.task_mode == 'regression':
            output['prediction'] = logits.squeeze(-1)
        else:
            output['logits'] = logits

        if predictor_protein_attention is not None:
            output['predictor_protein_attention'] = predictor_protein_attention
        if needs_relation_outputs:
            if self.alignment_target == 'response':
                drug_source = self.masked_mean(drug_update, drug_attn_mask)
                protein_source = self.masked_mean(protein_update, pr_mask_extended)
            elif self.alignment_target == 'pre':
                drug_source = self.masked_mean(relation_drug_pre, drug_attn_mask)
                protein_source = self.masked_mean(relation_protein_pre, pr_mask_extended)
            elif self.alignment_target == 'post':
                drug_source = self.masked_mean(relation_drug_post, drug_attn_mask)
                protein_source = self.masked_mean(relation_protein_post, pr_mask_extended)
            else:
                raise ValueError(f"Unsupported alignment_target={self.alignment_target}.")
            output.update({
                'drug_update': drug_update,
                'protein_update': protein_update,
                'drug_response': self._project_relation_response(drug_source, self.drug_response_head),
                'protein_response': self._project_relation_response(protein_source, self.protein_response_head),
                'drug_teacher_embedding': self._relation_drug_teacher(input_drugs, drug_attn_mask),
                'protein_teacher_embedding': self._relation_protein_teacher(input_proteins, pr_mask_extended),
                'alignment_target': self.alignment_target,
                'relation_alignment_side': self.relation_alignment_side,
            })

        output.update(induced_gate_out)
        if return_intermediates:
            returned_residue_mask = (
                pr_mask_extended.bool() if protein_residue_mask is None
                else protein_residue_mask
            )
            output.update({
                'protein_before': protein_before,
                'protein_after': fusion_protein,
                'protein_residue_mask': returned_residue_mask,
                'protein_cross_update': protein_update,
            })
            if self.protein_semantic_projection is not None:
                output['protein_semantic_change'] = self.protein_semantic_projection(
                    fusion_protein - protein_before
                )
            if self.protein_response_distillation_projection is not None:
                output['protein_gearnet_response_change'] = (
                    self.protein_response_distillation_projection(
                        fusion_protein - protein_before
                    )
                )
        return output
