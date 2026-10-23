import torch
import torch.nn as nn


class InducedFineLevelAdapter(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, gate_type="vector"):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )
        if gate_type not in {"vector", "scalar"}:
            raise ValueError("gate_type must be either 'vector' or 'scalar'.")

        self.hidden_dim = hidden_dim
        self.gate_type = gate_type
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        gate_dim = hidden_dim if gate_type == "vector" else 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, gate_dim),
        )

    def forward(self, protein_levels, drug_levels):
        if protein_levels.dim() != 3 or drug_levels.dim() != 3:
            raise ValueError("protein_levels and drug_levels must have shape [B, 3, H].")
        if protein_levels.size(1) != 3 or drug_levels.size(1) != 3:
            raise ValueError("InducedFineLevelAdapter currently requires exactly 3 RQ levels.")
        if protein_levels.shape != drug_levels.shape:
            raise ValueError(
                "protein_levels and drug_levels must have matching [B, 3, H] shapes."
            )
        if protein_levels.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {protein_levels.size(-1)}."
            )

        # Coarse-to-fine residual quantization representations: [B, H].
        p1, p2, p3 = protein_levels.unbind(dim=1)
        d1, d2, d3 = drug_levels.unbind(dim=1)
        drug = d1 + d2 + d3  # [B, H]

        query = p3.unsqueeze(1)  # [B, 1, H]
        key_value = drug_levels  # [B, 3, H]
        attn_out, attn_weights = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
            average_attn_weights=True,
        )
        delta_p3 = attn_out.squeeze(1)  # [B, H]

        gate_input = torch.cat([p3, drug], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))  # [B, H] or [B, 1]
        p3_tilde = p3 + gate * delta_p3  # [B, H]
        adapted_protein = p1 + p2 + p3_tilde  # [B, H]
        shift = p3_tilde - p3

        return {
            "adapted_protein": adapted_protein,
            "drug": drug,
            "P3_tilde": p3_tilde,
            "Delta_P3": delta_p3,
            "gate_value": gate,
            "attn_weights": attn_weights,  # [B, 1, 3]
            "shift": shift,
            "shift_loss": shift.pow(2).mean(),
        }
