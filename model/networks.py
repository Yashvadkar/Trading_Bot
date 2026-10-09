"""Multi-Neural-Network Debate Architecture (PyTorch).

Karpathy First-Principles Design:
- Zero wrapper bloat: direct nn.Module implementations.
- Explicit tensor shapes at every boundary.
- Compact footprint: <500K total parameters for instant CPU forward pass.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BullNetwork(nn.Module):
    """Optimistic Upside Seeker: Identifies momentum, breakout velocity, and catalyst upside."""

    def __init__(self, state_dim: int = 64, hidden_dim: int = 128, num_actions: int = 4):
        super().__init__()
        self.fc_in = nn.Linear(state_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        # Policy head: 0: Hold, 1: Buy 0.5x, 2: Buy 1.0x, 3: Exit
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        # Upside value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x: (batch, state_dim) or (batch, seq, state_dim).

        Returns: (action_probs, state_value, latent_embedding)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, state_dim)

        emb = F.silu(self.fc_in(x))
        out, h_next = self.gru(emb, h)
        latent = out[:, -1, :]  # (batch, hidden_dim)

        logits = self.policy_head(latent)
        action_probs = F.softmax(logits, dim=-1)
        val = self.value_head(latent)

        return action_probs, val, latent


class BearNetwork(nn.Module):
    """Adversarial Tail-Risk Seeker: Identifies traps, resistance, debt headwinds, and drawdown."""

    def __init__(self, state_dim: int = 64, hidden_dim: int = 128, num_actions: int = 4):
        super().__init__()
        self.fc_in = nn.Linear(state_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        # Policy head (tilted towards capital preservation / exit)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        # Downside risk estimate head
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        emb = F.silu(self.fc_in(x))
        out, h_next = self.gru(emb, h)
        latent = out[:, -1, :]

        logits = self.policy_head(latent)
        action_probs = F.softmax(logits, dim=-1)
        risk_val = self.risk_head(latent)

        return action_probs, risk_val, latent


class ArbiterNetwork(nn.Module):
    """The Moderator: Ingests Bull and Bear latent embeddings + disagreement variance to settle execution."""

    def __init__(self, hidden_dim: int = 128, num_actions: int = 4):
        super().__init__()
        # Ingests [h_bull, h_bear, disagreement_var] -> (2 * hidden_dim + 1)
        in_dim = 2 * hidden_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.SiLU()
        )
        self.calibrated_policy = nn.Linear(64, num_actions)
        self.calibrated_value = nn.Linear(64, 1)

    def forward(self, h_bull: torch.Tensor, h_bear: torch.Tensor, disagreement_var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Input: h_bull (batch, hidden_dim), h_bear (batch, hidden_dim), disagreement_var (batch, 1).

        Returns: (final_action_probs, final_value)
        """
        combined = torch.cat([h_bull, h_bear, disagreement_var], dim=-1)
        feat = self.mlp(combined)

        logits = self.calibrated_policy(feat)
        probs = F.softmax(logits, dim=-1)
        val = self.calibrated_value(feat)

        return probs, val
