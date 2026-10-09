"""Adversarial Debate Protocol between Bull and Bear Networks.

Computes epistemic disagreement variance (sigma^2_debate) and yields
the calibrated consensus trading decision.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from model.networks import BullNetwork, BearNetwork, ArbiterNetwork


@dataclass(frozen=True)
class DebateVerdict:
    chosen_action: int              # 0: Hold, 1: Buy 0.5x, 2: Buy 1.0x, 3: Exit
    action_confidence: float        # Probability of chosen action [0.0, 1.0]
    bull_probs: list[float]
    bear_probs: list[float]
    arbiter_probs: list[float]
    disagreement_variance: float    # Epistemic uncertainty
    is_consensus_approved: bool     # True if disagreement <= threshold
    disagreement_threshold: float


class DebateArena:
    """Orchestrates multi-neural-network debate rounds."""

    def __init__(
        self,
        bull_net: BullNetwork,
        bear_net: BearNetwork,
        arbiter_net: ArbiterNetwork,
        disagreement_threshold: float = 0.35,
        device: str = "cpu"
    ):
        self.bull_net = bull_net.to(device)
        self.bear_net = bear_net.to(device)
        self.arbiter_net = arbiter_net.to(device)
        self.disagreement_threshold = disagreement_threshold
        self.device = device

    def evaluate_debate(self, state: torch.Tensor) -> DebateVerdict:
        """Run one forward debate pass on state tensor of shape (1, 32)."""
        self.bull_net.eval()
        self.bear_net.eval()
        self.arbiter_net.eval()

        with torch.no_grad():
            s = state.to(self.device)

            # Round 1: Bull & Bear evaluate state
            bull_probs, bull_val, h_bull = self.bull_net(s)
            bear_probs, bear_val, h_bear = self.bear_net(s)

            # Round 2: Compute Epistemic Disagreement Variance
            # sigma^2 = 0.5 * sum((P_bull(a) - P_bear(a))^2)
            prob_diff = bull_probs - bear_probs
            disagreement_var = 0.5 * torch.sum(prob_diff ** 2, dim=-1, keepdim=True)
            var_scalar = float(disagreement_var.item())

            # Round 3: Arbiter synthesizes latent representations and variance
            arb_probs, arb_val = self.arbiter_net(h_bull, h_bear, disagreement_var)

            # High Epistemic Uncertainty Gate: If networks violently disagree, force HOLD (Action 0)
            is_approved = (var_scalar <= self.disagreement_threshold)
            final_probs = arb_probs.clone()

            if not is_approved:
                # Force capital preservation: set action 0 (HOLD) to highest probability
                final_probs[0, 0] = max(final_probs[0, 0].item(), 0.85)
                final_probs = F.softmax(final_probs, dim=-1)

            chosen_action = int(torch.argmax(final_probs, dim=-1).item())
            confidence = float(final_probs[0, chosen_action].item())

        return DebateVerdict(
            chosen_action=chosen_action,
            action_confidence=confidence,
            bull_probs=[float(p) for p in bull_probs[0].tolist()],
            bear_probs=[float(p) for p in bear_probs[0].tolist()],
            arbiter_probs=[float(p) for p in final_probs[0].tolist()],
            disagreement_variance=var_scalar,
            is_consensus_approved=is_approved,
            disagreement_threshold=self.disagreement_threshold
        )
