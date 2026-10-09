"""Asymmetric Co-Trainer with Active Inference (Free Energy Principle).

Karpathy First-Principles Standard:
- Explicit gradient steps without blackbox RL frameworks.
- Active Inference / Free Energy Principle (FEP):
  Arbiter minimizes Variational Free Energy F = Complexity (KL to prior) - Accuracy (TD Return)
  Epistemic uncertainty dynamically scales the complexity penalty to prevent overtrading in chaotic regimes.
- Asymmetric objective functions:
  Bull rewarded on upside capture.
  Bear rewarded on downside risk mitigation.
  Arbiter settles execution by minimizing variational free energy.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.networks import BullNetwork, BearNetwork, ArbiterNetwork
from model.replay_buffer import ReplayBuffer


@dataclass
class TrainMetrics:
    bull_loss: float
    bear_loss: float
    arbiter_loss: float
    free_energy: float
    total_steps: int


class EnsembleTrainer:
    """Trains the 3-network debate ensemble with Active Inference and asymmetric objectives."""

    def __init__(
        self,
        bull_net: BullNetwork,
        bear_net: BearNetwork,
        arbiter_net: ArbiterNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        fep_beta: float = 0.05,       # Free energy complexity temperature
        device: str = "cpu"
    ):
        self.bull_net = bull_net.to(device)
        self.bear_net = bear_net.to(device)
        self.arbiter_net = arbiter_net.to(device)
        self.gamma = gamma
        self.fep_beta = fep_beta
        self.device = device

        self.opt_bull = optim.AdamW(self.bull_net.parameters(), lr=lr, weight_decay=1e-4)
        self.opt_bear = optim.AdamW(self.bear_net.parameters(), lr=lr, weight_decay=1e-4)
        self.opt_arbiter = optim.AdamW(self.arbiter_net.parameters(), lr=lr, weight_decay=1e-4)

        # Conservative prior over actions [Hold, Buy 0.5x, Buy 1.0x, Exit]
        # Tilt towards Hold (0.4) and Exit (0.2)
        self.action_prior = torch.tensor([0.40, 0.20, 0.20, 0.20], dtype=torch.float32, device=device).unsqueeze(0)

        self.steps = 0

    def train_step(self, buffer: ReplayBuffer, batch_size: int = 32) -> TrainMetrics:
        """Executes one co-training optimization step with Active Inference Free Energy minimization."""
        if len(buffer) < batch_size:
            return TrainMetrics(bull_loss=0.0, bear_loss=0.0, arbiter_loss=0.0, free_energy=0.0, total_steps=self.steps)

        states, actions, rewards, next_states, dones, disagreements = buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        disagreements = disagreements.to(self.device)

        self.bull_net.train()
        self.bear_net.train()
        self.arbiter_net.train()

        # ----------------------------------------------------
        # 1. Bull Network Update: Focuses on upside gains (> 0)
        # ----------------------------------------------------
        upside_rewards = torch.clamp(rewards, min=0.0)
        bull_probs, bull_val, h_bull = self.bull_net(states)
        with torch.no_grad():
            _, next_bull_val, _ = self.bull_net(next_states)
            bull_target = upside_rewards + self.gamma * (1.0 - dones) * next_bull_val

        bull_action_probs = bull_probs.gather(1, actions.unsqueeze(1))
        bull_policy_loss = -torch.log(bull_action_probs + 1e-8) * (bull_target - bull_val.detach())
        bull_val_loss = F.mse_loss(bull_val, bull_target)
        bull_loss = bull_policy_loss.mean() + 0.5 * bull_val_loss

        self.opt_bull.zero_grad()
        bull_loss.backward()
        nn.utils.clip_grad_norm_(self.bull_net.parameters(), 1.0)
        self.opt_bull.step()

        # ----------------------------------------------------
        # 2. Bear Network Update: Focuses on avoiding drawdowns (< 0)
        # ----------------------------------------------------
        downside_penalties = torch.clamp(-rewards, min=0.0)
        bear_probs, bear_risk, h_bear = self.bear_net(states)
        with torch.no_grad():
            _, next_bear_risk, _ = self.bear_net(next_states)
            bear_target = downside_penalties + self.gamma * (1.0 - dones) * next_bear_risk

        bear_action_probs = bear_probs.gather(1, actions.unsqueeze(1))
        bear_policy_loss = -torch.log(bear_action_probs + 1e-8) * (bear_target - bear_risk.detach())
        bear_val_loss = F.mse_loss(bear_risk, bear_target)
        bear_loss = bear_policy_loss.mean() + 0.5 * bear_val_loss

        self.opt_bear.zero_grad()
        bear_loss.backward()
        nn.utils.clip_grad_norm_(self.bear_net.parameters(), 1.0)
        self.opt_bear.step()

        # ----------------------------------------------------
        # 3. Arbiter Network: Active Inference & Free Energy Minimization
        # ----------------------------------------------------
        arbiter_probs, arbiter_val = self.arbiter_net(h_bull.detach(), h_bear.detach(), disagreements)
        with torch.no_grad():
            _, _, next_h_bull = self.bull_net(next_states)
            _, _, next_h_bear = self.bear_net(next_states)
            next_prob_diff = self.bull_net(next_states)[0] - self.bear_net(next_states)[0]
            next_dis = 0.5 * torch.sum(next_prob_diff ** 2, dim=-1, keepdim=True)
            _, next_arb_val = self.arbiter_net(next_h_bull, next_h_bear, next_dis)
            arbiter_target = rewards + self.gamma * (1.0 - dones) * next_arb_val

        # Accuracy term: TD Error
        arb_action_probs = arbiter_probs.gather(1, actions.unsqueeze(1))
        arb_td_loss = -torch.log(arb_action_probs + 1e-8) * (arbiter_target - arbiter_val.detach())
        arb_val_loss = F.mse_loss(arbiter_val, arbiter_target)

        # Complexity term: KL divergence from prior beliefs D_KL(q || p)
        # Scaled by epistemic disagreement variance (higher uncertainty = stronger pull to prior)
        kl_complexity = torch.sum(
            arbiter_probs * (torch.log(arbiter_probs + 1e-8) - torch.log(self.action_prior + 1e-8)),
            dim=-1,
            keepdim=True
        )
        scaled_complexity = (1.0 + 2.0 * disagreements) * kl_complexity

        # Variational Free Energy: F = Accuracy Loss + Beta * Complexity
        free_energy = arb_td_loss.mean() + self.fep_beta * scaled_complexity.mean()
        arbiter_loss = free_energy + 0.5 * arb_val_loss

        self.opt_arbiter.zero_grad()
        arbiter_loss.backward()
        nn.utils.clip_grad_norm_(self.arbiter_net.parameters(), 1.0)
        self.opt_arbiter.step()

        self.steps += 1
        return TrainMetrics(
            bull_loss=float(bull_loss.item()),
            bear_loss=float(bear_loss.item()),
            arbiter_loss=float(arbiter_loss.item()),
            free_energy=float(free_energy.item()),
            total_steps=self.steps
        )
