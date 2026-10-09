"""Offline Pre-Trainer and Warm-Start Utility (Karpathy Heuristic Standard).

Generates synthetic or historical state-action trajectories to pre-train
the Bull, Bear, and Arbiter networks so the bot never deploys cold.
"""

from __future__ import annotations
import numpy as np
import torch
from model.networks import BullNetwork, BearNetwork, ArbiterNetwork
from model.replay_buffer import ReplayBuffer
from model.trainer import EnsembleTrainer


def generate_synthetic_experience(num_samples: int = 500, state_dim: int = 64) -> ReplayBuffer:
    """Generate physically plausible financial transitions for warm-start."""
    np.random.seed(42)
    buffer = ReplayBuffer(capacity=num_samples + 100)

    for _ in range(num_samples):
        # 64-dim state vector
        state = np.random.normal(0.0, 1.0, state_dim).astype(np.float32)

        # Action: 0: Hold, 1: Buy 0.5x, 2: Buy 1.0x, 3: Exit
        # Generate reward correlated with momentum features
        momentum = float(state[0] + state[1])
        action = 2 if momentum > 0.5 else (0 if abs(momentum) < 0.5 else 3)
        reward = float(momentum * 0.02 + np.random.normal(0, 0.01))

        next_state = state + np.random.normal(0, 0.2, state_dim).astype(np.float32)
        done = bool(np.random.rand() < 0.05)
        disagreement = float(np.random.uniform(0.05, 0.4))

        buffer.push(state, action, reward, next_state, done, disagreement)

    return buffer


def run_pretraining(
    bull_net: BullNetwork,
    bear_net: BearNetwork,
    arbiter_net: ArbiterNetwork,
    epochs: int = 25,
    batch_size: int = 32,
    state_dim: int = 64
) -> tuple[float, float, float]:
    """Pre-trains the ensemble on warm-start buffer and returns final losses."""
    buffer = generate_synthetic_experience(num_samples=1000, state_dim=state_dim)
    trainer = EnsembleTrainer(bull_net, bear_net, arbiter_net, lr=1e-3)

    last_bull = 0.0
    last_bear = 0.0
    last_arb = 0.0

    for _ in range(epochs):
        metrics = trainer.train_step(buffer, batch_size=batch_size)
        last_bull = metrics.bull_loss
        last_bear = metrics.bear_loss
        last_arb = metrics.arbiter_loss

    return last_bull, last_bear, last_arb
