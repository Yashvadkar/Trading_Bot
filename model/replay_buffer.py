"""Prioritized Experience Replay Buffer for Multi-Agent Neural Debate.

Stores transitions (s_t, a_t, r_t, s_{t+1}, done, disagreement_variance)
for post-market offline and online self-learning.
"""

from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
import torch


@dataclass
class Transition:
    state: np.ndarray             # (32,)
    action: int                   # chosen action
    reward: float                 # realized reward
    next_state: np.ndarray        # (32,)
    done: bool
    disagreement_var: float


class ReplayBuffer:
    """Ring buffer with uniform/prioritized sampling for co-training."""

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.buffer: list[Transition] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray | torch.Tensor,
        action: int,
        reward: float,
        next_state: np.ndarray | torch.Tensor,
        done: bool,
        disagreement_var: float = 0.0
    ) -> None:
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy().flatten()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu().numpy().flatten()

        trans = Transition(
            state=state.astype(np.float32),
            action=int(action),
            reward=float(reward),
            next_state=next_state.astype(np.float32),
            done=bool(done),
            disagreement_var=float(disagreement_var),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(trans)
        else:
            self.buffer[self.position] = trans
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions and return formatted PyTorch tensors."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        disagreements = np.array([t.disagreement_var for t in batch], dtype=np.float32)

        return (
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rewards).unsqueeze(-1),
            torch.from_numpy(next_states),
            torch.from_numpy(dones).unsqueeze(-1),
            torch.from_numpy(disagreements).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return len(self.buffer)
