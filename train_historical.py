"""Offline Historical Pre-Training Script with Active Inference.

Ingests multi-year pure quantitative data, extracts 64-dimensional multi-branch
feature representations, and pre-trains the Bull, Bear, and Arbiter networks
under the Free Energy Principle.

Checkpoints persistent weights to model/weights/ensemble_pretrained.pt.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import time
import torch
import torch.optim as optim

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model.networks import BullNetwork, BearNetwork, ArbiterNetwork
from model.trainer import EnsembleTrainer
from data.historical_loader import HistoricalDataLoader


def train_historical(
    symbols: list[str],
    period: str = "2y",
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    output_path: str | Path | None = None
) -> Path:
    """Pre-trains the ensemble on real historical market episodes."""
    root_dir = Path(__file__).resolve().parent
    if output_path is None:
        weights_dir = root_dir / "model" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        output_file = weights_dir / "ensemble_pretrained.pt"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("[INIT] INSTITUTIONAL QUANTITATIVE PRE-TRAINING ENGINE (ACTIVE INFERENCE)")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {period} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 70)

    # 1. Ingest historical dataset & build replay buffer
    t0 = time.time()
    print("[1/3] Ingesting multi-year data & computing 64-dim quant theory tensors...")
    loader = HistoricalDataLoader()
    buffer = loader.build_historical_replay_buffer(
        symbols=symbols,
        period=period,
        max_samples_per_symbol=250
    )
    ingest_time = time.time() - t0
    print(f"[OK] Ingestion complete in {ingest_time:.2f}s! Replay buffer contains {len(buffer)} transitions.")

    if len(buffer) < batch_size:
        raise ValueError(f"Replay buffer size ({len(buffer)}) is smaller than batch size ({batch_size}).")

    # 2. Instantiate 64-dim Neural Ensemble
    print("[2/3] Initializing 64-dim Bull, Bear, and Arbiter Networks...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution device: {device}")

    bull_net = BullNetwork(state_dim=64, hidden_dim=128).to(device)
    bear_net = BearNetwork(state_dim=64, hidden_dim=128).to(device)
    arbiter_net = ArbiterNetwork(hidden_dim=128).to(device)

    trainer = EnsembleTrainer(
        bull_net=bull_net,
        bear_net=bear_net,
        arbiter_net=arbiter_net,
        lr=lr,
        fep_beta=0.05,
        device=device
    )

    # Cosine Annealing learning rate schedulers
    scheduler_bull = optim.lr_scheduler.CosineAnnealingLR(trainer.opt_bull, T_max=epochs, eta_min=1e-5)
    scheduler_bear = optim.lr_scheduler.CosineAnnealingLR(trainer.opt_bear, T_max=epochs, eta_min=1e-5)
    scheduler_arbiter = optim.lr_scheduler.CosineAnnealingLR(trainer.opt_arbiter, T_max=epochs, eta_min=1e-5)

    # 3. Training Loop across Epochs
    print("[3/3] Training ensemble across historical regimes...")
    steps_per_epoch = max(10, len(buffer) // batch_size)

    for epoch in range(1, epochs + 1):
        epoch_bull_loss = 0.0
        epoch_bear_loss = 0.0
        epoch_arb_loss = 0.0
        epoch_free_energy = 0.0

        for _ in range(steps_per_epoch):
            metrics = trainer.train_step(buffer, batch_size=batch_size)
            epoch_bull_loss += metrics.bull_loss
            epoch_bear_loss += metrics.bear_loss
            epoch_arb_loss += metrics.arbiter_loss
            epoch_free_energy += metrics.free_energy

        scheduler_bull.step()
        scheduler_bear.step()
        scheduler_arbiter.step()

        avg_bull = epoch_bull_loss / steps_per_epoch
        avg_bear = epoch_bear_loss / steps_per_epoch
        avg_arb = epoch_arb_loss / steps_per_epoch
        avg_fe = epoch_free_energy / steps_per_epoch

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Bull Loss: {avg_bull:+.4f} | "
                f"Bear Loss: {avg_bear:+.4f} | "
                f"Arbiter Loss: {avg_arb:+.4f} | "
                f"Free Energy (FEP): {avg_fe:+.4f}"
            )

    # 4. Save Checkpoint
    checkpoint = {
        "bull_state_dict": bull_net.state_dict(),
        "bear_state_dict": bear_net.state_dict(),
        "arbiter_state_dict": arbiter_net.state_dict(),
        "state_dim": 64,
        "hidden_dim": 128,
        "symbols": symbols,
        "period": period,
        "epochs": epochs,
        "timestamp": time.time(),
        "metadata": {
            "version": "2.0-quant-64dim",
            "theories": [
                "Random Matrix Theory (RMT)",
                "Path Signatures (Rough Path)",
                "Transfer Entropy (TE)",
                "Chaos Theory (Hurst)",
                "Extreme Value Theory (EVT)",
                "Network Theory (MST)",
                "Active Inference (FEP)",
                "Extreme Learning Machines (ELM)",
                "Topological Data Analysis (TDA)",
                "Critical Slowing Down"
            ]
        }
    }
    torch.save(checkpoint, output_file)
    print("=" * 70)
    print(f"[SAVED] Pre-trained weights successfully saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    print("=" * 70)
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train Trading Bot Ensemble on Historical Quant Data")
    parser.add_argument("--symbols", type=str, default="SPY,QQQ,AAPL,NVDA,MSFT,TSLA,AMD", help="Comma-separated tickers")
    parser.add_argument("--period", type=str, default="2y", help="Historical period (e.g. 1y, 2y, 3y)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path")

    args = parser.parse_args()
    sym_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    train_historical(
        symbols=sym_list,
        period=args.period,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output
    )
