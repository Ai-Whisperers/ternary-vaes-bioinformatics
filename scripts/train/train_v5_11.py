"""V5.11 Training Script: Frozen Coverage + Hyperbolic Structure.

This script trains the V5.11 architecture:
1. Loads frozen v5.5 encoder (100% coverage preserved)
2. Trains HyperbolicProjection layer for radial hierarchy
3. Uses unified PAdicGeodesicLoss (hierarchy + correlation)
4. Optional: DifferentiableController for adaptive training

Usage:
    python scripts/train/train_v5_11.py
    python scripts/train/train_v5_11.py --config configs/ternary_v5_11.yaml
    python scripts/train/train_v5_11.py --epochs 100 --lr 1e-3

Key differences from v5.10:
- Encoder is FROZEN (no gradients, coverage preserved)
- Only projection layer and controller train
- Single unified geodesic loss (not separate ranking + radial)
- Controller outputs are tensors (gradients flow)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import TernaryVAEV5_11
from src.losses import PAdicGeodesicLoss, RadialHierarchyLoss, CombinedGeodesicLoss
from src.data.generation import generate_all_ternary_operations
from src.core import TERNARY


def parse_args():
    parser = argparse.ArgumentParser(description='Train V5.11 Ternary VAE')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--v5_5_checkpoint', type=str,
                        default='sandbox-training/checkpoints/v5_5/latest.pt',
                        help='Path to v5.5 checkpoint')
    parser.add_argument('--save_dir', type=str,
                        default='sandbox-training/checkpoints/v5_11',
                        help='Directory to save checkpoints')
    parser.add_argument('--use_controller', action='store_true', default=False,
                        help='Use differentiable controller')
    parser.add_argument('--curvature', type=float, default=1.0,
                        help='Hyperbolic curvature')
    parser.add_argument('--max_radius', type=float, default=0.95,
                        help='Maximum Poincare ball radius')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_full_dataset(device: str):
    """Create full dataset of all 19,683 ternary operations."""
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    indices = torch.arange(len(operations), device=device)
    return x, indices


def compute_metrics(model, x, indices, geodesic_loss_fn, radial_loss_fn, device):
    """Compute comprehensive metrics."""
    model.eval()

    with torch.no_grad():
        outputs = model(x, compute_control=False)
        z_A_hyp = outputs['z_A_hyp']
        z_B_hyp = outputs['z_B_hyp']

        # Geodesic loss
        geo_loss_A, geo_metrics_A = geodesic_loss_fn(z_A_hyp, indices)
        geo_loss_B, geo_metrics_B = geodesic_loss_fn(z_B_hyp, indices)

        # Radial loss
        rad_loss_A, rad_metrics_A = radial_loss_fn(z_A_hyp, indices)
        rad_loss_B, rad_metrics_B = radial_loss_fn(z_B_hyp, indices)

        # Radial distribution
        radii_A = torch.norm(z_A_hyp, dim=1).cpu().numpy()
        radii_B = torch.norm(z_B_hyp, dim=1).cpu().numpy()
        valuations = TERNARY.valuation(indices).cpu().numpy()

        # Radial hierarchy correlation (should be NEGATIVE)
        radial_corr_A = spearmanr(valuations, radii_A)[0]
        radial_corr_B = spearmanr(valuations, radii_B)[0]

        # Coverage check (using frozen decoder)
        logits_A = outputs['logits_A']
        preds = torch.argmax(logits_A, dim=-1) - 1
        targets = x.long()
        correct = (preds == targets).float().mean(dim=1)
        coverage = (correct == 1.0).sum().item() / len(x)

    return {
        'coverage': coverage,
        'geo_loss_A': geo_loss_A.item(),
        'geo_loss_B': geo_loss_B.item(),
        'rad_loss_A': rad_loss_A.item(),
        'rad_loss_B': rad_loss_B.item(),
        'radial_corr_A': radial_corr_A,
        'radial_corr_B': radial_corr_B,
        'mean_radius_A': radii_A.mean(),
        'mean_radius_B': radii_B.mean(),
        'distance_corr_A': geo_metrics_A.get('distance_correlation', 0),
        'distance_corr_B': geo_metrics_B.get('distance_correlation', 0)
    }


def train_epoch(model, optimizer, x, indices, geodesic_loss_fn, radial_loss_fn,
                batch_size, epoch, tau, device):
    """Train one epoch."""
    model.train()

    n_samples = len(x)
    n_batches = (n_samples + batch_size - 1) // batch_size
    perm = torch.randperm(n_samples, device=device)

    total_loss = 0.0
    total_geo = 0.0
    total_rad = 0.0

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_samples)
        batch_idx = perm[start:end]

        x_batch = x[batch_idx]
        idx_batch = indices[batch_idx]

        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_batch, compute_control=model.use_controller)

        z_A_hyp = outputs['z_A_hyp']
        z_B_hyp = outputs['z_B_hyp']

        # Compute losses
        geo_loss_A, _ = geodesic_loss_fn(z_A_hyp, idx_batch)
        geo_loss_B, _ = geodesic_loss_fn(z_B_hyp, idx_batch)
        rad_loss_A, _ = radial_loss_fn(z_A_hyp, idx_batch)
        rad_loss_B, _ = radial_loss_fn(z_B_hyp, idx_batch)

        geo_loss = geo_loss_A + geo_loss_B
        rad_loss = rad_loss_A + rad_loss_B

        # Get tau from controller if available, else use schedule
        if model.use_controller and 'control' in outputs:
            ctrl = outputs['control']
            tau_batch = ctrl['tau'].item() if isinstance(ctrl['tau'], torch.Tensor) else ctrl['tau']
        else:
            tau_batch = tau

        # Curriculum blend: (1-tau)*radial + tau*geodesic
        total_batch_loss = (1 - tau_batch) * rad_loss + tau_batch * geo_loss

        # Backward and optimize
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_geo += geo_loss.item()
        total_rad += rad_loss.item()

    return {
        'loss': total_loss / n_batches,
        'geo_loss': total_geo / n_batches,
        'rad_loss': total_rad / n_batches
    }


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override with command line args
        for key in ['epochs', 'lr', 'batch_size', 'curvature', 'max_radius']:
            if hasattr(args, key) and getattr(args, key) is not None:
                if key in config:
                    config[key] = getattr(args, key)
    else:
        config = vars(args)

    # Setup
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(config.get('save_dir', 'sandbox-training/checkpoints/v5_11'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('runs') / f'v5_11_{timestamp}'
    writer = SummaryWriter(log_dir=str(log_dir))

    # Create model
    print("\n=== Creating V5.11 Model ===")
    model = TernaryVAEV5_11(
        latent_dim=16,
        hidden_dim=config.get('hidden_dim', 64),
        max_radius=config.get('max_radius', 0.95),
        curvature=config.get('curvature', 1.0),
        use_controller=config.get('use_controller', False)
    )

    # Load v5.5 checkpoint
    v5_5_path = Path(config.get('v5_5_checkpoint', 'sandbox-training/checkpoints/v5_5/latest.pt'))
    if not v5_5_path.exists():
        print(f"ERROR: v5.5 checkpoint not found at {v5_5_path}")
        sys.exit(1)

    model.load_v5_5_checkpoint(v5_5_path, device)

    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Frozen: {param_counts['frozen']:,}")
    print(f"  Projection: {param_counts['projection']:,}")
    print(f"  Controller: {param_counts['controller']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Total: {param_counts['total']:,}")

    # Create dataset
    print("\n=== Loading Dataset ===")
    x, indices = create_full_dataset(device)
    print(f"Dataset size: {len(x)}")

    # Create loss functions
    geodesic_loss_fn = PAdicGeodesicLoss(
        curvature=config.get('curvature', 1.0),
        max_target_distance=config.get('max_target_distance', 3.0),
        n_pairs=config.get('n_pairs', 2000)
    ).to(device)

    radial_loss_fn = RadialHierarchyLoss(
        inner_radius=config.get('inner_radius', 0.1),
        outer_radius=config.get('outer_radius', 0.85)
    ).to(device)

    # Create optimizer (only trainable parameters)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )

    # Training loop
    print("\n=== Starting Training ===")
    n_epochs = config.get('epochs', 100)
    batch_size = config.get('batch_size', 512)

    best_radial_corr = float('inf')  # Want negative, so lower is better

    for epoch in range(n_epochs):
        # Curriculum: tau goes from 0 (radial focus) to 1 (geodesic focus)
        tau = min(1.0, epoch / (n_epochs * 0.7))

        # Train
        train_metrics = train_epoch(
            model, optimizer, x, indices,
            geodesic_loss_fn, radial_loss_fn,
            batch_size, epoch, tau, device
        )

        # Evaluate
        eval_metrics = compute_metrics(
            model, x, indices,
            geodesic_loss_fn, radial_loss_fn, device
        )

        # Update scheduler
        scheduler.step()

        # Log to TensorBoard
        writer.add_scalar('Train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('Train/geo_loss', train_metrics['geo_loss'], epoch)
        writer.add_scalar('Train/rad_loss', train_metrics['rad_loss'], epoch)
        writer.add_scalar('Train/tau', tau, epoch)
        writer.add_scalar('Eval/coverage', eval_metrics['coverage'], epoch)
        writer.add_scalar('Eval/radial_corr_A', eval_metrics['radial_corr_A'], epoch)
        writer.add_scalar('Eval/radial_corr_B', eval_metrics['radial_corr_B'], epoch)
        writer.add_scalar('Eval/distance_corr_A', eval_metrics['distance_corr_A'], epoch)
        writer.add_scalar('Eval/mean_radius_A', eval_metrics['mean_radius_A'], epoch)
        writer.add_scalar('Eval/mean_radius_B', eval_metrics['mean_radius_B'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Print progress
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"\nEpoch {epoch}/{n_epochs}")
            print(f"  Loss: {train_metrics['loss']:.4f} (geo: {train_metrics['geo_loss']:.4f}, rad: {train_metrics['rad_loss']:.4f})")
            print(f"  Coverage: {eval_metrics['coverage']*100:.1f}%")
            print(f"  Radial Hierarchy: A={eval_metrics['radial_corr_A']:.3f}, B={eval_metrics['radial_corr_B']:.3f}")
            print(f"  Distance Corr: A={eval_metrics['distance_corr_A']:.3f}, B={eval_metrics['distance_corr_B']:.3f}")
            print(f"  Mean Radius: A={eval_metrics['mean_radius_A']:.3f}, B={eval_metrics['mean_radius_B']:.3f}")
            print(f"  tau: {tau:.3f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model (best = most negative radial correlation)
        if eval_metrics['radial_corr_A'] < best_radial_corr:
            best_radial_corr = eval_metrics['radial_corr_A']
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'metrics': eval_metrics,
                'config': config
            }
            torch.save(checkpoint, save_dir / 'best.pt')
            print(f"  [NEW BEST] Radial hierarchy: {best_radial_corr:.4f}")

        # Periodic checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'metrics': eval_metrics,
                'config': config
            }, save_dir / f'epoch_{epoch}.pt')

    # Final checkpoint
    torch.save({
        'epoch': n_epochs - 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'metrics': eval_metrics,
        'config': config
    }, save_dir / 'latest.pt')

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal Metrics:")
    print(f"  Coverage: {eval_metrics['coverage']*100:.1f}%")
    print(f"  Radial Hierarchy: A={eval_metrics['radial_corr_A']:.3f}, B={eval_metrics['radial_corr_B']:.3f}")
    print(f"  Distance Correlation: A={eval_metrics['distance_corr_A']:.3f}")
    print(f"\nBest Radial Hierarchy: {best_radial_corr:.4f}")
    print(f"\nCheckpoints saved to: {save_dir}")
    print(f"TensorBoard logs: {log_dir}")

    writer.close()


if __name__ == '__main__':
    main()
