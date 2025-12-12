"""Training script for Ternary VAE v5.10 - Pure Hyperbolic Geometry.

This is the orchestration script that wires together components from src/.
All training logic is delegated to HyperbolicVAETrainer.

Usage:
    python scripts/train/train_ternary_v5_10.py --config configs/ternary_v5_10.yaml
"""

import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
import sys
from torch.utils.data import DataLoader, random_split

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_10 import DualNeuralVAEV5_10
from src.training import TernaryVAETrainer, HyperbolicVAETrainer, TrainingMonitor
from src.data import generate_all_ternary_operations, TernaryOperationDataset


def main():
    parser = argparse.ArgumentParser(description='Train Ternary VAE v5.10 - Pure Hyperbolic')
    parser.add_argument('--config', type=str, default='configs/ternary_v5_10.yaml')
    parser.add_argument('--log-dir', type=str, default='logs')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup monitor
    monitor = TrainingMonitor(
        eval_num_samples=config.get('eval_num_samples', 1000),
        tensorboard_dir=config.get('tensorboard_dir', 'runs'),
        log_dir=args.log_dir,
        log_to_file=True
    )

    _log_config_summary(monitor, config, args.config)

    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    monitor._log(f"\nDevice: {device}")

    # Create data loaders
    train_loader, val_loader = _create_data_loaders(config, seed, monitor)

    # Initialize model and trainers
    model = _create_model(config)
    base_trainer = TernaryVAETrainer(model, config, device)
    trainer = HyperbolicVAETrainer(base_trainer, model, device, config, monitor)

    # Training loop
    _run_training(trainer, base_trainer, model, train_loader, val_loader, config, monitor)

    monitor.close()


def _log_config_summary(monitor: TrainingMonitor, config: dict, config_path: str) -> None:
    """Log configuration summary."""
    monitor._log(f"{'='*80}")
    monitor._log("Ternary VAE v5.10 Training - PURE HYPERBOLIC GEOMETRY")
    monitor._log(f"{'='*80}")
    monitor._log(f"Config: {config_path}")

    padic = config.get('padic_losses', {})
    hyp_v10 = padic.get('hyperbolic_v10', {})

    monitor._log(f"\nv5.10 Modules:")
    monitor._log(f"  Hyperbolic Prior: {'ENABLED' if hyp_v10.get('use_hyperbolic_prior') else 'DISABLED'}")
    monitor._log(f"  Hyperbolic Recon: {'ENABLED' if hyp_v10.get('use_hyperbolic_recon') else 'DISABLED'}")
    monitor._log(f"  Centroid Loss: {'ENABLED' if hyp_v10.get('use_centroid_loss') else 'DISABLED'}")

    monitor._log(f"\nEvaluation Intervals:")
    monitor._log(f"  Coverage: every {config.get('coverage_check_interval', 5)} epochs")
    monitor._log(f"  Correlation: every {config.get('eval_interval', 20)} epochs")


def _create_data_loaders(config: dict, seed: int, monitor: TrainingMonitor):
    """Create train and validation data loaders."""
    monitor._log("\nGenerating dataset...")
    operations = generate_all_ternary_operations()
    dataset = TernaryOperationDataset(operations)
    monitor._log(f"Total operations: {len(dataset):,}")

    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, _ = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    monitor._log(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader


def _create_model(config: dict) -> DualNeuralVAEV5_10:
    """Create and return the model."""
    mc = config['model']
    return DualNeuralVAEV5_10(
        input_dim=mc['input_dim'], latent_dim=mc['latent_dim'],
        rho_min=mc['rho_min'], rho_max=mc['rho_max'],
        lambda3_base=mc['lambda3_base'], lambda3_amplitude=mc['lambda3_amplitude'],
        eps_kl=mc['eps_kl'], gradient_balance=mc.get('gradient_balance', True),
        adaptive_scheduling=mc.get('adaptive_scheduling', True),
        use_statenet=mc.get('use_statenet', True),
        statenet_lr_scale=mc.get('statenet_lr_scale', 0.1),
        statenet_lambda_scale=mc.get('statenet_lambda_scale', 0.02),
        statenet_ranking_scale=mc.get('statenet_ranking_scale', 0.3),
        statenet_hyp_sigma_scale=mc.get('statenet_hyp_sigma_scale', 0.05),
        statenet_hyp_curvature_scale=mc.get('statenet_hyp_curvature_scale', 0.02)
    )


def _run_training(trainer, base_trainer, model, train_loader, val_loader, config, monitor):
    """Execute the training loop."""
    checkpoint_dir = Path(config.get('checkpoint_dir', 'sandbox-training/checkpoints/v5_10'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['total_epochs']):
        base_trainer.epoch = epoch
        losses = trainer.train_epoch(train_loader, val_loader, epoch)

        base_trainer.monitor.check_best(losses['loss'])
        base_trainer.monitor.update_histories(
            losses['H_A'], losses['H_B'], losses['unique_A'], losses['unique_B']
        )

        # Build homeostatic metrics
        homeo = {k.replace('homeo_', ''): v for k, v in losses.items() if k.startswith('homeo_')}

        monitor.log_epoch_summary(
            epoch, config['total_epochs'], losses['loss'],
            losses['cov_A'], losses['cov_B'],
            losses['corr_A_hyp'], losses['corr_B_hyp'],
            losses['corr_A_euc'], losses['corr_B_euc'],
            losses['mean_radius_A'], losses['mean_radius_B'],
            losses['ranking_weight'],
            losses.get('coverage_evaluated', True),
            losses.get('correlation_evaluated', True),
            losses.get('hyp_kl_A', 0), losses.get('hyp_kl_B', 0),
            losses.get('centroid_loss', 0), losses.get('radial_loss', 0),
            homeo if homeo else None
        )

        monitor.log_hyperbolic_epoch(
            epoch, losses['corr_A_hyp'], losses['corr_B_hyp'],
            losses['corr_A_euc'], losses['corr_B_euc'],
            losses['mean_radius_A'], losses['mean_radius_B'],
            losses['ranking_weight'], losses.get('ranking_loss_hyp', 0),
            losses.get('radial_loss', 0), losses.get('hyp_kl_A', 0),
            losses.get('hyp_kl_B', 0), losses.get('centroid_loss', 0),
            homeo if homeo else None
        )

        if epoch % config.get('checkpoint_freq', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': base_trainer.optimizer.state_dict(),
                'best_corr_hyp': trainer.best_corr_hyp,
                'best_coverage': trainer.best_coverage,
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

    # Save final model
    monitor.print_training_summary()
    torch.save({
        'epoch': config['total_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': base_trainer.optimizer.state_dict(),
        'best_corr_hyp': trainer.best_corr_hyp,
        'best_corr_euc': trainer.best_corr_euc,
        'best_coverage': trainer.best_coverage,
        'correlation_history_hyp': trainer.correlation_history_hyp,
        'correlation_history_euc': trainer.correlation_history_euc,
        'coverage_history': trainer.coverage_history,
        'config': config
    }, checkpoint_dir / 'final_model.pt')
    monitor._log(f"\nFinal model saved to: {checkpoint_dir / 'final_model.pt'}")


if __name__ == '__main__':
    main()
