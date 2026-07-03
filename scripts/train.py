#!/usr/bin/env python3
"""Train a CMAT-DTI model.

Usage
-----
  # Train with default config
  python scripts/train.py --config configs/default.yaml

  # Train with cold-start split
  python scripts/train.py --config configs/cold_start.yaml

  # Override config values from the command line
  python scripts/train.py --config configs/default.yaml \\
      --train-path data/davis_train.csv \\
      --n-epochs 150 --batch-size 64 --lr 2e-4
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from src.data.dataset import DTIDataset, cold_start_split, create_dataloader
from src.models.cmat_dti import CMATDTI, ModelConfig
from src.training.trainer import Trainer
from src.utils.config import load_config, merge_configs, save_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CMAT-DTI")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--train-path", help="Override training data path")
    p.add_argument("--val-path", help="Override validation data path")
    p.add_argument("--n-epochs", type=int, help="Override number of training epochs")
    p.add_argument("--batch-size", type=int, help="Override batch size")
    p.add_argument("--lr", type=float, help="Override learning rate")
    p.add_argument("--split-mode", choices=["random", "cold_drug", "cold_target", "cold_both"],
                   help="Override dataset split mode (auto-splits a single CSV)")
    p.add_argument("--data-path", help="Single CSV path (will be split automatically)")
    p.add_argument("--checkpoint-dir", help="Override checkpoint directory")
    p.add_argument("--device", default=None, help="Device string (cpu/cuda/mps)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load base config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.train_path:
        cfg["data"]["train_path"] = args.train_path
    if args.val_path:
        cfg["data"]["val_path"] = args.val_path
    if args.n_epochs:
        cfg["training"]["n_epochs"] = args.n_epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.split_mode:
        cfg["data"]["split_mode"] = args.split_mode
    if args.checkpoint_dir:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir

    torch.manual_seed(args.seed)

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg_dict = cfg["model"]

    # ── Load or split dataset ─────────────────────────────────────────
    from src.data.dataset import DTIPair
    from src.data.featurizer import MolecularFeaturizer, ProteinFeaturizer

    mol_feat = MolecularFeaturizer(max_atoms=data_cfg["max_atoms"])
    prot_feat = ProteinFeaturizer(max_seq_len=data_cfg["max_seq_len"])

    if args.data_path:
        import pandas as pd
        df = pd.read_csv(args.data_path)
        pairs = [
            DTIPair(
                drug_id=str(row.get(data_cfg.get("drug_id_col", "drug_id"), i)),
                target_id=str(row.get(data_cfg.get("target_id_col", "target_id"), i)),
                smiles=row[data_cfg["smiles_col"]],
                sequence=row[data_cfg["sequence_col"]],
                label=float(row[data_cfg["label_col"]]),
            )
            for i, row in df.iterrows()
        ]
        train_pairs, val_pairs, test_pairs = cold_start_split(
            pairs,
            test_frac=data_cfg.get("test_frac", 0.1),
            val_frac=data_cfg.get("val_frac", 0.1),
            mode=data_cfg.get("split_mode", "random"),
            seed=args.seed,
        )
        train_ds = DTIDataset(train_pairs, mol_feat, prot_feat)
        val_ds = DTIDataset(val_pairs, mol_feat, prot_feat)
        test_ds = DTIDataset(test_pairs, mol_feat, prot_feat)
    else:
        train_ds = DTIDataset.from_csv(
            data_cfg["train_path"],
            smiles_col=data_cfg["smiles_col"],
            sequence_col=data_cfg["sequence_col"],
            label_col=data_cfg["label_col"],
            mol_featurizer=mol_feat,
            prot_featurizer=prot_feat,
        )
        val_ds = DTIDataset.from_csv(
            data_cfg["val_path"],
            smiles_col=data_cfg["smiles_col"],
            sequence_col=data_cfg["sequence_col"],
            label_col=data_cfg["label_col"],
            mol_featurizer=mol_feat,
            prot_featurizer=prot_feat,
        )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = create_dataloader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
    )

    # ── Build model ───────────────────────────────────────────────────
    # Use actual atom feature dimensionality from featurizer
    from src.data.featurizer import NUM_ATOM_FEATURES
    model_cfg_dict["atom_feat_dim"] = NUM_ATOM_FEATURES

    model_config = ModelConfig.from_dict(model_cfg_dict)
    model = CMATDTI(model_config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimiser & loss ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    task = train_cfg.get("task", "regression")
    if task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ── Train ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        task=task,
        scheduler_type=train_cfg.get("scheduler", "plateau"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        use_amp=train_cfg.get("use_amp", False),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
        log_dir=train_cfg.get("log_dir", None),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 30),
    )

    history = trainer.train(n_epochs=train_cfg["n_epochs"])
    print("Training complete.")

    # Save final config alongside checkpoint
    save_config(cfg, os.path.join(train_cfg.get("checkpoint_dir", "checkpoints"), "config.yaml"))
    print(f"Best val loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
