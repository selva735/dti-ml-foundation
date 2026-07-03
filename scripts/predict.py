#!/usr/bin/env python3
"""Run inference with a trained CMAT-DTI model.

Usage
-----
  # Single prediction
  python scripts/predict.py \\
      --checkpoint checkpoints/default/best_model.pt \\
      --smiles "CC(=O)Nc1ccc(O)cc1" \\
      --sequence "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPSVKELFTNLLKDFPSGPKLSPTAASSSSASSPTPQTAVHIVPGFGLASEFKDNMQPIWSGKLSTSGAHPLQDSSWFGHMTSDYNLLKIVRGERLPFQPVLRHIPDDPSFREVTPEVMHGGSVVDRFTTSVQHNIAQPGFMGALTCNNESVVQFLSASIFSIGEDATRRVDQQLKKSKAQQIIGSDSPQIFANVFDAALMGIAKAGFKTMPQQKTVLSTPNEGIPIEQSTTSGTLHQHWPKSVPTLSVPKGVSAGQLHAGSSAHDTIFLQMQSISYLFLEIQMKKSLAFQHEISHMKPTGAPSEAEDTSQLPWQHIAWDGDQNTAQVMQMEQAVEFPKVQELHHAFQQKAVQIDKPQMKLEEGDRPEPKVMQLLSQIHQLDQEVAEVQPAAQQQLALALASLGQDMDSVDEQDPLDPVTSSQKD"

  # Batch prediction from CSV with uncertainty
  python scripts/predict.py \\
      --checkpoint checkpoints/default/best_model.pt \\
      --input pairs.csv \\
      --output predictions.csv \\
      --uncertainty
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import pandas as pd

from src.data.featurizer import MolecularFeaturizer, ProteinFeaturizer, NUM_ATOM_FEATURES
from src.models.cmat_dti import CMATDTI, ModelConfig
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict DTI binding affinity with CMAT-DTI")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--config", help="Path to config YAML (optional; inferred from checkpoint dir)")
    p.add_argument("--smiles", help="Drug SMILES string for single prediction")
    p.add_argument("--sequence", help="Protein sequence for single prediction")
    p.add_argument("--input", help="Input CSV for batch prediction")
    p.add_argument("--output", default="predictions.csv", help="Output CSV path")
    p.add_argument("--smiles-col", default="smiles", help="SMILES column name in input CSV")
    p.add_argument("--sequence-col", default="sequence", help="Sequence column name in input CSV")
    p.add_argument("--uncertainty", action="store_true", help="Enable MC Dropout uncertainty")
    p.add_argument("--mc-samples", type=int, default=20, help="Number of MC Dropout samples")
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_model(checkpoint_path: str, config_path: str | None, device: str) -> CMATDTI:
    """Load a trained CMAT-DTI model from a checkpoint."""
    # Try to find config next to checkpoint if not specified
    if config_path is None:
        ckpt_dir = os.path.dirname(checkpoint_path)
        candidate = os.path.join(ckpt_dir, "config.yaml")
        config_path = candidate if os.path.exists(candidate) else None

    if config_path and os.path.exists(config_path):
        cfg = load_config(config_path)
        model_cfg_dict = cfg.get("model", {})
        model_cfg_dict["atom_feat_dim"] = NUM_ATOM_FEATURES
        model_config = ModelConfig.from_dict(model_cfg_dict)
    else:
        print("[predict] No config found – using default ModelConfig.")
        model_config = ModelConfig(atom_feat_dim=NUM_ATOM_FEATURES)

    model = CMATDTI(model_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(torch.device(device))
    return model


def predict_single(
    model: CMATDTI,
    smiles: str,
    sequence: str,
    device: str,
    uncertainty: bool = False,
    mc_samples: int = 20,
) -> dict:
    """Predict binding affinity for a single drug-target pair."""
    mol_feat = MolecularFeaturizer()
    prot_feat = ProteinFeaturizer()

    mol = mol_feat(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    prot = prot_feat(sequence)

    dev = torch.device(device)
    atom_features = mol["atom_features"].unsqueeze(0).to(dev)
    dist_matrix = mol["dist_matrix"].unsqueeze(0).to(dev)
    atom_mask = mol["atom_mask"].unsqueeze(0).to(dev)
    tokens = prot["tokens"].unsqueeze(0).to(dev)
    protein_padding_mask = prot["padding_mask"].unsqueeze(0).to(dev)

    if uncertainty:
        result = model.predict_with_uncertainty(
            atom_features=atom_features,
            dist_matrix=dist_matrix,
            protein_tokens=tokens,
            atom_mask=atom_mask,
            protein_padding_mask=protein_padding_mask,
            n_samples=mc_samples,
        )
        return {
            "prediction": result["mean"].item(),
            "uncertainty": result["std"].item(),
        }
    else:
        with torch.no_grad():
            pred = model(
                atom_features=atom_features,
                dist_matrix=dist_matrix,
                protein_tokens=tokens,
                atom_mask=atom_mask,
                protein_padding_mask=protein_padding_mask,
            )
        return {"prediction": pred.item()}


def main() -> None:
    args = parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = load_model(args.checkpoint, args.config, device)

    if args.smiles and args.sequence:
        result = predict_single(
            model, args.smiles, args.sequence, device,
            uncertainty=args.uncertainty, mc_samples=args.mc_samples
        )
        print(f"Prediction: {result['prediction']:.4f}")
        if "uncertainty" in result:
            print(f"Uncertainty (std): {result['uncertainty']:.4f}")
        return

    if args.input:
        df = pd.read_csv(args.input)
        mol_feat = MolecularFeaturizer()
        prot_feat = ProteinFeaturizer()

        predictions = []
        uncertainties = []
        for _, row in df.iterrows():
            try:
                res = predict_single(
                    model,
                    smiles=str(row[args.smiles_col]),
                    sequence=str(row[args.sequence_col]),
                    device=device,
                    uncertainty=args.uncertainty,
                    mc_samples=args.mc_samples,
                )
                predictions.append(res["prediction"])
                uncertainties.append(res.get("uncertainty", None))
            except Exception as e:
                print(f"Warning: skipping row – {e}")
                predictions.append(None)
                uncertainties.append(None)

        df["prediction"] = predictions
        if args.uncertainty:
            df["uncertainty"] = uncertainties
        df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
        return

    print("Provide either --smiles/--sequence for a single prediction or --input for batch.")


if __name__ == "__main__":
    main()
