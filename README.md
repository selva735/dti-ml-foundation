# dti-ml-foundation

## CMAT-DTI: Cross-Modal Attention Transformer for Drug-Target Interaction

A novel multi-modal deep learning framework for drug-target interaction (DTI) prediction combining:

- **Molecular Graph Transformer** — transformer-style self-attention over molecular graphs with graph-distance positional bias, enabling long-range atom interaction modeling beyond shallow GNN message-passing.
- **Protein Sequence Transformer** — lightweight transformer encoder over amino-acid tokens with sinusoidal positional encoding; supports pre-computed PLM (ESM-2/ProtBERT) embeddings.
- **Bidirectional Cross-Modal Attention** — symmetric attention where drugs attend to protein context *and* proteins attend to drug context simultaneously, producing richer context-aware representations.
- **Monte Carlo Dropout Uncertainty** — calibrated prediction intervals from stochastic forward passes, enabling risk-aware drug discovery prioritisation.
- **Cold-Start Evaluation** — rigorous evaluation under four scenarios: random, cold-drug, cold-target, and cold-both (unseen drugs and targets).

---

## Architecture

```
CMAT-DTI
├── Drug Encoder: Molecular Graph Transformer (MGT)
│   ├── Atom feature embedding (78-dim → embed_dim)
│   ├── MGT layers: graph-biased self-attention + FFN (pre-norm)
│   │   └── GraphPositionalEncoding: shortest-path distance → attention bias
│   └── Global pooling: mean | max | attention-weighted
│
├── Protein Encoder: Sequence Transformer
│   ├── Amino-acid token embedding + sinusoidal PE
│   ├── Transformer encoder layers (pre-norm, GELU)
│   └── [CLS] token output
│
├── Bidirectional Cross-Modal Attention Fusion
│   ├── Drug-to-Protein cross-attention (stacked layers)
│   ├── Protein-to-Drug cross-attention (stacked layers)
│   ├── Per-modality FFN refinement
│   └── Concatenation + projection → interaction embedding
│
└── Prediction Head
    ├── MLP with LayerNorm + GELU + Dropout
    └── Output: binding affinity / interaction score
```

---

## Installation

```bash
pip install -e ".[molecular,dev]"
```

Or install dependencies manually:

```bash
pip install torch>=2.2.0 numpy scipy scikit-learn pandas pyyaml
pip install rdkit>=2023.3.1   # for molecular featurization
```

---

## Quick Start

### Train a model

```python
from src.data.dataset import DTIPair, DTIDataset, cold_start_split, create_dataloader
from src.data.featurizer import MolecularFeaturizer, ProteinFeaturizer
from src.models.cmat_dti import CMATDTI, ModelConfig
from src.training.trainer import Trainer
import torch, torch.nn as nn

# Build dataset
pairs = [DTIPair("drug0", "target0", "CC(=O)Nc1ccc(O)cc1", "MKTAYIAK...", label=7.2)]
train_ds = DTIDataset(pairs)
train_loader = create_dataloader(train_ds, batch_size=32)

# Build model
config = ModelConfig()
model = CMATDTI(config)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = Trainer(model, train_loader, train_loader, optimizer, nn.MSELoss())
history = trainer.train(n_epochs=100)
```

### Predict with uncertainty

```python
result = model.predict_with_uncertainty(
    atom_features=atom_feat,
    dist_matrix=dist_mat,
    protein_tokens=tokens,
    n_samples=20,
)
print(f"Predicted: {result['mean'].item():.2f} ± {result['std'].item():.2f}")
```

### CLI usage

```bash
# Train with default config
python scripts/train.py --config configs/default.yaml

# Train with cold-start split on a single CSV
python scripts/train.py --data-path my_data.csv --split-mode cold_drug

# Predict single pair
python scripts/predict.py \
    --checkpoint checkpoints/default/best_model.pt \
    --smiles "CC(=O)Nc1ccc(O)cc1" \
    --sequence "MKTAYIAKQRQISFVK..."

# Batch prediction with uncertainty
python scripts/predict.py \
    --checkpoint checkpoints/default/best_model.pt \
    --input pairs.csv --output predictions.csv --uncertainty
```

---

## Project Structure

```
dti-ml-foundation/
├── src/
│   ├── models/
│   │   ├── drug_encoder.py       # Molecular Graph Transformer
│   │   ├── protein_encoder.py    # Protein Sequence Transformer
│   │   ├── cross_attention.py    # Bidirectional Cross-Modal Attention
│   │   └── cmat_dti.py           # Main model + ModelConfig + factory
│   ├── data/
│   │   ├── featurizer.py         # Molecule (RDKit) + protein featurizers
│   │   └── dataset.py            # DTIDataset, cold_start_split, collate_fn
│   ├── training/
│   │   ├── metrics.py            # MSE/RMSE/MAE/Pearson/CI/ROC-AUC + EarlyStopping
│   │   └── trainer.py            # Full training loop with AMP, LR scheduler
│   └── utils/
│       └── config.py             # YAML load/save/merge
├── scripts/
│   ├── train.py                  # CLI training script
│   └── predict.py                # CLI inference script
├── tests/
│   ├── test_models.py            # Model unit tests
│   ├── test_data.py              # Featurizer & dataset unit tests
│   └── test_training.py          # Metrics, EarlyStopping, Trainer tests
├── configs/
│   ├── default.yaml              # Standard training configuration
│   └── cold_start.yaml           # Cold-start optimised configuration
├── requirements.txt
├── setup.py
└── README.md
```

---

## Supported Datasets

| Dataset   | Drugs  | Proteins | Pairs   | Label type     |
|-----------|--------|----------|---------|----------------|
| Davis     | 68     | 442      | 30,056  | Kd (nM)        |
| KIBA      | 2,111  | 229      | 118,254 | KIBA score     |
| BindingDB | varies | varies   | varies  | IC50/Kd/Ki     |
| Custom    | any    | any      | any     | CSV: user-defined |

---

## Evaluation Metrics

**Regression:** MSE, RMSE, MAE, Pearson *r*, Spearman *ρ*, R², Concordance Index (CI)

**Classification:** ROC-AUC, PR-AUC, Accuracy

**Cold-start modes:** `random`, `cold_drug`, `cold_target`, `cold_both`

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Reference Papers

The following papers informed the design of CMAT-DTI:

- **DeepDTA** – Deep learning for drug-target binding affinity prediction
- **GraphDTA** – Predicting drug-target binding affinity with graph neural networks
- **DeepMGT-DTI** – Multi-graph transformer for DTI prediction
- **WIDEDTA** – Wide learning approach for DTI
