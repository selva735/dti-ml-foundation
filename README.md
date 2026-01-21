# DTI-ML-Foundation

**Multi-modal attention-based GNN framework for drug-target interaction prediction using molecular graphs, protein PLM embeddings, and topological features.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Overview

DTI-ML-Foundation is a comprehensive framework for predicting drug-target interactions (DTI) that combines:

- **Graph Neural Networks (GNN)** for molecular graph encoding (GAT/GIN)
- **Protein Language Models (PLM)** for protein sequence embeddings (ESM-2)
- **Multi-modal Attention** mechanisms for drug-protein interaction modeling
- **Uncertainty Estimation** via Monte Carlo Dropout and ensemble methods
- **Cold-Start Evaluation** for unseen drugs/targets scenarios

This framework enables drug discovery applications including virtual screening, drug repurposing, off-target prediction, and lead optimization.

## 📋 Features

- ✅ State-of-the-art GNN architectures (GAT, GIN)
- ✅ Pre-trained protein language model integration
- ✅ Cross-modal and self-attention mechanisms
- ✅ Comprehensive evaluation metrics
- ✅ Cold-start scenario handling
- ✅ Uncertainty quantification
- ✅ Distributed training support
- ✅ TensorBoard integration
- ✅ Extensive visualization tools
- ✅ Command-line interface for training/evaluation/prediction
- ✅ Modular and extensible design

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)

### Install via pip

```bash
# Clone the repository
git clone https://github.com/selva735/dti-ml-foundation.git
cd dti-ml-foundation

# Install the package
pip install -e .
```

### Install dependencies manually

```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Download Sample Data

```bash
python scripts/download_data.py --dataset sample --n-samples 1000
```

### 2. Train a Model

```bash
python scripts/train.py \
    --data-path data/sample_dataset.csv \
    --config configs/default.yaml \
    --n-epochs 50 \
    --device cuda
```

### 3. Evaluate the Model

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data-path data/sample_dataset.csv \
    --output-dir eval_results
```

### 4. Make Predictions

```bash
# Single prediction
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --drug-smiles "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \
    --protein-sequence "MSKGEELFTGVVPILVELDGDV..."

# Batch prediction
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --input predictions_input.csv \
    --output predictions_output.csv
```

## 📖 Documentation

### Model Architecture

The DTI prediction model consists of:

1. **Drug Encoder**: Graph neural network (GAT or GIN) for molecular graphs
2. **Protein Encoder**: Projection layers for PLM embeddings (ESM-2)
3. **Cross-Modal Attention**: Bidirectional attention between drug and protein
4. **Prediction Head**: MLP for affinity prediction

### Data Format

Input CSV file should contain:

| Column    | Description                | Example                    |
|-----------|----------------------------|----------------------------|
| smiles    | Drug SMILES string         | CC(C)CC1=CC=C(C=C1)C(C)... |
| sequence  | Protein amino acid sequence| MSKGEELFTGVVPILVELDGDV...  |
| affinity  | Binding affinity value     | 7.52                       |

### Configuration

Edit `configs/default.yaml` to customize model and training parameters.

### Cold-Start Evaluation

Evaluate on unseen drugs or targets using `configs/cold_start.yaml`.

## 🧪 Testing

Run tests:

```bash
pytest tests/
```

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{dti_ml_foundation,
  title={DTI-ML-Foundation: Multi-modal GNN for Drug-Target Interaction Prediction},
  author={DTI-ML-Foundation Contributors},
  year={2024},
  url={https://github.com/selva735/dti-ml-foundation}
}
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds upon research from DeepDTA, GraphDTA, and ESM-2. Referenced papers in the repository provide theoretical foundations.

---

**Status**: Active Development | **Version**: 0.1.0
