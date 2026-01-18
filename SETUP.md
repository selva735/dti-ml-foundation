# Setup Script for DTI-ML-Foundation

This script helps you set up the DTI-ML-Foundation framework.

## Installation Steps

### 1. Create a Python Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install PyTorch Geometric

```bash
# For CUDA 11.8
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# For CPU only
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### 4. Install RDKit

```bash
# Via conda (recommended)
conda install -c conda-forge rdkit

# Or via pip
pip install rdkit
```

### 5. Optional: Install Protein Language Models

```bash
# For ESM-2
pip install fair-esm

# For ProtBERT/ProtT5
pip install transformers sentencepiece
```

### 6. Optional: Install TDA Libraries

```bash
pip install gudhi ripser persim
```

### 7. Verify Installation

```bash
python -c "import torch; import torch_geometric; import rdkit; print('Installation successful!')"
```

## Download Datasets

### Davis Dataset

1. Download from: http://staff.cs.utu.fi/~aatapa/data/DrugTarget/
2. Extract to `data/Davis/`
3. Ensure files: `affinities.csv`, `compounds.csv`, `proteins.csv`

### KIBA Dataset

1. Download from: https://github.com/thinng/GraphDTA
2. Extract to `data/KIBA/`
3. Ensure files: `affinities.csv`, `compounds.csv`, `proteins.csv`

### BindingDB Dataset

1. Download from: https://www.bindingdb.org/
2. Preprocess and place in `data/BindingDB/`

## Directory Structure

After setup, your directory should look like:

```
dti-ml-foundation/
├── models/
│   ├── __init__.py
│   ├── drug_gnn.py
│   ├── protein_plm.py
│   ├── tda_encoder.py
│   ├── fusion_attention.py
│   └── evidential_head.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── Davis/
│   │   ├── README.md
│   │   └── [dataset files]
│   ├── KIBA/
│   │   ├── README.md
│   │   └── [dataset files]
│   └── BindingDB/
│       ├── README.md
│       └── [dataset files]
├── config/
│   ├── default_config.yaml
│   ├── davis_config.yaml
│   └── kiba_config.yaml
├── notebooks/
│   ├── 01_getting_started.ipynb
│   └── 02_cold_start_analysis.ipynb
├── train.py
├── eval.py
├── requirements.txt
└── README.md
```

## Quick Test

```bash
# Test training (dry run)
python train.py --help

# Test evaluation
python eval.py --help
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related errors:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU if needed
python train.py --device cpu
```

### RDKit Import Error

If RDKit fails to import:

```bash
# Try installing via conda
conda install -c conda-forge rdkit

# Or use mamba (faster)
mamba install -c conda-forge rdkit
```

### PyTorch Geometric Issues

```bash
# Reinstall PyG components
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## Next Steps

1. Explore the notebooks in `notebooks/`
2. Read the README.md for detailed usage
3. Try training a model on Davis dataset:
   ```bash
   python train.py --dataset davis --split_type random
   ```

For more help, see the [README.md](README.md) or open an issue on GitHub.
