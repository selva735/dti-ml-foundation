# KIBA Dataset

The KIBA (Kinase Inhibitor BioActivity) dataset is a large-scale benchmark for drug-target interaction prediction.

## Dataset Information

- **Drugs**: 2,111 compounds
- **Proteins**: 229 kinases
- **Interactions**: 118,254 drug-protein pairs
- **Affinity Metric**: KIBA score (combination of Ki, Kd, and IC50 values)
- **Range**: 0 - 17 (lower values indicate stronger binding)

## Original Source

Tang, J., Szwajda, A., Shakyawar, S., Xu, T., Hintsanen, P., Wennerberg, K., & Aittokallio, T. (2014). Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis. *Journal of Chemical Information and Modeling*, 54(3), 735-743.

## Data Files

Place the following files in the `data/KIBA/` directory:

- `affinities.csv`: Drug-protein interaction KIBA scores
- `compounds.csv`: Drug SMILES strings
- `proteins.csv`: Protein sequences

## Usage

```python
from data import DTIDataset

# Load KIBA dataset
dataset = DTIDataset(
    root='./data',
    dataset_name='kiba',
    split='train'
)
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{tang2014making,
  title={Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis},
  author={Tang, Jing and Szwajda, Agnieszka and Shakyawar, Sushil and Xu, Tao and Hintsanen, Petteri and Wennerberg, Krister and Aittokallio, Tero},
  journal={Journal of chemical information and modeling},
  volume={54},
  number={3},
  pages={735--743},
  year={2014},
  publisher={ACS Publications}
}
```
