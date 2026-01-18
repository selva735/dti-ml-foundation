# Davis Dataset

The Davis dataset is a benchmark dataset for drug-target interaction prediction, containing kinase inhibitor binding data.

## Dataset Information

- **Drugs**: 68 kinase inhibitors
- **Proteins**: 442 kinases
- **Interactions**: 30,056 drug-protein pairs
- **Affinity Metric**: Kd (dissociation constant) in nM, converted to -log10(Kd/1e9)
- **Range**: 5.0 - 10.8 (pKd values)

## Original Source

Davis, M. I., Hunt, J. P., Herrgard, S., Ciceri, P., Wodicka, L. M., Pallares, G., ... & Zarrinkar, P. P. (2011). Comprehensive analysis of kinase inhibitor selectivity. *Nature Biotechnology*, 29(11), 1046-1051.

## Data Files

Place the following files in the `data/Davis/` directory:

- `affinities.csv`: Drug-protein interaction affinities
- `compounds.csv`: Drug SMILES strings and metadata
- `proteins.csv`: Protein sequences and metadata

## Usage

```python
from data import DTIDataset

# Load Davis dataset
dataset = DTIDataset(
    root='./data',
    dataset_name='davis',
    split='train'
)
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{davis2011comprehensive,
  title={Comprehensive analysis of kinase inhibitor selectivity},
  author={Davis, Mindy I and Hunt, Jeremy P and Herrgard, Sonja and Ciceri, Pietro and Wodicka, Lisa M and Pallares, Gabriel and Hocker, Michael and Treiber, Daniel K and Zarrinkar, Patrick P},
  journal={Nature biotechnology},
  volume={29},
  number={11},
  pages={1046--1051},
  year={2011},
  publisher={Nature Publishing Group}
}
```
