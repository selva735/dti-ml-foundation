# BindingDB Dataset

BindingDB is a public database of measured binding affinities for drug-like molecules and proteins.

## Dataset Information

- **Drugs**: Variable (10,000+ compounds)
- **Proteins**: Variable (1,000+ targets)
- **Interactions**: Variable (100,000+ measurements)
- **Affinity Metrics**: Kd, Ki, IC50, EC50
- **Coverage**: Multiple protein families and drug types

## Original Source

Gilson, M. K., Liu, T., Baitaluk, M., Nicola, G., Hwang, L., & Chong, J. (2016). BindingDB in 2015: a public database for medicinal chemistry, computational chemistry and systems pharmacology. *Nucleic Acids Research*, 44(D1), D1045-D1053.

Website: https://www.bindingdb.org/

## Data Files

Download data from BindingDB and place in the `data/BindingDB/` directory:

- `affinities.csv`: Drug-protein binding affinities
- `compounds.csv`: Drug SMILES strings
- `proteins.csv`: Protein sequences
- `metadata.csv`: Additional binding metadata

## Preprocessing

BindingDB requires more preprocessing than Davis/KIBA:

1. Filter for high-quality measurements
2. Convert different affinity units to common scale
3. Handle multiple measurements per pair
4. Filter for complete drug-protein pairs

## Usage

```python
from data import DTIDataset

# Load BindingDB dataset
dataset = DTIDataset(
    root='./data',
    dataset_name='bindingdb',
    split='train'
)
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{gilson2016bindingdb,
  title={BindingDB in 2015: a public database for medicinal chemistry, computational chemistry and systems pharmacology},
  author={Gilson, Michael K and Liu, Tiqing and Baitaluk, Michael and Nicola, George and Hwang, Linda and Chong, Jenny},
  journal={Nucleic acids research},
  volume={44},
  number={D1},
  pages={D1045--D1053},
  year={2016},
  publisher={Oxford University Press}
}
```
