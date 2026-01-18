"""
Data preprocessing utilities for drugs, proteins, and topological features.
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import warnings

warnings.filterwarnings('ignore')


class DrugPreprocessor:
    """
    Preprocessor for drug molecules.
    
    Converts SMILES strings to molecular graphs and features.
    
    Args:
        max_atoms (int): Maximum number of atoms in molecule
        atom_features (list): List of atom features to extract
        use_chirality (bool): Whether to include chirality information
    """
    
    def __init__(
        self,
        max_atoms=100,
        atom_features=['atomic_num', 'degree', 'formal_charge', 'hybridization',
                      'is_aromatic', 'num_h'],
        use_chirality=True
    ):
        self.max_atoms = max_atoms
        self.atom_features = atom_features
        self.use_chirality = use_chirality
        
    def smiles_to_graph(self, smiles):
        """
        Convert SMILES string to molecular graph.
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dict: Graph data with node features and edge indices
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens if needed
            mol = Chem.AddHs(mol)
            
            # Get node features
            node_features = []
            for atom in mol.GetAtoms():
                features = self._get_atom_features(atom)
                node_features.append(features)
            
            node_features = np.array(node_features, dtype=np.float32)
            
            # Get edge indices
            edge_indices = []
            edge_features = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Add both directions (undirected graph)
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                
                bond_features = self._get_bond_features(bond)
                edge_features.append(bond_features)
                edge_features.append(bond_features)
            
            if len(edge_indices) == 0:
                edge_indices = np.array([[0, 0]], dtype=np.int64)
                edge_features = np.zeros((1, len(edge_features[0]) if edge_features else 4), dtype=np.float32)
            else:
                edge_indices = np.array(edge_indices, dtype=np.int64).T
                edge_features = np.array(edge_features, dtype=np.float32)
            
            return {
                'node_features': node_features,
                'edge_index': edge_indices,
                'edge_features': edge_features,
                'num_nodes': len(node_features)
            }
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")
            return None
    
    def _get_atom_features(self, atom):
        """Extract features from an atom."""
        features = []
        
        if 'atomic_num' in self.atom_features:
            # One-hot encoding of atomic number (common atoms)
            atomic_num = atom.GetAtomicNum()
            common_atoms = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
            features.extend([1 if atomic_num == x else 0 for x in common_atoms])
            features.append(1 if atomic_num not in common_atoms else 0)  # Other
        
        if 'degree' in self.atom_features:
            degree = atom.GetDegree()
            features.extend([1 if degree == x else 0 for x in range(6)])
        
        if 'formal_charge' in self.atom_features:
            charge = atom.GetFormalCharge()
            features.extend([1 if charge == x else 0 for x in [-1, 0, 1]])
        
        if 'hybridization' in self.atom_features:
            hybridization = atom.GetHybridization()
            hyb_types = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                        Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                        Chem.HybridizationType.SP3D2]
            features.extend([1 if hybridization == x else 0 for x in hyb_types])
        
        if 'is_aromatic' in self.atom_features:
            features.append(1 if atom.GetIsAromatic() else 0)
        
        if 'num_h' in self.atom_features:
            num_h = atom.GetTotalNumHs()
            features.extend([1 if num_h == x else 0 for x in range(5)])
        
        if self.use_chirality and 'chirality' in self.atom_features:
            try:
                chiral_tag = atom.GetChiralTag()
                features.extend([
                    1 if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else 0,
                    1 if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else 0
                ])
            except:
                features.extend([0, 0])
        
        return features
    
    def _get_bond_features(self, bond):
        """Extract features from a bond."""
        features = []
        
        # Bond type
        bond_type = bond.GetBondType()
        features.extend([
            1 if bond_type == Chem.BondType.SINGLE else 0,
            1 if bond_type == Chem.BondType.DOUBLE else 0,
            1 if bond_type == Chem.BondType.TRIPLE else 0,
            1 if bond_type == Chem.BondType.AROMATIC else 0
        ])
        
        # Is conjugated
        features.append(1 if bond.GetIsConjugated() else 0)
        
        # Is in ring
        features.append(1 if bond.IsInRing() else 0)
        
        return features
    
    def compute_molecular_descriptors(self, smiles):
        """
        Compute molecular descriptors.
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dict: Molecular descriptors
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {
            'mol_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_h_donors': Descriptors.NumHDonors(mol),
            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol)
        }
        
        return descriptors


class ProteinPreprocessor:
    """
    Preprocessor for protein sequences.
    
    Args:
        max_length (int): Maximum sequence length
        vocab (dict): Amino acid vocabulary
        use_bpe (bool): Whether to use byte-pair encoding
    """
    
    def __init__(
        self,
        max_length=1000,
        vocab=None,
        use_bpe=False
    ):
        self.max_length = max_length
        self.use_bpe = use_bpe
        
        # Standard amino acid vocabulary
        if vocab is None:
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            self.vocab = {aa: i+1 for i, aa in enumerate(amino_acids)}
            self.vocab['<PAD>'] = 0
            self.vocab['<UNK>'] = len(self.vocab)
        else:
            self.vocab = vocab
    
    def sequence_to_tokens(self, sequence):
        """
        Convert protein sequence to token IDs.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            np.ndarray: Token IDs
        """
        sequence = sequence.upper()[:self.max_length]
        
        tokens = [self.vocab.get(aa, self.vocab['<UNK>']) for aa in sequence]
        
        # Pad to max length
        if len(tokens) < self.max_length:
            tokens += [self.vocab['<PAD>']] * (self.max_length - len(tokens))
        
        return np.array(tokens, dtype=np.int64)
    
    def compute_protein_features(self, sequence):
        """
        Compute protein features.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            dict: Protein features
        """
        features = {
            'length': len(sequence),
            'molecular_weight': self._compute_molecular_weight(sequence),
            'charge': self._compute_charge(sequence),
            'hydrophobicity': self._compute_hydrophobicity(sequence)
        }
        
        return features
    
    def _compute_molecular_weight(self, sequence):
        """Compute molecular weight of protein."""
        aa_weights = {
            'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
            'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
            'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
            'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
        }
        
        weight = sum(aa_weights.get(aa, 0) for aa in sequence)
        return weight
    
    def _compute_charge(self, sequence):
        """Compute net charge of protein at pH 7."""
        positive = sequence.count('K') + sequence.count('R') + sequence.count('H')
        negative = sequence.count('D') + sequence.count('E')
        return positive - negative
    
    def _compute_hydrophobicity(self, sequence):
        """Compute hydrophobicity score."""
        hydrophobicity_scale = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        if len(sequence) == 0:
            return 0.0
        
        score = sum(hydrophobicity_scale.get(aa, 0) for aa in sequence) / len(sequence)
        return score


class TDAPreprocessor:
    """
    Preprocessor for topological data analysis features.
    
    Args:
        resolution (int): Resolution for persistence images
        num_points (int): Number of points for Betti curves and landscapes
    """
    
    def __init__(
        self,
        resolution=20,
        num_points=50
    ):
        self.resolution = resolution
        self.num_points = num_points
    
    def compute_persistence_features(self, mol_graph):
        """
        Compute persistence features from molecular graph.
        
        Args:
            mol_graph (dict): Molecular graph data
            
        Returns:
            dict: TDA features
        """
        # Placeholder for actual TDA computation
        # In practice, use libraries like gudhi or ripser
        
        # Generate dummy persistence diagram
        num_features = np.random.randint(10, 50)
        birth = np.random.rand(num_features)
        death = birth + np.random.rand(num_features) * 0.5
        persistence_diagram = np.column_stack([birth, death])
        
        # Compute persistence image
        from models.tda_encoder import compute_persistence_image, compute_betti_curve
        persistence_image = compute_persistence_image(
            persistence_diagram,
            resolution=self.resolution
        )
        
        # Compute Betti curve
        betti_curve = compute_betti_curve(
            persistence_diagram,
            num_points=self.num_points
        )
        
        # Compute persistence landscape (simplified)
        landscape = self._compute_landscape(persistence_diagram)
        
        return {
            'persistence_image': persistence_image,
            'betti_curve': betti_curve,
            'landscape': landscape,
            'persistence_diagram': persistence_diagram
        }
    
    def _compute_landscape(self, diagram):
        """Compute persistence landscape."""
        # Simplified landscape computation
        filtration_values = np.linspace(diagram[:, 0].min(), diagram[:, 1].max(), self.num_points)
        landscape = np.zeros(self.num_points)
        
        for i, t in enumerate(filtration_values):
            # Compute landscape function value at t
            values = []
            for birth, death in diagram:
                if birth <= t <= death:
                    values.append(min(t - birth, death - t))
            
            if values:
                landscape[i] = max(values)
        
        return landscape


def collate_dti_batch(batch):
    """
    Collate function for DTI data batches.
    
    Args:
        batch (list): List of samples
        
    Returns:
        dict: Batched data
    """
    # Separate drug graphs, protein sequences, and affinities
    drug_graphs = [s.get('drug_graph') for s in batch if s.get('drug_graph') is not None]
    protein_tokens = [s.get('protein_tokens') for s in batch if s.get('protein_tokens') is not None]
    affinities = [s.get('affinity') for s in batch if s.get('affinity') is not None]
    
    batched_data = {}
    
    if affinities:
        batched_data['affinity'] = torch.FloatTensor(affinities)
    
    if protein_tokens:
        batched_data['protein_tokens'] = torch.stack([torch.LongTensor(p) for p in protein_tokens])
    
    # Handle graph batching (would use PyTorch Geometric's Batch in practice)
    if drug_graphs:
        batched_data['drug_graphs'] = drug_graphs
    
    return batched_data
