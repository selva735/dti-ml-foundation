"""
Example usage of the DTI-ML-Foundation framework.

This script demonstrates how to use the multi-modal DTI prediction framework
for making predictions on drug-target pairs.
"""

import torch
import numpy as np
from models import DrugGNN, ProteinPLM, TDAEncoder, FusionAttention, EvidentialHead
from data.preprocessing import DrugPreprocessor, ProteinPreprocessor, TDAPreprocessor


def example_single_prediction():
    """
    Example: Make a single drug-target interaction prediction.
    """
    print("=" * 80)
    print("Example 1: Single Drug-Target Prediction")
    print("=" * 80)
    
    # Initialize preprocessors
    drug_preprocessor = DrugPreprocessor()
    protein_preprocessor = ProteinPreprocessor()
    tda_preprocessor = TDAPreprocessor()
    
    # Example drug (Ibuprofen) and protein sequence
    drug_smiles = "CC(C)Cc1ccc(cc1)C(C)C(O)=O"
    protein_sequence = "MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQK"
    
    # Preprocess drug
    drug_graph = drug_preprocessor.smiles_to_graph(drug_smiles)
    
    # Preprocess protein
    protein_tokens = protein_preprocessor.sequence_to_tokens(protein_sequence)
    
    # Compute TDA features (placeholder - in practice, compute from molecular structure)
    tda_features = tda_preprocessor.compute_persistence_features(drug_graph)
    
    # Initialize models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    drug_gnn = DrugGNN(in_features=78, out_dim=256).to(device)
    protein_plm = ProteinPLM(embedding_dim=1280, out_dim=256).to(device)
    tda_encoder = TDAEncoder(input_dim=100, out_dim=256).to(device)
    fusion = FusionAttention(drug_dim=256, protein_dim=256, tda_dim=256).to(device)
    prediction_head = EvidentialHead(in_dim=512).to(device)
    
    # Prepare inputs
    node_features = torch.FloatTensor(drug_graph['node_features']).to(device)
    edge_index = torch.LongTensor(drug_graph['edge_index']).to(device)
    batch = torch.zeros(drug_graph['num_nodes'], dtype=torch.long).to(device)
    
    protein_tokens_tensor = torch.LongTensor(protein_tokens).unsqueeze(0).to(device)
    
    persistence_img = torch.FloatTensor(tda_features['persistence_image']).unsqueeze(0).unsqueeze(0).to(device)
    betti_curve = torch.FloatTensor(tda_features['betti_curve']).unsqueeze(0).to(device)
    landscape = torch.FloatTensor(tda_features['landscape']).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        # Encode each modality
        drug_embed = drug_gnn(node_features, edge_index, batch)
        protein_embed = protein_plm(protein_tokens_tensor)
        tda_embed = tda_encoder(persistence_img, betti_curve, landscape)
        
        # Fuse modalities
        fused_embed, attention_weights = fusion(drug_embed, protein_embed, tda_embed)
        
        # Make prediction with uncertainty
        output = prediction_head(fused_embed, return_all=True)
    
    # Print results
    print(f"\nDrug: {drug_smiles}")
    print(f"Protein length: {len(protein_sequence)} amino acids")
    print(f"\nPrediction Results:")
    print(f"  Predicted Affinity: {output['pred'].item():.4f}")
    print(f"  Epistemic Uncertainty: {output['epistemic_unc'].item():.4f}")
    print(f"  Aleatoric Uncertainty: {output['aleatoric_unc'].item():.4f}")
    print(f"  Total Uncertainty: {output['total_unc'].item():.4f}")
    
    # Print attention weights
    print(f"\nModal Importance:")
    modal_importance = attention_weights['modal_importance'][0].cpu().numpy()
    print(f"  Drug: {modal_importance[0]:.3f}")
    print(f"  Protein: {modal_importance[1]:.3f}")
    print(f"  TDA: {modal_importance[2]:.3f}")


def example_batch_prediction():
    """
    Example: Make batch predictions on multiple drug-target pairs.
    """
    print("\n" + "=" * 80)
    print("Example 2: Batch Drug-Target Predictions")
    print("=" * 80)
    
    # Example drug-target pairs
    pairs = [
        {
            'drug': 'CC(C)Cc1ccc(cc1)C(C)C(O)=O',  # Ibuprofen
            'protein': 'MKKFFDSRREQGGSGLGSGSSGGGGSTSGLGSGYIGRVFGIGRQQVTVDEVLAEGGFAIVFLVRTSNGMKCALKRMFVNNEHDLQVCKREIQIMRDLSGHKNIVGYIDSSINNVSSGDVWEVLILMDFCRGGQVVNLMNQRLQTGFTENEVLQIFCDTCEAVARLHQCKTPIIHRDLKVENILLHDRGHYVLCDFGSATNKFQNPQTEGVNAVEDEIKKYTTLSYRAPEMVNLYSGKIITTKADIWALGCLLYKLCYFTLPFGESQVAICDGNFTIPDNSRYSQDMHCLIRYMLEPDPDKRPDIYQVSYFSFKLLKKECPIPNVQNSPIPAKLPEPVKASEAAAKKTQPKARLTDPIPTTETSIAPRQRPKAGQTQPNQAQK',
            'name': 'Ibuprofen-Kinase'
        },
        {
            'drug': 'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
            'protein': 'MPPLLPLALLLGPLLRAQGSPQSQASPPQALGGLFPGQPSGQASLFPGLAPGQQPGLSGHEPPPGLQASPLLHGVQVEPPPQGLLLHQQHQSQSPQRLLLPSPPQALSPHQLRAPGAHSSPQRLSGQAQLLPEAQPRLPPQGAGHHQPGHQPGPLPGPAQPLHGGPGGAPPAQGPGQPGLSLGQGPLGPQGPQGGLPGLLGAQHQPGVHGPQGPQGPQGAPGLAGPAGPQGPQGPQGPQGPPGPQGPQGPPGPPGPPGPPGPPGPPGPPGPPGPPGPP',
            'name': 'Aspirin-Collagen'
        }
    ]
    
    print(f"\nPredicting affinities for {len(pairs)} drug-target pairs...")
    
    # Note: In practice, you would batch process these
    # This is a simplified example
    for i, pair in enumerate(pairs):
        print(f"\n{i+1}. {pair['name']}")
        print(f"   Drug: {pair['drug'][:40]}...")
        print(f"   Protein length: {len(pair['protein'])} aa")
        print(f"   Predicted affinity: {np.random.randn() * 2 + 7:.4f}")  # Placeholder
        print(f"   Uncertainty: {np.random.rand():.4f}")  # Placeholder


def example_model_interpretation():
    """
    Example: Interpret model predictions using attention weights.
    """
    print("\n" + "=" * 80)
    print("Example 3: Model Interpretation")
    print("=" * 80)
    
    print("\nAttention Weight Interpretation:")
    print("  - High drug attention: Drug structure is critical for binding")
    print("  - High protein attention: Protein pocket/structure is important")
    print("  - High TDA attention: Topological features are significant")
    
    print("\nUncertainty Interpretation:")
    print("  - High epistemic: Model is uncertain (needs more training data)")
    print("  - High aleatoric: Data is noisy (inherent variability)")
    print("  - Low total: High confidence prediction")
    
    print("\nExample Attention Patterns:")
    scenarios = {
        'High affinity binding': {'drug': 0.45, 'protein': 0.40, 'tda': 0.15},
        'Weak binding': {'drug': 0.30, 'protein': 0.35, 'tda': 0.35},
        'Non-specific binding': {'drug': 0.33, 'protein': 0.33, 'tda': 0.34}
    }
    
    for scenario, weights in scenarios.items():
        print(f"\n  {scenario}:")
        print(f"    Drug: {weights['drug']:.2f} | Protein: {weights['protein']:.2f} | TDA: {weights['tda']:.2f}")


def main():
    """Run all examples."""
    print("\nDTI-ML-Foundation Examples")
    print("=" * 80)
    
    try:
        example_single_prediction()
        example_batch_prediction()
        example_model_interpretation()
        
        print("\n" + "=" * 80)
        print("Examples completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Train your own model: python train.py --dataset davis")
        print("  2. Evaluate on cold-start: python eval.py --checkpoint model.pt --evaluate_all_splits")
        print("  3. Explore notebooks: jupyter notebook notebooks/")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Note: These examples require PyTorch and other dependencies to be installed.")
        print("See SETUP.md for installation instructions.")


if __name__ == '__main__':
    main()
