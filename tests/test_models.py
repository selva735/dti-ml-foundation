"""Unit tests for CMAT-DTI model components.

Tests cover forward passes, output shapes, and the MC Dropout uncertainty
estimation, all running on CPU without external dependencies (RDKit, etc).
"""

import math
import pytest
import torch
import torch.nn as nn

from src.models.drug_encoder import MolecularGraphTransformer
from src.models.protein_encoder import ProteinSequenceTransformer
from src.models.cross_attention import BidirectionalCrossAttention
from src.models.cmat_dti import CMATDTI, ModelConfig, create_model


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

BATCH = 4
MAX_ATOMS = 20
ATOM_FEAT_DIM = 57
MAX_SEQ = 32
VOCAB_SIZE = 23  # ProteinSequenceTransformer default


def make_drug_batch(batch=BATCH, n_atoms=MAX_ATOMS, feat_dim=ATOM_FEAT_DIM):
    """Create random atom features, distance matrix, and padding mask."""
    atom_features = torch.rand(batch, n_atoms, feat_dim)
    # Random integer distances in [0, 9]
    dist_matrix = torch.randint(0, 10, (batch, n_atoms, n_atoms))
    # Make symmetric
    dist_matrix = (dist_matrix + dist_matrix.transpose(1, 2)) // 2
    dist_matrix.diagonal(dim1=1, dim2=2).fill_(0)
    # Last 5 atoms are padding
    atom_mask = torch.zeros(batch, n_atoms, dtype=torch.bool)
    atom_mask[:, -5:] = True
    return atom_features, dist_matrix, atom_mask


def make_protein_batch(batch=BATCH, seq_len=MAX_SEQ, vocab_size=VOCAB_SIZE):
    """Create random protein tokens and padding mask."""
    tokens = torch.randint(1, vocab_size - 1, (batch, seq_len))
    padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    padding_mask[:, -4:] = True
    return tokens, padding_mask


# ---------------------------------------------------------------------------
# Drug encoder tests
# ---------------------------------------------------------------------------

class TestMolecularGraphTransformer:
    @pytest.fixture
    def encoder(self):
        return MolecularGraphTransformer(
            atom_feat_dim=ATOM_FEAT_DIM,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            output_dim=128,
        )

    def test_output_shape(self, encoder):
        atom_features, dist_matrix, atom_mask = make_drug_batch()
        out = encoder(atom_features, dist_matrix, atom_mask)
        assert out.shape == (BATCH, 128), f"Expected (4, 128) got {out.shape}"

    def test_no_mask(self, encoder):
        atom_features, dist_matrix, _ = make_drug_batch()
        out = encoder(atom_features, dist_matrix)
        assert out.shape == (BATCH, 128)

    @pytest.mark.parametrize("pooling", ["mean", "max", "attention"])
    def test_pooling_strategies(self, pooling):
        enc = MolecularGraphTransformer(
            atom_feat_dim=ATOM_FEAT_DIM,
            embed_dim=32,
            num_layers=1,
            num_heads=4,
            pooling=pooling,
        )
        atom_features, dist_matrix, atom_mask = make_drug_batch()
        out = enc(atom_features, dist_matrix, atom_mask)
        assert out.shape == (BATCH, 32)

    def test_output_is_finite(self, encoder):
        atom_features, dist_matrix, atom_mask = make_drug_batch()
        out = encoder(atom_features, dist_matrix, atom_mask)
        assert torch.isfinite(out).all()

    def test_single_sample(self, encoder):
        atom_features, dist_matrix, atom_mask = make_drug_batch(batch=1)
        out = encoder(atom_features, dist_matrix, atom_mask)
        assert out.shape == (1, 128)


# ---------------------------------------------------------------------------
# Protein encoder tests
# ---------------------------------------------------------------------------

class TestProteinSequenceTransformer:
    @pytest.fixture
    def encoder(self):
        return ProteinSequenceTransformer(
            vocab_size=VOCAB_SIZE,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            ffn_dim=128,
            output_dim=128,
        )

    def test_output_shape(self, encoder):
        tokens, padding_mask = make_protein_batch()
        out = encoder(tokens=tokens, padding_mask=padding_mask)
        assert out.shape == (BATCH, 128)

    def test_no_mask(self, encoder):
        tokens, _ = make_protein_batch()
        out = encoder(tokens=tokens)
        assert out.shape == (BATCH, 128)

    def test_plm_input(self):
        plm_dim = 480
        enc = ProteinSequenceTransformer(
            vocab_size=VOCAB_SIZE,
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            output_dim=64,
            plm_input_dim=plm_dim,
        )
        plm_emb = torch.rand(BATCH, MAX_SEQ, plm_dim)
        out = enc(plm_embeddings=plm_emb)
        assert out.shape == (BATCH, 64)

    def test_output_is_finite(self, encoder):
        tokens, padding_mask = make_protein_batch()
        out = encoder(tokens=tokens, padding_mask=padding_mask)
        assert torch.isfinite(out).all()

    def test_requires_tokens_or_plm(self, encoder):
        with pytest.raises(ValueError):
            encoder()  # Neither tokens nor plm_embeddings provided


# ---------------------------------------------------------------------------
# Cross-attention fusion tests
# ---------------------------------------------------------------------------

class TestBidirectionalCrossAttention:
    @pytest.fixture
    def fusion(self):
        return BidirectionalCrossAttention(
            drug_dim=64,
            protein_dim=64,
            cross_dim=64,
            num_heads=4,
            num_layers=2,
            output_dim=128,
        )

    def test_output_shape(self, fusion):
        drug = torch.rand(BATCH, 64)
        prot = torch.rand(BATCH, 64)
        fused, d, p = fusion(drug, prot)
        assert fused.shape == (BATCH, 128)
        assert d.shape == (BATCH, 64)
        assert p.shape == (BATCH, 64)

    def test_output_is_finite(self, fusion):
        drug = torch.rand(BATCH, 64)
        prot = torch.rand(BATCH, 64)
        fused, _, _ = fusion(drug, prot)
        assert torch.isfinite(fused).all()


# ---------------------------------------------------------------------------
# Full CMAT-DTI model tests
# ---------------------------------------------------------------------------

class TestCMATDTI:
    @pytest.fixture
    def model(self):
        cfg = ModelConfig(
            atom_feat_dim=ATOM_FEAT_DIM,
            drug_embed_dim=64,
            drug_num_layers=2,
            drug_num_heads=4,
            drug_ffn_dim=128,
            protein_embed_dim=64,
            protein_num_layers=2,
            protein_num_heads=4,
            protein_ffn_dim=128,
            cross_dim=64,
            cross_num_heads=4,
            cross_num_layers=1,
            fusion_output_dim=128,
            head_hidden_dims=[64],
            output_dim=1,
            mc_dropout_samples=5,
        )
        return CMATDTI(cfg)

    def test_forward_shape(self, model):
        atom_features, dist_matrix, atom_mask = make_drug_batch()
        tokens, protein_padding_mask = make_protein_batch()
        out = model(
            atom_features=atom_features,
            dist_matrix=dist_matrix,
            protein_tokens=tokens,
            atom_mask=atom_mask,
            protein_padding_mask=protein_padding_mask,
        )
        assert out.shape == (BATCH, 1)

    def test_forward_is_finite(self, model):
        atom_features, dist_matrix, atom_mask = make_drug_batch()
        tokens, protein_padding_mask = make_protein_batch()
        out = model(
            atom_features=atom_features,
            dist_matrix=dist_matrix,
            protein_tokens=tokens,
            atom_mask=atom_mask,
            protein_padding_mask=protein_padding_mask,
        )
        assert torch.isfinite(out).all()

    def test_uncertainty_estimation(self, model):
        atom_features, dist_matrix, atom_mask = make_drug_batch()
        tokens, protein_padding_mask = make_protein_batch()
        result = model.predict_with_uncertainty(
            atom_features=atom_features,
            dist_matrix=dist_matrix,
            protein_tokens=tokens,
            atom_mask=atom_mask,
            protein_padding_mask=protein_padding_mask,
            n_samples=5,
        )
        assert "mean" in result
        assert "std" in result
        assert "samples" in result
        assert result["mean"].shape == (BATCH, 1)
        assert result["std"].shape == (BATCH, 1)
        assert result["samples"].shape == (5, BATCH, 1)
        # Uncertainty should be non-negative
        assert (result["std"] >= 0).all()

    def test_create_model_factory(self):
        model = create_model(atom_feat_dim=ATOM_FEAT_DIM, drug_embed_dim=32)
        assert isinstance(model, CMATDTI)

    def test_gradient_flow(self, model):
        """Verify that gradients flow through all parameters."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        atom_features, dist_matrix, atom_mask = make_drug_batch()
        tokens, protein_padding_mask = make_protein_batch()
        model.train()
        pred = model(
            atom_features=atom_features,
            dist_matrix=dist_matrix,
            protein_tokens=tokens,
            atom_mask=atom_mask,
            protein_padding_mask=protein_padding_mask,
        )
        loss = pred.mean()
        loss.backward()
        # Check at least some parameters got gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients found"
