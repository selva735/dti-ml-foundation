"""Unit tests for data featurization and dataset utilities.

All tests avoid RDKit by using mock/pre-built feature tensors where needed,
or by testing only the non-RDKit codepaths (tokenizer, cold-start splitter,
collate function).
"""

import pytest
import torch

from src.data.dataset import DTIPair, DTIDataset, cold_start_split, collate_fn
from src.models.protein_encoder import (
    tokenize_sequence,
    AA_TO_IDX,
    CLS_IDX,
    PAD_IDX,
    AA_VOCAB_SIZE,
)
from src.data.featurizer import ProteinFeaturizer


# ---------------------------------------------------------------------------
# Protein tokenizer tests (no RDKit)
# ---------------------------------------------------------------------------

class TestTokenizeSequence:
    def test_basic(self):
        tokens = tokenize_sequence("ACD")
        assert tokens[0] == CLS_IDX, "First token should be CLS"
        assert len(tokens) == 4  # CLS + 3 residues

    def test_truncation(self):
        long_seq = "A" * 200
        tokens = tokenize_sequence(long_seq, max_len=50)
        # CLS + 50 residues
        assert len(tokens) == 51

    def test_unknown_aa(self):
        tokens = tokenize_sequence("A*C")  # * is not a standard AA
        assert tokens[2] == AA_TO_IDX["X"]

    def test_lowercase_normalised(self):
        tokens_upper = tokenize_sequence("ACG")
        tokens_lower = tokenize_sequence("acg")
        assert tokens_upper == tokens_lower

    def test_empty_sequence(self):
        tokens = tokenize_sequence("")
        assert tokens == [CLS_IDX]


# ---------------------------------------------------------------------------
# ProteinFeaturizer tests
# ---------------------------------------------------------------------------

class TestProteinFeaturizer:
    @pytest.fixture
    def featurizer(self):
        return ProteinFeaturizer(max_seq_len=20)

    def test_output_keys(self, featurizer):
        result = featurizer("ACDEFG")
        assert "tokens" in result
        assert "padding_mask" in result

    def test_token_dtype(self, featurizer):
        result = featurizer("ACDEFG")
        assert result["tokens"].dtype == torch.long

    def test_mask_dtype(self, featurizer):
        result = featurizer("ACDEFG")
        assert result["padding_mask"].dtype == torch.bool

    def test_no_padding_in_single_seq(self, featurizer):
        result = featurizer("ACG")
        assert not result["padding_mask"].any()

    def test_collate_pads_correctly(self, featurizer):
        seqs = ["ACG", "ACDEFGHIKLM"]  # different lengths
        result = featurizer.collate(seqs)
        assert result["tokens"].shape[0] == 2
        assert result["tokens"].shape[1] == result["padding_mask"].shape[1]
        # Shorter sequence should have padding
        assert result["padding_mask"][0].any()
        # Longer sequence should NOT have padding
        assert not result["padding_mask"][1].any()


# ---------------------------------------------------------------------------
# cold_start_split tests (no RDKit)
# ---------------------------------------------------------------------------

def _make_pairs(n_drugs=5, n_targets=4, n_pairs=20):
    import random
    rng = random.Random(0)
    return [
        DTIPair(
            drug_id=f"d{rng.randint(0, n_drugs-1)}",
            target_id=f"t{rng.randint(0, n_targets-1)}",
            smiles="CC",
            sequence="ACG",
            label=rng.uniform(0, 10),
        )
        for _ in range(n_pairs)
    ]


class TestColdStartSplit:
    def test_random_split_sizes(self):
        pairs = _make_pairs(n_pairs=100)
        train, val, test = cold_start_split(pairs, test_frac=0.1, val_frac=0.1, mode="random")
        total = len(train) + len(val) + len(test)
        assert total == 100
        assert len(test) == 10
        assert len(val) == 10

    def test_cold_drug_no_overlap(self):
        pairs = _make_pairs(n_pairs=100)
        train, val, test = cold_start_split(pairs, mode="cold_drug", seed=42)
        train_drugs = {p.drug_id for p in train}
        test_drugs = {p.drug_id for p in test}
        val_drugs = {p.drug_id for p in val}
        assert train_drugs.isdisjoint(test_drugs), "Test drugs should not appear in train"
        assert train_drugs.isdisjoint(val_drugs), "Val drugs should not appear in train"

    def test_cold_target_no_overlap(self):
        pairs = _make_pairs(n_pairs=100)
        train, val, test = cold_start_split(pairs, mode="cold_target", seed=42)
        train_targets = {p.target_id for p in train}
        test_targets = {p.target_id for p in test}
        assert train_targets.isdisjoint(test_targets)

    def test_cold_both_no_overlap(self):
        pairs = _make_pairs(n_drugs=8, n_targets=6, n_pairs=100)
        train, val, test = cold_start_split(pairs, mode="cold_both", seed=42)
        train_drugs = {p.drug_id for p in train}
        test_drugs = {p.drug_id for p in test}
        train_targets = {p.target_id for p in train}
        test_targets = {p.target_id for p in test}
        assert train_drugs.isdisjoint(test_drugs)
        assert train_targets.isdisjoint(test_targets)

    def test_invalid_mode(self):
        pairs = _make_pairs()
        with pytest.raises(ValueError):
            cold_start_split(pairs, mode="unsupported")

    def test_reproducible_with_seed(self):
        pairs = _make_pairs(n_pairs=50)
        train1, val1, test1 = cold_start_split(pairs, mode="random", seed=7)
        train2, val2, test2 = cold_start_split(pairs, mode="random", seed=7)
        assert [p.drug_id for p in test1] == [p.drug_id for p in test2]


# ---------------------------------------------------------------------------
# collate_fn tests (no RDKit – manual feature dicts)
# ---------------------------------------------------------------------------

def _fake_item(n_atoms=10, seq_len=15):
    """Create a fake featurized DTIPair item."""
    return {
        "drug_id": "d0",
        "target_id": "t0",
        "atom_features": torch.rand(n_atoms, 57),
        "dist_matrix": torch.randint(0, 5, (n_atoms, n_atoms)),
        "atom_mask": torch.zeros(n_atoms, dtype=torch.bool),
        "tokens": torch.randint(1, 20, (seq_len,)),
        "padding_mask": torch.zeros(seq_len, dtype=torch.bool),
        "label": torch.tensor(5.0),
    }


class TestCollateFn:
    def test_batch_atom_features_padded(self):
        batch = [_fake_item(n_atoms=8), _fake_item(n_atoms=12)]
        out = collate_fn(batch)
        assert out["atom_features"].shape == (2, 12, 57)

    def test_batch_tokens_padded(self):
        batch = [_fake_item(seq_len=10), _fake_item(seq_len=20)]
        out = collate_fn(batch)
        assert out["tokens"].shape == (2, 20)

    def test_batch_labels(self):
        batch = [_fake_item(), _fake_item()]
        out = collate_fn(batch)
        assert out["label"].shape == (2,)

    def test_protein_padding_mask_correct(self):
        batch = [_fake_item(seq_len=5), _fake_item(seq_len=10)]
        out = collate_fn(batch)
        # First sequence is shorter → its last 5 positions should be masked
        assert out["protein_padding_mask"][0, 5:].all()
        assert not out["protein_padding_mask"][1, :].any()
