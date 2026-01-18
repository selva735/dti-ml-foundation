# Contributing to DTI-ML-Foundation

Thank you for your interest in contributing to DTI-ML-Foundation! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/dti-ml-foundation.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests and linting
6. Commit your changes: `git commit -m "Description of changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Code Style

### Python

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Maximum line length: 100 characters

Example:

```python
def compute_affinity(drug_embed: torch.Tensor, protein_embed: torch.Tensor) -> torch.Tensor:
    """
    Compute drug-protein binding affinity.
    
    Args:
        drug_embed: Drug embeddings [batch_size, embed_dim]
        protein_embed: Protein embeddings [batch_size, embed_dim]
        
    Returns:
        Predicted affinity scores [batch_size, 1]
    """
    # Implementation here
    pass
```

### Documentation

- Update README.md if you add new features
- Add docstrings to all new functions and classes
- Update SETUP.md if you change installation requirements
- Create example notebooks for new major features

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=. tests/
```

### Writing Tests

- Write tests for all new functionality
- Aim for >80% code coverage
- Use pytest for testing
- Place tests in `tests/` directory

Example:

```python
def test_drug_gnn_forward():
    """Test DrugGNN forward pass."""
    model = DrugGNN(in_features=78, out_dim=256)
    # Test implementation
    assert output.shape == (batch_size, 256)
```

## Pull Request Process

1. **Update Documentation**: Ensure all documentation is updated
2. **Add Tests**: Add tests for new functionality
3. **Run Linting**: Ensure code passes linting checks
4. **Update CHANGELOG**: Add entry to CHANGELOG.md (if applicable)
5. **Request Review**: Tag maintainers for review

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you ran

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added
- [ ] All tests pass
```

## Areas for Contribution

### High Priority

- [ ] Implement actual dataset loaders for Davis, KIBA, BindingDB
- [ ] Add real protein language model integration (ESM-2, ProtBERT)
- [ ] Implement actual TDA computation using gudhi/ripser
- [ ] Add comprehensive unit tests
- [ ] Create data preprocessing scripts
- [ ] Add model checkpoints and pre-trained weights

### Medium Priority

- [ ] Add more visualization tools
- [ ] Implement additional evaluation metrics
- [ ] Add data augmentation strategies
- [ ] Create web interface for predictions
- [ ] Add hyperparameter tuning scripts
- [ ] Implement ensemble methods

### Low Priority

- [ ] Add support for more datasets
- [ ] Create Docker container
- [ ] Add CI/CD pipeline
- [ ] Create tutorial videos
- [ ] Add multilingual documentation

## Code Review Guidelines

### For Reviewers

- Be constructive and respectful
- Check for code quality, not just functionality
- Ensure tests are comprehensive
- Verify documentation is clear
- Check for security issues

### For Contributors

- Respond to all review comments
- Update PR based on feedback
- Be open to suggestions
- Keep PR focused on single feature/fix

## Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Give credit where due
- Follow code of conduct

## Questions?

- Open an issue for questions
- Join discussions in GitHub Discussions
- Email maintainers for sensitive issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Attribution

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to DTI-ML-Foundation! 🎉
