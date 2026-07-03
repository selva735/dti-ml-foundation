"""CMAT-DTI: Cross-Modal Attention Transformer for Drug-Target Interaction."""

from setuptools import setup, find_packages

setup(
    name="cmat-dti",
    version="0.1.0",
    description=(
        "CMAT-DTI: Cross-Modal Attention Transformer for Drug-Target Interaction "
        "Prediction using molecular graph transformers, protein sequence transformers, "
        "and bidirectional cross-modal attention."
    ),
    author="DTI ML Foundation",
    packages=find_packages(where=".", include=["src*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "molecular": ["rdkit>=2023.3.1"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "tensorboard": ["tensorboard>=2.14.0"],
    },
    entry_points={
        "console_scripts": [
            "cmat-train=scripts.train:main",
            "cmat-predict=scripts.predict:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
