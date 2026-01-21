"""Setup script for dti-ml-foundation package."""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dti-ml-foundation',
    version='0.1.0',
    author='DTI-ML-Foundation Contributors',
    description='Multi-modal attention-based GNN framework for drug-target interaction prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/selva735/dti-ml-foundation',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.2.0',
        'torch-geometric>=2.3.0',
        'rdkit>=2023.3.1',
        'transformers>=4.30.0',
        'fair-esm',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.10.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
        'tensorboard>=2.13.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.1.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
            'mypy>=1.3.0',
        ],
        'wandb': ['wandb>=0.15.0'],
        'optuna': ['optuna>=3.2.0'],
    },
)
