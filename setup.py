"""
WavLM Vocoder for French
Setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="wavlm-vocoder-french",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Neural vocoder for French speech reconstruction from WavLM representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NassimaOULDOUALI/wavlm-vocoder-french",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.13.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "eval": [
            "pesq>=0.0.4",
            "pystoi>=0.3.3",
        ],
    },
)
