"""Package setup for quantitative-trading-ml-system."""

from setuptools import setup, find_packages

setup(
    name="quantitative-trading-ml-system",
    version="0.1.0",
    description="ML-based quantitative trading system for research",
    author="",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "statsmodels>=0.13.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
        "notebooks": ["jupyter>=1.0.0", "notebook>=6.4.0", "shap>=0.41.0"],
    },
)
