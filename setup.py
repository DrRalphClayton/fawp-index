from setuptools import setup, find_packages

setup(
    name="fawp-index",
    version="0.5.0",
    author="Ralph Clayton",
    description="FAWP Alpha Index v2.1 — Information-Control Exclusion Principle detector",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DrRalphClayton/fawp-index",
    packages=find_packages(),
    package_data={"fawp_index": ["data/*.csv"]},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
    ],
    extras_require={
        "plot": ["matplotlib>=3.4"],
        "finance": ["yfinance>=0.2"],
        "weather": ["requests>=2.28"],
        "sklearn": ["scikit-learn>=1.0"],
        "all": ["matplotlib>=3.4", "yfinance>=0.2", "requests>=2.28", "scikit-learn>=1.0"],
    },
    entry_points={
        "console_scripts": [
            "fawp-index=fawp_index.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial",
    ],
)
