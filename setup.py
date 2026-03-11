from setuptools import setup, find_packages

setup(
    name="fawp-index",
    version="0.1.0",
    author="Ralph Clayton",
    description="FAWP Alpha Index v2.1 — Information-Control Exclusion Principle detector",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://zenodo.org/records/18673949",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.21", "pandas>=1.3"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
    ],
)
