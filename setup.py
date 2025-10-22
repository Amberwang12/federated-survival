from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="federated-survival",
    version="0.2.0",
    author="Wenjun Wang",
    author_email="amber930422@163.com",
    description="A federated learning framework for survival analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Amberwang12/federated-survival",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.2.2",
        "scikit-learn>=1.3.0",
        "sklearn-pandas>=2.2.0",
        "lifelines>=0.27.0",
        "pycox>=0.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "scipy>=1.10.0",
        "tqdm>=4.67.0",
        "torchvision>=0.17.0",
        "torchtuples>=0.2.0",
    ],
) 