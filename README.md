# Simulated Variational Quantum Circuits for Hybrid Scientific Document Re-ranking

![Python](https://img.shields.io/badge/Python-3.13-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This repository contains the implementation used for the research paper:

**Simulated Variational Quantum Circuits for Hybrid Scientific Document Re-ranking**

The study investigates whether a simulated Variational Quantum Circuit (VQC) can improve second-stage document re-ranking in a hybrid scientific document retrieval system.

The proposed framework combines BM25 lexical retrieval, dense semantic retrieval, engineered query-document interaction features, Principal Component Analysis (PCA), and a simulated VQC to estimate document relevance. Different quantum-classical fusion strategies are evaluated and compared with a classical neural network baseline.

The experiments were conducted using the SciFact dataset from the BEIR benchmark.

---

## Main Features

- Hybrid retrieval using BM25 and dense retrieval
- Simulated Variational Quantum Circuit (VQC) for document re-ranking
- PCA-based feature reduction
- Evaluation of multiple VQC configurations
- Comparison of three fusion strategies
- Classical multilayer perceptron (MLP) baseline
- Standard information retrieval evaluation metrics
- Statistical significance analysis

---

## Repository Structure

```
VQC-RAG/
│
├── configs/          Configuration files
├── data/             Dataset preparation
├── experiments/      Experiment scripts
├── figures/          Figures used in the manuscript
├── results/          Experimental results
├── src/              Source code
├── tests/            Unit tests
│
├── README.md
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

---

## Dataset

The experiments use the **SciFact** dataset from the **BEIR benchmark**.

The dataset is publicly available from the BEIR project and is not redistributed in this repository.

Reference:

N. Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.*

---

## Installation

Clone the repository

```bash
git clone https://github.com/Kgalappaththi/VQC-RAG.git
cd VQC-RAG
```

Create the Python environment

```bash
conda create -n vqc-rag python=3.13
conda activate vqc-rag
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

The main experiments can be reproduced using the scripts in the `experiments` folder.

Typical workflow:

1. Prepare the SciFact dataset
2. Run BM25 retrieval
3. Run dense retrieval
4. Generate hybrid retrieval results
5. Build query-document interaction features
6. Train the VQC models
7. Evaluate VQC performance
8. Apply fusion strategies
9. Perform statistical analysis
10. Generate the publication figures

---

## Experimental Summary

The study evaluates simulated VQC-assisted document re-ranking under several circuit configurations and compares the results with classical retrieval methods.

The experiments show that the simulated VQC produces only small changes in retrieval performance compared with the hybrid retrieval baseline. Under the evaluated settings, these improvements were not statistically significant.

The repository provides a reproducible implementation for future research on quantum-assisted information retrieval.

---

## Software

The implementation is based on

- Python
- PennyLane
- PyTorch
- Sentence Transformers
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## Citation

If you use this repository, please cite the associated publication.

The citation will be updated after publication.

---

## License

This project is released under the MIT License.

---

## Contact

**Dr. Kalpana Galappaththi**

Division of Information Technology  
Institute of Technology  
University of Moratuwa  
Diyagama, Homagama  
Sri Lanka

Email: kgalappaththi@itum.mrt.ac.lk
