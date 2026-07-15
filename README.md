# VQC-RAG: Variational Quantum Circuit-Assisted Re-ranking for Hybrid Retrieval-Augmented Generation

# VQC-RAG

![Python](https://img.shields.io/badge/Python-3.13-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://github.com/Kgalappaththi/VQC-RAG/actions/workflows/python-tests.yml/badge.svg)

## Overview

This repository contains the complete implementation of a hybrid quantum-classical document re-ranking framework developed for Retrieval-Augmented Generation (RAG).

The proposed framework investigates whether Variational Quantum Circuits (VQCs) can provide complementary relevance information during document re-ranking after Hybrid Retrieval. Hybrid Retrieval combines BM25 lexical retrieval and dense semantic retrieval, while the VQC learns nonlinear query-document relevance from engineered interaction features.

The implementation accompanies the research paper:

**"A Systematic Evaluation of Variational Quantum Circuit-Assisted Re-ranking for Hybrid Retrieval-Augmented Generation"**

---

# Main Contributions

- Hybrid Retrieval baseline using BM25 and Dense Retrieval
- Variational Quantum Circuit (VQC) document reranking
- Pairwise ranking optimization
- Quantum capacity analysis (4-, 6-, and 8-qubit circuits)
- Classical MLP control model
- Multiple fusion strategies
  - Weighted Fusion
  - Rank Fusion
  - Reciprocal Rank Fusion
- Statistical significance analysis
- Fully reproducible experimental pipeline

---

# Repository Structure

```
VQC-RAG/
│
├── configs/
├── data/
├── experiments/
├── figures/
├── results/
├── src/
├── tests/
│
├── README.md
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

---

# Dataset

Experiments use the **SciFact** dataset from the **BEIR benchmark**.

Reference:

Thakur et al., BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.

---

# Installation

## Clone repository

```bash
git clone https://github.com/<YOUR_USERNAME>/VQC-RAG.git
cd VQC-RAG
```

## Create environment

```bash
conda create -n vqc-rag python=3.13
conda activate vqc-rag
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Reproducing the Experiments

## 1. Generate embeddings

```bash
python -m experiments.create_query_embeddings
```

## 2. Run BM25

```bash
python -m experiments.run_bm25
```

## 3. Run Dense Retrieval

```bash
python -m experiments.run_dense
```

## 4. Run Hybrid Retrieval

```bash
python -m experiments.run_baseline_hybrid
```

## 5. Export Hybrid candidates

```bash
python -m experiments.export_hybrid_candidates
```

## 6. Build VQC dataset

```bash
python -m experiments.build_vqc_dataset
```

## 7. Train the VQC

```bash
python -m experiments.run_vqc_training
```

## 8. Evaluate the trained model

```bash
python -m experiments.evaluate_vqc_model
```

## 9. Run VQC reranking

```bash
python -m experiments.run_vqc_reranking
```

## 10. Run the classical MLP control

```bash
python -m experiments.run_mlp_pairwise
```

## 11. Statistical analysis

```bash
python -m experiments.run_statistical_tests
```

## 12. Generate publication figures

```bash
python -m experiments.generate_publication_figures
```

---

# Experimental Results

Best Hybrid Retrieval

| Metric | Value |
|---------|------:|
| MRR | 0.6146 |
| nDCG@10 | 0.6456 |
| MAP | 0.6036 |

Best Quantum Configuration

- 6 qubits
- Depth = 4
- Pairwise Accuracy = 0.6437

The weighted Hybrid + VQC fusion achieved the strongest overall performance among the evaluated quantum-assisted methods.

Statistical hypothesis testing indicated that the observed improvements were **not statistically significant (α = 0.05)**.

---

# Software

- Python
- PennyLane
- PyTorch
- SentenceTransformers
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

# Citation

If you use this repository, please cite the accompanying publication.

```bibtex
(To be added after publication)
```

---

# License

MIT License

---

# Contact

Dr. Kalpana Galappaththi

Institute of Technology

University of Moratuwa

Sri Lanka