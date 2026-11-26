# Simulated Variational Quantum Circuits for Hybrid Scientific Document Re-ranking

![Python](https://img.shields.io/badge/Python-3.13-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Latest Update – Scientific Reports (Second-round Revision)

The experimental code has been updated to support the revised
manuscript submitted to Scientific Reports in November 2026.

The second-round revision includes:

- MLP-Hinge baseline for comparison under a similar training objective.
- Independent cross-domain evaluation using NFCorpus.
- Query-level statistical analysis and multiple-comparison correction.
- Diagnostic experiments and computational latency benchmarking.
- Updated experimental configurations and reproducibility documentation.

The revised experimental code is available in the
[scientific_reports_2026_r2](./scientific_reports_2026_r2) directory.

### Reproducibility Archive

The complete reproducibility package, including source code,
experimental configurations, model checkpoints, processed features,
results and supporting evidence, is available on Zenodo.

- Version: 1.0.0
- DOI: [10.5281/zenodo.24132662](https://doi.org/10.5281/zenodo.24132662)

The GitHub repository provides selected experimental code and
documentation. The Zenodo archive should be used for complete
experimental reproduction.

The previous implementations and experimental history are retained
for reference.

**Manuscript status:** Revised manuscript submitted to Scientific
Reports; awaiting editorial evaluation.

---

## Overview

This repository contains the implementation used for the research paper:

**Simulated Variational Quantum Circuits for Hybrid Scientific Document Re-ranking**

The study investigates whether a simulated Variational Quantum Circuit (VQC) can improve second-stage document re-ranking in a hybrid scientific document retrieval system.

The proposed framework combines BM25 lexical retrieval, dense semantic retrieval, engineered query-document interaction features, Principal Component Analysis (PCA), and a simulated VQC to estimate document relevance. Different quantum-classical fusion strategies are evaluated and compared with a classical neural network baseline.

The main experiments were conducted using the SciFact dataset from
the BEIR benchmark. NFCorpus was used for independent cross-domain
evaluation in the second-round revision.

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
├── configs/                     Configuration files
├── data/                        Dataset preparation
├── experiments/                 Earlier experiment scripts
├── paper/figures/               Manuscript figures
├── results/                     Earlier experimental results
├── src/                         Source code
├── tests/                       Unit tests
├── scientific_reports_2026/     First-round revision
├── scientific_reports_2026_r2/  Second-round revision
├── README.md
├── requirements.txt
├── environment.yml
├── CHANGELOG.md
├── LICENSE
└── CITATION.cff
```

---

## Dataset

The main experiments use the SciFact dataset from the BEIR benchmark.
NFCorpus is used for independent cross-domain evaluation.

Both datasets are publicly available through the BEIR project.

The complete experimental materials and reproducibility evidence
are available in the Zenodo archive:
https://doi.org/10.5281/zenodo.24132662

Reference:

N. Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.*

---

## Installation

Clone the repository

```bash
git clone -b scientific-reports-revision-2 https://github.com/Kgalappaththi/VQC-RAG.git
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
For the second-round experiments, follow the instructions in
[scientific_reports_2026_r2/README.md](./scientific_reports_2026_r2/README.md)
and the complete Zenodo reproducibility archive.

---

## Running the Experiments

The earlier experiments can be reproduced using the scripts in
the `experiments` folder.

For the second-round experiments, refer to the
[revised experimental code](./scientific_reports_2026_r2)
and the [complete Zenodo archive](https://doi.org/10.5281/zenodo.24132662).

The typical workflow for the original SciFact experiments is
given below.

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

The study evaluates simulated VQC-assisted document re-ranking
and compares its performance with classical re-ranking methods.

The exploratory SciFact experiments show small improvements
over the hybrid retrieval baseline. However, the results do
not establish superiority over the classical MLP baselines.

Independent cross-domain evaluation using NFCorpus shows
that the VQC-assisted improvements observed on SciFact
do not generalise to NFCorpus under the evaluated settings.

The results do not demonstrate a computational advantage
of quantum processing.

The repository provides experimental code and supporting
documentation for further research on quantum-assisted
information retrieval. The complete reproducibility package
is available on Zenodo.

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

If you use the code or experimental materials from this study,
please cite the Zenodo reproducibility archive:

[Simulated Variational Quantum Circuits for Hybrid Scientific
Document Re-ranking – Reproducibility Package](https://doi.org/10.5281/zenodo.24132662)

Version: 1.0.0

DOI: 10.5281/zenodo.24132662

The journal article citation will be added after publication.

---



## License

The original source code is released under the MIT License.

Third-party datasets, models and other external materials
remain subject to their respective licences and terms of use.

See [THIRD_PARTY_NOTICES.md](./scientific_reports_2026_r2/THIRD_PARTY_NOTICES.md)
for further information.

---

## Contact

**Dr. Kalpana Galappaththi**

Division of Information Technology  
Institute of Technology  
University of Moratuwa  
Diyagama, Homagama  
Sri Lanka

Email: kgalappaththi@itum.mrt.ac.lk

## Scientific Reports – First-round Revision (November 2026)

The experimental code and supporting materials from the
first-round revision are retained in the
[scientific_reports_2026](./scientific_reports_2026) folder.

These materials are preserved for reference and to maintain
the research development history.

For the second-round revision, refer to
[scientific_reports_2026_r2](./scientific_reports_2026_r2)
and the published Zenodo reproducibility archive.





