# Scientific Reports revision — GitHub update (November 2026)

This folder contains selected **code and recorded numerical results** from the reproducibility archive accompanying the revised manuscript *Simulated Variational Quantum Circuits for Hybrid Scientific Document Re-ranking*. It is a **supplementary GitHub update**, not a replacement for the existing `VQC-RAG` repository or the complete author-held archive.

## Experimental design

The original 300 BEIR SciFact evaluation queries were repartitioned into 180 training, 60 validation and 60 test queries (partition seed `20261106`). The partition is query-disjoint, but **not an external independent test set**. The test partition was examined during development, so all reported test comparisons are **exploratory**.

The revised pipeline fitted standardisation and PCA on training data. The reported simulated VQC used six qubits and four layers, with seeds 7, 21 and 42. Classical comparisons included MLP, logistic regression and RankSVM-style baselines; the manuscript's main comparisons use Hybrid, Hybrid + MLP, weighted Hybrid + VQC, RRF, and VQC alone.

## Recorded three-seed test nDCG@10

| Method | Mean |
|---|---:|
| Hybrid | 0.663326 |
| Hybrid + MLP | 0.681079 |
| Hybrid + weighted VQC | 0.687576 |
| RRF | 0.517302 |
| VQC alone | 0.276315 |

The weighted VQC comparison with Hybrid was positive in the exploratory test partition (Holm-adjusted p = 0.009690). The VQC versus MLP difference was **not statistically significant** (Holm-adjusted p = 0.399076). Validation did not establish a consistent VQC advantage. **No quantum advantage is claimed.**

## Included material

- `experiments/` — revised scripts as deposited in the source archive
- `original_pipeline/` — original source code, experiments and tests
- `configs/` and `data/splits/` — experiment configuration and recorded query splits
- `results/classical/` — numerical classical baseline evaluations
- `results/audit_seed*/` — per-query VQC evaluation and uncertainty summaries
- `results/vqc_seed*/`, `results/seed42/` — checkpoints, run manifests and training histories (document-level scores omitted)
- `results/final_statistical_analysis/` — matched test scores and paired statistical comparisons
- `final_statistics.py` — regenerate the statistical comparisons from saved per-query outputs

## Reproduce the final statistical analysis

Install dependencies using `requirements.txt` in an isolated Python environment. From **this folder** run:

```bash
python final_statistics.py
```

The script reads the included classical and VQC per-query evaluation CSVs and writes the outputs under `results/final_statistical_analysis/`. **Run in a copy** if you wish to retain the deposited CSV files byte-for-byte: rerunning the stochastic tests can produce different Monte Carlo values depending on environment and RNG implementation.

## Limitations of this public subset

To avoid unverified redistribution of third-party SciFact text and associated derived material, this public subset **excludes cached document/query embeddings, candidate document tables, source metadata and document-level ranking traces**. Consequently, **the full end-to-end retrieval and VQC training pipeline cannot be rerun from this public subset alone**. Some scripts expect omitted `data/results/` inputs and original SciFact query/qrel files. Obtain SciFact from its official BEIR distribution, subject to the applicable terms. The separate author-held archive is the source of the saved candidate and feature artefacts.

The original archive's README notes that regenerating PCA features may produce small numerical differences across software environments. A fully independent end-to-end rerun has not been established. The preserved saved per-query outputs support verification of the final statistical comparisons.

## Publication and archive status

Manuscript: *Simulated Variational Quantum Circuits for Hybrid Scientific Document Re-ranking* (revised submission under review at *Scientific Reports*).

Zenodo reserved DOI: `10.5281/zenodo.24132662` (**unpublished draft; do not treat as a publicly available DOI**). The corresponding article citation and public archive link should be added only after publication/release.

Software licence: MIT, as stated in `LICENSE`; third-party data rights are separate.
