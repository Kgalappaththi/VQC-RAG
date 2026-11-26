# Changelog

## v2.0.0 – Scientific Reports Second-round Revision

### Added
- MLP-Hinge baseline for comparison under a similar training objective.
- Independent cross-domain evaluation using NFCorpus.
- Frozen-configuration evaluation on NFCorpus.
- Query-level paired statistical analysis and Holm correction.
- Classical and VQC computational latency benchmarks.
- Diagnostic experiments and supporting analysis scripts.
- Second-round experimental code in `scientific_reports_2026_r2/`.

### Updated
- Experimental methodology and reproducibility documentation.
- README and citation metadata.
- Interpretation of the exploratory SciFact results and independent NFCorpus evaluation.
- Documentation of experimental limitations.

### Reproducibility Archive
- Zenodo archive version: 1.0.0
- DOI: https://doi.org/10.5281/zenodo.24132662
- The complete reproducibility package is available on Zenodo.

### Previous Versions
- Earlier experimental code, results, branches and tags are retained to preserve the research development history.

---

## v3.1-baseline-final

- Final validated classical retrieval baseline completed.
- BM25, Dense, and Hybrid retrieval evaluated.
- Extended metrics recorded: Recall@3, Recall@5, Recall@10, MRR, MAP, nDCG@5, nDCG@10, and latency.
- Repository ready for VQC re-ranking implementation.
