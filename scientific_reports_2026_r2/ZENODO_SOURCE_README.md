# Simulated Variational Quantum Circuits for Hybrid Scientific Document Re-ranking

Kalpana Galappaththi and Chathura Wattegama

This repository accompanies the manuscript of the same title. It contains source code, configurations, trained parameters, numerical features and evaluation evidence. SciFact findings are exploratory because test outcomes were inspected during development. The frozen NFCorpus evaluation is an independent cross-task assessment, not an exact replication; it did not reproduce the SciFact improvement. No quantum advantage is established.

## Repository guide

| Directory | Purpose |
|---|---|
| VQC_SciFact_Reproducibility_v1 | SciFact experiments, saved results and historical pipeline |
| Phase2_Execution | Frozen NFCorpus evaluation and model parameters |
| NFCorpus_Independent_Validation_Evidence | Original external evaluation results and protocol |
| NFCorpus_Preparation | Dataset preparation |
| MLP_Hinge | Objective-matched classical baseline |
| Latency_Benchmarks | Cached-feature scoring benchmarks |
| Diagnostic_Results | Post-hoc diagnostic analyses |
| verification | Historical reproducibility checks |

The frozen NFCorpus execution directory and SciFact root retain their tested names. Other directories use descriptive names; historical manifests retain original paths. Historical code and configurations are provenance records, not alternative final settings. Historical checksum manifests describe their original archives; SHA256SUMS.txt describes this release.

## Execution

Use Python 3.13.13 and the dependencies in the relevant requirements files. The archived environment records document the original versions. Obtain datasets from BEIR (https://github.com/beir-cellar/beir); raw corpus and query text and encoder weights are not distributed here. Preserve corpus order and identifiers. Text-dependent historical stages require upstream data and may require path adaptation.

From the repository root, verify integrity with `sha256sum -c SHA256SUMS.txt`. Run `python audit_nfcorpus.py` to check archived external statistics. For frozen NFCorpus execution, follow Phase2_Execution/README.txt. For SciFact statistics, change into VQC_SciFact_Reproducibility_v1 and run `python final_statistics.py` in a disposable copy. For the portable classical baseline, run `python MLP_Hinge/run.py --out hinge_verification` from the root. Use new output directories to preserve archived evidence.

The frozen NFCorpus pipeline was rerun on Windows and reproduced all archived ranking scores, ranking metrics and paired statistics. See FINAL_NFCORPUS_VERIFICATION.txt. This is not a claim of fresh-install or complete historical-pipeline verification.

## Licence and citation

See LICENSE and THIRD_PARTY_NOTICES.md. Cite this software using the version DOI assigned by Zenodo and the associated manuscript. Third-party material retains its upstream terms.
