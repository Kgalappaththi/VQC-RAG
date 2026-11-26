Frozen NFCorpus evaluation

Install dependencies from requirements.txt in a suitable Python environment. Obtain NFCorpus from https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip or the BEIR distribution. Preserve original corpus order. Obtain the all-MiniLM-L6-v2 encoder snapshot 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 from Hugging Face. The encoder gate checks archived query embeddings; the original SciFact encoder commit was not independently established.

From this directory:
python run.py --encoder /path/to/snapshot --dataset /path/to/nfcorpus --out reproduced_results

Use a new output directory. PROTOCOL.txt specifies the frozen settings. Do not tune on NFCorpus. VQC, MLP-BCE and MLP-Hinge output labels denote weighted fusion. Hinge reference scores were generated from recovered weights. Timing is descriptive cached-feature scoring, not end-to-end latency. Original results are in ../NFCorpus_Independent_Validation_Evidence. The frozen evaluation was successfully rerun on Windows; see ../FINAL_NFCORPUS_VERIFICATION.txt.
