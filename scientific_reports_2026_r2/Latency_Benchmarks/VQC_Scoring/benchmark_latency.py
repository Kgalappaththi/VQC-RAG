"""CPU cached-feature reranking inference benchmark, same 20 candidates/query.
Run in repository root: python path/to/benchmark_latency.py --repo . --repeats 5
Measures scoring only; excludes embedding generation, retrieval, I/O, preprocessing.
No training, no test-set tuning, no relevance labels accessed.
"""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--repeats',type=int,default=5);p.add_argument('--warmup',type=int,default=2);p.add_argument('--out',default='results/latency');a=p.parse_args()
assert a.repeats>=3 and a.warmup>=1
repo=Path(a.repo).resolve();out=repo/a.out;out.mkdir(parents=True,exist_ok=True)
with np.load(repo/'data/results/train_only_6component_features.npz',allow_pickle=False) as z:
    features=np.asarray(z['features'],dtype=np.float64); qids=np.asarray(z['query_ids'])
with np.load(repo/'results/seed42/checkpoint.npz',allow_pickle=False) as z: weights=np.asarray(z['weights'],dtype=np.float64)
assert features.shape==(6000,6) and weights.shape==(4,6,2)
# Fixed six-qubit depth-four original circuit
try:
 import pennylane as qml
except ImportError as e: raise SystemExit('PennyLane is required: pip install pennylane==0.45.0') from e
dev=qml.device('default.qubit',wires=6)
@qml.qnode(dev)
def circuit(x,w):
    for i in range(6):qml.RY(x[i],wires=i)
    for layer in range(4):
        for i in range(6):
            qml.RY(w[layer,i,0],wires=i)
            qml.RZ(w[layer,i,1],wires=i)
        for i in range(5):qml.CNOT(wires=[i,i+1])
    return tuple(qml.expval(qml.PauliZ(i)) for i in range(6))
def quantum_score(X):
    return np.array([(np.mean(circuit(x,weights))+1)/2 for x in X],dtype=float)
# Lightweight classical cached-feature scorer, explicitly NOT a trained MLP/RankSVM latency comparison.
def linear_reference(X):return X@np.ones(6,dtype=np.float64)
indices={q:np.flatnonzero(qids==q) for q in np.unique(qids)}
assert len(indices)==300 and all(len(ix)==20 for ix in indices.values())
# Fixed query subset across all methods, no labels consulted
selected=list(sorted(indices,key=str))[:30]
for _ in range(a.warmup):
 for q in selected[:3]: quantum_score(features[indices[q]]);linear_reference(features[indices[q]])
rows=[]
for repeat in range(a.repeats):
 for q in selected:
    X=features[indices[q]]
    for name,func in [('VQC_6q_depth4',quantum_score),('linear_reference_untrained',linear_reference)]:
        start=time.perf_counter_ns();scores=func(X);elapsed=(time.perf_counter_ns()-start)/1e6
        assert len(scores)==20 and np.all(np.isfinite(scores))
        rows.append({'repeat':repeat,'query_id':q,'method':name,'candidate_count':20,'milliseconds':elapsed})
df=pd.DataFrame(rows);df.to_csv(out/'raw_per_query_latency.csv',index=False)
summary=df.groupby('method').milliseconds.agg(['count','mean','median','std','min','max']).reset_index();summary.to_csv(out/'latency_summary.csv',index=False)
manifest={'platform':platform.platform(),'python':sys.version,'pennylane':qml.__version__,'cpu_only':True,'candidate_count_per_query':20,'queries_benchmarked':len(selected),'repeats':a.repeats,'warmup_passes':a.warmup,'scope':'scoring cached 6-component features only; excludes first-stage retrieval, feature engineering, disk I/O, embeddings and model loading','warning':'Linear reference is untrained and not a fair performance baseline. For a fair MLP/RankSVM latency comparison, load their fitted weights and score the same cached features.'}
(out/'benchmark_manifest.json').write_text(json.dumps(manifest,indent=2))
print(summary.to_string(index=False));print('Saved:',out)
