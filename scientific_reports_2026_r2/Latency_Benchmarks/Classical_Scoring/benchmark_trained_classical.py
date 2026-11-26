"""Controlled cached-feature scoring benchmark for trained classical baselines.
Run from any directory: python benchmark_trained_classical.py --repo D:\\QML\\VQC_RAG_Revised_Reproducible
Same 30 queries as Stage 8 VQC benchmark; 20 candidates/query; 5 repeats.
"""
import argparse, json, platform, sys, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import sklearn

p=argparse.ArgumentParser()
p.add_argument('--repo',required=True)
p.add_argument('--repeats',type=int,default=5)
p.add_argument('--warmup',type=int,default=2)
p.add_argument('--out',default='results/latency_classical')
p.add_argument('--rank-weights',default=None,help='Path to model_numeric_weights.npz, defaults to script folder')
a=p.parse_args()
assert a.repeats>=3 and a.warmup>=1
root=Path(a.repo).resolve();out=root/a.out;out.mkdir(parents=True,exist_ok=True)
feature_path=root/'data/results/train_only_6component_features.npz'
with np.load(feature_path,allow_pickle=False) as z:
 X=np.asarray(z['features'],dtype=np.float64)
 qids=np.asarray(z['query_ids']);train=np.asarray(z['training_pair_rows'])
assert X.shape==(6000,6) and len(train)==151
# Same fitted training specification as Stage 3; no validation or test data used in fitting.
idx=np.array([i for row in train for i in (int(row[1]),int(row[2]))],dtype=int)
y=np.tile([1,0],len(train))
mlp=MLPClassifier(hidden_layer_sizes=(16,),activation='relu',solver='adam',max_iter=500,random_state=42)
mlp.fit(X[idx],y)
# Plain numerical RankSVM parameters from the already fitted Stage 7 model (no pickle).
weight_path=Path(a.rank_weights) if a.rank_weights else Path(__file__).with_name('model_numeric_weights.npz')
with np.load(weight_path,allow_pickle=False) as z:
 coef=np.asarray(z['coef'],dtype=np.float64).reshape(-1)
 intercept=float(np.asarray(z['intercept'],dtype=np.float64).reshape(-1)[0])
 selected_C=float(np.asarray(z['C']).reshape(-1)[0])
assert coef.shape==(6,)
# Inference functions each score 20 candidates; model loading and preprocessing excluded.
def score_mlp(A):return mlp.predict_proba(A)[:,1]
def score_rank(A):return A@coef+intercept
indices={q:np.flatnonzero(qids==q) for q in np.unique(qids)}
assert len(indices)==300 and all(len(v)==20 for v in indices.values())
selected=list(sorted(indices,key=str))[:30]
for _ in range(a.warmup):
 for q in selected[:3]:
  A=X[indices[q]];score_mlp(A);score_rank(A)
rows=[]
for repeat in range(a.repeats):
 for q in selected:
  A=X[indices[q]]
  for name,fn in [('MLP_trained_seed42',score_mlp),('RankSVM_trained',score_rank)]:
   start=time.perf_counter_ns();v=fn(A);ms=(time.perf_counter_ns()-start)/1e6
   assert v.shape==(20,) and np.isfinite(v).all()
   rows.append(dict(repeat=repeat,query_id=q,method=name,candidate_count=20,milliseconds=ms))
df=pd.DataFrame(rows);df.to_csv(out/'raw_per_query_latency.csv',index=False)
summary=df.groupby('method').milliseconds.agg(['count','mean','median','std','min','max']).reset_index()
summary.to_csv(out/'latency_summary.csv',index=False)
# Save training-only model numerical parameters and deterministic score signatures for audit.
np.savez_compressed(out/'mlp_seed42_numeric_weights.npz',**{f'coef_{i}':v for i,v in enumerate(mlp.coefs_)},**{f'intercept_{i}':v for i,v in enumerate(mlp.intercepts_)})
full_mlp=score_mlp(X);full_rank=score_rank(X)
pd.DataFrame({'query_id':qids,'mlp_seed42_score':full_mlp,'rank_svm_score':full_rank}).to_csv(out/'classical_document_scores.csv',index=False)
manifest={'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,'sklearn':sklearn.__version__,
 'feature_sha256':hashlib.sha256(feature_path.read_bytes()).hexdigest(),
 'rank_weights_sha256':hashlib.sha256(weight_path.read_bytes()).hexdigest(),
 'MLP_spec':'MLPClassifier(hidden_layer_sizes=(16,),activation=relu,solver=adam,max_iter=500,random_state=42)',
 'MLP_train_rows':int(len(idx)),'RankSVM_selected_C':selected_C,
 'candidate_count_per_query':20,'queries_benchmarked':len(selected),'repeats':a.repeats,'warmup_passes':a.warmup,
 'scope':'cached six-component PCA feature scoring only; excludes feature computation, embeddings, retrieval, disk I/O and model fitting/loading',
 'query_selection':'first 30 sorted query IDs, matching Stage 8 VQC script',
 'warning':'Cross-method time ratios depend on implementation and device; VQC uses PennyLane QNode per candidate and MLP uses vectorised sklearn predict_proba. Report as measured implementation latency, not intrinsic quantum advantage.'}
(out/'benchmark_manifest.json').write_text(json.dumps(manifest,indent=2))
print(summary.to_string(index=False));print('Saved to:',out)
