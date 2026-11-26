"""Reproduce original 6-qubit depth-4 VQC with frozen, train-only PCA features.
Run in directory containing train_only_6component_features.npz and per_query_classical_metrics.csv.
No test-set tuning. Unjudged candidates used as weak negatives (original protocol).
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

parser=argparse.ArgumentParser()
parser.add_argument('--epochs',type=int,default=20)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--lr',type=float,default=.05)
parser.add_argument('--margin',type=float,default=.1)
parser.add_argument('--out',default='vqc_seed42')
a=parser.parse_args()
assert a.epochs>0 and a.lr>0
base=Path(__file__).resolve().parents[1]
out=base/'results'/a.out; out.mkdir(parents=True,exist_ok=True)
if (out/'run_manifest.json').exists(): raise SystemExit('Existing completed run: choose a different --out directory to avoid overwriting results')
with np.load(base/'data/results/train_only_6component_features.npz',allow_pickle=False) as data:
    Z=np.asarray(data['features'],dtype=float)
    qids=np.asarray(data['query_ids']);dids=np.asarray(data['document_ids'])
    tr=np.asarray(data['training_pair_rows']);va=np.asarray(data['validation_pair_rows']);te=np.asarray(data['test_pair_rows'])
assert Z.shape==(6000,6) and len(tr)==151 and len(va)==44 and len(te)==52
assert not (set(tr[:,0]) & set(va[:,0]) or set(tr[:,0]) & set(te[:,0]) or set(va[:,0]) & set(te[:,0]))
# Original circuit: RY data encoding, RY/RZ per layer, linear CNOT chain, mean Z readout.
dev=qml.device('default.qubit',wires=6)
@qml.qnode(dev,interface='autograd')
def circuit(x,w):
    for i in range(6):qml.RY(x[i],wires=i)
    for layer in range(4):
        for i in range(6):
            qml.RY(w[layer,i,0],wires=i)
            qml.RZ(w[layer,i,1],wires=i)
        for i in range(5):qml.CNOT(wires=[i,i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(6)]
def score(x,w):return (pnp.mean(pnp.stack(circuit(x,w)))+1)/2
rng=np.random.RandomState(a.seed)
w=pnp.array(rng.normal(0,.1,(4,6,2)),requires_grad=True)
opt=qml.AdamOptimizer(stepsize=a.lr)
# Full batch objective, identical to original src/quantum/trainer.py.
def loss(weights):
    total=0.
    for _,p,n in tr:
        total=total+pnp.maximum(0.,a.margin-score(Z[int(p)],weights)+score(Z[int(n)],weights))
    return total/len(tr)
history=[];start=time.perf_counter()
for epoch in range(1,a.epochs+1):
    w,l=opt.step_and_cost(loss,w)
    elapsed=time.perf_counter()-start
    history.append({'epoch':epoch,'loss_before_update':float(l),'elapsed_seconds':elapsed})
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    np.savez(out/'checkpoint.npz',weights=np.asarray(w),seed=a.seed,epochs_completed=epoch)
    print(f'Epoch {epoch}/{a.epochs}: loss before update={float(l):.6f}; elapsed={elapsed:.1f}s',flush=True)
# All 6000 candidates are scored under the same fitted model; no new embeddings needed.
sc=np.array([float(score(x,w)) for x in Z])
pd.DataFrame({'query_id':qids,'document_id':dids,'vqc_score':sc}).to_csv(out/'document_scores.csv',index=False)
def pair_accuracy(pairs):
    return float(np.mean([float(sc[int(p)]>sc[int(n)])+.5*float(sc[int(p)]==sc[int(n)]) for _,p,n in pairs]))
# Compare on the same queries using original Hybrid per-query metrics; VQC-only rank needs qrels.
# Source qrels are not in Stage3 ZIP, so do not invent relevance metrics here.
metrics={'seed':a.seed,'epochs':a.epochs,'n_qubits':6,'depth':4,'train_pairs':len(tr),'validation_pairs':len(va),'test_pairs':len(te),'training_seconds':time.perf_counter()-start,'pairwise_accuracy_train':pair_accuracy(tr),'pairwise_accuracy_validation':pair_accuracy(va),'pairwise_accuracy_test':pair_accuracy(te),'caveat':'Test pairwise accuracy is descriptive only; model selection must use validation. Non-qrels are unjudged, not confirmed irrelevant.'}
(out/'run_manifest.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2),flush=True)
print('Saved numerical scores and weights in',out.resolve())
