"""Fresh controlled classical experiment. No prior assistant-generated metrics are used."""
from pathlib import Path
import json, time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1] / 'data'
OUT=Path(__file__).resolve().parents[1] / 'results' / 'classical'; OUT.mkdir(parents=True,exist_ok=True)
SPLIT=ROOT/'splits/frozen_query_partitions.csv'
ca=pd.read_csv(ROOT/'results/hybrid_top20_candidates.csv'); qr=pd.read_csv(ROOT/'qrels/qrels.csv')
sp=pd.read_csv(SPLIT); qm=pd.read_csv(ROOT/'results/query_metadata.csv'); dm=pd.read_csv(ROOT/'results/document_metadata.csv')
qe=np.load(ROOT/'results/query_embeddings.npy'); de=np.load(ROOT/'results/document_embeddings.npy')
qidx={int(k):i for i,k in enumerate(qm.query_id)}; didx={int(k):i for i,k in enumerate(dm.document_id)}
assert len(ca)==6000 and ca.groupby('query_id').size().eq(20).all()
assert set(ca.query_id)==set(sp.query_id)
qrels={int(q):set(map(int,g.document_id)) for q,g in qr.groupby('query_id')}
parts={int(r.query_id):r.partition for r in sp.itertuples()}
# Each candidate's original within-query position is preserved for deterministic ties.
ca['original_position']=ca.groupby('query_id').cumcount()
qix=ca.query_id.map(qidx).to_numpy(); dix=ca.document_id.map(didx).to_numpy()
assert np.isfinite(qix).all() and np.isfinite(dix).all()
Q=qe[qix.astype(int)]; D=de[dix.astype(int)]
X=np.concatenate([Q,D,np.abs(Q-D),Q*D],axis=1)
assert X.shape==(6000,1536)
# Pair construction exactly matches original: first positive and first unmarked candidate.
pairs=[]
for q,g in ca.groupby('query_id',sort=False):
    pos=g[g.document_id.isin(qrels[int(q)])]; neg=g[~g.document_id.isin(qrels[int(q)])]
    if len(pos) and len(neg):pairs.append((int(q),int(pos.index[0]),int(neg.index[0])))
assert len(pairs)==247
train=[p for p in pairs if parts[p[0]]=='train'];val=[p for p in pairs if parts[p[0]]=='validation'];test=[p for p in pairs if parts[p[0]]=='test']
assert len(train)==151 and len(val)==44 and len(test)==52
train_idx=[i for _,p,n in train for i in (p,n)]
sc=StandardScaler().fit(X[train_idx]);Xp=sc.transform(X)
pca=PCA(n_components=6,random_state=42,svd_solver='randomized').fit(Xp[train_idx]);Z=pca.transform(Xp)
# Save only safe numerical objects; model output is not treated as VQC evidence.
np.savez_compressed(ROOT/'results/train_only_6component_features.npz',features=Z,query_ids=ca.query_id.to_numpy(),document_ids=ca.document_id.to_numpy(),training_pair_rows=np.array(train),validation_pair_rows=np.array(val),test_pair_rows=np.array(test),scaler_mean=sc.mean_,scaler_scale=sc.scale_,pca_components=pca.components_,pca_mean=pca.mean_)
idx=np.array(train_idx);y=np.tile([1,0],len(train))
models={'LogisticRegression':LogisticRegression(C=1.0,max_iter=2000,random_state=42)}
for seed in [7,21,42]:models[f'MLP_seed{seed}']=MLPClassifier(hidden_layer_sizes=(16,),activation='relu',solver='adam',max_iter=500,random_state=seed)
rows=[];scores={}; pair_rows=[]
for name,model in models.items():
    start=time.perf_counter();model.fit(Z[idx],y);seconds=time.perf_counter()-start
    s=model.predict_proba(Z)[:,1];scores[name]=s
    for partition,ps in [('train',train),('validation',val),('test',test)]:
        good=np.array([s[p]>s[n] for _,p,n in ps]);ties=np.array([s[p]==s[n] for _,p,n in ps]);
        pair_rows.append(dict(model=name,partition=partition,n_pairs=len(ps),pairwise_accuracy=float(np.mean(good+0.5*ties)),train_seconds=seconds))
    rows.append(dict(model=name,train_seconds=seconds,training_rows=len(idx)))
# Metric convention: full candidate recall at k, MRR@10, MAP@10, nDCG@5/10.
def metrics(ids,relevant):
    ids=list(map(int,ids));rel=set(relevant);hits=[int(d in rel) for d in ids]
    out={f'Recall@{k}':sum(hits[:k])/len(rel) if rel else 0 for k in (3,5,10,20)}
    first=next((i+1 for i,h in enumerate(hits[:10]) if h),None)
    out['MRR@10']=1/first if first else 0
    out['MAP@10']=sum(sum(hits[:i+1])/(i+1) for i,h in enumerate(hits[:10]) if h)/len(rel) if rel else 0
    for k in (5,10):
        dcg=sum(h/np.log2(i+2) for i,h in enumerate(hits[:k]));ideal=sum(1/np.log2(i+2) for i in range(min(k,len(rel))))
        out[f'nDCG@{k}']=dcg/ideal if ideal else 0
    return out
# Precommit score-fusion weight to original alpha=.8, not tuned using held-out data.
def norm(v):
    v=np.asarray(v,float);span=v.max()-v.min();return (v-v.min())/span if span>0 else np.full(len(v),0.5)
per=[]
for q,g in ca.groupby('query_id',sort=False):
    q=int(q);ix=g.index.to_numpy();h=g.hybrid_score.to_numpy();ids=g.document_id.to_numpy();rel=qrels[q]
    choices={'Hybrid':h}
    for name,s in scores.items():
        choices[name]=s[ix]
        choices['Hybrid+'+name]=.8*norm(h)+.2*norm(s[ix])
    for name,s in choices.items():
        # Stable descending score, ties retain original hybrid order.
        order=np.argsort(-s,kind='stable');met=metrics(ids[order],rel)
        per.append(dict(query_id=q,partition=parts[q],method=name,**met))
res=pd.DataFrame(per);res.to_csv(OUT/'per_query_classical_metrics.csv',index=False)
summary=res.groupby(['partition','method'],as_index=False).agg({c:'mean' for c in res.columns if c.startswith(('Recall','MRR','MAP','nDCG'))})
summary.to_csv(OUT/'classical_metrics_summary.csv',index=False)
pd.DataFrame(pair_rows).to_csv(OUT/'pairwise_diagnostics.csv',index=False)
manifest={'original_repository':'VQC-RAG-1.0-qmi-submission.zip','frozen_split_seed':20261106,'query_partition_counts':sp.partition.value_counts().to_dict(),'eligible_pair_counts':{'train':len(train),'validation':len(val),'test':len(test)},'feature_dimensions':1536,'pca_components':6,'pca_train_rows':len(train_idx),'explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),'fusion_alpha_hybrid':0.8,'ranking_ties':'stable original candidate order','metric_cutoffs':'MRR@10 MAP@10; other cutoffs explicitly labelled','models':list(models),'warning':'No VQC model trained. These classical controls are not equivalent to a pairwise hinge objective; MLP uses classification cross-entropy. No external test corpus.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('PCA explained variance:',manifest['explained_variance_ratio_sum'])
print('TRAIN',summary[summary.partition=='train'].to_string(index=False))
print('VALIDATION',summary[summary.partition=='validation'].to_string(index=False))
print('TEST',summary[summary.partition=='test'].to_string(index=False))
