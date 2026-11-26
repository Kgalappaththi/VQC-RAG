import numpy as np,pandas as pd,torch,zipfile,io,argparse
from pathlib import Path
from torch import nn
root=Path(__file__).resolve().parent
base=root.parent
p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,default=base/"VQC_SciFact_Reproducibility_v1"); p.add_argument("--out",type=Path,required=True); args=p.parse_args(); root=args.out.resolve(); root.mkdir(parents=True,exist_ok=True)
def read(p):return pd.read_csv(args.repo/p)
a=np.load(args.repo/'data/results/train_only_6component_features.npz'); X=a['features']; tr=a['training_pair_rows']; ca=read('data/results/hybrid_top20_candidates.csv'); sp=read('data/splits/frozen_query_partitions.csv').set_index('query_id').partition
qr=pd.read_csv(Path(__file__).with_name('qrels.csv')); rel=qr.groupby('query_id').document_id.apply(set).to_dict()
assert X.shape==(6000,6) and np.isfinite(X).all() and len(tr)==151
assert np.array_equal(a['query_ids'],ca.query_id) and np.array_equal(a['document_ids'],ca.document_id)
for key,part,count in [('training_pair_rows','train',151),('validation_pair_rows','validation',44),('test_pair_rows','test',52)]:
 p=a[key]; assert len(p)==count
 for q,i,j in p:
  assert sp[q]==part and ca.query_id.iloc[i]==q==ca.query_id.iloc[j] and ca.document_id.iloc[i] in rel[q] and ca.document_id.iloc[j] not in rel[q]
torch.set_num_threads(1); torch.use_deterministic_algorithms(True); T=torch.tensor(X,dtype=torch.float64); rows=[]; hist=[]
def norm(x):return (x-x.min())/np.ptp(x) if np.ptp(x)>0 else np.full(len(x),.5)
for seed in [7,21,42]:
 torch.manual_seed(seed); model=nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,1),nn.Sigmoid()).double(); opt=torch.optim.Adam(model.parameters(),lr=.05)
 for epoch in range(20):
  opt.zero_grad(); loss=(.1-model(T[tr[:,1]])+model(T[tr[:,2]])).clamp(min=0).mean(); loss.backward(); opt.step(); hist.append([seed,epoch+1,loss.item()])
 with torch.no_grad(): s=model(T).numpy().ravel()
 np.savez(root/f'model_seed{seed}.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
 for q,g in ca.groupby('query_id',sort=False):
  h=g.hybrid_score.to_numpy(); v=s[g.index]; ids=g.document_id.to_numpy(); r=rel[q]
  for method,score in [('Hybrid',h),('MLP-Hinge',v),('Hybrid+MLP-Hinge',.8*norm(h)+.2*norm(v))]:
   hits=np.isin(ids[np.argsort(-score,kind='stable')],list(r)); m={f'Recall@{k}':hits[:k].sum()/len(r) for k in [3,5,10,20]}; m.update({f'nDCG@{k}':sum(hits[:k]/np.log2(np.arange(k)+2))/sum(1/np.log2(np.arange(min(k,len(r)))+2)) for k in [5,10]}); ix=np.flatnonzero(hits[:10]); m['MRR@10']=1/(ix[0]+1) if len(ix) else 0; m['MAP@10']=sum(np.cumsum(hits[:10])[ix]/(ix+1))/len(r); rows.append(dict(seed=seed,query_id=q,partition=sp[q],method=method,**m))
df=pd.DataFrame(rows); df.to_csv(root/'per_query.csv',index=False); pd.DataFrame(hist,columns=['seed','epoch','loss_before_update']).to_csv(root/'training_history.csv',index=False); agg=df.groupby(['partition','method','seed']).mean(numeric_only=True).drop(columns='query_id'); agg.to_csv(root/'per_seed.csv'); print(agg['nDCG@10'].to_string()); print(agg.groupby(level=[0,1])['nDCG@10'].agg(['mean','std']))
