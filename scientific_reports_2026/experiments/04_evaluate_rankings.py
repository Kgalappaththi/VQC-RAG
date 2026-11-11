from pathlib import Path
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--run',required=True,help='VQC output folder name under results/')
parser.add_argument('--out',default=None,help='Audit folder name under results/')
a=parser.parse_args()
BASE=Path(__file__).resolve().parents[1]
import numpy as np,pandas as pd,json
R=BASE/'data'; O=BASE/'results'/(a.out or ('audit_'+a.run)); O.mkdir(parents=True,exist_ok=True)
ca=pd.read_csv(R/'results/hybrid_top20_candidates.csv');qr=pd.read_csv(R/'qrels/qrels.csv');sp=pd.read_csv(R/'splits/frozen_query_partitions.csv');v=pd.read_csv(BASE/'results'/a.run/'document_scores.csv');cl=pd.read_csv(BASE/'results/classical/per_query_classical_metrics.csv'); manifest=json.loads((BASE/'results'/a.run/'run_manifest.json').read_text())
assert len(v)==len(ca)==6000
assert not v.duplicated(['query_id','document_id']).any()
ca=ca.merge(v,on=['query_id','document_id'],how='left',validate='one_to_one');assert ca.vqc_score.notna().all()
part=dict(zip(sp.query_id,sp.partition));rel={int(q):set(map(int,g.document_id)) for q,g in qr.groupby('query_id')}
def norm(x):
 x=np.asarray(x,float); d=x.max()-x.min();return (x-x.min())/d if d>0 else np.full(len(x),.5)
def metric(ids,rels):
 h=[int(int(d) in rels) for d in ids];o={f'Recall@{k}':sum(h[:k])/len(rels) for k in [3,5,10,20]};p=next((i+1 for i,z in enumerate(h[:10]) if z),None);o['MRR@10']=1/p if p else 0;o['MAP@10']=sum(sum(h[:i+1])/(i+1) for i,z in enumerate(h[:10]) if z)/len(rels)
 for k in (5,10):
  dcg=sum(z/np.log2(i+2) for i,z in enumerate(h[:k]));ideal=sum(1/np.log2(i+2) for i in range(min(k,len(rels))));o[f'nDCG@{k}']=dcg/ideal if ideal else 0
 return o
rows=[];trace=[]
for q,g in ca.groupby('query_id',sort=False):
 q=int(q);g=g.reset_index(drop=True);h=g.hybrid_score.to_numpy();s=g.vqc_score.to_numpy();ids=g.document_id.to_numpy();hr=pd.Series(h).rank(ascending=False,method='first').to_numpy();vr=pd.Series(s).rank(ascending=False,method='first').to_numpy();rrf=1/(60+hr)+1/(60+vr)
 methods={'Hybrid':h,'VQC':s,'Hybrid+VQC_weighted_0.8':.8*norm(h)+.2*norm(s),'Hybrid+VQC_RRF_k60':rrf}
 for name,score in methods.items():
  ix=np.argsort(-score,kind='stable');rows.append(dict(query_id=q,partition=part[q],method=name,**metric(ids[ix],rel[q])))
 for j,d in enumerate(ids):trace.append(dict(query_id=q,partition=part[q],document_id=int(d),hybrid_score=float(h[j]),vqc_score=float(s[j]),hybrid_rank=int(hr[j]),vqc_rank=int(vr[j]),rrf_score=float(rrf[j]),rrf_rank=int(pd.Series(rrf).rank(ascending=False,method='first').iloc[j])))
res=pd.DataFrame(rows);res.to_csv(O/'vqc_per_query_metrics.csv',index=False);pd.DataFrame(trace).to_csv(O/'rrf_document_audit.csv',index=False)
sumry=res.groupby(['partition','method'],as_index=False).mean(numeric_only=True).drop(columns='query_id');sumry.to_csv(O/'vqc_metrics_summary.csv',index=False)
# Same 60 test queries for all methods; paired bootstrap and randomisation, with fixed RNG
base=res[(res.partition=='test') & (res.method=='Hybrid')].sort_values('query_id');b=base['nDCG@10'].to_numpy(); stats=[]
for method in ['VQC','Hybrid+VQC_weighted_0.8','Hybrid+VQC_RRF_k60']:
 other=res[(res.partition=='test') & (res.method==method)].sort_values('query_id');assert np.array_equal(base.query_id,other.query_id);d=other['nDCG@10'].to_numpy()-b;rng=np.random.default_rng(20261106);means=np.mean(d[rng.integers(0,len(d),(20000,len(d)))],axis=1);p=(1+np.count_nonzero(np.abs(np.mean(d*rng.choice([-1,1],(20000,len(d))),axis=1))>=abs(d.mean())))/20001
 stats.append(dict(method=method,mean_delta_nDCG10=float(d.mean()),bootstrap_95_low=float(np.quantile(means,.025)),bootstrap_95_high=float(np.quantile(means,.975)),randomisation_p_unadjusted=float(p)))
pd.DataFrame(stats).to_csv(O/'vqc_paired_uncertainty.csv',index=False)
print('VALIDATION');print(sumry[sumry.partition=='validation'].to_string(index=False));print('TEST');print(sumry[sumry.partition=='test'].to_string(index=False));print('PAIRED');print(pd.DataFrame(stats).to_string(index=False));print('manifest',manifest)
