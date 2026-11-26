import argparse, json, hashlib, time, platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parent
SEEDS=(7,21,42)
def norm(x,constant=.5):
    return (x-x.min())/np.ptp(x) if np.ptp(x)>0 else np.full(len(x),constant)
def load(name):
    with np.load(ROOT/'models'/name,allow_pickle=False) as a: return dict(a)
def transform(q,d,p):
    q=np.broadcast_to(q,d.shape); x=np.concatenate([q,d,abs(q-d),q*d],axis=1)
    return ((x-p['scaler_mean'])/p['scaler_scale']-p['pca_mean'])@p['pca_components'].T
def mlp(x,p,hinge=False):
    if hinge: return expit(np.maximum(0,x@p['0.weight'].T+p['0.bias'])@p['2.weight'].T+p['2.bias']).ravel()
    return expit(np.maximum(0,x@p['coef_0']+p['intercept_0'])@p['coef_1']+p['intercept_1']).ravel()
def metrics(ids,rel):
    grades=np.array([rel.get(i,0) for i in ids]); ideal=np.array(sorted(rel.values(),reverse=True)[:10]); discount=lambda n:np.log2(np.arange(n)+2)
    # BEIR/trec_eval nDCG uses linear graded gains.
    dcg=sum(grades[:10]/discount(min(10,len(grades)))); denom=sum(ideal/discount(len(ideal))); hits=np.flatnonzero(grades[:10]>0)
    return {'nDCG@10':dcg/denom if denom else 0.,'Recall@20':sum(grades[:20]>0)/sum(v>0 for v in rel.values()),'MRR@10':1/(hits[0]+1) if len(hits) else 0.}
def stats(df):
    a=df.groupby(['query','method'])['nDCG@10'].mean().unstack(); rows=[]
    for base in ['Hybrid','MLP-BCE','MLP-Hinge']:
        d=(a['VQC']-a[base]).to_numpy(); rng=np.random.default_rng(20261106); n=len(d); b=d[rng.integers(n,size=(20000,n))].mean(1); r=(d*rng.choice([-1,1],size=(20000,n))).mean(1); p=(1+sum(abs(r)>=abs(d.mean())))/20001
        rows.append(dict(comparison='VQC vs '+base,delta=d.mean(),low=np.quantile(b,.025),high=np.quantile(b,.975),p=p))
    t=pd.DataFrame(rows); ix=np.argsort(t.p.to_numpy()); t.loc[ix,'holm']=np.minimum(1,np.maximum.accumulate(t.p.to_numpy()[ix]*np.arange(3,0,-1))); return t

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--encoder',required=True,help='Local immutable encoder snapshot directory'); ap.add_argument('--dataset',required=True,help='Extracted official NFCorpus directory'); ap.add_argument('--out',required=True); a=ap.parse_args()
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import pennylane as qml
    from beir.datasets.data_loader import GenericDataLoader
    import importlib.metadata as md
    enc=Path(a.encoder); assert enc.is_dir(), 'Use local snapshot, not floating model name'
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    files=[*sorted((ROOT/'models').glob('*.npz')),Path(__file__),ROOT/'PROTOCOL.txt',*sorted(p for p in enc.rglob('*') if p.is_file()),*[Path(a.dataset)/p for p in ['corpus.jsonl','queries.jsonl','qrels/test.tsv']]]
    manifest={'hashes':{str(p):sha(p) for p in files},'platform':platform.platform(),'packages':{p:md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','sentence-transformers','rank-bm25','pennylane','beir']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    model=SentenceTransformer(str(enc),device='cpu')
    # Gate on archived SciFact query embeddings before reading external qrels.
    ref=load('encoder_reference.npz'); check=model.encode(ref['queries'].tolist(),show_progress_bar=False); diff=np.max(abs(check-ref['embeddings'])); print('Encoder max difference',diff,flush=True)
    if not np.allclose(check,ref['embeddings'],rtol=1e-4,atol=1e-5): raise RuntimeError('Encoder verification failed; stop before external evaluation')
    dev=qml.device('default.qubit',wires=6)
    @qml.qnode(dev)
    def circuit(x,w):
        for i in range(6): qml.RY(x[i],wires=i)
        for l in range(4):
            for i in range(6): qml.RY(w[l,i,0],wires=i); qml.RZ(w[l,i,1],wires=i)
            for i in range(5): qml.CNOT(wires=[i,i+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(6)]
    def vqc(x,w): return np.array([(np.mean(circuit(z,w))+1)/2 for z in x])
    p=load('preprocess.npz'); models={s:(load(f'vqc{s}.npz'),load(f'bce{s}.npz'),load(f'hinge{s}.npz')) for s in SEEDS}
    ref=load('scorer_reference.npz')
    for s,(v,b,h) in models.items():
        for name,score in [('vqc',vqc(ref['x'],v['weights'])),('bce',mlp(ref['x'],b)),('hinge',mlp(ref['x'],h,True))]:
            if not np.allclose(score,ref[f'{name}{s}'],atol=1e-5,rtol=1e-5): raise RuntimeError(f'Scorer gate failed {name} {s}')
    corpus,queries,qrels=GenericDataLoader(data_folder=a.dataset).load(split='test'); assert len(corpus)==3633 and len(queries)==323 and set(queries)==set(qrels)
    ids=np.array(list(corpus)); texts=[' '.join(str(corpus[i]['text']).split()) for i in ids]; D=model.encode(texts,show_progress_bar=True); Q=model.encode(list(queries.values()),show_progress_bar=True); bm=BM25Okapi([t.lower().split() for t in texts]); rows=[]; ranks=[]
    for j,(qid,q) in enumerate(queries.items()):
        h=.5*norm(bm.get_scores(q.lower().split()),0)+.5*norm(cosine_similarity(Q[j:j+1],D)[0],0); ix=np.argsort(-h,kind='stable')[:20]; hh=h[ix]; x=transform(Q[j],D[ix],p); rel=qrels[qid]; assert any(v>0 for v in rel.values())
        for s,(v,b,k) in models.items():
            for name,fn in [('Hybrid',lambda:hh),('VQC',lambda:vqc(x,v['weights'])),('MLP-BCE',lambda:mlp(x,b)),('MLP-Hinge',lambda:mlp(x,k,True))]:
                start=time.perf_counter(); raw=fn(); ms=(time.perf_counter()-start)*1000; score=raw if name=='Hybrid' else .8*norm(hh)+.2*norm(raw); order=np.argsort(-score,kind='stable'); ranked=ids[ix][order]; rows.append(dict(query=qid,seed=s,method=name,scoring_ms=ms,**metrics(ranked,rel))); ranks.extend(dict(query=qid,seed=s,method=name,rank=r+1,doc=d,score=score[order[r]]) for r,d in enumerate(ranked))
        if (j+1)%25==0: print(j+1,flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/'per_query.csv',index=False); pd.DataFrame(ranks).to_csv(out/'rankings.csv',index=False); cols=['nDCG@10','Recall@20','MRR@10','scoring_ms']; per=df.groupby(['method','seed'])[cols].mean(); per.to_csv(out/'per_seed.csv'); per.groupby('method').agg(['mean','std']).to_csv(out/'summary.csv'); stats(df).to_csv(out/'paired.csv',index=False)
if __name__=='__main__': main()
