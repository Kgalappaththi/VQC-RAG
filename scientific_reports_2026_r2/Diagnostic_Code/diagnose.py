import argparse, sys, json, hashlib, zipfile, io
from pathlib import Path
import numpy as np
import pandas as pd

def pair_summary(x, pairs, score, method, seed, split):
    delta=score[pairs[:,1]]-score[pairs[:,2]]
    return dict(method=method,seed=seed,split=split,n=len(delta),wins=int(sum(delta>0)),ties=int(sum(delta==0)),losses=int(sum(delta<0)),accuracy=float(np.mean(delta>0)),mean_margin=float(delta.mean()))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase2',required=True); ap.add_argument('--scifact',required=True); ap.add_argument('--dataset',required=True); ap.add_argument('--encoder',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    sys.path.insert(0,str(Path(a.phase2).resolve())); import run as r
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from beir.datasets.data_loader import GenericDataLoader
    import pennylane as qml
    out=Path(a.out); out.mkdir(exist_ok=False,parents=True)
    z=zipfile.ZipFile(a.scifact); get=lambda s:z.read(next(n for n in z.namelist() if n.endswith(s)))
    f=np.load(io.BytesIO(get('train_only_6component_features.npz'))); X=f['features']; p=r.load('preprocess.npz')
    model=SentenceTransformer(a.encoder,device='cpu'); ref=r.load('encoder_reference.npz'); assert np.allclose(model.encode(ref['queries'].tolist()),ref['embeddings'],atol=1e-5,rtol=1e-4)
    dev=qml.device('default.qubit',wires=6)
    @qml.qnode(dev)
    def circuit(x,w):
        for i in range(6): qml.RY(x[i],wires=i)
        for l in range(4):
            for i in range(6): qml.RY(w[l,i,0],wires=i); qml.RZ(w[l,i,1],wires=i)
            for i in range(5): qml.CNOT(wires=[i,i+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(6)]
    def vqc(x,w): return np.array([(np.mean(circuit(t,w))+1)/2 for t in x])
    models={s:(r.load(f'vqc{s}.npz'),r.load(f'bce{s}.npz'),r.load(f'hinge{s}.npz')) for s in r.SEEDS}
    ref=r.load('scorer_reference.npz'); pairs=[]
    for s,(v,b,h) in models.items():
        for name,score in [('vqc',vqc(ref['x'],v['weights'])),('bce',r.mlp(ref['x'],b)),('hinge',r.mlp(ref['x'],h,True))]: assert np.allclose(score,ref[f'{name}{s}'],atol=1e-5,rtol=1e-5), (name,s)
        path='results/seed42/document_scores.csv' if s==42 else f'results/vqc_seed{s}/document_scores.csv'
        d=pd.read_csv(io.BytesIO(get(path))); keys=pd.MultiIndex.from_arrays([f['query_ids'],f['document_ids']]); score=d.set_index(['query_id','document_id']).vqc_score.reindex(keys).to_numpy(); assert np.isfinite(score).all()
        for name,sc in [('VQC',score),('MLP-BCE',r.mlp(X,b)),('MLP-Hinge',r.mlp(X,h,True))]:
            for split,key in [('train','training_pair_rows'),('validation','validation_pair_rows'),('test','test_pair_rows')]: pairs.append(pair_summary(X,f[key],sc,name,s,split))
    pd.DataFrame(pairs).to_csv(out/'scifact_pairwise.csv',index=False)
    corpus,queries,qrels=GenericDataLoader(data_folder=a.dataset).load(split='test'); assert len(corpus)==3633 and len(queries)==323
    ids=np.array(list(corpus)); texts=[' '.join(str(corpus[i]['text']).split()) for i in ids]; D=model.encode(texts,show_progress_bar=True); Q=model.encode(list(queries.values()),show_progress_bar=True); bm=BM25Okapi([t.lower().split() for t in texts]); rows=[]; sensitivity=[]; features=[]
    for j,(qid,q) in enumerate(queries.items()):
        h=.5*r.norm(bm.get_scores(q.lower().split()),0)+.5*r.norm(r.cosine_similarity(Q[j:j+1],D)[0],0); ix=np.argsort(-h,kind='stable')[:20]; hh=h[ix]; x=r.transform(Q[j],D[ix],p); rel=qrels[qid]; features.append(x)
        for s,(v,b,k) in models.items():
            for name,raw in [('VQC',vqc(x,v['weights'])),('MLP-BCE',r.mlp(x,b)),('MLP-Hinge',r.mlp(x,k,True))]:
                for i,doc in enumerate(ids[ix]): rows.append(dict(query=qid,doc=doc,seed=s,method=name,grade=rel.get(doc,0),hybrid=hh[i],raw=raw[i],**{f'pc{t+1}':x[i,t] for t in range(6)}))
                for w in [0,.1,.2,.3,.4,.5,1]:
                    score=(1-w)*r.norm(hh)+w*r.norm(raw); order=np.argsort(-score,kind='stable'); sensitivity.append(dict(query=qid,seed=s,method=name,weight=w,**r.metrics(ids[ix][order],rel)))
        if (j+1)%25==0: print(j+1,flush=True)
    pd.DataFrame(rows).to_csv(out/'raw_scores_features.csv',index=False); pd.DataFrame(sensitivity).to_csv(out/'exploratory_fusion.csv',index=False)
    summary=[]
    for label,arr in [('SciFact_all_candidates',X),('SciFact_training_pair_rows',X[f['training_pair_rows'][:,1:].ravel()]),('NFCorpus_candidates',np.concatenate(features))]:
        for i,col in enumerate(arr.T): summary.append(dict(dataset=label,component=i+1,n=len(col),mean=col.mean(),sd=col.std(),min=col.min(),max=col.max(),q01=np.quantile(col,.01),q99=np.quantile(col,.99),fraction_abs_gt_pi=np.mean(abs(col)>np.pi)))
    pd.DataFrame(summary).to_csv(out/'feature_summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'status':'exploratory post-hoc; no tuning or training','runner_sha256':hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest(),'scifact_zip_sha256':hashlib.sha256(Path(a.scifact).read_bytes()).hexdigest()},indent=2))
if __name__=='__main__': main()
