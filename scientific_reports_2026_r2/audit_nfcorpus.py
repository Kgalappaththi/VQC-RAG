import numpy as np,pandas as pd,zipfile,io,hashlib
from scipy.stats import spearmanr
from pathlib import Path
root=Path(__file__).parent
class ExpandedEvidence:
 def read(self,n): return (root/'NFCorpus_Independent_Validation_Evidence'/n).read_bytes()
z=ExpandedEvidence()
def read(n): return pd.read_csv(io.BytesIO(z.read('results_nfcorpus/'+n+'.csv')))
a=read('per_query'); r=read('rankings'); old=read('paired'); w=a.groupby(['query','method'])['nDCG@10'].mean().unstack(); rows=[]
for base in ['Hybrid','MLP-BCE','MLP-Hinge']:
 d=(w.VQC-w[base]).values; rng=np.random.default_rng(20261106); n=len(d); b=np.mean(d[rng.integers(0,n,(20000,n))],axis=1); p=(1+np.count_nonzero(np.abs(np.mean(d*rng.choice([-1,1],(20000,n)),axis=1))>=abs(d.mean())))/20001
 rows.append(dict(comparison='VQC vs '+base,delta=d.mean(),low=np.quantile(b,.025),high=np.quantile(b,.975),p=p,wins=sum(d>1e-12),ties=sum(abs(d)<=1e-12),losses=sum(d< -1e-12)))
t=pd.DataFrame(rows); order=t.p.argsort(); t.loc[order,'holm']=np.minimum(1,np.maximum.accumulate(t.p.iloc[order].values*np.arange(3,0,-1))); print(t.to_string(index=False)); print('max statistical differences', (t[old.columns[1:]]-old.iloc[:,1:]).abs().max())
q=pd.read_csv(root/'test.txt',sep='\t'); rel=q.groupby('query-id')['corpus-id'].apply(set); h=r[(r.method=='Hybrid')&(r.seed==7)]; hits=h.groupby('query').apply(lambda x:len(set(x.doc)&rel[x.name])); print('candidate hits',hits.describe(), 'zero',sum(hits==0)); print('protocol',hashlib.sha256(z.read('PROTOCOL.txt')).hexdigest(),z.read('PROTOCOL_FROZEN_SHA256.txt')); print('manifest run hash',[(k,v) for k,v in __import__('json').loads(z.read('results_nfcorpus/manifest.json'))['hashes'].items() if k.endswith('run.py')]); print('local',hashlib.sha256((root/'Phase2_Execution/run.py').read_bytes()).hexdigest())
