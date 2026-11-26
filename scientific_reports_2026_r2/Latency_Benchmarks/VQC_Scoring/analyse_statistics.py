"""Exploratory paired test-set comparisons with Holm family-wise correction.
Uses original saved per-query scores; does not tune or train models.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
p=argparse.ArgumentParser();p.add_argument('--repo',default=None);p.add_argument('--stage7',default=None);p.add_argument('--output',default='statistical_comparisons.csv');a=p.parse_args()
root=Path(a.repo) if a.repo else Path(__file__).resolve().parents[1]/'VQC_RAG_Revised_Reproducible'
stage=Path(a.stage7) if a.stage7 else Path(__file__).resolve().parents[1]/'VQC_Stage7_Pairwise_RankSVM'
cl=pd.read_csv(root/'results/classical/per_query_classical_metrics.csv')
rank=pd.read_csv(stage/'per_query_metrics.csv')
qcol='query_id';metric='nDCG@10'
def get(df,method):
    s=df[(df.partition=='test')&(df.method==method)].set_index(qcol)[metric].sort_index()
    assert len(s)==60 and s.index.is_unique
    return s
baseline=get(cl,'Hybrid')
methods={'Hybrid+MLP':get(cl,'Hybrid+MLP_seed42'),'Hybrid+LogisticRegression':get(cl,'Hybrid+LogisticRegression'),'Hybrid+RankSVM':get(rank,'Hybrid+RankSVM_0.8'),'RankSVM':get(rank,'RankSVM')}
rng=np.random.default_rng(20261107)
rows=[]
for name,v in methods.items():
    assert baseline.index.equals(v.index)
    delta=(v-baseline).to_numpy()
    # sign-flip randomisation test under paired exchangeability, two-sided
    n=100000
    flips=rng.choice([-1,1],size=(n,len(delta)),replace=True)
    null=(flips*delta).mean(axis=1)
    observed=abs(delta.mean())
    pval=(1+np.sum(np.abs(null)>=observed-1e-14))/(n+1)
    bs_idx=rng.integers(0,len(delta),size=(20000,len(delta)))
    ci=np.quantile(delta[bs_idx].mean(axis=1),[.025,.975])
    rows.append({'method':name,'baseline':'Hybrid','n_queries':len(delta),'mean_delta_nDCG@10':delta.mean(),'ci95_low':ci[0],'ci95_high':ci[1],'p_randomisation_two_sided':pval})
out=pd.DataFrame(rows)
# Holm step-down over all four specified classical comparisons
order=np.argsort(out.p_randomisation_two_sided.to_numpy()); adjusted=np.empty(len(out));running=0.
for j,i in enumerate(order):
    running=max(running,(len(out)-j)*out.loc[i,'p_randomisation_two_sided'])
    adjusted[i]=min(1.,running)
out['p_holm_4_comparisons']=adjusted
out.to_csv(a.output,index=False)
print(out.to_string(index=False,float_format=lambda x:f'{x:.5f}'))
print('EXPLORATORY: test results were inspected before this family was fixed; not confirmatory.')
