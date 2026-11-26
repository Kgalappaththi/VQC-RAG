"""Create a reproducible query-disjoint 60/20/20 split of the original 300 SciFact test queries.
This is a NEW experimental partition of benchmark test queries, not an official BEIR split.
No model training or evaluation occurs here.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'data'
OUT = ROOT / 'splits'
OUT.mkdir(exist_ok=True)
SEED = 20261106
queries = pd.read_csv(ROOT/'queries/queries.csv',dtype={'query_id':str})
candidates = pd.read_csv(ROOT/'results/hybrid_top20_candidates.csv',dtype={'query_id':str,'document_id':str})
qrels = pd.read_csv(ROOT/'qrels/qrels.csv',dtype={'query_id':str,'document_id':str})
ids = sorted(queries.query_id.unique().tolist(),key=lambda x:int(x) if x.isdigit() else x)
assert len(ids)==300 and len(set(ids))==300
rng = np.random.default_rng(SEED)
ids = np.array(ids,dtype=object)[rng.permutation(len(ids))]
assignment={q: ('train' if i<180 else 'validation' if i<240 else 'test') for i,q in enumerate(ids)}
rows=[]
for q in ids:
    c=candidates.loc[candidates.query_id==q,'document_id'].tolist()
    rel=set(qrels.loc[(qrels.query_id==q)&(qrels.relevance>0),'document_id'])
    eligible=any(d in rel for d in c) and any(d not in rel for d in c)
    rows.append({'query_id':q,'partition':assignment[q],'top20_candidate_count':len(c),'positive_qrels':len(rel),'pair_eligible':int(eligible),'relevant_in_top20':sum(d in rel for d in c)})
frame=pd.DataFrame(rows)
assert len(frame)==300 and frame.query_id.is_unique
assert frame.groupby('partition').size().to_dict()=={'train':180,'validation':60,'test':60}
assert frame.top20_candidate_count.eq(20).all()
assert set(frame.query_id)==set(queries.query_id)
frame.to_csv(OUT/'frozen_query_partitions.csv',index=False)
summary=frame.groupby('partition').agg(queries=('query_id','count'),pair_eligible=('pair_eligible','sum'),relevant_in_top20=('relevant_in_top20','sum'),qrels=('positive_qrels','sum')).reset_index()
summary.to_csv(OUT/'partition_summary.csv',index=False)
manifest={'source':'Original uploaded repository VQC-RAG-1.0-qmi-submission','seed':SEED,'rng':'numpy.random.default_rng PCG64','split':'180 train, 60 validation, 60 test queries','eligibility':'at least one qrel-positive and one qrel-unmarked top20 candidate','selection':'uniform random split across all 300 query IDs; no outcome-driven optimisation','warning':'SciFact original test queries repartitioned for exploratory supervised re-ranking; no external independent test corpus'}
(OUT/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
print(summary.to_string(index=False));print('Total pair-eligible:',frame.pair_eligible.sum())
