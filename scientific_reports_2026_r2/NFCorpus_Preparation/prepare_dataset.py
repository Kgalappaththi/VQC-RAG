from pathlib import Path
import csv, json, hashlib, shutil
import pyarrow.parquet as pq
root=Path(__file__).resolve().parent
out=root/'nfcorpus'
corpus=pq.read_table(root/'corpus-00000-of-00001.parquet').to_pylist()
queries=pq.read_table(root/'queries-00000-of-00001.parquet').to_pylist()
with (root/'test.txt').open(encoding='utf-8-sig',newline='') as f:
    rel=list(csv.DictReader(f,delimiter='\t'))
for rows,fields in [(corpus,['_id','title','text']),(queries,['_id','text'])]:
    assert all(all(isinstance(r.get(k),str) for k in fields) for r in rows), 'Unexpected schema or null text'
    assert len({r['_id'] for r in rows})==len(rows), 'Duplicate IDs'
assert len(corpus)==3633, 'Unexpected corpus size'
assert len({r['query-id'] for r in rel})==323 and len(rel)==12334
assert len({(r['query-id'],r['corpus-id']) for r in rel})==len(rel)
assert {r['score'] for r in rel}=={'1','2'}
assert {r['query-id'] for r in rel}<={r['_id'] for r in queries}
assert {r['corpus-id'] for r in rel}<={r['_id'] for r in corpus}
(out/'qrels').mkdir(parents=True,exist_ok=False)
for name,rows in [('corpus',corpus),('queries',queries)]:
    with (out/(name+'.jsonl')).open('w',encoding='utf-8',newline='\n') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
shutil.copyfile(root/'test.txt',out/'qrels/test.tsv')
manifest={'documents':len(corpus),'all_queries':len(queries),'test_queries':323,'judgements':len(rel),'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'test.txt',*root.glob('*.parquet'),*out.rglob('*.jsonl'),out/'qrels/test.tsv']}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
