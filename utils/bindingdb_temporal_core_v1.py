"""Patent-only, single-chain, disk-backed BindingDB temporal core benchmark."""
import csv, json, sqlite3
from collections import Counter
from pathlib import Path
from rdkit import Chem
from utils.bindingdb_temporal_binary_v1 import _label,_normalize_sequence,_parse_ic50,_parse_year,_pair_id

def _get(r,c,*ns):
 for n in ns:
  i=c.get(n)
  if i is not None and i<len(r): return r[i].strip()
 return ''
def _smi(x):
 m=Chem.MolFromSmiles(x); return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True) if m else None
def build_core(input_tsv,output_root):
 input_tsv,output_root=Path(input_tsv),Path(output_root); dbp=output_root/'intermediate'/'pairs.sqlite'; dbp.parent.mkdir(parents=True,exist_ok=True)
 if dbp.exists(): dbp.unlink()
 db=sqlite3.connect(str(dbp)); db.execute('PRAGMA journal_mode=WAL'); db.execute('CREATE TABLE p(id TEXT PRIMARY KEY,s TEXT,q TEXT,m INTEGER,fa INTEGER,fe INTEGER)'); cnt=Counter()
 with input_tsv.open('r',newline='',encoding='utf-8-sig',errors='replace') as f:
  rd=csv.reader(f,delimiter='\t'); c={x.strip():i for i,x in enumerate(next(rd))}; chain='Number of Protein Chains in Target (>1 implies a multichain complex)'
  for r in rd:
   cnt['raw_rows']+=1
   if not _get(r,c,'Patent Number'): cnt['non_patent_dropped']+=1; continue
   if _get(r,c,chain)!='1': cnt['non_single_chain_dropped']+=1; continue
   v,o=_parse_ic50(_get(r,c,'IC50 (nM)')); y=_label(v,o) if v is not None else None
   if y is None: cnt['ic50_dropped']+=1; continue
   year=_parse_year(_get(r,c,'Date of publication')); s=_smi(_get(r,c,'Ligand SMILES')); q=_normalize_sequence(_get(r,c,'BindingDB Target Chain Sequence 1'))
   if year is None or s is None or q is None: cnt['metadata_dropped']+=1; continue
   e=year if year>=2013 else None; mask=2 if y else 1; ident=_pair_id(s,q)
   db.execute('''INSERT INTO p VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET m=(p.m|excluded.m),fa=MIN(p.fa,excluded.fa),fe=CASE WHEN p.fe IS NULL THEN excluded.fe WHEN excluded.fe IS NULL THEN p.fe ELSE MIN(p.fe,excluded.fe) END''',(ident,s,q,mask,year,e)); cnt['accepted_measurements']+=1
   if cnt['accepted_measurements']%10000==0: db.commit()
 db.commit(); out=output_root/'splits'; out.mkdir(parents=True,exist_ok=True); clauses={'train':'fe BETWEEN 2013 AND 2019','validation':'fe BETWEEN 2020 AND 2021','test':'fe BETWEEN 2022 AND 2025 AND fa>=2022','prospective_2026':'fe=2026 AND fa>=2026'}; stats={}
 for n,w in clauses.items():
  rows=pos=0; drugs=set(); prots=set()
  with (out/(n+'.csv')).open('w',newline='') as f:
   cw=csv.writer(f); cw.writerow(['SMILES','Protein','Y'])
   for s,q,m in db.execute('SELECT s,q,m FROM p WHERE m IN (1,2) AND '+w):
    y=1 if m==2 else 0; cw.writerow([s,q,y]); rows+=1; pos+=y; drugs.add(s); prots.add(q)
  stats[n]={'rows':rows,'positive':pos,'negative':rows-pos,'positive_ratio':pos/rows if rows else None,'unique_drugs':len(drugs),'unique_proteins':len(prots)}
 report={'protocol':'patent_only_single_chain_conflict_free','measurement_counts':dict(cnt),'conflicting_pairs_dropped':db.execute('SELECT count(*) FROM p WHERE m=3').fetchone()[0],'splits':stats}; (output_root/'audit').mkdir(parents=True,exist_ok=True); (output_root/'audit'/'split_statistics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); db.close(); return report
