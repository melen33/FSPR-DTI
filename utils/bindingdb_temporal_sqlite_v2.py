"""Disk-backed pair aggregation for BindingDB temporal binary v1."""
import csv, json, sqlite3
from pathlib import Path
from collections import Counter
from rdkit import Chem
from utils.bindingdb_temporal_binary_v1 import _label, _normalize_sequence, _parse_ic50, _parse_year, _pair_id

def _get(row, cols, *names):
    for n in names:
        i=cols.get(n)
        if i is not None and i < len(row): return row[i].strip()
    return ''

def _smiles(x):
    m=Chem.MolFromSmiles(x)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m else None

def _setup(db):
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE pairs(pair_id TEXT PRIMARY KEY, smiles TEXT, protein TEXT, label_mask INTEGER, first_any INTEGER, first_eligible INTEGER)')

def _upsert(db, pair, smiles, protein, label, year):
    eligible = year if year >= 2013 else None
    mask = 2 if label else 1
    db.execute('''INSERT INTO pairs VALUES(?,?,?,?,?,?) ON CONFLICT(pair_id) DO UPDATE SET
      label_mask=(pairs.label_mask | excluded.label_mask),
      first_any=MIN(pairs.first_any,excluded.first_any),
      first_eligible=CASE WHEN pairs.first_eligible IS NULL THEN excluded.first_eligible WHEN excluded.first_eligible IS NULL THEN pairs.first_eligible ELSE MIN(pairs.first_eligible,excluded.first_eligible) END''', (pair,smiles,protein,mask,year,eligible))

def build_sqlite_benchmark(input_tsv, output_root):
    input_tsv, output_root = Path(input_tsv), Path(output_root)
    dbpath=output_root/'intermediate'/'pairs.sqlite'; dbpath.parent.mkdir(parents=True,exist_ok=True)
    if dbpath.exists(): dbpath.unlink()
    db=sqlite3.connect(str(dbpath)); _setup(db); counts=Counter()
    with input_tsv.open('r',newline='',encoding='utf-8-sig',errors='replace') as h:
      reader=csv.reader(h,delimiter='\t'); cols={x.strip():i for i,x in enumerate(next(reader))}
      for row in reader:
        counts['raw_rows']+=1; v,op=_parse_ic50(_get(row,cols,'IC50 (nM)')); label=_label(v,op) if v is not None else None
        if label is None: counts['filtered_ic50']+=1; continue
        year=_parse_year(_get(row,cols,'Date of publication'))
        smi=_smiles(_get(row,cols,'Ligand SMILES')); seq=_normalize_sequence(_get(row,cols,'BindingDB Target Chain Sequence 1','BindingDB Target Chain Sequence'))
        patent=_get(row,cols,'Patent Number'); article=_get(row,cols,'Article DOI','PMID')
        if year is None or smi is None or seq is None or not (patent or article): counts['filtered_metadata']+=1; continue
        _upsert(db,_pair_id(smi,seq),smi,seq,label,year); counts['accepted_measurements']+=1
        if counts['accepted_measurements']%10000==0: db.commit()
    db.commit(); splitroot=output_root/'splits'; splitroot.mkdir(parents=True,exist_ok=True); stats={}
    clauses={'train':'first_eligible BETWEEN 2013 AND 2019','validation':'first_eligible BETWEEN 2020 AND 2021','test':'first_eligible BETWEEN 2022 AND 2025 AND first_any>=2022','prospective_2026':'first_eligible=2026 AND first_any>=2026'}
    for name,clause in clauses.items():
      path=splitroot/(name+'.csv'); n=pos=0; drugs=set(); proteins=set()
      with path.open('w',newline='',encoding='utf-8') as out:
        w=csv.writer(out); w.writerow(['SMILES','Protein','Y'])
        for smi,seq,mask in db.execute('SELECT smiles,protein,label_mask FROM pairs WHERE label_mask IN (1,2) AND '+clause):
          y=1 if mask==2 else 0; w.writerow([smi,seq,y]); n+=1; pos+=y; drugs.add(smi); proteins.add(seq)
      stats[name]={'rows':n,'positive':pos,'negative':n-pos,'positive_ratio':(pos/n if n else None),'unique_drugs':len(drugs),'unique_proteins':len(proteins)}
    conflicts=db.execute('SELECT COUNT(*) FROM pairs WHERE label_mask=3').fetchone()[0]
    audit={'measurement_counts':dict(counts),'conflicting_pairs_dropped':conflicts,'splits':stats,'pair_overlap_all_zero':True}
    (output_root/'audit').mkdir(parents=True,exist_ok=True); (output_root/'audit'/'split_statistics.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    db.close(); return audit
