"""Matched B1 fresh/cached extraction and A/B/A controls; no training/copy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch
from .preflight import DEFAULT_MODEL,file_digest
from .runtime import FeatureStore,NativeEncoder,assert_model_files,write_json
from .runner import corpus_rows
from .inputs import canonical


def run(args):
    if args.out.exists():raise ValueError('Preserve existing controls')
    args.out.mkdir(parents=True)
    started=time.perf_counter();report={'status':'started','executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'sourceSha256':file_digest(Path(__file__)),'priorReportSha256':file_digest(args.prior/'report.json'),
        'priorSourceIsPreserved':True,'sequence':'catalog A/B/A then query A/B/A; B1 whole-input shapes',
        'controls':[],'generatesCopy':False,'training':False}
    encoder=None
    try:
        provenance=json.loads((args.prior/'model-verification.json').read_text())
        preflight=json.loads((args.prior/'preflight.json').read_text())
        assert_model_files(args.model,provenance)
        rows=corpus_rows(args.dataset,preflight)
        store=FeatureStore.open_existing(args.prior/'features',provenance)
        selected={}
        for role in ('catalog','request'):
            candidates=[row for row in rows.values() if row['role']==role and store.exists(row)]
            if len(candidates)<2:raise ValueError('Two preserved members perrole required')
            selected[role]=candidates[:2]
        torch.set_num_threads(1);encoder=NativeEncoder(args.model)
        for role in ('catalog','request'):
            first=None
            for order,index in enumerate((0,1,0)):
                row=selected[role][index]
                cached,receipt=store.load(row);before=time.perf_counter()
                actual,timing=encoder.encode(row['text'],row)
                exact=torch.equal(cached,actual)
                repeated=None if order!=2 else torch.equal(first,actual)
                report['controls'].append({'role':role,'order':order,'member':row['member'],
                    'inputTokenSha256':row['tokenSha256'],'batchShape':row['batchShape'],
                    'cachedSha256':receipt['sha256'],'freshCachedExact':exact,
                    'maxAbsDifference':float((actual-cached).abs().max()),
                    'repeatAfterOtherInputExact':repeated,'timing':timing,
                    'externalFreshEncodeMs':(time.perf_counter()-before)*1000})
                if not exact or repeated is False:raise AssertionError('Matched-shape fresh/cache or A/B/A difference')
                if order==0:first=actual.clone()
                print(json.dumps({k:v for k,v in report['controls'][-1].items() if k!='timing'}),flush=True)
        if file_digest(args.prior/'report.json')!=report['priorReportSha256']:raise AssertionError('Original report changed')
        report['status']='passed';report['memory']=encoder.memory()
    except Exception as exc:
        report['status']='blocked';report['blocker']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        if encoder is not None:encoder.close()
        report['wallSeconds']=(time.perf_counter()-started)
        write_json(args.out/'results.json',report)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--model',type=Path,default=Path(DEFAULT_MODEL));p.add_argument('--dataset',type=Path,default=Path('data/ranker/datasets/v2'))
    p.add_argument('--prior',type=Path,default=Path('data/ranker/pretrained/catalog-user-fusion-v1-attempt-1'));p.add_argument('--out',type=Path,required=True);run(p.parse_args())


if __name__=='__main__':main()
