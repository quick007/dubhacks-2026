"""Exact old v5 query-feature replay of the preregistered fusion bound.

No new query is assigned an old vector. No new persona model is evaluated here:
this checks every simplex-vertex worst-case preference perturbation against the
32 exact historical v5 query texts. It reads no relevance/evaluation labels.
"""
from pathlib import Path
import argparse
import json
import os
import torch
from . import store_persona_context_v1 as context
from .store_persona_runtime_v1 import FUSION_POLICY


def check(asset_root,out):
    from .store_rank_session_v5 import read_export
    from .store_function_adaptation import load_head,request_text
    from .store_demo_engine import checked_request,ranking_input,guarded_order
    from .catalog_fusion.runtime import FeatureStore
    asset_root=Path(asset_root).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError('Unique CPU replay result required')
    if not out.is_relative_to(context.HERE.parents[2]):raise ValueError('Write only isolated worktree')
    os.chdir(asset_root);torch.set_num_threads(1)
    run=asset_root/'data/ranker/pretrained/actual-store-function-v5-attempt-1';export_path=run/'serving-head.json'
    export=read_export(export_path,context.sha(export_path));head=load_head(run/'learned.pt',export['head']['sha256'],export['head']['metadata'])
    packet=json.loads((run/'packet.json').read_text());store=FeatureStore.open_existing(run/'features',export['featureProvenance'])
    catalog=context.load_catalog_features(asset_root/'data/ranker/pretrained/actual-store-function-v3-attempt-1');c=catalog['vectors'];rows=[]
    queries=[r for r in packet['records'] if r['role']=='request' and r['split']=='evaluation']
    if len(queries)!=32:raise ValueError('Fixed32 original v5 native queries required')
    for row in queries:
        payload=json.loads(row['text'].split('\n',1)[1]);public=payload['request']
        if len(public['preferences'])!=1:raise ValueError('Exact historical storefront query shape required')
        request,policy=checked_request({'query':public['preferences'][0],'itemBudget':public['itemBudget'],'hardRequirements':public['hardRequirements']})
        if request_text(ranking_input(request))!=row['text']:raise ValueError('Refusing to relabel historical native query vector')
        q,_=store.load(row)
        with torch.no_grad():scores=(head.probabilities('catalog',c)@head.probabilities('query',q)[0])
        baseline,checks=guarded_order(scores.tolist(),request,policy,catalog['products']);alternatives=[]
        for index in range(16):
            addition=torch.zeros(16);addition[index]=FUSION_POLICY['specificQueryResidual']
            order,_=guarded_order((scores+addition).tolist(),request,policy,catalog['products'])
            alternatives.append({'preferenceVertex':index,'top1':order[0] if order else None,'sameTop1':order[:1]==baseline[:1]})
        ranked=scores.sort(descending=True).values
        rows.append({'caseId':row['caseId'],'queryTextSha256':row['textSha256'],'baselineTop1':baseline[0] if baseline else None,
            'rawScoreMargin':float(ranked[0]-ranked[1]),'all16WorstCaseTop1Stable':all(r['sameTop1'] for r in alternatives),
            'worstCaseChangedVertices':[r['preferenceVertex'] for r in alternatives if not r['sameTop1']]})
    report={'status':'completed-cpu-historical-replay','fixedQueries':32,'nativeCalls':0,'newQueriesEncoded':0,'goldRead':False,
        'personaModelEvaluated':False,'freshIndependent32Evaluated':False,'fusionPolicy':FUSION_POLICY,
        'stableAgainstAll16WorstCaseVertices':sum(r['all16WorstCaseTop1Stable'] for r in rows),
        'interpretation':'Sufficient worst-case score-bound check, not fused serving quality or a measured native latency benchmark',
        'sourcePins':{str(export_path):context.sha(export_path),str(run/'packet.json'):context.sha(run/'packet.json'),str(Path(__file__)):context.sha(__file__)},'rows':rows}
    out.parent.mkdir(parents=True,exist_ok=True);context.dump(out,report);return report

def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    r=check(a.asset_root,a.out);print(json.dumps({k:r[k] for k in ('status','fixedQueries','stableAgainstAll16WorstCaseVertices','nativeCalls')}))
if __name__=='__main__':main()
