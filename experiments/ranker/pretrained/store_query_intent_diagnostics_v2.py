"""Predeclared TRAIN-only intent sensitivity diagnostics; no fitting/selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from . import store_query_intent_gate_v2 as gate

VERSION='query-intent-TRAIN-diagnostics.v2'
REPORT_THRESHOLD=.5  # Reporting only; runtime uses continuous pSpecific.

def summarize(rows,probabilities):
    if len(rows)!=len(probabilities) or len({r['caseId'] for r in rows})!=len(rows):raise ValueError('Unique complete case/prediction join required')
    items=[{'caseId':r['caseId'],'query':r['query'],'intent':r['intent'],'contextIndex':r['contextIndex'],'matchedContext':r['matchedContext'],
        'labelSpecific':r['labelSpecific'],'pSpecific':float(p),'correct':int(float(p)>=REPORT_THRESHOLD)==r['labelSpecific']} for r,p in zip(rows,probabilities,strict=True)]
    def metrics(group):
        counts={str(label):sum(x['labelSpecific']==label for x in group) for label in (0,1)}
        accuracy={str(label):sum(x['correct'] for x in group if x['labelSpecific']==label)/counts[str(label)] if counts[str(label)] else None for label in (0,1)}
        return {'rows':len(group),'classCounts':counts,'accuracyByClass':accuracy,'classBalancedAccuracy':sum(accuracy.values())/2 if all(x is not None for x in accuracy.values()) else None,'correct':sum(x['correct'] for x in group)}
    grouped={}
    for item in items:grouped.setdefault((item['intent'],item['query']),[]).append(item)
    per_query=[]
    for (kind,query),group in sorted(grouped.items()):
        if len(group)!=8 or {x['contextIndex'] for x in group}!=set(range(8)):raise ValueError('Each TRAIN phrase needs all8matched contexts')
        values=[x['pSpecific'] for x in group]
        per_query.append({'intent':kind,'query':query,'rows':8,'minPSpecific':min(values),'maxPSpecific':max(values),'maxContextDelta':max(values)-min(values),
            'correct':sum(x['correct'] for x in group),'all8Correct':all(x['correct'] for x in group),'predictions':group})
    contexts={str(index):metrics([x for x in items if x['contextIndex']==index]) for index in range(8)}
    contrasts={'broadNullBudget':metrics([x for x in items if x['labelSpecific']==0 and x['matchedContext']['itemBudget'] is None]),
        'specificNumericBudgetAndGuard':metrics([x for x in items if x['labelSpecific']==1 and x['matchedContext']['itemBudget'] is not None and x['matchedContext']['hardRequirements']])}
    return {'version':VERSION,'reportingThreshold':REPORT_THRESHOLD,'selectionOrThresholdTuning':False,'scope':'Correlated matched TRAIN contexts only; no quality/generalization claim',
        'overall':metrics(items),'budgetGuardContexts':contexts,'contrastSubsets':contrasts,'perQuery':per_query,
        'maximumPerQueryContextDelta':max(x['maxContextDelta'] for x in per_query),'independentTRAINPhraseCount':len(per_query),'privateQualityLabelsRead':False}

def run(features,fit,out):
    features=Path(features);fit=Path(fit);out=Path(out)
    if out.exists():raise FileExistsError('Unique TRAIN diagnostic output required')
    plan=json.loads((features/'plan.json').read_text());report=json.loads((features/'report.json').read_text());freeze=json.loads((fit/'HEAD_FROZEN.json').read_text())
    if gate.sha(features/'query-features.pt')!=report['queryFeaturesSha256']:raise ValueError('Native TRAIN feature bundle changed')
    head,metadata=gate.load_intent_head(fit/'head.pt',freeze['headSha256'],expected_source_manifest_sha256=gate.sha(features/'plan.json'))
    bundle=torch.load(features/'query-features.pt',map_location='cpu',weights_only=True)
    if bundle['caseIds']!=[r['caseId'] for r in plan['rows']] or bundle['sourceManifestSha256']!=metadata['sourceManifestSha256']:raise ValueError('TRAIN feature identity mismatch')
    for vector in bundle['features']:gate._feature(vector)
    with torch.inference_mode():probabilities=head(bundle['features'].float()).sigmoid().tolist()
    result=summarize(plan['rows'],probabilities);result.update(headSha256=freeze['headSha256'],sourceManifestSha256=metadata['sourceManifestSha256'],diagnosticSourceSha256=gate.sha(__file__))
    gate.dump(out,result);return result
if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--fit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=run(a.features,a.fit,a.out);print(json.dumps({'overall':r['overall'],'maximumPerQueryContextDelta':r['maximumPerQueryContextDelta']},indent=2))
