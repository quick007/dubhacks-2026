"""CPU-only full-grid workload preparation, not a model or UI implementation."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

OUT=Path('experiments/ranker/pretrained/FULL_GRID_WORKLOAD.json')


def prepare():
    from transformers import AutoTokenizer
    from .evidence_plan import REVISION
    from .fusion import build_shared_prefix,observable_request
    from .boundary import guard_reasons
    source=Path('data/ranker/pretrained/parallel-decode-v1/parallel-case-plan.json')
    base=next(c for c in json.loads(source.read_text())['cases'] if c['caseId']=='evidence-page-scale-6')
    catalog=base['catalog'];surface='compact-card'
    pool=sorted(pid for pid,p in catalog.items() if p['inStock'] and any(surface in v['compatibleSurfaces'] for v in p['presentationVariants']))[:36]
    if len(pool)!=36:raise ValueError('Need36 real eligible public diagnostic products; never repeat/pad products')
    model=Path('.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots')/REVISION
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    cases=[]
    for cards in (12,24):
        for profile in ('A','B'):
            request=observable_request(copy.deepcopy(base['request']))
            request['recordId']=f'full-grid-{cards}-profile-{profile}'
            request['candidateProductIds']=pool[:]
            request['context'].update(task='Choose everyday travel backpacks for a complete storefront grid.',
                statedBudget=None,statedRequirements=[],statedPreferences=['compact','durable'] if profile=='A' else ['lightweight','roomy'],
                presentationSurface=surface,presentationPreferences=['detail-view'] if profile=='A' else ['in-use-view'])
            request['presentationCompatibility']={pid:{v['id']:surface in v['compatibleSurfaces'] for v in catalog[pid]['presentationVariants']} for pid in pool}
            if any(guard_reasons(request,catalog[pid]) or not any(request['presentationCompatibility'][pid].values()) for pid in pool):
                raise ValueError('Candidate eligibility changed')
            # CPU tokenization only: a high counting cap does not alter serving guards.
            tokens,actions,positions,_=build_shared_prefix(tokenizer,request,catalog,32768)
            cases.append({'caseId':request['recordId'],'requestedCards':cards,'candidateCount':36,
                'request':request,'completeDecisionReadouts':len(actions),'measuredRequiredRankingPrefixTokens':len(tokens),
                'rankingPrefixTokenIdsSha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),
                'current1536PrefixGuardWouldReject':len(tokens)>1536,'readoutPositionsCount':len(positions),
                'requiredSectionsAtThreeCards':cards//3,'requiredFreshTitles':cards,'requiredBadgeOutputs':cards,
                'requiredHeadingOutputs':cards//3,'totalIndependentFieldJobs':2*cards+cards//3,
                'proposedPerFieldCaps':{'title':24,'badge':8,'heading':16},
                'proposedTotalSampledStepCap':cards*32+(cards//3)*16,
                'completedNativeAttempts':0,'evaluatedFullGridQuality':None})
    needed=set(pool)|{e['productId'] for e in cases[0]['request']['history'] if e['productId'] in catalog}
    return {'schemaVersion':1,'synthetic':True,'inspectedDiagnosticsNotHeldOut':True,'executedInference':False,
        'sourceCasePlanSha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'modelRevision':REVISION,'cases':cases,'publicCatalog':{pid:catalog[pid] for pid in sorted(needed)},
        'contract':'All requested unique product/image choices and every requested fresh copy field must complete and pass strict validation; partial first-card readiness is separate.',
        'planningTargetNotMeasured':{'warmCompleteGridP50Ms':500,'coldCompleteGridMs':2500,
            'scope':'Full requested12/24-card grid including all required fresh titles/badges/headings and validation.',
            'coldDefinition':'Process/model load plus first complete request, separately from resident cache-cold and cache-warm.',
            'status':'Challenge targets selected for architecture evaluation, not user-agreed SLA or demonstrated feasibility. Current9B timings miss them.'},
        'requiredConditions':['model-cold full request','resident-cache-cold full request','resident-cache-warm full request',
            'A-to-B fresh public preference conditioning','return-to-A','catalog price/availability invalidation'],
        'acceptanceProtocol':['Fixed case/condition denominators include every failed and unattempted request.',
            'No candidate truncation, smaller grid, repeated products, approved-title or template substitution.',
            'Report complete-grid quality and full completion, not just first card or decision readiness.',
            'Retain source/runtime/head/token/input hashes, raw sampled IDs/EOS/text/stop, all field rejections, wall time and memory.',
            'Model/prompt representation changes require corresponding trained-head compatibility and quality measurements.',
            'Existing1/3/6 count/section/budget guards remain unchanged; this is a future benchmark input packet, not a runnable serving contract.']}


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    if args.out.exists():raise ValueError('Preserve full-grid packet; no overwrite')
    value=prepare();args.out.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'executedInference':False,'cases':[{'caseId':c['caseId'],'cards':c['requestedCards'],
        'candidates':c['candidateCount'],'prefixTokens':c['measuredRequiredRankingPrefixTokens'],
        'fieldJobs':c['totalIndependentFieldJobs']} for c in value['cases']]}))


if __name__=='__main__':main()
