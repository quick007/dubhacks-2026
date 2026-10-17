"""Revalidate saved raw generations without regeneration or changing baseline evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from .evidence_plan import catalog_at, validate_generated, dump


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--dataset',type=Path,default=Path('data/ranker/datasets/v2'))
    args=parser.parse_args();catalog=catalog_at(args.dataset)
    original=args.run/'benchmark.json';report=json.loads(original.read_text());revalidated=copy.deepcopy(report)
    cases={case['caseId']:case for case in json.loads((args.run/'case-plan.json').read_text())['cases']}
    for record in revalidated['records']:
        for arm,value in record['arms'].items():
            raw=value['generation'];request=cases[record['caseId']]['request']
            plan,quality=validate_generated(raw['text'],raw['finishReason'],value['pagePlan'],request,catalog)
            value['pagePlan']=plan;value['quality']=quality
    for arm in ('noPlan','plan'):
        values=[row['arms'][arm] for row in revalidated['records']]
        summary=revalidated['comparison'][arm]
        for key in ('newTitleCount','generatedBadgeCount','generatedHeadingCount'):
            summary[key]=sum(v['quality'][key] for v in values)
        summary['completePageCount']=sum(v['quality']['allFieldsFulfilled'] for v in values)
        summary['requestedTitleCount']=sum(len(s['products']) for v in values for s in v['pagePlan']['sections'])
        summary['requestedHeadingCount']=sum(len(v['pagePlan']['sections']) for v in values)
        for scale in summary['scale']:
            current=next(row['arms'][arm] for row in revalidated['records'] if row['kind']=='fixed-six-candidate-scale-pool' and row['cards']==scale['cards'])
            scale['quality']=current['quality']
        summary['correctNoneControl']=[row['arms'][arm]['quality'] for row in revalidated['records'] if row['kind']=='explicit-correct-NONE-control']
    revalidated['revalidation']={'originalSha256':hashlib.sha256(original.read_bytes()).hexdigest(),
        'generationRepeated':False,'timing':'Original full completed-attempt timing retained. No timing benefit claimed from post-hoc checks.',
        'fixes':['Aware timestamp cutoff and filter-before-bound','Exact explicit positive tag matching','Bound fact/reason evidence','Nonnull generated provenance and exact fallback','Request identity','Plaintext control rejection','Observed-alignment denominator excludes unsupported context'],
        'note':'Normal synthetic fixture input prompts unchanged by temporal/exact-tag fixes; all original raw completions retained.'}
    dump(args.run/'benchmark-revalidated.json',revalidated)
    compact={key:revalidated[key] for key in ('synthetic','recordedAtUtc','config','modelRevision','training','quality','correctedRouter','comparison','memory','limitations','revalidation')}
    compact['evidence']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (original,args.run/'benchmark-revalidated.json',args.run/'training.json',args.run/'case-plan.json')}
    dump(Path('experiments/ranker/pretrained/EVIDENCE_PLAN_RESULTS.json'),compact)
    print(json.dumps({'quality':compact['quality'],'comparison':{k:{name:v[name] for name in ('newTitleCount','requestedTitleCount','generatedHeadingCount','requestedHeadingCount','generatedBadgeCount','completePageCount','pageCount')} for k,v in compact['comparison'].items()},'revalidation':compact['revalidation']},indent=2))

if __name__=='__main__':main()
