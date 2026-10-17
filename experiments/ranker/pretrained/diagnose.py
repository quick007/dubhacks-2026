"""CPU-only audit of the preserved v2/shared-prefix and label-router baselines."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from .boundary import input_rows, visible_history
from .cli import load_heads, targets_for
from .routing import ROUTES, load_head, teacher_route

REVISION = '8b2b98c00a6b4d291155e4890773ca8f769aee53'


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(dataset, baseline, routing):
    torch.set_num_threads(1)
    catalog = {p['id']: p for p in json.loads((dataset/'catalog.json').read_text())['products']}
    requests = {r['recordId']: r for r in read_lines(dataset/'public/train.jsonl')}
    labels = {r['recordId']: r for r in read_lines(dataset/'labels/train.jsonl')}
    manifest = json.loads((dataset/'manifest.json').read_text())
    cache = torch.load(baseline/'training-embeddings.pt', weights_only=True)
    heads = load_heads(baseline/'adaptation', 4096, REVISION, 'shared-prefix')
    router = load_head(routing/'adaptation', REVISION)
    report = {'synthetic': True, 'baselinePreserved': True, 'stages': {}, 'audit': {},
              'fixedNextConfig': {'seed': 728, 'epochs': 80, 'learningRate': .03,
                 'productLoss': 'per-request best-product cross entropy after mean across variants',
                 'presentationLoss': 'conditional compatible-variant cross entropy, only products with informative nonzero labels',
                 'routingLoss': 'inverse-frequency weighted cross entropy',
                 'trainingData': 'all 24 train records together; no validation/test/private grading inputs for fitting',
                 'backbone': 'unchanged frozen 9B shared-prefix readouts; no architecture or decoder update'}}
    cosine = []
    alignment = True
    privacy = True
    history_unchanged = True
    for rid, (x, actions) in cache.items():
        request = requests[rid]
        alignment &= actions == input_rows(request, catalog)[1] and x.shape == (8, 4096)
        cosine.extend((x @ x.T)[torch.triu(torch.ones(8,8), diagonal=1).bool()].tolist())
        changed = copy.deepcopy(request)
        changed.update(hiddenPreferences='PRIVATE_SENTINEL', personaFamilyId='PRIVATE_SENTINEL', futureOutcome='PRIVATE_SENTINEL')
        changed['context']['privatePreference'] = 'PRIVATE_SENTINEL'
        for _ in range(5):
            changed['history'].append({'at': '2099-01-01T00:00:00Z', 'productId': actions[0]['productId'], 'description': 'FUTURE_SENTINEL'})
        privacy &= input_rows(request, catalog) == input_rows(changed, catalog)
        # Every original fixture history is sorted, <=2 events, and before cutoff.
        history_unchanged &= request['history'] == visible_history(request, catalog)
    for stage, ids in manifest['trainStageRecordIds'].items():
        x = torch.cat([cache[rid][0] for rid in ids])
        y = torch.cat([targets_for(requests[rid], cache[rid][1], labels) for rid in ids])
        targets = torch.tensor([ROUTES.index(teacher_route(requests[rid], catalog[a['productId']])) for rid in ids for a in cache[rid][1]])
        prediction = router(x).argmax(-1)
        report['stages'][stage] = {'rows':len(x), 'finalAdaptedDecisionMse': ((heads(x)-y)**2).mean().item(),
            'routerTargetCounts': dict(Counter(ROUTES[i] for i in targets.tolist())),
            'routerPredictionCounts': dict(Counter(ROUTES[i] for i in prediction.tolist())),
            'finalAdaptedRouterAccuracy': (prediction==targets).float().mean().item()}
    report['audit'] = {'cachedActionAndReadoutAlignment': bool(alignment), 'privateAndFutureMutationInvariant': bool(privacy),
        'fixtureHistoryPromptUnaffectedByBoundaryFix': bool(history_unchanged),
        'withinRequestReadoutCosine': {'mean':sum(cosine)/len(cosine),'min':min(cosine),'max':max(cosine)},
        'trainProductRelevanceCounts':dict(Counter(value for label in labels.values() for value in label['relevance'].values())),
        'trainPresentationTargetCounts':dict(Counter(value for label in labels.values() for scores in label['presentationScores'].values() for value in scores.values())),
        'missingStatedBudgetRecords':sum(r['context']['statedBudget'] is None for r in requests.values()),
        'historyCounts':dict(Counter(len(r['history']) for r in requests.values()))}
    report['findings'] = [
        'No action/label misalignment or injected private/future leakage found. Half the training requests hide their budget; private-rubric labels are partly unidentifiable from observations.',
        'The old history boundary slices before cutoff filtering and trusts list order. Future events can evict valid history. Fixed to filter/sort then bound; normal v2 fixture prompts are unchanged.',
        'Router targets are NONE/SELECT/GENERATE=48/36/12 per stage, and plain CE is unweighted. Final route accuracy is 100% on adaptation versus 80.2% on initial training families: fit plus forgetting, not generalization.',
        'Decision adaptation MSE is .025 on adaptation versus .122 on initial families. Adaptation sees only later 12 records, without replay.',
        'Pointwise suitability MSE does not directly optimize within-request ranking. Presentation zeros conflate ineligible product relevance with descriptor preference.',
        'Candidate readouts are causal and query-order-dependent. Cosine correlation is measured, not proof of a faulty backbone or architectural limit.',
        'Observed histories are synthetic and correlate with private preferred products by construction. This is an observable cue, not a deployment leakage guarantee; future outcomes/private fields never enter model text.'
    ]
    report['sourceInputs'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (dataset/'manifest.json', baseline/'training-embeddings.pt', baseline/'training.json', routing/'training.json')}
    return report


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('data/ranker/datasets/v2'))
    parser.add_argument('--baseline',type=Path,default=Path('data/ranker/pretrained/shared-prefix'))
    parser.add_argument('--routing',type=Path,default=Path('data/ranker/pretrained/routing'))
    parser.add_argument('--output',type=Path,default=Path('experiments/ranker/pretrained/DIAGNOSIS.json'))
    args=parser.parse_args(); report=audit(args.dataset,args.baseline,args.routing)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
