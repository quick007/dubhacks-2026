"""Bounded same-prompt sequential/native-batch complete-page comparison."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time

OUT = Path('data/ranker/pretrained/parallel-decode-v1')
PRIOR = Path('data/ranker/pretrained/full-page-reliability-v1/benchmark.json')
WIDTHS = (1, 2, 3, 6)


def selected_ids(plan):
    return [[card['productId'], card['imageCaptionId']]
            for section in plan['sections'] for card in section['products']]


def parity(reference, candidate):
    """Compare every raw generated token and completion, never accepted-only rows."""
    if reference.get('status') != 'completed-attempt' or candidate.get('status') != 'completed-attempt':
        return {'comparable': False, 'exact': False, 'reason': 'failed-or-unattempted-arm'}
    left = {row['rowId']: row for row in reference['rows']}
    right = {row['rowId']: row for row in candidate['rows']}
    same_set = left.keys() == right.keys()
    details = []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        exact = bool(a and b and a['tokenIds'] == b['tokenIds'] and a['finishReason'] == b['finishReason'])
        same_prompt = bool(a and b and a['promptSha256'] == b['promptSha256'] and a['maxTokens'] == b['maxTokens'])
        details.append({'rowId': key, 'exactTokensAndFinish': exact, 'samePromptAndBudget': same_prompt})
    return {'comparable': True, 'exact': same_set and all(r['exactTokensAndFinish'] and r['samePromptAndBudget'] for r in details),
            'samePrefix': reference['prefixSha256'] == candidate['prefixSha256'], 'rows': details}


def summary(records, expected_cases):
    from .routing import summary as latencies
    result = {}
    for name in ('manifest', 'sequential', 'batch2', 'batch3', 'batch6'):
        values = [row['arms'][name] for row in records if name in row['arms']]
        completed = [v for v in values if v.get('status') == 'completed-attempt']
        valid = [v for v in completed if v['quality']['allFieldsFulfilled']]
        result[name] = {'expectedCases': expected_cases, 'recordedCases': len(values), 'completedAttempts': len(completed),
            'failedAttempts': sum(v.get('status') == 'failed' for v in values),
            'unattempted': expected_cases - len(values) + sum(v.get('status') == 'not-attempted' for v in values),
            'validPages': len(valid), 'validPageRateAllFixedCases': len(valid) / expected_cases,
            'newTitleCount': sum(v['quality']['newTitleCount'] for v in completed),
            'acceptedHeadingCount': sum(v['quality']['generatedHeadingCount'] for v in completed),
            'acceptedBadgeCount': sum(v['quality']['generatedBadgeCount'] for v in completed),
            'fullAttemptMs': latencies([v['timing']['fullCompletionMs'] for v in completed]),
            'validPageMs': latencies([v['timing']['fullCompletionMs'] for v in valid]),
            'rawBaselineReplayCount': sum(v.get('baselineRawMatchesPrior', False) for v in completed),
            'exactSequentialParityCount': sum(v.get('parityAgainstSequential', {}).get('exact', False) for v in completed)}
    return result


def benchmark(args):
    import torch
    from .backbone import FrozenBackbone
    from .evidence_plan import load_corrected, REVISION, dump, digest
    from .full_page_reliability import run_arm
    from .generation_diagnosis import cases_and_saved
    from .parallel_decode import factorized_arm
    gate = json.loads(args.compatibility.read_text())
    if gate.get('passed') is not True:
        raise ValueError('Actual tiny native compatibility gate must pass before pretrained comparison')
    cases = [case for case in cases_and_saved(args.dataset) if case['source'] == 'evidence']
    if len(cases) != 12:
        raise ValueError('All twelve fixed evidence cases are required; no outcome selection')
    prior = {r['caseId']: r['arms']['manifest'] for r in json.loads(PRIOR.read_text())['records']}
    dump(args.out / 'parallel-case-plan.json', {'synthetic': True, 'training': False, 'validatorChanges': False,
        'cases': [{k: v for k, v in case.items() if k != 'saved'} for case in cases],
        'widths': WIDTHS, 'budgets': {'1': 64, '3': 128, '6': 224},
        'compatibility': gate, 'compatibilitySha256': digest(args.compatibility), 'priorSha256': digest(PRIOR),
        'order': 'Widths increase 1/2/3/6. Manifest before widths on even cases, after on odd cases; disclose order bias.',
        'scope': 'Same twelve evidence-source cases and user-first ranking context; catalog-first cache conditions excluded.'})
    torch.set_num_threads(1)
    heads, _, training = load_corrected(args.heads)
    start = time.perf_counter()
    backbone = FrozenBackbone(args.model, REVISION, 1536, 6)
    load_ms = (time.perf_counter()-start)*1000
    records = []
    abort = None
    for index, case in enumerate(cases):
        record = {'caseId': case['caseId'], 'cards': case['cards'], 'arms': {}}
        names = ['sequential', 'batch2', 'batch3', 'batch6']
        names.insert(0 if index % 2 == 0 else len(names), 'manifest')
        for name in names:
            if abort:
                value = {'status': 'not-attempted', 'reason': abort}
            elif name == 'manifest':
                value = run_arm(backbone, heads, None, case, False)
                value['status'] = 'completed-attempt'
                value['baselineRawMatchesPrior'] = value['generation']['text'] == prior[case['caseId']]['generation']['text']
                if not value['baselineRawMatchesPrior']:
                    abort = 'Manifest baseline text no longer reproduces; stop comparison and inspect.'
            else:
                width = 1 if name == 'sequential' else int(name[-1])
                value = factorized_arm(backbone, heads, case['request'], case['catalog'], case['cards'], width)
                if value['status'] == 'completed-attempt':
                    if selected_ids(value['pagePlan']) != selected_ids(prior[case['caseId']]['pagePlan']):
                        abort = 'Selected product/image changed; stop comparison.'
                    if width > 1:
                        value['parityAgainstSequential'] = parity(record['arms']['sequential'], value)
                        if not value['parityAgainstSequential']['samePrefix']:
                            abort = 'Common ranker prefix changed; stop comparison.'
                else:
                    # Do not raise memory allowance or continue using an uncertain runtime.
                    abort = name + ': ' + value.get('error', 'native failure')
            record['arms'][name] = value
            dump(args.out / f'parallel-case-{index}-{name}.json', value)
            print(json.dumps({'case': index+1, 'total': len(cases), 'cards': case['cards'], 'arm': name,
                'status': value['status'], 'fullMs': value.get('timing', {}).get('fullCompletionMs'),
                'quality': value.get('quality'), 'parity': value.get('parityAgainstSequential', {}).get('exact'),
                'error': value.get('error')}), flush=True)
            gc.collect()
            backbone.mx.clear_cache()
            backbone.mx.synchronize()
        records.append(record)
        dump(args.out / 'parallel-progress.json', {'records': records, 'abort': abort})
    report = {'synthetic': True, 'modelRevision': REVISION, 'training': False, 'validatorChanges': False,
        'residentLoadMs': load_ms, 'headCheckpoint': training['checkpoints']['heads'],
        'compatibility': gate, 'records': records, 'summary': summary(records, len(cases)), 'abort': abort,
        'memory': backbone.runtime_memory(), 'limitations': [
            'Same previously inspected synthetic cases, not human conversion or untouched generalization evidence.',
            'Whole-page manifest versus factorized sequential changes prompts; only identical factorized sequential/batch isolates concurrency.',
            'Widths run increasingly for bounded memory safety, causing order/warmth bias; one attempt per fixed case.',
            'Each arm ranks again and starts timing before prefill, then includes complete state materialization, suffixes, generation, assembly and strict validation.',
            'A fast invalid attempt is not a usable page. All fixed cases, rejected fields, failed and unattempted arms stay in denominators.',
            'Batching parallelizes independent autoregressive streams, not all tokens in one forward pass.',
            'No retry, raised memory limit, new runtime, MTP, prompt-cache splicing, generated-code execution or validator relaxation.']}
    dump(args.out / 'parallel-benchmark.json', report)
    compact = dict(report)
    compact['evidence'] = {str(p): digest(p) for p in (args.out / 'parallel-benchmark.json', args.out / 'parallel-case-plan.json', args.compatibility, PRIOR)}
    dump(Path('experiments/ranker/pretrained/PARALLEL_BENCHMARK_RESULTS.json'), compact)
    print(json.dumps({'summary': report['summary'], 'abort': abort, 'memory': report['memory']}, indent=2), flush=True)


def main():
    from .evidence_plan import REVISION
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('data/ranker/datasets/v2'))
    parser.add_argument('--heads', type=Path, default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--compatibility', type=Path, default=OUT / 'native-compatibility.json')
    args = parser.parse_args()
    expected = Path('.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots') / REVISION
    if args.model.resolve() != expected.resolve():
        parser.error('Use the existing verified pinned local four-bit checkpoint')
    benchmark(args)


if __name__ == '__main__':
    main()
