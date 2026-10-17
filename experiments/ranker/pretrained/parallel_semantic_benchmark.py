"""Separate, gated quality/latency experiment; preserves strict v1 parity abort."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time

from .parallel_benchmark import parity, selected_ids, summary, validate_gate

OUT = Path('data/ranker/pretrained/parallel-semantic-v1')
REPORT = Path('experiments/ranker/pretrained/PARALLEL_SEMANTIC_RESULTS.json')


def validate_controls(control):
    """Require actual matched-state/repeat controls, never infer from token ties."""
    expected = {f'w{w}-{mode}-r{r}' for w in (1, 2, 3)
                for mode in ('fork', 'independent', 'full') for r in (0, 1)}
    if control.get('status') != 'completed' or set(control.get('arms', {})) != expected:
        raise ValueError('All eighteen actual matched-shape controls must complete')
    budget = control['memoryLimitBytes']
    if budget != 6*2**30 or control['memory']['mlxPeakBytes'] > budget:
        raise ValueError(f"Native controls exceeded fixed six-GiB peak budget: {control['memory']['mlxPeakBytes']} > {budget}; stop heavyweight comparison")
    base = control['pristineBaseHash']
    for key, arm in control['arms'].items():
        if arm['status'] != 'completed' or arm['baseBefore'] != base or arm['baseAfter'] != base:
            raise ValueError('Pristine complete state changed or control failed: '+key)
        if arm['mode'] != 'full' and arm['branchMatchesPristine'] != [True]*arm['width']:
            raise ValueError('Rebuilt/forked complete prefix differs: '+key)
        probe = arm['rows']['badge-1']['probe']
        if not probe['alignment']['exact'] or probe['actualActiveWidth'] != arm['width']:
            raise ValueError('Raw logits are not exactly aligned with sampled probabilities')
        if any(len(row['rawSampledTokenIds']) != 7 for row in arm['rows'].values()):
            raise ValueError('Control ended before the fixed comparison position')
    for width in (1, 2, 3):
        for repeat in (0, 1):
            comparison = control['comparisons'][f'w{width}-independent-r{repeat}-vs-fork']
            if not comparison['sameEarlierContextAllRows'] or not all(comparison[k]['exact'] for k in ('raw', 'logprobs')):
                raise ValueError('Matched-shape independent rebuilding differs from fork')
        for mode in ('fork', 'independent', 'full'):
            comparison = control['comparisons'][f'w{width}-{mode}-r1-vs-repeat0']
            if not all(comparison[k]['exact'] for k in ('raw', 'logprobs')):
                raise ValueError('Repeated distribution differs')
            a, b = [control['arms'][f'w{width}-{mode}-r{r}']['rows'] for r in (0, 1)]
            if any(a[key]['rawSampledTokenIds'] != b[key]['rawSampledTokenIds'] or
                   a[key]['finishReason'] != b[key]['finishReason'] for key in a):
                raise ValueError('Repeated tokens or finish differ')
    # Fresh full-prompt lanes must independently show width sensitivity at the
    # identical earlier badge context. This still cannot localize a shared
    # native merge/kernel issue; it excludes reuse as its necessary cause.
    changed = False
    for width in (2, 3):
        for mode in ('fork', 'independent', 'full'):
            comparison = control['comparisons'][f'w{width}-{mode}-r0-vs-width1']
            if not comparison['sameEarlierBadgeContext']:
                raise ValueError('Cross-width comparison has a different causal context')
            if mode == 'full' and not comparison['raw']['exact']:
                changed = True
    if not changed:
        raise ValueError('No independently fresh batch-shape sensitivity observed')


def comparison_policy(reference, candidate):
    evidence = parity(reference, candidate)
    # Exact generated strings remain evidence, while changed input/state
    # contracts still abort. Output differences are evaluated, never repaired.
    if not evidence['comparable'] or not evidence['samePrefix']:
        raise ValueError('Noncomparable attempt or changed shared prefix')
    if len(reference['rows']) != len(candidate['rows']) or not evidence['rows']:
        raise ValueError('Changed or empty row set')
    if not all(row['samePromptAndBudget'] for row in evidence['rows']):
        raise ValueError('Changed factorized prompt, budget or row identity')
    if len({r['rowId'] for r in candidate['rows']}) != len(candidate['rows']):
        raise ValueError('Duplicate row identity')
    return evidence


def benchmark(args):
    import torch
    from importlib.metadata import version
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION, digest, dump, load_corrected
    from .generation_diagnosis import cases_and_saved
    from .parallel_decode import factorized_arm
    from .full_page_reliability import run_arm
    start = time.perf_counter()
    control = json.loads(args.controls.read_text())
    validate_controls(control)
    if control['sourceSha256'] != digest(Path(__file__).with_name('native_batch_diagnosis.py')):
        raise ValueError('Control source changed since execution')
    array_path = args.controls.parent/'position6-arrays.npz'
    if control['arraysSha256'] != digest(array_path):
        raise ValueError('Raw native control distributions changed')
    runtime = {name: version(name) for name in ('mlx', 'mlx-lm')}
    if runtime != control['runtime']:
        raise ValueError('Control runtime changed')
    gate = json.loads(args.compatibility.read_text())
    sources = {name: digest(Path(__file__).with_name(name)) for name in ('parallel_decode.py','test_parallel_decode.py')}
    validate_gate(gate, runtime, sources)
    if args.out.exists():
        raise ValueError('Preserve existing experiment outputs; no overwrite/retry')
    cases = [case for case in cases_and_saved(args.dataset) if case['source'] == 'evidence']
    if len(cases) != 12:
        raise ValueError('All twelve fixed cases are required')
    prior_path = Path('data/ranker/pretrained/full-page-reliability-v1/benchmark.json')
    prior = {row['caseId']: row['arms']['manifest'] for row in json.loads(prior_path.read_text())['records']}
    dump(args.out/'case-plan.json', {'cases': [{k:v for k,v in case.items() if k!='saved'} for case in cases],
        'widths':[1,2,3,6], 'controlSha256':digest(args.controls), 'nativeReceiptSha256':digest(args.compatibility),
        'priorSha256':digest(prior_path), 'decoderSourceSha256':sources,
        'policy':'Strict v1 abort preserved. This separate experiment retains raw mismatches and uses unchanged page validation.',
        'order':'Widths 1/2/3/6; manifest first on even indices, last on odd. Order/warmth bias.'})
    torch.set_num_threads(1)
    heads, _, receipt = load_corrected(args.heads)
    load_start = time.perf_counter()
    class PeakCheckedBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            peak = self.mx.get_peak_memory()
            if peak > self.memory_budget:
                raise MemoryError(f'Observed peak {peak} exceeds fixed {self.memory_budget}-byte budget')
    # Existing decoder calls this hook after native rounds. This detects an
    # excursion sooner; the allocator limit itself is not a hard ceiling.
    backbone = PeakCheckedBackbone(args.model, REVISION, 1536, 6)
    load_ms = (time.perf_counter()-load_start)*1000
    records, abort = [], None
    for index, case in enumerate(cases):
        record = {'caseId':case['caseId'], 'cards':case['cards'], 'arms':{}}
        names = ['sequential','batch2','batch3','batch6']
        names.insert(0 if index%2==0 else 4, 'manifest')
        for name in names:
            attempt_start = time.perf_counter()
            if abort:
                value = {'status':'not-attempted', 'reason':abort}
            else:
                value = None
                try:
                    if name == 'manifest':
                        value = run_arm(backbone, heads, None, case, False)
                        value['status'] = 'completed-attempt'
                        value['baselineRawMatchesPrior'] = value['generation']['text'] == prior[case['caseId']]['generation']['text']
                        if not value['baselineRawMatchesPrior']:
                            raise ValueError('Preserved manifest text no longer reproduces')
                    else:
                        width = 1 if name=='sequential' else int(name[-1])
                        value = factorized_arm(backbone, heads, case['request'], case['catalog'], case['cards'], width)
                        if value['status'] != 'completed-attempt':
                            abort = name+': '+value.get('error','native failure')
                        elif width > 1:
                            value['parityAgainstSequential'] = comparison_policy(record['arms']['sequential'], value)
                    if not abort and selected_ids(value['pagePlan']) != selected_ids(prior[case['caseId']]['pagePlan']):
                        raise ValueError('Selected product/image IDs changed')
                except Exception as error:
                    # Retain any already-produced output alongside failure evidence.
                    value = {'status':'failed', 'error':f'{type(error).__name__}: {error}',
                        'fullCompletionMs':(time.perf_counter()-attempt_start)*1000,
                        'producedAttempt':value, 'budgetRaised':False}
                    abort = name+': '+value['error']
            record['arms'][name] = value
            dump(args.out/f'case-{index}-{name}.json', value)
            if value['status'] != 'not-attempted':
                print(json.dumps({'case':index, 'cards':case['cards'], 'arm':name, 'status':value['status'],
                    'quality':value.get('quality'), 'fullMs':value.get('timing',{}).get('fullCompletionMs'),
                    'exactTokens':value.get('parityAgainstSequential',{}).get('exact'), 'abort':abort}), flush=True)
                try:
                    gc.collect(); backbone.mx.synchronize(); backbone.mx.clear_cache(); backbone.check_memory()
                    memory = backbone.runtime_memory()
                    if memory['mlxPeakBytes'] > memory['processBudgetBytes']:
                        raise MemoryError(f"Peak allocation {memory['mlxPeakBytes']} exceeds fixed {memory['processBudgetBytes']}-byte budget")
                except Exception as error:
                    value['cleanupError'] = f'{type(error).__name__}: {error}'
                    value['memoryAtCleanupFailure'] = backbone.runtime_memory()
                    abort = abort or 'Cleanup failure: '+value['cleanupError']
                    dump(args.out/f'case-{index}-{name}.json', value)
        records.append(record)
        dump(args.out/'progress.json', {'records':records,'abort':abort})
    report = {'schemaVersion':1, 'synthetic':True, 'training':False, 'validatorChanges':False,
        'modelRevision':REVISION, 'headCheckpoint':receipt['checkpoints']['heads'],
        'residentLoadMs':load_ms, 'totalWallMs':(time.perf_counter()-start)*1000,
        'controlSha256':digest(args.controls), 'compatibility':gate, 'records':records,
        'summary':summary(records,12), 'abort':abort, 'memory':backbone.runtime_memory(),
        'limitations':['Inspected synthetic cases; one attempt per case/arm; no conversion/generalization evidence.',
            'Fixed widths increase, with warmth/order bias; whole-page versus factorized prompts change a separate factor.',
            'Raw exact-token mismatches remain reported; original strict abort and its outputs remain intact.',
            'Matched fork/rebuild parity plus fresh-full sensitivity exclude prefix reuse as a necessary cause; a shared native merge/kernel issue is not universally ruled out.',
            'Unchanged validator measures lexical factuality, novelty and completion, not full coherence, identity preservation or all prompt constraints.',
            'Invalid attempts are never usable-page speedups; failed/unattempted cases remain in twelve-case denominators.',
            'No prompt, budget, precision, sampler, runtime or memory-limit changes. Section grouping is deterministic.']}
    dump(args.out/'benchmark.json', report)
    report['evidence'] = {str(path):digest(path) for path in (args.out/'benchmark.json',args.out/'case-plan.json',args.controls,array_path)}
    dump(REPORT, report)
    print(json.dumps({'summary':report['summary'], 'totalWallMs':report['totalWallMs'], 'abort':abort,
                      'memory':report['memory']}, indent=2), flush=True)


def main():
    from .evidence_plan import REVISION
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('data/ranker/datasets/v2'))
    parser.add_argument('--heads',type=Path,default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--compatibility',type=Path,default=Path('data/ranker/pretrained/parallel-decode-v1/native-compatibility.json'))
    parser.add_argument('--controls',type=Path,default=Path('data/ranker/pretrained/native-batch-diagnosis-v2/probe.json'))
    parser.add_argument('--out',type=Path,default=OUT)
    args = parser.parse_args()
    expected = Path('.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots')/REVISION
    if args.model.resolve() != expected.resolve():
        parser.error('Use the verified pinned local snapshot')
    benchmark(args)


if __name__ == '__main__':
    main()
