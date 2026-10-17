"""One fixed 9B log-probability probe of the preserved case-2 badge divergence.

Uses one pristine common rank state; no sampler/math/runtime/source modifications.
The observer subclasses BatchGenerator in this process only and retains CPU data.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_hash(cache):
    import numpy as np
    import mlx.core as mx
    result = hashlib.sha256()
    def visit(value):
        if isinstance(value, (list, tuple)):
            result.update(f"sequence:{len(value)}".encode())
            for child in value:
                visit(child)
        elif hasattr(value, "dtype"):
            result.update(str(value.dtype).encode())
            result.update(str(value.shape).encode())
            # Float32 exactly represents bfloat16 values; conversion is for
            # host hashing only and never replaces the live native state.
            array = np.array(value.astype(mx.float32) if str(value.dtype) == "mlx.core.bfloat16" else value)
            result.update(array.tobytes())
        else:
            result.update(json.dumps(value).encode())
    for layer in cache:
        result.update(type(layer).__name__.encode())
        visit(layer.state)
    return result.hexdigest()


def run(args):
    import numpy as np
    import torch
    import importlib
    generate = importlib.import_module("mlx_lm.generate")
    native_class = generate.BatchGenerator
    from importlib.metadata import version
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION, CONFIG, load_corrected, all_actions
    from .fusion import encode_shared_prefix, append_generation_suffix
    from .page_plan import make_page_plan, validate_page_plan
    from .parallel_decode import build_rows, decode_rows
    torch.set_num_threads(1)
    expected_model = Path.cwd()/'.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots'/REVISION
    if args.model.resolve() != expected_model.resolve():
        raise ValueError("Use the verified pinned snapshot")
    prior = {arm: json.loads((args.prior/f'parallel-case-2-{arm}.json').read_text()) for arm in ('sequential','batch2','batch3')}
    case = json.loads((args.prior/'parallel-case-plan.json').read_text())['cases'][2]
    request, catalog = case['request'], case['catalog']
    heads, _, receipt = load_corrected(args.heads)
    load_start = time.perf_counter()
    backbone = FrozenBackbone(args.model, REVISION, 1536, 6)
    model_load_ms = (time.perf_counter()-load_start)*1000
    embeddings, actions, prefix_timing, base = encode_shared_prefix(backbone, request, catalog, 1536)
    with torch.inference_mode():
        decision = all_actions(request, catalog, actions, heads(embeddings).tolist())
    plan = make_page_plan(request, catalog, decision, case['cards'])
    validate_page_plan(plan, request, catalog)
    rows = build_rows(plan, request, catalog)
    del embeddings
    baseline_rows = {row['rowId']: row for row in prior['sequential']['rows']}
    preflight = []
    for row in rows:
        dummy = object()
        opaque = {**base, 'cache': [], 'model': dummy, 'consumed': False}
        suffix, _ = append_generation_suffix(backbone.tokenizer, row['messages'], opaque, dummy)
        prompt_sha = hashlib.sha256(json.dumps(row['messages'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for saved in prior.values():
            reference = next(item for item in saved['rows'] if item['rowId']==row['rowId'])
            if (prompt_sha, len(suffix), row['maxTokens']) != (reference['promptSha256'],reference['suffixTokens'],reference['maxTokens']):
                raise AssertionError('Probe row differs from preserved prompt/suffix/budget')
        preflight.append({'rowId':row['rowId'],'promptSha256':prompt_sha,'suffixTokens':len(suffix),'maxTokens':row['maxTokens'],'matchedSavedArms':list(prior)})
    if any(base['prefixSha256'] != saved['prefixSha256'] for saved in prior.values()):
        raise AssertionError('Common prefix differs from preserved failing case')
    original_hash = state_hash(base['cache'])
    reports, full_logprobs = {}, {}
    args.out.mkdir(parents=True,exist_ok=True)
    def save_progress():
        (args.out/'progress.json').write_text(json.dumps({'caseId':case['caseId'],'arms':reports},indent=2,allow_nan=False)+'\n')
        np.savez(args.out/'badge-position6-logprobs.npz',**{key.replace('-','_'):value for key,value in full_logprobs.items()})
    try:
        for label, width in (('sequential',1),('batch2',2),('batch3',3),('batch3-repeat',3)):
            observed = {'rawTokens':{},'badgePosition6':None}
            class Observer(native_class):
                def __init__(self,*pargs,**kwargs):
                    super().__init__(*pargs,**kwargs)
                    self._probe_admitted=0
                    self._probe_rows={}
                    self._probe_steps={}
                def insert(self,*pargs,**kwargs):
                    uids=super().insert(*pargs,**kwargs)
                    for uid in uids:
                        row_id=rows[self._probe_admitted]['rowId']
                        self._probe_admitted+=1
                        self._probe_rows[uid]=row_id
                        self._probe_steps[uid]=0
                        observed['rawTokens'][row_id]=[]
                    return uids
                def next_generated(self):
                    responses=super().next_generated()
                    for response in responses:
                        row_id=self._probe_rows[response.uid]
                        index=self._probe_steps[response.uid]
                        observed['rawTokens'][row_id].append(int(response.token))
                        self._probe_steps[response.uid]+=1
                        if row_id=='badge-1' and index==6:
                            values=np.array(response.logprobs.astype(backbone.mx.float32)).reshape(-1).copy()
                            top=np.argpartition(values,-5)[-5:]
                            top=top[np.argsort(values[top])[::-1]]
                            observed['badgePosition6']={'sampledTokenId':int(response.token),'indexZeroBased':index,
                                'top5':[{'tokenId':int(t),'text':backbone.tokenizer.decode([int(t)]),'logprob':float(values[t])} for t in top],
                                'token85688':{'text':backbone.tokenizer.decode([85688]),'logprob':float(values[85688])},
                                'token417':{'text':backbone.tokenizer.decode([417]),'logprob':float(values[417])},
                                'meadowMinusBLogprob':float(values[85688]-values[417]),
                                'topMargin':float(values[top[0]]-values[top[1]])}
                            full_logprobs[label]=values
                    return responses
            generate.BatchGenerator=Observer
            before=state_hash(base['cache'])
            outputs,timing=decode_rows(backbone,base,rows,width)
            after=state_hash(base['cache'])
            if before!=original_hash or after!=original_hash:
                raise AssertionError('Pristine complete prefix state changed')
            row_checks=[]
            for row in outputs:
                prior_row=baseline_rows[row['rowId']]
                common=next((i for i,(a,b) in enumerate(zip(prior_row['rawSampledTokenIds'],row['rawSampledTokenIds'])) if a!=b),min(len(prior_row['rawSampledTokenIds']),len(row['rawSampledTokenIds'])))
                row_checks.append({'rowId':row['rowId'],'firstDivergenceVsSavedSequential':None if prior_row['rawSampledTokenIds']==row['rawSampledTokenIds'] else common,
                    'rawSampledTokenIds':row['rawSampledTokenIds'],'finishReason':row['finishReason'],'arSteps':row['arSteps'],'stopTokenId':row['stopTokenId'],'text':row['text']})
            reports[label]={'width':width,'observer':observed,'rows':row_checks,'baseStateBefore':before,'baseStateAfter':after,'timing':timing,'memory':backbone.runtime_memory()}
            save_progress()
            print(json.dumps({'arm':label,'badgeProbe':observed['badgePosition6'],'baseUnchanged':before==after}),flush=True)
            del outputs
            backbone.mx.clear_cache();backbone.mx.synchronize();backbone.check_memory()
    finally:
        generate.BatchGenerator=native_class
    if any(label not in full_logprobs for label in reports):
        raise AssertionError('Badge completed before the fixed divergence probe position')
    comparison={}
    for label,value in full_logprobs.items():
        control=full_logprobs['sequential']
        finite=np.isfinite(value)&np.isfinite(control)
        delta=np.abs(value[finite]-control[finite])
        comparison[label]={'finiteComparedCount':int(finite.sum()),'nonfiniteCount':int((~np.isfinite(value)).sum()),
            'nonfinitePatternMatchesSequential':bool(np.array_equal(np.isfinite(value),np.isfinite(control))),
            'maxAbsoluteLogprobDifferenceVsSequential':float(delta.max()) if len(delta) else None,
            'meanAbsoluteLogprobDifferenceVsSequential':float(delta.mean()) if len(delta) else None}
    exact_repeat=bool(np.array_equal(full_logprobs['batch3'],full_logprobs['batch3-repeat']))
    np.savez(args.out/'badge-position6-logprobs.npz',**{key.replace('-','_'):value for key,value in full_logprobs.items()})
    report={'schemaVersion':1,'synthetic':True,'recordedAtUtc':datetime.now(timezone.utc).isoformat(),'modelRevision':REVISION,
        'caseId':case['caseId'],'pinnedModelPath':str(args.model),'runtime':{name:version(name) for name in ('mlx','mlx-lm')},
        'scope':'One model process; one pristine common rank state, exact preserved rows, natural greedy decode, seq/batch2/batch3/repeat3 only.',
        'priorStartupFailure':{'sessionId':43596,'error':"AttributeError: 'function' object has no attribute 'BatchGenerator'",'stage':'After model/common-prefix prefill and row preflight, before any decoding.','resolution':'Use importlib.import_module and validate BatchGenerator before model loading; corrected retry separately authorized.'},
        'modelLoadMs':model_load_ms,'prefixSha256':base['prefixSha256'],'prefixTiming':prefix_timing,'rowsPreflight':preflight,
        'headCheckpoint':receipt['checkpoints']['heads'],'baseCompleteStateHash':original_hash,'arms':reports,'logprobComparison':comparison,
        'batch3RepeatLogprobsExactlyEqual':exact_repeat,'memory':backbone.runtime_memory(),'memoryLimitBytes':6*2**30,
        'inputHashes':{str(args.prior/f'parallel-case-2-{arm}.json'):digest(args.prior/f'parallel-case-2-{arm}.json') for arm in prior},
        'limitations':['Observes normalized sample log-probabilities, not pre-normalization raw logits.',
            'Stable base-state hashes and exact prompt identities rule out source-prefix mutation, not every native cache/merge/position bug.',
            'Batch-shape floating-point reduction/kernel differences are a hypothesis unless matched-state independent forward controls distinguish them.',
            'No arithmetic precision, sampler, validator, runtime, budget, prompt or model-weight changes. Observer exists only inside this process.',
            'Single diagnosed synthetic case; no speed, quality, conversion or general exact-parity claim.']}
    report['evidence']={str(args.out/'badge-position6-logprobs.npz'):digest(args.out/'badge-position6-logprobs.npz')}
    (args.out/'probe.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--heads',type=Path,default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--prior',type=Path,default=Path('data/ranker/pretrained/parallel-decode-v1'))
    parser.add_argument('--out',type=Path,default=Path('data/ranker/pretrained/native-batch-diagnosis-v1'))
    parser.add_argument('--report',type=Path,default=Path('experiments/ranker/pretrained/NATIVE_BATCH_DIAGNOSIS.json'))
    parser.add_argument('--stage', choices=('v1','v2','v3','v4'), default='v1')
    args=parser.parse_args()
    (run_controls if args.stage in ('v2','v3','v4') else run)(args)


# Stage two is deliberately separate from the original reproducer. The original
# report remains intact; only this fixed same-shape control stage is appended.
def control_rows(rows, width):
    selected = {1: ('badge-1',), 2: ('title-1', 'badge-1'),
                3: ('heading-1', 'title-1', 'badge-1')}
    if width not in selected:
        raise ValueError('Only one/two/three actual initial lanes are in scope')
    indexed = {row['rowId']: row for row in rows}
    return [indexed[key] for key in selected[width]]


def vector_comparison(left, right):
    import numpy as np
    finite = np.isfinite(left) & np.isfinite(right)
    difference = np.abs(left[finite] - right[finite])
    return {'exact': bool(np.array_equal(left, right)),
            'finiteComparedCount': int(finite.sum()),
            'maxAbsoluteDifference': float(difference.max()) if len(difference) else None,
            'meanAbsoluteDifference': float(difference.mean()) if len(difference) else None,
            'leftNonfiniteCount': int((~np.isfinite(left)).sum()),
            'rightNonfiniteCount': int((~np.isfinite(right)).sum())}


def run_controls(args):
    import gc
    import importlib
    import numpy as np
    import torch
    from importlib.metadata import version
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION, load_corrected, all_actions
    from .fusion import encode_shared_prefix, append_generation_suffix
    from .page_plan import make_page_plan, validate_page_plan
    from .parallel_decode import build_rows, clone_prefix_state
    native = importlib.import_module('mlx_lm.generate')
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    native_class = native.BatchGenerator  # Validate before any weight load.
    stage = 'stage'+args.stage[1:]
    prefill_step_size = 512 if stage == 'stage2' else 256
    if args.out.name != 'native-batch-diagnosis-'+args.stage:
        raise ValueError('Each stage requires its separately named native-batch-diagnosis directory')
    original_bytes = args.report.read_bytes()
    previous = json.loads(original_bytes)
    if stage in previous or args.out.exists():
        raise ValueError('Preserve existing stage evidence; do not overwrite it')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'preserved-v1-report.json').write_bytes(original_bytes)
    expected = Path.cwd()/'.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots'/REVISION
    if args.model.resolve() != expected.resolve():
        raise ValueError('Only the verified pinned snapshot may be loaded')
    torch.set_num_threads(1)
    case = json.loads((args.prior/'parallel-case-plan.json').read_text())['cases'][2]
    saved = json.loads((args.prior/'parallel-case-2-sequential.json').read_text())
    heads, _, _ = load_corrected(args.heads)
    process_started = time.perf_counter()
    result = {'schemaVersion': int(args.stage[1:]), 'stage': stage,
        'nativePrefillStepSize': prefill_step_size, 'synthetic': True, 'recordedAtUtc': datetime.now(timezone.utc).isoformat(),
        'prefillObservation': 'materialize native cache state only; preserve unused lazy logits' if stage == 'stage4' else 'legacy observation',
        'status': 'running', 'runtime': {name: version(name) for name in ('mlx', 'mlx-lm')},
        'modelRevision': REVISION, 'caseId': case['caseId'], 'memoryLimitBytes': 6*2**30,
        'scope': 'One model process, widths 1/2/3, fork/independent-prefix/fresh-full, each repeated once; seven greedy responses per row; no forcing.',
        'priorReportSha256': hashlib.sha256(original_bytes).hexdigest(),
        'sourceSha256': digest(Path(__file__)), 'arms': {}, 'comparisons': {},
        'limitations': ['Static initial widths: width 2 is title/badge, whereas the original width-2 badge was admitted late.',
            f'Independent prefix prefill uses identical single-988-token backbone segmentation to the pristine source; fresh-full uses native {prefill_step_size}-token prefill and therefore changes segmentation.',
            'Distribution comparisons require exact earlier sampled context, not merely a shared prompt hash.',
            'Raw-logit observation and extra normalization are measurement-only; no model/sampler/precision or validator changes.',
            'No semantic-quality, usable-page speed or general cache-correctness conclusion follows from one synthetic case.']}
    arrays = {}
    def persist():
        if arrays:
            np.savez(args.out/'position6-arrays.npz', **arrays)
            result['arraysSha256'] = digest(args.out/'position6-arrays.npz')
        (args.out/'probe.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        previous[stage] = result
        args.report.write_text(json.dumps(previous, indent=2, allow_nan=False)+'\n')
    backbone = None
    try:
        class PeakCheckedBackbone(FrozenBackbone):
            def check_memory(self):
                super().check_memory()
                peak = self.mx.get_peak_memory()
                if peak > self.memory_budget:
                    raise MemoryError(f'Observed peak {peak} exceeds fixed {self.memory_budget}-byte experiment guard')
        backbone = (FrozenBackbone if stage == 'stage2' else PeakCheckedBackbone)(args.model, REVISION, 1536, 6)
        mx = backbone.mx
        embeddings, actions, timing, base = encode_shared_prefix(backbone, case['request'], case['catalog'], 1536)
        with torch.inference_mode():
            decision = all_actions(case['request'], case['catalog'], actions, heads(embeddings).tolist())
        del embeddings
        plan = make_page_plan(case['request'], case['catalog'], decision, case['cards'])
        validate_page_plan(plan, case['request'], case['catalog'])
        rows = build_rows(plan, case['request'], case['catalog'])
        base_hash = state_hash(base['cache'])
        suffixes = {}
        for row in rows:
            dummy = object()
            opaque = {**base, 'cache': [], 'model': dummy, 'consumed': False}
            suffix, _ = append_generation_suffix(backbone.tokenizer, row['messages'], opaque, dummy)
            ref = next(item for item in saved['rows'] if item['rowId'] == row['rowId'])
            prompt_hash = hashlib.sha256(json.dumps(row['messages'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if (prompt_hash, len(suffix)) != (ref['promptSha256'], ref['suffixTokens']):
                raise AssertionError('Fixed case prompt changed')
            suffixes[row['rowId']] = suffix
        if base['prefixSha256'] != saved['prefixSha256']:
            raise AssertionError('Fixed prefix identity changed')
        result.update(prefixSha256=base['prefixSha256'], prefixTokens=len(base['tokens']), prefixTiming=timing,
                      pristineBaseHash=base_hash, rowPrompts=[{'rowId': r['rowId'], 'suffixTokens': len(suffixes[r['rowId']]),
                      'promptSha256': next(x['promptSha256'] for x in saved['rows'] if x['rowId']==r['rowId'])} for r in rows])
        persist()
        for width in (1, 2, 3):
            members = control_rows(rows, width)
            for repeat in (0, 1):
                for mode in ('fork', 'independent', 'full'):
                    label = f'w{width}-{mode}-r{repeat}'
                    record = {'width': width, 'mode': mode, 'repeat': repeat, 'status': 'running',
                              'rowIds': [r['rowId'] for r in members], 'baseBefore': state_hash(base['cache'])}
                    result['arms'][label] = record
                    persist()
                    started = time.perf_counter()
                    caches = []
                    branch_hashes = []
                    for _ in members:
                        if mode == 'fork':
                            branch = clone_prefix_state(backbone, base)
                            cache = branch['cache']
                            del branch
                        else:
                            cache = make_prompt_cache(backbone.model)
                            if mode == 'independent':
                                # Same 988-token backbone call as the source.
                                hidden = backbone.model.model(mx.array([base['tokens']]), cache=cache)
                                mx.eval([entry.state for entry in cache]); mx.synchronize()
                                del hidden
                                backbone.check_memory()
                        branch_hashes.append(state_hash(cache))
                        caches.append(cache)
                        backbone.check_memory()
                    record['branchHashesBeforeNativeMerge'] = branch_hashes
                    record['branchMatchesPristine'] = [h == base_hash for h in branch_hashes]
                    record['branchPreparationMs'] = (time.perf_counter()-started)*1000
                    raw_observation = {}
                    class ObservedModel:
                        def __init__(self):
                            self.single_token_calls = 0
                        def __getattr__(self, name):
                            return getattr(backbone.model, name)
                        def __call__(self, tokens, *pargs, **kwargs):
                            logits = backbone.model(tokens, *pargs, **kwargs)
                            if stage == 'stage3':
                                mx.eval(logits); mx.synchronize(); backbone.check_memory()
                            elif stage == 'stage4':
                                if tokens.shape[1] == 1:
                                    mx.eval(logits)
                                else:
                                    mx.eval([entry.state for entry in kwargs['cache']])
                                mx.synchronize(); backbone.check_memory()
                            if tokens.shape[1] == 1:
                                step = self.single_token_calls
                                self.single_token_calls += 1
                                if step == 6:
                                    raw = logits[:, -1, :]
                                    normalized = raw - mx.logsumexp(raw, axis=-1, keepdims=True)
                                    raw_observation.update(dtype=str(raw.dtype), forwardStepZeroBased=step,
                                        width=int(raw.shape[0]), raw=np.array(raw.astype(mx.float32)).copy(),
                                        reconstructed=np.array(normalized.astype(mx.float32)).copy())
                            return logits
                    observed_model = ObservedModel()
                    gen = native_class(observed_model, completion_batch_size=width, prefill_batch_size=width,
                                       prefill_step_size=prefill_step_size, sampler=make_sampler(temp=0.0))
                    prompts = [((base['tokens'] if mode == 'full' else [])+suffixes[r['rowId']]) for r in members]
                    prior_tokens = [([] if mode == 'full' else list(base['tokens'])) for _ in members]
                    eos = [[int(t)] for t in backbone.tokenizer.eos_token_ids]
                    records = {r['rowId']: {'rawSampledTokenIds': [], 'finishReason': None} for r in members}
                    try:
                        uids = gen.insert(prompts, max_tokens=[7]*width, caches=caches, all_tokens=prior_tokens,
                            samplers=[make_sampler(temp=0.0) for _ in members], logits_processors=[[] for _ in members],
                            stop_sequences=[native.StopSequences(eos) for _ in members])
                        mapping = dict(zip(uids, [r['rowId'] for r in members], strict=True))
                        del caches, cache
                        while True:
                            responses = gen.next_generated()
                            mx.synchronize()
                            if not responses:
                                break
                            for response in responses:
                                row_id = mapping[response.uid]
                                row_record = records[row_id]
                                index = len(row_record['rawSampledTokenIds'])
                                row_record['rawSampledTokenIds'].append(int(response.token))
                                row_record['finishReason'] = response.finish_reason
                                if row_id == 'badge-1' and index == 6:
                                    # GenerationBatch computes one step ahead:
                                    # use raw FORWARD call 6, not the latest call.
                                    if raw_observation.get('forwardStepZeroBased') != 6 or raw_observation['width'] != width:
                                        raise AssertionError('Target raw forward observation is missing or wrong shape')
                                    badge_index = [r['rowId'] for r in members].index('badge-1')
                                    raw = raw_observation['raw'][badge_index].copy()
                                    reconstructed = raw_observation['reconstructed'][badge_index].copy()
                                    actual = np.array(response.logprobs.astype(mx.float32)).reshape(-1).copy()
                                    arrays[label+'-raw'] = raw
                                    arrays[label+'-logprobs'] = actual
                                    arrays[label+'-reconstructed'] = reconstructed
                                    top = np.argsort(raw)[-5:][::-1]
                                    row_record['probe'] = {'sampledTokenId': int(response.token), 'rawDtype': raw_observation['dtype'],
                                        'forwardStepZeroBased': 6, 'responseIndexZeroBased': index, 'actualActiveWidth': width,
                                        'alignment': vector_comparison(reconstructed, actual),
                                        'topRaw': [{'tokenId': int(t), 'text': backbone.tokenizer.decode([int(t)]),
                                                    'rawLogit': float(raw[t]), 'logprob': float(actual[t])} for t in top],
                                        'rawTopTwoMargin': float(raw[top[0]]-raw[top[1]]),
                                        'meadowMinusBRaw': float(raw[85688]-raw[417]),
                                        'meadowMinusBLogprob': float(actual[85688]-actual[417])}
                                    if not np.array_equal(reconstructed, actual):
                                        raise AssertionError('Observed raw logits do not reconstruct this Response.logprobs exactly')
                            del response, responses
                            backbone.check_memory()
                    finally:
                        gen.close()
                        del gen, observed_model
                        gc.collect(); mx.synchronize(); mx.clear_cache()
                    record.update(status='completed', rows=records, baseAfter=state_hash(base['cache']),
                                  diagnosticWallMs=(time.perf_counter()-started)*1000, memory=backbone.runtime_memory())
                    if record['baseBefore'] != base_hash or record['baseAfter'] != base_hash:
                        raise AssertionError('Pristine source state changed')
                    if any(len(r['rawSampledTokenIds']) != 7 for r in records.values()):
                        raise AssertionError('A row finished before the matched-context probe')
                    persist()
                    print(json.dumps({'arm': label, 'probe': records['badge-1']['probe'],
                                      'baseUnchanged': True, 'branchMatchesPristine': record['branchMatchesPristine']}), flush=True)
                    del raw_observation
                    gc.collect(); mx.synchronize(); mx.clear_cache(); backbone.check_memory()
        for width in (1, 2, 3):
            for repeat in (0, 1):
                reference = f'w{width}-fork-r{repeat}'
                for mode in ('independent', 'full'):
                    candidate = f'w{width}-{mode}-r{repeat}'
                    same_context = all(result['arms'][reference]['rows'][key]['rawSampledTokenIds'][:6] ==
                                       result['arms'][candidate]['rows'][key]['rawSampledTokenIds'][:6]
                                       for key in result['arms'][reference]['rows'])
                    result['comparisons'][candidate+'-vs-fork'] = {'sameEarlierContextAllRows': same_context,
                        'raw': vector_comparison(arrays[reference+'-raw'], arrays[candidate+'-raw']),
                        'logprobs': vector_comparison(arrays[reference+'-logprobs'], arrays[candidate+'-logprobs'])}
            for mode in ('fork', 'independent', 'full'):
                left, right = f'w{width}-{mode}-r0', f'w{width}-{mode}-r1'
                result['comparisons'][right+'-vs-repeat0'] = {'allRawTokensExactlyEqual': all(
                    result['arms'][left]['rows'][key]['rawSampledTokenIds'] == result['arms'][right]['rows'][key]['rawSampledTokenIds']
                    for key in result['arms'][left]['rows']),
                    'raw': vector_comparison(arrays[left+'-raw'], arrays[right+'-raw']),
                    'logprobs': vector_comparison(arrays[left+'-logprobs'], arrays[right+'-logprobs'])}
        for width in (2, 3):
            for mode in ('fork', 'independent', 'full'):
                reference, candidate = f'w1-{mode}-r0', f'w{width}-{mode}-r0'
                a = result['arms'][reference]['rows']['badge-1']['rawSampledTokenIds']
                b = result['arms'][candidate]['rows']['badge-1']['rawSampledTokenIds']
                result['comparisons'][candidate+'-vs-width1'] = {'sameEarlierBadgeContext': a[:6]==b[:6],
                    'raw': vector_comparison(arrays[reference+'-raw'], arrays[candidate+'-raw']),
                    'logprobs': vector_comparison(arrays[reference+'-logprobs'], arrays[candidate+'-logprobs'])}
        result['status'] = 'completed'
        result['memory'] = backbone.runtime_memory()
    except BaseException as error:
        result['status'] = 'failed-stopped'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
        for arm in result['arms'].values():
            if arm['status'] == 'running':
                arm['status'] = 'failed'
                arm['failure'] = result['failure']
        if backbone is not None:
            result['memory'] = backbone.runtime_memory()
        persist()
        raise
    finally:
        result['totalWallMs'] = (time.perf_counter()-process_started)*1000
        persist()


if __name__ == '__main__':
    main()
