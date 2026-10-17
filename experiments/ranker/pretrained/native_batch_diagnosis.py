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
    run(parser.parse_args())


if __name__=='__main__':main()
