"""Six fixed heading-prompt attempts on one local resident model, after controls."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time


def run(args):
    import torch
    from importlib.metadata import version
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION, digest, dump, load_corrected
    from .heading_experiment import diagnosis_cases, heading_arm
    from .parallel_benchmark import selected_ids, validate_gate
    from .parallel_semantic_benchmark import validate_controls
    control = json.loads(args.controls.read_text())
    validate_controls(control)
    if control['nativePrefillStepSize'] != 256:
        raise ValueError('This fixed experiment requires actual 256-prefill controls')
    if control['sourceSha256'] != digest(Path(__file__).with_name('native_batch_diagnosis.py')):
        raise ValueError('Control source changed')
    if control['arraysSha256'] != digest(args.controls.parent/'position6-arrays.npz'):
        raise ValueError('Control arrays changed')
    runtime = {name:version(name) for name in ('mlx','mlx-lm')}
    if runtime != control['runtime']:
        raise ValueError('Control runtime changed')
    sources = {name:digest(Path(__file__).with_name(name)) for name in ('parallel_decode.py','test_parallel_decode.py')}
    gate = json.loads(args.compatibility.read_text()); validate_gate(gate,runtime,sources)
    if args.out.exists():
        raise ValueError('Preserve all previous output; no overwrite/retry')
    cases = diagnosis_cases(json.loads(Path('data/ranker/pretrained/parallel-decode-v1/parallel-case-plan.json').read_text())['cases'])
    dump(args.out/'case-plan.json', {'cases':cases,'arms':['baseline','compact'],'nativePrefillStepSize':256,
        'controlSha256':digest(args.controls),'compatibility':gate,
        'helperSha256':digest(Path(__file__).with_name('heading_experiment.py')),
        'order':'Baseline first on cases0/2, compact first on case1; one attempt each.'})
    class PeakCheckedBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory() > self.memory_budget:
                raise MemoryError(f'Observed peak {self.mx.get_peak_memory()} exceeds experiment guard {self.memory_budget}')
    start = time.perf_counter()
    torch.set_num_threads(1); heads,_,receipt = load_corrected(args.heads)
    backbone = PeakCheckedBackbone(args.model,REVISION,1536,6)
    records,abort = [],None
    for index,case in enumerate(cases):
        record = {'caseId':case['caseId'],'arms':{}}
        for compact in ((False,True) if index%2==0 else (True,False)):
            name = 'compact' if compact else 'baseline'
            if abort:
                value = {'status':'not-attempted','reason':abort}
            else:
                value = heading_arm(backbone,heads,case['request'],case['catalog'],1,compact=compact,
                                    width=1,prefill_step_size=256)
                if value['status'] != 'completed-attempt':
                    abort = value.get('error','native failure')
                elif selected_ids(value['pagePlan']) != selected_ids(json.loads(Path(f'data/ranker/pretrained/parallel-decode-v1/parallel-case-{index}-sequential.json').read_text())['pagePlan']):
                    value['selectionFailure'] = True; abort = 'Selected product/image changed'
            record['arms'][name] = value
            dump(args.out/f'case-{index}-{name}.json',value)
            if value['status'] != 'not-attempted':
                print(json.dumps({'case':index,'arm':name,'status':value['status'],'quality':value.get('quality'),
                    'heading':value.get('headingDiagnostics'),'abort':abort}),flush=True)
                try:
                    gc.collect();backbone.mx.synchronize();backbone.mx.clear_cache();backbone.check_memory()
                except Exception as error:
                    value['cleanupError'] = f'{type(error).__name__}: {error}'
                    abort = abort or value['cleanupError'];dump(args.out/f'case-{index}-{name}.json',value)
        if all(record['arms'][key]['status']=='completed-attempt' for key in ('baseline','compact')):
            left,right = [record['arms'][key] for key in ('baseline','compact')]
            a,b = [{r['rowId']:r for r in arm['rows'] if r['kind']!='heading'} for arm in (left,right)]
            record['cardPromptBudgetAndTokens'] = {key:{'samePromptAndBudget':a[key]['promptSha256']==b[key]['promptSha256'] and a[key]['maxTokens']==b[key]['maxTokens'],
                'exactRawTokens':a[key]['rawSampledTokenIds']==b[key]['rawSampledTokenIds']} for key in a}
            if not all(v['samePromptAndBudget'] for v in record['cardPromptBudgetAndTokens'].values()):
                abort = 'Heading-only intervention changed card prompt/budget'
        records.append(record);dump(args.out/'progress.json',{'records':records,'abort':abort})
    report={'schemaVersion':1,'synthetic':True,'training':False,'validatorChanges':False,'nativePrefillStepSize':256,
        'modelRevision':REVISION,'headCheckpoint':receipt['checkpoints']['heads'],'records':records,'abort':abort,
        'totalWallMs':(time.perf_counter()-start)*1000,'memory':backbone.runtime_memory(),
        'controlSha256':digest(args.controls),'compatibility':gate,
        'limitations':['Three previously inspected one-card cases, six attempts, one fixed prompt variant; no held-out quality estimate.',
            'Only heading prompts differ within pair. All token budgets/EOS stopping/card prompts/strict validators stay fixed.',
            '256-prefill segmentation differs from historical512 outputs; baseline is freshly measured at256.',
            'Global rejection can mask individually valid fields; lexical acceptance does not establish coherence.',
            'No output rewriting or retries; remaining arms stay unattempted on native/memory failure.']}
    dump(args.out/'benchmark.json',report)
    report['evidence']={str(p):digest(p) for p in (args.out/'benchmark.json',args.out/'case-plan.json',args.controls)}
    dump(Path('experiments/ranker/pretrained/HEADING_EXPERIMENT_RESULTS.json'),report)


def main():
    from .evidence_plan import REVISION
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--controls',type=Path,default=Path('data/ranker/pretrained/native-batch-diagnosis-v3/probe.json'))
    parser.add_argument('--compatibility',type=Path,default=Path('data/ranker/pretrained/parallel-decode-v2/native-compatibility.json'))
    parser.add_argument('--heads',type=Path,default=Path('data/ranker/pretrained/evidence-plan-v1'))
    parser.add_argument('--out',type=Path,default=Path('data/ranker/pretrained/heading-diagnosis-v1'))
    args=parser.parse_args()
    expected=Path('.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots')/REVISION
    if args.model.resolve()!=expected.resolve():parser.error('Use the verified local pinned snapshot')
    run(args)


if __name__=='__main__':
    main()
