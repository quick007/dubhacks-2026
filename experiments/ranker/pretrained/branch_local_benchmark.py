"""One six-card copy-stage locality comparison on explicitly inspected cases."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time
from .semantic_copy_training import ROOT,DATA,MODEL,dump,digest,records,disabled,null_context

PACKET=ROOT/'BRANCH_LOCAL_CASES.json'


def freeze():
    from .branch_local_copy import pending_plan
    from transformers import AutoTokenizer
    from .branch_local_copy import build_layout,token_inventory
    if PACKET.exists():raise ValueError('Preserve frozen packet')
    source=ROOT/'SEMANTIC_COPY_DIRECT_RESULTS.json';original=json.loads(source.read_text())
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False)
    cases=[]
    for split in ('dev','test'):
        for case in records(DATA/f'cases.{split}.jsonl'):
            arm=next(a for a in original['arms'] if a['split']==split and a['caseId']==case['caseId'] and a['arm']=='frozen')
            plan=pending_plan(arm['pagePlan']);root,rows,_=build_layout(plan,case['request'],case['catalog'])
            cases.append({'caseId':case['caseId'],'split':split,'request':case['request'],'catalog':case['catalog'],
                'plan':plan,'inventory':token_inventory(tokenizer,root,rows)})
    dump(PACKET,{'schemaVersion':1,'sourceResultSha256':digest(source),'sourceSourceCommit':original['executionCommit'],
        'cases':cases,'cards':6,'fixedCases':6,'arms':['frozen','global-copy-adapter'],
        'casesAlreadyInspected':True,'rankingExecution':False,'scope':'copy-stage only; new branch-local representation',
        'width':3,'caps':{'heading':16,'title':24,'badge':8},'totalBudgetPerPage':224,
        'learnedCheckpoint':str(Path('data/ranker/pretrained/semantic-copy-direct-v1/adapter.safetensors')),
        'learnedCheckpointSha256':original['checkpoint']['sha256'],
        'layoutSourceSha256':digest(ROOT/'branch_local_copy.py')})
    print(json.dumps({'frozenCases':len(cases),'maxFullContextWithCap':max(r['fullPromptTokens']+r['maxTokens'] for c in cases for r in c['inventory']['rows']),
        'maxSuffix':max(r['suffixTokens'] for c in cases for r in c['inventory']['rows'])}),flush=True)


def run(args):
    import psutil
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION
    from .branch_local_copy import build_layout,prepare_root,native_root_controls,run_locked_arm,token_inventory
    from mlx_lm.tuner.lora import LoRALinear
    from .catalog_fusion.runtime import assert_model_files
    if args.out.exists():raise ValueError('Preserve previous attempt')
    packet=json.loads(PACKET.read_text())
    if digest(ROOT/'branch_local_copy.py')!=packet['layoutSourceSha256']:raise ValueError('Frozen layout changed')
    checkpoint=Path(packet['learnedCheckpoint'])
    if digest(checkpoint)!=packet['learnedCheckpointSha256']:raise ValueError('Learned checkpoint changed')
    if psutil.virtual_memory().available<7*2**30:raise MemoryError('Less than declared7GiB available')
    args.out.mkdir(parents=True);started=time.perf_counter()
    report={'status':'started','packetSha256':digest(PACKET),'sourceSha256':digest(Path(__file__)),
        'executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'memoryGuardGiB':7,'fixedCases':6,'fixedArms':12,'cardsEach':6,'width':3,
        'casesAlreadyInspected':True,'endToEndRequestTiming':False,'learnedCheckpointSha256':packet['learnedCheckpointSha256'],
        'controls':{},'arms':[]}
    def save():
        report['wallMs']=(time.perf_counter()-started)*1000;dump(args.out/'results.json',report)
    save()
    class PeakBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory()>self.memory_budget:raise MemoryError('Observed peak exceeded declared7GiB')
    try:
        provenance=json.loads(Path('data/ranker/pretrained/catalog-user-fusion-v1-attempt-1/model-verification.json').read_text())
        assert_model_files(MODEL,provenance)
        bb=PeakBackbone(MODEL,REVISION,2048,7);model=bb.model;report['loadMs']=bb.load_ms
        adapter=LoRALinear.from_base(model.layers[-1].mlp.down_proj,r=4,scale=20,dropout=0)
        saved=bb.mx.load(str(checkpoint))
        if set(saved)!={'lora_a','lora_b'}:raise ValueError('Unexpected adapter members')
        adapter.load_weights(list(saved.items()),strict=False);adapter.freeze()
        model.layers[-1].mlp.down_proj=adapter;model.eval();bb.mx.eval(adapter.parameters())
        first=packet['cases'][0];root,rows,_=build_layout(first['plan'],first['request'],first['catalog'])
        for name in packet['arms']:
            dest=args.out/(name+'-controls');dest.mkdir()
            with disabled(model,adapter) if name=='frozen' else null_context():
                state,_=prepare_root(bb,root);control=native_root_controls(bb,state,root,rows,dest)
            report['controls'][name]=control;save()
            if control['status']!='completed':raise AssertionError('Branch-local native controls failed: '+str(control.get('abort')))
            del state;gc.collect();bb.mx.clear_cache()
        for index,case in enumerate(packet['cases']):
            root,rows,_=build_layout(case['plan'],case['request'],case['catalog'])
            if token_inventory(bb.tokenizer,root,rows)!=case['inventory']:raise ValueError('Actual native token inventory changed')
            for name in packet['arms'] if index%2==0 else list(reversed(packet['arms'])):
                begin=time.perf_counter()
                with disabled(model,adapter) if name=='frozen' else null_context():result=run_locked_arm(bb,case['plan'],case['request'],case['catalog'])
                result['externalCopyStageMs']=(time.perf_counter()-begin)*1000
                result['pagePlan']['method']['headingsAndTitles']='same quantized backbone, adapter '+('disabled' if name=='frozen' else 'enabled')+'; branch-local evidence'
                report['arms'].append({'caseId':case['caseId'],'split':case['split'],'arm':name,**result});save()
                print(json.dumps({'caseId':case['caseId'],'split':case['split'],'arm':name,'quality':result['quality'],'copyStageMs':result['externalCopyStageMs']}),flush=True)
                gc.collect();bb.mx.clear_cache()
        if len(report['arms'])!=12:raise AssertionError('Incomplete fixed comparative roster')
        report['status']='completed';report['memory']=bb.runtime_memory();save()
    except Exception as exc:
        report['status']='blocked';report['blocker']={'type':type(exc).__name__,'message':str(exc)}
        if 'bb' in locals():report['memory']=bb.runtime_memory()
        save();raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('freeze','run'));p.add_argument('--out',type=Path);args=p.parse_args()
    if args.action=='freeze':freeze()
    else:
        if args.out is None:p.error('run requires --out')
        run(args)


if __name__=='__main__':main()
