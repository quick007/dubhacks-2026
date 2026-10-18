"""One source-motivated native auxiliary-feature parity test; zero updates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time
from .semantic_copy_training import ROOT,DATA,MODEL,dump,digest,records,disabled,cache_proof


def run(out):
    import psutil
    import numpy as np
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION
    from .fusion import encode_shared_prefix
    from .shared_evidence_experiment import append_evidence
    from .semantic_copy_gradient_features import probe_gradient_features,GradientFeatureProbeError
    from .semantic_copy_training import teacher_tokens
    from mlx_lm.tuner.lora import LoRALinear
    if out.exists():raise ValueError('Preserve prior diagnostic')
    if psutil.virtual_memory().available<7*2**30:raise MemoryError('Less than declared7GiB available')
    out.mkdir(parents=True);started=time.perf_counter()
    report={'status':'started','executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'sourceSha256':digest(Path(__file__)),'probeSourceSha256':digest(ROOT/'semantic_copy_gradient_features.py'),
        'nativeUpdates':0,'memoryGuardGiB':7,'trainingAuthorizedByThisProbe':False,
        'oldFailuresPreserved':['SEMANTIC_COPY_SMOKE_V2.json','SEMANTIC_COPY_SMOKE_V3.json','SEMANTIC_COPY_SMOKE_V4.json']}
    def save():
        report['wallMs']=(time.perf_counter()-started)*1000;dump(out/'results.json',report)
    save()
    class PeakBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory()>self.memory_budget:raise MemoryError('Observed peak exceeded declared7GiB')
    try:
        bb=PeakBackbone(MODEL,REVISION,1792,7);model=bb.model;bb.mx.random.seed(19074)
        adapter=LoRALinear.from_base(model.layers[-1].mlp.down_proj,r=4,scale=20,dropout=0)
        model.layers[-1].mlp.down_proj=adapter;model.eval()
        adapter.lora_b=bb.mx.full(adapter.lora_b.shape,.0001,dtype=adapter.lora_b.dtype);bb.mx.eval(adapter.parameters())
        case=records(DATA/'cases.train.jsonl')[0];field=records(DATA/'fields.train.jsonl')[0]
        with disabled(model,adapter):
            _,_,_,rank=encode_shared_prefix(bb,case['request'],case['catalog'],1792)
            state,_=append_evidence(bb,rank,field['messages'][0])
        contract=teacher_tokens(bb.tokenizer,state['messages']+field['messages'][-1:],field['targetValue'],field['maxTokens'])
        report['cacheProof']=cache_proof(bb,adapter,{'prompt':contract['promptTokens'],'forcedContinuation':contract['targets'][:3]});save()
        try:receipt,item=probe_gradient_features(bb,adapter,state,field)
        except GradientFeatureProbeError as exc:
            report['probe']=exc.report
            raise
        report['probe']=receipt;report['status']=receipt['status']
        if item is not None:
            artifact=out/'gradient-features.npz';np.savez(artifact,**{k:item[k] for k in ('x','h','y')})
            metadata={k:v for k,v in item.items() if k not in ('x','h','y')};dump(out/'feature-metadata.json',metadata)
            report['featureArtifactSha256']=digest(artifact)
        report['memory']=bb.runtime_memory();save()
        print(json.dumps({'status':report['status'],'nativeUpdates':0,'comparisons':receipt.get('comparisons'),'memory':report['memory']}),flush=True)
    except Exception as exc:
        report['status']='blocked';report['blocker']={'type':type(exc).__name__,'message':str(exc)}
        if 'bb' in locals():report['memory']=bb.runtime_memory()
        save();raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)


if __name__=='__main__':main()
