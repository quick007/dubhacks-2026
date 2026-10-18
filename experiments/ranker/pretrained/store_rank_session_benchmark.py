"""Root-only resident actual-store action/seam diagnostic, no HTTP/UI mutation."""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import statistics
import time

from .catalog_fusion.generalization_v2 import native_owner_lock
from .catalog_fusion.runtime import NativeEncoder, write_json, verify_model
from .store_catalog_v1 import canonical
from .store_rank_session import StoreRankSession
from .store_copy_diagnostic import completed_copy_adapter


def run(rank_run, public_path, copy_report, out):
    out=Path(out)
    if out.exists():raise FileExistsError('Preserve prior attempt')
    started=time.perf_counter();rank_run=Path(rank_run)
    packet=json.loads((rank_run/'packet.json').read_text());model=packet['model']['modelPath']
    cases=[x for x in json.loads(Path(public_path).read_text()) if x['split']=='evaluation']
    if len(cases)!=16:raise ValueError('Fixed16 known-catalog development replay required')
    copy_receipt=completed_copy_adapter(copy_report)
    with native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'),'root'):
        out.mkdir(parents=True);encoder=None;results=[]
        report={'status':'started','fixedReplayCases':16,'fixedSeamCases':4,'fixedPolicyCases':4,'results':results,
            'scope':'resident model-side rank/actions and exact copy-adapter-disabled seam; no generated copy/browser/UI measurement',
            'sources':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['store_rank_session.py','store_rank_session_benchmark.py']}}
        try:
            report['modelVerification']=verify_model(model,packet['model']);write_json(out/'MODEL_VERIFIED.json',report['modelVerification'])
            encoder=NativeEncoder(model);session=StoreRankSession(encoder.backbone,rank_run)
            for case in cases:
                result=session.rank_request(case['request']);rid=case['caseId']
                reference=session.report['results']['evaluation']['learned']['cases'][rid]
                if result['scores16']!=reference['scores'] or result['productIndices']!=reference['topSix']['productIndices']:
                    raise ValueError('Resident action score/order parity differs from trained run')
                results.append({'caseId':rid,'arm':'frozen','result':result})
            from mlx_lm.tuner.lora import LoRALinear
            from .semantic_copy_training import parameter_hashes
            bb=encoder.backbone;model_obj=bb.model;before=parameter_hashes(model_obj.language_model)
            adapter=LoRALinear.from_base(model_obj.layers[-1].mlp.down_proj,r=4,scale=20,dropout=0)
            model_obj.layers[-1].mlp.down_proj=adapter
            weights=bb.mx.load(copy_receipt['checkpointPath']);adapter.load_weights(list(weights.items()),strict=False)
            adapter.freeze();model_obj.eval();bb.mx.eval(adapter.parameters());bb.mx.synchronize()
            for case in cases[:4]:
                result=session.rank_request(case['request'],copy_adapter=adapter)
                reference=next(x['result'] for x in results if x['caseId']==case['caseId'])
                if result['scores16']!=reference['scores16'] or result['cards']!=reference['cards']:
                    raise ValueError('Copy adapter disabled rank/action seam is not exact')
                results.append({'caseId':case['caseId'],'arm':'copy-adapter-disabled','result':result,'frozenScoreAndActionsExact':True})
            base={'query':'Trail mug','preferences':[],'itemBudget':None,'hardRequirements':[]}
            probes=[('exact-name',base),('blocked-exact-name',{**base,'itemBudget':23}),
              ('hard-color',{**base,'hardRequirements':[{'attribute':'color','value':'Cobalt'}]}),
              ('unknown-stock',{**base,'hardRequirements':[{'attribute':'stock','value':'in-stock'}]})]
            for name,query in probes:
                result=session.rank_request(query,copy_adapter=adapter)
                if name=='exact-name' and (not result['exactNamePolicy']['applied'] or result['productIndices'][0]!=0):raise ValueError('Exact name policy failed')
                if name=='blocked-exact-name' and 0 in result['productIndices']:raise ValueError('Exact name bypassed budget')
                if name=='hard-color' and any(c['facts']['selectedColor']!='Cobalt' for c in result['cards']):raise ValueError('Hard color variant binding failed')
                if name=='unknown-stock' and (result['productIndices'] or result['shortfall']!=6):raise ValueError('Unknown stock did not fail closed')
                results.append({'caseId':name,'arm':'policy','result':result})
            after={k.replace('.31.mlp.down_proj.linear.','.31.mlp.down_proj.'):v for k,v in parameter_hashes(model_obj.language_model).items()}
            if before!=after or len(before)!=927:raise ValueError('Frozen tensor roster changed')
            report.update(status='completed',nativeForwards=24,base927HashesUnchanged=True,exactSeamCases=4,
                residentLoadSeconds=encoder.load_seconds,memory=encoder.memory(),
                replayMedianFreshRankActionsMs=statistics.median(x['result']['freshRankToSerializedActionsMs'] for x in results if x['arm']=='frozen'))
        except BaseException as exc:
            report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
            raise
        finally:
            if encoder is not None:encoder.close()
            report['runWallSeconds']=time.perf_counter()-started;write_json(out/'report.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--rank-run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    p.add_argument('--copy-report',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    z=run(a.rank_run,a.public,a.copy_report,a.out);print(canonical({'status':z['status'],'nativeForwards':z['nativeForwards'],'medianMs':z['replayMedianFreshRankActionsMs']}))
