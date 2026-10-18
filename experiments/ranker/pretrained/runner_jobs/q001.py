"""Bounded external Mac badge-row job. CPU preflight is the default action."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

INPUT=Path(__file__).with_name('Q001_INPUT.json')
ROOT=Path(__file__).resolve().parents[4]
MODEL_RECEIPT=Path(__file__).with_name('Q001_MODEL_RECEIPT.json')
MODEL_RECEIPT_SHA='0151d0c2dd700673d4e1a0fc0f7bc09e95127fcc13200fc17fee6d814d4bd42a'
INPUT_SHA='55d78d540bfed50423184278b3c5196049ca92e53bd43a2c0ff8945f7b026b6e'
REVISION='8b2b98c00a6b4d291155e4890773ca8f769aee53'
IDS={'evidence-r-e1f73b869dc810bc15','evidence-r-c936cadc9fcdc5b182',
     'evidence-control-no-history-no-preferences'}


def hashed(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def file_digest(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()


def badge_score(raw,finish,record):
    """Existing badge lexical/support policy plus exact isolated-row protocol."""
    from ..fusion import _field_claim_errors,_normalized
    card=record['selectedCard'];product=record['catalog'][card['productId']]
    support=[x.rsplit(':',1)[1] for x in card['approvedFactIds']]
    errors=[];lines=raw.strip().splitlines()
    match=re.fullmatch(r'B1: (.+)',lines[0].strip()) if len(lines)==1 else None
    if not match:errors.append('unexpected-or-missing-single-B1-line')
    global_ok=(finish=='stop' and not any(ord(c)<32 and c not in '\n\r\t' or ord(c)==127 for c in raw)
        and not re.search(r'[<>`{}\[\]]|https?://|www\.|\*|__',raw,re.I))
    if not global_ok:errors.append('unfinished-or-markup-output')
    badge=match[1].strip() if match else None
    if not support:
        if badge!='NONE':errors.append('NONE-required')
    elif badge!='NONE':
        if badge is None or not 5<=len(badge)<=90:errors.append('badge-length')
        if badge is not None:errors.extend(_field_claim_errors(badge,product,record['catalog']))
        if not any(re.search(r'\b'+re.escape(_normalized(tag))+r'\b',_normalized(badge or '')) for tag in support):
            errors.append('unsupported-or-unresponsive-badge')
    return {'acceptedBadgeRow':not errors,'acceptedGeneratedBadge':not errors and badge!='NONE',
        'acceptedNone':not errors and badge=='NONE','rejections':errors,'badge':badge,
        'support':support,'completePageEvaluated':False}


def preflight():
    if hashlib.sha256(INPUT.read_bytes()).hexdigest()!=INPUT_SHA:raise ValueError('Q001 input hash changed')
    if file_digest(MODEL_RECEIPT)!=MODEL_RECEIPT_SHA:raise ValueError('Model receipt changed')
    for name,expected in {'backbone.py':'551dd6204419b2cee7b6f98cbc8ea239ca371cac6d2cf992bb7838f2d657aedc',
        'fusion.py':'ac50c90102709b11dbd8293ca09b6943035c28c872fc5d6a440d0c0c83dfe5df'}.items():
        if file_digest(INPUT.parent.parent/name)!=expected:raise ValueError('Pinned helper changed: '+name)
    packet=json.loads(INPUT.read_text());records=packet['records']
    expected={(cid,v) for cid in IDS for v in ('baseline','badge-evidence-only')}
    if len(records)!=6 or {(r['caseId'],r['variant']) for r in records}!=expected:
        raise ValueError('Require exactly the six fixed badge arms')
    for record in records:
        row=record['row']
        if record['modelRevision']!=REVISION or row['kind']!='badge' or row['fields']!=['B1'] or row['maxTokens']!=12:
            raise ValueError('Scope, model or token budget changed')
        for key,hashkey in (('messages','promptSha256'),('fullMessages','fullMessagesSha256')):
            if hashed(row[key])!=row[hashkey]:raise ValueError('Prompt hash changed')
    return packet


def summarize(records):
    return {v:{'fixedDenominator':3,
        'completed':sum(r['variant']==v and r['status']=='completed-attempt' for r in records),
        'acceptedBadgeRows':sum(r['variant']==v and r['status']=='completed-attempt' and
            r.get('quality',{}).get('acceptedBadgeRow',False) for r in records)}
        for v in ('baseline','badge-evidence-only')}


def execute(args,packet):
    import gc
    import platform
    import sys
    from importlib.metadata import version
    if platform.system()!='Darwin' or platform.machine()!='arm64':raise RuntimeError('Existing Apple silicon MLX runner required')
    versions={n:version(n) for n in ('mlx','mlx-lm','transformers','torch','numpy','psutil')}
    expected={'mlx':'0.32.4','mlx-lm':'0.32.2','transformers':'5.20.1','torch':'2.14.1','numpy':'2.5.4','psutil':'7.2.2'}
    if versions!=expected:raise RuntimeError('Pinned environment differs; report BLOCKED rather than install or substitute')
    model=args.model.resolve()
    if model.name!=REVISION or not (model/'model.safetensors.index.json').is_file():
        raise ValueError('Already-present pinned local snapshot required; no download')
    model_receipt=json.loads(MODEL_RECEIPT.read_text())
    for name,expected_file in model_receipt['files'].items():
        path=model/name
        if not path.is_file() or path.stat().st_size!=expected_file['bytes'] or file_digest(path)!=expected_file['sha256']:
            raise ValueError('Pinned model/tokenizer bytes differ: '+name+'; do not download or substitute')
    root=(ROOT/'experiments/ranker/pretrained/quality_offload').resolve()
    output=args.out.resolve()
    if not output.is_relative_to(root) or output==root or output.exists():
        raise ValueError('Use one new Q001 output directory within external-owned quality_offload')
    if args.expected_commit!=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip():
        raise ValueError('Run from the exact emailed pinned commit; no automatic checkout/reset')
    trusted_paths=[str(p.relative_to(ROOT)) for p in (Path(__file__),INPUT,MODEL_RECEIPT)]
    if subprocess.run(['git','diff','--quiet','HEAD','--',*trusted_paths],cwd=ROOT).returncode:
        raise ValueError('Trusted job sources are dirty; preserve them and report BLOCKED')
    import psutil
    before=psutil.virtual_memory()._asdict()
    if before['available']<7*2**30:raise MemoryError('Less than explicit7GiB guard available')
    from ..backbone import FrozenBackbone
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    class PeakBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory()>self.memory_budget:raise MemoryError('Peak exceeded7GiB guard')
    output.mkdir(parents=True);started=time.perf_counter();records=[];abort=None;backbone=None
    report={'jobId':'Q001','inputSha256':INPUT_SHA,'executionCommit':args.expected_commit,
        'runnerSourceSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'modelReceiptSha256':MODEL_RECEIPT_SHA,'verifiedModelFiles':model_receipt['files'],
        'hardware':{'machine':platform.machine(),'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                    'physicalRamBytes':before['total']},
        'python':sys.version,'runtime':versions,'modelRevision':REVISION,
        'modelConfigSha256':hashlib.sha256((model/'config.json').read_bytes()).hexdigest(),
        'modelIndexSha256':hashlib.sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest(),
        'memoryBefore':before,'records':records,'sampling':{'temperature':0.0,'enableThinking':False,
        'maxTokens':12,'prefillStepSize':256,'stop':'native tokenizer EOS only','freshCompleteCachePerArm':True},
        'limitations':['Three inspected diagnostic cases per variant, not held-out.',
          'Badge-only acceptance, no page acceptance or end-to-end serving latency.',
          'Fresh full prompts may differ numerically from historical segmented shared-prefix generation.',
          'External hardware timings are not target-Mac latency measurements.']}
    try:
        backbone=PeakBackbone(model,REVISION,2304,7)
        report['modelLoadMs']=(time.perf_counter()-started)*1000
    except Exception as error:abort=f'{type(error).__name__}: {error}'
    for index,item in enumerate(packet['records']):
        row=item['row'];result={'caseId':item['caseId'],'variant':item['variant'],'rowId':row['rowId'],
            'fullMessagesSha256':row['fullMessagesSha256'],'promptSha256':row['promptSha256'],
            'status':'not-attempted','rawSampledTokenIds':[],'text':'','finishReason':None}
        if not abort:
            begin=time.perf_counter();cache=None
            try:
                tokens=backbone.tokenizer.apply_chat_template(row['fullMessages'],tokenize=True,
                    return_dict=False,add_generation_prompt=True,enable_thinking=False)
                if len(tokens)>2304:raise ValueError('Full prompt exceeds fixed2304 token cap; refuse truncation')
                result.update(promptTokenIds=tokens,promptTokenIdsSha256=hashed(tokens),promptTokenCount=len(tokens))
                cache=make_prompt_cache(backbone.model);backbone.mx.synchronize();gen_start=time.perf_counter();last=None
                for last in stream_generate(backbone.model,backbone.tokenizer,prompt=tokens,max_tokens=12,
                    sampler=make_sampler(temp=0.0),prompt_cache=cache,prefill_step_size=256):
                    result['rawSampledTokenIds'].append(int(last.token));result['text']+=last.text
                    result['finishReason']=last.finish_reason;backbone.check_memory()
                backbone.mx.synchronize()
                if last is None or len(result['rawSampledTokenIds'])!=last.generation_tokens:
                    raise AssertionError('Native token accounting mismatch')
                backbone.check_memory()
                result.update(status='completed-attempt',nativeGenerationWallMs=(time.perf_counter()-gen_start)*1000,
                    sampledStepsIncludingEOS=len(result['rawSampledTokenIds']),
                    rawFinalTokenIsEOS=result['rawSampledTokenIds'][-1] in backbone.tokenizer.eos_token_ids,
                    quality=badge_score(result['text'],result['finishReason'],item))
            except Exception as error:
                abort=f'{type(error).__name__}: {error}';result.update(status='failed',error=abort)
            finally:
                result['totalAttemptWallMs']=(time.perf_counter()-begin)*1000
                result['memory']=backbone.runtime_memory();del cache
                try:gc.collect();backbone.mx.synchronize();backbone.mx.clear_cache()
                except Exception as error:
                    abort=f'cleanup {type(error).__name__}: {error}';result['cleanupError']=abort
                    result['status']='failed'
        if result['status']=='not-attempted':result['reason']=abort
        records.append(result);dump(output/f'{index+1:02d}-{item["variant"]}.json',result)
        print(json.dumps({k:result.get(k) for k in ('caseId','variant','status','quality','error','reason')}),flush=True)
    report.update(abort=abort,totalWallMs=(time.perf_counter()-started)*1000,memoryAfter=psutil.virtual_memory()._asdict(),
        memory=backbone.runtime_memory() if backbone else None,
        variants=summarize(records))
    dump(output/'results.json',report)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('preflight','run'))
    parser.add_argument('--model',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--expected-commit');args=parser.parse_args();packet=preflight()
    if args.action=='run':
        if not all((args.model,args.out,args.expected_commit)):parser.error('run needs --model, --out and --expected-commit')
        execute(args,packet)
    else:print(json.dumps({'jobId':'Q001','inputSha256':INPUT_SHA,'arms':6,'casesPerVariant':3,'nativeExecution':False}))


if __name__=='__main__':main()
