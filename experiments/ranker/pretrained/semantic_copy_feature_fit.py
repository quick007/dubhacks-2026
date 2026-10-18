"""Prospective fixed 504-row fit, gated on every gradient-mode feature row.

--preflight uses only the pinned local tokenizer. Native extraction, fitting
and evaluation require a separate explicit invocation owned by the parent.
No inference uses teacher targets. Fixture selection is locked and unlabeled.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

ROOT = Path('experiments/ranker/pretrained')
DATA = ROOT / 'semantic_copy_pairs'
MODEL = Path('.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53')
MODEL_SHARDS = {
    'model-00001-of-00002.safetensors':(5349771222,'a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd'),
    'model-00002-of-00002.safetensors':(600449850,'b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d')}
CONFIG = {'seed':19076, 'epochs':1, 'updates':504, 'rank':4, 'scale':20.,
    'learningRate':.0001, 'memoryGiB':7, 'maximumRankingPrefix':1792,
    'maximumContext':4096, 'maximumSuffix':256, 'prefillStep':256,
    'singleHostFeatureBudgetBytes':32*2**20, 'totalFeatureArtifactBudgetBytes':2*2**30,
    'featureArchiveReserveBytes':2**20, 'residentHostFeatureRows':1,
    'fieldWidth':3, 'fieldsPerCase':14, 'outputCapPerCase':224,
    'selection':'final fixed step; no dev/test checkpoint selection',
    'order':'one seeded global permutation of all 504 individual rows',
    'gates':'all 504 exact nonzero-B native/auxiliary/feature loss and A/B gradients, no tolerance'}
SOURCES = ('semantic_copy_feature_fit.py','semantic_copy_gradient_features.py','semantic_copy_tokens.py',
    'semantic_copy_training.py','semantic_copy_data/generator.py','fusion.py','heading_experiment.py',
    'parallel_decode.py','test_parallel_decode.py','shared_evidence_experiment.py','scheduler_identity_experiment.py',
    'evidence_plan.py','test_semantic_copy_feature_fit.py')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'))
def hashed(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def dump(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def records(path): return [json.loads(v) for v in Path(path).read_text().splitlines() if v]


def model_input_receipt():
    prior=Path('data/ranker/pretrained/catalog-user-fusion-v1-attempt-1/model-verification.json')
    receipt=json.loads(prior.read_text())
    if receipt['modelRevision']!=MODEL.name or receipt['shards']!={k:v[1] for k,v in MODEL_SHARDS.items()}:
        raise ValueError('Pinned model verification receipt differs from declared revision/shards')
    return {'modelPath':str(MODEL.resolve()),'revision':MODEL.name,
        'shards':{k:{'bytes':v[0],'sha256':v[1]} for k,v in MODEL_SHARDS.items()},
        'smallFileSha256':{n:digest(MODEL/n) for n in ('config.json','model.safetensors.index.json',
            'tokenizer.json','tokenizer_config.json','vocab.json','chat_template.jinja')},
        'priorWeightVerificationSha256':digest(prior),
        'freshShardVerificationRequiredBeforeNativeLoad':True}


def verify_model_files(receipt):
    """Fresh CPU byte-stream hashes before load; never expand checkpoint tensors."""
    if receipt!=model_input_receipt(): raise ValueError('Pinned model metadata changed after preflight')
    hashes={}
    for name,expected in receipt['shards'].items():
        path=MODEL/name;before=path.stat()
        if before.st_size!=expected['bytes']: raise ValueError('Pinned shard size mismatch: '+name)
        hasher=hashlib.sha256()
        with path.open('rb') as handle:
            while block:=handle.read(8*2**20): hasher.update(block)
        after=path.stat()
        if (before.st_ino,before.st_mtime_ns,before.st_size)!=(after.st_ino,after.st_mtime_ns,after.st_size) or hasher.hexdigest()!=expected['sha256']:
            raise ValueError('Pinned shard changed or checksum mismatch: '+name)
        hashes[name]=hasher.hexdigest()
    return {'freshShardsSha256':hashes,'smallFileSha256':receipt['smallFileSha256'],
        'modelRevision':MODEL.name,'verified':True,'method':'8 MiB CPU byte stream; no checkpoint tensor execution'}


def dataset_packet(data=DATA):
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest.get('datasetVersion')!='semantic-copy-pairs-v1' or manifest['audit']['status']!='passed':
        raise ValueError('The audited frozen paired dataset is required')
    for name,sha in manifest['fileSha256'].items():
        if digest(data/name)!=sha: raise ValueError('Frozen dataset payload changed: '+name)
    if digest(data/'generator.py')!=manifest['generatorSha256']: raise ValueError('Dataset generator changed')
    for name,sha in manifest['validatorSha256AtAudit'].items():
        if digest(ROOT/name)!=sha: raise ValueError('Validator changed since paired-data audit: '+name)
    packet={s:{'cases':records(data/f'cases.{s}.jsonl'),'fields':records(data/f'fields.{s}.jsonl')}
            for s in ('train','dev','test')}
    seen=set()
    for split,n in (('train',36),('dev',3),('test',3)):
        cases,fields=packet[split]['cases'],packet[split]['fields']
        if len(cases)!=n or len(fields)!=14*n: raise ValueError('Fixed split denominator changed')
        ids={c['caseId'] for c in cases}
        if len(ids)!=n or seen & ids or {f['caseId'] for f in fields}!=ids: raise ValueError('Case membership changed')
        seen |= ids
        for case in cases: serving_rows(case,[f for f in fields if f['caseId']==case['caseId']])
    return manifest,packet


def serving_rows(case,fields):
    """Render through the existing serving layout and compare frozen teacher inputs."""
    from .semantic_copy_training import layout
    common,rows,evidence=layout(case['pendingPagePlan'],case['request'],case['catalog'])
    if len(rows)!=14 or sum(r['maxTokens'] for r in rows)!=224 or len(fields)!=14:
        raise ValueError('Original fourteen-field roster or cap changed')
    by_id={f['rowId']:f for f in fields}
    if len(by_id)!=14 or set(by_id)!={r['rowId'] for r in rows}: raise ValueError('Frozen field membership changed')
    ordered=[]
    for row in rows:
        field=by_id[row['rowId']]
        if (field['messages'],field['maxTokens'],field['kind'],field['field']) != (
                [common]+row['messages'],row['maxTokens'],row['kind'],row['fields'][0]):
            raise ValueError('Frozen teacher prompt differs from the serving renderer')
        ordered.append(field)
    if evidence!=case['publicEvidence']: raise ValueError('Public selected evidence changed')
    return common,rows,ordered


def feature_bytes(suffix_tokens):
    if isinstance(suffix_tokens,bool) or not isinstance(suffix_tokens,int) or not 1<=suffix_tokens<=256:
        raise ValueError('Complete native suffix must have 1-256 tokens')
    return suffix_tokens*(12288+4096+4096)*4


def shuffled_order(count=504,seed=CONFIG['seed']):
    if count!=504: raise ValueError('Exactly 504 individual updates are required')
    indices=list(range(count));random.Random(seed).shuffle(indices)
    return indices


def compatibility_receipts():
    """Read actual prior native evidence; this function executes no model work."""
    from importlib.metadata import version
    from .parallel_benchmark import validate_gate
    native=Path('data/ranker/pretrained/parallel-decode-v3/native-compatibility.json')
    runtime={n:version(n) for n in ('mlx','mlx-lm')}
    validate_gate(json.loads(native.read_text()),runtime,
        {n:digest(ROOT/n) for n in ('parallel_decode.py','test_parallel_decode.py')})
    feature=Path('data/ranker/pretrained/gradient-feature-probe-v1/results.json')
    prior=json.loads(feature.read_text());probe=prior['probe']
    if (prior['status']!='exact-gate-passed' or prior['nativeUpdates']!=0 or
            prior['probeSourceSha256']!=digest(ROOT/'semantic_copy_gradient_features.py') or
            probe['status']!='exact-gate-passed' or not probe['nonzeroB'] or
            probe['runtime']!={**runtime,'numpy':version('numpy')} or
            not probe['sourceStatePreserved'] or not probe['adapterPreserved'] or
            set(probe['comparisons'])!={'nativeVsAuxiliaryNative','auxiliaryNativeVsFrozenFeatures','nativeVsFrozenFeatures'} or
            not all(v['exact'] for v in probe['comparisons'].values())):
        raise ValueError('Prior exact gradient-feature receipt does not match current helper/runtime')
    return {'nativeCompatibilitySha256':digest(native),'gradientFeatureProbeSha256':digest(feature),
        'runtime':runtime,'priorPassDoesNotReplace504PerRowGates':True}


def preflight(tokenizer,data=DATA):
    from .fusion import build_shared_prefix
    from .semantic_copy_gradient_features import build_native_contract
    manifest,packet=dataset_packet(data)
    result={'status':'tokenizer-preflight-passed','nativeExecution':False,'modelCalls':0,
        'trainingUpdates':0,'config':CONFIG,'datasetManifestSha256':digest(data/'manifest.json'),
        'compatibilityReceipts':compatibility_receipts(),
        'modelFiles':model_input_receipt(),
        'payloadSha256':manifest['fileSha256'],'sources':{name:digest(ROOT/name) for name in SOURCES},
        'tokenizerSha256':{p.name:digest(p) for p in MODEL.iterdir() if p.is_file() and
            p.name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','special_tokens_map.json')},
        'splits':{},'limitations':['Synthetic grammar and reused training catalogs; fresh ranking/image quality unlabeled.',
            'Locked fixture selection; copy comparison only, no learned ranking gain.',
            'Original 14 fields omit visible C copy; no complete above-fold claim.',
            'Every native row must still pass its own exact nonzero-B gate before any optimizer update.']}
    for split,group in packet.items():
        rows=[];cases=[]
        for case in group['cases']:
            common,serving,fields=serving_rows(case,[f for f in group['fields'] if f['caseId']==case['caseId']])
            rank,_,_,messages=build_shared_prefix(tokenizer,case['request'],case['catalog'],1792)
            complete=tokenizer.apply_chat_template(messages+[common],tokenize=True,return_dict=False,
                add_generation_prompt=False,enable_thinking=False)
            if complete[:len(rank)]!=rank or not 1<=len(complete)-len(rank)<=2048 or len(complete)>4096:
                raise ValueError('Common evidence violates exact rank boundary or context cap')
            state={'messages':messages+[common],'tokens':complete,'consumed':False}
            case_rows=[]
            for field,row in zip(fields,serving,strict=True):
                contract=build_native_contract(tokenizer,state,field)
                prompt=tokenizer.apply_chat_template(state['messages']+row['messages'],tokenize=True,
                    return_dict=False,add_generation_prompt=True,enable_thinking=False)
                if prompt!=contract['promptTokenIds'] or len(prompt)+row['maxTokens']>4096:
                    raise ValueError('Serving and teacher prompts differ or generation cap exceeds context')
                size=feature_bytes(len(contract['suffixTokenIds']))
                if size>CONFIG['singleHostFeatureBudgetBytes']: raise MemoryError('Single host feature cap exceeded')
                receipt={'caseId':case['caseId'],'rowId':field['rowId'],'kind':field['kind'],
                    'maxTokens':field['maxTokens'],'suffixTokens':len(contract['suffixTokenIds']),
                    'promptTokens':len(prompt),'fullTeacherTokens':contract['fullTokens'],
                    'targetTokens':len(contract['targetTokenIds']),'projectedHostFeatureBytes':size,
                    'promptSha256':contract['promptSha256'],'modelInputSha256':contract['modelInputSha256'],
                    'sourceTokenSha256':contract['sourceTokenSha256'],'predictionStartInBranch':contract['predictionStartInBranch']}
                rows.append(receipt);case_rows.append(receipt)
            cases.append({'caseId':case['caseId'],'lockedPlanSha256':hashed(case['pendingPagePlan']),
                'rankTokens':len(rank),'commonTokens':len(complete)-len(rank),'sharedTokens':len(complete),
                'rankTokensSha256':hashed(rank),'commonCompleteSha256':hashed(complete),
                'maximumSuffix':max(r['suffixTokens'] for r in case_rows)})
        result['splits'][split]={'cases':cases,'rows':rows,'caseCount':len(cases),'fieldCount':len(rows),
            'maximumRankTokens':max(c['rankTokens'] for c in cases),
            'maximumFullTeacherTokens':max(r['fullTeacherTokens'] for r in rows),
            'maximumSuffixTokens':max(r['suffixTokens'] for r in rows),
            'projectedHostFeatureBytes':sum(r['projectedHostFeatureBytes'] for r in rows)}
    projected=result['splits']['train']['projectedHostFeatureBytes']+CONFIG['featureArchiveReserveBytes']
    if projected>CONFIG['totalFeatureArtifactBudgetBytes']: raise MemoryError('Projected feature artifact cap exceeded')
    result['projectedTrainingArtifactsWithReserveBytes']=projected
    result['globalPermutationSha256']=hashed(shuffled_order())
    result['gateWork']={'rows':504,'autodiffPasses':1512,'fullNativeSuffixAutodiffPasses':1008,
        'featureAutodiffPasses':504,'completeSourceStateHashesAtLeast':1008,
        'costDisclosure':'Every row is gated without sampling; this may dominate extraction wall time. No speedup claim.'}
    return result


def load_feature(entry,out):
    """One host row at a time, archive and original per-tensor receipts verified."""
    import numpy as np
    from .semantic_copy_gradient_features import FEATURE_KEYS,validate_feature_contract
    archive=out/entry['archive'];metadata=out/entry['metadata']
    if digest(archive)!=entry['archiveSha256'] or digest(metadata)!=entry['metadataSha256']:
        raise ValueError('Frozen feature artifact changed')
    item=json.loads(metadata.read_text())
    with np.load(archive,allow_pickle=False) as data:
        if set(data.files)!=set(FEATURE_KEYS): raise ValueError('Wrong feature archive membership')
        for k in FEATURE_KEYS: item[k]=np.array(data[k])
    if item.get('gradientFeatureGatePassed') is not True: raise ValueError('Missing exact per-row gate')
    total=0
    for k in FEATURE_KEYS:
        value=item[k];total+=value.nbytes
        if value.dtype!=np.float32 or not np.isfinite(value).all(): raise ValueError('Nonfinite or inexact host feature encoding')
        receipt=entry['featureTensors'][k]
        actual=hashlib.sha256(item['dtypes'][k].encode()+str(tuple(value.shape)).encode()+value.tobytes()).hexdigest()
        if actual!=receipt['sha256'] or list(value.shape)!=receipt['shape'] or item['dtypes'][k]!=receipt['dtype']:
            raise ValueError('Original native feature receipt mismatch')
    if total>CONFIG['singleHostFeatureBudgetBytes']: raise MemoryError('Single host feature cap exceeded')
    validate_feature_contract({k:item[k] for k in FEATURE_KEYS},item['dtypes'],item['predictionStartInBranch'],item['targets'])
    return item


def summary(arms):
    result={}
    for split in ('dev','test'):
        for arm in ('frozen','learned'):
            values=[v for v in arms if v['split']==split and v['arm']==arm]
            done=[v for v in values if v['status']=='completed-attempt']
            result[split+'/'+arm]={'fixedCases':3,'completed':len(done),
                'validOriginal14FieldPages':sum(bool(v['quality']['allFieldsFulfilled']) for v in done),
                'validRateFixedCases':sum(bool(v['quality']['allFieldsFulfilled']) for v in done)/3,
                'acceptedTitles':sum(v['quality'].get('newTitleCount',0) for v in done),
                'acceptedHeadings':sum(v['quality'].get('generatedHeadingCount',0) for v in done),
                'acceptedNonNoneBadges':sum(v['quality'].get('generatedBadgeCount',0) for v in done),
                'failed':sum(v['status']=='failed' for v in values),
                'unattempted':sum(v['status']=='not-attempted' for v in values),
                'inProgress':sum(v['status']=='running' for v in values),
                'lockedPlanCopyMs':[v['lockedPlanCopyMs'] for v in done],
                'validLockedPlanCopyMs':[v['lockedPlanCopyMs'] for v in done if v['quality']['allFieldsFulfilled']],
                'completeAboveFoldClaim':False}
    return result


def run(out,preflight_path,data=DATA):
    """Parent-owned native invocation only; never called by CPU tests/preflight."""
    import numpy as np
    import psutil
    from transformers import AutoTokenizer
    # Complete CPU verification BEFORE importing/loading the native model.
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    expected=preflight(tokenizer,data)
    if json.loads(preflight_path.read_text())!=expected: raise ValueError('Preflight receipt differs from current frozen inputs/source')
    if out.exists(): raise ValueError('Preserve existing experiment directory')
    if psutil.virtual_memory().available<7*2**30: raise MemoryError('Less than declared 7 GiB available')
    model_verification=verify_model_files(expected['modelFiles'])
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.lora import LoRALinear
    from .backbone import FrozenBackbone
    from .evidence_plan import REVISION
    from .fusion import encode_shared_prefix
    from .parallel_decode import decode_rows
    from .shared_evidence_experiment import append_evidence,message_suffixes
    from .semantic_copy_training import disabled,parameter_hashes,cache_proof
    from .semantic_copy_gradient_features import probe_gradient_features,GradientFeatureProbeError,build_native_contract,feature_loss
    from .native_batch_diagnosis import state_hash
    from .scheduler_identity_experiment import assemble_values,mechanisms
    manifest,packet=dataset_packet(data)
    out.mkdir(parents=True);(out/'features').mkdir();(out/'gates').mkdir();started=time.perf_counter()
    arms=[{'split':s,'caseId':c['caseId'],'arm':a,'status':'not-attempted'} for s in ('dev','test')
        for i,c in enumerate(packet[s]['cases']) for a in (('frozen','learned') if i%2==0 else ('learned','frozen'))]
    report={'status':'started','executionCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'config':CONFIG,'preflightSha256':digest(preflight_path),'preflight':expected,
        'modelVerification':model_verification,
        'extraction':[],'steps':[],'arms':arms,'nativeUpdates':0,'optimizerUpdateCalls':0,
        'limitations':expected['limitations'],'phase':'extraction','featureArtifactBytes':0,'controls':{}}
    def save():
        report['wallMs']=(time.perf_counter()-started)*1000;report['summary']=summary(arms);dump(out/'results.json',report)
    save()
    class PeakBackbone(FrozenBackbone):
        def check_memory(self):
            super().check_memory()
            if self.mx.get_peak_memory()>self.memory_budget: raise MemoryError('Observed peak exceeded declared 7 GiB')
    try:
        mx.random.seed(CONFIG['seed']);bb=PeakBackbone(MODEL,REVISION,1792,7);model=bb.model
        model.freeze();model.eval()
        if dict(tree_flatten(model.trainable_parameters())): raise AssertionError('Base model was not fully frozen')
        base_hashes=parameter_hashes(model.language_model);bb.check_memory()
        adapter=LoRALinear.from_base(model.layers[-1].mlp.down_proj,r=4,scale=20,dropout=0)
        model.layers[-1].mlp.down_proj=adapter;model.eval();mx.eval(adapter.parameters())
        roster=dict(tree_flatten(model.trainable_parameters()))
        if len(roster)!=2 or sum(v.size for v in roster.values())!=65536 or any(
                '.31.mlp.down_proj.' not in k or k.rsplit('.',1)[-1] not in ('lora_a','lora_b') for k in roster):
            raise AssertionError('Exactly layer31 down-projection A/B, 65536 trainables, required')
        report['trainableParameters']={k:int(v.size) for k,v in roster.items()}
        initial_a=adapter.lora_a;zero_b=adapter.lora_b
        adapter.lora_b=mx.full(zero_b.shape,.0001,dtype=zero_b.dtype);mx.eval(adapter.parameters())
        first_proof=None;index=0
        train_receipts={(v['caseId'],v['rowId']):v for v in expected['splits']['train']['rows']}
        for case in packet['train']['cases']:
            fields=[f for f in packet['train']['fields'] if f['caseId']==case['caseId']]
            common,_,fields=serving_rows(case,fields)
            with disabled(model,adapter):
                embeddings,_,_,rank=encode_shared_prefix(bb,case['request'],case['catalog'],1792);del embeddings
                state,preparation=append_evidence(bb,rank,common)
            rank_hash=state_hash(rank['cache']);shared_hash=state_hash(state['cache'])
            for field in fields:
                begin=time.perf_counter();planned=train_receipts[(case['caseId'],field['rowId'])]
                if first_proof is None:
                    c=build_native_contract(bb.tokenizer,state,field)
                    first_proof={'prompt':c['promptTokenIds'],'forcedContinuation':c['targetTokenIds'][:3]}
                    report['controls']['nonzeroCacheProof']=cache_proof(bb,adapter,first_proof)
                try:gate,item=probe_gradient_features(bb,adapter,state,field)
                except GradientFeatureProbeError as exc:
                    gate_path=out/'gates'/f'{index:04d}.json';dump(gate_path,exc.report)
                    report['failedGate']={'index':index,'caseId':case['caseId'],'rowId':field['rowId'],
                        'receipt':str(gate_path),'status':'blocked'}
                    raise
                gate_path=out/'gates'/f'{index:04d}.json';dump(gate_path,gate)
                if gate['status']!='exact-gate-passed' or item is None:
                    report['failedGate']={'index':index,'caseId':case['caseId'],'rowId':field['rowId'],'receipt':str(gate_path)}
                    raise AssertionError('Exact per-row nonzero-B feature gate failed; no fit permitted')
                if gate['contract']['modelInputSha256']!=planned['modelInputSha256'] or gate['contract']['sourceTokenSha256']!=planned['sourceTokenSha256']:
                    raise AssertionError('Native row differs from tokenizer preflight')
                actual_bytes=sum(item[k].nbytes for k in ('x','h','y'))
                if actual_bytes!=planned['projectedHostFeatureBytes'] or actual_bytes>CONFIG['singleHostFeatureBudgetBytes']:
                    raise MemoryError('Native host feature size violates preflight')
                archive=out/'features'/f'{index:04d}.npz';metadata=out/'features'/f'{index:04d}.json'
                np.savez(archive,**{k:item[k] for k in ('x','h','y')})
                dump(metadata,{k:v for k,v in item.items() if k not in ('x','h','y')})
                report['featureArtifactBytes']+=archive.stat().st_size+metadata.stat().st_size
                if report['featureArtifactBytes']>CONFIG['totalFeatureArtifactBudgetBytes']: raise MemoryError('Total feature artifact budget exceeded')
                if state_hash(rank['cache'])!=rank_hash or state_hash(state['cache'])!=shared_hash:
                    raise AssertionError('Pristine rank or shared state changed')
                report['extraction'].append({'index':index,'caseId':case['caseId'],'rowId':field['rowId'],'kind':field['kind'],
                    'archive':str(archive.relative_to(out)),'metadata':str(metadata.relative_to(out)),
                    'archiveSha256':digest(archive),'metadataSha256':digest(metadata),'gateSha256':digest(gate_path),
                    'featureTensors':gate['featureTensors'],'hostFeatureBytes':actual_bytes,
                    'modelInputSha256':planned['modelInputSha256'],'sourceStateHash':shared_hash,
                    'gateExact':True,'ms':(time.perf_counter()-begin)*1000})
                index+=1;del item,gate;gc.collect();mx.clear_cache();save()
            del state,rank;gc.collect();mx.clear_cache()
        if len(report['extraction'])!=504: raise AssertionError('All 504 row gates must pass before fitting')
        # Start actual fitting from the original seeded A and exactly zero B.
        adapter.lora_a=initial_a;adapter.lora_b=zero_b;mx.eval(adapter.parameters());mx.synchronize()
        if not bool(mx.all(adapter.lora_b==0).item()): raise AssertionError('Actual fit must start with zero B')
        initial={k:np.array(v) for k,v in tree_flatten(adapter.trainable_parameters())}
        report['controls']['all504NonzeroBGatesPassed']=True;report['controls']['fitStartedFromZeroB']=True
        report['phase']='fit';order=shuffled_order();report['permutation']=order;save()
        optimizer=optim.Adam(learning_rate=CONFIG['learningRate'])
        grad_fn=nn.value_and_grad(adapter,lambda item:feature_loss(bb,adapter,item))
        for step,idx in enumerate(order):
            begin=time.perf_counter();entry=report['extraction'][idx];item=load_feature(entry,out)
            loss,gradients=grad_fn(item);mx.eval(loss,gradients)
            flat=dict(tree_flatten(gradients))
            if set(flat)!={'lora_a','lora_b'} or not bool(mx.isfinite(loss).item()) or not all(bool(mx.all(mx.isfinite(v)).item()) for v in flat.values()):
                raise FloatingPointError('Nonfinite loss or wrong/nonfinite gradient roster')
            norm=sum(float(mx.sum(v.astype(mx.float32)**2).item()) for v in flat.values())**.5
            step_receipt={'step':step,'featureIndex':idx,'caseId':entry['caseId'],'rowId':entry['rowId'],
                'kind':entry['kind'],'loss':float(loss.item()),'gradientL2':norm,
                'status':'update-attempted','updateEvaluated':False}
            report['steps'].append(step_receipt)
            optimizer.update(adapter,gradients);report['optimizerUpdateCalls']+=1
            mx.eval(adapter.parameters(),optimizer.state);mx.synchronize()
            report['nativeUpdates']+=1;step_receipt['updateEvaluated']=True;bb.check_memory()
            if not all(bool(mx.all(mx.isfinite(v)).item()) for _,v in tree_flatten(adapter.trainable_parameters())) or not all(
                    bool(mx.all(mx.isfinite(v)).item()) for _,v in tree_flatten(optimizer.state) if hasattr(v,'dtype')):
                raise FloatingPointError('Nonfinite adapter or optimizer state')
            step_receipt.update(status='completed',ms=(time.perf_counter()-begin)*1000)
            del item,loss,gradients,flat;save()
        if report['nativeUpdates']!=504: raise AssertionError('Fixed update count incomplete')
        changes={k:float(np.abs(np.array(v)-initial[k]).sum()) for k,v in tree_flatten(adapter.trainable_parameters())}
        if not changes['lora_b']>0: raise AssertionError('No real B update')
        report['parameterAbsoluteChanges']=changes;report['controls']['learnedCacheProof']=cache_proof(bb,adapter,first_proof)
        item=load_feature(report['extraction'][0],out);before=float(feature_loss(bb,adapter,item).item())
        checkpoint=out/'adapter.safetensors';mx.save_safetensors(str(checkpoint),dict(tree_flatten(adapter.trainable_parameters())))
        weights=mx.load(str(checkpoint))
        if set(weights)!={'lora_a','lora_b'}: raise AssertionError('Wrong adapter checkpoint membership')
        adapter.load_weights(list(weights.items()),strict=False);mx.eval(adapter.parameters())
        after=float(feature_loss(bb,adapter,item).item())
        if before!=after: raise AssertionError('Feature prediction changed on adapter reload')
        report['checkpoint']={'sha256':digest(checkpoint),'lossBeforeReload':before,'lossAfterReload':after,'exact':True};del item
        after_hashes={k.replace('.31.mlp.down_proj.linear.','.31.mlp.down_proj.'):v for k,v in parameter_hashes(model.language_model).items()}
        if base_hashes!=after_hashes: raise AssertionError('Frozen language weights changed')
        report['controls']['frozenLanguageHashesUnchanged']=True;report['languageParameterTensorsHashed']=len(base_hashes)
        report['phase']='evaluation';save();by_case={c['caseId']:c for s in ('dev','test') for c in packet[s]['cases']}
        expected_states={}
        for arm in arms:
            begin=time.perf_counter();case=by_case[arm['caseId']]
            arm['status']='running';save()
            try:
                fields=[f for f in packet[arm['split']]['fields'] if f['caseId']==case['caseId']]
                common,rows,_=serving_rows(case,fields)
                with disabled(model,adapter):
                    embeddings,_,rank_timing,rank=encode_shared_prefix(bb,case['request'],case['catalog'],1792);del embeddings
                    state,prep=append_evidence(bb,rank,common)
                rank_hash=state_hash(rank['cache']);source_hash=state_hash(state['cache'])
                pair=(rank_hash,source_hash,hashed(case['pendingPagePlan']))
                if case['caseId'] in expected_states and expected_states[case['caseId']]!=pair:
                    raise AssertionError('Paired frozen/learned decisions or pristine states differ')
                expected_states[case['caseId']]=pair
                from contextlib import nullcontext
                with disabled(model,adapter) if arm['arm']=='frozen' else nullcontext():
                    with message_suffixes():outputs,timing=decode_rows(bb,state,rows,3,prefill_step_size=256)
                native=__import__('mlx_lm.generate',fromlist=['generation_stream']);mx.synchronize(native.generation_stream);bb.check_memory()
                if state_hash(rank['cache'])!=rank_hash or state_hash(state['cache'])!=source_hash: raise AssertionError('Evaluation source state changed')
                result=assemble_values(outputs,copy.deepcopy(case['pendingPagePlan']),case['request'],case['catalog'])
                arm.update(status='completed-attempt',rows=outputs,**result,
                    lockedPlanSha256=pair[2],lockedPlanCopyMs=(time.perf_counter()-begin)*1000,
                    rankContextTiming=rank_timing,preparation=prep,copyTiming=timing,
                    original14FieldCap=224,actualOutputTokens=sum(len(r['tokenIds']) for r in outputs),
                    actualArSteps=sum(r['arSteps'] for r in outputs),rankStateHash=rank_hash,sharedStateHash=source_hash,
                    mechanisms=mechanisms(result['wireRows'],case['pendingPagePlan'],case['catalog'],True),memory=bb.runtime_memory())
            except Exception as exc:
                arm.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)},lockedPlanCopyMs=(time.perf_counter()-begin)*1000)
                save();raise
            finally:
                if 'state' in locals(): del state
                if 'rank' in locals(): del rank
                gc.collect();mx.clear_cache()
            save()
        report['status']='completed';report['phase']='completed';report['memory']=bb.runtime_memory();save()
    except Exception as exc:
        report['status']='blocked';report['blocker']={'type':type(exc).__name__,'message':str(exc)}
        if 'bb' in locals(): report['memory']=bb.runtime_memory()
        save();raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--preflight',action='store_true')
    p.add_argument('--preflight-report',type=Path,required=True);p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.preflight:
        if args.out is not None or args.preflight_report.exists(): raise ValueError('Preflight writes one new report only')
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
        receipt=preflight(tokenizer)
        import sys
        if any(k=='mlx' or k.startswith('mlx.') for k in sys.modules): raise AssertionError('Tokenizer-only preflight unexpectedly imported MLX')
        dump(args.preflight_report,receipt)
        print(json.dumps({s:{k:v for k,v in result.items() if k not in ('rows','cases')} for s,result in receipt['splits'].items()}),flush=True)
    else:
        if args.out is None: raise ValueError('Native invocation needs a new output directory')
        run(args.out,args.preflight_report)


if __name__=='__main__': main()
