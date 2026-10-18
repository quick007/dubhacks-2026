"""Read-only CPU audit of feature-fit evidence; no model or MLX imports.

Progress mode checks small receipts only. --verify-features is refused while
the owning native worker is still active according to its persisted status.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import struct

ROOT=Path('experiments/ranker/pretrained')
OUT=Path('data/ranker/pretrained/semantic-copy-feature-fit-v1')
PREFLIGHT=ROOT/'SEMANTIC_COPY_FEATURE_PREFLIGHT_V2.json'
FEATURE_WIDTHS={'x':12288,'h':4096,'y':4096}
COMPARISONS={'nativeVsAuxiliaryNative','auxiliaryNativeVsFrozenFeatures','nativeVsFrozenFeatures'}


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block:=handle.read(2**20):result.update(block)
    return result.hexdigest()


def audit(out=OUT,preflight_path=PREFLIGHT,verify_features=False):
    path=out/'results.json'
    # The worker writes directly. A partial concurrent write is not evidence of
    # a corrupt run; a subsequent scheduled read must resolve this snapshot.
    try: raw=path.read_bytes();run=json.loads(raw)
    except (FileNotFoundError,json.JSONDecodeError) as exc:
        return {'status':'snapshot-unavailable','reason':type(exc).__name__,'finalConclusion':False}
    terminal=run['status'] not in ('started',)
    if verify_features and not terminal:raise ValueError('Do not scan all feature archives while native worker is active')
    checks=[];issues=[]
    def check(name,condition,detail=None):
        row={'name':name,'passed':bool(condition)}
        if detail is not None:row['detail']=detail
        checks.append(row)
        if not condition:issues.append(row)
    pre=run['preflight'];cfg=run['config'];entries=run['extraction'];steps=run['steps']
    check('preflight bytes bound',digest(preflight_path)==run['preflightSha256'])
    check('preflight/config equality',pre['config']==cfg)
    changed={name:hash_value for name,hash_value in pre['sources'].items() if digest(ROOT/name)!=hash_value}
    check('all running source files unchanged',not changed,changed or None)
    dataset=ROOT/'semantic_copy_pairs'
    check('dataset manifest unchanged',digest(dataset/'manifest.json')==pre['datasetManifestSha256'])
    check('all frozen dataset payloads unchanged',all(digest(dataset/n)==s for n,s in pre['payloadSha256'].items()))
    model_receipt=run['modelVerification'];model_pre=pre['modelFiles']
    check('fresh pinned model verification receipt',model_receipt['verified'] is True
        and model_receipt['modelRevision']==model_pre['revision']
        and model_receipt['freshShardsSha256']=={k:v['sha256'] for k,v in model_pre['shards'].items()}
        and model_receipt['smallFileSha256']==model_pre['smallFileSha256'])
    expected=pre['splits']['train']['rows'];lookup={(r['caseId'],r['rowId']):r for r in expected}
    check('fixed training preflight denominator',len(expected)==len(lookup)==504)
    check('extraction prefix indices',all(e['index']==i for i,e in enumerate(entries)) and len(entries)<=504)
    seen=set();states={};control_parameters=None;archive_bytes=0;maximum_peak=0;archives_checked=0
    for entry in entries:
        i=entry['index'];key=(entry['caseId'],entry['rowId']);planned=lookup[key]
        check(f'row {i}: unique extraction membership',key not in seen and key==(expected[i]['caseId'],expected[i]['rowId']))
        seen.add(key)
        gate_path=out/'gates'/f'{i:04d}.json';gate=json.loads(gate_path.read_text())
        check(f'row {i}: gate receipt hash',digest(gate_path)==entry['gateSha256'])
        check(f'row {i}: exact nonzero-B no-update gate',gate['status']=='exact-gate-passed' and gate['nativeUpdates']==0
            and gate['nonzeroB'] is True and entry['gateExact'] is True)
        check(f'row {i}: gate source identity',gate['sourceSha256']==pre['sources']['semantic_copy_gradient_features.py'])
        check(f'row {i}: source cache and adapter preserved',gate['sourceStatePreserved'] is True and gate['adapterPreserved'] is True
            and gate['sourceStateHashBefore']==gate['sourceStateHashAfter']==entry['sourceStateHash']
            and gate['adapterBefore']==gate['adapterAfter'])
        if control_parameters is None:control_parameters=gate['adapterBefore']
        check(f'row {i}: fixed extraction adapter',gate['adapterBefore']==control_parameters)
        if entry['caseId'] in states:check(f'row {i}: case-local pristine shared state',states[entry['caseId']]==entry['sourceStateHash'])
        states[entry['caseId']]=entry['sourceStateHash']
        check(f'row {i}: tokenizer/input identity',all(gate['contract'][k]==planned[k] for k in
            ('modelInputSha256','promptSha256','sourceTokenSha256','predictionStartInBranch'))
            and entry['modelInputSha256']==planned['modelInputSha256'])
        check(f'row {i}: exact comparison flags',set(gate['comparisons'])==COMPARISONS and all(
            c['exact'] is True and all(v['shapeExact'] is True and v['dtypeExact'] is True and v['arrayExact'] is True
                and v['maxAbsDifference']==0 for v in [c['loss'],*c['gradients'].values()]) for c in gate['comparisons'].values()))
        direct=gate['arms']['direct']
        check(f'row {i}: independent raw loss/gradient receipts agree',math.isfinite(direct['loss']) and all(
            gate['arms'][arm]==direct for arm in ('auxiliaryNative','frozenFeatures'))
            and set(direct['gradientTensors'])=={'lora_a','lora_b'})
        check(f'row {i}: gradient shapes',direct['gradientTensors']['lora_a']['shape']==[12288,4]
            and direct['gradientTensors']['lora_b']['shape']==[4,4096])
        features=gate['featureTensors'];length=planned['suffixTokens']
        check(f'row {i}: full-shape/per-tensor receipts',features==entry['featureTensors'] and all(
            features[k]['shape']==[1,length,width] and features[k]['dtype']==gate['featureContract']['dtypes'][k]
            for k,width in FEATURE_WIDTHS.items()))
        metadata_path=out/entry['metadata'];archive_path=out/entry['archive'];metadata=json.loads(metadata_path.read_text())
        archive_bytes+=metadata_path.stat().st_size+archive_path.stat().st_size
        check(f'row {i}: metadata hash',digest(metadata_path)==entry['metadataSha256'])
        check(f'row {i}: metadata bound to native gate',metadata['contract']==gate['contract']
            and metadata['gradientFeatureGatePassed'] is True and metadata['caseId']==entry['caseId']
            and metadata['rowId']==entry['rowId'] and metadata['dtypes']==gate['featureContract']['dtypes']
            and metadata['predictionStartInBranch']==planned['predictionStartInBranch']
            and len(metadata['targets'])==planned['targetTokens']
            and metadata['targets'][-1]==metadata['contract']['stopTokenId'])
        check(f'row {i}: bounded host feature bytes',entry['hostFeatureBytes']==planned['projectedHostFeatureBytes']
            ==length*20480*4 and entry['hostFeatureBytes']<=cfg['singleHostFeatureBudgetBytes'])
        memory=gate['memory'];maximum_peak=max(maximum_peak,memory['mlxPeakBytes'])
        check(f'row {i}: active/peak native guard',memory['processBudgetBytes']==7*2**30
            and max(memory['mlxActiveBytes'],memory['mlxPeakBytes'])<=7*2**30)
        if verify_features:
            import numpy as np
            check(f'row {i}: archive hash',digest(archive_path)==entry['archiveSha256'])
            with np.load(archive_path,allow_pickle=False) as archive:
                check(f'row {i}: archive members',set(archive.files)==set(FEATURE_WIDTHS))
                for k in FEATURE_WIDTHS:
                    value=archive[k];receipt=features[k]
                    actual=hashlib.sha256(receipt['dtype'].encode()+str(tuple(value.shape)).encode()+value.tobytes()).hexdigest()
                    check(f'row {i}: {k} native feature bytes',value.dtype==np.float32 and list(value.shape)==receipt['shape']
                        and np.isfinite(value).all() and actual==receipt['sha256'])
                    del value
            archives_checked+=1
    check('recorded feature byte sum',archive_bytes==run['featureArtifactBytes'])
    check('feature artifact budget',archive_bytes<=cfg['totalFeatureArtifactBudgetBytes'])
    if run['phase']=='extraction':
        check('no updates before all gates complete',not steps and run['nativeUpdates']==run['optimizerUpdateCalls']==0)
    failed_gate=None
    if run.get('failedGate'):
        failure=run['failedGate'];failure_path=Path(failure['receipt'])
        failed_gate=json.loads(failure_path.read_text())
        check('failed gate is separate from accepted extraction prefix',failure['index']==len(entries)
            and failed_gate['status'] in ('strict-gate-failed','blocked') and run['nativeUpdates']==0)
        check('failed gate helper identity',failed_gate['sourceSha256']==pre['sources']['semantic_copy_gradient_features.py'])
    if steps or run['phase'] in ('fit','evaluation','completed'):
        check('all gates precede fitting',len(entries)==504 and run['controls'].get('all504NonzeroBGatesPassed') is True)
        check('zero-B reset before fit',run['controls'].get('fitStartedFromZeroB') is True)
        order=list(range(504));random.Random(cfg['seed']).shuffle(order)
        check('full seeded global permutation',run.get('permutation')==order)
        check('update prefix matches global permutation',all(s['step']==i and s['featureIndex']==order[i] for i,s in enumerate(steps)))
        check('evaluated update accounting',run['nativeUpdates']==sum(s['updateEvaluated'] is True for s in steps))
        check('optimizer call accounting',run['nativeUpdates']<=run['optimizerUpdateCalls']<=len(steps)<=504)
        check('finite training receipts',all(math.isfinite(s['loss']) and math.isfinite(s['gradientL2']) and s['gradientL2']>=0 for s in steps))
        check('step field membership',all((s['caseId'],s['rowId'],s['kind'])==(
            entries[s['featureIndex']]['caseId'],entries[s['featureIndex']]['rowId'],entries[s['featureIndex']]['kind']) for s in steps))
    for name in ('nonzeroCacheProof','learnedCacheProof'):
        if name in run['controls']:
            proof=run['controls'][name]
            check(name,proof['adapterNonzero'] is True and len(set(proof['completeStateHashes']))==1
                and proof['matchedForcedContinuationLogitsExact'] is True and math.isfinite(proof['maxAbsLogitEffect']) and proof['maxAbsLogitEffect']>0)
    if 'checkpoint' in run:
        receipt=run['checkpoint'];checkpoint=out/'adapter.safetensors'
        check('checkpoint bytes and exact same-mode reload',digest(checkpoint)==receipt['sha256'] and receipt['exact'] is True
            and math.isfinite(receipt['lossBeforeReload']) and receipt['lossBeforeReload']==receipt['lossAfterReload'])
        with checkpoint.open('rb') as handle:
            length=struct.unpack('<Q',handle.read(8))[0]
            if length>2**20:raise ValueError('Unexpected adapter header size')
            header=json.loads(handle.read(length))
        check('adapter-only checkpoint roster',set(header)-{'__metadata__'}=={'lora_a','lora_b'}
            and header['lora_a']['shape']==[12288,4] and header['lora_b']['shape']==[4,4096])
        check('frozen language verification receipt',run['controls'].get('frozenLanguageHashesUnchanged') is True
            and run.get('languageParameterTensorsHashed',0)>700)
    completed=[a for a in run['arms'] if a['status']=='completed-attempt']
    check('fixed twelve evaluation slots',len(run['arms'])==12 and len({(a['split'],a['caseId'],a['arm']) for a in run['arms']})==12)
    paired={}
    for arm in completed:
        key=arm['caseId'];value=(arm['rankStateHash'],arm['sharedStateHash'],arm['lockedPlanSha256'])
        if key in paired:check('paired immutable states '+key,paired[key]==value)
        paired[key]=value
        check('original field/cap scope '+key+'/'+arm['arm'],len(arm['rows'])==14 and arm['original14FieldCap']==224
            and sum(r['maxTokens'] for r in arm['rows'])==224)
        check('raw token accounting '+key+'/'+arm['arm'],arm['actualOutputTokens']==sum(len(r['tokenIds']) for r in arm['rows'])
            and arm['actualArSteps']==sum(r['arSteps'] for r in arm['rows']) and all(
                r['arSteps']==len(r['rawSampledTokenIds']) and r['finishReason'] in ('stop','length') for r in arm['rows']))
    if terminal and completed:
        # Replay only saved raw outputs. This is CPU validation, no generation.
        from .scheduler_identity_experiment import assemble_values
        cases={c['caseId']:c for split in ('dev','test') for c in (
            json.loads(line) for line in (dataset/f'cases.{split}.jsonl').read_text().splitlines() if line)}
        for arm in completed:
            case=cases[arm['caseId']]
            replay=assemble_values(copy.deepcopy(arm['rows']),copy.deepcopy(case['pendingPagePlan']),case['request'],case['catalog'])
            check('strict raw quality replay '+arm['caseId']+'/'+arm['arm'],all(replay[k]==arm[k] for k in
                ('quality','rawText','wireRows','valueProtocolRejections','pagePlan','audit')))
    if run['status']=='completed':
        check('completed extraction/update/evaluation coverage',len(entries)==len(steps)==run['nativeUpdates']==run['optimizerUpdateCalls']==504
            and len(completed)==12 and all(s['status']=='completed' for s in steps))
    summaries={}
    for split in ('dev','test'):
        for name in ('frozen','learned'):
            slots=[a for a in run['arms'] if a['split']==split and a['arm']==name]
            done=[a for a in slots if a['status']=='completed-attempt']
            valid=[a for a in done if a['quality']['allFieldsFulfilled']]
            summaries[split+'/'+name]={'fixedCases':3,'completed':len(done),'validOriginal14FieldPages':len(valid),
                'validRateFixedCases':len(valid)/3,'acceptedTitles':sum(a['quality']['newTitleCount'] for a in done),
                'acceptedHeadings':sum(a['quality']['generatedHeadingCount'] for a in done),
                'acceptedNonNoneBadges':sum(a['quality']['generatedBadgeCount'] for a in done),
                'failed':sum(a['status']=='failed' for a in slots),'unattempted':sum(a['status']=='not-attempted' for a in slots),
                'inProgress':sum(a['status']=='running' for a in slots),
                'lockedPlanCopyMs':[a['lockedPlanCopyMs'] for a in done],
                'validLockedPlanCopyMs':[a['lockedPlanCopyMs'] for a in valid],'completeAboveFoldClaim':False}
    check('independently recomputed fixed-denominator summaries',summaries==run['summary'])
    result={'status':('audit-passed' if not issues else 'audit-issues') if terminal else 'in-progress',
        'runStatus':run['status'],'phase':run['phase'],'finalConclusion':terminal,
        'experimentCompleted':run['status']=='completed','qualityComparisonComplete':len(completed)==12,
        'auditSourceSha256':digest(Path(__file__)),
        'snapshotSha256':hashlib.sha256(raw).hexdigest(),'executionCommit':run['executionCommit'],
        'extractedRows':len(entries),'evaluatedUpdates':run['nativeUpdates'],'completedArms':len(completed),
        'verifiedFeatureArchives':archives_checked,'fullFeatureArchiveVerification':verify_features,
        'maximumRecordedGatePeakBytes':maximum_peak,'featureArtifactBytes':archive_bytes,
        'checksRun':len(checks),'issues':issues,'runBlocker':run.get('blocker'),'failedGate':run.get('failedGate'),
        'failedGateReceipt':failed_gate,'runSummary':summaries,
        'scope':'Evidence/state/byte/CPU-validator audit; no model execution or complete above-fold claim.'}
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--preflight',type=Path,default=PREFLIGHT);p.add_argument('--verify-features',action='store_true')
    p.add_argument('--report',type=Path);args=p.parse_args();result=audit(args.out,args.preflight,args.verify_features)
    if args.report:
        if args.report.exists():raise ValueError('Preserve previous audit report; choose a new path')
        args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
