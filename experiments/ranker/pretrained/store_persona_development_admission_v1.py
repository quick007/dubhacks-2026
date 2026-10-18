"""Explicit technically verified development-demo admission; NEVER promotion.

The selected adapter FAILED the quality gate. This separate receipt admits only
known-public-profile UI/factual QA. Existing accepted/offline-pass selectors are
unchanged. No native, HTTP, browser, service, optimizer or profile write occurs.
"""
from copy import deepcopy
import json
from pathlib import Path

from . import store_persona_http_semantic_probe_v2 as probe
from . import store_persona_adapter_runtime_v2 as runtime

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
MAIN=Path('/Users/lukas/Documents/Projects/dubhacks-2026')
VERSION='persona-adapter-development-admission.v1'
FIXED={'adapter':'1e88bdc919e8d8dede8cc54f8f39aff6c8af297c7e496504d7ab6cada63c148b',
 'gate':'c524f37b2cbd9929bb34f45cb169523b6d143aa2062e40b3035caba27d9df92a',
 'v5':'5d07673e51450bbbbb1803c8a44ef99721b39fd3b1ae92edb9168e27a1377d6b',
 'offlineAudit':'107fe4e5e330d4176ba8f0bca7ee6b4329f2d705e06b2ca84ebf18f9c410254c',
 'offlineSource':'950cebc62d752b33dbfe7a09031958671afcba3e130bc2abc1428517c321907d',
 'queryMechanicalAudit':'b206798dc1cc88a97efad3cd7c56f8f9d839cbb7db27cc4719dbf97c9ed049da',
 'queryCombinationReplay':'0994eefbd86cad9f89e765a76053a8e1b5e6517386d602d849e3132ab1a51809',
 'integrationManifest':'16acf1713efcd3164be473b6f44c480cac4aa3cdfee760add21877c958aa8beb',
 'copyManifest':'4bf86af880a59ffb420646bbaf24cd200b743dccb5d0cf9274190d3d0a4acfae',
 'rankExport':'17a0a15a2da6e2fc90e651e1f5bb2c237d61241795ec0cca548a4f3ce319d703',
 'qualityGate':'73fa0d52c5ef4fe983834f610ef00cf89a899cf420cdb81008dd5a7909298c6e',
 'publicPacket':'2a0801c73440dfbcea50f73c0c1b2de3a96abc8c975caf8d54e92010a84fe312'}
SOURCES=tuple(dict.fromkeys((*probe.SOURCE_NAMES,'store_persona_development_admission_v1.py','audit_persona_choice_combined_v2.py')))


def need(value,message):
    if not value:raise ValueError(message)


def evidence(record,base,*,require_startup=True):
    need(record.get('version')==VERSION and record.get('status')=='technical-development-demo' and
        record.get('scope')=='known-public-profile-development-demo' and record.get('qualityGatePassed') is False and
        record.get('modelPromotion') is False and record.get('acceptedProfile') is False,
        'Explicit non-promoted failed-quality development demo required; accepted/offline-pass substitution forbidden')
    keys=['offlineAudit','admissionBeforeGold','queryMechanicalAudit','queryCombinationReplay','adapterManifest',
        'gateCheckpoint','v5Export','copyManifest','integrationManifest']
    if require_startup:keys.append('startup')
    paths={key:probe.artifact(record[key],base) for key in keys}
    pins={str(path):record[key]['sha256'] for key,path in paths.items()}
    for key in ('offlineAudit','queryMechanicalAudit','queryCombinationReplay','integrationManifest','copyManifest'):
        need(record[key]['sha256']==FIXED[key],'Exact reviewed development artifact required: '+key)
    need(record['v5Export']['sha256']==FIXED['rankExport'] and record['gateCheckpoint']['sha256']==FIXED['gate'],
        'Exact existing v5/query checkpoints required')
    for name in SOURCES:
        path=str((HERE/name).resolve())
        need(record.get('sourceSha256',{}).get(path)==probe.old.sha(path),'Current explicit development source pin missing: '+name)
    pins.update(record['sourceSha256'])
    audit=json.loads(paths['offlineAudit'].read_text());before=json.loads(paths['admissionBeforeGold'].read_text())
    need(audit.get('version')=='persona-choice-combined-independent-audit.v2' and audit.get('status')=='completed-offline-adjudication' and
        audit.get('sourceSha256')==FIXED['offlineSource'] and audit.get('gateSha256')==FIXED['qualityGate'] and
        audit.get('offlineCoreAndReplacementPass') is False and audit.get('overallPromotionPass') is False,
        'Actual corrected FAILED quality audit must remain explicit; no quality-pass assertion allowed')
    observed={'rawPrimaryTop1':audit['raw']['main']['learned']['primaryTop1'],
        'rawFrozenBasePrimaryTop1':audit['raw']['main']['frozen-base-lexical-decision']['primaryTop1'],
        'combinedPrimaryTop1':audit['combined']['main']['learned']['primaryTop1'],
        'combinedFrozenBasePrimaryTop1':audit['combined']['main']['frozen-base-lexical-decision']['primaryTop1'],'denominator':32}
    need(observed=={'rawPrimaryTop1':17,'rawFrozenBasePrimaryTop1':20,'combinedPrimaryTop1':17,'combinedFrozenBasePrimaryTop1':19,'denominator':32},
        'Exact observed development failure/better frozen-base comparison required')
    need(before.get('privateQualityJSONParsed') is False and before.get('adapterCheckpointSha256')==FIXED['adapter'] and
        before.get('packetManifestSha256')==FIXED['publicPacket'],'Original mechanical admission identity differs')
    serving,certificate=runtime.load_manifest(paths['adapterManifest'],record['adapterManifest']['sha256'])
    need(serving['adapter']['sha256']==FIXED['adapter'],'Only explicit technically verified existing adapter admitted; no silent replacement')
    mechanical=json.loads(paths['queryMechanicalAudit'].read_text());replay=json.loads(paths['queryCombinationReplay'].read_text())
    need(mechanical['version']=='query-intent-independent-TRAIN-audit.v3' and mechanical['status']=='passed-independent-TRAIN-replay' and
        mechanical['headSha256']==FIXED['gate'] and mechanical['sourceManifestSha256']==runtime.INTENT_TRAIN_MANIFEST_SHA and
        mechanical['replay']['observedReplayOptimizerSteps']==250 and mechanical['replay']['everyFinalParameterBitwiseMatches'] is True and
        mechanical['twoReloadsExactLogitParity'] is True and mechanical['reloadSavedParity']['exactSavedProbabilityParity'] is True,
        'Actual mechanically verified existing query head required')
    need(replay['version']=='persona-combined-query-replay-audit.v1' and replay['status']=='passed' and
        replay['adapterCheckpointSha256']==FIXED['adapter'] and replay['gateCheckpointSha256']==FIXED['gate'] and
        replay['v5HeadSha256']==FIXED['v5'] and replay['combinedFreezeSha256']==before['combinedFreezeSha256'] and
        replay['mechanicalQueryAuditSha256']==FIXED['queryMechanicalAudit'] and replay['rows']==344 and replay['modelRowsChecked']==1376 and
        replay['uniqueQueryVectorsReplayed']==292 and replay['exactQueryProbabilityAndV5Parity'] is True and
        replay['actualFloat32BlendAndGuardsExact'] is True and replay['overallPromotionPass'] is False,
        'Actual same-model independent mechanical combination replay required')
    pins.update(mechanical['inputSha256']);pins.update(replay['inputSha256']);pins.update(replay['sourceSha256'])
    export=json.loads(paths['v5Export'].read_text());copy=json.loads(paths['copyManifest'].read_text())
    need(export['head']['sha256']==FIXED['v5'] and copy['all48BindingsSupervised'] is True and copy['runtimeTeacherLookup'] is False,
        'Original v5 and public-only copy contracts required')
    expected={'adapterSha256':FIXED['adapter'],'adapterManifestSha256':record['adapterManifest']['sha256'],
        'gateSha256':FIXED['gate'],'gateSourceSha256':probe.old.sha(HERE/'store_query_intent_gate_v3.py'),
        'gateTRAINManifestSha256':runtime.INTENT_TRAIN_MANIFEST_SHA,'v5HeadSha256':FIXED['v5'],
        'copyHeadSha256':copy['headSha256'],'colorHeadSha256':runtime.COLOR_SHA,
        'rankExportSha256':record['v5Export']['sha256'],'integrationManifestSha256':FIXED['integrationManifest'],
        'startup':json.loads(paths['startup'].read_text()) if require_startup else None}
    probe.old.pins_current(pins)
    return {'mode':'technical-development-demo','developmentDemo':True,'qualityGatePassed':False,'modelPromotion':False,
        'acceptedProfile':False,'overallPromotionPass':False,'acceptanceStateWritten':False,
        'scope':'known public profile development demo and technical/factual UI QA; failed quality gate; no learned-adaptation advantage or unseen claim',
        'qualityFailure':observed,'offlineAuditSha256':FIXED['offlineAudit'],'queryMechanicalAuditSha256':FIXED['queryMechanicalAudit'],
        'queryCombinationReplaySha256':FIXED['queryCombinationReplay'],'expected':expected,'pins':pins}


def selection(receipt_path,expected_sha):
    path=Path(receipt_path).resolve()
    need(probe.old.sha(path)==runtime.contract.exact_sha(expected_sha),'Exact root-selected development receipt SHA required')
    result=evidence(json.loads(path.read_text()),path.parent)
    result['pins'][str(path)]=expected_sha
    return result


def build_record(startup_path=None):
    """CPU recipe for root to seal AFTER manual candidate startup, never acceptance."""
    data=ROOT/'data/ranker/pretrained';main_data=MAIN/'data/ranker/pretrained'
    paths={'offlineAudit':data/'persona-choice-independent-adjudication-v2-attempt-1/ADJUDICATION.json',
        'admissionBeforeGold':data/'persona-choice-independent-adjudication-v2-attempt-1/ADMISSION_PASSED_BEFORE_GOLD.json',
        'queryMechanicalAudit':data/'query-intent-gate-v3-audit-attempt-1/audit.json',
        'queryCombinationReplay':data/'persona-choice-combined-query-audit-v1-attempt-1/audit.json',
        'adapterManifest':data/'persona-choice-train-v1-attempt-1/serving.json',
        'gateCheckpoint':data/'query-intent-gate-v3-fit-attempt-1/head.pt',
        'v5Export':main_data/'actual-store-function-v5-attempt-1/serving-head.json',
        'copyManifest':main_data/'store-parallel-text-train-all-demo-v1-attempt-2/serving-manifest.json',
        'integrationManifest':HERE/'store_demo_integration_personalized_adapter_v2/manifest.json'}
    if startup_path is not None:paths['startup']=Path(startup_path).resolve()
    return {'version':VERSION,'status':'technical-development-demo','scope':'known-public-profile-development-demo',
        'qualityGatePassed':False,'modelPromotion':False,'acceptedProfile':False,
        **{key:{'file':str(path),'sha256':probe.old.sha(path)} for key,path in paths.items()},
        'sourceSha256':{str((HERE/name).resolve()):probe.old.sha(HERE/name) for name in SOURCES}}


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(__doc__);p.add_argument('--startup',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args(argv);record=build_record(args.startup);result=evidence(record,ROOT,require_startup=args.startup is not None)
    if args.out is not None:
        need(args.startup is not None,'Actual startup file required before sealing root development receipt')
        need(args.out.is_absolute() and args.out.resolve().is_relative_to(ROOT) and not args.out.exists(),'Unique absolute worktree development output required')
        args.out.mkdir(parents=True);probe.old.write(args.out/'receipt.json',record);probe.old.write(args.out/'preflight.json',result)
    print(json.dumps({'status':'CPU-development-evidence-verified','qualityGatePassed':False,'modelPromotion':False,
        'actualStartupBound':args.startup is not None,'qualityFailure':result['qualityFailure'],
        'receiptSha256':probe.old.sha(args.out/'receipt.json') if args.out else None},indent=2))


if __name__=='__main__':main()
