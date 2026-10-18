"""Predeclared fresh frozen-Qwen public-context candidate; CPU prepare by default.

The original40 are development replay after v1 inspection. Independent public
cases may be appended by explicit --independent-public, never private gold.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import nn
from . import store_persona_context_v1 as base

HERE = Path(__file__).resolve().parent
VERSION = 'public-persona-native-context.v1'
CONFIG = {**base.CONFIG, 'textDimensions':4096, 'steps':250, 'ownershipCoefficient':'one shared trainable scalar initialized0, additive public purchase indicator; no mask', 'learningRate':.003,'weightDecay':.0001,
    'seed':20261018,'hiddenWidth':64,'colorWidth':32,'variantLossWeight':.5,'maxTokens':512,'memoryGiB':7,
    'maxNativeSeconds':600, 'contextRepresentation':'fresh frozen Qwen4096 whole public observations; no hashed new text; catalog original Qwen4096',
    'oldHeldoutScope':'40 previously inspected synthetic cases are DEVELOPMENT REPLAY; no selection from their outcomes',
    'independentScope':'separate author32 frozen before model fit; known16 synthetic new combinations; no human or unseen catalog claim'}


def public_context_text(value):
    request, policy, context = base.checked_context_request(value)
    anchor = max((datetime.fromisoformat(e['occurredAt'].replace('Z','+00:00')) for e in context['publicHistory']),default=None)
    # Public real item names/colors survive. Opaque IDs and absolute time are not
    # semantic input; relative age and event type identify observed possessions.
    events=[]
    for e in context['publicHistory']:
        age=(anchor-datetime.fromisoformat(e['occurredAt'].replace('Z','+00:00'))).total_seconds()/86400
        events.append([e['eventType'],e['itemName'],e['category'],e.get('selectedColor'),round(age,3)])
    facts={'query':request['query'],'itemBudget':request['itemBudget'],'hardRequirements':request['hardRequirements'],
        'background':context['persona']['background'],'colorPreferences':context['persona']['colorPreferences'],
        'recentEvents_type_item_category_color_ageDays':events}
    return ('Encode public shopping observations. Purchases are already owned possessions; suggest useful complements rather than duplicates. '
        'Views and carts are interests, not ownership. Respect query, budget and hard requirements.\n'+base.canonical(facts))


def prepare(public_export, feature_root, out, *, demo_export=None, probe_public=None, independent_public=None, contrasts_public=None):
    from tokenizers import Tokenizer
    from .catalog_fusion.inputs import digest, canonical
    root=Path(feature_root);original=json.loads((root/'packet.json').read_text());report=json.loads((root/'report.json').read_text())
    # Full frozen catalog provenance validated on CPU, not fabricated context vectors.
    catalog=base.load_catalog_features(root)
    tokenizer=Tokenizer.from_file(str(Path(original['model']['modelPath'])/'tokenizer.json'))
    base.verify_seed(HERE/'synthetic_history_v1',Path(public_export))
    profiles=base.read_profiles(public_export);split=json.loads((HERE/'synthetic_history_v1/split.json').read_text())
    base.checked_split(profiles,split)
    sources={str(Path(__file__)):base.sha(__file__),str(Path(base.__file__)):base.sha(base.__file__),
        str(HERE/'store_persona_qwen_fit_v1.py'):base.sha(HERE/'store_persona_qwen_fit_v1.py'), str(HERE/'backbone.py'):base.sha(HERE/'backbone.py'),str(HERE/'catalog_fusion/runtime.py'):base.sha(HERE/'catalog_fusion/runtime.py')}
    for name in ('store_demo_engine.py','store_catalog_v1.py','store_function_adaptation.py','catalog_fusion/inputs.py','catalog_fusion/generalization_v2.py','store_copy_fit_v2.py','semantic_copy_training.py','store_rank_session.py'):
        sources[str(HERE/name)]=base.sha(HERE/name)
    files={str(Path(public_export).resolve()):base.sha(public_export)}
    annotation_pins={str((HERE/'synthetic_history_v1/labels.train.json').resolve()):'1b1ae769cb162bcb0deee76ea938ac619fbd7a94ee3ef39d7544d5d2491e8ccb'}
    rows=[]
    train=set(split['trainProfileIds'])
    incoming=[(p,'train' if p['profileId'] in train else 'development-replay') for p in profiles]
    if demo_export:
        files[str(Path(demo_export).resolve())]=base.sha(demo_export)
        people=json.loads(Path(demo_export).read_text())['people']
        incoming += [({'profileId':p['id'],'userGroupId':p['id'],'request':p['request']},'demo-nonfit') for p in people]
        for person in people:
            request=deepcopy(person['request']);request['query']='Browse the store'
            incoming.append(({'profileId':person['id']+'-default-query','userGroupId':person['id'],'request':request},'demo-default-query-nonfit'))
        files[str((HERE.parents[2]/'apps/storefront/src/local-model.ts').resolve())]=base.sha(HERE.parents[2]/'apps/storefront/src/local-model.ts')
    if probe_public:
        files[str(Path(probe_public).resolve())]=base.sha(probe_public)
        incoming += [({**pair[arm],'userGroupId':pair['pairId']},'history-diagnostic-nonfit') for pair in json.loads(Path(probe_public).read_text())['pairs'] for arm in ('profileA','profileB')]
    if contrasts_public:
        manifest_path=Path(contrasts_public).parent/'manifest.json'
        if base.sha(manifest_path)!='7ec75bc25321d3c235159a5b609893bbd428b1512c82ab1e2d07446d8247a017':raise ValueError('Exact reviewed TRAINcontrast manifest required')
        manifest=json.loads(manifest_path.read_text())
        for name,checksum in manifest['files'].items():
            if base.sha(manifest_path.parent/name)!=checksum:raise ValueError('Frozen TRAINcontrast file changed')
        files[str(manifest_path.resolve())]=base.sha(manifest_path)
        annotation_pins[str((manifest_path.parent/'labels.train.json').resolve())]=manifest['files']['labels.train.json']
        files[str(Path(contrasts_public).resolve())]=base.sha(contrasts_public)
        incoming += [(p,'train') for p in base.read_profiles(contrasts_public)]
    if independent_public:
        independent_manifest=Path(independent_public).parent/'MANIFEST.json'
        if base.sha(independent_manifest)!='2e7895cec205d0aa312dfdb9ced84bad829746fd288d165e978141f9e2149b17':raise ValueError('Exact independentv2 manifest required')
        files[str(independent_manifest.resolve())]=base.sha(independent_manifest)
        annotation_pins[str((Path(independent_public).parent/'labels.private.json').resolve())]=json.loads(independent_manifest.read_text())['files']['labels.private.json']
        if base.sha(independent_public)!='2994a34b2df30fe80600c88df2a878a962e2212a7436220d33032b0234b7a342':raise ValueError('Only frozen independentv2 PUBLIC32 allowed')
        files[str(Path(independent_public).resolve())]=base.sha(independent_public)
        value=json.loads(Path(independent_public).read_text()); incoming += [(p,'independent-evaluation') for p in value['profiles']]
    for p,split_name in incoming:
        base.checked_context_request(p['request'],catalog['products']);text=public_context_text(p['request']);ids=tokenizer.encode(text,add_special_tokens=False).ids
        if not 0<len(ids)<=CONFIG['maxTokens']:raise ValueError(f"Whole public context {p['profileId']} has {len(ids)}tokens; no truncation")
        rows.append({'role':'request','member':{'profileId':p['profileId']},'caseId':p['profileId'],'split':split_name,
            'text':text,'textSha256':digest(text),'tokenIds':ids,'tokenSha256':digest(canonical(ids)),'tokens':len(ids),'batchShape':[1,len(ids)],'publicProfile':p})
    if len({r['caseId'] for r in rows})!=len(rows):raise ValueError('Unique external profile membership required')
    packet={'version':VERSION,'config':CONFIG,'model':original['model'],'featureProvenance':report['featureProvenance'],
        'assetRoot':str(root.resolve().parents[3]),'originalCatalogRoot':str(root.resolve()),'originalCatalogPins':catalog['pins'],'publicFilesSha256':files,'sourceSha256':sources,'rows':rows,
        'annotationFilesExpectedSha256':annotation_pins,'expectedNativeForwards':len(rows),'nativeContextExtractionOnly':True,'privateGoldRead':False,
        'nativePreparation':{'B':1,'cache':None,'pooling':'last-valid-token','dtype':'float32','normalize':'l2','copyAdapter':'absent frozen base'},
        'tokenRange':[min(r['tokens'] for r in rows),max(r['tokens'] for r in rows)]}
    Path(out).mkdir(parents=True,exist_ok=False);base.dump(Path(out)/'preflight.json',packet)
    return packet


def extract(preflight, out, *, run=False):
    packet=json.loads(Path(preflight).read_text())
    def pins():
        for group in ('sourceSha256','publicFilesSha256','originalCatalogPins'):
            if any(base.sha(p)!=s for p,s in packet[group].items()):raise ValueError('Predeclared source/public/catalog pin changed')
    pins()
    if not run:return {'status':'cpu-preflight-only','expectedNativeForwards':packet['expectedNativeForwards'],'nativeExecuted':False}
    import os
    os.chdir(packet['assetRoot'])
    # Root dispatches this explicit native path only after shared cohort releases.
    from .store_ar_comparison_run_v1 import demo_listening
    from .catalog_fusion.generalization_v2 import native_owner_lock
    from .catalog_fusion.runtime import FeatureStore,verify_model,assert_model_files
    from .semantic_copy_training import parameter_hashes
    from .backbone import FrozenBackbone
    from .store_copy_fit_v2 import deadline
    if demo_listening():raise RuntimeError('Main8766 listening; gracefully stop before native extraction')
    out=Path(out)
    if not out.is_absolute() or not out.resolve().is_relative_to(base.HERE.parents[2]):raise ValueError('Explicit absolute unique worktree extraction output required')
    if out.exists():raise FileExistsError('Unique extraction output required')
    out.mkdir(parents=True)
    base.dump(out/'preflight.json',packet);bb=None;before=None;started=time.perf_counter()
    report={'version':VERSION,'status':'running','nativeForwards':0,'expectedNativeForwards':len(packet['rows']),
        'roster':[{'caseId':r['caseId'],'split':r['split'],'status':'unattempted'} for r in packet['rows']],
        'copyAdapterLoaded':False,'privateGoldRead':False,'frozen927HashesUnchanged':None,'timings':[]}
    def save():base.dump(out/'report.json',report)
    save()
    from contextlib import ExitStack
    resources=ExitStack()
    try:
        resources.enter_context(native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'),'root'))
        with deadline(CONFIG['maxNativeSeconds'],'persona native public context extraction lifecycle'):
            if demo_listening():raise RuntimeError('Main8766 became active')
            verify_model(packet['model']['modelPath'],packet['model']);assert_model_files(packet['model']['modelPath'],packet['featureProvenance'])
            bb=FrozenBackbone(packet['model']['modelPath'],packet['featureProvenance']['modelRevision'],max_length=CONFIG['maxTokens'],memory_gib=CONFIG['memoryGiB'])
            before=parameter_hashes(bb.model.language_model)
            if len(before)!=927:raise ValueError('Expected927 frozen source tensors')
            store=FeatureStore(out/'features',packet['featureProvenance'])
            for row,receipt in zip(packet['rows'],report['roster'],strict=True):
                pins();receipt['status']='running';report['activeCaseId']=row['caseId'];save()
                actual=bb.tokenizer.encode(row['text'],add_special_tokens=False)
                if actual!=row['tokenIds']:raise ValueError('Native token IDs differ from CPUpreflight')
                values,timing=bb.encode([row['text']]);report['nativeForwards']+=1
                if timing['batchRows']!=1 or timing['backboneForwardCalls']!=1 or timing['truncatedRows']!=0 or timing['paddedTokensPerRow']!=row['tokens']:raise ValueError('Whole freshB1 native receipt mismatch')
                store.save(row,values,timing);receipt['status']='completed';report['timings'].append({'caseId':row['caseId'],**timing});save()
            pins();report['serializedFeatureFilesSha256']={str(p.resolve()):base.sha(p) for row in packet['rows'] for p in store.paths(row['role'],row['member'])}
            report['status']='completed-context-features'
    except BaseException as error:
        for item in report['roster']:
            if item['status']=='running':item.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        try:
            if bb is not None:
                try:
                    after=parameter_hashes(bb.model.language_model)
                    report['frozen927HashesUnchanged']=before is not None and before==after and len(after)==927
                    report['memory']=bb.runtime_memory()
                    if not report['frozen927HashesUnchanged']:report['status']='failed-frozen-integrity'
                except BaseException as error:
                    report.update(status='failed-shutdown-integrity',shutdownIntegrityError={'type':type(error).__name__,'message':str(error)})
                finally:
                    mx=bb.mx;bb=None
                    import gc
                    try:
                        gc.collect();mx.clear_cache();mx.synchronize();report['shutdownCompleted']=True
                    except BaseException as error:
                        report.update(status='failed-shutdown',shutdownCompleted=False,shutdownError={'type':type(error).__name__,'message':str(error)})
        finally:
            try:
                report['elapsedSeconds']=time.perf_counter()-started;save()
            finally:resources.close()
    if report['status']!='completed-context-features':raise RuntimeError('Native context extraction did not complete integrity/shutdown: '+report['status'])
    return report


def main(argv=None):
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for key in ('public-export','feature-root','out'):a.add_argument('--'+key,type=Path,required=True)
    for key in ('demo-export','probe-public','independent-public','contrasts-public'):a.add_argument('--'+key,type=Path)
    b=sub.add_parser('extract');b.add_argument('--preflight',type=Path,required=True);b.add_argument('--out',type=Path,required=True);b.add_argument('--run',action='store_true')
    args=p.parse_args(argv)
    if args.command=='prepare':
        result=prepare(args.public_export,args.feature_root,args.out,demo_export=args.demo_export,probe_public=args.probe_public,independent_public=args.independent_public,contrasts_public=args.contrasts_public)
        print(json.dumps({'status':'prepared','expectedNativeForwards':result['expectedNativeForwards'],'tokenRange':result['tokenRange']}))
    else:print(json.dumps(extract(args.preflight,args.out,run=args.run)))
if __name__=='__main__':main()
