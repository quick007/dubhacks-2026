"""Three predeclared CPU optimization arms on immutable native Qwen features.

Every nonfit case is DEVELOPMENT, including the previously evaluated32. Default
operation freezes a plan without training. All three final heads/predictions are
frozen before any development annotation file is parsed. No native imports.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from . import store_persona_context_v1 as base
from . import store_persona_qwen_fit_v1 as previous

VERSION = 'public-persona-qwen-optimization.v2'
ARMS = [dict(name='raw-2500', centered=False, steps=2500),
        dict(name='centered-250', centered=True, steps=250),
        dict(name='centered-1000', centered=True, steps=1000)]
SPLITS = ('development-replay', 'independent-evaluation', 'history-diagnostic-nonfit',
          'demo-nonfit', 'demo-default-query-nonfit')
COUNTS = dict(zip(SPLITS, (40,32,20,10,10)))
CONFIG = {**previous.CONFIG, 'selection':'three-arm DEVELOPMENT comparison; no unseen claim',
          'independentScope':'previous independent32 now DEVELOPMENT; no untouched evaluation in this comparison',
          'arms':ARMS, 'contextCenter':'mean of320 TRAIN contexts only',
          'catalogCenter':'mean of16 public catalog vectors only',
          'centeringOperation':'subtract mean then L2 normalize with eps1e-12; same transform at inference'}

def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def checked_matrix(value):
    if value.ndim != 2 or value.shape[1] != 4096 or not torch.isfinite(value).all():
        raise ValueError('Finite Bx4096 native features required')
    return value

def centers(train, catalog, centered):
    checked_matrix(train); checked_matrix(catalog)
    return (train.mean(0) if centered else torch.zeros(4096),
            catalog.mean(0) if centered else torch.zeros(4096))

class OptimizationHead(previous.NativePersonaHead):
    def __init__(self, context_mean, catalog_mean, centered):
        super().__init__()
        self.register_buffer('context_mean', context_mean.detach().clone())
        self.register_buffer('catalog_mean', catalog_mean.detach().clone())
        self.centered = centered

    def transform(self, context, catalog):
        if self.centered:
            return F.normalize(context-self.context_mean,dim=-1,eps=1e-12), F.normalize(catalog-self.catalog_mean,dim=-1,eps=1e-12)
        return context, catalog

    def forward(self, context, catalog, colors, indices, owned):
        context,catalog=self.transform(context,catalog)
        return super().forward(context,catalog,colors,indices,owned)

def load_head(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['version']!=VERSION or saved['config']!=CONFIG or saved['arm'] not in ARMS:
        raise ValueError('Exact frozen optimization recipe required')
    state=saved['state_dict']
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite head')
    head=OptimizationHead(state['context_mean'],state['catalog_mean'],saved['arm']['centered'])
    head.load_state_dict(state,strict=True)
    return head.eval().requires_grad_(False)

@torch.no_grad()
def loss_diagnostics(ps, vs, pt, vt):
    p_loss=-(pt*ps.log_softmax(-1)).sum(-1).mean()
    v_loss=-(vt*vs.log_softmax(-1)).sum(-1).mean()
    entropy=lambda p:float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean())
    ph,vh=entropy(pt),entropy(vt)
    return {'loss':float(p_loss+CONFIG['variantLossWeight']*v_loss),
            'productCrossEntropy':float(p_loss),'variantCrossEntropy':float(v_loss),
            'productTargetEntropy':ph,'variantTargetEntropy':vh,
            'targetEntropyLossFloor':ph+CONFIG['variantLossWeight']*vh,
            'productPredictionEntropy':entropy(ps.softmax(-1)),
            'variantPredictionEntropy':entropy(vs.softmax(-1))}

def metrics(rows, labels, catalog):
    if len(labels) in (32,40):return previous.evaluate(rows,labels,catalog)
    result=base.evaluate_rows(rows,labels,catalog);n=len(labels)
    for k in ('topGrade3Rate','novelRelevantTop1Rate','ownedDuplicateTop1Rate','ndcg6','acceptableVariantTop1Rate'):result[k]*=40/n
    result['fixedDenominator']=n
    return result

def validate_pins(pins):
    for path, checksum in pins.items():
        if base.sha(path)!=checksum:raise ValueError('Immutable experiment input changed: '+path)

def prepare(features, seed_data, contrasts, dev32_gold, out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Unique plan output required')
    if not out.is_relative_to(base.HERE.parents[2]):raise ValueError('Worktree output required')
    feature_dir=Path(features).resolve();packet=json.loads((feature_dir/'preflight.json').read_text())
    extraction=json.loads((feature_dir/'report.json').read_text())
    if (extraction['status']!='completed-context-features' or extraction['nativeForwards']!=432
        or extraction['frozen927HashesUnchanged'] is not True or extraction['shutdownCompleted'] is not True
        or len(extraction['serializedFeatureFilesSha256'])!=864):raise ValueError('Completed432 native feature receipt required')
    pins={}
    for group in ('sourceSha256','publicFilesSha256','originalCatalogPins','annotationFilesExpectedSha256'):
        pins.update(packet[group])
    pins.update(extraction['serializedFeatureFilesSha256'])
    paths=[Path(__file__),Path(base.HERE)/'test_store_persona_qwen_optimize_v2.py',Path(base.HERE)/'audit_persona_qwen_candidate_v1.py',feature_dir/'preflight.json',feature_dir/'report.json',
           Path(seed_data)/'labels.heldout.json',Path(seed_data)/'history-only-probes.labels.json',
           Path(seed_data)/'history-only-probes.public.json',Path(contrasts)/'labels.train.json',
           Path(dev32_gold),Path(base.HERE.parents[2])/'data/ranker/pretrained/persona-qwen-fit-v1-attempt-1/results.json',
           Path(base.HERE.parents[2])/'data/ranker/pretrained/persona-qwen-fit-v1-attempt-1/head.pt']
    for path in paths:
        key=str(path.resolve());actual=base.sha(path)
        if key in pins and pins[key]!=actual:raise ValueError('Previously frozen pin mismatch')
        pins[key]=actual
    validate_pins(pins)
    plan={'version':VERSION,'config':CONFIG,'features':str(feature_dir),'seedData':str(Path(seed_data).resolve()),
          'contrasts':str(Path(contrasts).resolve()),'dev32Gold':str(Path(dev32_gold).resolve()),'sha256':pins,
          'expectedNonfitCounts':COUNTS,'developmentGoldParsed':False,'optimizerUpdatesStarted':0,
          'scope':'Known16 authored synthetic development selection across exactly3 arms; old32 no longer independent',
          'freezeOrder':'all3 checkpoints and all nonfit predictions before any DEV gold parse',
          'noNewNativeCalls':True,'outputCompatibility':'research-only v2 head; no existing serving loader changes'}
    out.mkdir(parents=True);base.dump(out/'plan.json',plan)
    return plan

def run(plan_path, plan_sha, out):
    if base.sha(plan_path)!=plan_sha:raise ValueError('Exact reviewed plan SHA required')
    plan=json.loads(Path(plan_path).read_text())
    if plan['version']!=VERSION or plan['config']!=CONFIG:raise ValueError('Frozen three-arm recipe mismatch')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Unique experiment output required')
    if not out.is_relative_to(base.HERE.parents[2]):raise ValueError('Worktree output required')
    out.mkdir(parents=True);start=time.monotonic()
    report={'version':VERSION,'status':'preflight','planSha256':plan_sha,'scope':plan['scope'],
            'nativeCalls':0,'actualUpdates':0,'arms':{},'developmentGoldParsed':False}
    def save():base.dump(out/'results.json',report)
    save()
    try:
        validate_pins(plan['sha256'])
        from .catalog_fusion.runtime import FeatureStore
        feature_dir=Path(plan['features']);packet=json.loads((feature_dir/'preflight.json').read_text())
        catalog=base.load_catalog_features(packet['originalCatalogRoot'])
        store=FeatureStore.open_existing(feature_dir/'features',packet['featureProvenance'])
        rows=packet['rows'];by={r['caseId']:checked_matrix(store.load(r)[0]) for r in rows}
        profiles=[r['publicProfile'] for r in rows if r['split']=='train']
        if len(profiles)!=320:raise ValueError('Fixed320 TRAIN required')
        train=torch.cat([by[p['profileId']] for p in profiles])
        labels=base.label_rows(Path(plan['contrasts'])/'labels.train.json')
        pt,vt=base.targets(profiles,labels,catalog);colors,indices=base.source_colors(catalog);owned=previous.ownership(profiles,catalog)
        rosters={split:[r['publicProfile'] for r in rows if r['split']==split] for split in SPLITS}
        if {s:len(v) for s,v in rosters.items()}!=COUNTS:raise ValueError('Fixed112 nonfit roster required')
        torch.set_num_threads(CONFIG['threads']);outputs={};frozen={}
        # All three fixed schedules execute before examining any DEV annotation.
        for arm in ARMS:
            name=arm['name'];directory=out/name;directory.mkdir()
            cm,pm=centers(train,catalog['vectors'],arm['centered'])
            head=OptimizationHead(cm,pm,arm['centered']);initial={k:v.detach().clone() for k,v in head.state_dict().items()}
            state={'status':'training','actualUpdates':0,'arm':arm,'losses':[],
                   'contextMeanSha256':tensor_sha(cm),'catalogMeanSha256':tensor_sha(pm)}
            report['arms'][name]=state;save()
            base.dump(directory/'ARM_FROZEN.json',{'arm':arm,'config':CONFIG,'planSha256':plan_sha,
                      'centersDerivedFromTrain320AndPublicCatalog16Only':True,
                      'contextMeanSha256':tensor_sha(cm),'catalogMeanSha256':tensor_sha(pm),
                      'initialStateSha256':{k:tensor_sha(v) for k,v in initial.items()},'developmentGoldParsed':False})
            with torch.no_grad():state['initialDiagnostics']=loss_diagnostics(*head(train,catalog['vectors'],colors,indices,owned),pt,vt)
            optimizer=torch.optim.AdamW(head.parameters(),lr=CONFIG['learningRate'],weight_decay=CONFIG['weightDecay'])
            nonzero=0
            for step in range(arm['steps']):
                optimizer.zero_grad();ps,vs=head(train,catalog['vectors'],colors,indices,owned)
                loss=-(pt*ps.log_softmax(-1)).sum(-1).mean()-CONFIG['variantLossWeight']*(vt*vs.log_softmax(-1)).sum(-1).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):raise FloatingPointError('Nonfinite/missing gradient')
                nonzero+=int(float(head.ownership.grad)!=0);optimizer.step()
                state['actualUpdates']=step+1;report['actualUpdates']+=1
                if step%25==0 or step==arm['steps']-1:
                    state['losses'].append({'step':step+1,'preUpdateLoss':float(loss.detach()),'ownership':float(head.ownership.detach())});save()
            head.eval()
            with torch.no_grad():state['finalDiagnostics']=loss_diagnostics(*head(train,catalog['vectors'],colors,indices,owned),pt,vt)
            state['changedParameters']=[k for k,v in head.named_parameters() if not torch.equal(v,initial[k])]
            state['ownershipGradientNonzeroSteps']=nonzero;state['ownershipCoefficient']=float(head.ownership.detach())
            if not nonzero or 'ownership' not in state['changedParameters']:raise ValueError('Actual ownership training required')
            torch.save({'version':VERSION,'config':CONFIG,'arm':arm,'state_dict':head.state_dict(),
                        'planSha256':plan_sha,'featureProvenance':packet['featureProvenance'],
                        'trainingScope':'320TRAIN; all112 nonfit public tensors not used for optimizer or centering'},directory/'head.pt')
            restored=load_head(directory/'head.pt')
            state['reloadExact']=all(torch.equal(v,restored.state_dict()[k]) for k,v in head.state_dict().items())
            # Exact wholeTRAIN plus all112 nonfit reloaded predictions, not just two rows.
            outputs[name]={}
            for split,roster in {'train':profiles,**rosters}.items():
                x=torch.cat([by[p['profileId']] for p in roster])
                before=previous.predict(head,roster,x,catalog);after=previous.predict(restored,roster,x,catalog)
                if before!=after:raise ValueError('Whole-roster reload prediction parity failed')
                outputs[name][split]=after;base.dump(directory/(split+'-predictions.json'),after)
            state['reloadPredictionParity']=True;state['trainMetrics']=metrics(outputs[name]['train'],labels,catalog)
            state['checkpointSha256']=base.sha(directory/'head.pt');state['status']='checkpoint-and-predictions-frozen'
            frozen[name]={'checkpointSha256':state['checkpointSha256'],
                          'files':{s:base.sha(directory/(s+'-predictions.json')) for s in outputs[name]}}
            validate_pins(plan['sha256']);save()
        base.dump(out/'ALL_ARMS_PREDICTIONS_FROZEN.json',{'planSha256':plan_sha,'arms':frozen,
                  'actualUpdates':report['actualUpdates'],'developmentGoldParsed':False,
                  'developmentSets':{'old40':40,'old32':32,'history':20,'demo':10,'demoDefault':10}})
        validate_pins(plan['sha256'])
        dev40=base.label_rows(Path(plan['seedData'])/'labels.heldout.json')
        dev32=base.label_rows(plan['dev32Gold'])
        probe_public=json.loads((Path(plan['seedData'])/'history-only-probes.public.json').read_text())
        probe_gold=base.label_rows(Path(plan['seedData'])/'history-only-probes.labels.json')
        from .audit_persona_qwen_candidate_v1 import history_metrics, factual_checks
        report['developmentGoldParsed']=True
        for arm in ARMS:
            name=arm['name'];state=report['arms'][name];o=outputs[name]
            state['development40Metrics']=metrics(o['development-replay'],dev40,catalog)
            state['development32Metrics']=metrics(o['independent-evaluation'],dev32,catalog)
            state['historyDevelopmentMetrics']=history_metrics(o['history-diagnostic-nonfit'],probe_public,probe_gold,catalog['products'])
            state['nonfitFactGuardChecks']={s:factual_checks(o[s],rosters[s],catalog) for s in SPLITS}
            state['status']='completed-development-comparison'
        validate_pins(plan['sha256']);report.update(status='completed',allOriginalPinsUnchanged=True,elapsedSeconds=time.monotonic()-start);save()
        return report
    except BaseException as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)},elapsedSeconds=time.monotonic()-start);save();raise

def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',action='store_true');p.add_argument('--out',type=Path,required=True)
    for key in ('features','seed-data','contrasts','dev32-gold','plan'):p.add_argument('--'+key,type=Path)
    p.add_argument('--plan-sha256');a=p.parse_args(argv)
    if a.run:
        if not a.plan or not a.plan_sha256:p.error('--run requires --plan and --plan-sha256')
        result=run(a.plan,a.plan_sha256,a.out)
        print(json.dumps({'status':result['status'],'actualUpdates':result['actualUpdates'],'arms':list(result['arms'])}))
    else:
        if any(getattr(a,k) is None for k in ('features','seed_data','contrasts','dev32_gold')):p.error('Prepare requires features, seed-data, contrasts, dev32-gold')
        prepare(a.features,a.seed_data,a.contrasts,a.dev32_gold,a.out)
        print(json.dumps({'status':'plan-frozen-no-optimizer','planSha256':base.sha(a.out/'plan.json')}))

if __name__=='__main__':main()
