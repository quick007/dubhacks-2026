"""Fixed CPU head fit on genuine native-public-context tensors; no GPU loads.

Original40 are development replay. Optional independent gold is opened only
AFTER checkpoint plus every independent prediction is durably frozen.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import argparse
import json
import torch
from torch import nn
from torch.nn import functional as F
from . import store_persona_context_v1 as base
from . import store_persona_qwen_v1 as native

VERSION='public-persona-qwen-head.v1'
CONFIG=native.CONFIG

class NativePersonaHead(nn.Module):
    def __init__(self):
        super().__init__();torch.manual_seed(CONFIG['seed'])
        self.context=nn.Linear(4096,CONFIG['hiddenWidth'])
        self.product=nn.Linear(4096,CONFIG['hiddenWidth'],bias=False)
        self.color_context=nn.Linear(4096,CONFIG['colorWidth'])
        self.color_candidate=nn.Linear(CONFIG['colorDimensions'],CONFIG['colorWidth'],bias=False)
        self.scale=nn.Parameter(torch.tensor(2.))
        self.ownership=nn.Parameter(torch.tensor(0.))
    def forward(self,context,catalog,colors,indices,owned):
        similarity=F.normalize(torch.tanh(self.context(context)),dim=-1)@F.normalize(self.product(catalog),dim=-1).T*self.scale.exp().clamp(max=30)
        products=similarity+self.ownership*owned
        color=F.normalize(torch.tanh(self.color_context(context)),dim=-1)@F.normalize(self.color_candidate(colors),dim=-1).T*4
        return products,products[:,indices]+color

def ownership(profiles,catalog):
    ids=[p['productId'] for p in catalog['products']]
    return torch.tensor([[float(any(e['eventType']=='purchase' and e.get('productId')==pid for e in p['request']['publicHistory'])) for pid in ids] for p in profiles])

def load(path,expected=None):
    if expected and base.sha(path)!=expected:raise ValueError('Qwen persona checkpoint changed')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value['version']!=VERSION or value['config']!=CONFIG:raise ValueError('Exact frozen native persona architecture required')
    if any(not torch.isfinite(v).all() for v in value['state_dict'].values()):raise ValueError('Nonfinite checkpoint')
    head=NativePersonaHead();head.load_state_dict(value['state_dict'],strict=True)
    return head.eval().requires_grad_(False),value

def predict(head,profiles,features,catalog,*,owned_override=None):
    colors,indices=base.source_colors(catalog)
    with torch.no_grad():ps,vs=head(features,catalog['vectors'],colors,indices,ownership(profiles,catalog) if owned_override is None else owned_override)
    from .store_demo_engine import guarded_order
    result=[]
    for profile,p,v in zip(profiles,ps.tolist(),vs.tolist(),strict=True):
        request,policy,_=base.checked_context_request(profile['request'],catalog['products']);order,guards=guarded_order(p,request,policy,catalog['products'])
        selected=[]
        for i in order[:6]:
            choices=[(j,variant) for j,variant in enumerate(catalog['variants']) if variant['productIndex']==i and all(r['attribute']!='color' or r['value']==variant['colorName'] for r in request['hardRequirements'])]
            j,variant=max(choices,key=lambda item:(v[item[0]],-item[0]))
            selected.append({'productId':catalog['products'][i]['productId'],'variantId':variant['variantId'],'selectedColor':variant['colorName'],'image':variant['images'][0]})
        result.append({'profileId':profile['profileId'],'userGroupId':profile['userGroupId'],'scores16':p,'rawLearnedOrder':sorted(range(16),key=lambda i:(-p[i],i)),'order':order,'guardChecks':guards,'selected':selected})
    return result

def evaluate(predictions,labels,catalog):
    n=len(labels)
    if n not in (32,40):raise ValueError('Fixed32 independent or40 development denominator required')
    metrics=base.evaluate_rows(predictions,labels,catalog)
    for key in ('topGrade3Rate','novelRelevantTop1Rate','ownedDuplicateTop1Rate','ndcg6','acceptableVariantTop1Rate'):metrics[key]*=40/n
    metrics['fixedDenominator']=n
    gold={r['profileId']:r for r in labels};ids=[p['productId'] for p in catalog['products']];extra=[]
    for p in predictions:
        label=gold[p['profileId']];top=ids[p['order'][0]] if p['order'] else None;raw=ids[p['rawLearnedOrder'][0]] if p.get('rawLearnedOrder') else None
        extra.append({'profileId':p['profileId'],'pairId':label.get('pairId',p['userGroupId']),
            'primaryTop1':top==label.get('primaryProductId'),'rawPrimaryTop1':raw==label.get('primaryProductId'),
            'positiveTop3':any(label['relevanceGrades'][ids[i]]>0 for i in p['order'][:3]),
            'strongPositiveTop3':any(label['relevanceGrades'][ids[i]]>=2 for i in p['order'][:3])})
    pairs={}
    for row in extra:pairs.setdefault(row['pairId'],[]).append(row)
    metrics.update(primaryTop1=sum(r['primaryTop1'] for r in extra),rawPrimaryTop1=sum(r['rawPrimaryTop1'] for r in extra),
        positiveTop3=sum(r['positiveTop3'] for r in extra),strongPositiveTop3=sum(r['strongPositiveTop3'] for r in extra),
        bothPairSidesPrimary=sum(len(rs)==2 and all(r['primaryTop1'] for r in rs) for rs in pairs.values()),
        bothPairSidesRawPrimary=sum(len(rs)==2 and all(r['rawPrimaryTop1'] for r in rs) for rs in pairs.values()),pairRows=extra)
    if any('primaryProductId' not in label for label in labels):
        for key in ('primaryTop1','rawPrimaryTop1','bothPairSidesPrimary','bothPairSidesRawPrimary'):metrics[key]=None
        for row in metrics['pairRows']:row['primaryTop1']=None;row['rawPrimaryTop1']=None
    return metrics

def fit(features,seed_data,contrasts,out,*,independent_gold=None):
    from .catalog_fusion.runtime import FeatureStore
    out=Path(out)
    if out.exists():raise FileExistsError('Unique model output required')
    if not out.resolve().is_relative_to(base.HERE.parents[2]):raise ValueError('Write only isolated worktree')
    out.mkdir(parents=True);report={'version':VERSION,'status':'preflight','actualUpdates':0,'config':CONFIG,'independentStatus':'unattempted','old40Scope':'development replay'}
    def save():base.dump(out/'results.json',report)
    save()
    try:
        feature_dir=Path(features);packet=json.loads((feature_dir/'preflight.json').read_text());extraction=json.loads((feature_dir/'report.json').read_text())
        if extraction['status']!='completed-context-features' or extraction['nativeForwards']!=packet['expectedNativeForwards'] or extraction['frozen927HashesUnchanged'] is not True:raise ValueError('Completed genuine context features and927integrity required')
        paths=[Path(__file__),Path(native.__file__),Path(base.__file__),feature_dir/'preflight.json',feature_dir/'report.json',Path(seed_data)/'labels.train.json',Path(seed_data)/'labels.heldout.json',Path(contrasts)/'labels.train.json']
        if independent_gold:
            if str(Path(independent_gold).resolve()) not in packet['annotationFilesExpectedSha256']:raise ValueError('Only frozen independentv2 private annotation path allowed')
            paths.append(Path(independent_gold))
        pins={str(p.resolve()):base.sha(p) for p in paths}
        pins.update(extraction['serializedFeatureFilesSha256'])
        if len(extraction['serializedFeatureFilesSha256'])!=2*packet['expectedNativeForwards']:raise ValueError('Exact two immutable serializedfiles per nativefeature required')
        base.dump(out/'RECIPE_AND_INPUTS_FROZEN.json',{'config':CONFIG,'sha256':pins,'independentGoldParsed':False,'old40PreviouslyInspected':True,'selection':'fixed final250step no validation/earlystop'})
        def assert_pins():
            for group in ('sourceSha256','publicFilesSha256','originalCatalogPins','annotationFilesExpectedSha256'):
                if any(base.sha(p)!=checksum for p,checksum in packet[group].items()):raise ValueError('Fit differs from native extraction source/public/catalog provenance')
            if any(base.sha(p)!=s for p,s in pins.items()):raise ValueError('Fixed fit source/input changed')
        assert_pins();catalog=base.load_catalog_features(packet['originalCatalogRoot'])
        store=FeatureStore.open_existing(feature_dir/'features',packet['featureProvenance'])
        rows=packet['rows'];by={r['caseId']:store.load(r)[0] for r in rows}
        profiles=[r['publicProfile'] for r in rows if r['split']=='train'];train_features=torch.cat([by[p['profileId']] for p in profiles])
        if len(profiles)!=320:raise ValueError('Exactly160original+160TRAINcontrasts required')
        labels=base.label_rows(Path(contrasts)/'labels.train.json')
        original_labels=base.label_rows(Path(seed_data)/'labels.train.json');joined={r['profileId']:r for r in labels}
        if any(joined.get(r['profileId'])!=r for r in original_labels):raise ValueError('Original frozen TRAIN annotations changed in combined contrast set')
        pt,vt=base.targets(profiles,labels,catalog);colors,indices=base.source_colors(catalog);owned=ownership(profiles,catalog)
        torch.set_num_threads(CONFIG['threads']);head=NativePersonaHead();initial={k:v.detach().clone() for k,v in head.state_dict().items()}
        optimizer=torch.optim.AdamW(head.parameters(),lr=CONFIG['learningRate'],weight_decay=CONFIG['weightDecay']);losses=[];ownership_gradients=[]
        for step in range(CONFIG['steps']):
            optimizer.zero_grad();ps,vs=head(train_features,catalog['vectors'],colors,indices,owned)
            loss=-(pt*ps.log_softmax(-1)).sum(-1).mean()-CONFIG['variantLossWeight']*(vt*vs.log_softmax(-1)).sum(-1).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native persona loss')
            loss.backward();ownership_gradients.append(float(head.ownership.grad));optimizer.step();report['actualUpdates']=step+1
            if step%25==0 or step==CONFIG['steps']-1:losses.append({'step':step+1,'loss':float(loss.detach()),'ownershipCoefficient':float(head.ownership.detach())})
            save()
        changed=[k for k,v in head.state_dict().items() if not torch.equal(v,initial[k])]
        if not changed or not any(g!=0 for g in ownership_gradients) or torch.equal(head.ownership,initial['ownership']):raise ValueError('Actual learned ownership update required')
        head.eval()
        demo_rows=[r for r in rows if r['split'] in ('demo-nonfit','demo-default-query-nonfit')]
        runtime_cache={'version':'public-persona-native-context-cache.v1','featureProvenance':packet['featureProvenance'], 'contextFormatterSha256':base.sha(Path(native.__file__)),
            'teacherOrLabels':False,'sourceExtractionReportSha256':base.sha(feature_dir/'report.json'),
            'features':{r['textSha256']:{'tensor':by[r['caseId']], 'tokenIds':r['tokenIds'],'textSha256':r['textSha256'],'split':r['split']} for r in demo_rows}}
        torch.save(runtime_cache,out/'public-demo-features.pt')
        torch.save({'version':VERSION,'config':CONFIG,'state_dict':head.state_dict(),'catalogPins':catalog['pins'],'featureProvenance':packet['featureProvenance'],'sourceFeaturePreflightSha256':base.sha(feature_dir/'preflight.json'),'publicDemoFeatureCacheFile':str((out/'public-demo-features.pt').resolve()),'publicDemoFeatureCacheSha256':base.sha(out/'public-demo-features.pt'),'trainingScope':'320authoredsyntheticTRAIN;160Tigerreadback+160disclosedTRAINcontrasts'},out/'head.pt')
        restored,_=load(out/'head.pt')
        if any(not torch.equal(v,restored.state_dict()[k]) for k,v in head.state_dict().items()):raise ValueError('Exact native head reload failed')
        first_train=profiles[:2];first_features=train_features[:2]
        if predict(head,first_train,first_features,catalog)!=predict(restored,first_train,first_features,catalog):raise ValueError('Native head reloaded prediction parity failed')
        report['reloadPredictionParity']=True
        # Freeze all nonfit predictions, including fresh independent32, before gold.
        outputs={}
        for split in ('development-replay','demo-nonfit','demo-default-query-nonfit','history-diagnostic-nonfit','independent-evaluation'):
            roster=[r['publicProfile'] for r in rows if r['split']==split]
            if roster:
                output=predict(restored,roster,torch.cat([by[p['profileId']] for p in roster]),catalog);base.dump(out/(split+'-predictions.json'),output);outputs[split]=output
        # Matched baselines fixed before independent gold access.
        independent=[r['publicProfile'] for r in rows if r['split']=='independent-evaluation']
        baseline_outputs={}
        if independent:
            ix=torch.cat([by[p['profileId']] for p in independent]);own=ownership(independent,catalog)
            baseline_outputs['ownership-term-disabled']=predict(restored,independent,ix,catalog,owned_override=torch.zeros_like(own))
            baseline_outputs['zero-context-owned-only']=predict(restored,independent,torch.zeros_like(ix),catalog)
            shuffled=deepcopy(independent)
            for i,p in enumerate(shuffled):
                other=independent[(i+17)%len(independent)]['request'];p['request']['persona']=deepcopy(other['persona']);p['request']['publicHistory']=deepcopy(other['publicHistory'])
            # Features and ownership both belong to the swapped public profile.
            baseline_outputs['shuffled-context']=predict(restored,shuffled,torch.roll(ix,-17,0),catalog)
            prior=pt.mean(0).tolist();baseline_outputs['training-popularity']=[base.baseline_prediction(p,prior,catalog) for p in independent]
            recent=[];product_ids=[p['productId'] for p in catalog['products']]
            for p in independent:
                scores=list(prior);events=[e for e in p['request']['publicHistory'] if e['eventType']=='purchase' and e.get('productId') in product_ids]
                if events:scores[product_ids.index(max(events,key=lambda e:e['occurredAt'])['productId'])]+=10
                recent.append(base.baseline_prediction(p,scores,catalog))
            baseline_outputs['most-recent-owned-repeat']=recent
            for name,output in baseline_outputs.items():base.dump(out/('independent-'+name+'-predictions.json'),output)
        base.dump(out/'PREDICTIONS_FROZEN.json',{'checkpointSha256':base.sha(out/'head.pt'),'files':{split:base.sha(out/(split+'-predictions.json')) for split in outputs},'baselineFiles':{name:base.sha(out/('independent-'+name+'-predictions.json')) for name in baseline_outputs},'independentGoldParsed':False})
        assert_pins();report.update(status='completed',losses=losses,checkpointSha256=base.sha(out/'head.pt'),reloadExact=True,changedTensors=changed,ownershipCoefficient=float(restored.ownership),ownershipGradientNonzeroSteps=sum(g!=0 for g in ownership_gradients))
        report['developmentReplayMetrics']=evaluate(outputs['development-replay'],base.label_rows(Path(seed_data)/'labels.heldout.json'),catalog)
        if independent_gold:
            labels=base.label_rows(independent_gold)
            metrics=evaluate(outputs['independent-evaluation'],labels,catalog)
            report['independentBaselineMetrics']={name:evaluate(output,labels,catalog) for name,output in baseline_outputs.items()}
            report['independentMetrics']=metrics;report['independentStatus']='completed-after-all-predictions-freeze'
        assert_pins();report['allFitPinsUnchanged']=True;save();return report
    except BaseException as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)});save();raise

def main(argv=None):
    p=argparse.ArgumentParser(__doc__)
    for key in ('features','seed-data','contrasts','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--independent-gold',type=Path);a=p.parse_args(argv)
    print(json.dumps(fit(a.features,a.seed_data,a.contrasts,a.out,independent_gold=a.independent_gold)))
if __name__=='__main__':main()
