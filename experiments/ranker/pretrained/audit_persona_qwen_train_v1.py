"""Independent saved-checkpoint TRAIN diagnosis; CPU inference, no optimizer."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from . import audit_persona_qwen_candidate_v1 as audit

HERE=Path(__file__).resolve().parent


def distribution(values):
    import torch
    x=torch.as_tensor(values,dtype=torch.float64).flatten()
    if not x.numel(): return {'count':0}
    q=torch.quantile(x,torch.tensor([0.,.05,.5,.95,1.],dtype=torch.float64))
    return {'count':x.numel(),'mean':x.mean().item(),'stdPopulation':x.std(unbiased=False).item(),
            **dict(zip(('min','p05','median','p95','max'),q.tolist(),strict=True))}


def independent_targets(profiles,labels,catalog):
    import torch
    ids=[p['productId'] for p in catalog['products']];by={r['profileId']:r for r in labels}
    audit.require(len(profiles)==len(labels)==len(by)==320 and set(by)=={p['profileId'] for p in profiles},'Exact320 TRAIN profile/target join')
    product=[];variant=[];aligned=[]
    for p in profiles:
        label=by[p['profileId']];g=label['relevanceGrades']
        audit.require(set(g)==set(ids) and all(type(v) is int and 0<=v<=3 for v in g.values()),'All16 bounded TRAIN grades')
        audit.require(any(v==3 for v in g.values()),'At least one TRAIN grade3 required; multi-positive contrasts allowed')
        observed={e['productId'] for e in p['request']['publicHistory'] if e['eventType']=='purchase' and e.get('productId') in ids}
        audit.require(observed==set(label['alreadyOwnedProductIds']) and all(g[k]==0 for k in observed),'Public ownership/annotation mismatch')
        gain=torch.tensor([2**g[k]-1 for k in ids],dtype=torch.float32)
        audit.require(gain.sum()>0,'Positive utility required')
        preferred=set(label['preferredVariantIds']);known={v['variantId'] for v in catalog['variants']}
        audit.require(preferred<=known,'Preferred TRAIN variants must be real')
        vg=torch.tensor([float(gain[v['productIndex']]) if v['variantId'] in preferred else 0. for v in catalog['variants']])
        if vg.sum()==0:vg=torch.tensor([float(gain[v['productIndex']]) for v in catalog['variants']])
        product.append(gain/gain.sum());variant.append(vg/vg.sum());aligned.append(label)
    return torch.stack(product),torch.stack(variant),aligned


def entropy(targets):
    return -(targets*targets.clamp(min=1e-30).log()).sum(-1)


def geometry(x,classes=None):
    import torch
    n=len(x);mask=~torch.eye(n,dtype=torch.bool);pair=x@x.T
    mean=x.mean(0);centered=x-mean;norms=centered.norm(dim=1)
    eig=torch.linalg.eigvalsh(centered@centered.T).clamp(min=0)
    prob=eig/eig.sum().clamp(min=1e-30)
    result={'rows':n,'dimensions':x.shape[1],'rowNorms':distribution(x.norm(dim=1)),
            'meanVectorNorm':mean.norm().item(),'offDiagonalCosines':distribution(pair[mask]),
            'centeredRowNorms':distribution(norms),'centeredVarianceTrace':eig.sum().item(),
            'centeredEffectiveRank':float(entropy(prob[None,:]).exp()),
            'firstCenteredPrincipalVarianceFraction':float(eig[-1]/eig.sum().clamp(min=1e-30))}
    if classes is not None:
        classes=torch.tensor(classes,dtype=torch.float32);same=(classes@classes.T)>0
        result['classDefinition']='Pair shares at least one grade3 product label; multiple grade3 TRAIN labels remain valid.'
        result['withinPrimaryClassCosines']=distribution(pair[mask&same])
        result['betweenPrimaryClassCosines']=distribution(pair[mask&~same])
        normalized=torch.nn.functional.normalize(centered,dim=1);cp=normalized@normalized.T
        result['centeredWithinPrimaryClassCosines']=distribution(cp[mask&same])
        result['centeredBetweenPrimaryClassCosines']=distribution(cp[mask&~same])
    return result


def run(model_dir,feature_dir,out):
    import torch
    from .catalog_fusion.runtime import FeatureStore
    from . import store_persona_context_v1 as base
    from .store_demo_engine import guarded_order
    from . import store_persona_qwen_v1 as native
    torch.set_num_threads(1)
    model_dir,feature_dir,out=map(Path,(model_dir,feature_dir,out))
    audit.require(not out.exists() and out.resolve().is_relative_to(HERE.parents[2]),'Unique worktree audit output required')
    packet=audit.read(feature_dir/'preflight.json');native_report=audit.read(feature_dir/'report.json');report=audit.read(model_dir/'results.json')
    audit.require(audit.sha(feature_dir/'preflight.json')==audit.PREFLIGHT_SHA256 and report['status']=='completed','Reviewed actual completed fit required')
    audit.require(audit.sha(model_dir/'head.pt')==report['checkpointSha256']=='ad641a62da8845e5a904d9a326fc67eaf32d739792f09f73436761d834b9163a','Exact actual candidate checkpoint')
    # Require completed fixed predictions even though this analysis uses TRAIN.
    audit.require_prediction_freeze(model_dir,packet)
    label_path=HERE/'synthetic_history_train_contrasts_v2/labels.train.json'
    pair_path=HERE/'synthetic_history_train_contrasts_v2/pair-links.json'
    pins={str(p.resolve()):audit.sha(p) for p in (Path(__file__),model_dir/'head.pt',model_dir/'results.json',model_dir/'PREDICTIONS_FROZEN.json',feature_dir/'preflight.json',feature_dir/'report.json',label_path,pair_path)}
    for group in ('sourceSha256','publicFilesSha256','originalCatalogPins','annotationFilesExpectedSha256'):pins.update(packet[group])
    pins.update(native_report['serializedFeatureFilesSha256'])
    for path,expected in pins.items():audit.require(audit.sha(path)==expected,'Input pin differs: '+path)
    out.mkdir(parents=True);audit.write(out/'INPUTS_FROZEN.json',{'sourceSha256':audit.sha(__file__),'sha256':pins,'mode':'saved TRAIN320 inference only; no optimizer/native/browser','new32Scope':'now inspected DEVELOPMENT replay, no new unseen claim'})
    checkpoint=torch.load(model_dir/'head.pt',map_location='cpu',weights_only=True);state=checkpoint['state_dict']
    audit.verify_state(state,report,packet['config'])
    catalog=base.load_catalog_features(packet['originalCatalogRoot']);ids=[p['productId'] for p in catalog['products']]
    rows=[r for r in packet['rows'] if r['split']=='train'];profiles=[r['publicProfile'] for r in rows]
    audit.require(len(rows)==320 and len({p['profileId'] for p in profiles})==320 and Counter(Counter(p['userGroupId'] for p in profiles).values())=={4:80},'Exactly320 TRAIN in80 groups of4')
    combined=audit.read(HERE/'synthetic_history_train_contrasts_v2/profiles.train.public.json')['profiles']
    audit.require({p['profileId']:p for p in combined}=={p['profileId']:p for p in profiles},'TRAIN frozen public profile contents/order identity')
    audit.require(all(native.public_context_text(r['publicProfile']['request'])==r['text'] for r in rows),'Public context formatter differs from extracted TRAIN text')
    store=FeatureStore.open_existing(feature_dir/'features',packet['featureProvenance'])
    x=torch.cat([store.load(row)[0] for row in rows])
    labels=audit.read(label_path)['labels'];pt,vt,aligned=independent_targets(profiles,labels,catalog)
    original_pt,original_vt=base.targets(profiles,labels,catalog)
    audit.require(torch.equal(pt,original_pt) and torch.equal(vt,original_vt),'Independent ID-aligned product/variant targets differ from actual consumer')
    with torch.no_grad():ps,vs=audit.replay_scores(state,x,catalog,profiles)
    lp=-(pt*ps.log_softmax(-1)).sum(-1);lv=-(vt*vs.log_softmax(-1)).sum(-1)
    hp,hv=entropy(pt),entropy(vt);weight=packet['config']['variantLossWeight']
    predictions=[];evaluated=[];groups=defaultdict(list);classes=defaultdict(list)
    for i,(profile,label,scores,variant_scores) in enumerate(zip(profiles,aligned,ps.tolist(),vs.tolist(),strict=True)):
        request,policy,_=base.checked_context_request(profile['request'],catalog['products']);order,guards=guarded_order(scores,request,policy,catalog['products'])
        raw=sorted(range(16),key=lambda j:(-scores[j],j));primaries=[k for k,g in label['relevanceGrades'].items() if g==3];primary_key='|'.join(primaries);primary_indices=[ids.index(k) for k in primaries]
        selected=[]
        for index in order[:6]:
            choices=[(j,v) for j,v in enumerate(catalog['variants']) if v['productIndex']==index and all(h['attribute']!='color' or h['value']==v['colorName'] for h in request['hardRequirements'])]
            _,variant=max(choices,key=lambda t:(variant_scores[t[0]],-t[0]))
            selected.append({'productId':ids[index],'variantId':variant['variantId'],'selectedColor':variant['colorName'],'image':variant['images'][0]})
        prediction={'profileId':profile['profileId'],'userGroupId':profile['userGroupId'],'scores16':scores,'rawLearnedOrder':raw,'order':order,'guardChecks':guards,'selected':selected};predictions.append(prediction)
        top=ids[order[0]] if order else None;grades=label['relevanceGrades'];owned=set(label['alreadyOwnedProductIds'])
        gain=[2**grades[ids[j]]-1 for j in order[:6]];ideal=sorted([2**g-1 for g in grades.values()],reverse=True)[:6]
        import math
        dcg=lambda a:sum(v/math.log2(j+2) for j,v in enumerate(a))
        ranks=[order.index(j)+1 for j in primary_indices if j in order]
        item={'profileId':profile['profileId'],'userGroupId':profile['userGroupId'],'primaryProductIds':primaries,'topProductId':top,'primaryTop1':top in primaries,'rawPrimaryTop1':raw[0] in primary_indices,'primaryRawRank':min(raw.index(j)+1 for j in primary_indices),'primaryGuardedRank':min(ranks) if ranks else None,'novelRelevantTop1':bool(top and grades[top]>=2 and top not in owned),'strongPositiveTop3':any(grades[ids[j]]>=2 for j in order[:3]),'ownedTop1':top in owned,'ndcg6':dcg(gain)/dcg(ideal),'acceptableVariantTop1':bool(selected and selected[0]['variantId'] in label.get('acceptableVariantIds',label['preferredVariantIds'])),'productLoss':float(lp[i]),'variantLoss':float(lv[i]),'jointLoss':float(lp[i]+weight*lv[i]),'jointEntropyFloor':float(hp[i]+weight*hv[i])}
        evaluated.append(item);groups[item['userGroupId']].append(item);classes[primary_key].append(item)
    def counts(rs):return {'rows':len(rs),**{k:sum(r[k] for r in rs) for k in ('primaryTop1','rawPrimaryTop1','novelRelevantTop1','strongPositiveTop3','ownedTop1','acceptableVariantTop1')},'meanNdcg6':sum(r['ndcg6'] for r in rs)/len(rs),'primaryRawRanks':distribution([r['primaryRawRank'] for r in rs])}
    factual=audit.factual_checks(predictions,profiles,catalog)
    quick={'version':'persona-qwen-independent-train-diagnosis.v1','status':'TRAIN-replay-completed','checkpointSha256':audit.sha(model_dir/'head.pt'),'fixedDenominator':320,'groups':80,'targetAlignmentExact':True,'publicProfileBindingExact':True,'metrics':counts(evaluated),'perPrimaryClass':{k:counts(v) for k,v in classes.items()},'perGroup':{k:counts(v) for k,v in groups.items()},'loss':{'productCrossEntropy':float(lp.mean()),'variantCrossEntropy':float(lv.mean()),'variantWeight':weight,'jointCrossEntropy':float((lp+weight*lv).mean()),'productEntropyFloor':float(hp.mean()),'variantEntropyFloor':float(hv.mean()),'jointEntropyFloor':float((hp+weight*hv).mean()),'jointExcessOverEntropy':float((lp-hp+weight*(lv-hv)).mean()),'producerLossTrace':report['losses'],'lastLossRecordedBeforeFinalUpdate':True},'sourceFactChecks':factual,'rows':evaluated,'noOptimizerNativeBrowserExecuted':True}
    audit.write(out/'TRAIN_PREDICTIONS.json',predictions);audit.write(out/'QUICK_TRAIN.json',quick)
    print(json.dumps({'quickTrainReady':True,'metrics':quick['metrics'],'loss':quick['loss'],'factViolations':factual['violationCount']}),flush=True)
    primary_classes=[[float(k in r['primaryProductIds']) for k in ids] for r in evaluated]
    geom={'TRAIN':geometry(x,primary_classes),'catalog':geometry(catalog['vectors'])}
    by={p['profileId']:i for i,p in enumerate(profiles)};links=audit.read(pair_path)['pairs'];pair_results=[]
    audit.require(len(links)==160,'Exactly160 TRAIN original/contrast pairs')
    for link in links:
        a,b=by[link['sourceProfileId']],by[link['contrastProfileId']];owned=ids.index(link['newlyOwnedProductId'])
        audit.require(profiles[a]['userGroupId']==profiles[b]['userGroupId']==link['userGroupId'],'Whole-group contrast alignment')
        before=predictions[a]['rawLearnedOrder'].index(owned)+1;after=predictions[b]['rawLearnedOrder'].index(owned)+1
        pair_results.append({'sourceProfileId':profiles[a]['profileId'],'contrastProfileId':profiles[b]['profileId'],'cosine':float(x[a]@x[b]),'newlyOwnedProductId':ids[owned],'rawRankBefore':before,'rawRankAfter':after,'ownedPriorityDemoted':after>before,'novelRelevantAfterTop1':evaluated[b]['novelRelevantTop1']})
    geom['originalContrastPairs']={'pairs':160,'cosines':distribution([p['cosine'] for p in pair_results]),'rawRankDemotions':sum(p['ownedPriorityDemoted'] for p in pair_results),'novelRelevantAfterTop1':sum(p['novelRelevantAfterTop1'] for p in pair_results),'rows':pair_results}
    with torch.no_grad():
        hidden=torch.nn.functional.normalize(torch.tanh(torch.nn.functional.linear(x,state['context.weight'],state['context.bias'])),dim=1)
        products=torch.nn.functional.normalize(torch.nn.functional.linear(catalog['vectors'],state['product.weight']),dim=1)
    geom['learnedTRAINContext']=geometry(hidden,primary_classes);geom['learnedCatalog']=geometry(products)
    for path,expected in pins.items():audit.require(audit.sha(path)==expected,'Original input changed during independent diagnosis: '+path)
    result={'version':quick['version'],'status':'completed-saved-TRAIN-diagnosis','sourceSha256':audit.sha(__file__),'inputsFreezeSha256':audit.sha(out/'INPUTS_FROZEN.json'),'quickTrainSha256':audit.sha(out/'QUICK_TRAIN.json'),'trainPredictionsSha256':audit.sha(out/'TRAIN_PREDICTIONS.json'),'geometry':geom,'allOriginalInputsUnchanged':True,'limitations':['Saved-checkpoint TRAIN analysis, no additional updates or new candidate.','All fresh32 outcomes now inspected; they are DEVELOPMENT replay.','Anisotropy measurements alone do not establish a causal fix.','Entropy floor is a target-distribution bound; shared parameters and guards can make it unattainable.']}
    audit.write(out/'DIAGNOSIS.json',result)
    print(json.dumps({'completed':True,'out':str(out),'trainMeanNorm':geom['TRAIN']['meanVectorNorm'],'trainPairCosine':geom['TRAIN']['offDiagonalCosines']['mean'],'withinClassCosine':geom['TRAIN']['withinPrimaryClassCosines']['mean'],'betweenClassCosine':geom['TRAIN']['betweenPrimaryClassCosines']['mean'],'pairCosine':geom['originalContrastPairs']['cosines']['mean'],'trainPairDemotions':geom['originalContrastPairs']['rawRankDemotions']}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('run','features','out'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.run,a.features,a.out)


if __name__=='__main__':main()
