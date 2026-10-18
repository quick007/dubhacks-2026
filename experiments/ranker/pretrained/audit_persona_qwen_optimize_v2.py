"""Independent saved three-arm CPU replay; no optimizer or native model."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

from . import audit_persona_qwen_candidate_v1 as audit
from .audit_persona_qwen_train_v1 import independent_targets,entropy

HERE=Path(__file__).resolve().parent
PLAN_SHA='b9ec986785923a00253e90609e7d72f0a2663f1f5a5ac7cf7a6765be091ad0c0'
ARMS={'raw-2500':(False,2500),'centered-250':(True,250),'centered-1000':(True,1000)}
SPLITS={'train':320,**audit.PRIMARY_SPLITS}
# Post-hoc manual interpretation of public demo facts, not frozen gold or a
# training/promotion input. These anchors are obvious functional examples only.
DEMO_ANCHORS={'bob':'blanket','maya':'lamp','priya':'notebook','lena':'jacket',
              'june':'headphones','grace':'lunchbox','sam':'duffel','ana':'organizer',
              'rosa':'towel','felix':'sneakers'}


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def grade_metrics(predictions,labels,products):
    ids=[p['productId'] for p in products];gold={r['profileId']:r for r in labels}
    audit.require(len(gold)==len(labels)==len(predictions) and set(gold)=={r['profileId'] for r in predictions},'Exact evaluated target join')
    result={'denominator':len(labels),'grade3Top1':0,'novelRelevantTop1':0,'strongPositiveTop3':0,'ownedTop1':0,'ndcg6':0.}
    import math
    for row in predictions:
        label=gold[row['profileId']];g=label['relevanceGrades'];order=[ids[i] for i in row['order']];top=order[0] if order else None
        result['grade3Top1']+=int(g.get(top,0)==3)
        result['novelRelevantTop1']+=int(g.get(top,0)>=2 and top not in label['alreadyOwnedProductIds'])
        result['strongPositiveTop3']+=int(any(g[k]>=2 for k in order[:3]))
        result['ownedTop1']+=int(top in label['alreadyOwnedProductIds'])
        dcg=lambda gains:sum(v/math.log2(j+2) for j,v in enumerate(gains))
        result['ndcg6']+=dcg([2**g[k]-1 for k in order[:6]])/dcg(sorted([2**v-1 for v in g.values()],reverse=True)[:6])/len(predictions)
    return result


def run(run_dir,plan_dir,out):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from . import store_persona_context_v1 as base
    from .catalog_fusion.runtime import FeatureStore
    from .store_demo_engine import guarded_order
    torch.set_num_threads(1)
    run_dir,plan_dir,out=map(Path,(run_dir,plan_dir,out))
    audit.require(not out.exists() and out.resolve().is_relative_to(HERE.parents[2]),'Unique worktree audit output required')
    plan_path=plan_dir/'plan.json';audit.require(audit.sha(plan_path)==PLAN_SHA,'Exact pre-optimizer plan required')
    plan=audit.read(plan_path);report=audit.read(run_dir/'results.json');freeze=audit.read(run_dir/'ALL_ARMS_PREDICTIONS_FROZEN.json')
    audit.require(report['status']=='completed' and report['actualUpdates']==3750 and report['nativeCalls']==0 and report['allOriginalPinsUnchanged'] is True,'Actual completed3750 CPU receipt required')
    audit.require(report['planSha256']==freeze['planSha256']==PLAN_SHA and freeze['actualUpdates']==3750 and freeze['developmentGoldParsed'] is False,'All-arm freeze chronology required')
    audit.require(set(report['arms'])==set(freeze['arms'])==set(ARMS),'Exact three-arm denominator')
    pins=dict(plan['sha256']);pins[str(plan_path)]=PLAN_SHA
    for path in (Path(__file__),run_dir/'results.json',run_dir/'ALL_ARMS_PREDICTIONS_FROZEN.json'):pins[str(path.resolve())]=audit.sha(path)
    outputs={}
    for name,(centered,steps) in ARMS.items():
        arm=report['arms'][name];f=freeze['arms'][name];directory=run_dir/name
        audit.require(arm['actualUpdates']==steps and arm['reloadExact'] is True and arm['reloadPredictionParity'] is True and arm['status']=='completed-development-comparison','Exact per-arm update/reload receipt required')
        audit.require(0<arm['ownershipGradientNonzeroSteps']<=steps,'Actual ownership gradients required')
        audit.require([r['step'] for r in arm['losses']]==[i+1 for i in range(steps) if i%25==0 or i==steps-1],'Fixed training trace denominator')
        audit.require(audit.sha(directory/'head.pt')==f['checkpointSha256']==arm['checkpointSha256'],'Frozen head differs')
        audit.require(set(f['files'])==set(SPLITS),'All432 prediction splits required')
        pins[str(directory/'head.pt')]=f['checkpointSha256'];pins[str(directory/'ARM_FROZEN.json')]=audit.sha(directory/'ARM_FROZEN.json')
        outputs[name]={}
        for split,expected in f['files'].items():
            path=directory/(split+'-predictions.json');audit.require(audit.sha(path)==expected,'Prediction file changed')
            pins[str(path)]=expected;outputs[name][split]=audit.read(path)
        audit.require((run_dir/'ALL_ARMS_PREDICTIONS_FROZEN.json').stat().st_mtime_ns>=max(Path(p).stat().st_mtime_ns for p in pins if str(directory) in p),'All-arm freeze must follow checkpoint/prediction artifacts')
    for path,expected in pins.items():audit.require(audit.sha(path)==expected,'Original/source pin differs: '+path)
    packet=audit.read(Path(plan['features'])/'preflight.json')
    audit.require(audit.sha(Path(plan['features'])/'preflight.json')==audit.PREFLIGHT_SHA256,'Actual original432 feature preflight required')
    rosters={s:[r['publicProfile'] for r in packet['rows'] if r['split']==s] for s in SPLITS}
    audit.require({s:len(v) for s,v in rosters.items()}==SPLITS,'Fixed432 public profile denominator')
    for name in ARMS:
        for split in SPLITS:audit.exact_rows(outputs[name][split],rosters[split],name+'/'+split)
    catalog=base.load_catalog_features(packet['originalCatalogRoot']);store=FeatureStore.open_existing(Path(plan['features'])/'features',packet['featureProvenance'])
    by={r['caseId']:store.load(r)[0] for r in packet['rows']}
    train=torch.cat([by[p['profileId']] for p in rosters['train']]);catalog_mean=catalog['vectors'].mean(0);context_mean=train.mean(0)
    # All three saved prediction rosters and hashes have been admitted before
    # annotation parsing here. These old annotations are explicitly DEVELOPMENT.
    train_labels=audit.read(Path(plan['contrasts'])/'labels.train.json')['labels']
    pt,vt,_=independent_targets(rosters['train'],train_labels,catalog)
    dev32=audit.read(plan['dev32Gold'])['labels'];dev40=audit.read(Path(plan['seedData'])/'labels.heldout.json')['labels']
    probe_public=audit.read(Path(plan['seedData'])/'history-only-probes.public.json');probe_gold=audit.read(Path(plan['seedData'])/'history-only-probes.labels.json')['labels']
    result={'status':'completed-independent-saved-replay','planSha256':PLAN_SHA,'actualOptimizerUpdatesProducerReceipt':3750,'independentlyReplayedRows':0,'arms':{},'scope':'All annotations DEVELOPMENT; no unseen holdout, no new optimizer/native/browser','overallPromotionPassed':False}
    for name,(centered,steps) in ARMS.items():
        directory=run_dir/name;checkpoint=torch.load(directory/'head.pt',map_location='cpu',weights_only=True);reload=torch.load(directory/'head.pt',map_location='cpu',weights_only=True)
        arm_frozen=audit.read(directory/'ARM_FROZEN.json');arm_report=report['arms'][name];state=checkpoint['state_dict'];params={k:v for k,v in state.items() if k not in ('context_mean','catalog_mean')}
        audit.require(checkpoint['version']==plan['version'] and checkpoint['config']==plan['config'] and checkpoint['arm']==arm_report['arm'] and checkpoint['planSha256']==PLAN_SHA,'Saved architecture/plan binding')
        audit.require(all(torch.equal(v,reload['state_dict'][k]) for k,v in state.items()),'Independent exact reload')
        cm=context_mean if centered else torch.zeros(4096);pm=catalog_mean if centered else torch.zeros(4096)
        audit.require(torch.equal(state['context_mean'],cm) and torch.equal(state['catalog_mean'],pm),'Checkpointed means must be TRAIN320/public16 only')
        audit.require(tensor_sha(cm)==arm_frozen['contextMeanSha256']==arm_report['contextMeanSha256'] and tensor_sha(pm)==arm_frozen['catalogMeanSha256']==arm_report['catalogMeanSha256'],'Exact derived center hashes')
        audit.verify_state(params,{'changedTensors':arm_report['changedParameters'],'ownershipCoefficient':arm_report['ownershipCoefficient']},plan['config'])
        with torch.random.fork_rng():
            torch.manual_seed(plan['config']['seed']);c=nn.Linear(4096,64);p=nn.Linear(4096,64,bias=False);cc=nn.Linear(4096,32);cv=nn.Linear(256,32,bias=False)
        initial={'scale':torch.tensor(2.),'ownership':torch.tensor(0.),'context.weight':c.weight.detach(),'context.bias':c.bias.detach(),'product.weight':p.weight.detach(),'color_context.weight':cc.weight.detach(),'color_context.bias':cc.bias.detach(),'color_candidate.weight':cv.weight.detach(),'context_mean':cm,'catalog_mean':pm}
        audit.require({k:tensor_sha(v) for k,v in initial.items()}==arm_frozen['initialStateSha256'],'Seeded optimizer initial-state proof differs')
        transformed_catalog={**catalog,'vectors':F.normalize(catalog['vectors']-pm,dim=-1,eps=1e-12) if centered else catalog['vectors']}
        replayed=0;fact_checks={};loss=None
        for split,profiles in rosters.items():
            x=torch.cat([by[p['profileId']] for p in profiles]);x=F.normalize(x-cm,dim=-1,eps=1e-12) if centered else x
            with torch.no_grad():ps,vs=audit.replay_scores(params,x,transformed_catalog,profiles)
            predictions=outputs[name][split];audit.require(ps.tolist()==[r['scores16'] for r in predictions],'Independent full score replay differs: '+name+'/'+split)
            for row,profile,vscore in zip(predictions,profiles,vs.tolist(),strict=True):
                request,policy,_=base.checked_context_request(profile['request'],catalog['products'])
                audit.require(row['order']==guarded_order(row['scores16'],request,policy,catalog['products'])[0],'Pinned guard order differs')
                for index,selected in zip(row['order'][:6],row['selected'],strict=True):
                    choices=[(j,v) for j,v in enumerate(catalog['variants']) if v['productIndex']==index and all(h['attribute']!='color' or h['value']==v['colorName'] for h in request['hardRequirements'])]
                    chosen=max(choices,key=lambda item:(vscore[item[0]],-item[0]))[1]
                    audit.require(selected['variantId']==chosen['variantId'],'Independent real-color argmax differs')
            fact_checks[split]=audit.factual_checks(predictions,profiles,catalog);replayed+=len(profiles)
            if split=='train':
                lp=-(pt*ps.log_softmax(-1)).sum(-1).mean();lv=-(vt*vs.log_softmax(-1)).sum(-1).mean();floor=(entropy(pt)+.5*entropy(vt)).mean()
                loss={'productCrossEntropy':float(lp),'variantCrossEntropy':float(lv),'jointLoss':float(lp+.5*lv),'jointEntropyFloor':float(floor),'jointExcess':float(lp+.5*lv-floor)}
                audit.require(abs(loss['jointLoss']-arm_report['finalDiagnostics']['loss'])<1e-6,'Independent final TRAIN loss differs')
        arm_result={'updatesProducerReceipt':steps,'replayedRows':replayed,'meanBuffersExact':True,'initialStateExact':True,'checkpointReloadExact':True,'TRAIN':grade_metrics(outputs[name]['train'],train_labels,catalog['products']),'TRAINLoss':loss,'DEV40':grade_metrics(outputs[name]['development-replay'],dev40,catalog['products']),'DEV32':audit.metrics(outputs[name]['independent-evaluation'],dev32,catalog['products'],rosters['independent-evaluation']),'DEVHistory10':audit.history_metrics(outputs[name]['history-diagnostic-nonfit'],probe_public,probe_gold,catalog['products']),'sourceFactChecks':fact_checks,'publicDemos':{}}
        for split,suffix in (('demo-nonfit',''),('demo-default-query-nonfit','-default-query')):
            demos=[]
            for row,profile in zip(outputs[name][split],rosters[split],strict=True):
                person=row['profileId'].removesuffix(suffix) if suffix else row['profileId'];anchor=DEMO_ANCHORS[person]
                selected=[v['productId'] for v in row['selected']];owned={e['productId'] for e in profile['request']['publicHistory'] if e['eventType']=='purchase' and 'productId' in e}
                demos.append({'person':person,'functionalAnchor':anchor,'anchorRank':selected.index(anchor)+1 if anchor in selected else None,'anchorFirst3':anchor in selected[:3],'selectedProducts':selected,'selectedColors':[v['selectedColor'] for v in row['selected']],'ownedInFirst3':sorted(set(selected[:3])&owned),'ownedInFirst6':sorted(set(selected)&owned),'publicColorPreferences':profile['request']['persona']['colorPreferences']})
            arm_result['publicDemos'][split]={'manualPosthocFunctionalReview':True,'notUnseenGold':True,'denominator':10,'functionalAnchorTop1':sum(r['anchorRank']==1 for r in demos),'functionalAnchorFirst3':sum(r['anchorFirst3'] for r in demos),'rowsWithOwnedFirst3':sum(bool(r['ownedInFirst3']) for r in demos),'rowsWithOwnedFirst6':sum(bool(r['ownedInFirst6']) for r in demos),'rows':demos}
        result['arms'][name]=arm_result;result['independentlyReplayedRows']+=replayed
    for path,expected in pins.items():audit.require(audit.sha(path)==expected,'Original artifact changed during review: '+path)
    result.update(allOriginalPinsUnchanged=True,filesSha256=pins,limitations=['Actual update counts and native927 integrity are producer receipts corroborated by pinned source/changed checkpoints; reviewer ran no optimizer/native.','Ten public demo anchors were manually interpreted after seeing outputs, not frozen benchmark labels.','Demos reuse TRAIN semantic motifs and do not establish broad novel-combination quality.','No generated copy, live query fusion, image decode, physical paint or serving latency verified.'])
    out.mkdir(parents=True);audit.write(out/'AUDIT.json',result)
    print({name:{'TRAINprimary':a['TRAIN']['grade3Top1'],'DEV32primary':a['DEV32']['primaryTop1'],'demoOriginalTop1':a['publicDemos']['demo-nonfit']['functionalAnchorTop1'],'demoBrowseFirst3':a['publicDemos']['demo-default-query-nonfit']['functionalAnchorFirst3']} for name,a in result['arms'].items()})
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('run','plan','out'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();run(args.run,args.plan,args.out)


if __name__=='__main__':main()
