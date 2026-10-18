"""Post-adjudication DEV error analysis. No training, regrading or new claims."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from . import audit_persona_choice_combined_v2 as audit

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
VERSION='persona-choice-known-DEV-error-analysis.v1'
FILES={'raw':{'learned':'learned-nonfit-predictions.json','base':'frozen-base-nonfit-predictions.json'},
       'fused':{'learned':'learned-combined-predictions.json','base':'frozen-base-lexical-decision-combined-predictions.json'}}


def top(row,ids):return ids[row['order'][0]] if row['order'] else None


def margin(row,target,ids):
    index=ids.index(target); audit.need(index in row['order'],'annotated primary must remain source-eligible')
    others=[row['scores16'][i] for i in row['order'] if i!=index]
    return row['scores16'][index]-max(others) if others else None


def case_detail(profile,label,row,ids,side,first_primary,second_primary):
    chosen=top(row,ids); owned=set(label['alreadyOwnedProductIds']); grades=label['relevanceGrades']
    if chosen==label['primaryProductId']:reason='primary-correct'
    elif side=='A' and chosen==second_primary:reason='less-urgent-secondary-before-first-gap'
    elif side=='B' and chosen==first_primary:reason='repeats-newly-purchased-usable-item'
    elif chosen in owned:reason='repeats-other-owned-item'
    elif grades.get(chosen,0)==1:reason='weaker-observed-interest'
    elif grades.get(chosen,0)>=2:reason='other-annotated-strong-positive'
    else:reason='no-annotated-positive-support'
    return {'profileId':profile['profileId'],'userGroupId':profile['userGroupId'],'side':side,
            'primaryProductId':label['primaryProductId'],'topProductId':chosen,'topGrade':grades.get(chosen,0),
            'correct':chosen==label['primaryProductId'],'ownedTop1':chosen in owned,
            'primaryEligibleRank':row['order'].index(ids.index(label['primaryProductId']))+1,
            'primaryMarginAgainstBestEligible':margin(row,label['primaryProductId'],ids),
            'failureType':reason,'query':profile['request']['query'],
            'publicBackground':profile['request']['persona']['background'],
            'publicBudget':profile['request'].get('itemBudget'),'publicHardRequirements':profile['request'].get('hardRequirements',[]),
            'publicHistory':[{'eventType':e['eventType'],'productId':e.get('productId'),
                              'itemName':e['itemName'],'selectedColor':e.get('selectedColor')} for e in profile['request']['publicHistory']]}


def summarize_model(details,pairs,rows,ids):
    cases={d['profileId']:d for d in details}; pair_results=[]
    for a,b in pairs:
        da,db=cases[a['profileId']],cases[b['profileId']]; ra,rb=rows[a['profileId']],rows[b['profileId']]
        acquired=da['primaryProductId'];i=ids.index(acquired)
        pair_results.append({'groupId':a['userGroupId'],'profileAId':a['profileId'],'profileBId':b['profileId'],
            'newlyPurchasedProductId':acquired,'bothCorrect':da['correct'] and db['correct'],
            'beforeCorrect':da['correct'],'afterCorrect':db['correct'],
            'topChoiceChanged':da['topProductId']!=db['topProductId'],
            'newlyOwnedRankBefore':ra['order'].index(i)+1,'newlyOwnedRankAfter':rb['order'].index(i)+1,
            'newlyOwnedPriorityDemoted':rb['order'].index(i)>ra['order'].index(i),
            'beforeTop':da['topProductId'],'afterTop':db['topProductId']})
    return {'fixedCases':32,'fixedPairs':16,'primaryCorrect':sum(d['correct'] for d in details),
        'beforePrimaryCorrect':sum(d['correct'] for d in details if d['side']=='A'),
        'afterPrimaryCorrect':sum(d['correct'] for d in details if d['side']=='B'),
        'bothPairSidesCorrect':sum(p['bothCorrect'] for p in pair_results),
        'pairOutcomeCounts':dict(Counter(('both-correct' if p['bothCorrect'] else 'before-only' if p['beforeCorrect'] else 'after-only' if p['afterCorrect'] else 'both-wrong') for p in pair_results)),
        'topChoiceFrequency':dict(Counter(d['topProductId'] for d in details)),
        'failureTypes':dict(Counter(d['failureType'] for d in details)),
        'ownedTop1':sum(d['ownedTop1'] for d in details),
        'newlyOwnedPriorityDemotions':sum(p['newlyOwnedPriorityDemoted'] for p in pair_results),'pairs':pair_results,'cases':details}


def compare(learned,base):
    la={r['profileId']:r for r in learned['cases']};ba={r['profileId']:r for r in base['cases']}
    loss=[];gain=[];both_right=[];both_wrong=[];choices_same=0
    for pid in la:
        a,b=la[pid],ba[pid];entry={'profileId':pid,'side':a['side'],'primaryProductId':a['primaryProductId'],
            'baseTop':b['topProductId'],'learnedTop':a['topProductId'],
            'basePrimaryRank':b['primaryEligibleRank'],'learnedPrimaryRank':a['primaryEligibleRank'],
            'basePrimaryMargin':b['primaryMarginAgainstBestEligible'], 'learnedPrimaryMargin':a['primaryMarginAgainstBestEligible'],
            'primaryMarginDelta':a['primaryMarginAgainstBestEligible']-b['primaryMarginAgainstBestEligible'],
            'learnedFailureType':a['failureType'],'publicBackground':a['publicBackground'],
            'publicHistory':a['publicHistory'],'publicBudget':a['publicBudget'],'publicHardRequirements':a['publicHardRequirements']}
        choices_same+=a['topProductId']==b['topProductId']
        (both_right if a['correct'] and b['correct'] else loss if b['correct'] else gain if a['correct'] else both_wrong).append(entry)
    return {'baseCorrectToLearnedWrong':len(loss),'baseWrongToLearnedCorrect':len(gain),
        'bothCorrect':len(both_right),'bothWrong':len(both_wrong),'sameTopChoice':choices_same,
        'changedTopChoice':32-choices_same,'sameCorrectnessOutcome':len(both_right)+len(both_wrong),
        'lossesByMostNegativePrimaryMarginChange':sorted(loss,key=lambda r:r['primaryMarginDelta']),
        'gains':gain,'bothWrongChoiceChanges':[r for r in both_wrong if r['baseTop']!=r['learnedTop']]}


def run(out):
    out=Path(out);audit.need(not out.exists(),'unique DEV analysis output required');out.mkdir(parents=True)
    packet=HERE/'persona_choice_independent_eval_v3';manifest=audit.read(packet/'MANIFEST.json')
    audit.need(audit.sha(packet/'MANIFEST.json')==audit.PACKET_SHA,'fixed packet identity')
    inputs={str(packet/'MANIFEST.json'):audit.PACKET_SHA}
    def packet_read(name):
        path=packet/name;audit.need(audit.sha(path)==manifest['files'][name],'DEV packet bytes changed')
        inputs[str(path)]=audit.sha(path);return audit.read(path)
    public=packet_read('main.public.json')['profiles'];replacement=packet_read('replacement.public.json')['profiles']
    labels={r['profileId']:r for r in packet_read('main.labels.private.json')['labels']}
    replacement_labels={r['profileId']:r for r in packet_read('replacement.labels.private.json')['labels']}
    ids=[p['productId'] for p in audit.read(HERE/'store_catalog_v1/external_mapping.json')['products']]
    audit.need(len(public)==32 and len(labels)==32 and len(replacement)==len(replacement_labels)==4,'fixed DEV denominator')
    pairs=list(zip(public[::2],public[1::2],strict=True));audit.need(len(pairs)==16,'fixed sixteen matched pairs')
    adjudication=ROOT/'data/ranker/pretrained/persona-choice-independent-adjudication-v2-attempt-1/ADJUDICATION.json'
    actual=audit.read(adjudication);audit.need(actual['status']=='completed-offline-adjudication' and actual['offlineCoreAndReplacementPass'] is False,'actual rejected adjudication required')
    inputs[str(adjudication)]=audit.sha(adjudication)
    directories={'raw':ROOT/'data/ranker/pretrained/persona-choice-train-v1-attempt-1',
                 'fused':ROOT/'data/ranker/pretrained/persona-choice-combined-v2-attempt-1'}
    raw_freeze=audit.read(directories['raw']/'ALL_PREDICTIONS_FROZEN.json')
    combined_freeze=audit.read(directories['fused']/'ALL_MODELS_PREDICTIONS_FROZEN.json');audit.pins(combined_freeze['sha256'])
    result={'version':VERSION,'scope':'Posthoc known DEVELOPMENT only; labels/thresholds/exclusions unchanged; descriptive associations, no causal or capacity claim',
            'gateSha256':audit.GATE_SHA,'nativeCalls':0,'optimizerUpdates':0,'candidateEdits':False,'sourceSha256':audit.sha(__file__)}
    for pipeline,directory in directories.items():
        all_rows={}; selected={}
        for model,filename in FILES[pipeline].items():
            path=directory/filename
            expected=(raw_freeze['learnedPredictionsSha256' if model=='learned' else 'frozenBasePredictionsSha256'] if pipeline=='raw' else
                      combined_freeze['models']['learned' if model=='learned' else 'frozen-base-lexical-decision']['sha256'])
            audit.need(audit.sha(path)==expected,'actual frozen model prediction changed');inputs[str(path)]=expected
            all_rows[model]=audit.read(path)
            selected[model]={r['profileId']:r for r in all_rows[model] if r['orderLabel']=='canonical'}
        details={model:[] for model in FILES[pipeline]}
        for a,b in pairs:
            audit.need(a['userGroupId']==b['userGroupId'],'group join changed')
            first=labels[a['profileId']]['primaryProductId'];second=labels[b['profileId']]['primaryProductId']
            for side,profile in [('A',a),('B',b)]:
                for model in details:details[model].append(case_detail(profile,labels[profile['profileId']],selected[model][profile['profileId']],ids,side,first,second))
        models={model:summarize_model(values,pairs,selected[model],ids) for model,values in details.items()}
        reference=actual['raw' if pipeline=='raw' else 'combined']['main']
        audit.need(models['learned']['primaryCorrect']==reference['learned']['primaryTop1'] and
                   models['base']['primaryCorrect']==reference['frozen-base-lexical-decision']['primaryTop1'],'analysis cannot change fixed metrics')
        extra={}
        for model in details:
            cases=[]
            for profile in replacement:
                label=replacement_labels[profile['profileId']];row=selected[model][profile['profileId']];chosen=top(row,ids)
                cases.append({'profileId':profile['profileId'],'primaryProductId':label['primaryProductId'],'topProductId':chosen,
                    'correct':chosen==label['primaryProductId'],'publicBackground':profile['request']['persona']['background'],
                    'publicReplacementRequested':label['requestedReplacementProductIds']})
            extra[model]={'fixedDenominator':4,'primaryCorrect':sum(r['correct'] for r in cases),'cases':cases}
        checks={model:audit.facts([r for r in all_rows[model] if r['orderLabel']=='canonical' and r['profileId'] in labels],public,
                [{**p,'productId':ids[i]} for i,p in enumerate(audit.read(HERE/'store_catalog_v1/model_inputs.json')['products'])]) for model in details}
        result[pipeline]={'models':models,'learnedVsBase':compare(models['learned'],models['base']),'replacement':extra,
            'sourceEligibilityViolations':{model:v['violationCount'] for model,v in checks.items()},
            'hardColorCases':{model:{'cases':sum(bool(p['request']['hardRequirements']) for p in public),
                'primaryCorrect':sum(d['correct'] for d in models[model]['cases'] if d['publicHardRequirements'])} for model in details}}
        history_public=audit.read(HERE/'synthetic_history_v1/history-only-probes.public.json')
        history_labels=audit.read(HERE/'synthetic_history_v1/history-only-probes.labels.json')['labels']
        result[pipeline]['knownHistory']={model:audit.history_metrics(all_rows[model],history_public,history_labels,
                                            [{'productId':pid} for pid in ids]) for model in details}
        if pipeline=='fused':
            probs=[selected['learned'][p['profileId']]['pSpecific'] for p in public]
            result['queryGateDescriptiveOnly']={'fixedCases':32,'minimum':min(probs),'median':statistics.median(probs),
                'maximum':max(probs),'abovePointFive':sum(p>.5 for p in probs),
                'scope':'continuous probabilities on authored generic-query contexts; no new routing threshold or accuracy gate'}
    prior=directories['raw']/'BASELINE_DEFINITIONS_FROZEN.json';inputs[str(prior)]=audit.sha(prior)
    result['TRAINPriorDescriptiveOnly']=audit.read(prior)['priorScoresByProductId']
    result['inputSha256']=inputs;audit.pins(inputs)
    audit.write(out/'ANALYSIS.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    value=run(args.out);print(json.dumps({'version':VERSION,'out':str(args.out),
        'raw':{k:v for k,v in value['raw']['learnedVsBase'].items() if isinstance(v,int)},
        'fused':{k:v for k,v in value['fused']['learnedVsBase'].items() if isinstance(v,int)}}))
