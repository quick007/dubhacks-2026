"""CPU query/v5/fusion replay, no optimizer, MLX, serving or quality labels.

This supplements the frozen quality adjudicator without changing its bytes.
The certificate covers saved query math, never specific-query behavioral or
browser acceptance. All inputs must already be durably frozen by their owners.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from . import audit_persona_choice_combined_v1 as evidence
from . import store_persona_combined_query_v1 as public_query
from . import store_persona_context_v1 as context
from .store_demo_engine import guarded_order

HERE=Path(__file__).resolve().parent
VERSION='persona-combined-query-replay-audit.v1'
QUERY_PLAN_SHA='8db50d9063df26f1662c705bee79e7fe73ddf42c58b30d13c52ec5897591961a'
GATE_PLAN_SHA='25fdb0c976dc3b6e01e826e7f30ea63fc65e3fbb85f56f24f4c83e8538703232'
MECHANICAL_AUDIT_SOURCE_SHA='aeb83de941339e852b70b12433aaec91629f091bf8bc6313a367dc24dfb356d0'


def finite(value, shape):
    evidence.need(isinstance(value,torch.Tensor) and value.device.type=='cpu' and
                  tuple(value.shape)==shape and value.dtype==torch.float32 and
                  bool(torch.isfinite(value).all()),'finite CPU float32 tensor shape required')
    return value


def query_probability(state, query):
    evidence.need(set(state)=={'linear.weight','linear.bias'},'exact query linear parameter roster')
    weight=finite(state['linear.weight'],(1,4096)); bias=finite(state['linear.bias'],(1,))
    finite(query,(1,4096))
    evidence.need(torch.allclose(query.norm(dim=-1),torch.ones(1),atol=.002,rtol=0),'query must be original unit-L2 base feature')
    with torch.inference_mode():
        return float(F.linear(query,weight,bias).squeeze(-1).sigmoid().item())


def v5_probability(state, role, values):
    evidence.need(role in ('catalog','query'),'explicit catalog/query role required')
    finite(state[role+'_mean'],(4096,)); finite(state[role+'_weight'],(16,4096)); finite(state[role+'_bias'],(16,))
    evidence.need(values.ndim==2 and values.shape[1]==4096 and values.dtype==torch.float32 and torch.isfinite(values).all(),'finite raw4096 v5 inputs')
    with torch.inference_mode():
        centered=values-state[role+'_mean']
        centered=centered/centered.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        return (centered@state[role+'_weight'].T+state[role+'_bias']).softmax(-1)


def v5_scores(state, query, catalog):
    evidence.need(set(state)=={role+'_'+field for role in ('query','catalog') for field in ('mean','weight','bias')},'exact v5 two-projection parameter roster')
    finite(catalog,(16,4096)); finite(query,(1,4096))
    evidence.need(torch.equal(state['catalog_mean'],catalog.double().mean(0).float()),'v5 TRAIN catalog mean differs from original16')
    with torch.inference_mode():
        return (v5_probability(state,'catalog',catalog)@v5_probability(state,'query',query)[0]).tolist()


def mechanical_receipt(path, gate_sha):
    receipt=evidence.read(path)
    evidence.need(receipt['version']=='query-intent-independent-TRAIN-audit.v3' and
        receipt['status']=='passed-independent-TRAIN-replay' and receipt['privateQualityLabelsRead'] is False and
        receipt['sourceManifestSha256']==GATE_PLAN_SHA and receipt['headSha256']==gate_sha and
        receipt['auditSourceSha256']==MECHANICAL_AUDIT_SOURCE_SHA and
        receipt['replay']['observedReplayOptimizerSteps']==250 and
        receipt['replay']['everyFinalParameterBitwiseMatches'] is True and
        receipt['twoReloadsExactLogitParity'] is True and
        receipt['reloadSavedParity']['rows']==1272 and receipt['reloadSavedParity']['exactSavedProbabilityParity'] is True,
        'actual fixed TRAIN mechanical replay receipt incomplete')
    evidence.pins(receipt['inputSha256'])
    return receipt


def query_inputs(run, expected_sha):
    run=Path(run).resolve(); plan_path=run/'plan.json'
    evidence.need(evidence.sha(plan_path)==expected_sha==QUERY_PLAN_SHA,'exact reviewed292 query plan required')
    plan=evidence.read(plan_path); report=evidence.read(run/'report.json')
    evidence.pins(plan['sha256'])
    evidence.need(plan['goldParsed'] is False and plan['targetOrTeacherInputs'] is False and
        plan['counts']=={'combinedRows':344,'historicalQueryOnlyRows':320,'allRows':664,'uniqueBaseQueryForwardsRequired':292},'fixed344+320/292 query roster required')
    evidence.need(report['version']==plan['version'] and report['status']=='completed-query-features' and report['goldParsed'] is False and
        report['sourceManifestSha256']==expected_sha and report['adapterInstalled'] is False and report['optimizerUpdates']==0 and
        all(report[k]==292 for k in ('attemptedNativeForwards','physicalNativeForwards','savedVectors','expectedNativeForwards')) and
        report['frozen927Unchanged'] is True and report['sourcePinsUnchanged'] is True and report['shutdownCompleted'] is True and
        len(report['beforeParameterSha256'])==927 and report['beforeParameterSha256']==report['afterParameterSha256'],'complete292 base-only forwards/927/shutdown proof required')
    evidence.need(len(report['roster'])==292 and all(r['status']=='completed' for r in report['roster']) and
        [r['textSha256'] for r in report['roster']]==[r['textSha256'] for r in plan['uniqueQueries']],'actual completed292 query identity roster')
    bundle_path=run/'query-features.pt'; evidence.need(evidence.sha(bundle_path)==report['queryFeaturesSha256'],'query feature bundle changed')
    bundle=torch.load(bundle_path,map_location='cpu',weights_only=True)
    evidence.need(set(bundle)=={'version','sourceManifestSha256','features'} and bundle['version']==plan['version'] and
        bundle['sourceManifestSha256']==expected_sha and len(plan['uniqueQueries'])==292 and
        set(bundle['features'])=={r['textSha256'] for r in plan['uniqueQueries']},'exact target-free query feature bundle')
    from tokenizers import Tokenizer
    tokenizer_path=Path(plan['modelFiles']['modelPath'])/'tokenizer.json'
    evidence.need(evidence.sha(tokenizer_path)==plan['modelFiles']['smallFileSha256']['tokenizer.json'],'query tokenizer pin')
    tokenizer=Tokenizer.from_file(str(tokenizer_path)); query_by={r['textSha256']:r for r in plan['uniqueQueries']}
    for key,row in query_by.items():
        evidence.need(public_query.digest_text(row['text'])==key and tokenizer.encode(row['text'],add_special_tokens=False).ids==row['tokenIds'] and
                      public_query.contract.token_hash(row['tokenIds'])==row['tokenIdsSha256'] and len(row['tokenIds'])==row['tokens'],'exact whole query UTF8/token identity')
        finite(bundle['features'][key],(1,4096))
    products=public_query.public_products()
    for row in plan['rows']:
        text=public_query.runtime_text(row['publicRequest'],products)
        evidence.need(text==query_by[row['queryTextSha256']]['text'] and
            public_query.digest_text(public_query.canonical(row['publicRequest']))==row['publicRequestSha256'],'query context gained history/labels or public request changed')
    return plan,bundle,{str(run/name):evidence.sha(run/name) for name in ('plan.json','report.json','query-features.pt')}


def check_combined_rows(rows, public_rows, expected_query, products, gate_sha):
    evidence.need(len(rows)==len(public_rows)==344,'fixed344 combined denominator')
    by={r['caseId']:r for r in public_rows}
    evidence.need(len({r['caseId'] for r in rows})==344 and {r['caseId'] for r in rows}==set(by),'combined exactcase join')
    for row in rows:
        public=by[row['caseId']]; p,v5=expected_query[public['queryTextSha256']]
        evidence.need(all(row.get(k)==public.get(k) for k in ('profileId','userGroupId','caseId','split','orderLabel','sourceProfileId','controlName','publicRequestSha256','queryTextSha256')),'combined public identity changed')
        evidence.need(row['pSpecific']==p and row['v5Scores16']==v5,'independent query probability/v5 replay differs')
        evidence.need(row['fusionPolicy']['gateCheckpointSha256']==gate_sha,'wrong query checkpoint')
        # Preserve the actual float32 CPU arithmetic of the runtime, while the
        # original quality auditor also checks its independent scalar formula.
        raw=torch.tensor(evidence.vector(row['rawSourceScores16']),dtype=torch.float32)
        v5_tensor=torch.tensor(v5,dtype=torch.float32); soft=raw.softmax(-1)
        scores=((1-p)*soft+p*(v5_tensor+.05*soft)).tolist()
        evidence.need(scores==row['scores16'],'combined float32 blend differs')
        request,policy,_=context.checked_context_request(public['publicRequest'],products)
        order,guards=guarded_order(scores,request,policy,products)
        evidence.need(row['order']==order and row['guardChecks']==guards and
            row['rawLearnedOrder']==sorted(range(16),key=lambda i:(-scores[i],i)) and
            row['rankedProductIds']==[products[i]['productId'] for i in order],'actual guard/exact-name/order binding differs')
        evidence.need(row['copyGenerated'] is False and row['variantsOrImagesScored'] is False,'offline prediction claims unsupported serving evidence')


def audit(combined_run, query_run, gate_head, v5_export, mechanical_query_audit, out):
    combined_run=Path(combined_run).resolve(); query_run=Path(query_run).resolve(); gate_head=Path(gate_head).resolve()
    v5_export=Path(v5_export).resolve(); out=Path(out)
    evidence.need(not out.exists(),'new supplemental audit output required'); out.mkdir(parents=True)
    result={'version':VERSION,'status':'starting','privateQualityLabelsRead':False,'nativeCalls':0,'optimizerUpdates':0,
            'overallPromotionPass':False,'servingPending':True,'specific320BehavioralRegressionComputed':False}
    start=time.perf_counter(); old_threads=torch.get_num_threads()
    try:
        frozen=evidence.read(HERE/'PERSONA_COMBINED_QUERY_REPLAY_AUDIT_FROZEN_V1.json'); evidence.pins(frozen['sha256'])
        combined_freeze_path=combined_run/'ALL_MODELS_PREDICTIONS_FROZEN.json'; combined=evidence.read(combined_freeze_path)
        evidence.need(combined['goldParsed'] is False and combined['independentPacketManifestSha256']==evidence.PACKET_SHA and
            set(combined['models'])==evidence.MODELS and all(combined['stagesCompleted'][k] is True for k in ('adapter','queryGate','allControls')),'global model admission incomplete')
        evidence.pins(combined['sha256']); gate_sha=evidence.sha(gate_head)
        evidence.need(combined['gateCheckpointSha256']==gate_sha and combined['sha256'].get(str(gate_head))==gate_sha,'query checkpoint not globalfreeze pinned')
        mechanical_receipt(mechanical_query_audit,gate_sha)
        plan,bundle,input_pins=query_inputs(query_run,QUERY_PLAN_SHA)
        evidence.need(all(combined['sha256'].get(p)==s for p,s in input_pins.items()),'combined uses different query artifact')
        intent=torch.load(gate_head,map_location='cpu',weights_only=True)
        evidence.need(intent['version']=='query-intent-linear.v3' and intent['metadata']['sourceManifestSha256']==GATE_PLAN_SHA and
                      intent['metadata']['trainingOnly'] is True and intent['metadata']['qualityEvaluationLabelsRead'] is False,'query intent provenance')
        export=evidence.read(v5_export)
        evidence.need(combined['sha256'].get(str(v5_export))==evidence.sha(v5_export) and export['teacherLookupAtServing'] is False,'v5 export not global pinned')
        head_path=v5_export.parent/export['head']['file']; head_sha=evidence.sha(head_path)
        evidence.need(head_sha==export['head']['sha256'] and combined['sha256'].get(str(head_path))==head_sha,'v5 checkpoint pin')
        head=torch.load(head_path,map_location='cpu',weights_only=True)
        evidence.need(head['metadata']==export['head']['metadata'],'v5 serialized metadata')
        catalog=context.load_catalog_features(export['sourceBaseRun']); evidence.pins(catalog['pins'])
        evidence.need(export['featureProvenance']==catalog['provenance'],'v5/export original catalog provenance differs')
        evidence.need(all(combined['sha256'].get(p)==s for p,s in catalog['pins'].items()),'original catalog source not global pinned')
        torch.set_num_threads(1)
        expected={key:(query_probability(intent['state'],value),v5_scores(head['state'],value,catalog['vectors'])) for key,value in bundle['features'].items()}
        rows=[r for r in plan['rows'] if r['scope']=='combined-all344']; evidence.need(len(rows)==344,'all344 nonfit required')
        model_pins={}; raw_files={'learned':'learned-nonfit-predictions.json','frozen-base-lexical-decision':'frozen-base-nonfit-predictions.json',
                                  'TRAIN-popularity':'training-popularity-nonfit-predictions.json','latest-owned-repeat':'latest-owned-repeat-nonfit-predictions.json'}
        for model,entry in combined['models'].items():
            path=evidence.local_file(combined_run,entry['file'],entry['sha256']); model_rows=evidence.read(path)
            check_combined_rows(model_rows,rows,expected,catalog['products'],gate_sha); model_pins[str(path)]=entry['sha256']
            raw_paths=[Path(p) for p in combined['sha256'] if Path(p).name==raw_files[model]]
            evidence.need(len(raw_paths)==1,'exact corresponding raw prediction file required')
            raw={r['caseId']:r for r in evidence.read(raw_paths[0])}
            evidence.need(len(raw)==344 and set(raw)=={r['caseId'] for r in model_rows},'raw344 identity roster')
            for row in model_rows:evidence.need(row['rawSourceScores16']==raw[row['caseId']]['scores16'],'combined source vector differs from frozen raw model')
        write_values=[{'queryTextSha256':key,'pSpecific':p,'v5Scores16':v5} for key,(p,v5) in sorted(expected.items())]
        evidence.write(out/'INDEPENDENT_QUERY_VALUES.json',write_values)
        input_pins.update({str(gate_head):gate_sha,str(v5_export):evidence.sha(v5_export),str(head_path):head_sha,
            str(Path(mechanical_query_audit).resolve()):evidence.sha(mechanical_query_audit),str(combined_freeze_path):evidence.sha(combined_freeze_path)})
        input_pins.update(catalog['pins']); input_pins.update(model_pins); evidence.pins(input_pins); evidence.pins(combined['sha256'])
        result.update(status='passed',gateCheckpointSha256=gate_sha,v5HeadSha256=head_sha,
            adapterCheckpointSha256=combined['adapterCheckpointSha256'],queryBundleSha256=input_pins[str(query_run/'query-features.pt')],
            combinedFreezeSha256=evidence.sha(combined_freeze_path),mechanicalQueryAuditSha256=evidence.sha(mechanical_query_audit),
            rows=344,modelRowsChecked=1376,uniqueQueryVectorsReplayed=292,exactQueryProbabilityAndV5Parity=True,
            actualFloat32BlendAndGuardsExact=True,inputSha256=input_pins,sourceSha256=frozen['sha256'],
            independentQueryValuesSha256=evidence.sha(out/'INDEPENDENT_QUERY_VALUES.json'))
    except Exception as error:
        result.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
    finally:
        torch.set_num_threads(old_threads); result['elapsedSeconds']=time.perf_counter()-start
        evidence.write(out/'audit.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    for name in ('combined-run','query-run','gate-head','v5-export','mechanical-query-audit','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv); result=audit(args.combined_run,args.query_run,args.gate_head,args.v5_export,args.mechanical_query_audit,args.out)
    print(json.dumps({k:result[k] for k in ('version','status','overallPromotionPass','servingPending')}))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
