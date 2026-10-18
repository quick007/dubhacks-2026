"""Bounded known-store multi-need adaptation; CPU prepare, root-only native run.

One frozen Qwen extraction per input: 16 catalog +64 TRAIN +24 fresh evaluation.
No copy, images, UI rendering, deployment or promotion. Evaluation labels are
parsed only after all learned/control predictions have been serialized/frozen.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F

from . import store_function_data_v4 as data_api
from .catalog_fusion.inputs import digest, tensor_digest
from .catalog_fusion.preflight import DEFAULT_MODEL, REVISION, file_digest
from .catalog_fusion.runtime import (FeatureStore, NativeEncoder, assert_model_files,
                                     verify_model, write_json)
from .store_catalog_v1 import canonical
from .store_model_bridge import prepare_current_catalog

ROOT = Path(__file__).resolve().parent
VERSION = 'actual-store-independent-function-adaptation.v4'
CONFIG = {'seed': 728, 'steps': 80, 'priorSteps': 80, 'learningRate': .03,
    'weightDecay': .001, 'epsilon': 1e-8, 'threads': 1, 'memoryGiB': 6,
    'maxTokens': 512, 'nativeForwards': 104, 'trainFeatures': 80,
    'evaluationCases': 24, 'maxLifecycleSeconds': 180,
    'dimensions': 4096, 'classes': 16, 'parameters': 131104,
    'semanticThreshold': .5, 'maxResults': 6,
    'permutedQueryOffsetTrain': 17, 'permutedQueryOffsetEvaluation': 1,
    'permutationClaim': 'fixed counterfactual shuffle, not guaranteed disjoint semantic targets; evaluation labels not consulted',
    'objective': 'equal role mean: flat TRAIN desired-function BCE + catalog identity CE',
    'priorObjective': 'flat TRAIN desired-function BCE for query-independent 16 logits',
    'unrequestedPolicy': 'zero is case-specific unrequested, not universally irrelevant; no negative reweighting',
    'exactNamePolicy': 'whole current query is original name, or find/show [me] [the item named] original name; optional please/quotes/punctuation; no incidental mention'}
ARMS = ('learned', 'prior', 'random', 'permuted-query')
SOURCE_FILES = ('store_function_adaptation_v4.py', 'test_store_function_adaptation_v4.py',
    'store_function_data_v4.py', 'test_store_function_data_v4.py', 'store_catalog_v1.py',
    'store_model_bridge.py', 'catalog_fusion/runtime.py', 'catalog_fusion/inputs.py',
    'catalog_fusion/preflight.py', 'catalog_fusion/generalization_v2.py', 'backbone.py')


def source_hashes():
    return {name: file_digest(ROOT/name) for name in SOURCE_FILES}


def read_public(data=data_api.DATA):
    """Hash every frozen file but never parse the evaluation label packet."""
    data = Path(data);receipt = json.loads((data/'FREEZE.json').read_text())
    if receipt['version'] != data_api.VERSION:
        raise ValueError('Frozen v4 dataset required')
    if receipt['sourceHashes'] != {n:file_digest(ROOT/n) for n in data_api.SOURCE_FILES}:
        raise ValueError('Frozen v4 data source changed')
    for name, expected in receipt['files'].items():
        if file_digest(data/name) != expected:
            raise ValueError('Frozen data artifact changed: ' + name)
    products = json.loads((data/'catalog.json').read_text())
    cases = json.loads((data/'public-cases.json').read_text())
    if (len(products) != 16 or len(cases) != 88
            or sum(c['split'] == 'train' for c in cases) != 64
            or sum(c['split'] == 'evaluation' for c in cases) != 24
            or len({c['caseId'] for c in cases}) != 88):
        raise ValueError('Fixed 16 catalog/64 TRAIN/24 evaluation roster required')
    for c in cases:
        data_api.validate_request(c['request'])
    if len({data_api.request_text(c['request']) for c in cases}) != 88:
        raise ValueError('Disjoint whole public requests required')
    bridge = prepare_current_catalog()
    if bridge['modelInputSha256'] != receipt['bridge']['modelInputSha256']:
        raise ValueError('Current public catalog changed')
    return receipt, products, cases


def prepare(data, model, prior):
    from tokenizers import Tokenizer
    frozen, products, cases = read_public(data)
    model = Path(model).resolve();old = json.loads((Path(prior)/'preflight.json').read_text())
    if model.name != REVISION or str(model) != old['modelPath']:
        raise ValueError('Pinned local model path/revision required')
    for name, key in (('tokenizer.json', 'tokenizerSha256'), ('config.json', 'configSha256'),
                      ('chat_template.jinja', 'chatTemplateSha256')):
        if file_digest(model/name) != old[key]:
            raise ValueError('Pinned tokenizer/config/template changed')
    provenance_preflight = {k:old[k] for k in ('quantization', 'tokenizerSha256',
        'configSha256', 'chatTemplateSha256')}
    provenance_preflight['modelPath'] = str(model)
    tokenizer = Tokenizer.from_file(str(model/'tokenizer.json'));rows = []
    for i,p in enumerate(products):
        rows.append({'role':'catalog', 'member':[VERSION,'catalog',i], 'caseId':str(i),
                     'split':'train', 'text':data_api.product_text(p)})
    for split in ('train', 'evaluation'):
        for c in [x for x in cases if x['split'] == split]:
            rows.append({'role':'request', 'member':[VERSION,c['caseId']],
                'caseId':c['caseId'], 'split':split, 'text':data_api.request_text(c['request'])})
    for row in rows:
        ids = tokenizer.encode(row['text'], add_special_tokens=False).ids
        if not 0 < len(ids) <= CONFIG['maxTokens']:
            raise ValueError('Whole input exceeds frozen token cap; no truncation')
        row.update(textSha256=digest(row['text']), tokenIds=ids,
                   tokenSha256=digest(canonical(ids)), tokens=len(ids), batchShape=[1,len(ids)])
    if len(rows) != 104 or [r['split'] for r in rows] != ['train']*80 + ['evaluation']*24:
        raise ValueError('Exactly 80 TRAIN then 24 evaluation native inputs required')
    packet = {'version':VERSION, 'sourceHashes':source_hashes(),
        'dataFreezeSha256':file_digest(Path(data)/'FREEZE.json'), 'model':provenance_preflight,
        'priorPacketSha256':file_digest(Path(prior)/'preflight.json'), 'config':deepcopy(CONFIG),
        'records':rows, 'tokenRange':[min(r['tokens'] for r in rows),max(r['tokens'] for r in rows)],
        'evaluationLabelsParsed':False, 'thresholdChosenBeforeEvaluation':.5}
    return packet, frozen, products, cases, rows


class IndependentFunctionHead(nn.Module):
    def __init__(self, means):
        super().__init__();generator = torch.Generator().manual_seed(CONFIG['seed'])
        for role in ('query', 'catalog'):
            self.register_buffer(role+'_mean', means[role].clone())
            setattr(self,role+'_weight',nn.Parameter(torch.randn(16,4096,generator=generator)*.05))
            setattr(self,role+'_bias',nn.Parameter(torch.zeros(16)))

    def logits(self, role, values):
        centered = values - getattr(self,role+'_mean')
        normalized = centered/centered.norm(dim=-1,keepdim=True).clamp_min(CONFIG['epsilon'])
        return normalized@getattr(self,role+'_weight').T + getattr(self,role+'_bias')

    def query_probabilities(self, values):
        return self.logits('query',values).sigmoid()

    def catalog_probabilities(self, values):
        return self.logits('catalog',values).softmax(-1)


def fit_inputs(store, rows, cases, labels):
    selected = [c for c in cases if c['split']=='train']
    ids = [c['caseId'] for c in selected];targets = {r['caseId']:r for r in labels}
    if len(labels) != 64 or len(targets) != 64 or set(targets) != set(ids):
        raise ValueError('Only the exact 64 TRAIN labels may fit the head')
    lookup = {(r['role'],r['caseId']):r for r in rows}
    q = torch.cat([store.load(lookup['request',rid])[0] for rid in ids])
    c = torch.cat([store.load(lookup['catalog',str(i)])[0] for i in range(16)])
    desired = torch.zeros((64,16),dtype=torch.float32)
    for i,rid in enumerate(ids):
        members = targets[rid]['desiredProductIndices']
        if (not members or members != sorted(set(members))
                or any(type(j) is not int or not 0<=j<16 for j in members)):
            raise ValueError('Explicit semantic desired-function TRAIN labels required')
        desired[i,members] = 1
    means = {'query':q.double().mean(0).float(), 'catalog':c.double().mean(0).float()}
    metadata = {'queryMembers':ids, 'catalogMembers':list(range(16)),
        'meanSha256':{k:tensor_digest(v) for k,v in means.items()},
        'semanticDesiredTargetsSha256':tensor_digest(desired),
        'targetSource':'TRAIN desiredProductIndices before deterministic eligibility masking',
        'config':deepcopy(CONFIG)}
    return q,c,desired,means,metadata


def save_head(path, model, metadata):
    with Path(path).open('xb') as handle:
        torch.save({'version':VERSION,'metadata':metadata,'state':model.state_dict()},handle)
    return file_digest(path)


def load_head(path, expected_sha, metadata):
    if file_digest(path) != expected_sha:
        raise ValueError('Head checksum changed')
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if saved['version'] != VERSION or saved['metadata'] != metadata:
        raise ValueError('Head metadata changed')
    means = {role:saved['state'][role+'_mean'] for role in ('query','catalog')}
    if any(tensor_digest(v) != metadata['meanSha256'][role] for role,v in means.items()):
        raise ValueError('TRAIN mean changed')
    model = IndependentFunctionHead(means);model.load_state_dict(saved['state'],strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('Nonfinite checkpoint')
    return model.eval()


def exact_query_target(query, products):
    """Recognize entire identity requests; negation/incidental mentions never route."""
    if type(query) is not str:
        raise ValueError('Public query string required')
    matches = []
    for i,p in enumerate(products):
        name = re.escape(p['name'])
        pattern = (r'\s*(?:(?:please\s+)?(?:find|show)(?:\s+me)?'
                   r'(?:\s+the\s+item\s+named)?\s+)?[\"\']?'+name+r'[\"\']?[.!?]?\s*')
        if re.fullmatch(pattern,query,flags=re.IGNORECASE):
            matches.append(i)
    return matches[0] if len(matches)==1 else None


def decide(scores, request, products, *, apply_exact_policy=False):
    data_api.validate_request(request)
    if (len(products)!=16 or len(scores)!=16 or any(type(x) not in (float,int)
            or not math.isfinite(x) or not 0<=x<=1 for x in scores)):
        raise ValueError('Full 16 finite semantic compatibility scores in [0,1] required')
    raw = sorted(range(16),key=lambda i:(-scores[i],i))
    allowed = [i for i,p in enumerate(products) if data_api.eligible(request,p)]
    eligible_rank = [i for i in raw if i in allowed]
    semantic = [i for i in eligible_rank if scores[i]>=CONFIG['semanticThreshold']]
    exact = exact_query_target(request['query'],products)
    selected = semantic[:6];route = apply_exact_policy and exact is not None
    if route:
        selected = [exact] if exact in allowed else []
    return {'selectedProductIndices':selected, 'rawRank':raw,
        'unthresholdedTopSix':raw[:6], 'unthresholdedEligibleTopSix':eligible_rank[:6],
        'eligibleProductIndices':allowed, 'eligibleSemanticProductIndices':semantic,
        'semanticSelectedBeforePolicy':semantic[:6], 'semanticThreshold':.5,
        'shortfallToSix':6-len(selected), 'hardEligibleCount':len(allowed),
        'exactNameTarget':exact, 'exactNameRouteTaken':bool(route),
        'exactNameGuardRejected':bool(route and exact not in allowed),
        'policy':'exact-name-then-hard-guards' if apply_exact_policy else 'semantic-strict',
        'tieBreak':'external source index; never an encoder input',
        'budgetScope':'inclusive item price; delivery excluded',
        'unknownSizeAndStock':'required unsupported attributes reject every product',
        'padding':False}


def prediction_payload(scores, request, products):
    return {'scores':scores, 'semantic':decide(scores,request,products),
            'policy':decide(scores,request,products,apply_exact_policy=True)}


def predict(models, prior_scores, queries, catalog, cases, products):
    """No labels accepted; controls use frozen states and public requests only."""
    ids = [c['caseId'] for c in cases];result = {}
    splits={c['split'] for c in cases}
    if len(splits)!=1 or next(iter(splits)) not in ('train','evaluation'):
        raise ValueError('One explicit TRAIN or evaluation split per prediction call required')
    offset=CONFIG['permutedQueryOffsetTrain' if next(iter(splits))=='train' else 'permutedQueryOffsetEvaluation']
    with torch.no_grad():
        for arm in ARMS:
            model = models['random' if arm=='random' else 'learned']
            catp = model.catalog_probabilities(catalog);rows = {}
            for i,case in enumerate(cases):
                source = (i+offset)%len(cases) if arm=='permuted-query' else i
                qp = model.query_probabilities(queries[source:source+1])[0]
                scores = prior_scores.tolist() if arm=='prior' else (catp@qp).tolist()
                rows[case['caseId']] = {**prediction_payload(scores,case['request'],products),
                    'querySource':None if arm=='prior' else ids[source],
                    'querySigmoid':None if arm=='prior' else qp.tolist(),
                    'catalogSoftmax':None if arm=='prior' else catp.tolist()}
            result[arm] = {'fixedCases':len(cases), 'cases':rows}
    return result


def evaluate(predictions, cases, labels, pairs):
    """Postfreeze evaluator; semantic and deterministic-policy quality stay separate."""
    result = {}
    for arm, rows in predictions.items():
        arm_result = {}
        for mode in ('semantic','policy'):
            outputs = {rid:r[mode]['selectedProductIndices'] for rid,r in rows['cases'].items()}
            arm_result[mode] = data_api.evaluate_rankings(outputs,cases,labels,pairs)
        unthresholded = {rid:r['semantic']['unthresholdedEligibleTopSix'] for rid,r in rows['cases'].items()}
        arm_result['unthresholdedGuardedRanking'] = data_api.evaluate_rankings(unthresholded,cases,labels,pairs)
        arm_result['exactNameRouteCount'] = sum(r['policy']['exactNameRouteTaken'] for r in rows['cases'].values())
        arm_result['exactNameGuardRejectedCount'] = sum(r['policy']['exactNameGuardRejected'] for r in rows['cases'].values())
        arm_result['policyChangedOutputCount'] = sum(r['semantic']['selectedProductIndices'] != r['policy']['selectedProductIndices'] for r in rows['cases'].values())
        result[arm] = arm_result
    return result


def check_elapsed(started):
    if time.perf_counter()-started > CONFIG['maxLifecycleSeconds']:
        raise TimeoutError('Fixed cooperative v4 lifecycle deadline exceeded')


def failure_receipt(out, error, state, started):
    value = {'status':'failed', 'phase':state['phase'], 'errorType':type(error).__name__,
        'error':str(error), 'completedNativeForwards':state['forwards'],
        'serializedFeatures':state['features'], 'completedEvaluationEndpoints':state['endpoints'],
        'completedLearnedUpdates':state['learnedSteps'], 'completedPriorUpdates':state['priorSteps'],
        'fixedNativeForwards':104, 'fixedTrainFeatures':80, 'fixedTrainingQueries':64,
        'fixedEvaluationCases':24, 'fixedLearnedUpdates':80, 'fixedPriorUpdates':80,
        'elapsedSeconds':time.perf_counter()-started, 'promotable':False,
        'timeoutKind':'cooperative checks; no hard interruption of a hung native call'}
    if not (Path(out)/'FAILED.json').exists():
        write_json(Path(out)/'FAILED.json',value)


def run(out, data, packet_path, model, prior, owner):
    """Own one new output directory; preserve every artifact on any failure."""
    out = Path(out)
    if out.exists():
        raise FileExistsError('Preserve previous v4 output')
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    state = {'phase':'owner-lock','forwards':0,'features':0,'endpoints':0,'learnedSteps':0,'priorSteps':0}
    try:
        from .catalog_fusion.generalization_v2 import native_owner_lock
        with native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'),owner):
            return _run(out,data,packet_path,model,prior,state,started)
    except BaseException as error:
        try:failure_receipt(out,error,state,started)
        except BaseException as receipt_error:
            error.add_note('Failure receipt could not be saved: '+str(receipt_error))
        raise


def _run(out, data, packet_path, model, prior, state, started):
    state['phase']='cpu-preparation';execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    packet,frozen,products,cases,rows = prepare(data,model,prior)
    if json.loads(Path(packet_path).read_text()) != packet:
        raise ValueError('Frozen v4 preflight changed')
    write_json(out/'packet.json',packet);check_elapsed(started)
    state['phase']='model-verification'
    provenance=verify_model(model,packet['model']);write_json(out/'model-verification.json',provenance)
    assert_model_files(model,provenance);check_elapsed(started)
    state['phase']='feature-store';store=FeatureStore(out/'features',provenance)
    encoder=None;observations=[];endpoints={};native_start=time.perf_counter()
    try:
        state['phase']='model-load';encoder=NativeEncoder(model)
        if encoder.evidence_source != provenance['evidenceSource']:
            raise ValueError('Native encoder provenance mismatch')
        check_elapsed(started)

        def extract(row):
            check_elapsed(started)
            values,timing=encoder.encode(row['text'],row);state['forwards']+=1
            if (timing['batchRows']!=1 or timing['paddedTokensPerRow']!=row['tokens']
                    or timing['truncatedRows']!=0 or timing['backboneForwardCalls']!=1):
                raise ValueError('Exactly one B1 whole-input native forward required')
            check_elapsed(started)
            _,receipt=store.save(row,values,timing);state['features']+=1
            observation={'member':row['member'],'timing':timing,'artifactSha256':receipt['artifactSha256']}
            observations.append(observation)
            with (out/'native-observations.jsonl').open('a') as handle:
                handle.write(json.dumps(observation,allow_nan=False)+'\n')
            print(json.dumps({'event':'v4-feature','completed':len(observations),'fixed':104}),flush=True)
            check_elapsed(started)
            return values

        state['phase']='train-extraction';train_rows=[r for r in rows if r['split']=='train']
        for row in train_rows:extract(row)
        if len(observations)!=80:raise ValueError('Exactly80 TRAIN features required before fit')
        write_json(out/'TRAIN_FEATURES_FROZEN.json',{'nativeForwards':80,'evaluationFeaturesRead':False,
            'files':{str(p.relative_to(out)):file_digest(p) for row in train_rows for p in store.paths(row['role'],row['member'])}})
        state['phase']='fit';torch.set_num_threads(1);torch.manual_seed(CONFIG['seed']);fit_start=time.perf_counter()
        labels=json.loads((Path(data)/'train-labels.json').read_text())
        q,c,desired,means,meta=fit_inputs(store,rows,cases,labels)
        meta.update(datasetFreezeSha256=packet['dataFreezeSha256'],packetSha256=file_digest(out/'packet.json'),
                    featureProvenanceSha256=digest(canonical(provenance)),representation=VERSION+'/whole-public-facts')
        learned=IndependentFunctionHead(means);random_head=IndependentFunctionHead(means)
        if sum(p.numel() for p in learned.parameters())!=131104:raise ValueError('Parameter count changed')
        optimizer=torch.optim.AdamW(learned.parameters(),lr=.03,weight_decay=.001);steps=[]
        for step in range(80):
            check_elapsed(started);optimizer.zero_grad()
            query_loss=F.binary_cross_entropy_with_logits(learned.logits('query',q),desired)
            catalog_loss=F.cross_entropy(learned.logits('catalog',c),torch.arange(16))
            loss=(query_loss+catalog_loss)/2
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();optimizer.step();state['learnedSteps']+=1
            steps.append({'step':step,'loss':loss.item(),'queryBCE':query_loss.item(),'catalogCE':catalog_loss.item()})
        prior_logits=nn.Parameter(torch.zeros(16));po=torch.optim.AdamW([prior_logits],lr=.03,weight_decay=.001);prior_steps=[]
        for step in range(80):
            check_elapsed(started);po.zero_grad();loss=F.binary_cross_entropy_with_logits(prior_logits.expand(64,-1),desired)
            if not torch.isfinite(loss):raise ValueError('Nonfinite prior loss')
            loss.backward();po.step();state['priorSteps']+=1;prior_steps.append({'step':step,'loss':loss.item()})
        hashes={};restored={}
        for name,head in (('learned',learned),('random',random_head)):
            path=out/(name+'.pt');head_sha=save_head(path,head,meta);restored[name]=load_head(path,head_sha,meta)
            if any(not torch.equal(v,restored[name].state_dict()[k]) for k,v in head.state_dict().items()):
                raise ValueError('Exact checkpoint reload failed')
            hashes[name]={'file':path.name,'sha256':head_sha,'metadata':meta}
        with (out/'prior.pt').open('xb') as handle:
            torch.save({'version':VERSION,'logits':prior_logits.detach(),'metadata':meta,'steps':80},handle)
        prior_sha=file_digest(out/'prior.pt');prior_saved=torch.load(out/'prior.pt',map_location='cpu',weights_only=True)
        if not torch.equal(prior_logits.detach(),prior_saved['logits']):raise ValueError('Prior reload changed')
        prior_scores=prior_saved['logits'].sigmoid()
        write_json(out/'HEADS_FROZEN.json',{'heads':hashes,'priorSha256':prior_sha,'metadata':meta,
            'trainingSteps':steps,'priorSteps':prior_steps,'evaluationLabelsParsed':False,
            'evaluationFeaturesEncoded':False,'fixedSemanticThreshold':.5,'packetSha256':file_digest(out/'packet.json')})
        fit_seconds=time.perf_counter()-fit_start
        state['phase']='fresh-evaluation-endpoints';lookup={(r['role'],r['caseId']):r for r in rows}
        for case in [x for x in cases if x['split']=='evaluation']:
            begin=time.perf_counter();query=extract(lookup['request',case['caseId']])
            catalog=torch.cat([store.load(lookup['catalog',str(i)])[0] for i in range(16)])
            if not torch.equal(catalog,c):raise ValueError('Verified warm catalog changed')
            with torch.no_grad():
                head=restored['learned'];scores=(head.catalog_probabilities(catalog)@head.query_probabilities(query)[0]).tolist()
            payload=prediction_payload(scores,case['request'],products);encoded=canonical(payload).encode()
            endpoint={'freshEncodeToSerializedDecisionsMs':(time.perf_counter()-begin)*1000,
                'payload':payload,'payloadSha256':hashlib.sha256(encoded).hexdigest(),'payloadBytes':len(encoded)}
            endpoints[case['caseId']]=endpoint;state['endpoints']+=1
            with (out/'fresh-endpoints.jsonl').open('a') as handle:
                handle.write(json.dumps({'caseId':case['caseId'],**endpoint},allow_nan=False)+'\n')
            check_elapsed(started)
        memory=encoder.memory();load_seconds=encoder.load_seconds;native_seconds=time.perf_counter()-native_start
        check_elapsed(started)
    finally:
        if encoder is not None:
            previous=state['phase'];state['phase']='native-cleanup'
            encoder.close();state['phase']=previous
    state['phase']='raw-feature-freeze';check_elapsed(started)
    write_json(out/'native-observations.json',observations);write_json(out/'FRESH_REQUEST_ENDPOINTS.json',endpoints)
    artifacts={str(p.relative_to(out)):file_digest(p) for row in rows for p in store.paths(row['role'],row['member'])}
    write_json(out/'RAW_FEATURES_FROZEN.json',{'files':artifacts,'nativeForwards':104,
        'evaluationFeaturesEncodedAfterHeadsFreeze':True})
    if state['forwards']!=104 or state['endpoints']!=24 or state['features']!=104:
        raise ValueError('Fixed104 features/24 fresh endpoints required')
    state['phase']='cached-control-predictions';predictions={}
    for split in ('train','evaluation'):
        selected=[x for x in cases if x['split']==split]
        queries=torch.cat([store.load(lookup['request',x['caseId']])[0] for x in selected])
        predictions[split]=predict(restored,prior_scores,queries,c,selected,products);check_elapsed(started)
    for rid,endpoint in endpoints.items():
        cached=predictions['evaluation']['learned']['cases'][rid]
        payload={k:cached[k] for k in ('scores','semantic','policy')}
        if endpoint['payload']!=payload or endpoint['payloadSha256']!=hashlib.sha256(canonical(payload).encode()).hexdigest():
            raise ValueError('Contiguous endpoint/replay payload mismatch')
    write_json(out/'PREDICTIONS_FROZEN.json',{'predictions':predictions,
        'headsFreezeSha256':file_digest(out/'HEADS_FROZEN.json'),
        'rawFeatureFreezeSha256':file_digest(out/'RAW_FEATURES_FROZEN.json'),
        'evaluationLabelsParsed':False,'fixedSemanticThreshold':.5})
    state['phase']='postfreeze-evaluation'
    evaluation=json.loads((Path(data)/'evaluation-labels.json').read_text())
    pairs=json.loads((Path(data)/'preference-pairs.json').read_text());results={}
    for split,external_labels in (('train',labels),('evaluation',evaluation)):
        selected=[x for x in cases if x['split']==split]
        results[split]=evaluate(predictions[split],selected,external_labels,pairs)
    state['phase']='final-integrity';check_elapsed(started)
    read_public(data);final_meta=fit_inputs(store,rows,cases,labels)[-1]
    if any(final_meta[k]!=meta[k] for k in final_meta):raise ValueError('TRAIN metadata changed')
    if source_hashes()!=packet['sourceHashes'] or file_digest(out/'prior.pt')!=prior_sha:
        raise ValueError('Source or prior changed during run')
    if any(file_digest(out/v['file'])!=v['sha256'] for v in hashes.values()):raise ValueError('Head changed')
    state['phase']='report';check_elapsed(started)
    report={'status':'completed','version':VERSION,'executionCommit':execution_commit,
        'packetSha256':file_digest(out/'packet.json'),'dataFreezeSha256':packet['dataFreezeSha256'],
        'modelInputSha256':frozen['bridge']['modelInputSha256'],'sourceHashes':packet['sourceHashes'],
        'heads':hashes,'priorSha256':prior_sha,'featureProvenance':provenance,
        'predictionsFreezeSha256':file_digest(out/'PREDICTIONS_FROZEN.json'),
        'rawFeaturesFreezeSha256':file_digest(out/'RAW_FEATURES_FROZEN.json'),
        'nativeForwards':104,'trainFeatures':80,'trainingQueries':64,'evaluationQueries':24,
        'learnedUpdates':80,'priorUpdates':80,'modelLoadSeconds':load_seconds,
        'nativeLifecycleSecondsIncludingLoadFitAndEndpoints':native_seconds,'fitSeconds':fit_seconds,
        'totalWallSeconds':time.perf_counter()-started,'memory':memory,'results':results,
        'freshRequestEndpoints':endpoints,
        'freshRequestMedianMs':statistics.median(x['freshEncodeToSerializedDecisionsMs'] for x in endpoints.values()),
        'latencyScope':'Fresh query encode, immutable query receipt and verification, 16 verified warm catalog reads, '
            'independent learned scoring, threshold/guards, semantic-strict and exact-name-policy decisions, '
            'and serialized payload. Model already resident. No generated copy, images or browser rendering.',
        'semanticThreshold':.5,'thresholdChosenBeforeEvaluation':True,
        'exactPolicyIsLearned':False,'guardPolicyIsLearned':False,'promoted':False,
        'limitations':frozen['limitations'] + [
            'Known16-product function adaptation; all identities/functions TRAIN-seen.',
            'Flat BCE zeros mean unrequested in these synthetic cases, not universal incompatibility.',
            'Fixed 0.5 threshold is a predeclared experiment, not calibrated quality evidence.',
            'Exact-name routing is deterministic and reported separately from semantic model quality.',
            'Fewer returned cards are disclosed shortfalls, not a copy/UI latency shortcut.',
            'v3 is inspected development replay; v4 is author-visible synthetic evaluation, not sealed.',
            'Cooperative deadline cannot interrupt a hung native call.',
            'No promotion or complete useful above-fold/full-page timing.']}
    write_json(out/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--data',type=Path,default=data_api.DATA)
    parser.add_argument('--model',type=Path,default=Path(DEFAULT_MODEL))
    parser.add_argument('--prior',type=Path,default=Path('data/ranker/pretrained/catalog-user-fusion-v1-attempt-1'))
    parser.add_argument('--packet',type=Path,required=True);parser.add_argument('--out',type=Path);parser.add_argument('--owner')
    args=parser.parse_args()
    if args.action=='prepare':
        packet=prepare(args.data,args.model,args.prior)[0];write_json(args.packet,packet)
        print(canonical({'nativeForwards':104,'tokenRange':packet['tokenRange'],'evaluationLabelsParsed':False}))
    else:
        if args.out is None or args.owner!='root':raise ValueError('Root --owner root/new --out required')
        result=run(args.out,args.data,args.packet,args.model,args.prior,args.owner)
        print(canonical({'status':result['status'],'nativeForwards':result['nativeForwards']}))


if __name__=='__main__':
    main()
