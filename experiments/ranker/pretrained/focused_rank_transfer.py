"""No-fit focused-v2 transfer to the inspected 36-product pool; root owns native.

prepare is CPU-only. run requires a reviewed exact packet and exactly 60 B1
forwards. Product ranking only: no image selection, copy or full-page claim.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import time
import torch

from .boundary import guard_reasons
from .catalog_fusion.fresh_eval_data import verify_dataset
from .catalog_fusion.fresh_eval_runner import public_inputs, opposite_preferences
from .catalog_fusion.generalization_v2 import catalog_text_v2, query_text_v2, load_head, require_frozen, native_owner_lock
from .catalog_fusion.inputs import canonical, digest
from .catalog_fusion.paired_data import ATOMS
from .catalog_fusion.preflight import DEFAULT_MODEL, REVISION, file_digest
from .catalog_fusion.runner import execution_receipt, source_hashes
from .catalog_fusion.runtime import FeatureStore, NativeEncoder, assert_model_files, verify_model, write_json

ROOT = Path(__file__).resolve().parent
VERSION = 'focused-rank-transfer.v1'
SOURCE = Path('data/ranker/pretrained/generalization-v2-attempt-1')
DATA = ROOT / 'catalog_fusion/fresh_eval_v1'
ARMS = ('learned', 'prior', 'random', 'opposing-query', 'public-tag-oracle')
OFFSETS, COUNTS = (0, 12, 24), (12, 24)
LIMITS = {'nativeForwards': 60, 'catalogForwards': 36, 'queryForwards': 24,
          'memoryGiB': 6, 'maxTokens': 512, 'nativeSeconds': 180, 'pilotSeconds': 60}
DENOMINATORS = {'requests': 24, 'products': 36, 'preferencePairs': 12,
                'rosterRotations': 3, 'topCounts': [12, 24], 'arms': list(ARMS)}


def sources():
    return {**source_hashes(), **{str((ROOT / n).resolve()): file_digest(ROOT / n)
            for n in ('focused_rank_transfer.py', 'test_focused_rank_transfer.py')}}


def heads(source):
    source = Path(source); packet = json.loads((source / 'packet.json').read_text())
    models = require_frozen(source, packet)
    report = json.loads((source / 'report.json').read_text())
    if report['status'] != 'completed' or report['nativeForwards'] != 192:
        raise ValueError('Completed immutable v2 source required; no interim checkpoint')
    frozen = json.loads((source / 'HEADS_FROZEN.json').read_text())
    receipt = {'sourceDirectory': str(source.resolve()), 'headFreezeSha256': file_digest(source / 'HEADS_FROZEN.json'),
        'sourceReportSha256': file_digest(source / 'report.json'),
        'heads': {a: frozen['heads']['focused-' + ('fusion' if a == 'learned' else a)] for a in ARMS[:3]}}
    return receipt, {a: models['focused-' + ('fusion' if a == 'learned' else a)] for a in ARMS[:3]}, packet


def prepare(dataset, source, model, tokenizer):
    frozen, catalog, requests = public_inputs(dataset)
    receipt, _, old = heads(source); model = Path(model).resolve()
    provenance = json.loads((Path(source) / 'model-verification.json').read_text())
    assert_model_files(model, provenance)
    if model.name != REVISION or str(model) != old['modelPath']:
        raise ValueError('Pinned model path required')
    for name, key in (('tokenizer.json', 'tokenizerSha256'), ('config.json', 'configSha256'), ('chat_template.jinja', 'chatTemplateSha256')):
        if file_digest(model / name) != old[key]: raise ValueError('Pinned model/tokenizer changed')
    rows = [{'role': 'catalog', 'member': pid, 'text': catalog_text_v2(p, 'focused')} for pid, p in catalog.items()]
    rows += [{'role': 'request', 'member': rid, 'text': query_text_v2(q, catalog, 'focused')} for rid, q in requests.items()]
    for row in rows:
        ids = tokenizer.encode(row['text'], add_special_tokens=False).ids
        if not 0 < len(ids) <= 512: raise ValueError('No token truncation permitted')
        row.update(textSha256=digest(row['text']), tokenIds=ids, tokenSha256=digest(canonical(ids)), tokens=len(ids), batchShape=[1, len(ids)])
    if len(rows) != 60: raise ValueError('Exactly 36 catalog +24 query rows required')
    overlap = {role: {split: sum(r['textSha256'] in {z['textSha256'] for z in old['records']
        if z['mode'] == 'focused' and z['role'] == role and z['split'] == split} for r in rows if r['role'] == role)
        for split in ('train', 'validation', 'test')} for role in ('catalog', 'request')}
    packet = {'version': VERSION, 'datasetDirectory': str(Path(dataset).resolve()),
        'dataFreezeSha256': file_digest(Path(dataset) / 'FREEZE.json'), 'sourceHeads': receipt,
        'sourceHashes': sources(), 'limits': LIMITS, 'fixedDenominators': DENOMINATORS,
        **{k: old[k] for k in ('modelPath', 'modelRevision', 'quantization', 'tokenizerSha256', 'configSha256', 'chatTemplateSha256')},
        'sourceModelVerificationSha256': file_digest(Path(source) / 'model-verification.json'),
        'records': [{k: v for k, v in r.items() if k != 'text'} for r in rows],
        'tokenRange': {role: [min(r['tokens'] for r in rows if r['role'] == role), max(r['tokens'] for r in rows if r['role'] == role)] for role in ('catalog', 'request')},
        'sourceFocusedExactTextOverlapOccurrences': overlap,
        'limitations': ['Inspected old fresh_eval_v1 replay, not unseen/sealed test; no fit or new labels.',
            'Focused positive-atom task only; budget/stock/shipping guards remain deterministic.',
            '24 requests repeat12 preference sets across image styles; no image action is evaluated.',
            'Head rows canonicalized by product ID before GEMM; scores rebound to roster for exact rotation invariance.',
            'Product-only component timing; no copy, grouping, browser or full-page claim.']}
    return packet, catalog, requests, rows, provenance


def ranked(scores, order, request, catalog):
    if scores.shape != (36,) or not torch.isfinite(scores).all() or len(order) != 36 or set(order) != set(catalog):
        raise ValueError('Finite complete 36-product score vector required')
    raw = sorted(order, key=lambda p: (-scores[order.index(p)].item(), p))
    eligible = [p for p in raw if not guard_reasons(request, catalog[p])]
    return {'rawRank': raw, 'eligibleRank': eligible, 'top': {str(k): eligible[:k] for k in COUNTS},
            'shortfall': {str(k): max(0, k-len(eligible)) for k in COUNTS}}


def score(model, query, matrix, order):
    # Canonical rows prevent Torch GEMM rounding from depending on roster order.
    indices=sorted(range(len(order)),key=lambda i:order[i]); inverse=[indices.index(i) for i in range(len(order))]
    with torch.no_grad():
        qp, cp = model.probabilities(query, matrix[indices])
        return (cp @ qp)[inverse], qp, cp[inverse]


def evaluate(predictions, labels, controls):
    result = {}
    pairs = [p for p in controls['pairs'] if p['kind'] == 'product']
    if len(labels) != 24 or set(labels) != set(predictions) or len(pairs) != 12 or any(set(p['recordIds']) - set(labels) for p in pairs): raise ValueError('Fixed label/pair denominator changed')
    if controls['requestedCardCounts'] != list(COUNTS) or controls['mandatoryRosterRotationOffsets'] != list(OFFSETS): raise ValueError('Fixed card/rotation controls changed')
    for arm in ARMS:
        cases = {}; correct = 0
        for rid, arms in predictions.items():
            readout = arms[arm]['readout']; truth = labels[rid]; rel = truth['relevance']
            chosen = readout['eligibleRank'][0] if readout['eligibleRank'] else None
            correct += chosen == truth['bestProductId']
            dcg = lambda vs: sum((2**v-1) / math.log2(i+2) for i,v in enumerate(vs))
            cases[rid] = {'correctUniqueGold': chosen == truth['bestProductId'],
                'ndcg': {str(k): dcg([rel[p] for p in readout['top'][str(k)]]) / dcg(sorted(rel.values(), reverse=True)[:k]) for k in COUNTS}}
        result[arm] = {'fixedRequests': 24, 'uniqueGoldCorrect': correct,
            'hardCompliance': {'violations': 0, 'checkedGridComponents': 144},
            'preferencePairs': 12, 'pairChoiceChanges': sum(predictions[p['recordIds'][0]][arm]['readout']['eligibleRank'][0] != predictions[p['recordIds'][1]][arm]['readout']['eligibleRank'][0] for p in pairs), 'pairBothCorrect': sum(all(cases[r]['correctUniqueGold'] for r in p['recordIds']) for p in pairs),
            'meanNdcg': {str(k): sum(c['ndcg'][str(k)] for c in cases.values()) / 24 for k in COUNTS}, 'cases': cases}
    if result['public-tag-oracle']['uniqueGoldCorrect'] != 24: raise AssertionError('Unique public gold/oracle must pass24/24')
    return result


def run(out, prepared, encoder_factory):
    packet, catalog, requests, rows, provenance = prepared; out = Path(out)
    receipt, models, _ = heads(packet['sourceHeads']['sourceDirectory'])
    if receipt != packet['sourceHeads'] or sources() != packet['sourceHashes']: raise ValueError('Frozen source/head changed')
    for arm, info in receipt['heads'].items():
        shutil.copyfile(Path(receipt['sourceDirectory']) / info['file'], out / info['file'])
        models[arm] = load_head(out / info['file'], info['sha256'], info['metadata'])
    write_json(out / 'SOURCE_HEADS.json', receipt)
    store = FeatureStore(out / 'features', provenance); mapping = {(r['role'], r['member']): r for r in rows}
    query = {}; observations = []; begin = time.perf_counter(); encoder = encoder_factory(); forwards = 0
    def encode(row):
        nonlocal forwards
        if forwards >= 60 or time.perf_counter()-begin > 180 or (forwards < 2 and time.perf_counter()-begin > 60): raise TimeoutError('Fixed60/pilot/time guard')
        values, timing = encoder.encode(row['text'], row); forwards += 1
        if timing['batchRows'] != 1 or timing['paddedTokensPerRow'] != row['tokens'] or timing['backboneForwardCalls'] != 1 or timing['truncatedRows'] != 0: raise ValueError('Exact native B1 receipt required')
        if time.perf_counter()-begin > 180 or (forwards <= 2 and time.perf_counter()-begin > 60): raise TimeoutError('Fixed native budget exceeded')
        store.save(row, values, timing)
        with (out / 'native-observations.jsonl').open('a') as h: h.write(json.dumps({'role':row['role'],'member':row['member'],'timing':timing,'memory':encoder.memory()})+'\n')
        return values[0], timing
    try:
        if encoder.evidence_source != provenance["evidenceSource"]: raise ValueError("Encoder provenance mismatch")
        cold = time.perf_counter()
        for row in rows:
            if row['role'] == 'catalog': encode(row)
        cold_ms = (time.perf_counter()-cold)*1000
        for rid, request in requests.items():
            start = time.perf_counter(); query[rid], timing = encode(mapping['request', rid]); query_ms=(time.perf_counter()-start)*1000
            cache = time.perf_counter(); order=request['candidateProductIds']; matrix=torch.stack([store.load(mapping['catalog',p])[0][0] for p in order]); cache_ms=(time.perf_counter()-cache)*1000
            before=time.perf_counter(); scores,_,_=score(models['learned'],query[rid],matrix,order); score_ms=(time.perf_counter()-before)*1000
            before=time.perf_counter(); readout=ranked(scores,order,request,catalog); grid_ms=(time.perf_counter()-before)*1000
            observations.append({'recordId':rid,'freshQueryNativeMs':timing['encodeTotalMs'],'querySaveVerifyMs':query_ms,
                'verifiedWarmCatalog36Ms':cache_ms,'headMs':score_ms,'guardsTop12Top24Ms':grid_ms,
                'queryWarmCatalogHeadGuardsTop12Top24Ms':(time.perf_counter()-start)*1000,'catalogHits':36,'freshQueryForwards':1})
            with (out / 'progress.jsonl').open('a') as h: h.write(json.dumps({'completedForwards':forwards,'fixedForwards':60,'completedQueries':len(observations)})+'\n')
        memory=encoder.memory(); load_seconds=encoder.load_seconds
    finally: encoder.close()
    if forwards != 60 or len(observations) != 24: raise ValueError('Incomplete fixed extraction')
    opposite=opposite_preferences(requests); predictions={}
    for rid, request in requests.items():
        order=request['candidateProductIds'];matrix=torch.stack([store.load(mapping['catalog',p])[0][0] for p in order]);predictions[rid]={}
        for arm in ARMS:
            origin=opposite[rid] if arm=='opposing-query' else rid
            if arm=='public-tag-oracle':
                qp=torch.tensor([float(a in request['context']['statedPreferences']) for a in ATOMS]);cp=torch.tensor([[float(a in catalog[p]['tags']) for a in ATOMS] for p in order]);scores=cp@qp
            else:scores,qp,cp=score(models['learned' if arm=='opposing-query' else arm],query[origin],matrix,order)
            views={}
            for offset in OFFSETS:
                changed=order[offset:]+order[:offset];indices=[order.index(p) for p in changed]
                rotated=scores[indices] if arm=='public-tag-oracle' else score(models['learned' if arm=='opposing-query' else arm],query[origin],matrix[indices],changed)[0]
                if not torch.equal(rotated,scores[indices]):raise AssertionError('Fatal same-shape roster score invariance mismatch')
                view=ranked(rotated,changed,request,catalog)
                if any(guard_reasons(request,catalog[p]) for k in COUNTS for p in view['top'][str(k)]):raise AssertionError('Hard compliance violation')
                views[str(offset)]=view
            if any(v!=views['0'] for v in views.values()):raise AssertionError('Roster rotation changed ID-bound ranking')
            predictions[rid][arm]={'querySource':origin,'productIds':order,'scores':scores.tolist(),'queryProbabilities':qp.tolist(),
                'catalogProbabilities':cp.tolist(),'readout':views['0'],'rosterViews':views}
    write_json(out/'predictions.json',predictions)
    sealed={str(p.relative_to(out)):file_digest(p) for p in (out/'features').glob('*/*')}
    write_json(out/'PREDICTIONS_FROZEN.json',{'packetSha256':file_digest(out/'packet.json'),'predictionsSha256':file_digest(out/'predictions.json'),
        'files':sealed,'nativeForwards':60,'fits':0,'privateLabelsOpened':False,'sourceHeads':receipt})
    freeze=json.loads((out/'PREDICTIONS_FROZEN.json').read_text())
    if file_digest(out/'predictions.json') != freeze['predictionsSha256'] or any(file_digest(out/n) != sha for n,sha in freeze['files'].items()): raise ValueError('Frozen predictions/features changed')
    if file_digest(Path(packet['datasetDirectory'])/'FREEZE.json') != packet['dataFreezeSha256']: raise ValueError('Dataset freeze changed')
    verify_dataset(packet['datasetDirectory'],private=True)
    labels={r['recordId']:r for r in map(json.loads,(Path(packet['datasetDirectory'])/'private/truth.jsonl').read_text().splitlines())}
    controls=json.loads((Path(packet['datasetDirectory'])/'private/controls.json').read_text());quality=evaluate(predictions,labels,controls)
    if sources()!=packet['sourceHashes'] or heads(receipt['sourceDirectory'])[0]!=receipt:raise ValueError('Source/head changed during transfer')
    write_json(out/'report.json',{'status':'completed','version':VERSION,'execution':execution_receipt(),'packetSha256':file_digest(out/'packet.json'),
        'evidenceSource':provenance['evidenceSource'],'fixedDenominators':DENOMINATORS,'nativeForwards':60,'fits':0,'totalRunWallSeconds':time.perf_counter()-begin,'coldCatalogEncodeSaveVerifyMs':cold_ms,'residentLoadSeconds':load_seconds,
        'observations':observations,'warmRequestMedianMs':statistics.median(o['queryWarmCatalogHeadGuardsTop12Top24Ms'] for o in observations),
        'quality':quality,'memory':memory,'privateLabelsOpenedAfterPredictionsFreeze':True,'limitations':packet['limitations']})


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('prepare','run'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=DATA);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--model',type=Path,default=Path(DEFAULT_MODEL))
    p.add_argument('--packet',type=Path);p.add_argument('--reviewed-packet-sha256');p.add_argument('--native-owner',choices=('root',));p.add_argument('--confirm-serialized',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Preserve previous result')
    a.out.mkdir(parents=True);start=time.perf_counter();torch.set_num_threads(1)
    try:
        from tokenizers import Tokenizer
        prepared=prepare(a.dataset,a.source,a.model,Tokenizer.from_file(str(a.model/'tokenizer.json')));packet=prepared[0]
        if a.action=='run' and (a.native_owner!='root' or not a.confirm_serialized or a.packet is None or file_digest(a.packet)!=a.reviewed_packet_sha256 or json.loads(a.packet.read_text())!=packet):raise ValueError('Exact reviewed packet and serialized root ownership required')
        write_json(a.out/'packet.json',packet)
        if a.action=='prepare':print(canonical({'nativeForwards':60,'packetSha256':file_digest(a.out/'packet.json'),'tokenRange':packet['tokenRange']}));return
        with native_owner_lock(Path('data/ranker/pretrained/.native-generalization-v2.lock'),a.native_owner):
            verified=verify_model(a.model,packet)
            if any(verified[k]!=prepared[-1][k] for k in ('weightsSha256','tokenizerSha256','templateSha256','quantization','configSha256','evidenceSource')):raise ValueError('Full model receipt differs')
            assert_model_files(a.model,verified)
            write_json(a.out/'model-verification.json',verified)
            run(a.out,prepared,lambda: NativeEncoder(a.model))
    except BaseException as error:
        write_json(a.out/'FAILED.json',{'status':'failed','type':type(error).__name__,'message':str(error),'fixedDenominators':DENOMINATORS,
            'completedForwards':len((a.out/'native-observations.jsonl').read_text().splitlines()) if (a.out/'native-observations.jsonl').exists() else 0,
            'completedQueries':len((a.out/'progress.jsonl').read_text().splitlines()) if (a.out/'progress.jsonl').exists() else 0,
            'fixedNativeForwards':60,'partialSuccessNotPromoted':True,'wallSeconds':time.perf_counter()-start});raise


if __name__=='__main__':main()
