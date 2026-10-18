"""CPU checkpoint/feature replay; writes new audit receipts, never the run."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics

import torch
from experiments.ranker.pretrained import store_function_adaptation as base
from experiments.ranker.pretrained import store_function_adaptation_v5 as corrected
from experiments.ranker.pretrained.catalog_fusion.inputs import canonical, digest, tensor_digest
from experiments.ranker.pretrained.catalog_fusion.runtime import FeatureStore
from experiments.ranker.pretrained.store_catalog_v1 import load_snapshot
from experiments.ranker.pretrained.store_demo_engine import guarded_order
from experiments.ranker.pretrained.store_rank_session import action_bindings

OUT = Path(__file__).resolve().parent
RUN = Path('data/ranker/pretrained/actual-store-function-v5-attempt-1')
DATA = Path('experiments/ranker/pretrained/store_function_v5')
counts = Counter()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(category, condition, detail):
    counts[category] += 1
    if not condition:
        raise AssertionError(category + ': ' + detail)


def save(name, value):
    path = OUT / name
    if path.exists():
        if read(path) != json.loads(json.dumps(value)):
            raise ValueError('Preserved audit receipt differs: ' + str(path))
        return
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def inventory():
    return {str(p.relative_to(RUN)): sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}


original = read(OUT / 'ORIGINAL_RUN_FILES_SHA256.json')['files']
check('original', inventory() == original, 'original run differs from pre-edit inventory')
report = read(RUN / 'report.json'); packet = read(RUN / 'packet.json')
head_freeze = read(RUN / 'HEADS_FROZEN.json'); pred_freeze = read(RUN / 'PREDICTIONS_FROZEN.json')
predicted = pred_freeze['predictions']; freeze = read(DATA / 'FREEZE.json')
check('pins', sha(RUN / 'packet.json') == report['packetSha256'], 'packet/report hash')
check('pins', sha(DATA / 'FREEZE.json') == packet['datasetFreezeSha256'], 'dataset freeze')
for name, expected in packet['datasetFilesSha256'].items():
    check('pins', sha(DATA / name) == expected, name)
for name, expected in packet['sourceSha256'].items():
    check('pins', sha(RUN / 'source' / name) == expected, 'archived ' + name)
for path, expected in packet['reusedFeatureFilesSha256'].items():
    check('pins', sha(path) == expected, 'reused ' + path)
check('pins', pred_freeze['headsFreezeSha256'] == sha(RUN / 'HEADS_FROZEN.json'), 'prediction/head freeze chain')
check('pins', report['servingHeadExportSha256'] == sha(RUN / 'serving-head.json'), 'serving export')
for receipt in (packet, head_freeze, pred_freeze):
    key = 'evaluationLabelsParsed' if receipt is packet else 'evaluationLabelsRead'
    check('boundary', receipt[key] is False, key)

cases = read(DATA / 'public-cases.json'); labels = read(DATA / 'evaluation-labels.json')
train = [c for c in cases if c['split'] == 'train']; evaluation = [c for c in cases if c['split'] == 'evaluation']
train_labels = read(DATA / 'train-labels.json'); target_map = {r['caseId']: r['targetProductIndex'] for r in train_labels}
targets = {r['caseId']: r['targetProductIndex'] for r in labels}
check('roster', len(train) == 96 and len(evaluation) == 32 and len(target_map) == 96, 'query roster')
check('roster', len({base.request_text(c['request']) for c in cases}) == 128, 'request disjointness')
check('roster', Counter(target_map.values()) == Counter({i: 6 for i in range(16)}), 'balanced TRAIN')
check('roster', Counter(targets.values()) == Counter({i: 2 for i in range(16)}), 'balanced evaluation')
rows = packet['records']; lookup = {(r['role'], r['caseId']): r for r in rows}
check('roster', len(rows) == 144 and len(lookup) == 144, 'feature roster')
check('roster', sum(r['origin'] == 'verified-v3-reuse' for r in rows) == 48, 'reuse48')
check('roster', sum(r['origin'] == 'new-native-required' for r in rows) == 96, 'new96')
store = FeatureStore.open_existing(RUN / 'features', packet['featureProvenance'])
old_store = FeatureStore.open_existing(Path(packet['prior']) / 'features', packet['featureProvenance'])
vectors = {}; feature_receipts = {}
for row in rows:
    key = (row['role'], row['caseId']); value, receipt = store.load(row)
    check('features', value.shape == (1, 4096) and value.dtype == torch.float32, str(key))
    check('features', torch.isfinite(value).all().item(), 'finite ' + str(key))
    vectors[key] = value; feature_receipts[key] = receipt
    if row['origin'] == 'verified-v3-reuse':
        old_value, old_receipt = old_store.load(row)
        check('reuse', torch.equal(value, old_value) and receipt['rawKeys'] == old_receipt['rawKeys'], str(key))
        check('reuse', receipt['timing'] == old_receipt['timing'], 'original timing ' + str(key))
observations = report['nativeObservations']
expected_new = [r for r in rows if r['split'] == 'train' and r['origin'] == 'new-native-required'] + [lookup['request', c['caseId']] for c in evaluation]
check('native', report['nativeForwards'] == report['serializedNewFeatures'] == len(observations) == 96, 'forward count')
check('native', [r['caseId'] for r in observations] == [r['caseId'] for r in expected_new], 'forward order TRAIN then evaluation')
for obs, row in zip(observations, expected_new, strict=True):
    timing = obs['timing']
    check('native', timing['batchRows'] == timing['backboneForwardCalls'] == 1 and timing['truncatedRows'] == 0, row['caseId'])
    check('native', timing['paddedTokensPerRow'] == row['tokens'] and 0 < row['tokens'] <= 512, 'complete ' + row['caseId'])
    check('native', feature_receipts['request', row['caseId']]['timing'] == timing, 'receipt ' + row['caseId'])
check('native', report['frozen927HashesUnchanged'] is True, 'producer927 shutdown receipt')
check('native', all(r['status'] == 'completed' for r in report['evaluationRoster']) and len(report['evaluationRoster']) == 32, 'completed eval roster')

query = torch.cat([vectors['request', c['caseId']] for c in train])
catalog = torch.cat([vectors['catalog', str(i)] for i in range(16)])
eval_vectors = torch.cat([vectors['request', c['caseId']] for c in evaluation])
train_targets = [target_map[c['caseId']] for c in train]
means = {'query': query.double().mean(0).float(), 'catalog': catalog.double().mean(0).float()}
models = {}
for name in ('learned', 'random'):
    meta = report['heads'][name]
    check('heads', head_freeze['heads'][name] == meta and sha(RUN / meta['file']) == meta['sha256'], name)
    models[name] = base.load_head(RUN / meta['file'], meta['sha256'], meta['metadata'])
    check('boundary', meta['metadata']['queryMembers'] == [c['caseId'] for c in train], 'TRAIN membership ' + name)
    check('boundary', meta['metadata']['trainTargetsSha256'] == digest(canonical(train_targets)), 'TRAIN targets ' + name)
    for role in ('query', 'catalog'):
        check('boundary', torch.equal(getattr(models[name], role + '_mean'), means[role]), 'TRAIN-only mean ' + role)
        check('boundary', meta['metadata']['meanSha256'][role] == tensor_digest(means[role]), 'mean checksum ' + role)
initial = base.FunctionHead(means)
check('heads', all(torch.equal(v, initial.state_dict()[k]) for k, v in models['random'].state_dict().items()), 'random is initial state')
check('heads', any(not torch.equal(v, models['random'].state_dict()[k]) for k, v in models['learned'].state_dict().items()), 'learned actual update')
losses = head_freeze['trainingLosses']
check('heads', len(losses) == 80 and [v['step'] for v in losses] == list(range(1, 81)), 'fixed80 steps')
first_eval_path = store.paths('request', lookup['request', evaluation[0]['caseId']]['member'])[0]
check('boundary', (RUN / 'HEADS_FROZEN.json').stat().st_mtime_ns <= first_eval_path.stat().st_mtime_ns, 'heads frozen before eval features')
archived_driver = (RUN / 'source/store_function_adaptation_v5.py').read_text()
check('boundary', archived_driver.index("write_json(out / 'PREDICTIONS_FROZEN.json'") < archived_driver.index("labels = json.loads((directory / 'evaluation-labels.json')"), 'source ordering of label parse')
prior_report = read(Path(packet['prior']) / 'report.json'); old_head = prior_report['heads']['learned']
check('pins', sha(Path(packet['prior']) / 'report.json') == packet['priorReportSha256'], 'prior report')
models['frozen-v3'] = base.load_head(Path(packet['prior']) / old_head['file'], old_head['sha256'], old_head['metadata'])
replayed = corrected.predictions(models, eval_vectors, catalog, evaluation, read(DATA / 'catalog.json'))
check('replay', replayed == predicted, 'all five frozen arms exact CPU checkpoint replay')
snapshot = load_snapshot(); products = snapshot['inputs']['products']
for case in evaluation:
    rid = case['caseId']; endpoint = report['freshRequestEndpoints'][rid]; payload = endpoint['payload']
    check('endpoints', payload['scores16'] == predicted['learned'][rid]['scores16'], 'fresh/cached ' + rid)
    order, guards = guarded_order(payload['scores16'], case['request'], {'budgetExclusive': False, 'waterproofRequired': False}, products)
    check('endpoints', payload == {'scores16': payload['scores16'], 'productIndices': order[:6], 'cards': action_bindings(case['request'], order[:6], snapshot), 'guards': guards}, 'actual bindings ' + rid)
    check('endpoints', hashlib.sha256(canonical(payload).encode()).hexdigest() == endpoint['payloadSha256'], 'payload hash ' + rid)
    for arm, arm_rows in predicted.items():
        ranked = sorted(range(16), key=lambda i: (-arm_rows[rid]['scores16'][i], i))
        check('ranking', ranked == arm_rows[rid]['rawRank'], arm + '/' + rid)

results = corrected.score_predictions(predicted, labels)
changes = []
for arm, value in results.items():
    before = report['results'][arm]
    normalized_value = json.loads(json.dumps({k:v for k,v in value.items() if k != 'misses'}))
    check('results', normalized_value == {k:v for k,v in before.items() if k != 'misses'}, 'aggregate counters unchanged ' + arm)
    for old, new in zip(before['misses'], value['misses'], strict=True):
        check('results', {k:v for k,v in old.items() if k != 'targetRank'} == {k:v for k,v in new.items() if k != 'targetRank'}, 'diagnostic only ' + arm)
        check('results', new['targetRank'] == predicted[arm][new['caseId']]['rawRank'].index(targets[new['caseId']]) + 1, 'per-case rank ' + arm)
        if old['targetRank'] != new['targetRank']:
            changes.append({'arm':arm,'caseId':new['caseId'],'target':new['target'],'originalTargetRank':old['targetRank'],'correctedTargetRank':new['targetRank']})
check('results', results['learned']['topOneCorrect'] == 29 and results['learned']['targetInTopSix'] == 32, 'v5 confirmed')
check('results', results['frozen-v3']['topOneCorrect'] == 14 and results['frozen-v3']['targetInTopSix'] == 29, 'v3 confirmed')
check('guards', len(report['guardChecks']) == 16 and all(r['passed'] for r in report['guardChecks']), 'primary16 controls')
check('original', inventory() == original, 'all original310 files unchanged after audit')
latencies = [x['freshEncodeToSerializedActionsMs'] for x in report['freshRequestEndpoints'].values()]
check('timing', statistics.median(latencies) == report['freshRequestMedianMs'], 'endpoint median')
old_ok = {rid:row['rawRank'][0] == targets[rid] for rid,row in predicted['frozen-v3'].items()}
new_ok = {rid:row['rawRank'][0] == targets[rid] for rid,row in predicted['learned'].items()}
correction = {'originalRun':str(RUN),'originalReportSha256':original['report.json'],
    'frozenPredictionsSha256':original['PREDICTIONS_FROZEN.json'],'evaluationLabelsSha256':sha(DATA / 'evaluation-labels.json'),
    'checkpointSha256':original['learned.pt'],'correctedScorerSha256':sha('experiments/ranker/pretrained/store_function_adaptation_v5.py'),
    'scope':'targetRank diagnostic only; raw predictions/checkpoints/report immutable; no native model execution/retraining; CPU checkpoint inference only',
    'changes':changes,'results':results}
save('CORRECTED_RESULTS.json', correction)
corrected_report = deepcopy(report); corrected_report['results'] = results
corrected_report['independentCorrection'] = {k:v for k,v in correction.items() if k not in ('results','changes')}
save('CORRECTED_REPORT.json', corrected_report)
audit = {'version':'function-v5-native-independent-audit.v1','nativeExecutedByReviewer':False,
    'originalFilesUnchanged':len(original),'checks':dict(counts),'totalChecks':sum(counts.values()),
    'originalReportSha256':original['report.json'],'correctedReportSha256':sha(OUT / 'CORRECTED_REPORT.json'),
    'correctedResultsSha256':sha(OUT / 'CORRECTED_RESULTS.json'),'diagnosticRowsChanged':len(changes),
    'results':{name:{k:value[k] for k in ('cases','topOneCorrect','targetInTopSix')} for name,value in results.items()},
    'learnedMisses':results['learned']['misses'],
    'paired':{'improved':sum(new_ok[i] and not old_ok[i] for i in new_ok),'regressed':sum(old_ok[i] and not new_ok[i] for i in new_ok),'bothCorrect':sum(old_ok[i] and new_ok[i] for i in new_ok),'bothWrong':sum(not old_ok[i] and not new_ok[i] for i in new_ok)},
    'timing':{'recordedMedianFreshEncodeThroughSerializedActionsMs':report['freshRequestMedianMs'],
        'medianEvaluationEncodeTotalMs':statistics.median(x['timing']['encodeTotalMs'] for x in observations if x['split']=='evaluation'),
        'scope':'Rank experiment includes feature/report archives, catalog verification and repeated load_snapshot provenance/asset checks; no copy/UI/browser; not live serving latency; snapshot phase not separately timed'},
    'knownLanguageGaps':report['knownLanguageGaps'],'frozen927Integrity':'Producer receipt true; archived source hashes show actual before/after comparison. CPU audit does not recreate native hashes.',
    'noEvalLeakageEvidence':'96 TRAIN-only membership/target digest/means verified; fixed80 source; head freeze precedes evaluation features; hash-linked prediction freeze precedes label parse in archived source. Not a syscall-level access attestation.',
    'recommendation':'Useful known16 single-function ranking candidate for explicit integration/live testing, with all48copy bindings already TRAIN seen; not automatic promotion, unseen generalization, broad profile personalization, or latency success.',
    'remainingLimits':['3/32 top-one misses remain, ranks4/2/5; minimum one correct per function','Including-delivery parser gap remains explicitly recorded','Recorded1.357s rank endpoint misses300ms review goal; independently measure lean live adapter rather than subtract unmeasured overhead','Copy/browser/visible readiness were not rerun with v5 head']}
save('INDEPENDENT_AUDIT.json', audit)
print(json.dumps({'checks':sum(counts.values()),'originalFilesUnchanged':len(original),'diagnosticRowsChanged':len(changes),'results':audit['results'],'learnedMisses':audit['learnedMisses'],'timing':audit['timing']}))
