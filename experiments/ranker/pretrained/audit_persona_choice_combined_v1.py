"""Independent saved-evidence audit; no native calls, optimizer or early gold.

The adapter and combined producer freezes must both pass admission before any
private quality JSON is parsed. Native parity is verified from hashed measured
receipts, not recreated or misrepresented as an independent CPU LM replay.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
import zipfile

HERE = Path(__file__).resolve().parent
VERSION = 'persona-choice-combined-independent-audit.v1'
GATE_SHA = '73fa0d52c5ef4fe983834f610ef00cf89a899cf420cdb81008dd5a7909298c6e'
PACKET_SHA = '2a0801c73440dfbcea50f73c0c1b2de3a96abc8c975caf8d54e92010a84fe312'
MODELS = {'learned', 'frozen-base-lexical-decision', 'TRAIN-popularity', 'latest-owned-repeat'}
CONTROL_NAMES = {'owned-only-public-context': 'owned-only-public-context',
                 'shuffled-public-profile': 'shuffled-profile',
                 'purchase-history-removed': 'purchase-history-removed'}
STRICT = {'frozen-base-lexical-decision', 'TRAIN-popularity', 'latest-owned-repeat',
          'owned-only-public-context', 'shuffled-public-profile'}
SPLITS = {'train': 896, 'evaluation-main': 64, 'evaluation-replacement': 8,
          'evaluation-controls': 192, 'development-history': 40,
          'demo': 20, 'demo-default-query': 20}


class AuditError(ValueError):
    pass


def need(value, message):
    if not value:
        raise AuditError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(AuditError('nonfinite JSON')))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def pins(items):
    need(type(items) is dict and bool(items), 'nonempty immutable pins required')
    for filename, digest in items.items():
        need(sha(filename) == digest, 'artifact/source changed: ' + filename)


def local_file(directory, filename, digest):
    path = (Path(directory) / filename).resolve()
    need(path.is_relative_to(Path(directory).resolve()), 'artifact escapes run directory')
    need(sha(path) == digest, 'artifact checksum: ' + filename)
    return path


def vector(values):
    need(type(values) is list and len(values) == 16 and
         all(type(v) in (int, float) and math.isfinite(v) for v in values), 'finite16 scores required')
    return values


def raw_order(row):
    return sorted(range(16), key=lambda i: (-row['scores16'][i], i))


def index_rows(rows, profiles, *, both_orders=True):
    need(type(rows) is list, 'prediction array required')
    by = {p['profileId']: p for p in profiles}
    need(len(by) == len(profiles), 'duplicate public profiles')
    expected = {(pid, order) for pid in by for order in (('canonical', 'reversed') if both_orders else ('canonical',))}
    result = {}
    for row in rows:
        key = (row['profileId'], row['orderLabel'])
        need(key not in result and key in expected, 'duplicate/unexpected prediction identity')
        need(row['userGroupId'] == by[key[0]]['userGroupId'], 'prediction group join mismatch')
        for field in ('sourceProfileId','controlName'):
            if field in by[key[0]]: need(row.get(field) == by[key[0]][field],'prediction control/source metadata changed')
        if 'caseId' in row: need(row['caseId'] == key[0]+'-'+key[1],'order-specific case identity changed')
        vector(row['scores16'])
        order = row['order']
        need(type(order) is list and len(set(order)) == len(order) and
             all(type(i) is int and 0 <= i < 16 for i in order), 'invalid guarded order')
        need(row['rawLearnedOrder'] == raw_order(row), 'scores/raw order mismatch')
        result[key] = row
    need(set(result) == expected, 'fixed prediction denominator/roster missing')
    return result


def softmax(values):
    values = vector(values)
    exps = [math.exp(v - max(values)) for v in values]
    return [v / sum(exps) for v in exps]


def fuse(source, p, v5):
    need(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1, 'invalid query probability')
    return [(1-p)*a + p*(b + .05*a) for a, b in zip(softmax(source), vector(v5), strict=True)]


def close(a, b):
    return len(a) == len(b) and all(math.isclose(x, y, abs_tol=2e-6, rel_tol=2e-6) for x, y in zip(a, b, strict=True))


def checkpoint_tensors(path, schedule):
    """Read the two small F32 arrays directly; no MLX/Torch model import."""
    raw = Path(path).read_bytes(); need(len(raw) >= 8,'short safetensors checkpoint')
    size = struct.unpack('<Q',raw[:8])[0]
    need(0 < size <= len(raw)-8,'bad safetensors header length')
    header = json.loads(raw[8:8+size]); header.pop('__metadata__',None)
    need(set(header) == {'lora_a','lora_b'},'checkpoint exactA/B tensor roster')
    offsets = []
    for name,shape in [('lora_a',[12288,4]),('lora_b',[4,4096])]:
        entry = header[name]; start,end = entry['data_offsets']
        need(entry['dtype'] == 'F32' and entry['shape'] == shape and
             type(start) is int and type(end) is int and 0 <= start < end <= len(raw)-8-size and
             end-start == math.prod(shape)*4,'checkpoint tensor dtype/shape/range')
        tensor = raw[8+size+start:8+size+end]
        values = [v[0] for v in struct.iter_unpack('<f',tensor)]
        need(all(math.isfinite(v) for v in values) and any(v != 0 for v in values),'nonfinite/unlearned checkpoint array')
        need(hashlib.sha256(tensor).hexdigest() != schedule['initialAdapterSha256'][name],'serialized adapter did not change')
        offsets.append((start,end))
    need(sorted(offsets) == [(0,sorted(offsets)[0][1]),(sorted(offsets)[0][1],len(raw)-8-size)],'checkpoint tensor overlap/trailing data')
    zero_b = hashlib.sha256(bytes(4*4096*4)).hexdigest()
    need(schedule['initialAdapterSha256']['lora_b'] == zero_b,'initialB was not truezero')


def exact_gate(gate):
    need(gate['status'] == 'exact-gate-passed' and gate['sourceStatePreserved'] is True and
         gate['adapterPreserved'] is True and bool(gate['comparisons']) and
         all(c['exact'] is True and c['routeLogitsExact'] is True for c in gate['comparisons'].values()), 'native exact gate failed')


def verify_adapter(run):
    """Hash every actual gate/feature, schedule, base927 and postfit receipts."""
    run = Path(run)
    report = read(run/'report.json'); plan = read(run/'plan.json')
    freeze = read(run/'ALL_PREDICTIONS_FROZEN.json')
    need(report['status'] == 'completed-fixed-fit-and-predictions' and report['actualUpdates'] == 3584, 'actual3584 update lifecycle incomplete')
    need(report['privateEvaluationLabelsParsed'] is False and freeze['privateEvaluationLabelsParsed'] is False, 'adapter gold admission already violated')
    need(report['planSha256'] == sha(run/'plan.json'), 'adapter plan changed')
    need(plan['expectedCounts'] == SPLITS, 'predeclared1240 counts changed')
    rows = plan['rows']
    need(Counter(r['split'] for r in rows) == SPLITS and len({r['caseId'] for r in rows}) == 1240, 'plan fixed roster')
    pins(plan['sha256'])
    need(report['frozen927HashesUnchanged'] is True and report['shutdownCompleted'] is True and report['sourcePinsUnchanged'] is True, 'base/source/cleanup incomplete')
    need(len(report['frozen927Before']) == 927 and report['frozen927Before'] == report['frozen927After'], 'raw927 hash equality failed')
    need(Counter(r['status'] for r in report['featureRoster']) == {'completed': 1240} and
         {r['caseId'] for r in report['featureRoster']} == {r['caseId'] for r in rows}, 'failed/unattempted feature rows')
    extraction = read(run/'EXTRACTION_FROZEN.json')
    need(extraction['status'] == 'completed-choice-feature-extraction' and extraction['allGatesExact'] is True and
         extraction['frozen927HashesUnchanged'] is True and len(extraction['rows']) == 1240, 'incomplete exact extraction')
    need(extraction['successfulExactGateSuffixForwards'] == 2480, 'successful extraction gate forward count')
    by = {r['caseId']: r for r in rows}
    need({r['caseId'] for r in extraction['rows']} == set(by), 'extraction roster changed')
    for entry in extraction['rows']:
        row = by[entry['caseId']]
        need(all(entry[k] == row[k] for k in ('split', 'profileId', 'order', 'targetPurpose', 'targetWeights', 'routeTokenIds')), 'feature target/public join changed')
        for key in ('archive', 'metadata', 'gate'):
            local_file(run, entry[key], entry[key+'Sha256'])
        gate = read(run/entry['gate'])
        exact_gate(gate)
        metadata = read(run/entry['metadata'])
        need(all(metadata[k] == row[k] for k in ('caseId','targetWeights','routeTokenIds')) and metadata['singleChoiceGradientGatePassed'] is True,'serialized feature target identity')
        if row['split'] != 'train':
            need(row['targetPurpose'] == 'arbitrary-first-route-numerical-control-not-evaluation-gold' and row['targetWeights'] == [1.]+[0.]*15,'nonTRAIN targets contain quality gold')
    fitted = read(run/'fit/report.json'); schedule = read(run/'fit/FIT_FROZEN.json')
    need(fitted['actualUpdates'] == 3584 and fitted['status'] == 'completed-feature-fit' and fitted['reloadExact'] is True and
         fitted['allFitPinsUnchanged'] is True and set(fitted['changedAdapterTensors']) == {'lora_a', 'lora_b'}, 'real update/reload proof incomplete')
    train = [r for r in rows if r['split'] == 'train']
    need(len({r['profileId'] for r in train}) == 448 and all(r['targetPurpose'] == 'maximum-grade-TRAIN-supervision' for r in train), 'TRAIN-only supervision')
    need(schedule['TRAINCases'] == [r['caseId'] for r in train] and schedule['developmentGoldParsed'] is False, 'optimizer TRAIN roster')
    expected = []
    for epoch in range(4):
        order = list(range(896)); random.Random(schedule['recipe']['seed']+epoch).shuffle(order); expected.extend(order)
    need(schedule['order'] == expected, 'fixed four-pass optimizer order changed')
    checkpoint = local_file(run, 'fit/adapter.safetensors', freeze['checkpointSha256'])
    checkpoint_tensors(checkpoint,schedule)
    need(fitted['checkpointSha256'] == sha(checkpoint), 'checkpoint/reload receipt mismatch')
    parity = read(run/'inference-parity.json')
    need(parity['adapterSha256'] == sha(checkpoint) and parity['cachedScoringMode'] == 'plain' and
         parity['dynamicMode'] == 'gradient-public-XHY-last16' and all(parity[k] is True for k in
         ('cachedScoreMatchesTrainingFeaturesExact', 'dynamicMatchesCachedFeatureScoresExact', 'zeroAndNonzeroControlPassed', 'learnedPostfitPassed')), 'target-free inference certificate missing')
    controls = read(run/'POSTFIT_ORIGINAL_ARCHIVE_PARITY.json')['records']
    need(len(controls) == 4 and Counter(c['split'] for c in controls) == {'train': 2, 'evaluation-main': 2}, 'four original-archive controls required')
    original_entries = {r['caseId']:r for r in extraction['rows']}
    chosen = [r for r in extraction['rows'] if r['split'] == 'train'][:2]+[r for r in extraction['rows'] if r['split'] == 'evaluation-main'][:2]
    need([c['caseId'] for c in controls] == [r['caseId'] for r in chosen],'postfit controls differ from fixed first two TRAIN/public rows')
    for i, control in enumerate(controls):
        need(all(control[k] is True for k in ('freshNativeVsOriginalArchiveExact','cacheBaseVsLearnedAdapterExact','targetFreeLearnedInferenceExact')), 'learned inference/native control failed')
        path = local_file(run, f'POSTFIT_INFERENCE_CONTROL_{i}.json', control['inferenceProofSha256'])
        proof = read(path)
        need(proof['cachedPlainScoreMatchesTrainingFeaturesExact'] is True and proof['labelFreeGradientPathExact'] is True, 'actual target-free receipt failed')
        postfit = read(local_file(run, f'POSTFIT_NATIVE_GATE_{i}.json', control['gateSha256'])); exact_gate(postfit)
        original = original_entries[control['caseId']]
        need(control['originalArchiveSha256'] == original['archiveSha256'] and
             postfit['featureTensors'] == read(run/original['gate'])['featureTensors'],'learned native controls not bound to original features')
    files = {'learned': ('learned-nonfit-predictions.json', freeze['learnedPredictionsSha256']),
             'frozen-base-lexical-decision': ('frozen-base-nonfit-predictions.json', freeze['frozenBasePredictionsSha256'])}
    for public, name in [('TRAIN-popularity','training-popularity'), ('latest-owned-repeat','latest-owned-repeat')]:
        files[public] = (name+'-nonfit-predictions.json', freeze['baselineFilesSha256'][name])
    profiles = list({r['profileId']: {'profileId':r['profileId'], 'userGroupId':r['userGroupId'], 'request':r['publicRequest'],
                                    'sourceProfileId':r.get('sourceProfileId'),'controlName':r.get('controlName')} for r in rows if r['split'] != 'train'}.values())
    need(len(profiles) == 172 and freeze['nonfitPromptRecords'] == 344 and freeze['canonicalIsPrimary'] is True and freeze['reversedRobustnessOnly'] is True, 'complete nonfit prediction freeze required')
    outputs = {}
    for model, (filename, digest) in files.items():
        outputs[model] = read(local_file(run, filename, digest)); index_rows(outputs[model], profiles)
    # Independently reconstruct simple controls and frozen-B0 canonical binding.
    prediction_maps = {name:{r['caseId']:r for r in values} for name,values in outputs.items()}
    ids = [c['productId'] for c in next(r for r in train if r['order'] == 'canonical')['choices']]
    prior = {pid:0. for pid in ids}
    canonical_train = [r for r in train if r['order'] == 'canonical']
    need(len(canonical_train) == 448, 'prior must use448 distinct TRAIN profiles')
    for row in canonical_train:
        need(math.isclose(sum(row['targetWeights']),1,abs_tol=1e-12), 'TRAIN target distribution')
        for choice,weight in zip(row['choices'],row['targetWeights'],strict=True): prior[choice['productId']] += weight/448
    for entry in extraction['rows']:
        if entry['split'] == 'train': continue
        row = by[entry['caseId']]; choices = [c['productId'] for c in row['choices']]
        base = dict(zip(choices,entry['frozenBaseRouteScores'],strict=True))
        need(close(prediction_maps['frozen-base-lexical-decision'][row['caseId']]['scores16'],[base[pid] for pid in ids]), 'frozen base scores changed from native archive')
        need(close(prediction_maps['TRAIN-popularity'][row['caseId']]['scores16'],[prior[pid] for pid in ids]), 'TRAIN popularity not reconstructed')
        recent = dict(prior)
        purchases = [e for e in row['publicRequest']['publicHistory'] if e['eventType'] == 'purchase' and e.get('productId') in prior]
        if purchases: recent[max(purchases,key=lambda e:e['occurredAt'])['productId']] += 10
        need(close(prediction_maps['latest-owned-repeat'][row['caseId']]['scores16'],[recent[pid] for pid in ids]), 'latest-owned baseline not reconstructed')
    public_cache = read(run/'public-features/index.json')
    need(public_cache['teacherOrTargetsIncluded'] is False and len(public_cache['entries']) == 20,'exact20 target-free public demo cache')
    allowed = {'archive','archiveSha256','dtypes','routeTokenIds','promptTokenIdsSha256','promptTokens','prefixPath','nativeInputSuffixTokens'}
    for key,entry in public_cache['entries'].items():
        need(set(entry) == allowed and entry['promptTokenIdsSha256'] == key and entry['nativeInputSuffixTokens'] == 16,'public cache metadata includes identifiers/supervision')
        path = local_file(run/'public-features',entry['archive'],entry['archiveSha256'])
        with zipfile.ZipFile(path) as archive:
            need(set(archive.namelist()) == {'x.npy','h.npy','y.npy'},'public cache contains target/teacher/dummy arrays')
    return {'report':report, 'plan':plan, 'freeze':freeze, 'models':outputs,
            'checkpointSha256':sha(checkpoint), 'profiles':profiles}


def verify_combined(run, adapter, public_profiles):
    """Producer manifest hashes every model/query artifact before private parse."""
    run = Path(run); freeze = read(run/'ALL_MODELS_PREDICTIONS_FROZEN.json')
    need(freeze['goldParsed'] is False and freeze['independentPacketManifestSha256'] == PACKET_SHA, 'global gold admission barrier')
    need(all(freeze['stagesCompleted'].get(k) is True for k in ('adapter','queryGate','allControls')), 'all models must finish before gold')
    need(freeze['adapterCheckpointSha256'] == adapter['checkpointSha256'] and set(freeze['models']) == MODELS, 'combined adapter/control identity')
    pins(freeze['sha256'])
    need(freeze.get('gateCheckpointSha256') in freeze['sha256'].values(), 'query head not pinned')
    models = {}; all_profiles = adapter['profiles']
    for name, item in freeze['models'].items():
        rows = read(local_file(run, item['file'], item['sha256']))
        profiles = all_profiles if len(rows) == 344 else public_profiles
        index_rows(rows, profiles); models[name] = rows
    need({len(v) for v in models.values()} in ({264}, {344}), 'same complete combined model roster required')
    raw = {name: {(r['profileId'],r['orderLabel']):r for r in rows} for name,rows in adapter['models'].items()}
    candidate = {(r['profileId'],r['orderLabel']):r for r in models['learned']}
    for name, rows in models.items():
        for row in rows:
            key = (row['profileId'],row['orderLabel']); source = raw[name][key]
            need(close(vector(row['rawSourceScores16']), source['scores16']), 'fused source differs from frozen raw control')
            need(close(row['scores16'], fuse(row['rawSourceScores16'],row['pSpecific'],row['v5Scores16'])), 'independent fusion arithmetic mismatch')
            primary = candidate[key]
            need(row['pSpecific'] == primary['pSpecific'] and row['v5Scores16'] == primary['v5Scores16'], 'control borrowed different gate/v5 scores')
            need(row['fusionPolicy']['gateCheckpointSha256'] == freeze['gateCheckpointSha256'], 'wrong query head in fusion')
    for profile in profiles:
        a,b = (candidate[(profile['profileId'],order)] for order in ('canonical','reversed'))
        need(a['pSpecific'] == b['pSpecific'] and a['v5Scores16'] == b['v5Scores16'],'catalog order changed query-only branch')
    # Input ablations retain query/budget/guards. The same actual query branch
    # must therefore also be used across natural and transformed profiles.
    for profile in public_profiles:
        if 'controlName' not in profile: continue
        for order in ('canonical','reversed'):
            row = candidate[(profile['profileId'],order)]; source = candidate[(profile['sourceProfileId'],order)]
            need(row['pSpecific'] == source['pSpecific'] and row['v5Scores16'] == source['v5Scores16'], 'input ablation changed query branch')
    return {'freeze':freeze, 'models':models}


def history_metrics(rows, public, labels, products):
    """Known, previously inspected DEVELOPMENT intervention, never unseen gold."""
    ids = [p['productId'] for p in products]
    profiles = [{**pair[side],'userGroupId':pair['pairId']} for pair in public['pairs'] for side in ('profileA','profileB')]
    subset = [r for r in rows if r['profileId'] in {p['profileId'] for p in profiles}]
    indexed = index_rows(subset,profiles)
    gold = {r['pairId']:r for r in labels}
    need(len(public['pairs']) == len(gold) == 10 and set(gold) == {p['pairId'] for p in public['pairs']}, 'fixed10 DEV history pair join')
    results = []
    for pair in public['pairs']:
        label = gold[pair['pairId']]; changed = ids.index(label['changedPurchaseProductId'])
        a,b = (indexed[(pair[side]['profileId'],'canonical')] for side in ('profileA','profileB'))
        rank_a,rank_b = (raw_order(r).index(changed)+1 for r in (a,b))
        top = ids[b['order'][0]] if b['order'] else None
        after = label['profileB']
        results.append({'pairId':pair['pairId'],'rawRankBefore':rank_a,'rawRankAfter':rank_b,
                        'rawOwnedPriorityDemoted':rank_b > rank_a,
                        'novelRelevantAfterTop1':bool(top and top not in after['alreadyOwnedProductIds'] and after['relevanceGrades'][top] >= 2)})
    demotions = sum(r['rawOwnedPriorityDemoted'] for r in results)
    novel = sum(r['novelRelevantAfterTop1'] for r in results)
    return {'developmentOnly':True,'fixedPairs':10,'rawOwnedPriorityDemotions':demotions,
            'novelRelevantAfterTop1':novel,'pass':demotions >= 8 and novel >= 8,'rows':results}


def effective_constraints(request):
    bounds = []
    if request.get('itemBudget') is not None: bounds.append((request['itemBudget'],False))
    pattern = r'\b(under|below|less than|up to|at most|no more than|max(?:imum)?(?: budget)?|budget(?: of| is)?)\s*:?\s*\$\s*(\d+(?:\.\d+)?)'
    for match in re.finditer(pattern, request['query'], re.I):
        bounds.append((float(match[2]), match[1].lower() in ('under','below','less than')))
    bound = min(bounds, key=lambda v:(v[0],not v[1])) if bounds else (None,False)
    return bound, request.get('hardRequirements',[])


def public_controls(main, controls, manifest):
    need(len(main) == 32 and len(controls) == 96,'fixed public control counts')
    by = {p['profileId']:p for p in main}
    need(Counter(c['controlName'] for c in controls) == {n:32 for n in CONTROL_NAMES.values()},'all three input control rosters required')
    for control in controls:
        source = by[control['sourceProfileId']]; expected = deepcopy(source['request'])
        need(control['userGroupId'] == source['userGroupId'],'control changed source group')
        if control['controlName'] == 'owned-only-public-context':
            expected['persona'] = {'background':[],'colorPreferences':[]}
            expected['publicHistory'] = [e for e in expected['publicHistory'] if e['eventType'] == 'purchase']
        elif control['controlName'] == 'purchase-history-removed':
            expected['publicHistory'] = [e for e in expected['publicHistory'] if e['eventType'] != 'purchase']
        else:
            donor = by[manifest['shuffledProfileDonors'][source['profileId']]]
            need(donor['userGroupId'] != source['userGroupId'],'shuffle donor same matched pair')
            expected['persona'] = donor['request']['persona']; expected['publicHistory'] = donor['request']['publicHistory']
        need(control['request'] == expected,'input control changed query/guards or wrong context transformation')


def facts(rows, profiles, products):
    """All eligible product positions, not just a convenient first prediction."""
    by = {p['profileId']:p for p in profiles}; violations = []
    for row in rows:
        request = by[row['profileId']]['request']; (budget,exclusive), hard = effective_constraints(request)
        for index in row['order']:
            product = products[index]; reasons = []
            if budget is not None and (product['itemPrice'] >= budget if exclusive else product['itemPrice'] > budget): reasons.append('budget')
            for h in hard:
                attr,value = h['attribute'],h['value']
                supported = (any(c['colorName'] == value for c in product['colorVariants']) if attr == 'color' else
                             product['category'] == value if attr == 'category' else False)
                if not supported: reasons.append('unsupported-or-unsatisfied-'+attr)
            colors = [h['value'] for h in hard if h['attribute'] == 'color']
            if colors and not any(all(c['colorName'] == v for v in colors) for c in product['colorVariants']): reasons.append('no-single-source-variant')
            if reasons: violations.append({'profileId':row['profileId'],'productId':product['productId'],'reasons':reasons})
    return {'violationCount':len(violations), 'violations':violations,
            'scope':'source product item prices and typed eligibility; selected variants/images/copy remain serving-pending',
            'selectedVariantImageCopyVerified':False}


def metrics(rows, profiles, labels, products, denominator, *, replacement=False):
    """Canonical or reversed independently; controls join original source gold."""
    need(len(rows) == len(profiles) == len(labels) == denominator, 'fixed quality denominator')
    index_rows(rows, profiles, both_orders=False)
    gold = {r['profileId']:r for r in labels}; ids = [p['productId'] for p in products]
    need(len(gold) == denominator and set(gold) == {p['profileId'] for p in profiles}, 'exact private/public label join')
    groups = defaultdict(list); result = []
    public = {p['profileId']:p for p in profiles}
    for row in rows:
        label = gold[row['profileId']]; grades = label['relevanceGrades']; primary = label['primaryProductId']
        need(set(grades) == set(ids) and all(type(g) is int and 0 <= g <= 3 for g in grades.values()) and
             primary in ids and grades[primary] == 3 and sum(g == 3 for g in grades.values()) == 1, 'bounded16 grades/unique primary')
        owned = {e['productId'] for e in public[row['profileId']]['request']['publicHistory'] if e['eventType'] == 'purchase' and e.get('productId') in ids}
        need(owned == set(label['alreadyOwnedProductIds']), 'private ownership differs from observed public purchases')
        if not replacement: need(all(grades[p] == 0 for p in owned), 'core addition gold has positive owned utility')
        order = [ids[i] for i in row['order']]; top = order[0] if order else None
        gain = lambda seq:sum((2**g-1)/math.log2(i+2) for i,g in enumerate(seq))
        detail = {'profileId':row['profileId'], 'userGroupId':row['userGroupId'],
                  'topProductId':top,'primaryTop1':top == primary,
                  'novelRelevantTop1':bool(top and top not in owned and grades[top] >= 2),
                  'strongPositiveTop3':any(grades[p] >= 2 for p in order[:3]),
                  'ownedTop1':top in owned,
                  'ndcg6':gain([grades[p] for p in order[:6]])/gain(sorted(grades.values(),reverse=True)[:6])}
        result.append(detail); groups[row['userGroupId']].append(detail)
    need(len(groups) == (4 if replacement else 16) and all(len(v) == (1 if replacement else 2) for v in groups.values()), 'whole-session matched group denominator')
    return {'fixedDenominator':denominator,'groups':len(groups),
            **{key:sum(r[key] for r in result) for key in ('primaryTop1','novelRelevantTop1','strongPositiveTop3','ownedTop1')},
            'ndcg6':sum(r['ndcg6'] for r in result)/denominator,
            'perGroup':{k:sum(r['primaryTop1'] for r in v) for k,v in groups.items()},'rows':result}


def choose(rows, profiles, order='canonical', control=None):
    wanted = {p['profileId'] for p in profiles}; selected = []
    for row in rows:
        if row['orderLabel'] != order: continue
        if control:
            if row.get('controlName') != CONTROL_NAMES[control] or row.get('sourceProfileId') not in wanted: continue
            row = {**row,'profileId':row['sourceProfileId'], 'orderLabel':'canonical'}
        else:
            if row['profileId'] not in wanted: continue
            row = {**row,'orderLabel':'canonical'}
        selected.append(row)
    return selected


def evaluate_pipeline(models, main, replacement, main_labels, replacement_labels, products):
    core = {}; controls = {}; replacement_result = {}
    for name, rows in models.items():
        core[name] = metrics(choose(rows,main),main,main_labels,products,32)
        replacement_result[name] = metrics(choose(rows,replacement),replacement,replacement_labels,products,4,replacement=True)
    for name in CONTROL_NAMES:
        controls[name] = metrics(choose(models['learned'],main,control=name),main,main_labels,products,32)
    candidate = core['learned']; comparisons = {name:(core[name] if name in core else controls[name])['primaryTop1'] for name in STRICT}
    natural = choose(models['learned'],main)+choose(models['learned'],replacement)
    factual = facts(natural,main+replacement,products)
    passed = (candidate['primaryTop1'] >= 20 and candidate['novelRelevantTop1'] >= 24 and
              candidate['strongPositiveTop3'] >= 28 and factual['violationCount'] == 0 and
              replacement_result['learned']['primaryTop1'] >= 3 and
              all(candidate['primaryTop1'] > value for value in comparisons.values()))
    reversed_metrics = metrics(choose(models['learned'],main,'reversed'),main,main_labels,products,32)
    return {'main':core,'inputControls':controls,'replacement':replacement_result,
            'reversedRobustnessOnly':reversed_metrics, 'strictBaselinePrimaryCounts':comparisons,
            'facts':factual,'offlineCoreAndReplacementPass':passed}


def run(adapter_run, combined_run, packet_dir, out):
    out = Path(out); need(not out.exists(),'new output directory required'); out.mkdir(parents=True)
    receipt = {'version':VERSION,'status':'admitting','privateQualityJSONParsed':False,
               'sourceSha256':sha(__file__),'gateSha256':GATE_SHA,'overallPromotionPass':False}
    try:
        frozen = read(HERE/'PERSONA_CHOICE_COMBINED_ADJUDICATOR_FROZEN_V1.json')
        pins(frozen['sha256'])
        need(sha(HERE/'PERSONA_CHOICE_COMBINED_PROMOTION_GATE_V2.json') == GATE_SHA,'root frozen gate changed')
        packet_dir = Path(packet_dir); need(sha(packet_dir/'MANIFEST.json') == PACKET_SHA,'packet identity changed')
        manifest = read(packet_dir/'MANIFEST.json')
        main = read(local_file(packet_dir,'main.public.json',manifest['files']['main.public.json']))['profiles']
        replacement = read(local_file(packet_dir,'replacement.public.json',manifest['files']['replacement.public.json']))['profiles']
        controls = read(local_file(packet_dir,'controls.public.json',manifest['files']['controls.public.json']))['profiles']
        need((len(main),len(replacement),len(controls)) == (32,4,96),'public packet denominator')
        public_controls(main,controls,manifest)
        adapter = verify_adapter(adapter_run)
        public_by_id = {p['profileId']:p for p in main+replacement+controls}
        evaluation_rows = [r for r in adapter['plan']['rows'] if r['split'].startswith('evaluation-')]
        need(len(evaluation_rows) == 264,'adapter evaluation prompt count')
        for row in evaluation_rows:
            public = public_by_id[row['profileId']]
            need(row['publicRequest'] == public['request'] and row['userGroupId'] == public['userGroupId'] and
                 all(row.get(k) == public.get(k) for k in ('controlName','sourceProfileId')),'adapter prompt differs from independent frozen public request')
        combined = verify_combined(combined_run,adapter,main+replacement+controls)
        for path,digest in manifest['catalogSha256'].items(): need(sha(HERE.parents[2]/path) == digest,'catalog pin changed')
        inputs = read(HERE/'store_catalog_v1/model_inputs.json')['products']
        mapping = read(HERE/'store_catalog_v1/external_mapping.json')['products']
        products = [{**p,'productId':m['productId']} for p,m in zip(inputs,mapping,strict=True)]
        # Durable admission receipt written BEFORE the only private JSON loads.
        write(out/'ADMISSION_PASSED_BEFORE_GOLD.json',{'adapterCheckpointSha256':adapter['checkpointSha256'],
              'adapterFreezeSha256':sha(Path(adapter_run)/'ALL_PREDICTIONS_FROZEN.json'),
              'combinedFreezeSha256':sha(Path(combined_run)/'ALL_MODELS_PREDICTIONS_FROZEN.json'),
              'privateQualityJSONParsed':False,'packetManifestSha256':PACKET_SHA})
        main_labels = read(local_file(packet_dir,'main.labels.private.json',manifest['files']['main.labels.private.json']))['labels']
        replacement_labels = read(local_file(packet_dir,'replacement.labels.private.json',manifest['files']['replacement.labels.private.json']))['labels']
        receipt['privateQualityJSONParsed'] = True
        receipt['raw'] = evaluate_pipeline(adapter['models'],main,replacement,main_labels,replacement_labels,products)
        receipt['combined'] = evaluate_pipeline(combined['models'],main,replacement,main_labels,replacement_labels,products)
        known = HERE/'synthetic_history_v1'
        history_public = read(known/'history-only-probes.public.json')
        history_labels = read(known/'history-only-probes.labels.json')['labels']
        receipt['knownDevelopmentHistory'] = history_metrics(adapter['models']['learned'],history_public,history_labels,products)
        if len(combined['models']['learned']) == 344:
            receipt['combinedKnownDevelopmentHistory'] = history_metrics(combined['models']['learned'],history_public,history_labels,products)
        receipt.update(status='completed-offline-adjudication',
             offlineCoreAndReplacementPass=receipt['raw']['offlineCoreAndReplacementPass'] and receipt['combined']['offlineCoreAndReplacementPass'] and receipt['knownDevelopmentHistory']['pass'],
             pending=['query32x10 regression review',
                      'query linear head/vector provenance replay','ten public demo coherence',
                      'selected variants/images/fresh copy source facts','live browser and useful first3/full6 readiness'],
             limitations='Hash/receipt validation of native full-LM parity, not independent CPU reproduction of quantized native projection. Offline pass alone never promotes.')
        pins(adapter['plan']['sha256']); pins(combined['freeze']['sha256'])
    except Exception as error:
        receipt.update(status='failed-admission-or-audit',error={'type':type(error).__name__,'message':str(error)})
    write(out/'ADJUDICATION.json',receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--adapter-run',type=Path,required=True); parser.add_argument('--combined-run',type=Path,required=True)
    parser.add_argument('--packet',type=Path,default=HERE/'persona_choice_independent_eval_v3')
    parser.add_argument('--out',type=Path,required=True); args = parser.parse_args(argv)
    result = run(args.adapter_run,args.combined_run,args.packet,args.out)
    print(json.dumps({'status':result['status'],'overallPromotionPass':False,'out':str(args.out)}))
    return 2 if result['status'] == 'completed-offline-adjudication' else 1


if __name__ == '__main__':
    raise SystemExit(main())
