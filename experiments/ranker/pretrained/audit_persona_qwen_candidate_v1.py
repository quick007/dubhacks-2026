"""Independent CPU adjudication. Never optimize, load MLX, or open gold early.

Fresh labels are opened only after complete frozen prediction/checkpoint proofs,
native receipt validation and exact CPU score replay. Serving acceptance stays
pending; this script cannot approve a storefront or claim physical readiness.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
VERSION = 'persona-qwen-independent-adjudicator.v1'
GATE_SHA256 = '89e243b2444eea4516b420491c1c70a18a5b115a686d1fc0f1324f4cd061aa4f'
PREFLIGHT_SHA256 = '71c9cabc533083e726cfd5938fed9e2cacf88e4ef841b102b4dc3a73ba619c35'
PRIMARY_SPLITS = {'development-replay': 40, 'demo-nonfit': 10,
                  'demo-default-query-nonfit': 10, 'history-diagnostic-nonfit': 20,
                  'independent-evaluation': 32}
BASELINES = {'ownership-term-disabled', 'zero-context-owned-only', 'shuffled-context',
             'training-popularity', 'most-recent-owned-repeat'}
DOMINANCE = {'training-popularity', 'most-recent-owned-repeat',
             'zero-context-owned-only', 'shuffled-context'}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for k, v in items:
            require(k not in out, 'Duplicate JSON key')
            out[k] = v
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(AuditError('Nonfinite JSON')))


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def exact_rows(rows, profiles, context):
    require(len({p['profileId'] for p in profiles}) == len(profiles), context + ': duplicate public profile')
    require(type(rows) is list and len(rows) == len(profiles), context + ': fixed denominator missing')
    require(len({r['profileId'] for r in rows}) == len(rows), context + ': duplicate prediction')
    expected = {p['profileId']: p for p in profiles}
    require({r['profileId'] for r in rows} == set(expected), context + ': wrong prediction roster')
    for row in rows:
        require(row['userGroupId'] == expected[row['profileId']]['userGroupId'], context + ': group changed')
        scores = row['scores16']
        require(type(scores) is list and len(scores) == 16 and all(type(v) in (int, float) and math.isfinite(v) for v in scores), context + ': invalid scores')
        order = row['order']
        require(type(order) is list and all(type(i) is int and 0 <= i < 16 for i in order) and len(set(order)) == len(order), context + ': invalid order')
        require(type(row['selected']) is list and len(row['selected']) == min(6, len(order)), context + ': card denominator')
        if 'rawLearnedOrder' in row:
            require(row['rawLearnedOrder'] == sorted(range(16), key=lambda i: (-scores[i], i)), context + ': raw score/rank mismatch')
    return {r['profileId']: r for r in rows}


def require_prediction_freeze(run, packet):
    """Pure JSON admission barrier. Does not open or hash any private label."""
    run = Path(run)
    freeze = read(run / 'PREDICTIONS_FROZEN.json')
    require(freeze.get('independentGoldParsed') is False, 'Prediction freeze must precede independent gold')
    require(set(freeze.get('files', {})) == set(PRIMARY_SPLITS), 'All five primary prediction files required')
    require(set(freeze.get('baselineFiles', {})) == BASELINES, 'All five baseline files required before gold')
    require(sha(run / 'head.pt') == freeze['checkpointSha256'], 'Checkpoint freeze digest mismatch')
    outputs = {}; baselines = {}; prediction_paths = []
    for split, n in PRIMARY_SPLITS.items():
        profiles = [r['publicProfile'] for r in packet['rows'] if r['split'] == split]
        require(len(profiles) == n, 'Frozen split denominator changed: ' + split)
        path = run / (split + '-predictions.json')
        require(sha(path) == freeze['files'][split], 'Prediction digest mismatch: ' + split)
        outputs[split] = read(path); exact_rows(outputs[split], profiles, split)
        prediction_paths.append(path)
    profiles = [r['publicProfile'] for r in packet['rows'] if r['split'] == 'independent-evaluation']
    for name in sorted(BASELINES):
        path = run / ('independent-' + name + '-predictions.json')
        require(sha(path) == freeze['baselineFiles'][name], 'Baseline digest mismatch: ' + name)
        baselines[name] = read(path); exact_rows(baselines[name], profiles, name)
        prediction_paths.append(path)
    require((run / 'PREDICTIONS_FROZEN.json').stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in [*prediction_paths, run / 'head.pt']), 'Freeze file precedes its artifacts')
    return freeze, outputs, baselines


def raw_order(row):
    return sorted(range(16), key=lambda i: (-row['scores16'][i], i))


def metrics(rows, labels, products, profiles):
    """Independent integer counts; no producer evaluator or /40 rescaling."""
    exact_rows(rows, profiles, 'independent32')
    require(len(rows) == len(labels) == 32, 'Exactly32 independent labels and predictions required')
    gold = {r['profileId']: r for r in labels}
    require(len(gold) == 32 and set(gold) == {r['profileId'] for r in rows}, 'Exact32 label join required')
    ids = [p['productId'] for p in products]; results = []; groups = defaultdict(list)
    for row in rows:
        label = gold[row['profileId']]; grades = label['relevanceGrades']
        require(set(grades) == set(ids) and all(type(g) is int and 0 <= g <= 3 for g in grades.values()), 'All16 bounded gold grades required')
        require(label['primaryProductId'] in ids and grades[label['primaryProductId']] == 3 and sum(v == 3 for v in grades.values()) == 1, 'Exactly one primary grade3 required')
        owned = set(label['alreadyOwnedProductIds'])
        require(all(grades[p] == 0 for p in owned), 'Owned gold cannot have positive utility')
        ordered = [ids[i] for i in row['order']]; raw = [ids[i] for i in raw_order(row)]
        top = ordered[0] if ordered else None
        gain = [2 ** grades[p] - 1 for p in ordered[:6]]
        ideal = sorted([2 ** v - 1 for v in grades.values()], reverse=True)[:6]
        dcg = lambda values: sum(g / math.log2(i + 2) for i, g in enumerate(values))
        r = {'profileId': row['profileId'], 'userGroupId': row['userGroupId'], 'pairId': label['pairId'],
             'topProductId': top, 'topGrade': grades.get(top, 0),
             'primaryTop1': top == label['primaryProductId'],
             'rawPrimaryTop1': raw[0] == label['primaryProductId'],
             'novelRelevantTop1': bool(top and grades[top] >= 2 and top not in owned),
             'strongPositiveTop3': any(grades[p] >= 2 for p in ordered[:3]),
             'ownedTop1': top in owned, 'ndcg6': dcg(gain) / max(dcg(ideal), 1e-12)}
        results.append(r); groups[r['pairId']].append(r)
    require(len(groups) == 16 and all(len(v) == 2 for v in groups.values()), 'Sixteen matched pairs required')
    require(len({r['userGroupId'] for r in results}) == 16 and all(len({r['userGroupId'] for r in rs}) == 1 for rs in groups.values()), 'Sixteen independent whole matched groups required')
    return {'fixedDenominator': 32, 'groups': 16,
            'primaryTop1': sum(r['primaryTop1'] for r in results),
            'rawPrimaryTop1': sum(r['rawPrimaryTop1'] for r in results),
            'novelRelevantTop1': sum(r['novelRelevantTop1'] for r in results),
            'strongPositiveTop3': sum(r['strongPositiveTop3'] for r in results),
            'ownedTop1': sum(r['ownedTop1'] for r in results),
            'ndcg6': sum(r['ndcg6'] for r in results) / 32,
            'bothPairSidesPrimary': sum(all(r['primaryTop1'] for r in rs) for rs in groups.values()),
            'perPair': {k: {'primaryTop1': sum(r['primaryTop1'] for r in rs), 'novelRelevantTop1': sum(r['novelRelevantTop1'] for r in rs), 'meanNdcg6': sum(r['ndcg6'] for r in rs) / 2} for k, rs in groups.items()},
            'rows': results}


def history_metrics(rows, public, labels, products):
    ids = [p['productId'] for p in products]
    profiles = [{**pair[k], 'userGroupId': pair['pairId']} for pair in public['pairs'] for k in ('profileA', 'profileB')]
    by_id = exact_rows(rows, profiles, 'known DEV history20')
    require(len(public['pairs']) == len(labels) == 10, 'Ten known development pairs required')
    gold = {r['pairId']: r for r in labels}; result = []
    require(len(gold) == 10 and set(gold) == {p['pairId'] for p in public['pairs']}, 'Exact ten diagnostic label joins required')
    for pair in public['pairs']:
        label = gold[pair['pairId']]; changed = label['changedPurchaseProductId']
        a, b = (by_id[pair[k]['profileId']] for k in ('profileA', 'profileB'))
        rank_a, rank_b = (raw_order(r).index(ids.index(changed)) + 1 for r in (a, b))
        top = ids[b['order'][0]] if b['order'] else None
        after = label['profileB']
        result.append({'pairId': pair['pairId'], 'changedProductId': changed, 'rawRankBefore': rank_a,
                       'rawRankAfter': rank_b, 'ownedPriorityDemoted': rank_b > rank_a,
                       'novelRelevantAfterTop1': bool(top and top not in after['alreadyOwnedProductIds'] and after['relevanceGrades'][top] >= 2), 'topAfter': top})
    return {'fixedPairs': 10, 'scope': 'previously inspected DEVELOPMENT diagnostic',
            'ownedPriorityDemotions': sum(r['ownedPriorityDemoted'] for r in result),
            'novelRelevantAfterTop1': sum(r['novelRelevantAfterTop1'] for r in result), 'rows': result}


def effective_budget(request):
    values = []
    if request.get('itemBudget') is not None: values.append((request['itemBudget'], False))
    pattern = r'\b(under|below|less than|up to|at most|no more than)\s*\$\s*(\d+(?:\.\d+)?)'
    for m in re.finditer(pattern, request['query'], re.I):
        values.append((float(m[2]), m[1].lower() in ('under', 'below', 'less than')))
    return min(values, key=lambda v: (v[0], not v[1])) if values else (None, False)


def factual_checks(rows, profiles, catalog):
    """Independently join source facts; fresh copy/physical image readiness absent."""
    by_id = {p['profileId']: p for p in profiles}; products = catalog['products']; violations = []
    def fail(pid, reason): violations.append({'profileId': pid, 'reason': reason})
    for row in rows:
        request = by_id[row['profileId']]['request']; budget, exclusive = effective_budget(request)
        for i in row['order']:
            p = products[i]
            if budget is not None and (p['itemPrice'] >= budget if exclusive else p['itemPrice'] > budget): fail(row['profileId'], 'budget')
            for h in request.get('hardRequirements', []):
                attr, value = h['attribute'], h['value']
                supported = (any(c['colorName'] == value for c in p['colorVariants']) if attr == 'color' else
                             p['category'] == value if attr == 'category' else p['name'] == value if attr == 'name' else
                             'waterproof' in p['category'].lower() if attr == 'waterproof' and value.lower() in ('yes', 'true', 'required') else False)
                if not supported: fail(row['profileId'], 'unsupported-or-unsatisfied-hard-' + attr)
        for i, card in zip(row['order'][:6], row['selected'], strict=True):
            p = products[i]
            if card['productId'] != p['productId']: fail(row['profileId'], 'product-identity')
            variant = next((v for v in p['variants'] if v['variantId'] == card['variantId']), None)
            if variant is None:
                fail(row['profileId'], 'unknown-variant'); continue
            if card.get('selectedColor') != variant['colorName']: fail(row['profileId'], 'selected-color-fact')
            if card.get('image') != variant['images'][0]: fail(row['profileId'], 'canonical-image-binding')
            for h in request.get('hardRequirements', []):
                if h['attribute'] == 'color' and variant['colorName'] != h['value']: fail(row['profileId'], 'variant-hard-color')
            for field, expected in (('itemPrice', p['itemPrice']), ('originalName', p['name'])):
                if field in card and card[field] != expected: fail(row['profileId'], field + '-fact')
    return {'violationCount': len(violations), 'violations': violations,
            'copyFieldsVerified': False, 'physicalImagesVerified': False,
            'scope': 'source product prices, typed constraints, real selected variant/color and canonical image binding only'}


def hash_color(text, dimensions=256):
    import torch
    from torch.nn import functional as F
    result = torch.zeros(dimensions)
    for word in re.findall(r"[\w'-]+", text.casefold()):
        padded = '^' + word + '$'
        tokens = ['w:' + word] + ['c:' + padded[i:i+n] for n in (3, 4) for i in range(max(0, len(padded)-n+1))]
        for token in tokens:
            h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), 'little')
            result[h % dimensions] += 1 if h >> 63 else -1
    return F.normalize(result, dim=0)


def replay_scores(state, context, catalog, profiles, *, owned_override=None):
    import torch
    from torch.nn import functional as F
    ids = [p['productId'] for p in catalog['products']]
    owned = torch.tensor([[float(any(e['eventType'] == 'purchase' and e.get('productId') == pid for e in p['request']['publicHistory'])) for pid in ids] for p in profiles]) if owned_override is None else owned_override
    c = F.normalize(torch.tanh(F.linear(context, state['context.weight'], state['context.bias'])), dim=-1)
    p = F.normalize(F.linear(catalog['vectors'], state['product.weight']), dim=-1)
    scores = c @ p.T * state['scale'].exp().clamp(max=30) + state['ownership'] * owned
    colors = torch.stack([hash_color(v['colorName']) for v in catalog['variants']])
    color = F.normalize(torch.tanh(F.linear(context, state['color_context.weight'], state['color_context.bias'])), dim=-1) @ F.normalize(F.linear(colors, state['color_candidate.weight']), dim=-1).T * 4
    indices = torch.tensor([v['productIndex'] for v in catalog['variants']])
    return scores, scores[:, indices] + color


def verify_state(state, report, config):
    """Recreate only the seeded initialization, never an optimizer/update."""
    import torch
    from torch import nn
    with torch.random.fork_rng():
        torch.manual_seed(config['seed'])
        c = nn.Linear(4096, config['hiddenWidth'])
        p = nn.Linear(4096, config['hiddenWidth'], bias=False)
        cc = nn.Linear(4096, config['colorWidth'])
        cv = nn.Linear(config['colorDimensions'], config['colorWidth'], bias=False)
    initial = {'scale': torch.tensor(2.), 'ownership': torch.tensor(0.),
               'context.weight': c.weight.detach(), 'context.bias': c.bias.detach(),
               'product.weight': p.weight.detach(), 'color_context.weight': cc.weight.detach(),
               'color_context.bias': cc.bias.detach(), 'color_candidate.weight': cv.weight.detach()}
    require(set(state) == set(initial), 'Exact checkpoint parameter roster required')
    require(all(type(v) is torch.Tensor and v.dtype == torch.float32 and v.shape == initial[k].shape and torch.isfinite(v).all() for k,v in state.items()), 'Exact finite float32 parameter shapes required')
    changed = {k for k in state if not torch.equal(state[k], initial[k])}
    require(changed == set(report['changedTensors']) and 'ownership' in changed, 'Actual seeded parameter-change receipt mismatch')
    require(state['ownership'].item() == report['ownershipCoefficient'], 'Learned ownership value receipt mismatch')
    return sorted(changed)


def simple_baseline_scores(labels, train_profiles, products, independent):
    """Recompute TRAIN popularity and observable latest-purchase scores."""
    import torch
    ids = [p['productId'] for p in products]
    by_id = {r['profileId']: r for r in labels}
    require(len(by_id) == len(labels) == len(train_profiles) == 320 and set(by_id) == {p['profileId'] for p in train_profiles}, 'Exact320 TRAIN labels required for prior')
    targets = []
    for profile in train_profiles:
        grades = by_id[profile['profileId']]['relevanceGrades']
        require(set(grades) == set(ids) and all(type(g) is int and 0 <= g <= 3 for g in grades.values()), 'Invalid TRAIN grades for prior')
        gain = torch.tensor([2 ** grades[k] - 1 for k in ids], dtype=torch.float32)
        require(gain.sum() > 0, 'TRAIN prior requires positive utility')
        targets.append(gain / gain.sum())
    prior = torch.stack(targets).mean(0).tolist()
    recent = []
    for profile in independent:
        scores = list(prior)
        events = [e for e in profile['request']['publicHistory'] if e['eventType'] == 'purchase' and e.get('productId') in ids]
        if events: scores[ids.index(max(events, key=lambda e: e['occurredAt'])['productId'])] += 10
        recent.append(scores)
    return {'training-popularity': [list(prior) for _ in independent], 'most-recent-owned-repeat': recent}


def verify_public_demo_cache(checkpoint, run, packet, features, extraction_path):
    import torch
    path = Path(checkpoint['publicDemoFeatureCacheFile'])
    require(path.resolve() == (Path(run)/'public-demo-features.pt').resolve() and sha(path) == checkpoint['publicDemoFeatureCacheSha256'], 'Exact local public demo cache binding')
    cache = torch.load(path, map_location='cpu', weights_only=True)
    native_source = HERE/'store_persona_qwen_v1.py'
    require(cache['version'] == 'public-persona-native-context-cache.v1' and cache['teacherOrLabels'] is False and cache['featureProvenance'] == packet['featureProvenance'], 'Public-only native demo cache provenance')
    require(cache['contextFormatterSha256'] == sha(native_source) and cache['sourceExtractionReportSha256'] == sha(extraction_path), 'Public demo cache source binding')
    # Duplicate identical public texts collapse exactly as disclosed by the producer.
    expected = {r['textSha256']: r for r in packet['rows'] if r['split'] in ('demo-nonfit','demo-default-query-nonfit')}
    require(set(cache['features']) == set(expected), 'Only exact20 public demo contexts allowed in serving cache')
    for key, row in expected.items():
        saved = cache['features'][key]
        require(set(saved) == {'tensor','tokenIds','textSha256','split'} and saved['tokenIds'] == row['tokenIds'] and saved['textSha256'] == key and saved['split'] == row['split'] and torch.equal(saved['tensor'], features[row['caseId']]), 'Public cache feature/token identity differs')
    return len(expected)


def adjudicate(run, feature_root, out):
    """All fresh-label reading occurs below the explicit admission/replay barrier."""
    import torch
    from .catalog_fusion.runtime import FeatureStore
    from . import store_persona_context_v1 as source_loader
    run, feature_root, out = map(Path, (run, feature_root, out))
    require(not out.exists(), 'Unique audit output required')
    require(out.resolve().is_relative_to(HERE.parents[2]), 'Audit outputs must stay in integration worktree')
    auditor_freeze = read(HERE/'PERSONA_QWEN_ADJUDICATOR_FROZEN_V1.json')
    for name, expected in auditor_freeze['sourceSha256'].items():
        require(sha(HERE/name) == expected, 'Pre-outcome adjudicator source changed: ' + name)
    gate_path = HERE / 'PERSONA_QWEN_PROMOTION_GATE_V1.json'
    require(sha(gate_path) == GATE_SHA256, 'Frozen root promotion gate changed')
    gate = read(gate_path)
    require(set(gate['freshIndependent32']['primaryTop1StrictlyExceeds']) == DOMINANCE, 'Frozen dominance arms changed')
    require(sha(feature_root / 'preflight.json') == PREFLIGHT_SHA256 == gate['preflightSha256'], 'Reviewed preflight changed')
    packet = read(feature_root / 'preflight.json'); extraction = read(feature_root / 'report.json')
    report = read(run / 'results.json'); recipe = read(run / 'RECIPE_AND_INPUTS_FROZEN.json')
    require(report['status'] == 'completed' and report['independentStatus'] == 'completed-after-all-predictions-freeze' and report['allFitPinsUnchanged'] is True, 'Actual completed scored immutable fit required')
    # No private labels have been opened by this adjudicator at this point.
    freeze, outputs, baselines = require_prediction_freeze(run, packet)
    run_pins = {str(p.resolve()):sha(p) for p in [run/'head.pt',run/'PREDICTIONS_FROZEN.json',run/'results.json',run/'RECIPE_AND_INPUTS_FROZEN.json',feature_root/'preflight.json',feature_root/'report.json',
                 *(run/(split+'-predictions.json') for split in PRIMARY_SPLITS),
                 *(run/('independent-'+name+'-predictions.json') for name in BASELINES)]}
    for group in ('sourceSha256', 'publicFilesSha256', 'originalCatalogPins', 'annotationFilesExpectedSha256'):
        for path, expected in packet[group].items(): require(sha(path) == expected, 'Input/source pin changed: ' + path)
    for path, expected in recipe['sha256'].items(): require(sha(path) == expected, 'Fit frozen pin changed: ' + path)
    require(recipe['independentGoldParsed'] is False and recipe['config'] == packet['config'] == report['config'], 'Recipe/config chronology changed')
    require(extraction['status'] == 'completed-context-features' and extraction['frozen927HashesUnchanged'] is True and extraction['shutdownCompleted'] is True, 'Native integrity/shutdown required')
    require(extraction['privateGoldRead'] is False and extraction['copyAdapterLoaded'] is False and packet['privateGoldRead'] is False and packet['nativeContextExtractionOnly'] is True, 'Public-only frozen-base extraction required')
    require(extraction['nativeForwards'] == packet['expectedNativeForwards'] == 432, 'Exact432 native forwards required')
    require(Counter(r['split'] for r in packet['rows']) == Counter({'train': 320, **PRIMARY_SPLITS}), 'Exact native split roster')
    require([(r['caseId'], r['split']) for r in extraction['roster']] == [(r['caseId'], r['split']) for r in packet['rows']] and all(r['status'] == 'completed' for r in extraction['roster']), 'Native fixed roster incomplete')
    timings = {r['caseId']: r for r in extraction['timings']}
    require(len(timings) == 432, 'Exact native timing denominator')
    for r in packet['rows']:
        t = timings[r['caseId']]
        require(t['batchRows'] == t['backboneForwardCalls'] == 1 and t['truncatedRows'] == 0 and t['paddedTokensPerRow'] == r['tokens'], 'WholeB1 token receipt differs')
        require(r['caseId'] == r['publicProfile']['profileId'] and r['member'] == {'profileId':r['caseId']} and r['role'] == 'request' and r['batchShape'] == [1,r['tokens']], 'Exact public profile/member binding')
        require(len(r['tokenIds']) == r['tokens'] and sha_bytes(r['text'].encode()) == r['textSha256'] and sha_bytes(json.dumps(r['tokenIds'],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()) == r['tokenSha256'], 'Public text/token hashes differ')
    feature_pins = extraction['serializedFeatureFilesSha256']
    require(len(feature_pins) == 864, 'Exact864 serialized feature files')
    for path, expected in feature_pins.items(): require(sha(path) == expected, 'Native serialized feature changed')
    def recheck_all_pins():
        for group in ('sourceSha256','publicFilesSha256','originalCatalogPins','annotationFilesExpectedSha256'):
            for path,expected in packet[group].items():require(sha(path)==expected,'Audit input changed: '+path)
        for pins in (recipe['sha256'], feature_pins, run_pins):
            for path,expected in pins.items():require(sha(path)==expected,'Audit artifact changed: '+path)
    torch.set_num_threads(1)
    checkpoint = torch.load(run / 'head.pt', map_location='cpu', weights_only=True)
    require(checkpoint['version'] == 'public-persona-qwen-head.v1', 'Exact native context head format required')
    require(checkpoint['config'] == packet['config'] and checkpoint['sourceFeaturePreflightSha256'] == PREFLIGHT_SHA256, 'Checkpoint recipe/source binding')
    require(sha(run / 'head.pt') == report['checkpointSha256'] == freeze['checkpointSha256'], 'Checkpoint/report/prediction chain')
    require(report['actualUpdates'] == 250 and report['reloadExact'] is True and report['reloadPredictionParity'] is True, 'Actual250 update/reload receipts required')
    require(0 < report['ownershipGradientNonzeroSteps'] <= 250, 'Learned ownership gradient receipt required')
    state = checkpoint['state_dict']
    changed_tensors = verify_state(state, report, packet['config'])
    reloaded = torch.load(run/'head.pt', map_location='cpu', weights_only=True)
    require(set(reloaded['state_dict']) == set(state) and all(torch.equal(v,reloaded['state_dict'][k]) for k,v in state.items()), 'Independent exact checkpoint reload failed')
    catalog = source_loader.load_catalog_features(packet['originalCatalogRoot'])
    require(checkpoint['catalogPins'] == catalog['pins'] and checkpoint['featureProvenance'] == packet['featureProvenance'], 'Checkpoint actual catalog/model pins')
    store = FeatureStore.open_existing(feature_root / 'features', packet['featureProvenance'])
    features = {}; raw = {}
    for row in packet['rows']:
        value, receipt = store.load(row); features[row['caseId']] = value; raw[row['caseId']] = row
        require(tuple(value.shape) == (1,4096) and torch.isfinite(value).all() and torch.allclose(value.norm(dim=1),torch.ones(1),atol=1e-4), 'Finite normalized native context4096 required')
        for path in store.paths(row['role'], row['member']): require(feature_pins.get(str(path.resolve())) == sha(path), 'Exact feature member receipt pin')
    demo_cache_count = verify_public_demo_cache(checkpoint, run, packet, features, feature_root/'report.json')
    run_pins[checkpoint['publicDemoFeatureCacheFile']] = checkpoint['publicDemoFeatureCacheSha256']
    replay_count = 0
    def replay(name, predictions, profiles, x, owned_override=None):
        nonlocal replay_count
        with torch.no_grad(): scores, variants = replay_scores(state, x, catalog, profiles, owned_override=owned_override)
        require([p['scores16'] for p in predictions] == scores.tolist(), 'Independent checkpoint score replay differs: ' + name)
        # Variant argmax is independently checked within actual typed colors.
        from .store_demo_engine import guarded_order
        for row, values, profile in zip(predictions, variants.tolist(), profiles, strict=True):
            checked, policy, _ = source_loader.checked_context_request(profile['request'],catalog['products'])
            expected_order, _ = guarded_order(row['scores16'],checked,policy,catalog['products'])
            require(row['order'] == expected_order, 'Pinned hard guard/order replay differs: ' + name)
            for index, selected in zip(row['order'][:6], row['selected'], strict=True):
                candidates = [(j,v) for j,v in enumerate(catalog['variants']) if v['productIndex'] == index and all(h['attribute'] != 'color' or h['value'] == v['colorName'] for h in profile['request'].get('hardRequirements', []))]
                chosen = max(candidates, key=lambda item: (values[item[0]], -item[0]))[1]
                require(selected['variantId'] == chosen['variantId'], 'Learned real-color argmax differs: ' + name)
        replay_count += len(predictions)
    for split, predictions in outputs.items():
        profiles = [r['publicProfile'] for r in packet['rows'] if r['split'] == split]
        replay(split, predictions, profiles, torch.cat([features[p['profileId']] for p in profiles]))
    independent = [r['publicProfile'] for r in packet['rows'] if r['split'] == 'independent-evaluation']
    ix = torch.cat([features[p['profileId']] for p in independent])
    replay('ownership-term-disabled', baselines['ownership-term-disabled'], independent, ix, torch.zeros((32,16)))
    replay('zero-context-owned-only', baselines['zero-context-owned-only'], independent, torch.zeros_like(ix))
    shuffled = deepcopy(independent)
    for i,p in enumerate(shuffled):
        other = independent[(i+17)%32]['request']; p['request']['persona'] = deepcopy(other['persona']); p['request']['publicHistory'] = deepcopy(other['publicHistory'])
    replay('shuffled-context', baselines['shuffled-context'], shuffled, torch.roll(ix,-17,0))
    # TRAIN-only labels are permitted after the admission barrier; fresh32 gold
    # remains unopened until every baseline's numeric scores have been checked.
    train_profiles = [r['publicProfile'] for r in packet['rows'] if r['split'] == 'train']
    train_path = HERE/'synthetic_history_train_contrasts_v2/labels.train.json'
    require(str(train_path.resolve()) in packet['annotationFilesExpectedSha256'], 'Exact frozen TRAIN annotation path required')
    simple_scores = simple_baseline_scores(read(train_path)['labels'], train_profiles, catalog['products'], independent)
    from .store_demo_engine import guarded_order
    for name, expected in simple_scores.items():
        require([r['scores16'] for r in baselines[name]] == expected, 'Independent simple baseline scores differ: ' + name)
        for row, profile in zip(baselines[name],independent,strict=True):
            checked,policy,_ = source_loader.checked_context_request(profile['request'],catalog['products'])
            require(row['order'] == guarded_order(row['scores16'],checked,policy,catalog['products'])[0], 'Simple baseline guard/order differs: ' + name)
            for i, card in zip(row['order'][:6], row['selected'], strict=True):
                product = catalog['products'][i]
                variant = next(v for v in product['variants'] if all(h['attribute'] != 'color' or h['value'] == v['colorName'] for h in checked['hardRequirements']))
                require(card == {'productId':product['productId'],'variantId':variant['variantId']}, 'Simple baseline source variant differs: ' + name)
    # ADMISSION BARRIER COMPLETE. Only now parse the fresh32 private gold.
    recheck_all_pins()
    gold_path = HERE / 'persona_qwen_independent_eval_v2/labels.private.json'
    require(str(gold_path.resolve()) in packet['annotationFilesExpectedSha256'], 'Correct fresh private label path required')
    labels = read(gold_path)['labels']
    main_metrics = metrics(outputs['independent-evaluation'], labels, catalog['products'], independent)
    baseline_metrics = {name: metrics(rows, labels, catalog['products'], independent) for name,rows in baselines.items()}
    factual = factual_checks(outputs['independent-evaluation'], independent, catalog)
    all_factual = {split:factual_checks(predictions,[r['publicProfile'] for r in packet['rows'] if r['split']==split],catalog) for split,predictions in outputs.items()}
    probe_public = read(HERE / 'synthetic_history_v1/history-only-probes.public.json')
    probe_gold = read(HERE / 'synthetic_history_v1/history-only-probes.labels.json')['labels']
    history = history_metrics(outputs['history-diagnostic-nonfit'], probe_public, probe_gold, catalog['products'])
    recheck_all_pins()
    thresholds = gate['freshIndependent32']; dev = gate['knownDevelopment10HistoryPairs']
    checks = {'primary20': main_metrics['primaryTop1'] >= thresholds['primaryGrade3Top1Minimum'],
              'novel24': main_metrics['novelRelevantTop1'] >= thresholds['novelRelevantGradeAtLeast2Top1Minimum'],
              'strongTop3_28': main_metrics['strongPositiveTop3'] >= thresholds['gradeAtLeast2CoverageTop3Minimum'],
              'zeroFactConstraintViolations': factual['violationCount'] <= thresholds['budgetTypedConstraintVariantOrFactViolationsMaximum'],
              'devOwnedDemotions8': history['ownedPriorityDemotions'] >= dev['ownedProductPriorityDemotionsMinimum'],
              'devNovelAfter8': history['novelRelevantAfterTop1'] >= dev['novelRelevantAfterPurchaseTop1Minimum']}
    checks.update({'primaryStrictlyExceeds_' + name: main_metrics['primaryTop1'] > baseline_metrics[name]['primaryTop1'] for name in sorted(DOMINANCE)})
    offline_pass = all(checks.values())
    result = {'version': VERSION, 'status': 'offline-gate-passes-serving-pending' if offline_pass else 'fails-offline-gate',
              'offlineGatePassed': offline_pass, 'overallPromotionPassed': False, 'gateSha256': GATE_SHA256,
              'checks': checks, 'independent32': main_metrics, 'baselines': baseline_metrics,
              'knownDevelopmentHistory': history, 'factualChecks': factual, 'allNonfitSourceFactChecks': all_factual,
              'evidence': {'checkpointSha256': sha(run/'head.pt'), 'predictionFreezeSha256': sha(run/'PREDICTIONS_FROZEN.json'),
                           'adjudicatorSourceSha256': sha(__file__), 'adjudicatorFreezeSha256': sha(HERE/'PERSONA_QWEN_ADJUDICATOR_FROZEN_V1.json'),
                           'extractionReportSha256': sha(feature_root/'report.json'), 'nativeForwards': 432, 'actualOptimizerUpdatesProducerReceipt': 250,
                           'nativeFrozen927ProducerReceipt': True, 'reviewerRecreatedNativeHashes': False,
                           'independentCheckpointScoreRowsReplayed': replay_count, 'simpleBaselineScoreRowsReplayed': 64,
                           'independentReloadExact': True, 'seededChangedTensorsVerified': changed_tensors,
                           'publicOnlyDemoContextCacheEntriesVerified': demo_cache_count, 'sourceAndFeaturePinsVerified': True,
                           'freshGoldOpenedOnlyAfterAdmissionAndReplay': True},
              'servingAcceptance': {name: {'status': 'pending', 'required': required} for name,required in gate['servingAcceptance'].items()},
              'limitations': ['Synthetic known-catalog combinations and TRAIN-seen ownership intervention; not human behavior.',
                              'Old40 and ten history pairs are development replay.', 'Ownership-term-disabled is report-only, not a strict-dominance arm.',
                              'Producer loaded nonfit public vectors before fitting; this is not a zero-heldout-feature-access experiment. Labels remain excluded until prediction freeze.',
                              'No new native one-hot or no-history representation claim.', 'No optimizer/native/browser executed by adjudicator.',
                              'Browser, full fresh copy, query-fusion and useful visible readiness remain separate pending evidence.']}
    out.mkdir(parents=True)
    write(out/'ADJUDICATION.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    for name in ('run', 'features', 'out'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(argv)
    result = adjudicate(a.run, a.features, a.out)
    print(json.dumps({'status': result['status'], 'offlineGatePassed': result['offlineGatePassed'], 'overallPromotionPassed': False}))
    return 0 if result['offlineGatePassed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
