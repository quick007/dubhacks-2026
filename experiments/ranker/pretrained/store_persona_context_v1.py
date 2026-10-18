"""Public-observation CPU learner on genuine frozen Qwen catalog vectors.

No native model or browser loads. Context features are explicitly hashed public
text/history, NOT purported Qwen persona embeddings. Gold is offline only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import time
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
VERSION = 'public-persona-context-model.v1'
CONFIG = {'seed': 20261018, 'steps': 250, 'learningRate': .003, 'weightDecay': .0001,
    'textDimensions': 2048, 'colorDimensions': 256, 'hiddenWidth': 64, 'colorWidth': 32,
    'threads': 1, 'halfLifeDays': 30., 'variantLossWeight': .5,
    'selection': 'single predeclared final step; no validation or heldout tuning',
    'contextRepresentation': 'signed public word/character ngrams plus observed real-item Qwen history mean; no Qwen persona encoding',
    'scope': 'synthetic known16 catalog; family/group heldout; no human or unseen-catalog claim'}
NEUTRAL = {'browse', 'browse the store', 'show me the store', 'show me some products', ''}

def canonical(value): return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def checked_context_request(value, products=None):
    from .store_demo_engine import checked_request
    allowed = {'query', 'itemBudget', 'hardRequirements', 'persona', 'publicHistory'}
    if type(value) is not dict or set(value) - allowed: raise ValueError('Only public query/constraints/persona/history fields allowed')
    base = {k: deepcopy(value[k]) for k in ('query', 'itemBudget', 'hardRequirements') if k in value}
    request, policy = checked_request(base)
    # Preserve the stricter public profile ceiling and explicit query ceiling.
    parsed, parsed_policy = checked_request({**base, 'itemBudget': None})
    if parsed['itemBudget'] is not None and (request['itemBudget'] is None or parsed['itemBudget'] <= request['itemBudget']):
        request['itemBudget'] = parsed['itemBudget']; policy.update(parsed_policy)
    if products is not None:
        colors = {v['colorName'] for p in products for v in p['variants']}
        for color in sorted(colors):
            if re.search(r'\b(?:only|must be|color|colour|in)\s+' + re.escape(color) + r'\b', request['query'], re.I):
                requirement = {'attribute': 'color', 'value': color}
                if requirement not in request['hardRequirements']: request['hardRequirements'].append(requirement)
    persona = value.get('persona', {'background': [], 'colorPreferences': []})
    if type(persona) is not dict or set(persona) != {'background', 'colorPreferences'}: raise ValueError('Exact public persona observation fields required')
    for key in persona:
        if type(persona[key]) is not list or len(persona[key]) > 20 or any(type(s) is not str or not s.strip() or len(s) > 500 for s in persona[key]):
            raise ValueError('Bounded public background/color strings required')
    history = value.get('publicHistory', [])
    if type(history) is not list or len(history) > 24: raise ValueError('At most24 observable events')
    known = {} if products is None else {p['productId']: p for p in products}
    seen_events = set()
    for event in history:
        required = {'eventId', 'eventType', 'itemName', 'category', 'occurredAt', 'source'}
        optional = {'selectedColor', 'productId', 'variantId', 'external'}
        if type(event) is not dict or not required <= set(event) or set(event) - required - optional: raise ValueError('Exact public history event fields required')
        if event['eventType'] not in ('view', 'cart', 'purchase') or event['source'] not in ('demo-seed', 'storefront-action'): raise ValueError('Observable event type/source required')
        if any(type(event[k]) is not str or not event[k].strip() or len(event[k]) > 500 for k in required - {'eventType', 'source'}): raise ValueError('Bounded event strings required')
        if datetime.fromisoformat(event['occurredAt'].replace('Z', '+00:00')).tzinfo is None: raise ValueError('Timezone-aware observable timestamp required')
        if event['eventId'] in seen_events: raise ValueError('Unique observable event IDs required')
        seen_events.add(event['eventId'])
        if 'external' in event and type(event['external']) is not bool: raise ValueError('External must be boolean')
        if 'selectedColor' in event and (type(event['selectedColor']) is not str or not event['selectedColor'].strip()): raise ValueError('Observed color must be a string')
        if event.get('external') is True and ('productId' in event or 'variantId' in event): raise ValueError('External purchase must not invent a catalog SKU')
        if 'variantId' in event and 'productId' not in event: raise ValueError('Observed variant requires observed product')
        if products is not None and 'productId' in event:
            if event['productId'] not in known: raise ValueError('Observed product must exist in actual catalog')
            product = known[event['productId']]
            if event['itemName'] != product.get('title', product.get('name')) or event['category'] != product['category']: raise ValueError('Observed SKU facts must match real product')
            if 'variantId' in event:
                variant = next((v for v in product['variants'] if v['variantId'] == event['variantId']), None)
                if variant is None or event.get('selectedColor') != variant['colorName']: raise ValueError('Observed variant/color must match real product')
            elif 'selectedColor' in event and event['selectedColor'] not in {v['colorName'] for v in product['variants']}: raise ValueError('Observed color not available on real product')
    return request, policy, {'persona': deepcopy(persona), 'publicHistory': deepcopy(history)}

def hashed_text(texts, dimensions):
    values = torch.zeros(dimensions, dtype=torch.float32)
    for text, weight in texts:
        words = re.findall(r"[\w'-]+", text.casefold())
        features = ['w:' + w for w in words]
        for word in words:
            padded = '^' + word + '$'
            features.extend('c:' + padded[i:i+n] for n in (3, 4) for i in range(max(0, len(padded)-n+1)))
        for token in features:
            raw = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), 'little')
            values[raw % dimensions] += weight * (1 if raw >> 63 else -1)
    return F.normalize(values, dim=0)

def context_features(context, catalog, *, omit_history=False):
    persona = context['persona']; history = [] if omit_history else context['publicHistory']
    texts = [('background ' + s, 1.) for s in persona['background']]
    texts += [('preferred color ' + s, 1.5) for s in persona['colorPreferences']]
    vectors = []; weights = []
    anchor = max((datetime.fromisoformat(e['occurredAt'].replace('Z', '+00:00')) for e in history), default=None)
    by_id = {p['productId']: i for i, p in enumerate(catalog['products'])}
    for event in history:
        when = datetime.fromisoformat(event['occurredAt'].replace('Z', '+00:00'))
        days = max(0., (anchor - when).total_seconds() / 86400)
        weight = {'view': .25, 'cart': .5, 'purchase': 1.}[event['eventType']] * 2 ** (-days / CONFIG['halfLifeDays'])
        # Event/profile/session IDs and absolute timestamps are never text inputs.
        texts.append((event['eventType'] + ' ' + event['itemName'] + ' ' + event['category'] + ' ' + event.get('selectedColor', ''), weight))
        if event.get('productId') in by_id:
            vectors.append(catalog['vectors'][by_id[event['productId']]]); weights.append(weight)
    history_vector = (sum(v * w for v, w in zip(vectors, weights)) / sum(weights)) if vectors else torch.zeros(4096)
    return hashed_text(texts, CONFIG['textDimensions']), F.normalize(history_vector, dim=0)

def load_catalog_features(feature_root):
    """Read-only original real tensors, validated against native source receipts."""
    from .catalog_fusion.runtime import FeatureStore
    from .store_catalog_v1 import load_snapshot
    from .store_function_adaptation import product_text
    root = Path(feature_root).resolve(); report = json.loads((root / 'report.json').read_text()); packet = json.loads((root / 'packet.json').read_text())
    anchor = json.loads((HERE / 'store_function_v5_prepare_v3/preflight.json').read_text())
    if sha(root / 'report.json') != anchor['priorReportSha256']: raise ValueError('Original report must match reviewed v5 source anchor')
    freeze_path = root / 'RAW_FEATURES_FROZEN.json'
    if sha(freeze_path) != report['rawFeaturesFreezeSha256']: raise ValueError('Original raw feature freeze changed')
    raw_pins = json.loads(freeze_path.read_text())['files']
    if any(sha(root / name) != checksum for name, checksum in raw_pins.items() if name.startswith('features/catalog/')): raise ValueError('Original native catalog archive bytes changed')
    if report['featureProvenance']['evidenceSource'] != 'native-frozen-qwen': raise ValueError('Genuine frozen Qwen provenance required')
    if report['status'] != 'completed' or report['nativeForwards'] != 64 or sha(root / 'packet.json') != report['packetSha256']: raise ValueError('Exact completed native64 v3 catalog source required')
    snapshot = load_snapshot(); rows = [r for r in packet['records'] if r['role'] == 'catalog']
    if [r['caseId'] for r in rows] != [str(i) for i in range(16)]: raise ValueError('Exactly16 original catalog members required')
    store = FeatureStore.open_existing(root / 'features', report['featureProvenance'])
    tensors = []; pins = {str(root / n): sha(root / n) for n in ('report.json', 'packet.json', 'RAW_FEATURES_FROZEN.json')}; products = []
    for index, row in enumerate(rows):
        if row['text'] != product_text(snapshot['inputs']['products'][index]): raise ValueError('Native catalog feature text differs from actual public catalog')
        tensor, receipt = store.load(row)
        if tuple(tensor.shape) != (1, 4096) or not torch.isfinite(tensor).all(): raise ValueError('Actual finite4096 Qwen vectors required; no fake replacements')
        tensors.append(tensor)
        for path in store.paths(row['role'], row['member']): pins[str(path)] = sha(path)
        p = snapshot['inputs']['products'][index]; external = snapshot['externalMapping']['products'][index]
        variants = [{'variantId': v['variantId'], 'colorName': c['colorName'], 'images': deepcopy(v['images'])}
            for c, v in zip(p['colorVariants'], external['colorVariants'], strict=True)]
        products.append({**deepcopy(p), 'productId': external['productId'], 'title': external['originalName'] if 'originalName' in external else p['name'], 'category': external['category'] if 'category' in external else p.get('category', ''), 'variants': variants})
    vectors = torch.cat(tensors).float()
    if not torch.allclose(vectors.norm(dim=1), torch.ones(16), atol=1e-4): raise ValueError('Original normalized source vectors required')
    variants = [{**v, 'productIndex': i} for i, p in enumerate(products) for v in p['variants']]
    return {'products': products, 'variants': variants, 'vectors': vectors, 'snapshot': snapshot,
        'pins': pins, 'provenance': report['featureProvenance'], 'nativeFeatureEvidence': '16 genuine frozenQwen4096 catalog encodings; public context uses separate CPU text/history features'}

class PersonaContextHead(nn.Module):
    def __init__(self, *, catalog_dimensions=4096):
        super().__init__(); torch.manual_seed(CONFIG['seed'])
        self.text = nn.Linear(CONFIG['textDimensions'], CONFIG['hiddenWidth'])
        self.history = nn.Linear(4096, CONFIG['hiddenWidth'], bias=False)
        self.product = nn.Linear(catalog_dimensions, CONFIG['hiddenWidth'], bias=False)
        self.color_context = nn.Linear(CONFIG['textDimensions'], CONFIG['colorWidth'])
        self.color_candidate = nn.Linear(CONFIG['colorDimensions'], CONFIG['colorWidth'], bias=False)
        self.scale = nn.Parameter(torch.tensor(2.))
    def forward(self, text, history, catalog_vectors, color_features, variant_product_indices):
        context = F.normalize(torch.tanh(self.text(text) + self.history(history)), dim=-1)
        products = F.normalize(self.product(catalog_vectors), dim=-1)
        product_scores = context @ products.T * self.scale.exp().clamp(max=30)
        colors = F.normalize(torch.tanh(self.color_context(text)), dim=-1) @ F.normalize(self.color_candidate(color_features), dim=-1).T * 4
        return product_scores, product_scores[:, variant_product_indices] + colors

def source_inputs(profiles, catalog, *, omit_history=False):
    features = [context_features(checked_context_request(p['request'], catalog['products'])[2], catalog, omit_history=omit_history) for p in profiles]
    return torch.stack([x[0] for x in features]), torch.stack([x[1] for x in features])

def source_colors(catalog):
    return (torch.stack([hashed_text([(v['colorName'], 1.)], CONFIG['colorDimensions']) for v in catalog['variants']]),
        torch.tensor([v['productIndex'] for v in catalog['variants']], dtype=torch.long))

def read_profiles(path):
    value = json.loads(Path(path).read_text())
    return value['profiles']

def label_rows(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, list): return value
    for key in ('labels', 'profiles', 'rows'):
        if key in value: return value[key]
    raise ValueError('Explicit label row array required')

def targets(profiles, labels, catalog):
    by_id = {r['profileId']: r for r in labels}; product_ids = [p['productId'] for p in catalog['products']]
    if set(by_id) != {p['profileId'] for p in profiles}: raise ValueError('Exact split label/profile join required')
    product = []; variant = []
    for profile in profiles:
        label = by_id[profile['profileId']]; grades = label['relevanceGrades']
        if set(grades) != set(product_ids) or any(type(g) is not int or not 0 <= g <= 3 for g in grades.values()): raise ValueError('All16 relevance grades required')
        gain = torch.tensor([2 ** grades[k] - 1 for k in product_ids], dtype=torch.float32)
        if gain.sum() == 0: raise ValueError('Positive TRAIN relevance required')
        preferred = set(label['preferredVariantIds'])
        if preferred - {v['variantId'] for v in catalog['variants']}: raise ValueError('Gold variant must be real source variant')
        vg = torch.tensor([gain[v['productIndex']] if v['variantId'] in preferred else 0. for v in catalog['variants']])
        if vg.sum() == 0: vg = torch.tensor([gain[v['productIndex']] for v in catalog['variants']])
        product.append(gain / gain.sum()); variant.append(vg / vg.sum())
    return torch.stack(product), torch.stack(variant)

def fit_model(text, history, catalog_vectors, colors, variant_indices, product_targets, variant_targets, *, progress=None):
    model = PersonaContextHead(catalog_dimensions=catalog_vectors.shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learningRate'], weight_decay=CONFIG['weightDecay'])
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    losses = []
    for step in range(CONFIG['steps']):
        optimizer.zero_grad(); ps, vs = model(text, history, catalog_vectors, colors, variant_indices)
        loss = -(product_targets * ps.log_softmax(-1)).sum(-1).mean() - CONFIG['variantLossWeight'] * (variant_targets * vs.log_softmax(-1)).sum(-1).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
        loss.backward(); optimizer.step()
        if progress is not None: progress(step + 1)
        if step % 25 == 0 or step == CONFIG['steps'] - 1: losses.append({'step': step + 1, 'loss': float(loss.detach())})
    model.training_change_proof = {'changedTensors': [key for key, value in model.state_dict().items() if not torch.equal(value, initial[key])], 'actualOptimizerUpdates': CONFIG['steps']}
    if not model.training_change_proof['changedTensors']: raise ValueError('Optimizer made no actual weight changes')
    return model.eval(), losses

def predict(model, profiles, catalog, *, omit_history=False, catalog_vectors=None):
    text, history = source_inputs(profiles, catalog, omit_history=omit_history); colors, indices = source_colors(catalog)
    with torch.no_grad(): product, variant = model(text, history, catalog['vectors'] if catalog_vectors is None else catalog_vectors, colors, indices)
    output = []
    from .store_demo_engine import guarded_order
    for profile, ps, vs in zip(profiles, product.tolist(), variant.tolist(), strict=True):
        request, policy, _ = checked_context_request(profile['request'], catalog['products'])
        order, checks = guarded_order(ps, request, policy, catalog['products'])
        selected = []
        for i in order[:6]:
            eligible = [(j, v) for j, v in enumerate(catalog['variants']) if v['productIndex'] == i
                and all(r['attribute'] != 'color' or r['value'] == v['colorName'] for r in request['hardRequirements'])]
            if not eligible: raise ValueError('Hard color guard and real variant roster differ')
            j, v = max(eligible, key=lambda item: (vs[item[0]], -item[0]))
            selected.append({'productId': catalog['products'][i]['productId'], 'productIndex': i,
                'variantId': v['variantId'], 'selectedColor': v['colorName'], 'image': deepcopy(v['images'][0])})
        output.append({'profileId': profile['profileId'], 'userGroupId': profile['userGroupId'],
            'scores16': ps, 'rawLearnedOrder': sorted(range(16), key=lambda i: (-ps[i], i)), 'order': order, 'selected': selected, 'shortfall': max(0, 6 - len(selected)), 'guardChecks': checks})
    return output

def checkpoint(model, path, catalog, metadata, *, catalog_dimensions=4096):
    saved = {'version': VERSION, 'config': CONFIG, 'catalogDimensions': catalog_dimensions,
        'state_dict': model.state_dict(), 'catalogPins': catalog['pins'], 'catalogProvenance': catalog['provenance'],
        'publicProducts': catalog['products'], 'metadata': metadata}
    torch.save(saved, path)
    loaded, restored = load_head(path)
    if any(not torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items()): raise ValueError('Exact checkpoint reload failed')
    return sha(path)

def load_head(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if any(not torch.isfinite(value).all() for value in saved['state_dict'].values()): raise ValueError('Finite learned state required')
    if saved['version'] != VERSION or saved['config'] != CONFIG: raise ValueError('Exact frozen persona architecture required')
    model = PersonaContextHead(catalog_dimensions=saved['catalogDimensions']); model.load_state_dict(saved['state_dict'], strict=True)
    return model.eval().requires_grad_(False), saved

def checked_split(profiles, split):
    by_id = {p['profileId']: p for p in profiles}
    train_ids, held_ids = split['trainProfileIds'], split['heldoutProfileIds']
    if len(by_id) != 200 or len(train_ids) != 160 or len(held_ids) != 40 or set(train_ids) & set(held_ids) or set(by_id) != set(train_ids) | set(held_ids): raise ValueError('Exact160/40 unique public profile split required')
    train = [by_id[i] for i in train_ids]; held = [by_id[i] for i in held_ids]
    tg, eg = {p['userGroupId'] for p in train}, {p['userGroupId'] for p in held}
    if tg & eg or len(tg) != 80 or len(eg) != 20 or tg & set(split['demoGroupIds']) or eg & set(split['demoGroupIds']): raise ValueError('Whole groups and demo groups must be disjoint')
    if any(n != 2 for n in Counter(p['userGroupId'] for p in profiles).values()): raise ValueError('Two sessions per anonymous group required')
    return train, held

def baseline_prediction(profile, scores, catalog):
    from .store_demo_engine import guarded_order
    request, policy, _ = checked_context_request(profile['request'], catalog['products'])
    order, checks = guarded_order(scores, request, policy, catalog['products'])
    return {'profileId': profile['profileId'], 'userGroupId': profile['userGroupId'], 'scores16': scores,
        'order': order, 'selected': [{'productId': catalog['products'][i]['productId'],
            'variantId': next(v['variantId'] for v in catalog['products'][i]['variants'] if all(r['attribute'] != 'color' or r['value'] == v['colorName'] for r in request['hardRequirements']))} for i in order[:6]],
        'guardChecks': checks}

def evaluate_rows(predictions, labels, catalog):
    gold = {l['profileId']: l for l in labels}; products = [p['productId'] for p in catalog['products']]
    if set(gold) != {p['profileId'] for p in predictions}: raise ValueError('Exact heldout prediction/gold join required')
    rows = []
    for prediction in predictions:
        label = gold[prediction['profileId']]; grades = label['relevanceGrades']; ids = [products[i] for i in prediction['order'][:6]]
        gains = [2 ** grades[p] - 1 for p in ids]; ideal = sorted([2 ** g - 1 for g in grades.values()], reverse=True)[:6]
        dcg = lambda values: sum(g / math.log2(i + 2) for i, g in enumerate(values))
        owned = set(label['alreadyOwnedProductIds']); top = ids[0] if ids else None
        variants = label.get('acceptableVariantIds', label['preferredVariantIds'])
        rows.append({'profileId': prediction['profileId'], 'userGroupId': prediction['userGroupId'],
            'topProductId': top, 'topGrade': grades.get(top, 0), 'topGrade3': grades.get(top, 0) == 3,
            'novelRelevantTop1': bool(top and top not in owned and grades[top] >= 2),
            'ownedDuplicateTop1': top in owned, 'ownedDuplicatesTop6': sum(p in owned for p in ids),
            'ndcg6': dcg(gains) / max(dcg(ideal), 1e-8),
            'acceptableVariantTop1': bool(prediction['selected'] and prediction['selected'][0]['variantId'] in variants)})
    groups = {}
    for row in rows: groups.setdefault(row['userGroupId'], []).append(row)
    return {'fixedDenominator': 40, 'predictions': len(rows), 'userGroups': len(groups),
        'semanticClassCoverage': len({max(l['relevanceGrades'], key=l['relevanceGrades'].get) for l in labels}),
        'topGrade3Rate': sum(r['topGrade3'] for r in rows) / 40,
        'novelRelevantTop1Rate': sum(r['novelRelevantTop1'] for r in rows) / 40,
        'ownedDuplicateTop1Rate': sum(r['ownedDuplicateTop1'] for r in rows) / 40,
        'ndcg6': sum(r['ndcg6'] for r in rows) / 40,
        'acceptableVariantTop1Rate': sum(r['acceptableVariantTop1'] for r in rows) / 40,
        'groupNovelRelevantTop1Rates': {g: sum(r['novelRelevantTop1'] for r in rs) / len(rs) for g, rs in groups.items()},
        'rows': rows}

def _run_fit(data, public_export, feature_root, out, *, tiger_receipt=None, provisional=False, evaluate=False):
    """Final fit requires actual Tiger-readback receipt. Provisional runs cannot
    emit heldout predictions. All architecture/data hashes precede public parse;
    heldout gold is opened only after frozen checkpoint and all baseline outputs.
    """
    data, public_export, out = map(Path, (data, public_export, out))
    if not out.resolve().is_relative_to(HERE.parents[2]): raise ValueError('All writes must stay in isolated persona worktree')
    if out.exists(): raise FileExistsError('Preserve every prior persona model attempt')
    if provisional and evaluate: raise ValueError('Provisional seed-only fit cannot evaluate final heldout')
    if not provisional and tiger_receipt is None: raise ValueError('Final fit requires actual Tiger readback/export proof')
    out.mkdir(parents=True); torch.set_num_threads(CONFIG['threads'])
    files = [public_export, data / 'split.json', data / 'labels.train.json', data / 'labels.heldout.json', data / 'manifest.json']
    frozen = {'version': VERSION, 'config': CONFIG, 'dataSha256': {str(p.resolve()): sha(p) for p in files},
        'sourceSha256': sha(Path(__file__)), 'provisionalSeedOnly': provisional,
        'frozenBeforeAnyProfileOrGoldParse': True, 'heldoutUsedForFitOrSelection': False}
    dump(out / 'ARCHITECTURE_AND_INPUTS_FROZEN.json', frozen)
    if tiger_receipt is not None:
        proof = json.loads(Path(tiger_receipt).read_text())
        if (proof.get('source') != 'tiger' or proof.get('reportId') != 'authored-public-history-v1-200profiles'
                or proof.get('sha256') != '176054e7b514ad21b127f411dec247718595d93915d9ebb55e8258e1f0a38c03'
                or sha(tiger_receipt) != '4032d059888c53376f477892c8ecba197c8d900012daf492d3e00dec198d5c22'
                or Path(tiger_receipt).parent.resolve() != public_export.parent.resolve()
                or proof['files']['profiles.public.json'] != sha(public_export)):
            raise ValueError('Exact root-reviewed actual Tiger-readback manifest required')
        for name, checksum in proof['files'].items():
            if sha(public_export.parent / name) != checksum: raise ValueError('Tiger export file mismatch')
        dump(out / 'TIGER_INPUT_PROOF.json', {'receiptSha256': sha(tiger_receipt), **proof})
    verify_seed(data, public_export)
    profiles = read_profiles(public_export); split = json.loads((data / 'split.json').read_text()); train, held = checked_split(profiles, split)
    catalog = load_catalog_features(feature_root); train_labels = label_rows(data / 'labels.train.json')
    text, history = source_inputs(train, catalog); colors, variant_indices = source_colors(catalog)
    pt, vt = targets(train, train_labels, catalog)
    report = {'version': VERSION, 'status': 'training', 'config': CONFIG, 'provisionalSeedOnly': provisional,
        'publicExportSha256': sha(public_export), 'trainProfiles': 160, 'trainUserGroups': 80,
        'heldoutProfiles': 40, 'heldoutUserGroups': 20, 'catalogPins': catalog['pins'],
        'featureEvidence': catalog['nativeFeatureEvidence'], 'nativeCalls': 0, 'optimizerUpdatesPerModel': 0,
        'scope': 'Known16 synthetic family-heldout motifs; catalog projection may learn class codes; no Qwen persona encoding/semantic advantage claim',
        'heldoutStatus': 'unattempted', 'currentV5QueryRegression': 'pending root native release; new persona text never reuses unrelated cached query vectors'}
    dump(out / 'results.json', report)
    source_pins = {str(p): sha(p) for p in [Path(__file__), HERE / 'store_demo_engine.py', HERE / 'store_catalog_v1.py', HERE / 'store_function_adaptation.py', HERE / 'catalog_fusion/runtime.py']}
    def verify_sources():
        if any(sha(p) != checksum for p, checksum in source_pins.items()): raise ValueError('Model/helper sources changed during experiment')
    dump(out / 'SOURCE_PINS.json', source_pins)
    try:
        verify_sources()
        def progress(model_name):
            def record(step): dump(out / 'OPTIMIZER_PROGRESS.json', {'activeModel': model_name, 'actualUpdatesThisModel': step, 'priorModelsCompletedUpdates': 0 if model_name == 'qwen-catalog' else CONFIG['steps']})
            return record
        head, loss = fit_model(text, history, catalog['vectors'], colors, variant_indices, pt, vt, progress=progress('qwen-catalog'))
        # Predeclared one-hot catalog ablation: measures whether catalog Qwen
        # vectors confer an advantage over arbitrary known-item identity codes.
        onehot = torch.eye(16); ablation, aloss = fit_model(text, history, onehot, colors, variant_indices, pt, vt, progress=progress('onehot-catalog'))
        popularity = pt.mean(0).tolist()
        metadata = {'publicExportSha256': sha(public_export), 'trainLabelsSha256': sha(data / 'labels.train.json'),
            'splitSha256': sha(data / 'split.json'), 'provisionalSeedOnly': provisional, 'trainingPopularityScores': popularity}
        head_sha = checkpoint(head, out / 'head.pt', catalog, metadata)
        ablation_sha = checkpoint(ablation, out / 'onehot-ablation.pt', catalog, metadata, catalog_dimensions=16)
        reloaded, _ = load_head(out / 'head.pt')
        reload_parity = predict(head, train[:2], catalog) == predict(reloaded, train[:2], catalog)
        if not reload_parity: raise ValueError('Reloaded predictions differ')
        report['checkpointPredictionParity'] = reload_parity
        report.update(status='completed-provisional-train-only' if provisional else 'completed-final-training',
            optimizerUpdatesPerModel=CONFIG['steps'], totalOptimizerUpdates=2 * CONFIG['steps'], trainLoss=loss, onehotTrainLoss=aloss,
            checkpointSha256=head_sha, onehotCheckpointSha256=ablation_sha, checkpointReloadExact=True, weightChanges=head.training_change_proof, onehotWeightChanges=ablation.training_change_proof)
        dump(out / 'CHECKPOINTS_FROZEN.json', {'headSha256': head_sha, 'ablationSha256': ablation_sha,
            'sourceSha256': frozen['sourceSha256'], 'config': CONFIG, 'heldoutGoldParsed': False})
        if evaluate:
            # All predictions, including predeclared baselines, precede heldout gold.
            outputs = {'context-qwen-catalog': predict(head, held, catalog),
                'onehot-catalog-ablation': predict(ablation, held, catalog, catalog_vectors=onehot),
                'no-history': predict(head, held, catalog, omit_history=True)}
            blank = deepcopy(held)
            for p in blank: p['request']['persona'] = {'background': [], 'colorPreferences': []}; p['request']['publicHistory'] = []
            outputs['no-profile'] = predict(head, blank, catalog)
            shuffled = deepcopy(held)
            for i, p in enumerate(shuffled):
                other = held[(i + 21) % 40]['request']; p['request']['persona'] = deepcopy(other['persona']); p['request']['publicHistory'] = deepcopy(other['publicHistory'])
            outputs['shuffled-profile'] = predict(head, shuffled, catalog)
            outputs['training-popularity'] = [baseline_prediction(p, popularity, catalog) for p in held]
            recent = []
            by_id = {p['productId']: i for i, p in enumerate(catalog['products'])}
            for p in held:
                scores = list(popularity); owned_events = [e for e in p['request']['publicHistory'] if e['eventType'] == 'purchase' and e.get('productId') in by_id]
                if owned_events: scores[by_id[max(owned_events, key=lambda e: e['occurredAt'])['productId']]] += 10
                recent.append(baseline_prediction(p, scores, catalog))
            outputs['most-recent-owned-repeat'] = recent
            for name, predictions in outputs.items(): dump(out / (name + '-predictions.json'), predictions)
            dump(out / 'PREDICTIONS_FROZEN.json', {'fixedHeldoutProfiles': 40,
                'files': {name: sha(out / (name + '-predictions.json')) for name in outputs},
                'checkpointSha256': head_sha, 'heldoutGoldParsed': False})
            verify_sources()
            held_labels = label_rows(data / 'labels.heldout.json')
            report['heldoutMetrics'] = {name: evaluate_rows(predictions, held_labels, catalog) for name, predictions in outputs.items()}
            family_by_group = {g['userGroupId']: g['familyId'] for g in split['groups']}
            for metrics in report['heldoutMetrics'].values():
                families = {}
                for row in metrics['rows']: families.setdefault(family_by_group[row['userGroupId']], []).append(row)
                metrics['perFamily'] = {name: {'sessions': len(rows), 'groups': len({r['userGroupId'] for r in rows}), 'novelRelevantTop1Rate': sum(r['novelRelevantTop1'] for r in rows)/len(rows), 'topGrade3Rate': sum(r['topGrade3'] for r in rows)/len(rows)} for name, rows in families.items()}
            report['heldoutStatus'] = 'completed-fixed40-after-predictions-freeze'
        if any(sha(p) != checksum for p, checksum in catalog['pins'].items()) or any(sha(p) != checksum for p, checksum in frozen['dataSha256'].items()): raise ValueError('Input native feature/data files changed during fit')
        verify_sources()
        report['allInputPinsUnchanged'] = True
    except BaseException as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        dump(out / 'results.json', report); raise
    dump(out / 'results.json', report); return report

def verify_seed(data, public_export):
    expected = {'profiles.public.json': 'b27fb617a57e8785fb100bb9363263602106b3afbbb0b9343cb926e7baf03451',
        'split.json': 'bf377038be8118472d98c517bc36de8d4404470acee54ddc4784936156a30b3d',
        'labels.train.json': '1b1ae769cb162bcb0deee76ea938ac619fbd7a94ee3ef39d7544d5d2491e8ccb',
        'labels.heldout.json': '81b5a341eada3b151fc61b12cf98f87a83c8c4459b442c7c55d8d44460a4fd51'}
    manifest = json.loads((data / 'manifest.json').read_text())
    for name, checksum in manifest['files'].items():
        if sha(data / name) != checksum: raise ValueError('Frozen seed manifest mismatch: ' + name)
    for name, checksum in expected.items():
        if sha(data / name) != checksum: raise ValueError('Predeclared seed source changed: ' + name)
    if canonical(read_profiles(public_export)) != canonical(read_profiles(data / 'profiles.public.json')):
        raise ValueError('Tiger public readback differs from frozen public profile requests')

def run_fit(data, public_export, feature_root, out, **kwargs):
    started = time.monotonic()
    try:
        return _run_fit(data, public_export, feature_root, out, **kwargs)
    except BaseException as error:
        directory = Path(out)
        if directory.exists():
            dump(directory / 'OUTER_FAILURE.json', {'status': 'failed', 'elapsedSeconds': time.monotonic()-started,
                'error': {'type': type(error).__name__, 'message': str(error)}, 'heldoutNotAssumedCompleted': True})
        raise

def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    for name in ('data', 'public-export', 'feature-root', 'out'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--tiger-receipt', type=Path); parser.add_argument('--provisional-train-only', action='store_true'); parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args(argv)
    report = run_fit(args.data, args.public_export, args.feature_root, args.out, tiger_receipt=args.tiger_receipt,
        provisional=args.provisional_train_only, evaluate=args.evaluate)
    print(json.dumps({'status': report['status'], 'updates': report['totalOptimizerUpdates'], 'heldout': report['heldoutStatus']}))

if __name__ == '__main__': main()
