"""Exact-live-format known-store v5 rank fit; CPU prepare unless --run.

Reuse verified16 catalog and32 original TRAIN vectors. Extract64 new TRAIN
utterances, fit once for80 TRAIN-only steps, then encode32 separately frozen
fresh evaluation requests. Truthful96 new native forwards/144 feature vectors.
No live runtime mutation, copy generation, hidden profiles or deployment.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from . import store_function_data_v5 as data
from . import store_function_adaptation as base
from .catalog_fusion.inputs import digest, tensor_digest
from .catalog_fusion.runtime import FeatureStore, assert_model_files, write_json
from .store_catalog_v1 import canonical, load_snapshot
from .store_demo_engine import checked_request, guarded_order
from .store_rank_session import action_bindings

HERE = Path(__file__).resolve().parent
VERSION = 'actual-store-function-adaptation.v5'
PREPARATION_VERSION = 'actual-store-function-v5-preparation.v2'
CONFIG = {'seed': 728, 'steps': 80, 'learningRate': .03, 'weightDecay': .001, 'threads': 1,
    'dimensions': 4096, 'classes': 16, 'maxTokens': 512, 'memoryGiB': 6,
    'reusedFeatures': 48, 'newTrainForwards': 64, 'newEvaluationForwards': 32,
    'newNativeForwards': 96, 'totalFeatureVectors': 144,
    'objective': 'mean(query96-class cross entropy, catalog16-identity cross entropy); balanced6 TRAIN rows/function',
    'selection': 'one fixed final80-step checkpoint; no eval-driven schedule/early stop',
    'permutedQueryOffset': 17, 'permutationClaim': 'fixed public row rotation, no target consulted; actual same-target rate reported postfreeze',
    'controls': ['frozen-v3', 'prior', 'random', 'permuted-query'],
    'runtimeCompatibility': 'same FunctionHead/v3 request and softmax scoring; original StoreRankSession64-forward report guard requires NEW explicit serving adapter; do not fake64'}
SOURCES = ('store_function_adaptation_v5.py', 'store_function_data_v5.py', 'store_function_adaptation.py',
    'store_rank_session.py', 'store_demo_engine.py', 'store_demo_catalog.py', 'store_demo_image_binding_v1.py', 'catalog_fusion/runtime.py',
    'catalog_fusion/inputs.py', 'catalog_fusion/preflight.py', 'store_copy_fit_v2.py', 'semantic_copy_training.py', 'backbone.py')


def sources():
    return {name: data.sha(HERE / name) for name in SOURCES}


@contextmanager
def cpu_threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def read_public(directory):
    directory = Path(directory); freeze = json.loads((directory / 'FREEZE.json').read_text())
    if freeze['version'] != data.VERSION or freeze['builderSha256'] != data.sha(HERE / 'store_function_data_v5.py'):
        raise ValueError('Frozen v5 builder changed')
    for name, expected in freeze['files'].items():
        if data.sha(directory / name) != expected:
            raise ValueError('Frozen v5 file changed: ' + name)
    products = json.loads((directory / 'catalog.json').read_text())
    cases = json.loads((directory / 'public-cases.json').read_text())
    if len(products) != 16 or len(cases) != 128 or sum(c['split'] == 'train' for c in cases) != 96:
        raise ValueError('Fixed16 catalog/96 TRAIN/32 evaluation required')
    for case in cases:
        base.validate_request(case['request'])
    return freeze, products, cases


def prepare(directory, prior, integration_manifest, *, catalog_binding_factory=None):
    """CPU-only preflight; hash eval labels but never parse them."""
    from tokenizers import Tokenizer
    from .store_demo_catalog import DemoCatalogBinding
    directory, prior = Path(directory), Path(prior)
    freeze, products, cases = read_public(directory)
    snapshot = load_snapshot()
    binding_factory = DemoCatalogBinding if catalog_binding_factory is None else catalog_binding_factory
    binding = binding_factory(integration_manifest)
    binding.assert_current_sources(snapshot['receipt'])
    image_helpers = {}
    if binding_factory.__module__ == 'experiments.ranker.pretrained.store_demo_image_binding_v1':
        image_helpers = {name: data.sha(HERE.parents[2] / name) for name in
            ('scripts/prepare-product-webp.mjs', 'scripts/prepare-product-responsive-webp.mjs')}
    if products != snapshot['inputs']['products']:
        raise ValueError('Current16 public catalog differs from frozen v5')
    report = json.loads((prior / 'report.json').read_text()); packet = json.loads((prior / 'packet.json').read_text())
    if (report['status'] != 'completed' or report['nativeForwards'] != 64 or
            data.sha(prior / 'report.json') != freeze['v3ReportSha256'] or data.sha(prior / 'packet.json') != report['packetSha256']):
        raise ValueError('Exact completed v3 source report/packet required')
    provenance = report['featureProvenance']; model_path = Path(packet['model']['modelPath'])
    assert_model_files(model_path, provenance)
    if any(data.sha(model_path / name) != expected for name, expected in provenance['tokenizerFiles'].items()):
        raise ValueError('Verified v3 tokenizer changed')
    tokenizer = Tokenizer.from_file(str(model_path / 'tokenizer.json'))
    prior_rows = {(r['role'], r['caseId']): r for r in packet['records']}
    rows = []
    for i, product in enumerate(products):
        row = deepcopy(prior_rows['catalog', str(i)])
        if row['text'] != base.product_text(product):
            raise ValueError('Exact catalog text/order changed; cached feature reuse forbidden')
        row['origin'] = 'verified-v3-reuse'; rows.append(row)
    for case in cases:
        text = base.request_text(case['request'])
        if case['caseId'] in freeze['reuseTrainIds']:
            row = deepcopy(prior_rows['request', case['caseId']])
            if row['split'] != 'train' or row['text'] != text:
                raise ValueError('Only exact original TRAIN text may reuse features')
            row['origin'] = 'verified-v3-reuse'
        else:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if not 0 < len(ids) <= 512:
                raise ValueError('Whole live request must fit512 tokens, no truncation')
            row = {'role': 'request', 'member': [VERSION, case['caseId']], 'caseId': case['caseId'],
                'split': case['split'], 'text': text, 'textSha256': digest(text),
                'tokenIds': ids, 'tokenSha256': digest(canonical(ids)), 'tokens': len(ids),
                'batchShape': [1, len(ids)], 'origin': 'new-native-required'}
        rows.append(row)
    reuse = [r for r in rows if r['origin'] == 'verified-v3-reuse']
    if len(rows) != 144 or len(reuse) != 48 or sum(r['split'] == 'evaluation' for r in rows) != 32:
        raise ValueError('Truthful48 reused/96 new/144 total roster required')
    store = FeatureStore.open_existing(prior / 'features', provenance)
    reuse_pins = {}
    for row in reuse:
        store.load(row)  # exact raw key/provenance, shape, vector and artifact hashes
        for path in store.paths(row['role'], row['member']):
            reuse_pins[str(path)] = data.sha(path)
    return {'version': VERSION, 'preparationVersion': PREPARATION_VERSION,
        'catalogBinding': {'module': binding_factory.__module__, 'class': binding_factory.__name__,
            'manifestVersion': json.loads(Path(integration_manifest).read_text())['version'],
            'helperSourceSha256': image_helpers,
            'policy': 'explicit chosen verifier; original strict rules unchanged, no fallback'}, 'config': deepcopy(CONFIG), 'sourceSha256': sources(),
        'datasetFreezeSha256': data.sha(directory / 'FREEZE.json'), 'datasetFilesSha256': deepcopy(freeze['files']), 'integrationManifest': str(integration_manifest),
        'integrationManifestSha256': data.sha(integration_manifest), 'prior': str(prior),
        'priorReportSha256': data.sha(prior / 'report.json'), 'priorPacketSha256': data.sha(prior / 'packet.json'),
        'model': packet['model'], 'featureProvenance': provenance, 'reusedFeatureFilesSha256': reuse_pins,
        'records': rows, 'evaluationLabelsParsed': False, 'nativeExecuted': False,
        'tokenRange': [min(r['tokens'] for r in rows), max(r['tokens'] for r in rows)],
        'reviewGoals': freeze['minimumReviewGoals'], 'limitations': freeze['limitations']}


def assert_packet_pins(packet, directory, manifest):
    if data.sha(Path(directory) / 'FREEZE.json') != packet['datasetFreezeSha256']:
        raise ValueError('Original data freeze changed')
    if any(data.sha(Path(directory) / name) != expected for name, expected in packet['datasetFilesSha256'].items()):
        raise ValueError('Original frozen dataset file changed')
    prior = Path(packet['prior'])
    if data.sha(prior / 'report.json') != packet['priorReportSha256'] or data.sha(prior / 'packet.json') != packet['priorPacketSha256']:
        raise ValueError('Original v3 feature/head report changed')
    if any(data.sha(HERE.parents[2] / name) != expected for name, expected in packet['catalogBinding']['helperSourceSha256'].items()):
        raise ValueError('Pinned image-verification helper changed')
    if data.sha(manifest) != packet['integrationManifestSha256']:
        raise ValueError('Reviewed integration manifest changed')


def fit_head(query, catalog, targets, case_ids, packet_sha):
    """No evaluation inputs/labels accepted by this bounded fitting function."""
    if query.shape != (96, 4096) or catalog.shape != (16, 4096) or targets.shape != (96,):
        raise ValueError('Exact TRAIN tensor shapes required')
    if not torch.equal(torch.bincount(targets, minlength=16), torch.full((16,), 6)):
        raise ValueError('Six TRAIN targets per function required')
    means = {'query': query.double().mean(0).float(), 'catalog': catalog.double().mean(0).float()}
    metadata = {'queryMembers': case_ids, 'catalogMembers': list(range(16)),
        'meanSha256': {k: tensor_digest(v) for k, v in means.items()}, 'vocabulary': list(base.VOCAB),
        'config': deepcopy(CONFIG), 'packetSha256': packet_sha, 'trainTargetsSha256': digest(canonical(targets.tolist())),
        'trainingScope': 'known16 catalog,96 public single-function TRAIN paraphrases; no hidden profiles'}
    with cpu_threads():
        learned, random = base.FunctionHead(means), base.FunctionHead(means)
        optimizer = torch.optim.AdamW(learned.parameters(), lr=.03, weight_decay=.001); losses = []
        for step in range(80):
            optimizer.zero_grad()
            loss = (F.cross_entropy(learned.logits('query', query), targets) +
                F.cross_entropy(learned.logits('catalog', catalog), torch.arange(16))) / 2
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite TRAIN loss')
            loss.backward(); optimizer.step()
            losses.append({'step': step + 1, 'loss': float(loss.detach())})
    return learned.eval(), random.eval(), metadata, losses


def predictions(models, queries, catalog, cases, products):
    result = {}
    with cpu_threads(), torch.inference_mode():
        for name, head in models.items():
            catp = head.probabilities('catalog', catalog)
            result[name] = {}
            for index, case in enumerate(cases):
                qp = head.probabilities('query', queries[index:index + 1])[0]
                scores = (catp @ qp).tolist(); order = sorted(range(16), key=lambda i: (-scores[i], i))
                result[name][case['caseId']] = {'scores16': scores, 'rawRank': order}
        result['prior'] = {c['caseId']: {'scores16': [1 / 16] * 16, 'rawRank': list(range(16))} for c in cases}
        # Shuffle fresh query rows only, not labels or known answers.
        result['permuted-query'] = {}
        for index, case in enumerate(cases):
            source = cases[(index + CONFIG['permutedQueryOffset']) % len(cases)]['caseId']
            result['permuted-query'][case['caseId']] = {**deepcopy(result['learned'][source]), 'querySource': source}
    return result


def score_predictions(predicted, labels):
    targets = {r['caseId']: r['targetProductIndex'] for r in labels}; result = {}
    for arm, rows in predicted.items():
        if set(rows) != set(targets):
            raise ValueError('Exact frozen evaluation label/prediction roster required')
        by_function = {i: {'correct': 0, 'cases': 0} for i in range(16)}
        for rid, row in rows.items():
            target = targets[rid]; by_function[target]['cases'] += 1
            by_function[target]['correct'] += row['rawRank'][0] == target
        result[arm] = {'cases': len(rows), 'topOneCorrect': sum(r['rawRank'][0] == targets[rid] for rid, r in rows.items()),
            'targetInTopSix': sum(targets[rid] in r['rawRank'][:6] for rid, r in rows.items()), 'perFunction': by_function,
            'sameTargetQuerySources': sum(targets.get(r.get('querySource')) == targets[rid] for rid, r in rows.items() if 'querySource' in r),
            'misses': [{'caseId': rid, 'target': targets[rid], 'actualTopOne': r['rawRank'][0],
                'targetRank': r['rawRank'].index(targets[rid]) + 1} for rid, r in rows.items() if r['rawRank'][0] != targets[rid]]}
    return result


def guard_checks(probes, products, snapshot):
    results = []
    for probe in probes:
        try:
            request, policy = checked_request(probe['request'])
            order, receipts = guarded_order([0.] * 16, request, policy, products)
            cards = action_bindings(request, order[:6], snapshot)
            passed = (not probe.get('expectedRequestError') and
                (not probe.get('expectedEmpty') or not cards) and
                ('expectedFirst' not in probe or bool(cards) and cards[0]['productIndex'] == probe['expectedFirst']) and
                ('expectedColor' not in probe or bool(cards) and cards[0]['facts']['selectedColor'] == probe['expectedColor']))
            results.append({'caseId': probe['caseId'], 'passed': passed, 'selected': [c['productIndex'] for c in cards], 'guards': receipts})
        except ValueError as error:
            results.append({'caseId': probe['caseId'], 'passed': probe.get('expectedRequestError', False), 'requestError': str(error)})
    return results


def run(out, directory, prior, manifest, packet_path, *, execute=False, catalog_binding_factory=None):
    """No native action unless execute=True; service never stopped/bypassed."""
    out, directory = Path(out), Path(directory)
    if out.exists():
        raise FileExistsError('Unused output required; preserve earlier attempts')
    packet = prepare(directory, prior, manifest, catalog_binding_factory=catalog_binding_factory)
    if json.loads(Path(packet_path).read_text()) != packet:
        raise ValueError('Frozen CPU preflight changed')
    if not execute:
        return {'status': 'cpu-prepared-only', 'nativeExecuted': False, 'newNativeForwards': 96}
    from .store_ar_comparison_run_v1 import demo_listening
    if demo_listening():
        raise RuntimeError('Demo8766 is listening; owner must gracefully stop it first')
    from .catalog_fusion.generalization_v2 import native_owner_lock
    from .catalog_fusion.runtime import NativeEncoder, verify_model
    from .semantic_copy_training import parameter_hashes
    encoder = None; before = None; observations = []; started = time.perf_counter(); report = {'version': VERSION,
        'status': 'running', 'config': CONFIG, 'nativeForwards': 0, 'reusedFeatures': 48, 'totalFeatureVectors': 144,
        'servingCompatibility': CONFIG['runtimeCompatibility'], 'catalogBinding': packet['catalogBinding'], 'promoted': False,
        'evaluationRoster': [{'caseId': r['caseId'], 'status': 'unattempted'} for r in packet['records'] if r['split'] == 'evaluation']}
    with native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), 'root'):
        if demo_listening():
            raise RuntimeError('Demo8766 became active; refuse native load')
        out.mkdir(parents=True, exist_ok=False); write_json(out / 'packet.json', packet)
        for name in SOURCES:
            target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((HERE / name).read_bytes())
        for name in packet['catalogBinding']['helperSourceSha256']:
            target = out / 'source' / 'repo' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((HERE.parents[2] / name).read_bytes())
        def save():
            temporary = out / 'report.tmp'; temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            temporary.replace(out / 'report.json')
        save(); alarms = ExitStack()
        try:
            from .store_copy_fit_v2 import deadline
            alarms.enter_context(deadline(180, 'v5 native load/TRAIN extraction/fit/fresh evaluation lifecycle'))
            provenance = verify_model(packet['model']['modelPath'], packet['model'])
            if any(provenance[k] != packet['featureProvenance'][k] for k in ('modelRevision', 'weightsSha256', 'tokenizerSha256', 'templateSha256', 'quantization')):
                raise ValueError('Fresh verified model differs from reusable feature provenance')
            # Use the exact original provenance for raw feature keys; fresh hashes
            # independently prove identical bytes, while timing/stat receipts differ.
            provenance = packet['featureProvenance']; store = FeatureStore(out / 'features', provenance)
            oldstore = FeatureStore.open_existing(Path(prior) / 'features', provenance)
            lookup = {(r['role'], r['caseId']): r for r in packet['records']}
            for row in packet['records']:
                if row['origin'] == 'verified-v3-reuse':
                    values, original_receipt = oldstore.load(row)
                    store.save(row, values, original_receipt['timing'])
            encoder = NativeEncoder(packet['model']['modelPath'])
            before = parameter_hashes(encoder.backbone.model.language_model)
            if len(before) != 927:
                raise ValueError('Expected927 source tensors')
            def extract(row):
                if time.perf_counter() - started > 180:
                    raise TimeoutError('Fixed180-second native lifecycle exceeded')
                report['activeRecord'] = {'caseId': row['caseId'], 'split': row['split'], 'role': row['role']}
                values, timing = encoder.encode(row['text'], row)
                report['nativeForwards'] += 1
                store.save(row, values, timing)
                if time.perf_counter() - started > 180:
                    raise TimeoutError('Fixed180-second native lifecycle exceeded after encode')
                observations.append({'caseId': row['caseId'], 'split': row['split'], 'timing': timing})
                report['serializedNewFeatures'] = len(observations); save()
                return values
            for row in packet['records']:
                if row['split'] == 'train' and row['origin'] == 'new-native-required':
                    extract(row)
            _, products, cases = read_public(directory); train = [c for c in cases if c['split'] == 'train']
            assert_packet_pins(packet, directory, manifest)
            label_rows = json.loads((directory / 'train-labels.json').read_text()); targets = {r['caseId']: r['targetProductIndex'] for r in label_rows}
            if set(targets) != {c['caseId'] for c in train}:
                raise ValueError('Exact96 TRAIN labels required')
            q = torch.cat([store.load(lookup['request', c['caseId']])[0] for c in train])
            c = torch.cat([store.load(lookup['catalog', str(i)])[0] for i in range(16)])
            learned, random, metadata, losses = fit_head(q, c, torch.tensor([targets[t['caseId']] for t in train]),
                [t['caseId'] for t in train], data.sha(out / 'packet.json'))
            heads = {}; restored = {}
            for name, head in [('learned', learned), ('random', random)]:
                path = out / (name + '.pt'); checksum = base.save_head(path, head, metadata)
                restored[name] = base.load_head(path, checksum, metadata)
                heads[name] = {'file': path.name, 'sha256': checksum, 'metadata': metadata}
            old_report = json.loads((Path(prior) / 'report.json').read_text()); oldhead = old_report['heads']['learned']
            restored['frozen-v3'] = base.load_head(Path(prior) / oldhead['file'], oldhead['sha256'], oldhead['metadata'])
            write_json(out / 'HEADS_FROZEN.json', {'heads': heads, 'trainingLosses': losses, 'evaluationLabelsRead': False})
            report['heads'] = heads; endpoints = {}; evaluations = [r for r in cases if r['split'] == 'evaluation']; eval_values = []
            for case in evaluations:
                case_record = next(r for r in report['evaluationRoster'] if r['caseId'] == case['caseId'])
                case_record['status'] = 'running'
                begin = time.perf_counter(); values = extract(lookup['request', case['caseId']]); eval_values.append(values)
                verified = torch.cat([store.load(lookup['catalog', str(i)])[0] for i in range(16)])
                with cpu_threads(), torch.inference_mode():
                    scores = (restored['learned'].probabilities('catalog', verified) @ restored['learned'].probabilities('query', values)[0]).tolist()
                order, guards = guarded_order(scores, case['request'], {'budgetExclusive': False, 'waterproofRequired': False}, products)
                actions = action_bindings(case['request'], order[:6], load_snapshot())
                payload = {'scores16': scores, 'productIndices': order[:6], 'cards': actions, 'guards': guards}
                encoded = canonical(payload).encode()
                endpoints[case['caseId']] = {'freshEncodeToSerializedActionsMs': (time.perf_counter() - begin) * 1000,
                    'payloadSha256': hashlib.sha256(encoded).hexdigest(), 'payload': payload}
                case_record.update(status='completed', endpoint=endpoints[case['caseId']]); save()
            predicted = predictions(restored, torch.cat(eval_values), c, evaluations, products)
            write_json(out / 'PREDICTIONS_FROZEN.json', {'predictions': predicted, 'evaluationLabelsRead': False,
                'headsFreezeSha256': data.sha(out / 'HEADS_FROZEN.json')})
            for case in evaluations:
                if predicted['learned'][case['caseId']]['scores16'] != endpoints[case['caseId']]['payload']['scores16']:
                    raise AssertionError('Fresh endpoint and cached evaluation scoring differ')
            assert_packet_pins(packet, directory, manifest)
            labels = json.loads((directory / 'evaluation-labels.json').read_text())
            guards = guard_checks(json.loads((directory / 'guard-probes.json').read_text()), products, load_snapshot())
            report.update(status='completed-not-promoted', results=score_predictions(predicted, labels), guardChecks=guards,
                freshRequestEndpoints=endpoints, freshRequestMedianMs=statistics.median(r['freshEncodeToSerializedActionsMs'] for r in endpoints.values()),
                knownLanguageGaps=guard_checks(json.loads((directory / 'known-language-gaps.json').read_text()), products, load_snapshot()),
                featureProvenance=provenance, packetSha256=data.sha(out / 'packet.json'),
                latencyScope='fresh native query encode, immutable feature receipt,16 verified catalog reads, learned head, deterministic guards and serialized real actions; no copy/browser/image time')
            if len(observations) != 96 or not all(g['passed'] for g in guards):
                raise AssertionError('Fixed native forward roster or hard guard checks failed')
            assert_packet_pins(packet, directory, manifest)
            if any(data.sha(out / head['file']) != head['sha256'] for head in heads.values()):
                raise ValueError('Frozen v5 head checkpoint changed after reload')
            if sources() != packet['sourceSha256'] or any(data.sha(p) != expected for p, expected in packet['reusedFeatureFilesSha256'].items()):
                raise ValueError('Frozen source/reused feature files changed')
        except BaseException as error:
            for row in report['evaluationRoster']:
                if row['status'] == 'running':
                    row.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
            report.update(status='failed-not-promoted', error={'type': type(error).__name__, 'message': str(error)})
        finally:
            alarms.close()
            if encoder is not None:
                try:
                    after = parameter_hashes(encoder.backbone.model.language_model)
                    report['frozen927HashesUnchanged'] = before is not None and after == before
                    if before is None or after != before:
                        report.update(status='failed-not-promoted', integrityError='Frozen927 source hashes unavailable/changed')
                    report['memory'] = encoder.memory()
                except BaseException as error:
                    report.update(status='failed-not-promoted', shutdownError={'type': type(error).__name__, 'message': str(error)})
                finally:
                    try:
                        encoder.close()
                    except BaseException as error:
                        report.update(status='failed-not-promoted', closeError={'type': type(error).__name__, 'message': str(error)})
            if report['status'] == 'completed-not-promoted' and report.get('frozen927HashesUnchanged') is True:
                export = {'version': 'store-rank-head-v5-export.v1', 'scope': 'known16 single-function paraphrase adaptation; not promoted',
                    'head': report['heads']['learned'], 'sourceBaseRun': str(prior),
                    'sourceBaseReportSha256': packet['priorReportSha256'], 'catalogBinding': packet['catalogBinding'], 'featureProvenance': packet['featureProvenance'],
                    'originalCatalogFeatureFilesSha256': {p: checksum for p, checksum in packet['reusedFeatureFilesSha256'].items() if '/catalog/' in p},
                    'frozen927HashesUnchanged': True, 'newNativeForwards': report['nativeForwards'],
                    'totalFeatureVectors': 144, 'requestEncodingUnchanged': True, 'teacherLookupAtServing': False,
                    'integrationRequirement': 'NEW explicit adapter may replace verified v3 session head; old StoreRankSession report64 guard intentionally unchanged'}
                try:
                    write_json(out / 'serving-head.json', export)
                    report['servingHeadExportSha256'] = data.sha(out / 'serving-head.json')
                except BaseException as error:
                    report.update(status='failed-not-promoted', exportError={'type': type(error).__name__, 'message': str(error)})
            report.update(nativeObservations=observations, wholeWallMs=(time.perf_counter() - started) * 1000)
            save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run')); parser.add_argument('--data', type=Path, default=data.DATA)
    parser.add_argument('--prior', type=Path, default=data.PRIOR); parser.add_argument('--integration-manifest', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True); parser.add_argument('--out', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--catalog-binding', choices=('standard', 'image-delivery'), default='standard')
    args = parser.parse_args()
    catalog_binding_factory = None
    if args.catalog_binding == 'image-delivery':
        from .store_demo_image_binding_v1 import ImageDeliveryCatalogBinding
        catalog_binding_factory = ImageDeliveryCatalogBinding
    if args.action == 'prepare':
        packet = prepare(args.data, args.prior, args.integration_manifest, catalog_binding_factory=catalog_binding_factory); write_json(args.packet, packet)
        print(canonical({'newNativeForwards': 96, 'totalFeatureVectors': 144, 'tokenRange': packet['tokenRange']}))
    else:
        if args.out is None:
            raise ValueError('Unused --out required')
        result = run(args.out, args.data, args.prior, args.integration_manifest, args.packet, execute=args.run, catalog_binding_factory=catalog_binding_factory)
        print(canonical(result['status'])); return int(result['status'].startswith('failed'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
