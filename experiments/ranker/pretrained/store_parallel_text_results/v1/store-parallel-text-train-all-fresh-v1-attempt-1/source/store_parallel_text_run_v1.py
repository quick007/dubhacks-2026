"""Explicit exclusive native extraction + CPU parallel text candidate pilot.

Default is CPU preflight. --run is required. This driver never stops a service,
changes live serving, downloads weights, reads frontier relevance labels or
reuses an old output directory. Root is the sole native GPU owner.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
from statistics import median
import time

import torch

from .store_catalog_v1 import canonical, load_snapshot
from . import store_parallel_text_data_v1 as data
from . import store_parallel_text_head_v1 as model
from .store_demo_engine import checked_request, ranking_input, guarded_order
from .store_rank_session import action_bindings

ROOT = Path(__file__).resolve().parent
PACKET = ROOT / 'store_parallel_text_v1'
FRONTIER = ROOT / 'store_speed_quality_frontier_v1'
DEFAULT_MANIFEST = ROOT / 'store_demo_integration_v3/manifest.json'
VERSION = 'store-parallel-text-native-run.v1'
CONFIG = {'trainingSteps': 160, 'learningRate': .004, 'hiddenWidth': 128,
          'cpuThreads': 1, 'featureRows': 48, 'featureTokenCap': 512,
          'frontierRequests': 32, 'cardsPerRequest': 6, 'fieldsPerRequest': 20,
          'publicStyle': 'plain', 'styleCounterfactualRequests': 8,
          'requestTimeoutSeconds': 30, 'extractionTimeoutSeconds': 180}
SOURCES = ('store_parallel_text_run_v1.py', 'store_parallel_text_head_v1.py',
           'store_parallel_text_data_v1.py', 'store_demo_engine.py', 'store_rank_session.py',
           'store_demo_catalog.py', 'backbone.py', 'semantic_copy_training.py',
           'store_copy_fit_v2.py', 'store_copy_wide_v2.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def demo_listening():
    try:
        with socket.create_connection(('127.0.0.1', 8766), timeout=.5):
            return True
    except ConnectionRefusedError:
        return False
    except OSError as error:
        raise RuntimeError('Cannot establish whether8766 is stopped; refuse native launch') from error


def load_packet(directory=PACKET):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schemaVersion'] != data.VERSION or manifest['sourceSha256'] != sha(ROOT / 'store_parallel_text_data_v1.py'):
        raise ValueError('Prepared packet schema/data source changed')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha(directory / name) != expected:
            raise ValueError('Prepared packet file changed: ' + name)
    expected = data.build(load_snapshot())
    actual = {'featureRecords': json.loads((directory / 'public-feature-contract.json').read_text())['records'],
              'vocabulary': json.loads((directory / 'vocabulary.json').read_text()),
              'labels': {s: json.loads((directory / f'labels.{s}.json').read_text()) for s in ('train', 'eval')},
              'cases': {s: json.loads((directory / f'cases.{s}.json').read_text()) for s in ('train', 'eval')},
              'audit': json.loads((directory / 'audit.json').read_text())}
    for key, value in actual.items():
        if json.loads(canonical(value)) != json.loads(canonical(expected[key])):
            raise ValueError('Prepared packet differs from exact public source renderer: ' + key)
    return expected


def frontier_cases(directory=FRONTIER):
    """Only this public file is read; gold-bearing case IDs never enter model."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    path = directory / 'public-requests.json'
    if (manifest['version'] != 'store-speed-quality-frontier.v1' or manifest['cases'] != 32 or
            sha(path) != manifest['files']['public-requests.json']):
        raise ValueError('Frozen public frontier file changed')
    cases = json.loads(path.read_text())
    if len(cases) != 32 or len({r['caseId'] for r in cases}) != 32:
        raise ValueError('Exactly32 distinct public frontier cases required')
    for case in cases:
        if (set(case) != {'caseId', 'request', 'requestedCardCount', 'requestedCopyFields'} or
                case['requestedCardCount'] != 6 or case['requestedCopyFields'] != 20):
            raise ValueError('Frozen six-card/twenty-field public case shape required')
        checked_request(case['request'])
    return cases


def cpu_preflight(manifest=DEFAULT_MANIFEST, *, tokenizer=None):
    from .store_demo_catalog import DemoCatalogBinding
    packet = load_packet(); cases = frontier_cases()
    binding = DemoCatalogBinding(manifest)
    binding.assert_current_sources(load_snapshot()['receipt'])
    if tokenizer is None:
        from transformers import AutoTokenizer
        from .semantic_copy_feature_fit import MODEL
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
    rows = []
    for record in packet['featureRecords']:
        ids = tokenizer.encode(record['text'], add_special_tokens=False)
        if not 0 < len(ids) <= CONFIG['featureTokenCap']:
            raise ValueError('Complete public fact input exceeds512tokens; no truncation')
        rows.append({'featureKey': record['featureKey'], 'textSha256': record['textSha256'],
                     'tokenIds': ids, 'tokens': len(ids), 'tokenIdsSha256': hashlib.sha256(canonical(ids).encode()).hexdigest()})
    return {'version': VERSION, 'nativeExecution': False, 'config': CONFIG, 'cases': cases, 'featureRows': rows,
        'sourceSha256': {n: sha(ROOT / n) for n in SOURCES},
        'packetManifestSha256': sha(PACKET / 'manifest.json'),
        'frontierPublicSha256': sha(FRONTIER / 'public-requests.json'),
        'integrationManifestSha256': sha(manifest), 'dataAudit': packet['audit'],
        'inferenceBoundary': 'request only; case ID/archive ordinal/gold/teacher labels never enter rank or generation',
        'copyBoundary': 'public bound facts + cached frozen Qwen features + explicit public style; no query intent copy claim',
        'timingPolicy': 'warm resident model and verified in-memory offline catalog features; fresh query encoding through validated canonical JSON for first3/full6; excludes startup/offline48features/training/archive/browser/images',
        'warmupPolicy': 'one unscored fresh-query six-card request before32 timed requests; no response cache',
        'qualityPolicy': 'versioned short-body factual screen and title+body-per-card proxy; raw output retained; separate independent review required, no human conversion claim'}


def extract_features(engine, records, preflight_rows, out):
    from .semantic_copy_training import disabled
    from .store_copy_wide_v2 import guard_memory
    from .store_copy_fit_v2 import deadline
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    values, receipts = [], []; native_start = time.perf_counter()
    initial_forwards = engine.bb.forward_calls
    if len(records) != 48 or len(preflight_rows) != 48:
        raise ValueError('Exactly48 color-specific source records required')
    with disabled(engine.bb.model, engine.adapter), deadline(CONFIG['extractionTimeoutSeconds'], 'parallel text48feature extraction'):
        if engine.bb.model.layers[-1].mlp.down_proj is not engine.adapter.linear:
            raise ValueError('Copy adapter was not disabled for frozen feature extraction')
        for index, (record, expected) in enumerate(zip(records, preflight_rows, strict=True)):
            engine.binding.assert_current_sources(engine.session.snapshot['receipt'])
            ids = engine.bb.tokenizer.encode(record['text'], add_special_tokens=False)
            if ids != expected['tokenIds'] or expected['featureKey'] != record['featureKey']:
                raise ValueError('Native feature tokenizer/key differs from complete CPU preflight')
            vector, timing = engine.bb.encode([record['text']])
            if (list(vector.shape) != [1, 4096] or timing['batchRows'] != 1 or
                    timing['paddedTokensPerRow'] != len(ids) or timing['backboneForwardCalls'] != 1 or timing['truncatedRows'] != 0):
                raise ValueError('Exact complete B1 frozen feature contract failed')
            values.append(vector.detach().cpu().float().clone())
            receipt = {'index': index, 'featureKey': record['featureKey'], 'text': record['text'],
                'textSha256': record['textSha256'], 'tokenIds': ids, 'tokenIdsSha256': expected['tokenIdsSha256'],
                'featureTensorSha256': model.tensor_sha(values[-1]), 'timing': timing, 'memory': guard_memory(engine.bb),
                'copyAdapterEnabled': False, 'teacherTextRead': False, 'evidenceSource': 'native-frozen-qwen'}
            receipts.append(receipt); dump(out / f'{index:03d}.json', receipt)
            print(canonical({'event': 'parallel-text-native-feature', 'completed': index + 1, 'total': 48}), flush=True)
    forwards = engine.bb.forward_calls - initial_forwards
    if forwards != 48:
        raise ValueError('Native feature forward count differs from48')
    provenance = {'evidenceSource': 'native-frozen-qwen', 'modelRevision': engine.bb.revision,
        'copyAdapterEnabled': False, 'nativeForwards': forwards, 'truncatedRows': 0,
        'pooling': 'last-valid-token unit L2', 'modelVerification': engine.model_verification,
        'offlineExtractionWallMs': (time.perf_counter() - native_start) * 1000,
        'rowReceiptSha256': {f'{i:03d}.json': sha(out / f'{i:03d}.json') for i in range(48)}}
    return model.make_feature_bundle(records, torch.cat(values), provenance)


def runtime_rows(cards, style):
    if len(cards) != 6 or style not in data.STYLES:
        raise ValueError('Explicit six-card workload and public style required; no padding/shortfall fallback')
    rows = []
    for section in range(2):
        group = cards[section * 3:section * 3 + 3]
        rows.append({'field': f'S{section + 1}', 'input': data.serving_row('section', style, section_index=section),
                     'featureKeys': [data.public_record(c['facts'])['featureKey'] for c in group]})
        for card_index in range(section * 3, section * 3 + 3):
            for prefix, kind in zip('TBC', data.KINDS[1:], strict=True):
                source = data.serving_row(kind, style, cards[card_index]['facts'])
                rows.append({'field': f'{prefix}{card_index + 1}', 'input': source,
                             'featureKeys': [source['source']['featureKey']]})
    return rows


def endpoint(cards, rows, generated, *, card_limit):
    selected = cards[:card_limit]
    if card_limit not in (3, 6) or len(rows) != 20 or len(generated['fields']) != 20:
        raise ValueError('Explicit first3/full6 endpoint required')
    by_field = {r['field']: result for r, result in zip(rows, generated['fields'], strict=True)}
    def text(field):
        result = by_field[field]
        return result['text'] if result['accepted'] else None
    sections = []
    for start in range(0, len(selected), 3):
        contents = []
        for i in range(start, start + 3):
            card = cards[i]; facts, action = card['facts'], card['action']
            contents.append({'productId': action['productId'], 'variantId': action['variantId'],
                'originalName': facts['name'], 'selectedColor': facts['selectedColor'], 'itemPrice': facts['itemPrice'],
                'image': deepcopy(action['image']), 'title': text(f'T{i + 1}'), 'badge': text(f'B{i + 1}'),
                'bodyCopy': text(f'C{i + 1}'), 'stock': None, 'size': None, 'shipping': None})
        sections.append({'heading': text(f'S{start // 3 + 1}'), 'cards': contents})
    fields = [r['field'] for r in rows[:10 if card_limit == 3 else 20]]
    acceptance = [{'field': field, 'accepted': by_field[field]['accepted'],
                   'rejections': by_field[field]['rejections']} for field in fields]
    useful = all(contents['title'] is not None and contents['bodyCopy'] is not None for section in sections for contents in section['cards'])
    return {'schemaVersion': VERSION, 'sections': sections, 'cardCount': card_limit,
        'validation': {'requiredFields': len(fields), 'acceptedFields': sum(r['accepted'] for r in acceptance),
            'allFieldsAccepted': all(r['accepted'] for r in acceptance), 'titleBodyPerCardProxy': useful,
            'fields': acceptance, 'contract': 'short color+item-price body v1; differs from frozen copy-v2'},
        'scope': 'experimental learned parallel word head; original names/prices/action metadata canonical; rejected text null, no repair/template'}


def run_request(engine, head, feature_map, value, *, style='plain', clock=time.perf_counter):
    """Request-only inference API. No case ID, gold, teacher, or row target input."""
    started = clock(); request, policy = checked_request(value)
    ranked = engine.session.rank_request(ranking_input(request), copy_adapter=engine.adapter)
    order, guards = guarded_order(ranked['scores16'], request, policy, engine.session.products)
    cards = action_bindings(request, order[:6], engine.session.snapshot)
    if len(cards) != 6:
        raise ValueError('Fixed six-card benchmark has an eligible shortfall; refuse padding or a changed workload')
    rank_json = canonical({'order': order, 'guards': guards, 'cards': cards})
    rank_ready = (clock() - started) * 1000
    rows = runtime_rows(cards, style)
    generated = model.generate(head, model.inference_rows(rows), feature_map)
    engine.binding.assert_current_sources(engine.session.snapshot['receipt'])
    first = endpoint(cards, rows, generated, card_limit=3); first_json = canonical(first)
    first_ms = (clock() - started) * 1000
    full = endpoint(cards, rows, generated, card_limit=6); full_json = canonical(full)
    full_ms = (clock() - started) * 1000
    return {'status': 'completed-attempt', 'input': deepcopy(value), 'publicStyle': style,
        'effectiveRequest': request, 'ranked': ranked, 'order': order, 'guards': guards, 'cards': cards,
        'runtimeRows': rows, 'generated': generated, 'firstThree': first, 'fullSix': full,
        'serialized': {'rankSha256': hashlib.sha256(rank_json.encode()).hexdigest(),
            'firstThreeJson': first_json, 'fullSixJson': full_json,
            'firstThreeBytes': len(first_json.encode()), 'fullSixBytes': len(full_json.encode())},
        'timing': {'rankThroughJsonMs': rank_ready, 'attemptFirstThreeMs': first_ms, 'attemptFullSixMs': full_ms,
            'usefulFirstThreeMs': first_ms if first['validation']['titleBodyPerCardProxy'] else None,
            'usefulFullSixMs': full_ms if full['validation']['titleBodyPerCardProxy'] else None,
            'scope': 'contiguous fresh query Qwen encode+rank+guards+cached offline feature head+all-field validation+source freshness+canonical JSON; no startup/offline48features/training/archive/browser/images',
            'firstThreePolicy': 'all20slots generated/validated in one pass, then first3 and full6 serialized separately'}}


def summarize(receipts, roster):
    def med(key):
        values = [r['timing'][key] for r in receipts if r['timing'][key] is not None]
        return median(values) if values else None
    completed = sum(r['status'] == 'completed-attempt' for r in roster)
    return {'plannedRequests': len(roster), 'completedRequests': completed,
        'failedRequests': sum(r['status'] == 'failed-attempt' for r in roster),
        'unattemptedRequests': sum(r['status'] == 'unattempted' for r in roster),
        'runningRequests': sum(r['status'] == 'running' for r in roster),
        'sixCardsEveryRequest': completed == len(roster) and bool(receipts) and all(len(r['cards']) == 6 for r in receipts),
        'twentyFieldsEveryRequest': completed == len(roster) and bool(receipts) and all(r['generated']['fieldCount'] == 20 for r in receipts),
        'firstThreeAllFieldPasses': sum(r['firstThree']['validation']['allFieldsAccepted'] for r in receipts),
        'fullSixAllFieldPasses': sum(r['fullSix']['validation']['allFieldsAccepted'] for r in receipts),
        'firstThreeTitleBodyProxyPasses': sum(r['firstThree']['validation']['titleBodyPerCardProxy'] for r in receipts),
        'fullSixTitleBodyProxyPasses': sum(r['fullSix']['validation']['titleBodyPerCardProxy'] for r in receipts),
        'medianRankThroughJsonMs': med('rankThroughJsonMs'), 'medianAttemptFirstThreeMs': med('attemptFirstThreeMs'),
        'medianAttemptFullSixMs': med('attemptFullSixMs'), 'medianUsefulFirstThreeMs': med('usefulFirstThreeMs'),
        'medianUsefulFullSixMs': med('usefulFullSixMs'),
        'independentHumanQualityClaim': False, 'rankingRelevanceScoredHere': False,
        'latencyDenominator': 'completed attempts only; fixed planned/failed/unattempted counts retained separately',
        'scope': 'curated development frontier; no unseen generalization, physical paint or human-conversion claim'}


def run_candidate(out, *, run=False, manifest=DEFAULT_MANIFEST, _dependencies=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Output must be unused; preserve every prior attempt')
    deps = {} if _dependencies is None else _dependencies
    if run and deps.get('service_listening', demo_listening)():
        raise RuntimeError('Demo8766 is listening; refuse without stopping or bypassing service')
    preflight = deps.get('preflight', cpu_preflight)(manifest)
    out.mkdir(parents=True); dump(out / 'preflight.json', preflight)
    if not run:
        return {'status': 'cpu-preflight-only', 'nativeExecution': False, 'preflight': preflight}
    if _dependencies is None:
        from .catalog_fusion.generalization_v2 import native_owner_lock
        from .store_demo_engine import ResidentDemoEngine
        lock = lambda: native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), 'root')
        factory = lambda: ResidentDemoEngine(output=out / 'resident', width=10, integration_manifest=manifest)
    else:
        lock, factory = deps['native_lock'], deps['engine_factory']
    packet = deps.get('packet', load_packet)()
    extractor = deps.get('extractor', extract_features); fitter = deps.get('fitter', model.fit)
    feature_validator = deps.get('feature_validator', model.validate_bundle)
    requester = deps.get('requester', run_request)
    report = {'version': VERSION, 'status': 'started', 'nativeExecution': True, 'config': CONFIG,
              'requests': [{'caseId': c['caseId'], 'input': deepcopy(c['request']), 'status': 'unattempted',
                            'file': f'request-{i:03d}.json'} for i, c in enumerate(preflight['cases'])],
              'styleCounterfactuals': [], 'preflightSha256': sha(out / 'preflight.json')}
    engine = None; receipts = []; start = time.perf_counter()
    def save():
        report['summary'] = summarize(receipts, report['requests']); dump(out / 'results.json', report)
    save()
    try:
        with ExitStack() as ownership:
            ownership.enter_context(lock())
            try:
                if deps.get('service_listening', demo_listening)():
                    raise RuntimeError('Demo8766 became active before native load; refuse without automatic stop')
                engine = factory()
                if engine.startup.get('sourceTensorsHashed') != 927:
                    raise ValueError('Verified927 frozen tensors required')
                report['startup'] = engine.startup; save()
                bundle = extractor(engine, packet['featureRecords'], preflight['featureRows'], out / 'feature-rows')
                torch.save(bundle, out / 'features.pt')
                report['featureBundle'] = {'file': 'features.pt', 'sha256': sha(out / 'features.pt'),
                    'tensorSha256': bundle['featureTensorSha256'], 'provenance': bundle['provenance']}; save()
                head, fit_report = fitter(packet, bundle, steps=160, learning_rate=.004, width=128)
                dump(out / 'fit.json', fit_report); report['fit'] = fit_report
                torch.save({'schemaVersion': model.HEAD_VERSION, 'state_dict': head.state_dict(),
                    'vocabulary': list(head.vocabulary), 'width': head.width,
                    'trainingConfig': CONFIG, 'sourcePacketManifestSha256': preflight['packetManifestSha256']}, out / 'head.pt')
                report['checkpoint'] = {'file': 'head.pt', 'sha256': sha(out / 'head.pt')}
                feature_map = feature_validator(bundle, packet)
                # Require safe CPU reload parity before any reported benchmark.
                saved = torch.load(out / 'head.pt', map_location='cpu', weights_only=True)
                restored = model.ParallelTextHead(saved['vocabulary'], width=saved['width'])
                restored.load_state_dict(saved['state_dict'], strict=True)
                with model.bounded_cpu_threads(CONFIG['cpuThreads']):
                    heldout = model.evaluate_cases(head, packet['cases']['eval'], feature_map)
                    reloaded = model.evaluate_cases(restored, packet['cases']['eval'], feature_map)
                    a = [[f['slotIds'] for f in r['result']['fields']] for r in heldout['receipts']]
                    b = [[f['slotIds'] for f in r['result']['fields']] for r in reloaded['receipts']]
                    if a != b:
                        raise ValueError('Reloaded head slot predictions differ')
                    dump(out / 'heldout.json', heldout)
                    report['heldout'] = {k: v for k, v in heldout.items() if k != 'receipts'}
                    report['reloadExactSlotParity'] = True; save()
                    # First public request warmup is unscored and archived.
                    from .store_copy_fit_v2 import deadline
                    report['activeStage'] = 'unscored-warmup'; save()
                    try:
                        with deadline(CONFIG['requestTimeoutSeconds'], 'parallel text complete warmup request'):
                            warmup = requester(engine, restored, feature_map, deepcopy(preflight['cases'][0]['request']))
                    except BaseException as error:
                        dump(out / 'warmup-failure.json', {'status': 'failed-unscored-warmup',
                            'input': deepcopy(preflight['cases'][0]['request']),
                            'error': {'type': type(error).__name__, 'message': str(error)}})
                        report['warmupFailure'] = 'warmup-failure.json'
                        raise
                    dump(out / 'warmup.json', warmup)
                    report['warmupExcluded'] = 'warmup.json'
                    for index, case in enumerate(preflight['cases']):
                        report['activeRequest'] = {'index': index, 'caseId': case['caseId'], 'request': deepcopy(case['request'])}
                        report['activeStage'] = 'timed-fresh-query'
                        roster = report['requests'][index]; roster['status'] = 'running'
                        save()
                        attempt_started = time.perf_counter()
                        path = out / roster['file']
                        try:
                            with deadline(CONFIG['requestTimeoutSeconds'], 'parallel text fresh complete request'):
                                result = requester(engine, restored, feature_map, deepcopy(case['request']))
                        except BaseException as error:
                            failed = {'status': 'failed-attempt', 'input': deepcopy(case['request']),
                                'archiveCaseId': case['caseId'], 'failedAtMs': (time.perf_counter() - attempt_started) * 1000,
                                'error': {'type': type(error).__name__, 'message': str(error)}}
                            dump(path, failed)
                            roster.update(status='failed-attempt', sha256=sha(path), error=failed['error'], failedAtMs=failed['failedAtMs'])
                            save()
                            raise
                        receipts.append(result)
                        dump(path, result)
                        roster.update(status='completed-attempt', sha256=sha(path),
                            timing=result['timing'], selectedProductIndices=result['order'][:6])
                        if index < CONFIG['styleCounterfactualRequests']:
                            variants = []
                            for style in data.STYLES:
                                rows = runtime_rows(result['cards'], style)
                                generated = model.generate(restored, model.inference_rows(rows), feature_map)
                                variants.append({'publicStyle': style, 'rows': rows, 'generated': generated})
                            file = out / f'style-{index:03d}.json'; dump(file, {'sameSourceCards': result['cards'], 'variants': variants,
                                'scope': 'explicit public style counterfactual; cached selected facts, head-only timing, no fresh query encode'})
                            texts = [[f['text'] for f in v['generated']['fields']] for v in variants]
                            report['styleCounterfactuals'].append({'caseId': case['caseId'], 'file': file.name,
                                'changedFieldsPlainVsBrowse': sum(a != b for a, b in zip(texts[0], texts[1])),
                                'changedFieldsPlainVsCompact': sum(a != b for a, b in zip(texts[0], texts[2]))})
                        save()
                        print(canonical({'event': 'parallel-text-fresh-request', 'completed': index + 1,
                            'total': len(preflight['cases']), 'fullAttemptMs': result['timing']['attemptFullSixMs']}), flush=True)
                if len(receipts) != 32:
                    raise ValueError('Exactly32 frontier fresh-query attempts required')
                engine.binding.assert_current_sources(engine.session.snapshot['receipt'])
                if deps.get('source_hashes', lambda: {n: sha(ROOT / n) for n in SOURCES})() != preflight['sourceSha256']:
                    raise ValueError('Experiment source changed during native attempt')
                report['status'] = 'completed-candidate-not-promoted'
                report.pop('activeRequest', None)
                report.pop('activeStage', None)
            finally:
                # Keep native ownership through integrity hashes and teardown.
                if engine is not None:
                    report['shutdownIntegrity'] = engine.close()
                    if not all(report['shutdownIntegrity'].get(k) is True for k in ('frozen927HashesUnchanged', 'copyCheckpointUnchanged')):
                        report['status'] = 'failed-shutdown-integrity'
    except BaseException as error:
        report.update(status='failed-candidate', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        report['totalWallMsIncludingSetupOfflineFeaturesFitEvaluationArchive'] = (time.perf_counter() - start) * 1000
        save()
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--run', action='store_true', help='Explicit isolated native work; caller must first stop serving')
    args = parser.parse_args()
    result = run_candidate(args.out, run=args.run, manifest=args.manifest)
    print(canonical({'status': result['status'], 'nativeExecution': result['nativeExecution'], 'summary': result.get('summary')}))
    if args.run and result['status'] != 'completed-candidate-not-promoted':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
