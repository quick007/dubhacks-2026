"""Explicit experimental copy backend; existing native demo remains unchanged.

Parallel mode predicts fresh word slots using a TRAIN-ALL public-catalog head.
Generated-cache mode serves verified earlier native text and generates an entire
selected page on a miss. Neither mode repairs rejected words or uses templates.
Run only explicitly through this module's CLI, under the existing single worker.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
import time

from .store_catalog_v1 import canonical
from .store_demo_engine import (DEFAULT_COPY, DEFAULT_RANK, NEUTRAL_COPY_REQUEST,
    RequestCancelled, ResidentDemoEngine, checked_request, demo_layout, guarded_order,
    public_sections, ranking_input)
from .store_rank_session import action_bindings

VERSION = 'store-candidate-engine.v1'
HERE = Path(__file__).resolve().parent
PUBLIC_CONTRACT = HERE / 'store_parallel_text_v1/public-feature-contract.json'
TIMING_SCOPE = 'request admission through fresh Qwen query/ranking, hard guards, real variant binding, copy generation or verified cache read, validation and serialized server events; excludes startup, network, browser paint and images'
IMAGE_POLICY = 'existing deterministic storefront gallery; declared metadata; no learned image or pixel claim'
SOURCE_FILES = ('store_candidate_engine_v1.py', 'store_parallel_text_head_v1.py',
    'store_parallel_text_data_v1.py', 'store_parallel_text_demo_all_v1.py',
    'store_generated_copy_cache_v1.py', 'store_demo_engine.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_parallel_assets(candidate_run, *, feature_bundle=None, public_contract=None):
    """CPU load reads only model metadata/weights/features and public facts.

    In particular no packet loader, TRAIN-ALL-labels, replay, teacher or gold
    request file is consulted. Checksum mismatches stop startup before GPU load.
    """
    import torch
    from . import store_parallel_text_head_v1 as model
    from .store_parallel_text_demo_all_v1 import load_head
    from .store_parallel_text_data_v1 import public_record
    run = Path(candidate_run); manifest_path = run / 'serving-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('scope') != 'TRAIN-ALL public catalog demo; no heldout evaluation' or
            manifest.get('all48BindingsSupervised') is not True or manifest.get('runtimeTeacherLookup') is not False or
            manifest.get('heldoutQualityClaim') is not False):
        raise ValueError('Explicit public-only TRAIN-ALL serving export required; no heldout claim')
    head_path = Path(manifest['headFile'])
    feature_path = Path(feature_bundle) if feature_bundle is not None else Path(manifest['featureBundleFile'])
    contract_path = Path(public_contract) if public_contract is not None else Path(manifest['publicFeatureContractFile'])
    if head_path.resolve() != (run / 'head.pt').resolve():
        raise ValueError('Expected exact candidate head.pt artifact')
    pins = {str(head_path): manifest['headSha256'], str(feature_path): manifest['featureBundleSha256'],
        str(contract_path): manifest['publicFeatureContractSha256'], str(manifest_path): sha(manifest_path)}
    if any(sha(path) != expected for path, expected in pins.items()):
        raise ValueError('Parallel checkpoint/feature/public-fact checksum mismatch')
    contract = json.loads(contract_path.read_text()); records = contract['records']
    if (contract.get('schemaVersion') != 'parallel-word-public-runtime-features.v1' or
            contract.get('teacherOrLabelContent') is not False or
            len(records) != 48 or len({r['featureKey'] for r in records}) != 48 or
            any(public_record(r['facts']) != r for r in records)):
        raise ValueError('Exactly48 complete pure public fact records required')
    head, saved = load_head(head_path)
    if saved['featureBundleSha256'] != pins[str(feature_path)]:
        raise ValueError('Head was fitted on another feature bundle')
    bundle = torch.load(feature_path, map_location='cpu', weights_only=True)
    features = model.validate_bundle(bundle, {'featureRecords': records})
    return head, features, records, {'version': VERSION, 'kind': 'fresh-parallel-head',
        'checkpointSha256': pins[str(head_path)], 'featuresSha256': pins[str(feature_path)],
        'publicFactsSha256': pins[str(contract_path)], 'assetSha256': pins,
        'trainingScope': manifest['scope'], 'heldoutBindings': 0, 'teacherStringsReadAtServing': False,
        'offlineFeatures': '48 frozen-base Qwen fact encodings; no request/target features',
        'headSlots': 14, 'vocabularySize': len(head.vocabulary), 'style': 'plain'}


def parallel_rows(cards, style='plain'):
    """1..6 real cards; only full groups have the trained three-source heading.

    Short group headings abstain. We never invent extra cards or duplicate a
    feature to meet the trained section mean shape.
    """
    from .store_parallel_text_data_v1 import serving_row, public_record
    if not 1 <= len(cards) <= 6:
        raise ValueError('One to six real cards required')
    rows = []
    for start in range(0, len(cards), 3):
        group = cards[start:start + 3]
        if len(group) == 3:
            rows.append({'field': f'S{start // 3 + 1}',
                'input': serving_row('section', style, section_index=start // 3),
                'featureKeys': [public_record(c['facts'])['featureKey'] for c in group]})
        for i, card in enumerate(group, start + 1):
            for prefix, kind in zip('TBC', ('title', 'badge', 'copy'), strict=True):
                inp = serving_row(kind, style, card['facts'])
                rows.append({'field': f'{prefix}{i}', 'input': inp,
                    'featureKeys': [inp['source']['featureKey']]})
    return rows


def parallel_payload(cards, rows, generated, limit):
    """Serialize predicted text/nulls and canonical actions, never target words."""
    if not 0 <= limit <= len(cards) <= 6 or len(rows) != len(generated['fields']):
        raise ValueError('Exact card/field roster required')
    expected = parallel_rows(cards) if cards else []
    if rows != expected:
        raise ValueError('Exact source-conditioned parallel field roster required')
    by_field = {row['field']: value for row, value in zip(rows, generated['fields'], strict=True)}
    sections = []; validation = []
    def field_value(field):
        result = by_field.get(field)
        if result is None:
            validation.append({'field': field, 'accepted': False, 'rejections': ['short-section-heading-abstention']})
            return None
        from .store_parallel_text_data_v1 import field_rejections
        row = next(r for r in rows if r['field'] == field)
        inp = row['input']
        errors = list(dict.fromkeys(list(result['rejections']) + field_rejections(inp['kind'], result.get('text'), inp.get('source', {}).get('facts'))))
        if result.get('finishReason') != 'eos' or result.get('accepted') is not True:
            if not errors:
                errors.append('invalid-parallel-slot-receipt')
        validation.append({'field': field, 'accepted': not errors, 'rejections': errors})
        return result['text'] if not errors else None
    for start in range(0, limit, 3):
        heading = field_value(f'S{start // 3 + 1}'); contents = []
        for i in range(start, min(start + 3, limit)):
            card = cards[i]; facts, action = card['facts'], card['action']
            contents.append({'productId': action['productId'], 'variantId': action['variantId'],
                'productIndex': card['productIndex'], 'originalName': facts['name'],
                'selectedColor': facts['selectedColor'], 'itemPrice': facts['itemPrice'],
                'title': field_value(f'T{i + 1}'), 'badge': field_value(f'B{i + 1}'),
                'bodyCopy': field_value(f'C{i + 1}'), 'stock': None, 'size': None, 'shipping': None})
        sections.append({'heading': heading, 'cards': contents})
    accepted = sum(r['accepted'] for r in validation)
    return sections, {'requiredFields': len(validation), 'acceptedFields': accepted,
        'withheldFields': len(validation) - accepted, 'fields': validation,
        'contract': 'parallel word-head selected-color/item-price body gate v1; original identity in canonical metadata; rejected output null, no repair',
        'allFieldsAccepted': all(r['accepted'] for r in validation),
        'titleBodyPerCardProxy': all(c['title'] is not None and c['bodyCopy'] is not None for s in sections for c in s['cards'])}


def source_for_cache(cache):
    hits, misses = cache['initialHits'], cache['initialMisses']
    kind = 'generated-earlier' if not misses else 'mixed-native-cache' if hits else 'fresh-native'
    entries = cache['entries']
    source = {'kind': kind, 'generatedEarlier': bool(hits),
        'freshPerRequestGeneration': bool(misses),
        'initialHits': hits, 'initialMisses': misses,
        'nativeGenerationMsThisRequest': cache['nativeColdGenerationMsThisRequest'],
        'bindingSha256': cache['bindingSha256'], 'entries': entries,
        'disclosure': cache['disclosure']}
    ages = [e['ageMs'] for e in entries if e['generatedEarlier']]
    if ages:
        source['maxAgeMs'] = max(ages)
    return source


class CandidateDemoEngine(ResidentDemoEngine):
    def __init__(self, *, mode, candidate_run, feature_bundle=None, public_contract=None, **kwargs):
        if mode not in ('parallel', 'generated-cache'):
            raise ValueError('Explicit parallel or generated-cache mode required; base native CLI remains available')
        self.mode = mode; self.candidate_run = Path(candidate_run)
        self.candidate_sources = {name: sha(HERE / name) for name in SOURCE_FILES}
        if mode == 'parallel':
            self.head, self.feature_map, self.public_records, self.copy_metadata = load_parallel_assets(
                candidate_run, feature_bundle=feature_bundle, public_contract=public_contract)
        else:
            preflight = json.loads((self.candidate_run / 'preflight.json').read_text())
            report = json.loads((self.candidate_run / 'results.json').read_text())
            control_path = self.candidate_run / 'native-controls-width10/results.json'
            control = json.loads(control_path.read_text())
            if (control.get('status') != 'exact-controls-passed' or
                    sha(control_path) != report.get('sameWidthControlsSha256')):
                raise ValueError('Exact archived same-width native controls required for cache misses')
            self.cache_spec = preflight['producerSpec']; self.control_sha = sha(control_path)
            self.cache_source = Path(report.get('warmCacheReuse', {}).get('sourceDirectory', self.candidate_run))
            from .store_generated_copy_cache_run_v1 import verify_reuse
            self.cache_reuse = verify_reuse(self.cache_source, preflight)
            if self.cache_spec['nativeWidth'] != 10:
                raise ValueError('Reviewed cache native producer width10 required')
            self.copy_metadata = {'kind': 'generated-cache', 'candidateRun': str(self.candidate_run),
                'originalRunStatus': report['status'], 'sameWidthControlsSha256': self.control_sha,
                'assetSha256': {str(control_path): self.control_sha}}
        super().__init__(**kwargs)
        try:
            if mode == 'parallel':
                from .store_generated_copy_cache_v1 import warm_plan
                from .store_parallel_text_data_v1 import public_record
                real_records = {canonical(public_record(c['facts'])) for g in warm_plan(self.session.snapshot)['groups'] for c in g['cards']}
                if real_records != {canonical(r) for r in self.public_records}:
                    raise ValueError('Public48 feature facts differ from current actual storefront')
                if self.head is None:
                    raise ValueError('No parallel decoder loaded; native fallback forbidden')
            else:
                from .store_generated_copy_cache_v1 import GeneratedCopyCache, hashed, producer_spec
                small = self.model_verification['smallFileSha256']
                tokenizer = hashed({name: small[name] for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'chat_template.jinja')})
                expected = producer_spec(self.copy_sha, self.bb.revision, tokenizer, native_width=10)
                if self.cache_spec != expected:
                    raise ValueError('Cache producer differs from verified resident checkpoint/tokenizer/sources')
                import shutil
                cache_directory = self.output / 'runtime-cache'
                shutil.copytree(self.cache_source / 'warm-cache', cache_directory)
                self.copy_cache = GeneratedCopyCache(cache_directory, self.session.snapshot, self.cache_spec)
                self.copy_metadata.update(verifiedOriginalPublicBindings=48, originalCachePreserved=True, cacheSource=str(self.cache_source))
            self.startup.update(candidateVersion=VERSION, copyMode=mode, copyCandidate=self.copy_metadata,
                candidateSourcesSha256=self.candidate_sources, nativeFreshBackend='store_demo_service CLI remains available')
            (self.output / 'startup.json').write_text(json.dumps(self.startup, indent=2) + '\n')
        except BaseException as error:
            try:
                super().close()
            except BaseException as cleanup_error:
                error.add_note('Native initialization cleanup failed: ' + str(cleanup_error))
            raise

    def _assert_candidate(self):
        if any(sha(HERE / name) != expected for name, expected in self.candidate_sources.items()):
            raise ValueError('Candidate inference source changed since startup')
        if any(sha(path) != expected for path, expected in self.copy_metadata['assetSha256'].items()):
            raise ValueError('Candidate checkpoint/features/controls changed since startup')

    def close(self):
        # Always complete native teardown/hashes even when candidate pins fail.
        try:
            self._assert_candidate()
        finally:
            result = super().close()
        result['candidateAssetsUnchanged'] = True
        (self.output / 'shutdown.json').write_text(json.dumps(result, indent=2) + '\n')
        return result

    def _native_cache_miss(self, cards, directory, cancelled):
        from .branch_local_copy import prepare_root
        from .fused_prefix import _complete_cache
        from .native_batch_diagnosis import state_hash
        from .store_copy_wide_v2 import guard_memory
        from .store_generated_copy_cache_v1 import utc_now
        from .store_demo_engine import decode_rows
        common, jobs = demo_layout(cards)
        directory.mkdir(parents=True, exist_ok=False)
        receipt = {'status': 'running', 'startedAt': utc_now(), 'cards': deepcopy(cards), 'common': common,
            'jobs': jobs, 'provenance': {'nativeGeneration': True, 'fallback': False,
                'checkpointSha256': self.copy_sha, 'modelRevision': self.bb.revision,
                'tokenizerSha256': self.cache_spec['tokenizerSha256'], 'nativeWidth': 10,
                'sameWidthControlsSha256': self.control_sha}}
        started = time.perf_counter(); state = None
        try:
            if cancelled():
                raise RequestCancelled('Cancelled before complete native cache miss generation')
            state, preparation = prepare_root(self.bb, common)
            _complete_cache(state['cache'], len(state['tokens'])); original = state_hash(state['cache'])
            rows, timing = decode_rows(self.bb, state, jobs, 10, lambda done: None, cancelled)
            receipt.update(rows=rows, preparation=preparation, decodeTiming=timing)
            for row in rows:
                if (row['text'] != self.bb.tokenizer.decode(row['tokenIds']) or
                        (row['finishReason'] == 'stop' and row['stopTokenId'] not in self.bb.tokenizer.eos_token_ids) or
                        row['arSteps'] != len(row['rawSampledTokenIds']) or
                        row['tokenIds'] != (row['rawSampledTokenIds'][:-1] if row['finishReason'] == 'stop' else row['rawSampledTokenIds'])):
                    raise AssertionError('Native cache miss actual sampled-token/text/EOS receipt differs')
            if state_hash(state['cache']) != original:
                raise AssertionError('Shared prefix mutated during native cache miss')
            if cancelled():
                raise RequestCancelled('Cancelled before native cache insertion')
            receipt.update(status='completed', prefixUnchanged=True, actualTokenizerDecodeAndEOSVerified=True, memory=guard_memory(self.bb))
        except BaseException as error:
            receipt.update(status='cancelled' if isinstance(error, RequestCancelled) else 'failed',
                error={'type': type(error).__name__, 'message': str(error)},
                partialRows=getattr(error, 'partial_rows', receipt.get('rows', [])))
            raise
        finally:
            receipt.update(completedAt=utc_now(), coldGenerationMs=(time.perf_counter() - started) * 1000)
            (directory / 'native.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
            if state is not None:
                del state
            self.bb.mx.clear_cache()
        return receipt

    def personalize(self, value, request_id, started, emit, cancelled):
        from .store_copy_wide_v2 import guard_memory
        request, policy = checked_request(value)
        directory = self.output / 'requests' / request_id; directory.mkdir(parents=True, exist_ok=False)
        receipt = {'version': VERSION, 'requestId': request_id, 'input': deepcopy(value),
            'effectiveRequest': request, 'policy': policy, 'status': 'running', 'mode': self.mode,
            'copyConditioning': deepcopy(NEUTRAL_COPY_REQUEST), 'copyShopperQueryUsed': False,
            'copyCandidate': self.copy_metadata, 'copyInputDisclosure': 'parallel word head: selected public name/category/itemPrice/color and fixed plain style; native cache producer: same selected facts and neutral public request; shopper query affects ranking only', 'events': [], 'rawRows': []}
        def save():
            path = directory / 'receipt.tmp'; path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
            path.replace(directory / 'receipt.json')
        def event(kind, **body):
            if cancelled():
                raise RequestCancelled('Cancelled before candidate event serialization')
            self._assert_candidate()
            result = {'type': kind, 'requestId': request_id, **body}
            canonical(result)
            result.setdefault('timing', {})['readyMs'] = (time.perf_counter() - started) * 1000
            if kind in ('firstThree', 'complete'):
                result['timing']['first3readyMs' if kind == 'firstThree' else 'fullReadyMs'] = result['timing']['readyMs']
            canonical(result); receipt['events'].append(deepcopy(result)); emit(result)
        save()
        try:
            self._assert_candidate()
            if cancelled():
                raise RequestCancelled('Cancelled before native ranking admission')
            ranked = self.session.rank_request(ranking_input(request), copy_adapter=self.adapter)
            order, guards = guarded_order(ranked['scores16'], request, policy, self.session.products)
            cards = action_bindings(request, order[:6], self.session.snapshot)
            rank_ms = (time.perf_counter() - started) * 1000
            provenance = {'ranking': 'fresh frozen Qwen query encoding and trained actual-store rank head; deterministic public hard guards and exact-name override',
                'rankCheckpointSha256': self.session.head_sha, 'nativeCopyCheckpointSha256': self.copy_sha,
                'copy': self.copy_metadata, 'images': IMAGE_POLICY, 'hiddenProfileInputs': False,
                'timingScope': TIMING_SCOPE, 'candidateMode': self.mode,
                'qualityScope': 'TRAIN-ALL catalog demo; no heldout generalization claim' if self.mode == 'parallel' else 'verified original native field validation; cache hits generated earlier'}
            if hasattr(self.session, 'startup_provenance'):
                provenance['rankSession'] = deepcopy(self.session.startup_provenance)
            receipt.update(ranking=ranked, finalOrder=order, guards=guards, cards=cards, provenance=provenance)
            metadata, _ = public_sections(cards, [], self.session.products)
            shared = {'eligibleCount': len(order), 'shortfall': max(0, 6 - len(cards)), 'provenance': provenance}
            event('rank', productIds=[self.session.snapshot['externalMapping']['products'][i]['productId'] for i in order],
                selectedProductIds=[c['action']['productId'] for c in cards],
                variants={c['action']['productId']: c['action']['variantId'] for c in cards},
                sections=metadata, timing={'rankMs': rank_ms}, **shared)
            save()
            if cancelled():
                raise RequestCancelled('Cancelled after ranking')
            if cards and self.mode == 'parallel':
                from . import store_parallel_text_head_v1 as model
                rows = parallel_rows(cards)
                with model.bounded_cpu_threads(1):
                    generated = model.generate(self.head, model.inference_rows(rows), self.feature_map)
                receipt.update(runtimeRows=rows, parallelGeneration=generated)
                source = {**self.copy_metadata, 'kind': 'fresh-parallel-head', 'generatedEarlier': False, 'freshPerRequestGeneration': True,
                    'generation': 'fresh independent word-slot predictions from cached frozen public fact features',
                    'autoregressiveSteps': 0, 'feedforwardCalls': generated['feedforwardCalls']}
                payload = lambda limit: parallel_payload(cards, rows, generated, limit)
            elif cards:
                resolved = self.copy_cache.resolve(cards, request_started=started,
                    native_generate=lambda selected: self._native_cache_miss(selected, directory / 'native-miss', cancelled))
                receipt.update(copyCache=resolved['cache'], rawRows=resolved['rawRows'], validation=resolved['validation'])
                source = source_for_cache(resolved['cache'])
                payload = lambda limit: public_sections(cards, resolved['rawRows'], self.session.products, limit)
            else:
                source = {'kind': 'no-generation',
                    'generatedEarlier': False, 'freshPerRequestGeneration': False, 'noEligibleProducts': True}
                payload = lambda limit: public_sections([], [], self.session.products, 0)
            # All parallel slots / cache resolution finish before either endpoint;
            # never claim earlier first3 readiness than actual validation+JSON.
            for kind, limit in ([('firstThree', min(3, len(cards))), ('complete', len(cards))] if cards else [('complete', 0)]):
                if cancelled():
                    raise RequestCancelled('Cancelled before candidate payload validation')
                sections, validation = payload(limit)
                self.binding.assert_current_sources(self.session.snapshot['receipt'])
                event(kind, sections=sections, validation=validation, copySource=source,
                    **{key: validation[key] for key in ('acceptedFields', 'withheldFields', 'requiredFields')},
                    timing={'rankMs': rank_ms, 'copyMs': (time.perf_counter() - started) * 1000 - rank_ms}, **shared)
            self._assert_candidate()
            receipt.update(status='completed', memory=guard_memory(self.bb)); save()
        except BaseException as error:
            receipt.update(status='cancelled' if isinstance(error, RequestCancelled) else 'failed',
                error={'type': type(error).__name__, 'message': str(error)},
                partialRows=getattr(error, 'partial_rows', []), failedAtMs=(time.perf_counter() - started) * 1000)
            save(); raise


def main():
    from .store_demo_service import ModelWorker, handler_for
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=('parallel', 'generated-cache'))
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--feature-bundle', type=Path)
    parser.add_argument('--public-contract', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--rank-run', type=Path, default=DEFAULT_RANK)
    parser.add_argument('--copy-run', type=Path, default=DEFAULT_COPY)
    parser.add_argument('--integration-manifest', type=Path, required=True)
    args = parser.parse_args()
    # Require an unused output; don't replace the live service or old receipts.
    if args.output.exists():
        raise FileExistsError('Candidate service output must be unused')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(None))
    worker = ModelWorker(lambda: CandidateDemoEngine(mode=args.mode, candidate_run=args.candidate_run,
        feature_bundle=args.feature_bundle, public_contract=args.public_contract, output=args.output,
        rank_run=args.rank_run, copy_run=args.copy_run, width=10, integration_manifest=args.integration_manifest))
    server.RequestHandlerClass = handler_for(worker); server.daemon_threads = True
    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    print(json.dumps({'event': 'candidate-demo-listening', 'mode': args.mode,
        'url': f'http://127.0.0.1:{args.port}', 'nativeWorker': 'starting', 'baseNativeBackend': 'store_demo_service'}), flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        if not worker.close():
            raise RuntimeError('Candidate native worker did not stop; shutdown integrity unconfirmed')


if __name__ == '__main__':
    main()
