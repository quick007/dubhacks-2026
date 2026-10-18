"""Scoped host critical-path observer for an exclusive new candidate worker.

Never attach to an HTTP/live worker. These process-local patches exist only
inside a standalone owner job and are restored even on failure. No additional
GPU synchronization, sampling, caching, output repair or algorithm changes.
"""
from __future__ import annotations
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import functools
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

VERSION = 'store-fast-path-host-profile.v1'


class PhaseTrace:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock; self.origin = None; self.nodes = []; self.stack = []; self.marks = {}

    @contextmanager
    def phase(self, name, *, start=None, metadata=None):
        before = self.clock() if start is None else start
        node = {'id': len(self.nodes), 'name': name, 'parent': self.stack[-1] if self.stack else None,
            'start': before, 'end': None, 'metadata': metadata or {}}
        self.nodes.append(node); self.stack.append(node['id'])
        try:
            yield
        finally:
            node['end'] = self.clock(); self.stack.pop()

    def wrap(self, original, label, metadata=None):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            name = label(args, kwargs) if callable(label) else label
            with self.phase(name, metadata=metadata):
                return original(*args, **kwargs)
        return wrapped

    def mark(self, name):
        self.marks[name] = self.clock()

    def report(self, cutoff=None):
        if self.origin is None or self.stack or any(n['end'] is None for n in self.nodes):
            raise ValueError('Completed rooted trace required')
        end = cutoff if cutoff is not None else max(n['end'] for n in self.nodes)
        def duration(node):
            return max(0., min(node['end'], end) - max(node['start'], self.origin)) * 1000
        phases = {}; timeline = []
        for node in self.nodes:
            inclusive = duration(node)
            if inclusive <= 0:
                continue
            children = sum(duration(child) for child in self.nodes if child['parent'] == node['id'])
            exclusive = max(0., inclusive - children)
            group = phases.setdefault(node['name'], {'calls': 0, 'inclusiveMs': 0., 'exclusiveMs': 0., 'maxInclusiveMs': 0.})
            group['calls'] += 1; group['inclusiveMs'] += inclusive; group['exclusiveMs'] += exclusive
            group['maxInclusiveMs'] = max(group['maxInclusiveMs'], inclusive)
            timeline.append({'id': node['id'], 'parent': node['parent'], 'name': node['name'],
                'startMs': max(0., node['start'] - self.origin) * 1000,
                'endMs': max(0., min(node['end'], end) - self.origin) * 1000,
                'inclusiveMs': inclusive, 'exclusiveMs': exclusive, 'metadata': node['metadata']})
        wall = max(0., end - self.origin) * 1000
        accounted = sum(v['exclusiveMs'] for v in phases.values())
        return {'version': VERSION, 'wallMs': wall, 'phases': phases, 'timeline': timeline,
            'exclusiveAccountedMs': accounted, 'unaccountedMs': wall - accounted,
            'accounting': 'inclusive times overlap; exclusive removes direct child intervals; request exclusive is remaining Python/archive/observer work',
            'nativeBoundary': 'existing bb.encode already evals/synchronizes its native output; observer adds no synchronization',
            'scope': 'single-thread synchronous worker job; no HTTP queue, socket transfer, browser or image readiness'}


class JsonProxy:
    def __init__(self, original, trace):
        self.original = original; self.dumps = trace.wrap(original.dumps, 'serialization.receiptJson')
    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def scoped_profile(engine, trace):
    """Only use on the NEW standalone engine under its sole owner lock."""
    from . import store_candidate_engine_v1 as candidate
    from . import store_rank_session as rank_session
    from . import store_rank_session_v5 as rank_v5
    from . import store_parallel_text_head_v1 as text_head
    from . import store_parallel_text_data_v1 as text_data
    from .catalog_fusion import runtime, inputs
    with ExitStack() as stack:
        def wrap(obj, name, label, optional=False):
            if optional and not hasattr(obj, name):
                return
            original = getattr(obj, name)
            stack.enter_context(patch.object(obj, name, trace.wrap(original, label)))
        wrap(engine, '_assert_candidate', 'engineLifetimeIntegrity')
        wrap(engine.binding, 'assert_current_sources', 'sourceValidation')
        wrap(engine.session, 'source_validator', 'sourceValidation')
        wrap(engine.session, '_assert_head_assets', 'rankHeadIntegrity', optional=True)
        wrap(engine.session, 'integrity_receipt', 'rankHeadIntegrityReceipt', optional=True)
        wrap(engine.session, 'rank_request', 'rankSession')
        # MLX TokenizerWrapper delegates attribute writes to its HF tokenizer;
        # patching the wrapper makes mock cleanup delete a nonexistent local
        # attribute. Own the underlying method for both install and restore.
        tokenizer_target = getattr(engine.bb.tokenizer, '_tokenizer', engine.bb.tokenizer)
        wrap(tokenizer_target, 'encode', 'tokenizerEncode')
        wrap(engine.bb, 'encode', 'freshBackboneEncode')
        wrap(engine.session.store, 'load', 'catalogFeatureLoad')
        wrap(engine.session.head, 'probabilities', lambda args, kwargs: 'rankHead.' + str(args[0]) + 'Probabilities')
        wrap(engine.head, 'forward', 'parallelHeadForward')
        wrap(candidate, 'checked_request', 'admissionSchemaAndHardPolicy')
        wrap(candidate, 'guarded_order', 'finalHardGuards')
        wrap(rank_session, 'rank', 'rankGuards')
        wrap(rank_session, 'action_bindings', 'actionBinding')
        wrap(candidate, 'action_bindings', 'actionBinding')
        wrap(candidate, 'parallel_rows', 'parallelPublicRows')
        wrap(candidate, 'parallel_payload', 'copyPayloadValidation')
        wrap(candidate, 'public_sections', 'metadataPayloadValidation')
        wrap(text_head, 'generate', 'parallelTextGeneration')
        wrap(text_head, 'serving_tensors', 'parallelInputTensors')
        wrap(text_head, 'decode_ids', 'parallelWordSlotDecode')
        wrap(text_head, 'field_rejections', 'copyFieldSafetyGate')
        wrap(text_data, 'field_rejections', 'copyFieldSafetyGate')
        for module in (candidate, rank_session, rank_v5, text_data, runtime, inputs):
            if hasattr(module, 'canonical'):
                original = module.canonical
                stack.enter_context(patch.object(module, 'canonical', trace.wrap(original,
                    'serialization.canonical', metadata={'module': module.__name__})))
        stack.enter_context(patch.object(candidate, 'json', JsonProxy(candidate.json, trace)))
        for name in ('write_text', 'mkdir', 'replace'):
            original = getattr(Path, name)
            stack.enter_context(patch.object(Path, name, trace.wrap(original, 'archive.' + name)))
        yield


def receipt_signature(receipt, events):
    """Exact logical parity excludes timing, request IDs and archive locations."""
    generated = receipt.get('parallelGeneration', {})
    fields = generated.get('fields', [])
    signature = {'scores16': receipt.get('ranking', {}).get('scores16'), 'finalOrder': receipt.get('finalOrder'),
        'cards': receipt.get('cards'), 'runtimeRows': receipt.get('runtimeRows'),
        'copyWordFields': [{k: f.get(k) for k in ('text', 'slotIds', 'finishReason', 'firstEosSlot', 'rejections', 'accepted')} for f in fields],
        'endpoints': [{k: event.get(k) for k in ('type', 'productIds', 'selectedProductIds', 'variants', 'sections',
            'acceptedFields', 'withheldFields', 'requiredFields', 'validation', 'provenance', 'copySource')} for event in events]}
    encoded = json.dumps(signature, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'sha256': hashlib.sha256(encoded).hexdigest(), 'logical': signature,
        'cardCount': len(receipt.get('cards', [])), 'copyFieldCount': len(fields),
        'freshFeedforwardCalls': generated.get('feedforwardCalls'), 'freshCopySource': events[-1].get('copySource', {}).get('kind') if events else None}


def measure_request(engine, value, request_id, *, instrumented=False, clock=time.perf_counter):
    """Same complete original personalization path in both arms, serialized inline.

    Hook installation/restoration is excluded but recorded. The observer itself
    remains inside B's timing; paired A/B measures that overhead, never hides it.
    """
    trace = PhaseTrace(clock); events = []; wire = []; ready = {}; setup = clock()
    with ExitStack() as stack:
        if instrumented:
            stack.enter_context(scoped_profile(engine, trace))
        setup_ms = (clock() - setup) * 1000
        started = clock(); trace.origin = started
        def emit(event):
            if instrumented:
                with trace.phase('serialization.ndjsonWire'):
                    encoded = (json.dumps(event, allow_nan=False) + '\n').encode()
            else:
                encoded = (json.dumps(event, allow_nan=False) + '\n').encode()
            events.append(deepcopy(event)); wire.append(encoded)
            ready[event['type']] = (clock() - started) * 1000
            trace.mark(event['type'] + 'Wire')
        try:
            if instrumented:
                with trace.phase('request', start=started):
                    engine.personalize(value, request_id, started, emit, lambda: False)
            else:
                engine.personalize(value, request_id, started, emit, lambda: False)
        except BaseException as error:
            error.profile_failure_timing = {'failedAtMs': (clock() - started) * 1000, 'requestId': request_id}
            if instrumented:
                error.profile_partial_trace = trace.report()
            raise
        outer_ms = (clock() - started) * 1000
    receipt_path = Path(engine.output) / 'requests' / request_id / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    signature = receipt_signature(receipt, events)
    if (receipt.get('status') != 'completed' or signature['cardCount'] != 6 or signature['copyFieldCount'] != 20 or
            signature['freshFeedforwardCalls'] != 1 or signature['freshCopySource'] != 'fresh-parallel-head' or
            [e['type'] for e in events] != ['rank', 'firstThree', 'complete'] or events[-1]['requiredFields'] != 20):
        raise ValueError('Unchanged full6/20 fresh parallel-copy workload required; no shortfall/downscope/cache substitution')
    result = {'requestId': request_id, 'arm': 'B-profile' if instrumented else 'A-original', 'status': 'completed',
        'input': deepcopy(value), 'receiptPath': str(receipt_path), 'parity': signature,
        'timing': {'rankWireReadyMs': ready['rank'], 'firstThreeWireReadyMs': ready['firstThree'],
            'fullSixWireReadyMs': ready['complete'], 'outerWallIncludingFinalReceiptMs': outer_ms,
            'observerSetupMsExcluded': setup_ms, 'timerScope': 'caller entry through inline serialization of original NDJSON event; no queue/network/browser/images'},
        'wireBytes': {event['type']: len(encoded) for event, encoded in zip(events, wire, strict=True)},
        'quality': {event['type']: {k: event[k] for k in ('acceptedFields', 'withheldFields', 'requiredFields')}
            for event in events if event['type'] != 'rank'}}
    if instrumented:
        result['profile'] = {'throughFullSixWire': trace.report(trace.marks['completeWire']),
            'throughRequestReturn': trace.report(),
            'observerOverheadPolicy': 'Compare counterbalanced paired A/B full readiness; no added GPU sync or automatic overhead subtraction'}
    return result


# Importing this optional subclass does not load a model or start a worker.
from .store_candidate_image_service_v1 import ImageCandidateDemoEngine


class ProfileCandidateEngine(ImageCandidateDemoEngine):
    def measured_request(self, value, request_id, *, instrumented=False):
        return measure_request(self, value, request_id, instrumented=instrumented)
