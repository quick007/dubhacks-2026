"""Root-owned native API for compact copy. Importing this module is CPU-only.

Caller supplies a resident backbone and final-down adapter; this module never
loads a model, installs dependencies, downloads weights, or starts a service.
"""
from __future__ import annotations
from collections import OrderedDict
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import time

from . import store_compact_copy_v1 as compact
from .store_copy_fit_v2 import deadline

MODES = ('frozen', 'adapted')
PATHS = ('common-root', 'fact-prefix')
DIAGNOSTIC_INDICES = (0, 1, 8, 9, 16, 17, 24, 25)
CONFIG = {'width': 20, 'fixedCases': 32, 'fixedArms': 128, 'sampledCap': 128,
    'factPrefixLruEntries': 2, 'memoryGiB': 7, 'requestTimeoutSeconds': 120,
    'controlTimeoutSeconds': 240, 'trainingUpdates': 0,
    'scope': 'copy only, complete first3/full6 validated UTF8 payload; excludes ranking/startup/images/browser',
    'controlScope': 'same-width fork/rebuild/repeat at identical common/fact chunk boundaries; no cross-path numeric equality claim'}


def experiment_plan(name):
    if name == 'diagnostic8':
        return {'name': name, 'caseIndices': list(DIAGNOSTIC_INDICES), 'paths': ['fact-prefix'],
            'fixedCases': 8, 'fixedArms': 16, 'scope': 'predeclared small diagnostic; not the full32-case comparison'}
    if name == 'full32':
        return {'name': name, 'caseIndices': list(range(32)), 'paths': list(PATHS),
            'fixedCases': 32, 'fixedArms': 128, 'scope': 'complete32-case two-mode/two-prefix-path development comparison'}
    raise ValueError('Explicit diagnostic8 or full32 compact plan required')


def summary(arms, fixed_cases=32, paths=PATHS):
    result = {}
    for mode in MODES:
        for scheme in paths:
            selected = [a for a in arms if (a['mode'], a['path']) == (mode, scheme)]
            done = [a for a in selected if a['status'] == 'completed-attempt']
            first = [a for a in done if a['firstThree']['validation']['valid']]
            full = [a for a in done if a['fullSix']['validation']['valid']]
            result[mode + '/' + scheme] = {'fixedCases': fixed_cases, 'completed': len(done),
                'failed': sum(a['status'] == 'failed' for a in selected),
                'unattempted': sum(a['status'] == 'unattempted' for a in selected),
                'validFirstThree': len(first), 'validFullSix': len(full),
                'validFirstThreeRateFixedDenominator': len(first) / fixed_cases,
                'validFullSixRateFixedDenominator': len(full) / fixed_cases,
                'attemptFirstThreeMs': [a['firstThreeReadyMs'] for a in done],
                'attemptFullSixMs': [a['fullSixReadyMs'] for a in done],
                'validFirstThreeMs': [a['firstThreeReadyMs'] for a in first],
                'validFullSixMs': [a['fullSixReadyMs'] for a in full], 'semanticProof': False}
    return result


def _guard(bb):
    from .store_copy_wide_v2 import guard_memory
    guard_memory(bb)


def exact_suffix(tokenizer, messages, state, model):
    """Preserve separate ordinary chat turns; legacy suffix helper joins them."""
    if state['model'] is not model or state.get('consumed'):
        raise ValueError('Same-model pristine source branch required')
    full = tokenizer.apply_chat_template(state['messages'] + messages, tokenize=True,
        return_dict=False, add_generation_prompt=True, enable_thinking=False)
    if full[:len(state['tokens'])] != state['tokens']:
        raise ValueError('Complete compact field does not extend the exact native prefix')
    suffix = full[len(state['tokens']):]
    if not 1 <= len(suffix) <= 256:
        raise ValueError('Complete compact suffix required; no truncation')
    state['consumed'] = True
    return suffix, state['cache']


def _fact_state(bb, common_state, message):
    from .parallel_decode import clone_prefix_state
    from .fused_prefix import _complete_cache
    tokens = bb.tokenizer.apply_chat_template(common_state['messages'] + [message], tokenize=True,
        return_dict=False, add_generation_prompt=False, enable_thinking=False)
    if tokens[:len(common_state['tokens'])] != common_state['tokens']:
        raise ValueError('Exact ordinary chat fact prefix does not extend common root')
    state = clone_prefix_state(bb, common_state); tail = tokens[len(state['tokens']):]
    for offset in range(0, len(tail), 256):
        hidden = bb.model.model(bb.mx.array([tail[offset:offset + 256]]), cache=state['cache'])
        bb.mx.eval([entry.state for entry in state['cache']]); bb.mx.synchronize(); del hidden; _guard(bb)
    state.update(tokens=list(tokens), messages=state['messages'] + [message], consumed=False)
    _complete_cache(state['cache'], len(tokens))
    return state


class PrefixPool:
    """One exact common root plus at most two immutable native fact states."""
    def __init__(self, bb, common, use_facts):
        from .branch_local_copy import prepare_root
        from .native_batch_diagnosis import state_hash
        self.bb, self.common, self.use_facts = bb, common, use_facts
        self.root, self.preparation = prepare_root(bb, common)
        self.root_hash = state_hash(self.root['cache']); self.entries = OrderedDict()
        self.known_hashes = {}
        self.hits = self.misses = self.evictions = 0

    def source(self, job, *, rebuild=False):
        from .branch_local_copy import prepare_root
        from .native_batch_diagnosis import state_hash
        root = prepare_root(self.bb, self.common)[0] if rebuild else self.root
        if rebuild and state_hash(root['cache']) != self.root_hash:
            raise AssertionError('Independent same-boundary common root differs')
        if not self.use_facts:
            return root, job['messages']
        key = job['factKey']
        if not rebuild and key in self.entries:
            self.hits += 1; state, expected = self.entries.pop(key); self.entries[key] = (state, expected)
            return state, [job['fieldMessage']]
        state = _fact_state(self.bb, root, job['factMessage'])
        actual_hash = state_hash(state['cache'])
        if key in self.known_hashes and actual_hash != self.known_hashes[key]:
            raise AssertionError('Independent same-boundary fact prefix differs')
        self.known_hashes[key] = actual_hash
        if not rebuild:
            self.misses += 1
            if len(self.entries) >= CONFIG['factPrefixLruEntries']:
                _, (old, expected) = self.entries.popitem(last=False)
                if state_hash(old['cache']) != expected:
                    raise AssertionError('Evicted source fact cache mutated')
                self.evictions += 1
            self.entries[key] = (state, actual_hash)
        return state, [job['fieldMessage']]

    def verify(self):
        from .native_batch_diagnosis import state_hash
        if state_hash(self.root['cache']) != self.root_hash or any(
                state_hash(state['cache']) != expected for state, expected in self.entries.values()):
            raise AssertionError('Immutable compact source cache changed')
        return {'commonRootSha256': self.root_hash, 'factEntries': len(self.entries),
            'factCacheHits': self.hits, 'factCacheMisses': self.misses, 'factCacheEvictions': self.evictions,
            'sourceCachesUnchanged': True}


def decode(bb, jobs, pool, *, first_ready=lambda rows: None, rebuild=False, distributions=False):
    """All20 native rows at width20/original candidate caps, no fallback/repair."""
    from mlx_lm.generate import BatchGenerator, StopSequences, generation_stream
    from mlx_lm.sample_utils import make_sampler
    from .parallel_decode import clone_prefix_state
    mx = bb.mx; prompts = []; caches = []; prefix_tokens = []; rows = []
    for job in jobs:
        source, messages = pool.source(job, rebuild=rebuild)
        branch = clone_prefix_state(bb, source)
        suffix, cache = exact_suffix(bb.tokenizer, messages, branch, bb.model)
        full_sha = compact.hashed(source['tokens'] + suffix)
        if job.get('expectedPromptSha256', full_sha) != full_sha:
            raise ValueError('Native complete prompt differs from reviewed tokenizer preflight')
        if not 1 <= len(suffix) <= 256 or len(source['tokens']) + len(suffix) + job['maxTokens'] > 1024:
            raise ValueError('Untruncated compact native context bounds violated')
        prompts.append(suffix); caches.append(cache); prefix_tokens.append(list(source['tokens']))
        rows.append({**job, 'tokenIds': [], 'rawSampledTokenIds': [], 'stopTokenId': None,
            'arSteps': 0, 'finishReason': None, 'prefixTokens': len(source['tokens']), 'suffixTokens': len(suffix),
            'fullPromptTokenSha256': full_sha,
            **({'distributionSha256': []} if distributions else {})})
    generator = BatchGenerator(bb.model, completion_batch_size=20, prefill_batch_size=20,
        prefill_step_size=256, sampler=make_sampler(temp=0.0))
    first_done = False; rounds = 0; maximum = 0
    try:
        uids = generator.insert(prompts, max_tokens=[j['maxTokens'] for j in jobs], caches=caches,
            all_tokens=prefix_tokens, samplers=[make_sampler(temp=0.0) for _ in jobs],
            logits_processors=[[] for _ in jobs], stop_sequences=[StopSequences(
                [[int(t)] for t in bb.tokenizer.eos_token_ids]) for _ in jobs])
        by_uid = dict(zip(uids, rows, strict=True)); del caches, cache, branch, source
        while responses := generator.next_generated():
            mx.synchronize(generation_stream); rounds += 1; maximum = max(maximum, len(responses))
            for response in responses:
                row = by_uid[response.uid]
                if row['finishReason'] is not None:
                    raise AssertionError('Native row emitted after completion')
                row['arSteps'] += 1; row['rawSampledTokenIds'].append(int(response.token))
                if distributions:
                    import numpy as np
                    values = response.logprobs.astype(mx.float32); mx.eval(values); mx.synchronize(generation_stream)
                    host = np.array(values)
                    if not np.isfinite(host).all():
                        raise FloatingPointError('Nonfinite full native logprob vector')
                    row['distributionSha256'].append(hashlib.sha256(host.tobytes()).hexdigest()); del host, values
                if response.finish_reason != 'stop':
                    row['tokenIds'].append(int(response.token))
                if response.finish_reason is not None:
                    row.update(finishReason=response.finish_reason,
                        stopTokenId=int(response.token) if response.finish_reason == 'stop' else None,
                        text=bb.tokenizer.decode(row['tokenIds']))
            del response, responses; _guard(bb)
            if not first_done and all(r['finishReason'] is not None for r in rows[:10]):
                first_ready(rows[:10]); first_done = True
        if len(rows) != 20 or not first_done or any(r['finishReason'] is None for r in rows) or maximum != 20:
            raise AssertionError('All20 genuinely sampled fields at actual width20 required')
        return rows, {'generationRounds': rounds, 'maximumRowsEmittingTogether': maximum,
            'actualSampledTokensIncludingEos': sum(r['arSteps'] for r in rows), 'sampledCap': 128}
    except BaseException as error:
        error.partial_rows = rows; raise
    finally:
        generator.close(); mx.synchronize(generation_stream)


def controls(bb, case, pool, out):
    out = Path(out); out.mkdir(); _, jobs = compact.layout(case)
    report = {'status': 'running', 'arms': [], 'width': 20, 'sampledCap': 128,
        'scope': CONFIG['controlScope'], 'numericCrossPathEquivalenceClaim': False}
    try:
        reference = None
        for label in ('fork', 'independent-rebuild', 'fork-repeat'):
            report['activeLabel'] = label; compact.dump(out / 'results.json', report)
            rows, timing = decode(bb, jobs, pool, rebuild=label == 'independent-rebuild', distributions=True)
            observed = [{k: r[k] for k in ('field', 'rawSampledTokenIds', 'tokenIds', 'finishReason',
                'stopTokenId', 'distributionSha256', 'fullPromptTokenSha256')} for r in rows]
            arm = {'label': label, 'rows': observed, 'timing': timing, 'cacheVerification': pool.verify()}
            compact.dump(out / (label + '.json'), arm); report['arms'].append(arm)
            if reference is None:
                reference = observed
            elif observed != reference:
                raise AssertionError('Same-boundary width20 token/EOS/full-distribution mismatch')
        report['status'] = 'exact-controls-passed'; report.pop('activeLabel', None)
        return report
    except BaseException as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)},
            partialRows=getattr(error, 'partial_rows', [])); raise
    finally:
        compact.dump(out / 'results.json', report)


def run_native_comparison(bb, adapter, checkpoint, checkpoint_sha, out, preflight_path, *,
        native_owner=None, confirm_serialized=False, reviewed_preflight_sha256=None, data=compact.DATA,
        plan='diagnostic8'):
    """Root supplies loaded bb/adapter; root controls serialized GPU scheduling.

    Root load example: FrozenBackbone -> attach final-down LoRALinear(r4,scale20)
    -> call this function with completed actual-store adapter and reviewed receipt.
    This API acquires the shared owner lock; do not nest another owner-lock scope.
    """
    out, checkpoint, preflight_path = Path(out), Path(checkpoint), Path(preflight_path)
    if native_owner != 'root' or not confirm_serialized or compact.digest(preflight_path) != reviewed_preflight_sha256:
        raise ValueError('Root owner, serialized confirmation and reviewed compact preflight SHA required')
    if out.exists():
        raise FileExistsError('Preserve previous compact native attempt')
    declaration = experiment_plan(plan)
    from .catalog_fusion.generalization_v2 import native_owner_lock
    with native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'), native_owner):
        return _run(bb, adapter, checkpoint, checkpoint_sha, out, preflight_path, data, declaration)


def _run(bb, adapter, checkpoint, checkpoint_sha, out, preflight_path, data, declaration):
    from mlx_lm.generate import generation_stream
    from .semantic_copy_training import disabled, parameter_hashes
    from .store_hybrid_benchmark_v2 import completed_adapter
    started = time.perf_counter(); out.mkdir(parents=True)
    frozen = json.loads(preflight_path.read_text()); _, universe = compact.load_cases(data)
    cases = [universe[index] for index in declaration['caseIndices']]
    arms = [{'caseId': c['caseId'], 'mode': mode, 'path': scheme, 'status': 'unattempted'}
        for mode in MODES for c in cases for scheme in declaration['paths']]
    report = {'version': compact.VERSION, 'status': 'started',
        'config': {**CONFIG, 'fixedCases': declaration['fixedCases'], 'fixedArms': declaration['fixedArms']},
        'plan': declaration, 'caseUniverse': 32, 'fixedCaseIds': [c['caseId'] for c in cases],
        'unattemptedUniverseCaseIds': [c['caseId'] for c in universe if c not in cases],
        'checkpointSha256': checkpoint_sha, 'preflightSha256': compact.digest(preflight_path),
        'arms': arms, 'controls': {}, 'nativeUpdates': 0, 'stage': 'preflight',
        'limitations': ['Copy-only inspected actual-store development comparison; no ranking/browser latency claim.',
            'New short contract; bounded lexical safety is not comprehensive semantic factuality.',
            'Original names retained as metadata; no exact teacher-output targets or fallback strings.']}
    def save():
        report['wallMs'] = (time.perf_counter() - started) * 1000
        report['summary'] = summary(arms, declaration['fixedCases'], declaration['paths'])
        compact.dump(out / 'results.json', report)
    save()
    try:
        if bb.memory_budget != CONFIG['memoryGiB'] * 2**30:
            raise ValueError('Caller must use the unchanged7 GiB native budget')
        if compact.token_preflight(bb.tokenizer, data) != frozen:
            raise ValueError('Compact source/tokenizer/cases changed after preflight')
        completed, actual = completed_adapter(checkpoint.parent)
        if actual.resolve() != checkpoint.resolve() or compact.digest(checkpoint) != checkpoint_sha or completed['checkpoint']['sha256'] != checkpoint_sha:
            raise ValueError('Exact completed actual-store adapted checkpoint required')
        if bb.model.layers[-1].mlp.down_proj is not adapter:
            raise ValueError('Caller must attach the final-down adapter to the supplied model')
        if adapter.scale != 20:
            raise ValueError('Completed final-down adapter scale20 required')
        weights = bb.mx.load(str(checkpoint))
        if set(weights) != {'lora_a', 'lora_b'} or tuple(weights['lora_a'].shape) != (12288, 4) or tuple(weights['lora_b'].shape) != (4, 4096):
            raise ValueError('Exact final-down rank4 checkpoint shapes required')
        adapter.load_weights(list(weights.items()), strict=False); adapter.freeze(); bb.model.eval(); bb.mx.eval(adapter.parameters()); bb.mx.synchronize()
        if any(getattr(adapter, key).dtype != value.dtype or not bool(bb.mx.all(getattr(adapter, key) == value).item()) for key, value in weights.items()):
            raise AssertionError('Loaded adapted A/B checkpoint bytes differ')
        compact.catalog_colors()  # Source-verified CPU validator setup, outside readiness clocks.
        def base_hashes():
            return {k.replace('.31.mlp.down_proj.linear.', '.31.mlp.down_proj.'): v
                for k, v in parameter_hashes(bb.model.language_model).items()}
        before = base_hashes()
        if len(before) != 927 or before != completed['frozenBaseParameterHashes']:
            raise AssertionError('Completed run927 frozen base hashes differ')
        report['frozenBaseParameterHashes'] = before
        planned = {(row['caseId'], row['field']): row for row in frozen['rows']}
        with bb.mx.stream(generation_stream):
            for mode in MODES:
                with disabled(bb.model, adapter) if mode == 'frozen' else nullcontext():
                    for scheme in declaration['paths']:
                        report.update(stage='common-root-preparation', activeMode=mode, activePath=scheme); save()
                        with deadline(CONFIG['requestTimeoutSeconds'], 'compact common root'):
                            pool = PrefixPool(bb, compact.layout(cases[0])[0], scheme == 'fact-prefix')
                        report['controls'][mode + '/' + scheme + '/rootPreparation'] = pool.preparation
                        report['stage'] = 'matched-native-controls'; save()
                        with deadline(CONFIG['controlTimeoutSeconds'], 'compact width20 controls'):
                            report['controls'][mode + '/' + scheme] = controls(bb, cases[0], pool, out / ('controls-' + mode + '-' + scheme))
                        for case in cases:
                            arm = next(a for a in arms if (a['caseId'], a['mode'], a['path']) == (case['caseId'], mode, scheme))
                            arm.update(status='running', stage='layout'); report['stage'] = 'native-request'; save()
                            begin = time.perf_counter(); partial = {}
                            cache_before = (pool.hits, pool.misses, pool.evictions)
                            def first_ready(rows):
                                partial['firstThree'] = compact.assemble(case, rows, first_three=True)
                                partial['firstThreeReadyMs'] = (time.perf_counter() - begin) * 1000
                            try:
                                with deadline(CONFIG['requestTimeoutSeconds'], 'compact copy request'):
                                    common, jobs = compact.layout(case)
                                    for job in jobs:
                                        job['expectedPromptSha256'] = planned[(case['caseId'], job['field'])]['promptSha256']
                                    if common != pool.common:
                                        raise AssertionError('Request changed the supposedly static common root')
                                    rows, timing = decode(bb, jobs, pool, first_ready=first_ready)
                                    full = compact.assemble(case, rows)
                                    full_at = (time.perf_counter() - begin) * 1000
                                    proof = pool.verify()
                                    proof['requestFactHits'] = pool.hits - cache_before[0]
                                    proof['requestFactMisses'] = pool.misses - cache_before[1]
                                    proof['requestFactEvictions'] = pool.evictions - cache_before[2]
                                arm.update(partial, status='completed-attempt', stage='completed', rows=rows,
                                    fullSix=full, fullSixReadyMs=full_at, generation=timing, sourceCacheProof=proof,
                                    requestWallIncludingPostReadinessProofMs=(time.perf_counter() - begin) * 1000,
                                    memory=bb.runtime_memory())
                            except BaseException as error:
                                arm.update(partial, status='failed', error={'type': type(error).__name__, 'message': str(error)},
                                    rows=getattr(error, 'partial_rows', []), elapsedMs=(time.perf_counter() - begin) * 1000)
                                save(); raise
                            save(); gc.collect(); bb.mx.clear_cache()
                        del pool; gc.collect(); bb.mx.clear_cache()
        report['stage'] = 'final-preservation'; save()
        if base_hashes() != before or compact.digest(checkpoint) != checkpoint_sha or compact.token_preflight(bb.tokenizer, data) != frozen:
            raise AssertionError('Frozen weights/checkpoint or compact sources changed')
        report.update(status='completed', stage='completed', all927FrozenUnchanged=True,
            checkpointUnchanged=True, sourceUnchanged=True, memory=bb.runtime_memory()); save()
        return report
    except BaseException as error:
        report.update(status='blocked', error={'type': type(error).__name__, 'message': str(error)})
        for arm in arms:
            if arm['status'] == 'running':
                arm.update(status='failed', error=report['error'])
        save(); raise
