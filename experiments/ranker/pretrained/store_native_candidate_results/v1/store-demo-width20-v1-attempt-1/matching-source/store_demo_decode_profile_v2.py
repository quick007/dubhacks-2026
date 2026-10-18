"""Synchronized native phase profiler; no live/library source mutations or load.

The candidate preserves the pinned decoder's admission/sampling algorithm and
uses local subclasses of the installed MLX-LM batch classes. Extra boundaries
and cache materialization deliberately disturb asynchronous overlap. Report
ordinary uninstrumented timings separately and require exact token parity.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import importlib
from importlib.metadata import distribution
from pathlib import Path
import time
from types import FunctionType

from . import store_demo_decode_profile_v1 as outer

GENERATE_SHA256 = '9d46803799f99d6fc7c35fff2d4357c79b1a25ae98cc63efeaaebeb50a7b5f33'


def assert_reference_file():
    """Safe when the owner temporarily redirects engine.decode_rows in memory."""
    from . import store_demo_engine
    source = Path(store_demo_engine.__file__).read_text()
    node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'decode_rows')
    segment = ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    digest = hashlib.sha256(segment.encode()).hexdigest()
    if digest != outer.REFERENCE_SHA256:
        raise ValueError('Physical serving decoder differs from reviewed profile reference')
    return digest


def assert_installed_sources():
    package = distribution('mlx-lm')
    path = Path(package.locate_file('mlx_lm/generate.py'))
    if package.version != '0.32.2' or hashlib.sha256(path.read_bytes()).hexdigest() != GENERATE_SHA256:
        raise ValueError('Installed MLX-LM generate implementation changed; re-audit copied transitions')
    return {'mlxLmVersion': package.version, 'generateSourceSha256': GENERATE_SHA256}


class PhaseCollector:
    """Nested inclusive/exclusive phase times; pre-boundary waits kept separate."""
    def __init__(self, synchronize, *, clock=time.perf_counter):
        self.synchronize = synchronize; self.clock = clock; self.events = []; self.stack = []

    def measure(self, phase, operation, *, metadata=None, materialize=None, synchronized=True):
        entry = self.clock()
        if synchronized:
            self.synchronize()
        begin = self.clock()
        frame = {'childrenTotalMs': 0.}; self.stack.append(frame)
        result = None; failed = False
        try:
            result = operation()
            if materialize is not None:
                materialize(result)
            return result
        except BaseException:
            failed = True; raise
        finally:
            if synchronized:
                self.synchronize()
            end = self.clock(); self.stack.pop()
            inclusive = (end - begin) * 1000; before = (begin - entry) * 1000
            total = (end - entry) * 1000
            event = {'phase': phase, 'inclusiveMs': inclusive,
                'exclusiveMs': inclusive - frame['childrenTotalMs'], 'preBoundarySyncMs': before,
                'synchronized': synchronized, 'failed': failed, **(metadata or {})}
            self.events.append(event)
            if self.stack:
                self.stack[-1]['childrenTotalMs'] += total

    def report(self):
        phases = defaultdict(lambda: {'calls': 0, 'inclusiveMs': 0., 'exclusiveMs': 0., 'preBoundarySyncMs': 0.})
        for event in self.events:
            row = phases[event['phase']]; row['calls'] += 1
            for key in ('inclusiveMs', 'exclusiveMs', 'preBoundarySyncMs'):
                row[key] += event[key]
        return {'phases': dict(phases), 'events': self.events,
            'accountedExclusivePhaseMs': sum(row['exclusiveMs'] for row in phases.values()),
            'preBoundarySyncMs': sum(row['preBoundarySyncMs'] for row in phases.values()),
            'inclusivePhaseTimesAreAdditive': False,
            'interpretation': 'sum exclusive phase times and separate pre-boundary waits; inclusive parent phases contain child work'}


def profiled_classes(native, collector, mx, stream):
    """Local derived classes; no assignment to installed-module attributes."""
    def materialize_batch(batch):
        values = [[entry.state for entry in batch.prompt_cache]]
        # Cache concat/filter and next-token slices must be realized here to
        # distinguish their lazy work from the next model forward. Never force
        # the unused full-vocabulary prefill model output or extracted EOS cache.
        tokens = getattr(batch, '_next_tokens', None)
        if tokens is not None:
            values.append(tokens)
        probabilities = getattr(batch, '_next_logprobs', None)
        if probabilities:
            values.append(probabilities)
        mx.eval(*values)

    class ProfiledGenerationBatch(native.GenerationBatch):
        def _step(self):
            return collector.measure('autoregressiveStep', lambda: super(ProfiledGenerationBatch, self)._step(),
                metadata={'rows': len(self.uids), 'inputTokensPerRow': 1})

        def extend(self, batch):
            return collector.measure('generationCacheExtend', lambda: super(ProfiledGenerationBatch, self).extend(batch),
                metadata={'existingRows': len(self.uids), 'incomingRows': len(batch.uids)},
                materialize=lambda _: materialize_batch(self))

        def filter(self, keep):
            return collector.measure('generationCacheFilter', lambda: super(ProfiledGenerationBatch, self).filter(keep),
                metadata={'beforeRows': len(self.uids), 'afterRows': len(keep)},
                materialize=lambda _: materialize_batch(self))

        def extract_cache(self, index):
            return collector.measure('finishedRowCacheExtractionHost',
                lambda: super(ProfiledGenerationBatch, self).extract_cache(index), synchronized=False)

        def next(self):
            return collector.measure('generationResponseAndStopHandling',
                lambda: super(ProfiledGenerationBatch, self).next(), metadata={'rows': len(self.uids)})

    class ProfiledPromptProcessingBatch(native.PromptProcessingBatch):
        def __init__(self, *args, **kwargs):
            collector.measure('promptCacheMergeConstruction',
                lambda: super(ProfiledPromptProcessingBatch, self).__init__(*args, **kwargs),
                metadata={'rows': len(kwargs.get('uids', []))}, materialize=lambda _: materialize_batch(self))

        def extend(self, batch):
            return collector.measure('promptCacheExtend', lambda: super(ProfiledPromptProcessingBatch, self).extend(batch),
                metadata={'existingRows': len(self.uids), 'incomingRows': len(batch.uids)},
                materialize=lambda _: materialize_batch(self))

        def filter(self, keep):
            return collector.measure('promptCacheFilter', lambda: super(ProfiledPromptProcessingBatch, self).filter(keep),
                metadata={'beforeRows': len(self.uids), 'afterRows': len(keep)},
                materialize=lambda _: materialize_batch(self))

        def split(self, indices):
            def materialize_split(result):
                materialize_batch(self); materialize_batch(result)
            return collector.measure('promptCacheSplitAndCopy',
                lambda: super(ProfiledPromptProcessingBatch, self).split(indices),
                metadata={'beforeRows': len(self.uids), 'splitRows': len(indices),
                    'allRowsTransition': len(indices) == len(self.uids)}, materialize=materialize_split)

        def prompt(self, tokens):
            if not tokens:
                return super().prompt(tokens)
            lengths = [len(row) for row in tokens]
            return collector.measure('suffixPrefill', lambda: super(ProfiledPromptProcessingBatch, self).prompt(tokens),
                metadata={'rows': len(tokens), 'usefulInputTokens': sum(lengths),
                    'paddedInputTokens': max(lengths) * len(tokens), 'rowTokenLengths': lengths})

        def generate(self, tokens):
            # Exact installed0.32.2 PromptProcessingBatch.generate, changing
            # only the class instantiated so its INITIAL _step is measured.
            def transition():
                if any(len(row) > 1 for row in tokens):
                    self.prompt([row[:-1] for row in tokens])
                last_token = mx.array([row[-1] for row in tokens])
                generation = ProfiledGenerationBatch(self.model, self.uids, last_token,
                    self.prompt_cache, self.tokens, self.samplers, self.fallback_sampler,
                    self.logits_processors, self.stop_sequences, self.max_tokens)
                self.uids = []; self.prompt_cache = []; self.tokens = []
                self.samplers = []; self.logits_processors = []; self.max_tokens = []
                return generation
            return collector.measure('promptToGenerationTransition', transition,
                metadata={'rows': len(tokens)}, materialize=materialize_batch)

    class ProfiledBatchGenerator(native.BatchGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Original constructor creates ordinary EMPTY batches. Replace
            # those with equivalent empty derived batches; no model step is
            # performed for zero rows. Extra setup is instrumentation overhead.
            self._prompt_batch = ProfiledPromptProcessingBatch.empty(self.model, self.sampler,
                prefill_step_size=self.prefill_step_size)
            self._generation_batch = ProfiledGenerationBatch.empty(self.model, self.sampler)

        def _make_batch(self, count):
            # Exact installed0.32.2 _make_batch, changing only the prompt class.
            def construct():
                uids = []; caches = []; tokens = []; samplers = []; processors = []; caps = []; stops = []
                for _ in range(count):
                    sequence = self._unprocessed_sequences.popleft()
                    uids.append(sequence[0]); caches.append(sequence[3]); tokens.append(sequence[4])
                    samplers.append(sequence[5]); processors.append(sequence[6]); caps.append(sequence[2]); stops.append(sequence[7])
                    self._currently_processing.append([sequence[1], 0, sum(len(segment) for segment in sequence[1])])
                return ProfiledPromptProcessingBatch(model=self.model, uids=uids, caches=caches,
                    tokens=tokens, prefill_step_size=self.prefill_step_size, samplers=samplers,
                    fallback_sampler=self.sampler, logits_processors=processors, stop_sequences=stops, max_tokens=caps)
            return collector.measure('admissionBatchConstruction', construct, metadata={'rows': count})

    return ProfiledBatchGenerator, ProfiledPromptProcessingBatch, ProfiledGenerationBatch


def token_parity(reference_rows, profiled_rows):
    """Fail if instrumentation changes any field's actual sampled token receipt."""
    fields = ('fields', 'kind', 'maxTokens', 'suffixTokens', 'rawSampledTokenIds',
        'tokenIds', 'stopTokenId', 'arSteps', 'finishReason', 'text')
    def project(rows):
        return [{key: row.get(key) for key in fields} for row in rows]
    before, after = project(reference_rows), project(profiled_rows)
    equal = before == after
    report = {'exactTokensEOSAndFields': equal, 'fieldsCompared': list(fields),
        'referenceRows': len(reference_rows), 'profiledRows': len(profiled_rows),
        'referenceSha256': hashlib.sha256(str(before).encode()).hexdigest(),
        'profiledSha256': hashlib.sha256(str(after).encode()).hexdigest()}
    if not equal:
        error = AssertionError('Synchronized profiler changed sampled tokens/EOS/field receipts')
        error.parity_report = report
        raise error
    return report


def _isolated_outer_loop():
    # Reuse the EXACT v1 loop code object, without mutating v1 globals. Its
    # physical source verifier remains correct while the owner redirects the
    # engine's in-memory decode_rows attribute for this profiling arm.
    function = FunctionType(outer.decode_rows_profiled.__code__,
        {**outer.decode_rows_profiled.__globals__, 'assert_reference': assert_reference_file},
        name='isolated_profile_outer_loop', argdefs=outer.decode_rows_profiled.__defaults__)
    function.__kwdefaults__ = dict(outer.decode_rows_profiled.__kwdefaults__ or {})
    return function


def decode_rows_profiled(bb, state, jobs, width, first_ready, cancelled, *,
                         _dependencies=None, _clock=time.perf_counter):
    """Drop-in candidate for selected profile arms; no model load or GPU launch."""
    assert_reference_file()
    collector = PhaseCollector(lambda: bb.mx.synchronize(stream), clock=_clock)
    if _dependencies is None:
        installed = assert_installed_sources()
        native = importlib.import_module('mlx_lm.generate')
        deps = outer.native_dependencies()
        stream = deps['generation_stream']
        deps['BatchGenerator'] = profiled_classes(native, collector, bb.mx, stream)[0]
    else:
        deps = dict(_dependencies); stream = deps['generation_stream']
        installed = {'mlxLmVersion': 'CPU-injected-test', 'generateSourceSha256': None}
        # Tests may inject a generator factory bound to this actual collector.
        factory = deps.pop('profiled_generator_factory', None)
        if factory is not None:
            deps['BatchGenerator'] = factory(collector)
    def attach(profile):
        profile.update(version='store-demo-native-decode-profile.v2', installedNative=installed,
            nativePhases=collector.report(), synchronizedNativePhases=True,
            GPUKernelPrefillDecodeSplitMeasured=False,
            measurementLimitations=[
                'synchronized host/native method phases, not isolated hardware kernel traces',
                'extra barriers disrupt original asynchronous token lookahead and can increase total wall time',
                'cache concat/filter/split state is explicitly materialized for attribution; fusion/allocation behavior can change',
                'initial empty derived-batch construction adds setup overhead without generating tokens',
                'use paired uninstrumented latency and exact raw-token/EOS parity, never substitute profile wall time for serving latency'])
    try:
        rows, timing = _isolated_outer_loop()(bb, state, jobs, width, first_ready, cancelled,
            _dependencies=deps, _clock=_clock)
        attach(timing['profile'])
        return rows, timing
    except BaseException as error:
        profile = getattr(error, 'partial_profile', {})
        attach(profile); error.partial_profile = profile
        raise
