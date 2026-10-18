"""Isolated frozen-base B1 query-prefix candidate; importing is CPU/stdlib only.

No worker/model/service is constructed here. Root must own the native lock and
serialize calls with all other GPU work. No live encoder/session is patched.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import sysconfig
import time

VERSION = 'store-query-prefix-reuse.v1'
INSTRUCTION = 'Encode the complete public shopping request as data.\n'
# Literal formatter constants only, stopping BEFORE the first dynamic value.
# One boundary token is always withheld; never choose this using evaluation text.
PREFIX = INSTRUCTION + '{"budgetScope":"inclusive item price only; delivery excluded","request":{"hardRequirements":'
WITHHOLD_BOUNDARY_TOKENS = 1
MAX_TOKENS = 512
MIN_COSINE = .99999
MAX_ABS_ERROR = .001
REQUIRED_ASSET_GROUPS = {'model', 'tokenizer', 'source', 'catalogFeatures', 'head'}

def canonical(value): return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
def hashed(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def request_text(request):
    from .store_function_adaptation import request_text as original
    return original(request)

def prefix_tokens(tokenizer):
    ids = list(tokenizer.encode(PREFIX, add_special_tokens=False))
    if len(ids) <= WITHHOLD_BOUNDARY_TOKENS: raise ValueError('Nonempty literal prefix required')
    return ids[:-WITHHOLD_BOUNDARY_TOKENS]

def query_layout(tokenizer, request, *, formatter=request_text):
    text = formatter(request)
    if not text.startswith(PREFIX): raise ValueError('Public request formatter differs from fixed literal prefix')
    # Tokenize the WHOLE actual request. Never concatenate separately tokenized suffixes.
    full = list(tokenizer.encode(text, add_special_tokens=False)); prefix = prefix_tokens(tokenizer)
    if not 0 < len(full) <= MAX_TOKENS: raise ValueError('Whole public request must fit512tokens; no truncation')
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError('Actual whole-request token IDs do not extend cached prefix exactly')
    return {'text': text, 'fullTokenIds': full, 'prefixTokenIds': prefix, 'suffixTokenIds': full[len(prefix):],
        'wholeTextSha256': hashlib.sha256(text.encode()).hexdigest(), 'wholeTokenSha256': hashed(full)}

def freeze_assets(groups, *, frozen927_hashes_sha256, model_revision):
    """Explicit expensive full SHA pass, outside timed requests. Include actual
    model shards/config, tokenizer files, source, head and catalog feature files.
    No query feature or response cache is read or written by this module.
    """
    if set(groups) != REQUIRED_ASSET_GROUPS or not all(groups.values()): raise ValueError('All five nonempty asset groups required')
    if re.fullmatch('[0-9a-f]{64}', frozen927_hashes_sha256) is None: raise ValueError('Verified927-parameter-map digest required')
    entries = []
    for group, paths in sorted(groups.items()):
        for path in sorted(map(Path, paths)):
            if path.is_symlink() or not path.is_file(): raise ValueError('Actual nonsymlink asset files required')
            stat = path.stat()
            entries.append({'group': group, 'path': str(path.resolve()), 'bytes': stat.st_size,
                'mtimeNs': stat.st_mtime_ns, 'sha256': sha(path)})
    return {'version': VERSION, 'assets': entries, 'frozen927HashesSha256': frozen927_hashes_sha256,
        'modelRevision': model_revision, 'assetKey': hashed(entries), 'prefixLiteralSha256': hashlib.sha256(PREFIX.encode()).hexdigest(),
        'withholdBoundaryTokens': WITHHOLD_BOUNDARY_TOKENS, 'cacheMode': 'frozen-base-copy-adapter-disabled'}

def assert_assets(binding, *, full_sha=False):
    if binding['assetKey'] != hashed(binding['assets']) or binding['prefixLiteralSha256'] != hashlib.sha256(PREFIX.encode()).hexdigest():
        raise ValueError('Asset binding or literal prefix changed')
    for row in binding['assets']:
        path = Path(row['path'])
        if path.is_symlink() or not path.is_file(): raise ValueError('Bound asset missing or symlinked')
        stat = path.stat()
        if stat.st_size != row['bytes'] or stat.st_mtime_ns != row['mtimeNs'] or (full_sha and sha(path) != row['sha256']):
            raise ValueError('Bound model/tokenizer/source/head/catalog feature asset changed: ' + row['path'])
    return True

def installed_source_receipt(purelib=None):
    """Read installed Apple runtime source files; does not import MLX."""
    root = Path(purelib or sysconfig.get_path('purelib')) / 'mlx_lm/models'
    names = ('cache.py', 'qwen3_5.py', 'base.py', 'gated_delta.py', 'qwen3_next.py')
    return {'mlxVersion': version('mlx'), 'mlxLmVersion': version('mlx-lm'),
        'files': {str(root / name): sha(root / name) for name in names},
        'capability': 'make_prompt_cache/model.make_cache; from_state(deepcopy(full.state));24ArraysCache+8KVCache. No recurrent trim/rollback used.',
        'nativeParity': 'Not executed by source inspection; required separately'}

def clone_complete_cache(cache, *, materialize=lambda states: None):
    result = [type(entry).from_state(copy.deepcopy(entry.state)) for entry in cache]
    materialize([entry.state for entry in result])
    return result

@contextmanager
def frozen_base(backbone, adapter=None):
    model = backbone.model
    if getattr(model, 'training', False) is not False: raise ValueError('Ordinary frozen eval mode required')
    active = model.layers[-1].mlp.down_proj
    if adapter is None:
        if hasattr(active, 'lora_a'): raise ValueError('Attached copy adapter must be explicitly supplied and disabled')
        yield
    else:
        if active is not adapter: raise ValueError('Copy adapter must belong to this resident model')
        from .semantic_copy_training import disabled
        with disabled(model, adapter): yield

def native_parameter_hashes(backbone):
    from .semantic_copy_training import parameter_hashes
    return {k.replace('.31.mlp.down_proj.linear.', '.31.mlp.down_proj.'): v
        for k, v in parameter_hashes(backbone.model.language_model).items()}

def check_native_parameters(backbone, binding):
    actual = native_parameter_hashes(backbone)
    if len(actual) != 927 or hashed(actual) != binding['frozen927HashesSha256']:
        raise ValueError('Actual927 frozen tensors differ from verified cache binding')
    return {'tensorsHashed': 927, 'sha256': hashed(actual), 'matchesBinding': True}

def _native_dependencies():
    from mlx_lm.models.cache import make_prompt_cache
    from .fused_prefix import _complete_cache, _memory
    from .native_batch_diagnosis import state_hash
    return {'cache_factory': make_prompt_cache, 'validate_cache': _complete_cache,
        'memory_guard': _memory, 'state_hash': state_hash, 'parameter_check': check_native_parameters}

class QueryPrefixReuse:
    """One immutable public constant root; each query forks all32 native states.

    Root-only prepare() is explicit. No query text/tokens/features/results are
    retained on this object. This is not a drop-in change to FrozenBackbone.encode.
    """
    @classmethod
    def prepare(cls, backbone, binding, *, copy_adapter=None, native_owner=None, confirm_serialized=False, _dependencies=None):
        if native_owner != 'root' or confirm_serialized is not True: raise ValueError('Root must own/serialize native work before explicit prepare')
        if backbone.memory_budget != 7 * 2**30: raise ValueError('Unchanged7GiB guard required')
        if backbone.revision != binding['modelRevision']: raise ValueError('Resident model revision differs from bound assets')
        deps = _native_dependencies() if _dependencies is None else _dependencies
        started = time.perf_counter(); assert_assets(binding, full_sha=True)
        obj = cls(); obj.bb = backbone; obj.binding = copy.deepcopy(binding); obj.adapter = copy_adapter; obj.deps = deps
        obj.model = backbone.model; obj.tokens = prefix_tokens(backbone.tokenizer)
        obj.cache_key = hashed({'assetKey': binding['assetKey'], 'frozen927HashesSha256': binding['frozen927HashesSha256'],
            'modelRevision': binding['modelRevision'], 'prefixTokenIds': obj.tokens,
            'mode': 'frozen-base-copy-adapter-disabled', 'maximumWholeRequestTokens': MAX_TOKENS, 'version': VERSION})
        mx = backbone.mx; deps['memory_guard'](backbone)
        with frozen_base(backbone, copy_adapter):
            parameters = deps['parameter_check'](backbone, binding)
            obj.cache = deps['cache_factory'](backbone.model)
            inputs = mx.array([obj.tokens]); mx.eval(inputs)
            hidden = backbone.model.model(inputs, cache=obj.cache)
            mx.eval(hidden, [e.state for e in obj.cache]); mx.synchronize()
            backbone.forward_calls += 1
            deps['validate_cache'](obj.cache, len(obj.tokens))
            obj.source_hash = deps['state_hash'](obj.cache)
        obj.preparation = {'status': 'prepared-not-benchmark-validated', 'version': VERSION,
            'prefixTokens': len(obj.tokens), 'prefixTokenSha256': hashed(obj.tokens),
            'prefixLiteralSha256': binding['prefixLiteralSha256'], 'withholdBoundaryTokens': 1,
            'backboneForwardCalls': 1, 'prefixOnlyNoDynamicRequestFields': True,
            'cacheTypes': [type(e).__name__ for e in obj.cache], 'sourceStateSha256': obj.source_hash,
            'cacheArrayBytesReportedByRuntime': (sum(e.nbytes for e in obj.cache) if all(hasattr(e, 'nbytes') for e in obj.cache) else None),
            'assetKey': binding['assetKey'], 'cacheBindingKey': obj.cache_key, 'frozen927': parameters,
            'prepareMsExcludedFromWarmRequests': (time.perf_counter() - started) * 1000,
            'cacheSharing': 'all KV history+offset and both convolution/SSM states, full deepcopy per B1 suffix; no trim, no shared mutable query branch'}
        deps['memory_guard'](backbone)
        return obj

    def encode(self, request, *, formatter=request_text, host_converter=None):
        bb, mx, deps = self.bb, self.bb.mx, self.deps
        started = time.perf_counter(); assert_assets(self.binding)
        if bb.model is not self.model: raise ValueError('Prefix belongs to a different resident model')
        layout = query_layout(bb.tokenizer, request, formatter=formatter)
        if layout['prefixTokenIds'] != self.tokens: raise ValueError('Tokenizer prefix changed after preparation')
        tokenize_ms = (time.perf_counter() - started) * 1000
        with frozen_base(bb, self.adapter):
            deps['memory_guard'](bb)
            begin = time.perf_counter(); before = deps['state_hash'](self.cache)
            if before != self.source_hash: raise AssertionError('Immutable source prefix changed before query')
            proof_before = (time.perf_counter() - begin) * 1000
            begin = time.perf_counter()
            def materialize(values): mx.eval(values); mx.synchronize()
            cache = clone_complete_cache(self.cache, materialize=materialize)
            deps['validate_cache'](cache, len(self.tokens))
            clone_ms = (time.perf_counter() - begin) * 1000
            suffix = layout['suffixTokenIds']; inputs = mx.array([suffix]); mx.eval(inputs)
            begin = time.perf_counter(); hidden = bb.model.model(inputs, cache=cache)
            # EXACT original pooling/dtype/L2 operation order, last valid suffix token.
            pooled = hidden[mx.arange(1), mx.array([len(suffix)]) - 1].astype(mx.float32)
            pooled = pooled / mx.maximum(mx.linalg.norm(pooled, axis=1, keepdims=True), 1e-8)
            mx.eval(pooled, [entry.state for entry in cache]); mx.synchronize()
            if not bool(mx.all(mx.isfinite(pooled)).item()): raise FloatingPointError('Nonfinite fresh suffix feature')
            bb.forward_calls += 1
            forward_ms = (time.perf_counter() - begin) * 1000
            begin = time.perf_counter()
            if host_converter is None:
                import torch
                features = torch.from_numpy(bb.np.array(pooled)).clone()
            else: features = host_converter(pooled)
            host_ms = (time.perf_counter() - begin) * 1000
            feature_ready = (time.perf_counter() - started) * 1000
            begin = time.perf_counter(); deps['validate_cache'](cache, len(layout['fullTokenIds']))
            after = deps['state_hash'](self.cache)
            if after != before: raise AssertionError('Fresh query mutated reusable source KV/SSM state')
            assert_assets(self.binding); deps['memory_guard'](bb)
            proof_after = (time.perf_counter() - begin) * 1000
        return features, {'version': VERSION, 'batchRows': 1, 'wholeRequestTokens': len(layout['fullTokenIds']),
            'reusedPrefixTokens': len(self.tokens), 'freshSuffixTokens': len(suffix), 'paddedTokensPerRow': len(suffix),
            'wholeTextSha256': layout['wholeTextSha256'], 'wholeTokenSha256': layout['wholeTokenSha256'],
            'backboneForwardCalls': 1, 'prefixPreparationCallsAmortizedSeparately': 1, 'truncatedRows': 0,
            'tokenizeAndAssetStatMs': tokenize_ms, 'sourceProofBeforeMs': proof_before, 'cacheCloneMaterializeMs': clone_ms,
            'freshSuffixForwardNormalizeMs': forward_ms, 'hostTorchCopyMs': host_ms,
            'featureReadyMsBeforeFinalProof': feature_ready, 'finalProofAndMemoryGuardMs': proof_after,
            'totalEncodeIncludingProofMs': (time.perf_counter() - started) * 1000,
            'sourceBeforeSha256': before, 'sourceAfterSha256': after, 'sourceUnchanged': True,
            'noResponseOrQueryFeatureCache': True, 'assetStatChecksInsideTiming': True,
            'fullAssetAnd927HashChecks': 'preparation and required final development gate; not cryptographic per-request disk verification'}

def compare_features(reference, candidate):
    import torch
    if tuple(reference.shape) != (1, 4096) or tuple(candidate.shape) != (1, 4096): raise ValueError('Exact B1/4096 feature shape required')
    a, b = reference.float(), candidate.float()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()): raise FloatingPointError('Finite features required')
    cosine = float(torch.nn.functional.cosine_similarity(a, b).item()); error = float((a - b).abs().max().item())
    return {'bitwiseEqual': torch.equal(a, b), 'cosine': cosine, 'maximumAbsoluteDifference': error,
        'minimumCosineDeclared': MIN_COSINE, 'maximumAbsoluteErrorDeclared': MAX_ABS_ERROR,
        'numericalGate': cosine >= MIN_COSINE and error <= MAX_ABS_ERROR}

def score_function_head(session, catalog, feature, request):
    """Existing head and guard functions only; never alters session/head/catalog."""
    import torch
    from .store_function_adaptation import rank
    from .store_rank_session import exact_name_order, action_bindings
    session.source_validator(session.snapshot['receipt'])
    if hasattr(session, '_assert_head_assets'): session._assert_head_assets()
    with torch.no_grad(): scores = (session.head.probabilities('catalog', catalog) @ session.head.probabilities('query', feature)[0]).tolist()
    guarded = rank(scores, request, session.products, 6); order, identity = exact_name_order(request, session.products, guarded)
    return {'scores16': scores, 'learnedRank': guarded, 'productIndices': order[:6], 'eligibleCount': len(order),
        'shortfall': max(0, 6 - len(order)), 'exactNamePolicy': identity,
        'cards': action_bindings(request, order[:6], session.snapshot)}

def decision_parity(reference, candidate):
    keys = {'scores16', 'learnedRank', 'productIndices', 'eligibleCount', 'shortfall', 'exactNamePolicy', 'cards'}
    if set(reference) != keys or set(candidate) != keys: raise ValueError('Full scoring/guard/identity/action receipt required')
    if len(reference['scores16']) != 16 or len(candidate['scores16']) != 16: raise ValueError('All16 scores required')
    if not all(math.isfinite(v) for r in (reference, candidate) for v in r['scores16']): raise FloatingPointError('Finite scores required')
    score_error = max(abs(a - b) for a, b in zip(reference['scores16'], candidate['scores16'], strict=True))
    same = canonical({k: reference[k] for k in keys - {'scores16'}}) == canonical({k: candidate[k] for k in keys - {'scores16'}})
    return {'all16OrderGuardsIdentityAndActionsExact': same, 'maxScoreAbsoluteDifference': score_error,
        'decisionGate': same and score_error <= MAX_ABS_ERROR}

def development_controls(encoder, cases, score_and_bind, *, out):
    """Root calls AFTER cohort window under its native owner lock. Fixed32
    public requests only, no labels. New output required; partial failures saved.
    All64 fresh forwards are controls; no throughput/UI benchmark is implied.
    """
    if len(cases) != 32 or len({c['caseId'] for c in cases}) != 32 or any(set(c) != {'caseId', 'request'} for c in cases):
        raise ValueError('Exactly32 unique caseId/request-only public controls required; no gold/targets')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    report = {'version': VERSION, 'status': 'started', 'scope': 'Development replay feature and all16 rank/guard/action controls; no UI/throughput or unseen generalization claim',
        'fixedDenominator': 32, 'prefixPreparation': encoder.preparation, 'arms': [{'caseId': c['caseId'], 'status': 'unattempted'} for c in cases],
        'numericPolicy': {'minimumCosine': MIN_COSINE, 'maximumAbsoluteError': MAX_ABS_ERROR}, 'trainingUpdates': 0}
    def save(): (out / 'controls.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    try:
        for index, (case, arm) in enumerate(zip(cases, report['arms'], strict=True)):
            arm.update(status='running', order=['full-cache-none', 'reused-prefix'] if index % 2 == 0 else ['reused-prefix', 'full-cache-none']); save()
            features = {}; timings = {}
            for label in arm['order']:
                if label == 'full-cache-none':
                    with frozen_base(encoder.bb, encoder.adapter): features[label], timings[label] = encoder.bb.encode([request_text(case['request'])])
                else: features[label], timings[label] = encoder.encode(case['request'])
            a = score_and_bind(features['full-cache-none'], case['request']); b = score_and_bind(features['reused-prefix'], case['request'])
            arm.update(status='completed', featureParity=compare_features(features['full-cache-none'], features['reused-prefix']),
                decisionParity=decision_parity(a, b), baseline=a, candidate=b, timings=timings); save()
        assert_assets(encoder.binding, full_sha=True)
        with frozen_base(encoder.bb, encoder.adapter): report['final927Hashes'] = encoder.deps['parameter_check'](encoder.bb, encoder.binding)
        if encoder.deps['state_hash'](encoder.cache) != encoder.source_hash: raise AssertionError('Prefix source changed over32 controls')
        report.update(status='completed', fullAssetHashesUnchanged=True, prefixSourceUnchanged=True,
            all32NumericAndRankGuardIdentityGatesPassed=all(a['featureParity']['numericalGate'] and a['decisionParity']['decisionGate'] for a in report['arms']))
        report['usableLatencyInterpretationGate'] = report['all32NumericAndRankGuardIdentityGatesPassed']
    except BaseException as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)}, usableLatencyInterpretationGate=False)
        for arm in report['arms']:
            if arm['status'] == 'running': arm.update(status='failed', error=report['error'])
        save(); raise
    save(); return report
