"""Public query-only plan/extraction for combined adjudication; no default native.

The full LoRA packet is projected to public nonfit requests. Only original v5
base query4096 is encoded, with no rank/copy adapter installed, no context model,
no targets, labels or continuation. Root owns explicit native execution.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time
import torch

from . import store_persona_adapter_contract_v2 as contract
from . import store_qwen_lexical_choice_inputs_v2 as formatter
from . import store_persona_context_v1 as context
from .store_demo_engine import ranking_input
from .store_function_adaptation import request_text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAIN = Path('/Users/lukas/Documents/Projects/dubhacks-2026')
VERSION = 'persona-combined-public-query.v1'
EXPECTED_NONFIT = {'evaluation-main': 64, 'evaluation-replacement': 8, 'evaluation-controls': 192,
    'development-history': 40, 'demo': 20, 'demo-default-query': 20}
RECIPE = {'dimensions': 4096, 'maxTokens': 512, 'memoryGiB': 7, 'maxNativeSeconds': 600,
    'batchRows': 1, 'adapter': 'none', 'feature': 'exact-v5-runtime-query4096-L2',
    'dedup': 'exact UTF8 runtime query text; canonical/reversed share query', 'optimizerUpdates': 0}
SOURCES = [Path(__file__), Path(contract.__file__), Path(formatter.__file__), Path(context.__file__),
    HERE/'store_demo_engine.py', HERE/'store_function_adaptation.py', HERE/'backbone.py',
    HERE/'semantic_copy_training.py', HERE/'semantic_copy_feature_fit.py', HERE/'store_copy_fit_v2.py',
    HERE/'catalog_fusion/generalization_v2.py']


def sha(path): return contract.sha(path)
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
def digest_text(value): return hashlib.sha256(value.encode('utf8')).hexdigest()
def dump(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def pincheck(pins):
    for filename, expected in pins.items():
        if sha(filename) != contract.exact_sha(expected): raise ValueError('Combined query source/input pin changed: '+filename)


@contextmanager
def cwd(path):
    import os
    previous = Path.cwd(); os.chdir(path)
    try: yield
    finally: os.chdir(previous)


def public_products():
    catalog = formatter.load_public_catalog()
    return [{**p, 'productId': p['id'], 'title': p['name'],
        'variants': [{'variantId': v['id'], 'colorName': v['colorName']} for v in p['variants']]} for p in catalog]


def runtime_text(value, products=None):
    request = contract.common_request(value, formatter.load_public_catalog(), products or public_products())[0]
    return request_text(ranking_input(request))


def vector(values):
    if isinstance(values, torch.Tensor) and values.device.type != 'cpu': raise ValueError('CPU base query vector required')
    values = torch.as_tensor(values, dtype=torch.float32)
    if values.ndim == 1: values = values[None, :]
    if (tuple(values.shape) != (1, 4096) or not torch.isfinite(values).all() or
            not torch.allclose(values.norm(dim=-1), torch.ones(1), atol=.002, rtol=0)):
        raise ValueError('Finite unit-L2 original base query4096 required')
    return values


def project_rows(full_plan, historical=None):
    """Explicit public-only projection; TRAIN targets/feature tensors are discarded."""
    nonfit = [r for r in full_plan['rows'] if r['split'] != 'train']
    if {s: sum(r['split'] == s for r in nonfit) for s in EXPECTED_NONFIT} != EXPECTED_NONFIT or len(nonfit) != 344:
        raise ValueError('Exact fixed all344 nonfit row roster required')
    rows = []
    for source in nonfit:
        if source['caseId'] != source['profileId']+'-'+source['order'] or source['order'] not in ('canonical', 'reversed'):
            raise ValueError('Exact public profile/order/case identity required')
        rows.append({key: deepcopy(source.get(key)) for key in
            ('caseId', 'profileId', 'userGroupId', 'split', 'sourceProfileId', 'controlName', 'publicRequest')}
            | {'orderLabel': source['order'], 'scope': 'combined-all344'})
    if len({r['caseId'] for r in rows}) != 344: raise ValueError('Duplicate nonfit case identity')
    if historical is not None:
        if len(historical['cases']) != 320 or historical.get('publicOnly') is not True:
            raise ValueError('Exact separate320 public historical request roster required')
        for source in historical['cases']:
            rows.append({'caseId': 'historical-query-only/'+source['probeCaseId'], 'profileId': source['publicDemoId'],
                'userGroupId': source['publicDemoId'], 'split': 'historical-specific-query-only',
                'sourceProfileId': None, 'controlName': None, 'publicRequest': deepcopy(source['request']),
                'orderLabel': None, 'scope': 'query-only; LoRA context decisions not computed or claimed'})
    if len({r['caseId'] for r in rows}) != len(rows): raise ValueError('Duplicate query roster identity')
    return rows


def prepare(full_plan_path, full_plan_sha, out, *, historical_plan_path=None, historical_plan_sha=None):
    from tokenizers import Tokenizer
    full_plan_path = Path(full_plan_path).resolve(); out = Path(out)
    if out.exists(): raise FileExistsError('Preserve prior query preparation outputs')
    if sha(full_plan_path) != contract.exact_sha(full_plan_sha): raise ValueError('Explicit exact full1240 plan SHA required')
    full = json.loads(full_plan_path.read_text()); historical = None
    pins = {str(path.resolve()): sha(path) for path in SOURCES}
    pins[str(full_plan_path)] = full_plan_sha
    if historical_plan_path is not None:
        path = Path(historical_plan_path).resolve()
        if sha(path) != contract.exact_sha(historical_plan_sha): raise ValueError('Exact selected historical320 public plan required')
        historical = json.loads(path.read_text()); pins[str(path)] = historical_plan_sha
    elif historical_plan_sha is not None: raise ValueError('Historical path and SHA must be supplied together')
    source_rows = project_rows(full, historical); products = public_products()
    model_files = full['modelFiles']; tokenizer_file = Path(model_files['modelPath'])/'tokenizer.json'
    if sha(tokenizer_file) != model_files['smallFileSha256']['tokenizer.json']: raise ValueError('Exact pinned local tokenizer required')
    tokenizer = Tokenizer.from_file(str(tokenizer_file)); unique = {}; rows = []
    for source in source_rows:
        text = runtime_text(source['publicRequest'], products); key = digest_text(text)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not 0 < len(ids) <= 512: raise ValueError('Whole actual runtime query exceeds512tokens; no clipping')
        item = {'text': text, 'textSha256': key, 'tokenIds': ids, 'tokens': len(ids),
            'tokenIdsSha256': contract.token_hash(ids)}
        if key in unique and unique[key] != item: raise ValueError('Query SHA collision')
        unique[key] = item
        rows.append({**source, 'publicRequestSha256': digest_text(canonical(source['publicRequest'])), 'queryTextSha256': key})
    # Changed public profile/history alone must not change the query-feature path.
    by_profile = {r['profileId']: r for r in rows if r['scope'] == 'combined-all344' and r['orderLabel'] == 'canonical'}
    for row in rows:
        if row['scope'] != 'combined-all344': continue
        original = by_profile.get(row['sourceProfileId'] or row['profileId'])
        if original is not None and row['queryTextSha256'] != original['queryTextSha256']:
            raise ValueError('Input control changed actual query text; do not reuse source pSpecific/v5 silently')
    packet = {'version': VERSION, 'recipe': RECIPE, 'rows': rows, 'uniqueQueries': list(unique.values()),
        'sha256': pins, 'modelFiles': model_files, 'nativeOwnerLock': str(MAIN/'data/ranker/pretrained/native-owner.lock'),
        'counts': {'combinedRows': 344, 'historicalQueryOnlyRows': len(rows)-344, 'allRows': len(rows),
            'uniqueBaseQueryForwardsRequired': len(unique)}, 'goldParsed': False, 'targetOrTeacherInputs': False,
        'historicalLoRAOrCombinedRegressionComputed': False, 'nativeExecuted': False, 'optimizerUpdates': 0}
    out.mkdir(parents=True); dump(out/'plan.json', packet)
    dump(out/'PLAN_FROZEN.json', {'version': VERSION, 'planSha256': sha(out/'plan.json'), 'counts': packet['counts'],
        'goldParsed': False, 'nativeExecuted': False, 'optimizerUpdates': 0})
    return packet


def extract(plan_path, expected_sha, out, *, execute=False, owner=None):
    plan_path = Path(plan_path).resolve()
    if sha(plan_path) != contract.exact_sha(expected_sha): raise ValueError('Exact frozen query plan SHA required')
    packet = json.loads(plan_path.read_text())
    if packet['version'] != VERSION or packet['recipe'] != RECIPE: raise ValueError('Exact query-only recipe required')
    pincheck(packet['sha256'])
    if not execute: return {'status': 'cpu-preflight-only', 'nativeExecuted': False, 'counts': packet['counts']}
    if owner != 'root': raise ValueError('Root is sole native owner; no implicit execute')
    from .store_qwen_choice_pilot_v1 import listening
    if any(listening(p) for p in (8766, 8767)): raise RuntimeError('Serving active; no competing native worker')
    out = Path(out)
    if not out.is_absolute() or not out.is_relative_to(ROOT) or out.exists(): raise ValueError('Unique absolute worktree query output required')
    out.mkdir(parents=True); dump(out/'plan.json', packet)
    from .catalog_fusion.generalization_v2 import native_owner_lock
    from .semantic_copy_feature_fit import verify_model_files
    from .semantic_copy_training import parameter_hashes
    from .store_copy_fit_v2 import deadline
    from .backbone import FrozenBackbone
    report = {'version': VERSION, 'status': 'starting', 'sourceManifestSha256': expected_sha, 'goldParsed': False,
        'adapterInstalled': False, 'optimizerUpdates': 0, 'attemptedNativeForwards': 0, 'physicalNativeForwards': 0,
        'savedVectors': 0, 'expectedNativeForwards': len(packet['uniqueQueries']), 'frozen927Unchanged': None,
        'shutdownCompleted': False, 'roster': [{'textSha256': r['textSha256'], 'status': 'unattempted'} for r in packet['uniqueQueries']]}
    resources = ExitStack(); bb = before = None; values = {}; start = time.perf_counter()
    save = lambda: dump(out/'report.json', report); save()
    try:
        resources.enter_context(native_owner_lock(Path(packet['nativeOwnerLock']), owner))
        with deadline(RECIPE['maxNativeSeconds'], 'public combined query-only extraction'):
            if any(listening(p) for p in (8766, 8767)): raise RuntimeError('Serving became active')
            with cwd(MAIN): report['modelVerification'] = verify_model_files(packet['modelFiles'])
            bb = FrozenBackbone(packet['modelFiles']['modelPath'], packet['modelFiles']['revision'], max_length=512, memory_gib=7)
            bb.model.freeze(); bb.model.eval(); before = parameter_hashes(bb.model.language_model)
            if len(before) != 927: raise ValueError('Exactly927 pristine base tensors required')
            report['beforeParameterSha256'] = before; save()
            for row, receipt in zip(packet['uniqueQueries'], report['roster'], strict=True):
                receipt['status'] = 'running'; save()
                if bb.tokenizer.encode(row['text'], add_special_tokens=False) != row['tokenIds']: raise ValueError('Whole actual runtime query tokenizer differs')
                report['attemptedNativeForwards'] += 1; save()
                result, timing = bb.encode([row['text']]); report['physicalNativeForwards'] += 1
                if (timing['batchRows'] != 1 or timing['truncatedRows'] != 0 or timing['backboneForwardCalls'] != 1 or
                        timing['paddedTokensPerRow'] != row['tokens']): raise ValueError('Whole exactB1 native query forward required')
                result = vector(result); values[row['textSha256']] = result.clone()
                torch.save({'version': VERSION, 'sourceManifestSha256': expected_sha, 'features': values}, out/'query-features.pt')
                report['savedVectors'] += 1; receipt.update(status='completed', timing=timing)
                memory = bb.runtime_memory()
                if max(memory['mlxPeakBytes'], memory['mlxActiveBytes']) > 7*2**30: raise MemoryError('Explicit7GiB native guard exceeded')
                pincheck(packet['sha256']); save()
            report.update(status='completed-query-features', queryFeaturesSha256=sha(out/'query-features.pt'))
    except BaseException as error:
        for row in report['roster']:
            if row['status'] == 'running': row.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)}); raise
    finally:
        try:
            if bb is not None:
                try:
                    after = parameter_hashes(bb.model.language_model)
                    report['afterParameterSha256'] = after
                    report['frozen927Unchanged'] = before is not None and len(after) == 927 and before == after
                    report['memory'] = bb.runtime_memory()
                    if not report['frozen927Unchanged']: report['status'] = 'failed-frozen-integrity'
                except BaseException as error: report.update(status='failed-shutdown-integrity', shutdownError=str(error))
                finally:
                    mx = bb.mx; bb = None; gc.collect()
                    try: mx.clear_cache(); mx.synchronize(); report['shutdownCompleted'] = True
                    except BaseException as error: report.update(status='failed-shutdown', shutdownError=str(error))
            try: pincheck(packet['sha256']); report['sourcePinsUnchanged'] = True
            except BaseException as error: report.update(status='failed-source-change', sourceError=str(error))
        finally:
            report['elapsedSeconds'] = time.perf_counter()-start; save(); resources.close()
    if (report['status'] != 'completed-query-features' or not report['frozen927Unchanged'] or not report['shutdownCompleted'] or
            report['savedVectors'] != report['expectedNativeForwards']): raise RuntimeError('Query extraction lifecycle incomplete')
    return report


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(__doc__); p.add_argument('--prepare', action='store_true'); p.add_argument('--run', action='store_true')
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--plan-sha256', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--historical-plan', type=Path); p.add_argument('--historical-plan-sha256'); p.add_argument('--owner')
    args = p.parse_args(argv)
    if args.prepare and args.run: p.error('Choose prepare or explicit native run, never both')
    result = (prepare(args.plan,args.plan_sha256,args.out,historical_plan_path=args.historical_plan,historical_plan_sha=args.historical_plan_sha256)
        if args.prepare else extract(args.plan,args.plan_sha256,args.out,execute=args.run,owner=args.owner))
    print(json.dumps(result if 'rows' not in result else {'status':'prepared','counts':result['counts'],'planSha256':sha(args.out/'plan.json')}, indent=2))


if __name__ == '__main__': main()
