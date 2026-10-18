"""Freeze all four combined candidate/control models before private gold.

CPU only. Each raw16 model uses the identical learned query-intent probability,
verified original base query4096, v5 scores and strict public guards. No variant,
image, copy or historical LoRA regression outcome is fabricated by this writer.
"""
from copy import deepcopy
import json
from pathlib import Path
import torch

from . import store_persona_combined_query_reuse_v2 as queries
from . import store_persona_adapter_contract_v2 as contract
from . import store_persona_adapter_runtime_v2 as runtime
from . import store_persona_context_v1 as context
from . import store_query_intent_gate_v3 as gate
from .store_demo_engine import guarded_order
from .store_rank_session_v5 import read_export
from .store_function_adaptation import load_head

VERSION = 'persona-choice-clear-combined-predictions.v1'
MODEL_FILES = {'learned': 'learned-nonfit-predictions.json',
    'frozen-base-lexical-decision': 'frozen-base-nonfit-predictions.json',
    'TRAIN-popularity': 'training-popularity-nonfit-predictions.json',
    'latest-owned-repeat': 'latest-owned-repeat-nonfit-predictions.json'}
FUSION = '(1-p)*softmax(source)+p*(v5+0.05*softmax(source))'
SOURCES = [Path(__file__), Path(queries.__file__), Path(contract.__file__), Path(runtime.__file__),
    Path(context.__file__), Path(gate.__file__), Path(__file__).with_name('store_demo_engine.py'),
    Path(__file__).with_name('store_function_adaptation.py'), Path(__file__).with_name('store_rank_session_v5.py')]


def score_vector(value):
    result = torch.as_tensor(value, dtype=torch.float32)
    if tuple(result.shape) != (16,) or not torch.isfinite(result).all(): raise ValueError('Finite canonical16 source scores required')
    return result


def prediction_map(records, rows):
    if type(records) is not list or len(records) != len(rows): raise ValueError('All344 raw predictions required; no selective roster')
    indexed = {r['caseId']: r for r in records}
    if len(indexed) != len(records) or set(indexed) != {r['caseId'] for r in rows}: raise ValueError('Raw source case roster differs')
    for row in rows:
        source = indexed[row['caseId']]
        for key in ('profileId', 'userGroupId', 'caseId', 'split', 'sourceProfileId', 'controlName'):
            if source.get(key) != row.get(key): raise ValueError('Raw source/public identity mismatch: '+key)
        if source['orderLabel'] != row['orderLabel']: raise ValueError('Raw source catalog order label mismatch')
        score_vector(source['scores16'])
    return indexed


def compose_rows(rows, raw_predictions, query_values, products, *, gate_sha):
    """Inject public p/v5 for CPU tests; production validates native feature proof."""
    if set(raw_predictions) != set(MODEL_FILES): raise ValueError('All four raw models required')
    by_model = {name: prediction_map(records, rows) for name, records in raw_predictions.items()}
    output = {name: [] for name in MODEL_FILES}
    by_profile = {r['profileId']: r for r in rows if r['orderLabel'] == 'canonical'}
    for row in rows:
        key = row['queryTextSha256']
        if key not in query_values: raise ValueError('Missing exact actual-runtime query feature/prediction')
        p, v5_values = query_values[key]; v5 = score_vector(v5_values)
        original = by_profile.get(row['sourceProfileId'] or row['profileId'])
        if original is not None and original['queryTextSha256'] != key:
            raise ValueError('Input control has different runtime query; source p/v5 sharing is invalid')
        request, policy, _ = context.checked_context_request(row['publicRequest'], products)
        request_sha = queries.digest_text(queries.canonical(row['publicRequest']))
        if request_sha != row['publicRequestSha256']: raise ValueError('Public request bytes changed')
        for name in MODEL_FILES:
            raw = score_vector(by_model[name][row['caseId']]['scores16'])
            scores = runtime.blended_scores(v5, raw, p).tolist()
            order, guards = guarded_order(scores, request, policy, products)
            output[name].append({key: deepcopy(row.get(key)) for key in
                ('profileId', 'userGroupId', 'caseId', 'split', 'orderLabel', 'sourceProfileId', 'controlName')}
                | {'publicRequestSha256': request_sha, 'queryTextSha256': row['queryTextSha256'],
                    'rawSourceScores16': deepcopy(by_model[name][row['caseId']]['scores16']), 'pSpecific': float(p), 'v5Scores16': v5.tolist(), 'scores16': scores,
                    'fusionPolicy': {'version': 'learned-query-intent-blend.v1', 'formula': FUSION,
                        'gateCheckpointSha256': gate_sha, 'sameGateV5AndGuardsForEveryModel': True},
                    'rawLearnedOrder': sorted(range(16), key=lambda i: (-scores[i], i)), 'order': order,
                    'rankedProductIds': [products[i]['productId'] for i in order], 'guardChecks': guards,
                    'copyGenerated': False, 'variantsOrImagesScored': False})
    return output


def query_bundle(run, expected_plan_sha):
    return queries.bundle(run, expected_plan_sha)

def write_predictions(raw_run, raw_plan_path, raw_plan_sha, query_run, query_plan_sha, out, *,
                      gate_checkpoint, gate_sha, v5_export, v5_export_sha, catalog_feature_root,
                      independent_packet_manifest, independent_packet_manifest_sha):
    """No evaluator imported, no private gold path accepted, no native imports."""
    raw_run = Path(raw_run).resolve(); raw_plan_path = Path(raw_plan_path).resolve(); out = Path(out)
    if out.exists(): raise FileExistsError('Preserve every prior combined attempt')
    if not out.is_absolute() or not out.is_relative_to(queries.ROOT): raise ValueError('Unique absolute worktree combined output required')
    if queries.sha(raw_plan_path) != contract.exact_sha(raw_plan_sha): raise ValueError('Exact original frozen1240 plan required')
    full = json.loads(raw_plan_path.read_text()); expected_rows = queries.project_rows(full)
    manifest_file = Path(independent_packet_manifest).resolve()
    if queries.sha(manifest_file) != contract.exact_sha(independent_packet_manifest_sha): raise ValueError('Exact independent public-packet manifest required')
    if full['sha256'].get(str(manifest_file)) != independent_packet_manifest_sha: raise ValueError('Independent packet was not frozen in original LoRA plan')
    pins = {str(path.resolve()): queries.sha(path) for path in SOURCES}
    pins.update({str(raw_plan_path): raw_plan_sha, str(manifest_file): independent_packet_manifest_sha})
    qplan, query_features, query_pins = query_bundle(query_run, query_plan_sha); pins.update(query_pins); pins.update(qplan['sha256'])
    if qplan['sha256'].get(str(raw_plan_path)) != raw_plan_sha: raise ValueError('Query features not from this exact public LoRA plan')
    rows = [r for r in qplan['rows'] if r['scope'] == 'combined-all344']
    identity = lambda r: {k: r[k] for k in expected_rows[0]}
    if [identity(r) for r in rows] != expected_rows: raise ValueError('Exact fixed all344 public request/order roster required')
    report = json.loads((raw_run/'report.json').read_text()); frozen = json.loads((raw_run/'ALL_PREDICTIONS_FROZEN.json').read_text())
    if (full.get('version') != 'persona-choice-clear-training.v2' or report.get('version') != 'persona-choice-clear-training.v2' or report['status'] != 'completed-fixed-fit-and-predictions' or report['planSha256'] != raw_plan_sha or
            report['actualUpdates'] != 3584 or report['privateEvaluationLabelsParsed'] is not False or
            report['frozen927HashesUnchanged'] is not True or report['shutdownCompleted'] is not True or report['sourcePinsUnchanged'] is not True or
            frozen['privateEvaluationLabelsParsed'] is not False or frozen['nonfitPromptRecords'] != 344):
        raise ValueError('Completed frozen all344 LoRA/control predictions and cleanup required')
    serving_path = raw_run/'serving.json'; serving, cert = runtime.load_manifest(serving_path, queries.sha(serving_path))
    adapter_sha = serving['adapter']['sha256']
    if (frozen['checkpointSha256'] != adapter_sha or report['fit']['checkpointSha256'] != adapter_sha or
            serving.get('nativeReportSha256') != queries.sha(raw_run/'report.json')):
        raise ValueError('Raw report/predictions and serving adapter differ')
    file_shas = {'learned-nonfit-predictions.json': frozen['learnedPredictionsSha256'],
        'frozen-base-nonfit-predictions.json': frozen['frozenBasePredictionsSha256'],
        **frozen['baselineFilesSha256']}
    # Trainer uses names without .json only in baselineFilesSha256 keys.
    raw_predictions = {}
    for model, filename in MODEL_FILES.items():
        checksum = file_shas.get(filename)
        if checksum is None: checksum = file_shas.get(filename.removesuffix('-nonfit-predictions.json'))
        if queries.sha(raw_run/filename) != contract.exact_sha(checksum): raise ValueError('Frozen raw model prediction file changed')
        pins[str(raw_run/filename)] = checksum; raw_predictions[model] = json.loads((raw_run/filename).read_text())
    for name in ('report.json', 'ALL_PREDICTIONS_FROZEN.json', 'BASELINE_DEFINITIONS_FROZEN.json', 'serving.json',
                 serving['adapter']['file'], serving['publicCache']['file'], serving['inferenceParity']['file']):
        pins[str(raw_run/name)] = queries.sha(raw_run/name)
    catalog = context.load_catalog_features(catalog_feature_root); pins.update(catalog['pins'])
    export_path = Path(v5_export).resolve(); export = read_export(export_path, v5_export_sha)
    if export['featureProvenance'] != catalog['provenance']: raise ValueError('V5 and catalog base feature provenance differ')
    head_path = export_path.parent/export['head']['file']
    head = load_head(head_path, export['head']['sha256'], export['head']['metadata']).eval().requires_grad_(False)
    if not torch.equal(head.catalog_mean, catalog['vectors'].double().mean(0).float()): raise ValueError('Original16 v5 catalog mean differs')
    gate_checkpoint = Path(gate_checkpoint).resolve()
    intent, gate_metadata = gate.load_intent_head(gate_checkpoint, gate_sha,
        expected_source_manifest_sha256=runtime.INTENT_TRAIN_MANIFEST_SHA)
    pins.update({str(export_path): v5_export_sha, str(head_path): export['head']['sha256'], str(gate_checkpoint): gate_sha})
    query_values = {}
    with torch.inference_mode():
        catalog_prob = head.probabilities('catalog', catalog['vectors'])
        for key in {r['queryTextSha256'] for r in rows}:
            qraw = queries.vector(query_features[key])
            query_values[key] = gate.specific_probability(intent, qraw), (catalog_prob @ head.probabilities('query', qraw)[0]).tolist()
    output = compose_rows(rows, raw_predictions, query_values, catalog['products'], gate_sha=gate_sha)
    queries.pincheck(pins); out.mkdir(parents=True)
    models = {}
    for name, records in output.items():
        filename = name+'-combined-predictions.json'; queries.dump(out/filename, records)
        models[name] = {'file': filename, 'sha256': queries.sha(out/filename)}
    queries.dump(out/'QUERY_REPLAY.json', {'version': VERSION, 'queryPlanSha256': query_plan_sha,
        'queryFeatureBundleSha256': query_pins[str(Path(query_run).resolve()/'query-features.pt')],
        'gateCheckpointSha256': gate_sha, 'gateTRAINManifestSha256': gate_metadata['sourceManifestSha256'],
        'rows': [{'queryTextSha256': key, 'pSpecific': p, 'v5Scores16': values} for key, (p, values) in sorted(query_values.items())],
        'exactFullRuntimeQueryText': True, 'adapterParticipatedInQueryEncoding': False, 'goldParsed': False})
    pins[str((out/'QUERY_REPLAY.json').resolve())] = queries.sha(out/'QUERY_REPLAY.json')
    queries.pincheck(pins)
    freeze = {'version': VERSION, 'goldParsed': False, 'stagesCompleted': {'adapter': True, 'queryGate': True, 'allControls': True},
        'models': models, 'sha256': pins, 'adapterCheckpointSha256': adapter_sha, 'gateCheckpointSha256': gate_sha,
        'independentPacketManifestSha256': independent_packet_manifest_sha, 'canonicalIsPrimary': True,
        'reversedRobustnessOnly': True, 'fixedRowsPerModel': 344, 'caseIds': [r['caseId'] for r in rows],
        'variantsOrImagesScored': False, 'copyGenerated': False, 'specific320LoRARegressionComputed': False,
        'scope': 'offline combined/control predictions only; no quality, serving or live acceptance claim'}
    queries.dump(out/'ALL_MODELS_PREDICTIONS_FROZEN.json', freeze)
    return {'status': 'all-four-models-predictions-frozen-before-gold', 'models': models,
        'freezeSha256': queries.sha(out/'ALL_MODELS_PREDICTIONS_FROZEN.json'), 'rowsPerModel': 344, 'goldParsed': False}


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(__doc__)
    for name in ('raw-run', 'raw-plan', 'query-run', 'out', 'gate-checkpoint', 'v5-export', 'catalog-feature-root', 'independent-packet-manifest'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('raw-plan-sha256', 'query-plan-sha256', 'gate-sha256', 'v5-export-sha256', 'independent-packet-manifest-sha256'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(argv)
    print(json.dumps(write_predictions(a.raw_run,a.raw_plan,a.raw_plan_sha256,a.query_run,a.query_plan_sha256,a.out,
        gate_checkpoint=a.gate_checkpoint,gate_sha=a.gate_sha256,v5_export=a.v5_export,v5_export_sha=a.v5_export_sha256,
        catalog_feature_root=a.catalog_feature_root,independent_packet_manifest=a.independent_packet_manifest,
        independent_packet_manifest_sha=a.independent_packet_manifest_sha256),indent=2))


if __name__ == '__main__': main()
