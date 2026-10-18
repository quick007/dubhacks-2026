"""CPU integration receipt for the pinned jacket contract; no model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from .jacket_inputs import load_snapshot, catalog_text, task_text
from .jacket_plan import build_plan, validate_plan, copy_jobs


def run(out):
    started = time.perf_counter()
    out = Path(out)
    if out.exists():
        raise FileExistsError('Preserve prior output; choose a new result directory')
    snapshot = load_snapshot()
    products = snapshot['products']
    cases = []
    for index, task in enumerate(snapshot['tasks']):
        before = time.perf_counter()
        plan = build_plan(task, products)
        validate_plan(plan, task, products)
        jobs = copy_jobs(plan, task, products)
        elapsed = (time.perf_counter()-before)*1000
        if plan['eligibleCount'] != snapshot['guardReceipt']['tasks'][index]['eligibleCount']:
            raise AssertionError('New boundary differs from independently frozen eligibility')
        cases.append({'taskIndex': index, 'taskText': task_text(task), 'plan': plan,
                      'copyJobs': jobs, 'contractValidated': True,
                      'selectionValidationAndJobPreparationMs': elapsed})
    files = ('jacket_inputs.py', 'jacket_plan.py', 'jacket_contract_runner.py')
    report = {'schemaVersion': 'jacket-contract-integration.v1', 'status': 'completed',
        'scope': 'CPU public-input/guard/plan/job construction only; no inference or generated-copy validation',
        'sourceSha256': {n: hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in files},
        'snapshotReceipt': snapshot['receipt'], 'externalMapping': snapshot['externalMapping'],
        'catalogTexts': [catalog_text(p) for p in products], 'cases': cases,
        'summary': {'fixedTasks': len(cases), 'contractValid': len(cases),
            'eligibleCounts': [c['plan']['eligibleCount'] for c in cases],
            'returnedCards': sum(c['plan']['returnedCardCount'] for c in cases),
            'requestedCards': sum(c['plan']['requestedCardCount'] for c in cases),
            'shortfalls': [c['plan']['shortfall'] for c in cases],
            'preparedIndependentCopyJobs': sum(len(c['copyJobs']) for c in cases),
            'medianSelectionValidationAndJobPreparationMs': statistics.median(c['selectionValidationAndJobPreparationMs'] for c in cases)},
        'nativeForwards': 0, 'optimizerUpdates': 0, 'generatedFields': 0,
        'learnedRankingEstablished': False, 'learnedGroupingEstablished': False,
        'completePageQualityOrLatencyEstablished': False,
        'cpuRunWallMsExcludingImportAndReportIO': (time.perf_counter()-started)*1000}
    out.mkdir(parents=True, exist_ok=False)
    with (out/'report.json').open('x') as handle:
        json.dump(report,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n')
    return report['summary']


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--out',type=Path,required=True)
    print(json.dumps(run(parser.parse_args().out),sort_keys=True))
