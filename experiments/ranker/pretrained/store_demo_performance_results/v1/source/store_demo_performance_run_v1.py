"""Exclusive instrumented native comparison; default is CPU-only preflight.

The live service is never patched or stopped by this program. Process-local
function substitution selects an algorithm-preserving observer for paired arms.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from unittest.mock import patch
import uuid

from .store_ar_comparison_run_v1 import DEFAULT_MANIFEST, cpu_preflight, demo_listening, fixed_cases, reset_runtime
from .store_copy_fit_v2 import deadline
from .store_copy_data_v2 import CAPS, FIELDS

ROOT = Path(__file__).resolve().parent
ORDER = (('baseline', 'profile-v2'), ('profile-v2', 'baseline'), ('baseline', 'profile-v2'))
TIMER_POLICY = ('caller entry before rank through serializedfirst3/full6; archivewriteexcluded; '
    'controls/startupexcluded; profileadds explicit phase synchronization')
CACHE_POLICY = ('one warm resident backbone; one excluded complete baseline warmup; fresh request '
    'caches and allocator clear before each arm; no response/prompt-result caching')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_signature(receipt):
    """Timing-independent complete identity/field/token/EOS signature."""
    return {'cards': receipt['cards'], 'finalOrder': receipt['finalOrder'],
        'rows': [{key: row[key] for key in ('fields', 'tokenIds', 'rawSampledTokenIds',
            'stopTokenId', 'arSteps', 'finishReason', 'text')} for row in receipt['rawRows']]}


def run(out, *, native=False, width=10, _dependencies=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Unused output required; preserve prior experiments')
    if type(width) is not int or width not in (10, 20):
        raise ValueError('Reviewed comparison widths10 or20 required')
    deps = {} if _dependencies is None else _dependencies
    if native and deps.get('service_listening', demo_listening)():
        raise RuntimeError('Serving8766 still listening; coordinate before exclusive experiment')
    source_sha = deps.get('source_sha', sha)
    preflight = deepcopy(deps.get('preflight', cpu_preflight)(DEFAULT_MANIFEST))
    preflight.pop('approvalPolicy', None)  # Historical AR preparation item is resolved.
    preflight.update(version='store-demo-performance.v1',performanceVersion='store-demo-performance.v1', selectedWidth=width,
        pairedOrder=[list(pair) for pair in ORDER],
        armNames={'baseline':'ordinary resident hybrid decode', 'profile-v2':'same decode with synchronized phase observer'},
        timerPolicy=TIMER_POLICY,cachePolicy=CACHE_POLICY,
        profileSourceSha256=source_sha(ROOT / 'store_demo_decode_profile_v2.py'),
        profileSupportSourceSha256=source_sha(ROOT / 'store_demo_decode_profile_v1.py'),
        runnerSourceSha256=source_sha(Path(__file__)),
        scope='same full query-to-serialized20field path, baseline versus synchronized phase observer; no browser',
        executionPolicy='CPU preparation now; explicit native run only in the coordinated later performance window; never bypass an active service or owner lock')
    for case,order in zip(preflight.get('cases',[]),ORDER,strict=bool(preflight.get('cases'))):
        case['armOrder']=list(order)
        for key in ('conventionalPromptTokens','conventionalPromptSha256'):
            case.pop(key,None)
    out.mkdir(parents=True)
    (out/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
    if not native:
        return {'status':'cpu-preflight-only','nativeExecution':False,'preflight':preflight}
    if _dependencies is None:
        from .catalog_fusion.generalization_v2 import native_owner_lock
        from .store_demo_engine import ResidentDemoEngine
        from .store_demo_decode_profile_v2 import decode_rows_profiled
        lock=lambda: native_owner_lock(Path('data/ranker/pretrained/native-owner.lock'),'root')
        factory=lambda: ResidentDemoEngine(output=out/'engine',width=width,integration_manifest=DEFAULT_MANIFEST)
        profile=decode_rows_profiled
    else:
        lock=deps['lock']; factory=deps['factory']; profile=deps['profile']
    clock=deps.get('clock',time.perf_counter)
    reset=deps.get('reset',reset_runtime)
    report={'version':'store-demo-performance.v1','nativeExecution':True,'status':'started',
        'width':width,'preflightSha256':sha(out/'preflight.json'),'records':[], 'parity':[],
        'timerPolicy':TIMER_POLICY,'cachePolicy':CACHE_POLICY,
        'generatedTextSubstituted':False,'liveServingEdited':False}
    engine=None; ownership=ExitStack()
    def save():
        (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def one(case,mode,warmup=False):
        reset(engine)
        request_id=str(uuid.uuid4()); events=[]
        path=Path(engine.output)/'requests'/request_id/'receipt.json'
        row={'caseId':case['caseId'],'mode':mode,'warmupExcluded':warmup,'requestId':request_id,
            'resultPath':str(path),'status':'running'}
        report['records'].append(row); save()
        # Receipt persistence stays before the caller timer; no report write
        # separates the timer from request preparation and native execution.
        started=clock()
        try:
            with ExitStack() as stack:
                if mode=='profile-v2':
                    stack.enter_context(patch('experiments.ranker.pretrained.store_demo_engine.decode_rows',profile))
                with deadline(120,'complete timed performance request'):
                    engine.personalize(case['request'],request_id,started,events.append,lambda:False)
            value=json.loads(path.read_text()); bytype={e['type']:e for e in value['events']}
            timings={key:bytype[kind]['timing']['readyMs'] for key,kind in (
                ('rankMs','rank'),('firstThreeAttemptMs','firstThree'),('fullAttemptMs','complete'))}
            if (any(type(t) not in (float,int) or not math.isfinite(t) or t<0 for t in timings.values())
                    or not timings['rankMs']<=timings['firstThreeAttemptMs']<=timings['fullAttemptMs']):
                raise ValueError('Native serialized readiness timings must be finite, nonnegative and ordered')
            row.update(status=value['status'],cardCount=len(value['cards']),
                **timings,
                acceptedFields=bytype['complete']['acceptedFields'],requiredFields=bytype['complete']['requiredFields'],
                preparation=value['preparation'],decodeTiming=value['decodeTiming'],memory=value['memory'])
            rows=value['rawRows']
            if (value['status']!='completed' or row['cardCount']!=6 or row['requiredFields']!=20
                    or len(rows)!=20 or tuple(r['fields'] for r in rows)!=tuple([field] for field in FIELDS)
                    or any(r['maxTokens']!=CAPS[r['fields'][0][0]] for r in rows)
                    or sum(r['maxTokens'] for r in rows)!=416):
                raise ValueError('Must retain allsixcards and20freshfield attempts with unchanged416token cap')
            row['signatureSha256']=hashlib.sha256(json.dumps(token_signature(value),sort_keys=True).encode()).hexdigest()
            return value
        except BaseException as error:
            row.update(status='failed-attempt',error={'type':type(error).__name__,'message':str(error)},
                failedAtMs=(clock()-started)*1000)
            raise
        finally:
            row['outerCallWallIncludingArchiveMs']=(clock()-started)*1000
            save()
    started=clock()
    try:
        ownership.enter_context(lock())
        if deps.get('service_listening', demo_listening)():
            raise RuntimeError('Serving8766 became active before native load; refuse without stopping it')
        engine=factory(); report['startup']=engine.startup
        one(fixed_cases()[0],'baseline',True)
        for case,order in zip(fixed_cases(),ORDER,strict=True):
            receipts={mode:one(case,mode) for mode in order}
            parity=token_signature(receipts['baseline'])==token_signature(receipts['profile-v2'])
            report['parity'].append({'caseId':case['caseId'],'fullTokensBindingsAndEOSExact':parity})
            save()
            if not parity:
                raise AssertionError('Instrumented observer changed actual generation; preserve and investigate')
        report['status']='completed-profile-comparison'
    except BaseException as error:
        report.update(status='failed-profile-comparison',error={'type':type(error).__name__,'message':str(error)})
    finally:
        if engine is not None:
            try:
                report['shutdownIntegrity']=engine.close()
                if not all(report['shutdownIntegrity'].get(k) is True for k in ('frozen927HashesUnchanged','copyCheckpointUnchanged')):
                    raise AssertionError('Shutdown frozen/checkpoint integrity failed')
            except BaseException as error:
                report.update(status='failed-profile-comparison',shutdownError={'type':type(error).__name__,'message':str(error)})
        try:
            if (source_sha(ROOT / 'store_demo_decode_profile_v2.py')!=preflight['profileSourceSha256']
                    or source_sha(ROOT / 'store_demo_decode_profile_v1.py')!=preflight['profileSupportSourceSha256']
                    or source_sha(Path(__file__))!=preflight['runnerSourceSha256']):
                raise ValueError('Performance driver or observer source changed after preflight')
            report['runtimeSourceHashesUnchanged']=True
        except BaseException as error:
            report.update(status='failed-profile-comparison',sourceError={'type':type(error).__name__,'message':str(error)})
        try:
            ownership.close()
        except BaseException as error:
            report.update(status='failed-profile-comparison',ownershipError={'type':type(error).__name__,'message':str(error)})
        report['wholeRunWallMsIncludingSetupWarmup']=(clock()-started)*1000
        summary={}
        for mode in ('baseline','profile-v2'):
            rows=[r for r in report['records'] if r['mode']==mode and not r['warmupExcluded']]
            completed=[r for r in rows if r['status']=='completed']
            summary[mode]={'fixedCases':3,'attempts':len(rows),'completedAttempts':len(completed),
                'failedAttempts':len(rows)-len(completed),'unattempted':3-len(rows),
                **{key:statistics.median([r[key] for r in completed]) if completed else None
                for key in ('rankMs','firstThreeAttemptMs','fullAttemptMs')}}
        report['summary']=summary; save()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True); parser.add_argument('--run',action='store_true')
    parser.add_argument('--width',type=int,choices=(10,20),default=10)
    args=parser.parse_args(); report=run(args.out,native=args.run,width=args.width)
    print(json.dumps({key:report[key] for key in ('status','nativeExecution')}))
    return 1 if report['status'].startswith('failed') else 0

if __name__=='__main__':
    raise SystemExit(main())
