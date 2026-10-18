"""Exact-text base-query reuse plus only missing fresh forwards; CPU default.

No rank/copy adapter, labels, profile vector substitution or approximate cache
key. Prior tensors are reused only after the original927/no-adapter evidence and
the complete text/token/model/encoder contract match byte for byte.
"""
import argparse
from pathlib import Path
import json
import torch
from . import store_persona_combined_query_v1 as original
from . import store_persona_combined_predictions_v1 as old_writer

VERSION='persona-combined-public-query-reuse.v2'
RECIPE={**original.RECIPE,'reuse':'exact whole runtime query text/token IDs + frozen model/source contract only',
        'nativeSelection':'only missing exact query keys; prior292 never relabeled or reencoded unnecessarily'}
ROOT=original.ROOT;MAIN=original.MAIN
sha=original.sha;dump=original.dump;pincheck=original.pincheck;vector=original.vector
canonical=original.canonical;digest_text=original.digest_text;project_rows=original.project_rows
def read(p):return json.loads(Path(p).read_text())

def select_reuse(requested,prior):
    by={q['textSha256']:q for q in prior};reused=[];fresh=[]
    for row in requested:
        if row['textSha256'] in by:
            if row!=by[row['textSha256']]:raise ValueError('Same query SHA has different complete text/token contract')
            reused.append(row['textSha256'])
        else:fresh.append(row['textSha256'])
    return reused,fresh

def prepare(full_plan,full_sha,prior_run,prior_plan_sha,out):
    out=Path(out)
    if out.exists() or not out.is_absolute() or not out.is_relative_to(ROOT):raise ValueError('Unique absolute worktree preparation required')
    prior_run=Path(prior_run).resolve();prior,values,proofs=old_writer.query_bundle(prior_run,prior_plan_sha)
    out.mkdir(parents=True)
    base=original.prepare(full_plan,full_sha,out/'public-query-preflight')
    # The original producer validates every prior tensor and complete927 proof.
    if base['modelFiles']!=prior['modelFiles']:raise ValueError('Original frozen model receipt differs; reuse refused')
    for source in original.SOURCES:
        key=str(source.resolve())
        if base['sha256'].get(key)!=prior['sha256'].get(key):raise ValueError('Actual query encoder/formatter/guard source differs; reuse refused')
    reused,fresh=select_reuse(base['uniqueQueries'],prior['uniqueQueries'])
    if any(k not in values for k in reused):raise ValueError('Prior tensor missing despite text contract')
    pins={**base['sha256'],**proofs,**{str(p.resolve()):sha(p) for p in [Path(__file__),Path(old_writer.__file__),Path(__file__).with_name('store_persona_choice_clear_combined_predictions_v1.py'),Path(__file__).with_name('test_store_persona_combined_reuse_v2.py')]}}
    fresh_plan={**base,'uniqueQueries':[q for q in base['uniqueQueries'] if q['textSha256'] in set(fresh)],'sha256':pins,
        'counts':{**base['counts'],'uniqueBaseQueryForwardsRequired':len(fresh),'priorExactReusedQueries':len(reused)},
        'scope':'fresh missing-query subset only; full combined roster remains in parent reuse plan'}
    dump(out/'fresh-native-plan.json',fresh_plan)
    packet={**base,'version':VERSION,'recipe':RECIPE,'sha256':pins,'priorRun':str(prior_run),'priorPlanSha256':prior_plan_sha,
        'reusedKeys':reused,'freshKeys':fresh,'priorSourceVectors':len(prior['uniqueQueries']),
        'freshNativePlan':{'file':str((out/'fresh-native-plan.json').resolve()),'sha256':sha(out/'fresh-native-plan.json')},
        'counts':{**base['counts'],'uniqueBaseQueryForwardsRequired':len(fresh),'uniqueQueryVectors':len(base['uniqueQueries']),
                  'exactPriorVectorsReused':len(reused),'oldVectorsNotUsed':len(prior['uniqueQueries'])-len(reused)}}
    pincheck(pins);dump(out/'plan.json',packet);dump(out/'PLAN_FROZEN.json',{'version':VERSION,'planSha256':sha(out/'plan.json'),'counts':packet['counts'],'nativeExecuted':False,'goldParsed':False})
    return packet

def assemble_values(packet,prior_plan,prior_values,fresh_plan=None,fresh_values=None):
    reused,fresh=select_reuse(packet['uniqueQueries'],prior_plan['uniqueQueries'])
    if reused!=packet['reusedKeys'] or fresh!=packet['freshKeys']:raise ValueError('Exact reuse partition changed')
    fresh_values=fresh_values or {}
    if set(fresh_values)!=set(fresh):raise ValueError('Every missing query needs actual fresh native evidence, no substitutions')
    if fresh:
        by={q['textSha256']:q for q in packet['uniqueQueries']}
        if fresh_plan is None or fresh_plan['uniqueQueries']!=[by[k] for k in fresh]:raise ValueError('Fresh native query token contract differs')
    result={}
    for key in reused:result[key]=vector(prior_values[key]).clone()
    for key in fresh:result[key]=vector(fresh_values[key]).clone()
    if set(result)!={q['textSha256'] for q in packet['uniqueQueries']}:raise ValueError('Incomplete exact query bundle')
    return result

def bundle(run,expected_sha):
    run=Path(run).resolve();packet=read(run/'plan.json');report=read(run/'report.json')
    if (sha(run/'plan.json')!=expected_sha or packet['version']!=VERSION or packet['recipe']!=RECIPE or
        report['version']!=VERSION or report['sourceManifestSha256']!=expected_sha or report['status']!='completed-query-features' or
        report['goldParsed'] is not False or report['adapterInstalled'] is not False or report['optimizerUpdates']!=0 or
        report['freshNativeQueryForwards']!=len(packet['freshKeys']) or report['exactPriorVectorsReused']!=len(packet['reusedKeys']) or
        report['savedVectors']!=len(packet['uniqueQueries']) or report['sourcePinsUnchanged'] is not True or sha(run/'query-features.pt')!=report['queryFeaturesSha256']):raise ValueError('Completed exact reuse/fresh evidence required')
    pincheck(packet['sha256']);prior,pvalues,proofs=old_writer.query_bundle(packet['priorRun'],packet['priorPlanSha256'])
    fplan=fvalues=None
    if packet['freshKeys']:
        fplan,fvalues,fpins=old_writer.query_bundle(run/'fresh-native',packet['freshNativePlan']['sha256']);proofs.update(fpins)
    expected=assemble_values(packet,prior,pvalues,fplan,fvalues)
    actual=torch.load(run/'query-features.pt',map_location='cpu',weights_only=True)
    if set(actual)!={'version','sourceManifestSha256','features'} or actual['version']!=VERSION or actual['sourceManifestSha256']!=expected_sha or set(actual['features'])!=set(expected):raise ValueError('Exact target-free merged query bundle required')
    for key in expected:
        if not torch.equal(vector(actual['features'][key]),expected[key]):raise ValueError('Merged tensor differs from original/fresh native source')
    proofs.update({str(run/name):sha(run/name) for name in ('plan.json','report.json','query-features.pt')})
    return packet,actual['features'],proofs

def extract(plan_path,expected_sha,out,*,execute=False,owner=None):
    plan_path=Path(plan_path).resolve();packet=read(plan_path)
    if sha(plan_path)!=expected_sha or packet['version']!=VERSION or packet['recipe']!=RECIPE:raise ValueError('Exact frozen reuse plan required')
    pincheck(packet['sha256'])
    if sha(packet['freshNativePlan']['file'])!=packet['freshNativePlan']['sha256']:raise ValueError('Frozen missing-query plan changed')
    if not execute:return {'status':'cpu-preflight-only','nativeExecuted':False,'counts':packet['counts']}
    if owner!='root':raise ValueError('Explicit root execution required even for all-reuse assembly')
    out=Path(out)
    if out.exists() or not out.is_absolute() or not out.is_relative_to(ROOT):raise ValueError('Unique absolute worktree output required')
    out.mkdir(parents=True);dump(out/'plan.json',packet)
    report={'version':VERSION,'status':'starting','sourceManifestSha256':expected_sha,'goldParsed':False,'adapterInstalled':False,'optimizerUpdates':0,
        'freshNativeQueryForwards':0,'exactPriorVectorsReused':0,'savedVectors':0,'newModelLoaded':False,
        'new927HashOrShutdownClaim':False,'prior927ProofRevalidated':False};save=lambda:dump(out/'report.json',report);save()
    try:
        prior,pvalues,proofs=old_writer.query_bundle(packet['priorRun'],packet['priorPlanSha256']);report['prior927ProofRevalidated']=True;save()
        fplan=fvalues=None
        if packet['freshKeys']:
            report['freshNativeJobAttempted']=True;save()
            original.extract(packet['freshNativePlan']['file'],packet['freshNativePlan']['sha256'],out/'fresh-native',execute=True,owner='root')
            fplan,fvalues,fpins=old_writer.query_bundle(out/'fresh-native',packet['freshNativePlan']['sha256'])
            report['newModelLoaded']=True;report['freshNativeQueryForwards']=len(fvalues);report['new927HashOrShutdownClaim']=True
        values=assemble_values(packet,prior,pvalues,fplan,fvalues)
        torch.save({'version':VERSION,'sourceManifestSha256':expected_sha,'features':values},out/'query-features.pt')
        reloaded=torch.load(out/'query-features.pt',map_location='cpu',weights_only=True)
        if set(reloaded['features'])!=set(values) or any(not torch.equal(reloaded['features'][k],v) for k,v in values.items()):raise ValueError('Exact merged feature reload failed')
        pincheck(packet['sha256']);report.update(status='completed-query-features',exactPriorVectorsReused=len(packet['reusedKeys']),savedVectors=len(values),
            queryFeaturesSha256=sha(out/'query-features.pt'),sourcePinsUnchanged=True,reloadExact=True)
        return report
    except BaseException as error:report.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    finally:
        native=out/'fresh-native/report.json'
        if native.exists():
            partial=read(native);report['freshNativeQueryForwards']=partial['physicalNativeForwards']
            report['freshNativeReportSha256']=sha(native);report['newModelLoaded']=len(partial.get('beforeParameterSha256',{}))==927
        save()

def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--owner',choices=['root']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prior-run',type=Path);p.add_argument('--prior-plan-sha256');a=p.parse_args(argv)
    if a.prepare:
        if a.run or not a.prior_run or not a.prior_plan_sha256:p.error('Separate prepare requires exact completed prior query run')
        result=prepare(a.plan,a.plan_sha256,a.prior_run,a.prior_plan_sha256,a.out);print(json.dumps({'status':'prepared','planSha256':sha(a.out/'plan.json'),'counts':result['counts']}))
    else:print(json.dumps(extract(a.plan,a.plan_sha256,a.out,execute=a.run,owner=a.owner)))
if __name__=='__main__':main()
