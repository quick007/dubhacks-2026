// Explicit expanded35 development entry. Historical16 commands unchanged.
import {createHash}from'node:crypto';import{readFile,lstat,realpath}from'node:fs/promises';import{spawnSync}from'node:child_process';import path from'node:path';import{pathToFileURL}from'node:url';
import{buildPlan as fastBuild,preflight as fastPreflight,MODULE as OLD_MODULE,MAIN_ASSET_ROOT}from'./dev-personalized-fast.mjs';
import{normalMainFile}from'./dev-personalized-fast-ui.mjs';import{safeEnvironment,run as runChildren}from'./dev-personalized.mjs';
export const VERSION='personalized-expanded-development-launch.v1';export const MODULE='experiments.ranker.pretrained.store_expanded_engine_v1';
export const SOURCES=['store_expanded_engine_v1.py','store_expanded_rank_v1.py','store_expanded_binding_v1.py','store_expanded_parallel_copy_v1.py'].map(p=>'experiments/ranker/pretrained/'+p);
const sha=b=>createHash('sha256').update(b).digest('hex');const exact=s=>typeof s==='string'&&/^[a-f0-9]{64}$/.test(s);const need=(ok,msg)=>{if(!ok)throw Error('Expanded demo: '+msg);};
export function verifyExpanded(python,root,manifest,manifestSha,copyRun,copySha,execute=spawnSync){
 const code=`import json,sys; from pathlib import Path; from experiments.ranker.pretrained.store_expanded_binding_v1 import ExpandedBinding,ROOT; from experiments.ranker.pretrained.store_expanded_parallel_copy_v1 import load_expanded_parallel_assets; from experiments.ranker.pretrained.store_expanded_engine_v1 import load_catalog_vectors; from experiments.ranker.pretrained.store_qwen_lexical_choice_inputs_v2 import load_public_catalog; assert str(ROOT.resolve())==sys.argv[5]; b=ExpandedBinding(sys.argv[1]); assert b.manifest_sha==sys.argv[2]; h,f,r,m=load_expanded_parallel_assets(sys.argv[3],sys.argv[4]); c=load_catalog_vectors(m,b,[p['id'] for p in load_public_catalog()]); print(json.dumps({'version':'expanded-startup-cpu-verified.v1','catalogSha256':b.catalog_sha,'products':len(c['products']),'variants':len(c['variants']),'copyBindings':len(r),'bindingManifestSha256':b.manifest_sha,'copyManifestSha256':sys.argv[4],'sourceSha256':{str(ROOT/p):s for p,s in b.sources.items()},'assetSha256':m['assetSha256'],'modelPromotion':False,'freshNeuralCopyRequired':True}))`;
 const result=execute(python,['-c',code,manifest,manifestSha,copyRun,copySha,MAIN_ASSET_ROOT],{cwd:root,env:safeEnvironment(),encoding:'utf8',timeout:60_000,maxBuffer:8*1024*1024,stdio:['ignore','pipe','pipe']});
 need(!result.error&&result.status===0,'actual35 binding/native catalog/copy verification failed: '+(result.error?.message??result.stderr??'unknown'));
 const value=JSON.parse(result.stdout);need(value.version==='expanded-startup-cpu-verified.v1'&&value.products===35&&value.variants===105&&value.copyBindings===105&&value.bindingManifestSha256===manifestSha&&value.copyManifestSha256===copySha&&value.modelPromotion===false&&value.freshNeuralCopyRequired===true,'complete actual35/105 dynamic model proof required');return value;
}
export async function buildPlan(options={},deps={}){
 need(options.integrationManifest&&exact(options.integrationManifestSha256)&&options.expandedCopyRun&&exact(options.expandedCopyManifestSha256),'explicit new binding and completed expanded-copy run/digests required');
 const base=await(deps.baseBuild??fastBuild)(options,deps);need(base.root===MAIN_ASSET_ROOT&&base.assetRoot===MAIN_ASSET_ROOT,'actual MAIN code/assets required');
 need(base.qualityGatePassed===false&&base.modelPromotion===false&&base.acceptedProfile===false,'explicit non-promoted development evidence required');
 const manifest=await normalMainFile(options.integrationManifest,deps);need(sha(await(deps.read??readFile)(manifest))===options.integrationManifestSha256,'binding checksum differs');
 const copyRun=path.resolve(options.expandedCopyRun);const copyFile=await normalMainFile(path.join(copyRun,'serving-manifest.json'),deps);need(sha(await(deps.read??readFile)(copyFile))===options.expandedCopyManifestSha256,'expanded copy manifest checksum differs');
 const checked=(deps.verify??verifyExpanded)(base.python,base.root,manifest,options.integrationManifestSha256,copyRun,options.expandedCopyManifestSha256);
 const nativeArgs=[...base.nativeArgs];need(nativeArgs[0]==='-m'&&nativeArgs[1]===OLD_MODULE,'fixed prior practical args required');nativeArgs[1]=MODULE;
 for(const[flag,value]of[['--integration-manifest',manifest],['--candidate-run',copyRun]]){const indices=nativeArgs.flatMap((x,i)=>x===flag?[i]:[]);need(indices.length===1&&nativeArgs[indices[0]+1],'one exact '+flag+' argument required');nativeArgs[indices[0]+1]=value;}
 nativeArgs.push('--expanded-copy-manifest-sha256',options.expandedCopyManifestSha256);
 const sources={...base.currentSourceSha256,...checked.sourceSha256,[manifest]:options.integrationManifestSha256,...checked.assetSha256};
 for(const relative of SOURCES){const filename=path.join(base.root,relative);sources[filename]=sha(await(deps.read??readFile)(filename));}
 const own=path.join(base.root,'scripts/dev-personalized-expanded.mjs');
 return{...base,version:VERSION,nativeArgs,currentSourceSha256:sources,launcherSourceSha256:{...base.launcherSourceSha256,[own]:sha(await(deps.read??readFile)(own)),[path.join(base.root,'scripts/dev-personalized-fast-ui.mjs')]:sha(await(deps.read??readFile)(path.join(base.root,'scripts/dev-personalized-fast-ui.mjs')))},
  profile:{...base.profile,integrationManifest:{file:manifest,sha256:options.integrationManifestSha256},expandedCopyManifest:{file:copyFile,sha256:options.expandedCopyManifestSha256}},
  assets:[...base.assets,{file:manifest,sha256:options.integrationManifestSha256},{file:copyFile,sha256:options.expandedCopyManifestSha256}],expandedCatalog:checked,
  measurementPolicy:{...base.measurementPolicy,publicProducts:35,publicBindings:105,defaultPublicQueryKeysPreencoded:24,strictSpecificSearchEligibility:true,newSkuLoRATrainingClaim:false},scope:base.scope+'; expanded35 strict specific-search eligibility; LoRA16 families only; fresh105-binding neural copy; quality unvalidated'};
}
export async function preflight(options,deps={}){return fastPreflight(options,{...deps,build:deps.build??buildPlan});}
export async function execute(plan,{run=runChildren,read=readFile}={}){
 need(plan.version===VERSION&&plan.modelPromotion===false&&plan.qualityGatePassed===false&&plan.acceptedProfile===false&&plan.nativeArgs?.[1]===MODULE,'explicit non-promoted expanded plan required');
 for(const[file,sum]of Object.entries({...plan.currentSourceSha256,...plan.launcherSourceSha256}))need(sha(await read(file))===sum,'source/artifact changed before spawn: '+file);
 console.log('Expanded35 development demo: strict search eligibility first; learned16-family priors, genuine35 catalog vectors, learned variants and fresh neural copy. Quality unvalidated; no model promotion.');await run(plan);
}
export async function main(argv=process.argv.slice(2),deps={}){
 const options={};let run=false;const names={'--receipt':'receipt','--receipt-sha256':'receiptSha256','--asset-root':'assetRoot','--session':'session','--integration-manifest':'integrationManifest','--integration-manifest-sha256':'integrationManifestSha256','--expanded-copy-run':'expandedCopyRun','--expanded-copy-manifest-sha256':'expandedCopyManifestSha256'};
 for(let i=0;i<argv.length;i++){if(argv[i]==='--run')run=true;else if(names[argv[i]]&&argv[i+1])options[names[argv[i]]]=argv[++i];else throw Error('Unknown expanded argument: '+argv[i]);}
 const plan=await(deps.preflight??preflight)(options,deps);if(!run){console.log(JSON.stringify({...plan,execution:'CPU preflight only; no services/session created'},null,2));return plan;}process.umask(0o077);await(deps.execute??execute)(plan);return plan;
}
if(process.argv[1]&&pathToFileURL(path.resolve(process.argv[1])).href===import.meta.url)main().catch(e=>{console.error(e.message);process.exitCode=1;});
