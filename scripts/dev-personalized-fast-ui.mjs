// Explicit reviewed UI-binding replacement only; fast model policy unchanged.
import {createHash} from 'node:crypto';
import {readFile,lstat,realpath} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {buildPlan as fastBuild,preflight as fastPreflight,execute as fastExecute,VERSION as FAST_VERSION,MODULE,MAIN_ASSET_ROOT} from './dev-personalized-fast.mjs';
import {safeEnvironment,run as runChildren} from './dev-personalized.mjs';
export const VERSION='personalized-practical-ui-development-launch.v2';
const sha=b=>createHash('sha256').update(b).digest('hex');
const exact=s=>typeof s==='string'&&/^[a-f0-9]{64}$/.test(s);
const need=(ok,msg)=>{if(!ok)throw Error('Fast UI demo: '+msg);};

export async function normalMainFile(filename,deps={}) {
  need(typeof filename==='string'&&path.isAbsolute(filename),'absolute MAIN manifest file required');
  const relative=path.relative(MAIN_ASSET_ROOT,filename);
  need(relative&&!path.isAbsolute(relative)&&relative!=='..'&&!relative.startsWith('../'),'manifest must be inside MAIN');
  need(path.resolve(filename)===filename,'normalized explicit manifest path required');
  let current=MAIN_ASSET_ROOT;
  for(const piece of relative.split(path.sep)) {
    current=path.join(current,piece);const info=await (deps.lstat??lstat)(current);
    need(!info.isSymbolicLink(),'symlink manifest path forbidden: '+current);
    if(current===filename)need(info.isFile(),'manifest must be a normal file');
    else need(info.isDirectory(),'manifest parent must be directory');
  }
  need(await (deps.realpath??realpath)(filename)===filename,'manifest resolved outside explicit MAIN path');
  return filename;
}

export function verifyBinding(python,root,filename,checksum,execute=spawnSync) {
  const code=`import json,sys; from pathlib import Path; from experiments.ranker.pretrained.store_demo_image_binding_v1 import ImageDeliveryCatalogBinding,ROOT; from experiments.ranker.pretrained.store_catalog_v1 import load_snapshot; assert ROOT.resolve()==Path(sys.argv[3]).resolve(); b=ImageDeliveryCatalogBinding(sys.argv[1]); assert b.manifest_sha==sys.argv[2]; m=b.manifest; pins={r['path']:r['sha256'] for r in load_snapshot()['receipt']['sources']}; [pins.update(m.get(k,{})) for k in ('rendererOnlyExceptions','auxiliarySourceExceptions','frontendSha256','verificationSourcesSha256')]; print(json.dumps({'version':'reviewed-fast-ui-binding.v1','manifestSha256':b.manifest_sha,'root':str(ROOT.resolve()),'publicCatalogAndModelFactsUnchanged':True,'sourceSha256':{str(ROOT/p):s for p,s in pins.items()}}))`;
  const result=execute(python,['-c',code,filename,checksum,MAIN_ASSET_ROOT],{cwd:root,env:safeEnvironment(),encoding:'utf8',timeout:60_000,maxBuffer:8*1024*1024,stdio:['ignore','pipe','pipe']});
  need(!result.error&&result.status===0,'actual MAIN public-catalog/image binding failed: '+(result.error?.message??result.stderr??'unknown error'));
  const value=JSON.parse(result.stdout);
  need(value.version==='reviewed-fast-ui-binding.v1'&&value.root===MAIN_ASSET_ROOT&&value.manifestSha256===checksum&&value.publicCatalogAndModelFactsUnchanged===true,'verified MAIN binding proof required');
  need(Object.keys(value.sourceSha256??{}).length>0,'full checked source inventory required');
  for(const [file,sum]of Object.entries(value.sourceSha256))need(file.startsWith(MAIN_ASSET_ROOT+'/')&&exact(sum),'MAIN checked source pins required');
  return value;
}

export async function buildPlan(options={},deps={}) {
  need(options.integrationManifest&&exact(options.integrationManifestSha256),'explicit --integration-manifest and --integration-manifest-sha256 required');
  // Old admission/old manifest checks run unchanged before the explicit replacement.
  const base=await (deps.baseBuild??fastBuild)(options,deps);
  need(base.root===MAIN_ASSET_ROOT&&base.assetRoot===MAIN_ASSET_ROOT,'actual MAIN code and assets required for reviewed UI replacement');
  need(base.version===FAST_VERSION&&base.nativeArgs?.[0]==='-m'&&base.nativeArgs[1]===MODULE,'fixed fast production entry required');
  const file=await normalMainFile(options.integrationManifest,deps);const checksum=options.integrationManifestSha256;
  const bytes=await (deps.read??readFile)(file);need(sha(bytes)===checksum,'new integration manifest checksum differs');
  const binding=(deps.binding??verifyBinding)(base.python,base.root,file,checksum);
  const nativeArgs=[...base.nativeArgs];const indices=nativeArgs.flatMap((v,i)=>v==='--integration-manifest'?[i]:[]);
  need(indices.length===1&&nativeArgs[indices[0]+1]===base.profile.integrationManifest.file,'one exact prior integration manifest argument required');
  nativeArgs[indices[0]+1]=file;
  const own=path.join(base.root,'scripts/dev-personalized-fast-ui.mjs');
  return {...base,version:VERSION,nativeArgs,
    profile:{...base.profile,integrationManifest:{file,sha256:checksum}},
    assets:[...base.assets,{file,sha256:checksum,bytes:bytes.length}],
    currentSourceSha256:{...base.currentSourceSha256,...binding.sourceSha256,[file]:checksum},
    launcherSourceSha256:{...base.launcherSourceSha256,[own]:sha(await (deps.read??readFile)(own))},
    uiIntegrationReplacement:{version:'reviewed-fast-ui-binding.v1',historicalManifest:base.profile.integrationManifest,activeManifest:{file,sha256:checksum},binding,
      scope:'reviewed rendering source replacement; public catalog/model facts and model/checkpoint/ranking/guard/copy policy unchanged'},
    scope:base.scope+'; explicitly verified current MAIN UI binding replaces historical integration manifest only'};
}
export async function preflight(options,deps={}) {return fastPreflight(options,{...deps,build:deps.build??buildPlan});}
export async function execute(plan,deps={}) {
  need(plan.version===VERSION&&plan.uiIntegrationReplacement?.activeManifest?.sha256===plan.profile.integrationManifest.sha256,'explicit verified current UI plan required');
  // Re-run exact fast lifetime/source checks; only wrapper version is adapted.
  return fastExecute({...plan,version:FAST_VERSION},{...deps,run:async()=>{
    await (deps.run??runChildren)(plan); // Preserve V2 UI-plan provenance in launch.json.
  }});
}
export async function main(argv=process.argv.slice(2),deps={}) {
  const options={};let run=false;const names={'--receipt':'receipt','--receipt-sha256':'receiptSha256','--asset-root':'assetRoot','--session':'session','--integration-manifest':'integrationManifest','--integration-manifest-sha256':'integrationManifestSha256'};
  for(let i=0;i<argv.length;i++){if(argv[i]==='--run')run=true;else if(names[argv[i]]&&argv[i+1])options[names[argv[i]]]=argv[++i];else throw Error('Unknown fast UI argument: '+argv[i]);}
  const plan=await (deps.preflight??preflight)(options,deps);
  if(!run){console.log(JSON.stringify({...plan,execution:'CPU preflight only; no services/session created'},null,2));return plan;}
  process.umask(0o077);await (deps.execute??execute)(plan);return plan;
}
if(process.argv[1]&&pathToFileURL(path.resolve(process.argv[1])).href===import.meta.url)main().catch(e=>{console.error(e.message);process.exitCode=1;});
