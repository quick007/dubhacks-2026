// Fixed fast development entry. Historical/accepted launchers remain unchanged.
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {buildPlan as factorizedBuild,preflight as factorizedPreflight,validateFactorizedSelection,MODULE as FACTORIZED_MODULE} from './dev-personalized-factorized.mjs';
import {run as runChildren} from './dev-personalized.mjs';
export const VERSION='personalized-practical-development-launch.v1';
export const MODULE='experiments.ranker.pretrained.store_persona_practical_engine_v1';
export const MAIN_ASSET_ROOT='/Users/lukas/Documents/Projects/dubhacks-2026';
export const SOURCE_PINS={
  "experiments/ranker/pretrained/store_persona_practical_engine_v1.py": "5c9d7e5e37440a0828a63805089df75194c7f57f693e1a38865ff83c6d65498a",
  "experiments/ranker/pretrained/store_persona_query_projection_runtime_v1.py": "cbb052fff219ecf0e8ab54f0ed4fb4575cd1d99c19bda5f2066e366257724f15",
  "experiments/ranker/pretrained/store_persona_query_projection_engine_v1.py": "fcbba38e1c82428bd6b98d38db684e5c3ef161c505670bd5db81d1cd6288d954",
  "experiments/ranker/pretrained/store_persona_route_projection_v2.py": "d1d45fffe658860ba63b0f492e2ce836856e62956bd662cc0d224706ae293316",
  "experiments/ranker/pretrained/store_persona_query_prefix_runtime_v1.py": "9f67d2e9e6fe6590154897ed8b41cfa4a54d8322f52de22af541b993d735ac66",
  "experiments/ranker/pretrained/store_persona_query_prefix_engine_v1.py": "ef23e048d7539b4067acc426c4e0273c1073d75d06fe9f03b9a79d8990269dcd",
  "experiments/ranker/pretrained/store_query_prefix_reuse_v1.py": "ff0d370779000a28d534b1fc0870d86ff69512d28e4758602363428b5233d68f",
  "experiments/ranker/pretrained/store_persona_portless_http_v1.py": "cf7d8025af0ec9ba302451fd7d7771ecab2e39ca8a6f993b8d19d783d94fde78"
};
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const need=(ok,message)=>{if(!ok)throw Error('Fast development demo: '+message);};

export async function buildPlan(options={},deps={}) {
  need(path.resolve(options.assetRoot??MAIN_ASSET_ROOT)===MAIN_ASSET_ROOT,'MAIN asset root required; worktree runtime fallback forbidden');
  const base=await (deps.baseBuild??factorizedBuild)({...options,assetRoot:MAIN_ASSET_ROOT},deps);
  validateFactorizedSelection(base.developmentEvidence);
  need(base.assetRoot===MAIN_ASSET_ROOT&&base.qualityGatePassed===false&&base.modelPromotion===false&&base.acceptedProfile===false,'MAIN assets and explicit non-promoted development evidence required');
  need(base.nativeArgs?.[0]==='-m'&&base.nativeArgs[1]===FACTORIZED_MODULE,'exact factorized argument layout required');
  const nativeArgs=[...base.nativeArgs];nativeArgs[1]=MODULE;
  const currentSourceSha256={...base.currentSourceSha256};
  for(const [relative,expected] of Object.entries(SOURCE_PINS)) {
    const filename=path.join(base.root,relative);
    need(sha(await (deps.read??readFile)(filename))===expected,'reviewed practical source changed: '+relative);
    currentSourceSha256[filename]=expected;
  }
  const filename=path.join(base.root,'scripts/dev-personalized-fast.mjs');
  return {...base,version:VERSION,nativeArgs,currentSourceSha256,
    launcherSourceSha256:{...base.launcherSourceSha256,[filename]:sha(await (deps.read??readFile)(filename))},
    measurementPolicy:{...base.measurementPolicy,forceFreshQuery:false,queryEncoding:'ordinary',routeProjection:'selected',queryFeatureCache:'lru32',queryFeatureCacheCapacity:32,
      defaultPublicQueryKeysPreencoded:8,measurementControlsExposed:false,prefixReuseEnabled:false,compactQueryEnabled:false,
      responseWordsCached:false,scoresRankingsOrAnswersCached:false,freshFusionVariantSelectionAndCopy:true,
      performanceScope:'Exact query-feature hits report native0; novel misses report freshB1. Prior paired development measurements do not substitute for this lifetime browser/shutdown evidence.'},
    scope:base.scope+'; fixed ordinary query + selected16 projection + exact LRU32 query features; historical fresh-query policy superseded explicitly; quality remains unvalidated/non-promoted'};
}
export async function preflight(options,deps={}) {
  return factorizedPreflight(options,{...deps,build:deps.build??buildPlan});
}
export async function execute(plan,{run=runChildren,read=readFile}={}) {
  need(plan.version===VERSION&&plan.qualityGatePassed===false&&plan.modelPromotion===false&&plan.acceptedProfile===false,'explicit non-promoted practical plan required');
  validateFactorizedSelection(plan.developmentEvidence);
  need(plan.assetRoot===MAIN_ASSET_ROOT&&plan.nativeArgs?.[0]==='-m'&&plan.nativeArgs[1]===MODULE,'fixed practical module and MAIN assets required');
  for(const [filename,checksum] of Object.entries({...plan.currentSourceSha256,...plan.launcherSourceSha256}))
    need(sha(await read(filename))===checksum,'source changed before owned-process spawn: '+filename);
  console.log('Fast development demo: ordinary query, selected projection, exact query-feature LRU32; fresh scoring/variants/copy. Quality gate remains FAILED; V4 conditioning quality UNVALIDATED; model promotion false.');
  await run(plan);
}
export async function main(argv=process.argv.slice(2),deps={}) {
  const options={};let shouldRun=false;
  const names={'--receipt':'receipt','--receipt-sha256':'receiptSha256','--asset-root':'assetRoot','--session':'session'};
  for(let i=0;i<argv.length;i++) {
    if(argv[i]==='--run')shouldRun=true;
    else if(names[argv[i]]&&argv[i+1])options[names[argv[i]]]=argv[++i];
    else throw Error('Unknown fast development argument: '+argv[i]);
  }
  const plan=await (deps.preflight??preflight)(options,deps);
  if(!shouldRun){console.log(JSON.stringify({...plan,execution:'CPU preflight only; no services/session created'},null,2));return plan;}
  process.umask(0o077);await (deps.execute??execute)(plan);return plan;
}
if(process.argv[1]&&pathToFileURL(path.resolve(process.argv[1])).href===import.meta.url)
  main().catch(error=>{console.error(error.message);process.exitCode=1;});
