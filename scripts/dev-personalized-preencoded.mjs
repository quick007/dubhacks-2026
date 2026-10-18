// Explicit non-promoted V3 measurement launcher. V2/accepted entries unchanged.
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { buildPlan as originalBuildPlan, preflight as originalPreflight } from './dev-personalized-development.mjs';
import { safeEnvironment, run as runChildren } from './dev-personalized.mjs';

export const VERSION='personalized-preencoded-development-launch.v1';
export const MODULE='experiments.ranker.pretrained.store_persona_adapter_engine_v3';
export const HELPER='experiments.ranker.pretrained.store_persona_warm_http_probe_v1';
const ORIGINAL_MODULE='experiments.ranker.pretrained.store_persona_adapter_engine_v2';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const need=(ok,message)=>{if(!ok)throw Error('Preencoded development demo: '+message);};

export function validateWarmSelection(selected) {
  const persona=selected?.expected?.startup?.persona;const pre=persona?.featurePreencoding;
  need(selected?.mode==='technical-development-demo'&&selected.developmentDemo===true&&
    selected.qualityGatePassed===false&&selected.modelPromotion===false&&selected.acceptedProfile===false&&
    selected.overallPromotionPass===false,'explicit non-promoted failed-quality development evidence required');
  need(persona?.measurementEngine==='persona-adapter-engine.v3-measurement'&&persona.forceFreshQuery===true&&
    persona.wholePromptRankCacheIncludesQuery===true&&persona.noResponseWarmupOrTextCache===true&&
    pre?.version==='persona-adapter-runtime.v3'&&pre.enabled===true&&pre.rows?.length===10&&
    pre.rankingWarmResponses===0&&pre.copyWarmResponses===0&&pre.responseWordsCached===false,
    'reviewed warm admission with ten public profile features and fresh per-request queries required');
  return selected;
}

export function verifyWarmEvidence(python,evidenceRoot,receipt,receiptSha,execute=spawnSync) {
  const code=`import json,sys; from ${HELPER} import selection; print(json.dumps(selection(sys.argv[1],sys.argv[2]),allow_nan=False))`;
  const result=execute(python,['-c',code,receipt,receiptSha],{cwd:evidenceRoot,env:safeEnvironment(),encoding:'utf8',
    timeout:30_000,maxBuffer:16*1024*1024,stdio:['ignore','pipe','pipe']});
  need(!result.error&&result.status===0,`warm evidence verification failed: ${result.error?.message??result.stderr??'unknown error'}`);
  return validateWarmSelection(JSON.parse(result.stdout));
}

export async function buildPlan(options={},deps={}) {
  // Reuse all original receipt/source-equivalence/main-assets checks. Only the
  // explicitly pinned warm selector replaces the V2 evidence selector.
  const base=await (deps.baseBuild??originalBuildPlan)(options,{...deps,evidence:deps.evidence??verifyWarmEvidence});
  validateWarmSelection(base.developmentEvidence);
  need(base.qualityGatePassed===false&&base.modelPromotion===false&&base.acceptedProfile===false,'promotion flags must remain false');
  need(base.nativeArgs?.[0]==='-m'&&base.nativeArgs[1]===ORIGINAL_MODULE,'exact original adapter argument layout required');
  const nativeArgs=[...base.nativeArgs];nativeArgs[1]=MODULE;
  nativeArgs.push('--force-fresh-query','--preencode-public-profiles','--color-cache-entries','32');
  const filename=path.join(base.root,'scripts/dev-personalized-preencoded.mjs');
  const source=sha(await (deps.read??readFile)(filename));
  return {...base,version:VERSION,nativeArgs,
    launcherSourceSha256:{...base.launcherSourceSha256,[filename]:source},
    measurementPolicy:{runtime:'v3',preencodePublicProfiles:10,forceFreshQuery:true,colorCacheEntries:32,
      responseWordsCached:false,rankingWarmResponses:0,copyWarmResponses:0,
      rankAndColorCacheIncludeActualQuery:true,newUniqueQueryMayMiss:true,
      performanceScope:'startup profile encoding recorded separately; native/browser performance requires measured receipts'},
    scope:base.scope+'; exact public Browse context features preencoded, every real query freshly encoded; unique queries may miss context/rank caches'};
}

export async function preflight(options,deps={}) {
  return originalPreflight(options,{...deps,build:deps.build??buildPlan});
}

export async function execute(plan,{run=runChildren,read=readFile}={}) {
  need(plan.version===VERSION&&plan.qualityGatePassed===false&&plan.modelPromotion===false&&plan.acceptedProfile===false,
    'explicit non-promoted preencoded plan required');
  validateWarmSelection(plan.developmentEvidence);
  need(plan.nativeArgs?.[0]==='-m'&&plan.nativeArgs[1]===MODULE&&
    plan.nativeArgs.slice(-4).join(' ')==='--force-fresh-query --preencode-public-profiles --color-cache-entries 32',
    'fixed V3 module and fresh-query/preencoding flags required');
  for(const [filename,checksum] of Object.entries({...plan.currentSourceSha256,...plan.launcherSourceSha256}))
    need(sha(await read(filename))===checksum,`source changed before owned-process spawn: ${filename}`);
  console.log('Preencoded development demo: quality gate FAILED; model promotion false. Startup profile encoding is separate; queries and copy are fresh.');
  await run(plan); // Original global lock, owned child lifetime/cleanup and logs.
}

export async function main(argv=process.argv.slice(2),deps={}) {
  const options={};let shouldRun=false;
  const names={'--receipt':'receipt','--receipt-sha256':'receiptSha256','--asset-root':'assetRoot','--session':'session'};
  for(let i=0;i<argv.length;i++) {
    if(argv[i]==='--run')shouldRun=true;
    else if(names[argv[i]]&&argv[i+1])options[names[argv[i]]]=argv[++i];
    else throw Error(`Unknown preencoded development argument: ${argv[i]}`);
  }
  const plan=await (deps.preflight??preflight)(options,deps);
  if(!shouldRun){console.log(JSON.stringify({...plan,execution:'CPU preflight only; no services/session created'},null,2));return plan;}
  process.umask(0o077);await (deps.execute??execute)(plan);return plan;
}
if(process.argv[1]&&pathToFileURL(path.resolve(process.argv[1])).href===import.meta.url)
  main().catch(error=>{console.error(error.message);process.exitCode=1;});
