import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {buildPlan,preflight,execute,main,verifyFactorizedEvidence,validateFactorizedSelection,VERSION,MODULE,HELPER} from './dev-personalized-factorized.mjs';
const hash=value=>createHash('sha256').update(value).digest('hex');
const digest='a'.repeat(64);
function evidence(){return {mode:'technical-development-demo',developmentDemo:true,qualityGatePassed:false,modelPromotion:false,
 acceptedProfile:false,overallPromotionPass:false,factorizedServingEvidenceVersion:'persona-factorized-development-admission.v1',conditioningPolicy:'factorized-public-persona-history.v1',technicalSmokePassed:true,actualShutdownVerified:true,v4ConditioningQualityValidated:false,expected:{startup:{persona:{measurementEngine:'persona-adapter-engine.v4-factorized-profile',
 forceFreshQuery:true,wholePromptRankCacheIncludesQuery:false,noResponseWarmupOrTextCache:true,featurePreencoding:{
 version:'persona-adapter-runtime.v4-factorized-profile',enabled:true,rows:Array.from({length:10},(_,i)=>({publicPersonId:String(i)})),
 rankingWarmResponses:0,copyWarmResponses:0,responseWordsCached:false}}}}};}
function base(){return {version:'personalized-development-demo-launch.v1',root:'/main',assetRoot:'/main',
 scope:'failed-quality development',qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,
 developmentEvidence:evidence(),nativeArgs:['-m','experiments.ranker.pretrained.store_persona_adapter_engine_v2',
 '--asset-root','/main','--serving-manifest','/main/data/serving.json'],
 currentSourceSha256:{'/main/runtime.py':hash('same')},launcherSourceSha256:{'/main/old-launcher.mjs':hash('same')}};}
async function plan(changes={}){return buildPlan({}, {baseBuild:async()=>({...base(),...changes}),read:async()=>Buffer.from('same')});}

test('old V2-only evidence and promotion aliases cannot authorize factorized launcher',()=>{
 assert.equal(validateFactorizedSelection(evidence()).modelPromotion,false);
 for(const mutate of [v=>v.qualityGatePassed=true,v=>v.modelPromotion=true,v=>v.acceptedProfile=true,
  v=>v.expected.startup.persona.forceFreshQuery=false,v=>v.technicalSmokePassed=false,
  v=>v.actualShutdownVerified=false,v=>delete v.expected.startup.persona.measurementEngine]){
  const value=evidence();mutate(value);assert.throws(()=>validateFactorizedSelection(value));
 }
});

test('reviewed Python factual/shutdown selector uses exact receipt/source cwd and offline lexical Python',()=>{
 let actual;
 const selected=verifyFactorizedEvidence('/main/.tools/pretrained-venv/bin/python','/reviewed','/receipt.json',digest,(command,args,options)=>{
  actual={command,args,options};return {status:0,stdout:JSON.stringify(evidence())};
 });
 assert.equal(selected.modelPromotion,false);assert.equal(actual.command,'/main/.tools/pretrained-venv/bin/python');
 assert.ok(actual.args[1].includes(HELPER));assert.deepEqual(actual.args.slice(-2),['/receipt.json',digest]);
 assert.equal(actual.options.cwd,'/reviewed');assert.equal(actual.options.env.HF_HUB_OFFLINE,'1');
 assert.equal(actual.options.env.OPENAI_API_KEY,undefined);assert.equal(actual.options.shell,undefined);
 assert.throws(()=>verifyFactorizedEvidence('python','/r','/r.json',digest,()=>({status:2,stderr:'warm source pin absent'})),/factorized evidence verification/);
});

test('wrapper preserves main assets and argument bytes, selects only V4 and fixed fresh policy',async()=>{
 const original=base();const snapshot=structuredClone(original);let selectedVerifier;
 const result=await buildPlan({}, {baseBuild:async(options,deps)=>{selectedVerifier=deps.evidence;return original;},read:async()=>Buffer.from('same')});
 assert.equal(selectedVerifier,verifyFactorizedEvidence);assert.equal(result.version,VERSION);
 assert.equal(result.nativeArgs[1],MODULE);assert.deepEqual(result.nativeArgs.slice(2),original.nativeArgs.slice(2));
 assert.ok(!result.nativeArgs.includes('--force-fresh-query'));assert.equal(result.measurementPolicy.forceFreshQuery,true);
 assert.equal(result.qualityGatePassed,false);assert.equal(result.modelPromotion,false);assert.equal(result.acceptedProfile,false);
 assert.equal(result.measurementPolicy.responseWordsCached,false);assert.equal(result.measurementPolicy.unknownPublicProfileMayMiss,true);
 assert.equal(result.launcherSourceSha256['/main/scripts/dev-personalized-factorized.mjs'],hash('same'));
 assert.deepEqual(original,snapshot);
});

test('original argument layout must be V2 exact adapter path before explicit replacement',async()=>{
 await assert.rejects(plan({nativeArgs:['-m','arbitrary.module']}),/original adapter argument layout/);
 await assert.rejects(plan({modelPromotion:true}),/promotion flags/);
 const bad=evidence();delete bad.expected.startup.persona.featurePreencoding;
 await assert.rejects(plan({developmentEvidence:bad}),/actual V4 factual smoke/);
});

test('busy native ports stop before build, evidence subprocess or output creation',async()=>{
 let builds=0;
 await assert.rejects(preflight({}, {ports:async()=>{throw Error('8767 busy');},build:async()=>{builds++;return base();}}),/8767 busy/);
 assert.equal(builds,0);
});

test('original preflight lock and verified checkout behavior are reused',async()=>{
 const calls=[];const fake={...base(),python:'python',lock:'main-lock'};
 const value=await preflight({}, {ports:async()=>calls.push('ports'),build:async()=>fake,
  lock:(python,filename)=>calls.push(filename),checkout:async()=>({reused:true,origin:'http://127.0.0.1:4331'})});
 assert.deepEqual(calls,['ports','main-lock']);assert.equal(value.checkout.reused,true);
});

test('runner receives exact factorized plan and source checks reject changes before spawn',async()=>{
 const value=await plan();let runs=0;const originalLog=console.log;console.log=()=>{};
 try{
  await execute(value,{read:async()=>Buffer.from('same'),run:async selected=>{runs++;assert.equal(selected,value);}});
  assert.equal(runs,1);
  await assert.rejects(execute(value,{read:async()=>Buffer.from('changed'),run:async()=>runs++}),/source changed/);
  await assert.rejects(execute({...value,nativeArgs:base().nativeArgs},{read:async()=>Buffer.from('same'),run:async()=>runs++}),/fixed V4/);
  await assert.rejects(execute({...value,modelPromotion:true},{run:async()=>runs++}),/non-promoted/);
  assert.equal(runs,1);
 }finally{console.log=originalLog;}
});

test('no-run remains preflight-only and controls cannot be disabled with CLI overrides',async()=>{
 let runs=0;const originalLog=console.log;console.log=()=>{};
 try{
  await main(['--receipt','/r','--receipt-sha256',digest],{preflight:async()=>base(),execute:async()=>runs++});
  assert.equal(runs,0);
  await main(['--run'],{preflight:async()=>base(),execute:async()=>runs++});assert.equal(runs,1);
  for(const flag of ['--no-force-fresh-query','--engine','--accepted-profile'])
   await assert.rejects(main([flag,'anything']),/Unknown factorized/);
 }finally{console.log=originalLog;}
});
