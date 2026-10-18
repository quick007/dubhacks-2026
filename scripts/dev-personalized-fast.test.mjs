import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {buildPlan,execute,preflight,main,MODULE,VERSION,MAIN_ASSET_ROOT,SOURCE_PINS} from './dev-personalized-fast.mjs';
import {MODULE as OLD_MODULE} from './dev-personalized-factorized.mjs';
const root=path.resolve(fileURLToPath(new URL('../',import.meta.url)));
const evidence=()=>({mode:'technical-development-demo',developmentDemo:true,qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,overallPromotionPass:false,
 factorizedServingEvidenceVersion:'persona-factorized-development-admission.v1',conditioningPolicy:'factorized-public-persona-history.v1',technicalSmokePassed:true,actualShutdownVerified:true,v4ConditioningQualityValidated:false,
 expected:{startup:{persona:{measurementEngine:'persona-adapter-engine.v4-factorized-profile',forceFreshQuery:true,wholePromptRankCacheIncludesQuery:false,featurePreencoding:{version:'persona-adapter-runtime.v4-factorized-profile'}}}}});
const base=()=>({root,assetRoot:MAIN_ASSET_ROOT,scope:'development',qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,developmentEvidence:evidence(),nativeArgs:['-m',OLD_MODULE,'--asset-root',MAIN_ASSET_ROOT,'--guard','unchanged'],currentSourceSha256:{},launcherSourceSha256:{},measurementPolicy:{forceFreshQuery:true}});
const plan=()=>buildPlan({}, {baseBuild:async()=>base()});
test('exact source pins and argument bytes preserved with explicit practical policy',async()=>{
 const original=base();const snapshot=structuredClone(original);
 const result=await buildPlan({}, {baseBuild:async(options)=>{assert.equal(options.assetRoot,MAIN_ASSET_ROOT);return original;}});
 assert.equal(result.version,VERSION);assert.equal(result.nativeArgs[1],MODULE);assert.deepEqual(result.nativeArgs.slice(2),original.nativeArgs.slice(2));
 assert.deepEqual(original,snapshot);assert.equal(result.measurementPolicy.queryFeatureCache,'lru32');assert.equal(result.measurementPolicy.forceFreshQuery,false);
 assert.equal(result.measurementPolicy.defaultPublicQueryKeysPreencoded,8);assert.equal(result.measurementPolicy.measurementControlsExposed,false);
 assert.equal(result.measurementPolicy.compactQueryEnabled,false);assert.equal(result.modelPromotion,false);
 for(const [relative,checksum]of Object.entries(SOURCE_PINS))assert.equal(result.currentSourceSha256[path.join(root,relative)],checksum);
});
test('changed source, non-MAIN assets and promoted evidence fail before native',async()=>{
 let calls=0;
 await assert.rejects(buildPlan({assetRoot:'/worktree'},{baseBuild:async()=>{calls++;return base();}}),/MAIN asset/);assert.equal(calls,0);
 await assert.rejects(buildPlan({}, {baseBuild:async()=>base(),read:async()=>Buffer.from('altered')}),/source changed/);
 await assert.rejects(buildPlan({}, {baseBuild:async()=>({...base(),modelPromotion:true})}),/non-promoted/);
});
test('busy preflight fails before building and spawning',async()=>{
 let builds=0;await assert.rejects(preflight({}, {ports:async()=>{throw Error('8767 busy');},build:async()=>{builds++;return base();}}),/8767 busy/);assert.equal(builds,0);
});
test('owned runner receives fixed plan, source mutation and module replacement rejected',async()=>{
 const value=await plan();let runs=0;const log=console.log;console.log=()=>{};
 try{
  await execute(value,{run:async selected=>{assert.equal(selected,value);runs++;}});assert.equal(runs,1);
  await assert.rejects(execute(value,{read:async()=>Buffer.from('changed'),run:async()=>runs++}),/source changed/);
  await assert.rejects(execute({...value,nativeArgs:['-m',OLD_MODULE]},{run:async()=>runs++}),/fixed practical/);assert.equal(runs,1);
 }finally{console.log=log;}
});
test('preflight only default and no encoding/cache/engine override flags',async()=>{
 let runs=0;const log=console.log;console.log=()=>{};
 try{
  await main([],{preflight:async()=>base(),execute:async()=>runs++});assert.equal(runs,0);
  await main(['--run'],{preflight:async()=>base(),execute:async()=>runs++});assert.equal(runs,1);
  for(const flag of ['--engine','--compact','--cache','--accepted-profile'])await assert.rejects(main([flag,'anything']),/Unknown fast/);
 }finally{console.log=log;}
});
