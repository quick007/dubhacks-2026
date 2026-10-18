import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { buildPlan,sourceEquivalence,runtimeAsset,validateRecord,verifyEvidence,preflight,execute,main,VERSION,HELPER } from './dev-personalized-development.mjs';
const hash=x=>createHash('sha256').update(x).digest('hex');
const exact='a'.repeat(64);
function record(){return {version:'persona-adapter-development-admission.v1',status:'technical-development-demo',scope:'known-public-profile-development-demo',qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,
 sourceSha256:{['/reviewed/'+HELPER]:hash('same-source')},
 adapterManifest:{file:'/reviewed/data/ranker/pretrained/persona-choice-train-v1-attempt-1/serving.json',sha256:exact},
 gateCheckpoint:{file:'/reviewed/data/ranker/pretrained/query-intent-gate-v3-fit-attempt-1/head.pt',sha256:exact},
 v5Export:{file:'/assets/data/ranker/pretrained/actual-store-function-v5-attempt-1/serving-head.json',sha256:exact},
 copyManifest:{file:'/assets/data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-2/serving-manifest.json',sha256:exact},
 integrationManifest:{file:'/reviewed/experiments/ranker/pretrained/store_demo_integration_personalized_adapter_v2/manifest.json',sha256:exact}};}
function fixture(){
 const value=record();const bytes=Buffer.from(JSON.stringify(value));const checked=[];
 const serving={adapter:{file:'fit/adapter.safetensors'},publicCache:{file:'public-features/index.json',sha256:exact},inferenceParity:{file:'inference-parity.json',sha256:exact}};
 const index={entries:Object.fromEntries(Array.from({length:20},(_,i)=>[String(i),{archive:String(i)+'.npz',archiveSha256:exact}]))};
 const dependencies={dotenv:async()=>{},equivalence:async()=>({evidenceRoot:'/reviewed',currentSourceSha256:{['/code/'+HELPER]:hash('same-source')},sourceByteEquivalenceVerified:true}),
 evidence:()=>({mode:'technical-development-demo',developmentDemo:true,qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,overallPromotionPass:false,
 expected:{adapterSha256:exact,gateSha256:exact,colorHeadSha256:exact,gateTRAINManifestSha256:'25fdb0c976dc3b6e01e826e7f30ea63fc65e3fbb85f56f24f4c83e8538703232'}}),
 read:async filename=>filename==='/fixture/receipt.json'?bytes:filename.endsWith('/serving.json')?Buffer.from(JSON.stringify(serving)):
 filename.endsWith('/index.json')?Buffer.from(JSON.stringify(index)):Buffer.from('same-source'),
 checked:async(filename,sha,options)=>{checked.push(filename);return {file:filename,bytes:filename.endsWith('model-00001-of-00002.safetensors')?5349771222:filename.endsWith('model-00002-of-00002.safetensors')?600449850:1};}};
 return {options:{root:'/code',assetRoot:'/assets',receipt:'/fixture/receipt.json',receiptSha256:hash(bytes),session:'/code/data/session'},dependencies,checked};
}

test('failed-quality development receipt cannot become accepted/profile alias',()=>{
 const original=record();assert.equal(validateRecord(original).evidenceRoot,'/reviewed');
 for(const change of [{modelPromotion:true},{qualityGatePassed:true},{acceptedProfile:true},{status:'accepted'},{version:'personalized-local-adapter-profile.v2'}])
  assert.throws(()=>validateRecord({...original,...change}),/non-promoted/);
});

test('historical evidence and current main source bytes must match explicitly',async()=>{
 const recordValue=record();const proof=await sourceEquivalence(recordValue,'/main',async()=>Buffer.from('same-source'));
 assert.equal(proof.sourceByteEquivalenceVerified,true);assert.equal(proof.currentSourceSha256['/main/'+HELPER],hash('same-source'));
 await assert.rejects(sourceEquivalence(recordValue,'/main',async filename=>Buffer.from(filename.startsWith('/main/')?'changed':'same-source')),/source bytes differ/);
});

test('runtime paths map to main assets with no worktree fallback',()=>{
 assert.equal(runtimeAsset({file:'/reviewed/data/ranker/pretrained/run/head.pt',sha256:exact},'/reviewed','/main').file,'/main/data/ranker/pretrained/run/head.pt');
 assert.throws(()=>runtimeAsset({file:'/elsewhere/other/head.pt',sha256:exact},'/reviewed','/main'),/no worktree fallback/);
});

test('plan uses exact adapter flags and main runtime assets plus current-code manifest',async()=>{
 const f=fixture();const plan=await buildPlan(f.options,f.dependencies);
 assert.equal(plan.qualityGatePassed,false);assert.equal(plan.modelPromotion,false);assert.equal(plan.acceptedProfile,false);
 assert.equal(plan.profile.version,VERSION);assert.ok(plan.nativeArgs.includes('experiments.ranker.pretrained.store_persona_adapter_engine_v2'));
 assert.ok(!plan.nativeArgs.includes('--persona-head'));assert.ok(!plan.nativeArgs.includes('--port'));
 assert.ok(!plan.nativeArgs.some(value=>String(value).startsWith('/reviewed/')));
 assert.equal(plan.profile.integrationManifest.file,'/code/experiments/ranker/pretrained/store_demo_integration_personalized_adapter_v2/manifest.json');
 assert.equal(f.checked.filter(filename=>filename.endsWith('.npz')).length,20);
 assert.ok(f.checked.every(filename=>!filename.startsWith('/reviewed/')));
 assert.equal(plan.lock,'/assets/data/ranker/pretrained/native-owner.lock');
});

test('missing copied main bundle fails even while historical receipt names original files',async()=>{
 const f=fixture();const read=f.dependencies.read;const attempts=[];
 f.dependencies.read=async filename=>{attempts.push(filename);if(filename==='/assets/data/ranker/pretrained/persona-choice-train-v1-attempt-1/serving.json')throw Error('missing main bundle');return read(filename);};
 await assert.rejects(buildPlan(f.options,f.dependencies),/missing main bundle/);
 assert.ok(!attempts.includes('/reviewed/data/ranker/pretrained/persona-choice-train-v1-attempt-1/serving.json'));
});

test('Python admission uses reviewed source and lexical interpreter without model or shell invocation',()=>{
 let actual;
 const selected=verifyEvidence('/main/.tools/pretrained-venv/bin/python','/reviewed','/receipt.json',exact,(command,args,options)=>{
  actual={command,args,options};return {status:0,stdout:JSON.stringify({mode:'technical-development-demo',developmentDemo:true,qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,overallPromotionPass:false,expected:{adapterSha256:exact,gateSha256:exact}})};
 });
 assert.equal(selected.modelPromotion,false);assert.equal(actual.options.cwd,'/reviewed');assert.equal(actual.command,'/main/.tools/pretrained-venv/bin/python');
 assert.equal(actual.options.env.HF_HUB_OFFLINE,'1');assert.equal(actual.options.env.OPENAI_API_KEY,undefined);assert.equal(actual.options.shell,undefined);
 assert.deepEqual(actual.args.slice(-2),['/receipt.json',exact]);
 assert.throws(()=>verifyEvidence('python','/reviewed','/r',exact,()=>({status:1,stderr:'admission failed'})),/verification failed/);
});

test('busy ports fail before any evidence subprocess or output/session creation',async()=>{
 let builds=0;
 await assert.rejects(preflight({}, {ports:async()=>{throw Error('8767 busy');},build:async()=>{builds++;}}),/8767 busy/);
 assert.equal(builds,0);
});

test('preflight keeps explicit shared lock and reuses verified checkout',async()=>{
 const calls=[];
 const result=await preflight({}, {ports:async()=>calls.push('ports'),build:async()=>({python:'python',lock:'main-lock',assetRoot:'/main',root:'/code'}),
 lock:(python,filename)=>calls.push(filename),checkout:async()=>({reused:true,origin:'http://127.0.0.1:4331'})});
 assert.deepEqual(calls,['ports','main-lock']);assert.equal(result.checkout.reused,true);assert.equal(result.go,undefined);
});

test('explicit runner retains false promotion and rejects source changes before owned-process runner',async()=>{
 let runs=0;const plan={version:VERSION,qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,currentSourceSha256:{'/code/model.py':hash('same-source')},launcherSourceSha256:{}};
 await execute(plan,{read:async()=>Buffer.from('same-source'),run:async p=>{runs++;assert.equal(p.modelPromotion,false);}});assert.equal(runs,1);
 await assert.rejects(execute(plan,{read:async()=>Buffer.from('changed'),run:async()=>runs++}),/source changed/);assert.equal(runs,1);
 await assert.rejects(execute({...plan,modelPromotion:true},{run:async()=>runs++}),/non-promoted/);assert.equal(runs,1);
});

test('without --run main stays preflight-only and unknown acceptance flag rejects',async()=>{
 let runs=0;const original=console.log;console.log=()=>{};
 try {await main(['--receipt','/r','--receipt-sha256',exact],{preflight:async()=>({version:VERSION}),execute:async()=>runs++});assert.equal(runs,0);
 await main(['--run'],{preflight:async()=>({version:VERSION}),execute:async()=>runs++});assert.equal(runs,1);
 await assert.rejects(main(['--accepted-profile','/anything']),/Unknown/);}finally{console.log=original;}
});
