import test from'node:test';import assert from'node:assert/strict';import{createHash}from'node:crypto';import{buildPlan,execute,verifyExpanded,VERSION,MODULE,SOURCES}from'./dev-personalized-expanded.mjs';import{MAIN_ASSET_ROOT as ROOT,MODULE as OLD}from'./dev-personalized-fast.mjs';
const bytes=Buffer.from('fixture');const sum=createHash('sha256').update(bytes).digest('hex');const manifest=ROOT+'/new/manifest.json',copy=ROOT+'/data/ranker/pretrained/expanded-copy';
const options={integrationManifest:manifest,integrationManifestSha256:sum,expandedCopyRun:copy,expandedCopyManifestSha256:sum};
const original=()=>({root:ROOT,assetRoot:ROOT,python:'python',profile:{personaHead:{sha256:'a'.repeat(64)},queryIntentHead:{sha256:'b'.repeat(64)},integrationManifest:{file:ROOT+'/old.json'}},nativeArgs:['-m',OLD,'--candidate-run',ROOT+'/oldcopy','--integration-manifest',ROOT+'/old.json','--serving-manifest','unchanged'],assets:[],currentSourceSha256:{},launcherSourceSha256:{},measurementPolicy:{},qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,scope:'old'});
const proof=()=>({version:'expanded-startup-cpu-verified.v1',products:35,variants:105,copyBindings:105,bindingManifestSha256:sum,copyManifestSha256:sum,modelPromotion:false,freshNeuralCopyRequired:true,sourceSha256:{},assetSha256:{}});
const deps=(changes={})=>({baseBuild:async()=>original(),lstat:async()=>({isFile:()=>true,isDirectory:()=>true,isSymbolicLink:()=>false}),realpath:async p=>p,read:async()=>bytes,verify:()=>proof(),...changes});
test('only module/catalog/copy bindings replaced; model checkpoint and policy remain explicit',async()=>{
 const prior=original(),snapshot=structuredClone(prior);const plan=await buildPlan(options,deps({baseBuild:async()=>prior}));
 assert.equal(plan.version,VERSION);assert.equal(plan.nativeArgs[1],MODULE);assert.deepEqual(plan.profile.personaHead,prior.profile.personaHead);assert.deepEqual(plan.profile.queryIntentHead,prior.profile.queryIntentHead);assert.equal(plan.nativeArgs.at(-2),'--expanded-copy-manifest-sha256');assert.equal(plan.measurementPolicy.defaultPublicQueryKeysPreencoded,24);assert.equal(plan.modelPromotion,false);assert.deepEqual(prior,snapshot);for(const name of SOURCES)assert.equal(plan.currentSourceSha256[ROOT+'/'+name],sum);
});
test('wrong checksums and actual public/native/copy disagreement fail closed',async()=>{
 await assert.rejects(buildPlan({...options,integrationManifestSha256:'e'.repeat(64)},deps()),/binding checksum/);
 await assert.rejects(buildPlan(options,deps({verify:()=>{throw Error('catalog vectors not real');}})),/not real/);
 assert.throws(()=>verifyExpanded('python',ROOT,manifest,sum,copy,sum,()=>({status:2,stderr:'bad105'})),/verification failed/);
});
test('CPU native-proof verifier requires105bindings/35products and offline actualMAIN context',()=>{
 let seen;const value=verifyExpanded('python',ROOT,manifest,sum,copy,sum,(cmd,args,opts)=>{seen={cmd,args,opts};return{status:0,stdout:JSON.stringify(proof())};});assert.equal(value.products,35);assert.equal(seen.opts.cwd,ROOT);assert.equal(seen.opts.env.HF_HUB_OFFLINE,'1');assert.ok(seen.args[1].includes('load_catalog_vectors'));
 assert.throws(()=>verifyExpanded('python',ROOT,manifest,sum,copy,sum,()=>({status:0,stdout:JSON.stringify({...proof(),products:30})})),/35\/105/);
});
test('owned spawn rejects mutated artifacts and preserves nonpromotion',async()=>{
 const plan=await buildPlan(options,deps());let runs=0;const log=console.log;console.log=()=>{};
 try{await execute(plan,{read:async()=>bytes,run:async p=>{runs++;assert.equal(p,plan);}});assert.equal(runs,1);await assert.rejects(execute(plan,{read:async()=>Buffer.from('changed'),run:async()=>runs++}),/changed before spawn/);await assert.rejects(execute({...plan,modelPromotion:true}),/non-promoted/);assert.equal(runs,1);}finally{console.log=log;}
});
