import test from 'node:test';import assert from 'node:assert/strict';import {createHash}from'node:crypto';
import {buildPlan,execute,normalMainFile,verifyBinding,VERSION}from'./dev-personalized-fast-ui.mjs';
import {VERSION as FAST,MODULE,MAIN_ASSET_ROOT as ROOT}from'./dev-personalized-fast.mjs';
const hash=b=>createHash('sha256').update(b).digest('hex');const bytes=Buffer.from('reviewed');const sum=hash(bytes);const file=ROOT+'/experiments/new/manifest.json';
const info={isSymbolicLink:()=>false,isFile:()=>true,isDirectory:()=>true};
function base(){return{version:FAST,root:ROOT,assetRoot:ROOT,python:ROOT+'/.tools/pretrained-venv/bin/python',scope:'old',nativeArgs:['-m',MODULE,'--integration-manifest',ROOT+'/old.json','--serving-manifest','head-unchanged'],profile:{integrationManifest:{file:ROOT+'/old.json',sha256:'a'.repeat(64)},personaHead:{sha256:'b'.repeat(64)},queryIntentHead:{sha256:'c'.repeat(64)}},assets:[],qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,currentSourceSha256:{},launcherSourceSha256:{}};}
const options={integrationManifest:file,integrationManifestSha256:sum};
function deps(changes={}){return{baseBuild:async()=>base(),lstat:async()=>info,realpath:async x=>x,read:async()=>bytes,binding:()=>({sourceSha256:{[ROOT+'/source.py']:'d'.repeat(64)}}),...changes};}
test('only integration manifest/profile field changes; all checkpoint args/flags preserved',async()=>{
 const prior=base();const snapshot=structuredClone(prior);const result=await buildPlan(options,deps({baseBuild:async()=>prior}));
 assert.equal(result.version,VERSION);const expected=[...prior.nativeArgs];expected[3]=file;assert.deepEqual(result.nativeArgs,expected);
 const expectedProfile={...prior.profile,integrationManifest:{file,sha256:sum}};assert.deepEqual(result.profile,expectedProfile);assert.deepEqual(prior,snapshot);
 for(const key of['qualityGatePassed','modelPromotion','acceptedProfile'])assert.equal(result[key],false);
 assert.equal(result.currentSourceSha256[file],sum);assert.equal(result.uiIntegrationReplacement.historicalManifest.file,ROOT+'/old.json');
});
test('wrong checksum/catalog delta reject before any owned runner',async()=>{
 await assert.rejects(buildPlan({...options,integrationManifestSha256:'e'.repeat(64)},deps()),/checksum differs/);
 await assert.rejects(buildPlan(options,deps({binding:()=>{throw Error('Actual public catalog differs');}})),/catalog differs/);
 assert.throws(()=>verifyBinding('python',ROOT,file,sum,()=>({status:2,stderr:'catalog differs'})),/catalog.*binding failed/);
});
test('symlink leaf/ancestor, outside path, directory and realpath escape rejected',async()=>{
 for(const target of ['/outside/manifest.json',ROOT+'/../other.json'])await assert.rejects(normalMainFile(target,deps()),/inside MAIN|normalized/);
 for(const target of [file,ROOT+'/experiments'])await assert.rejects(normalMainFile(file,deps({lstat:async p=>p===target?{...info,isSymbolicLink:()=>true}:info})),/symlink/);
 await assert.rejects(normalMainFile(file,deps({realpath:async()=>'/outside/manifest.json'})),/resolved outside/);
 await assert.rejects(normalMainFile(file,deps({lstat:async p=>p===file?{...info,isFile:()=>false}:info})),/normal file/);
});
test('actual binding subprocess uses MAIN cwd, unchanged public model contract and safe offline env',()=>{
 let seen;const result=verifyBinding('python',ROOT,file,sum,(cmd,args,opts)=>{seen={cmd,args,opts};return{status:0,stdout:JSON.stringify({version:'reviewed-fast-ui-binding.v1',manifestSha256:sum,root:ROOT,publicCatalogAndModelFactsUnchanged:true,sourceSha256:{[ROOT+'/source.py']:'d'.repeat(64)}})};});
 assert.equal(result.root,ROOT);assert.equal(seen.opts.cwd,ROOT);assert.equal(seen.opts.env.HF_HUB_OFFLINE,'1');assert.equal(seen.opts.shell,undefined);assert.ok(seen.args[1].includes('ImageDeliveryCatalogBinding'));assert.deepEqual(seen.args.slice(-3),[file,sum,ROOT]);
});
test('missing required pins, duplicate manifest flags and WT roots rejected',async()=>{
 await assert.rejects(buildPlan({},deps()),/explicit/);
 await assert.rejects(buildPlan(options,deps({baseBuild:async()=>({...base(),root:'/wt'})})),/MAIN code/);
 const bad=base();bad.nativeArgs.push('--integration-manifest','extra');await assert.rejects(buildPlan(options,deps({baseBuild:async()=>bad})),/one exact/);
});
