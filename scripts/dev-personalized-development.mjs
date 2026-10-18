// Explicit non-promoted development-demo entry; accepted profile policy unchanged.
import { createHash, randomUUID } from 'node:crypto';
import { readFile, stat, access } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { assertFreePorts, assertNoFrontendDotenv, assertNativeLockFree, chooseCheckout,
  safeEnvironment, run as runChildren } from './dev-personalized.mjs';
import { adapterArguments } from './personalized-adapter-profile.mjs';

export const ROOT=fileURLToPath(new URL('../',import.meta.url));
export const VERSION='personalized-development-demo-launch.v1';
export const HELPER='experiments/ranker/pretrained/store_persona_development_admission_v1.py';
const ORIGINAL_CATALOG='data/ranker/pretrained/actual-store-function-v3-attempt-1';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const exact=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
const need=(ok,message)=>{if(!ok)throw Error(`Development demo: ${message}`);};

export function validateRecord(record) {
  need(record?.version==='persona-adapter-development-admission.v1'&&record.status==='technical-development-demo'&&
    record.scope==='known-public-profile-development-demo'&&record.qualityGatePassed===false&&record.modelPromotion===false&&
    record.acceptedProfile===false,'explicit failed-quality non-promoted development receipt required; accepted profiles are not substituted');
  const helpers=Object.keys(record.sourceSha256??{}).filter(name=>name.endsWith('/'+HELPER));
  need(helpers.length===1&&exact(record.sourceSha256[helpers[0]]),'one exact historical admission helper source required');
  return {evidenceRoot:helpers[0].slice(0,-HELPER.length-1)};
}

export async function sourceEquivalence(record,root,read=readFile) {
  const {evidenceRoot}=validateRecord(record);const result={};
  for(const [original,expected] of Object.entries(record.sourceSha256)) {
    const relative=path.relative(evidenceRoot,original);
    need(!path.isAbsolute(relative)&&!relative.startsWith('..')&&relative.endsWith('.py')&&exact(expected),'explicit historical source inventory required');
    const current=path.join(root,relative);
    need(sha(await read(original))===expected&&sha(await read(current))===expected,`source bytes differ between reviewed evidence and launch code: ${relative}`);
    result[current]=expected;
  }
  return {evidenceRoot,currentSourceSha256:result,sourceByteEquivalenceVerified:true};
}

export function runtimeAsset(record,evidenceRoot,assetRoot) {
  need(record&&exact(record.sha256)&&typeof record.file==='string','explicit runtime artifact reference required');
  const relative=path.relative(evidenceRoot,record.file);
  const marker='data/ranker/pretrained/';
  let selected;
  if(!path.isAbsolute(relative)&&!relative.startsWith('..')&&relative.startsWith(marker))selected=relative;
  else {
    const assetRelative=path.relative(assetRoot,record.file);
    need(!path.isAbsolute(assetRelative)&&!assetRelative.startsWith('..')&&assetRelative.startsWith(marker),
      'runtime artifacts must map explicitly to main asset data; no worktree fallback');
    selected=assetRelative;
  }
  return {file:path.join(assetRoot,selected),sha256:record.sha256,reviewedOriginalFile:record.file};
}

export function verifyEvidence(python,evidenceRoot,receipt,receiptSha,execute=spawnSync) {
  const code="import json,sys; from experiments.ranker.pretrained.store_persona_development_admission_v1 import selection; print(json.dumps(selection(sys.argv[1],sys.argv[2]),allow_nan=False))";
  const result=execute(python,['-c',code,receipt,receiptSha],{cwd:evidenceRoot,env:safeEnvironment(),encoding:'utf8',timeout:30_000,maxBuffer:16*1024*1024,stdio:['ignore','pipe','pipe']});
  need(!result.error&&result.status===0,`reviewed development evidence verification failed: ${result.error?.message??result.stderr??'unknown error'}`);
  const selected=JSON.parse(result.stdout);
  need(selected.mode==='technical-development-demo'&&selected.developmentDemo===true&&selected.qualityGatePassed===false&&
    selected.modelPromotion===false&&selected.acceptedProfile===false&&selected.overallPromotionPass===false&&
    selected.expected?.adapterSha256&&selected.expected?.gateSha256,'mechanical evidence must retain the failed quality/promotion state');
  return selected;
}

async function checked(filename,expected,{executable=false}={}) {
  const info=await stat(filename).catch(()=>{throw Error(`Development demo: required local file missing: ${filename}; no download attempted`);});
  need(info.isFile()&&info.size>0,`nonempty local file required: ${filename}`);
  if(executable)await access(filename,1);
  if(expected)need(exact(expected)&&sha(await readFile(filename))===expected,`pinned artifact changed: ${filename}`);
  return {file:filename,bytes:info.size,...(expected?{sha256:expected}:{})};
}

export async function buildPlan({root=ROOT,assetRoot,receipt,receiptSha256,session}={},deps={}) {
  root=path.resolve(root);need(receipt&&exact(receiptSha256),'explicit --receipt and --receipt-sha256 required');
  receipt=path.resolve(receipt);const bytes=await (deps.read??readFile)(receipt);need(sha(bytes)===receiptSha256,'development receipt checksum differs');
  const record=JSON.parse(bytes);validateRecord(record);
  await (deps.dotenv??assertNoFrontendDotenv)(root);
  assetRoot=path.resolve(assetRoot??(await stat(path.join(root,'.tools/pretrained-venv/bin/python')).then(()=>root).catch(()=>path.join(root,'../dubhacks-2026'))));
  const python=path.join(assetRoot,'.tools/pretrained-venv/bin/python'); // Keep lexical venv path.
  const equivalence=await (deps.equivalence??sourceEquivalence)(record,root);
  const selected=await (deps.evidence??verifyEvidence)(python,equivalence.evidenceRoot,receipt,receiptSha256);
  const candidate=path.join(assetRoot,'data/ranker/pretrained/store-parallel-text-train-all-demo-v1-attempt-2');
  const catalogFeatureRoot=path.join(assetRoot,ORIGINAL_CATALOG);
  const colorHead={file:path.join(equivalence.evidenceRoot,'data/ranker/pretrained/persona-qwen-optimize-v2-attempt-1/raw-2500/head.pt'),sha256:selected.expected.colorHeadSha256};
  const colorCache={file:path.join(equivalence.evidenceRoot,'data/ranker/pretrained/persona-qwen-fit-v1-attempt-1/public-demo-features.pt'),sha256:'dbf8c335a533120f5220a19f88ad163e4984631e7d9a7e80e5894080d02af28e'};
  const servingManifest=runtimeAsset(record.adapterManifest,equivalence.evidenceRoot,assetRoot);
  const queryIntentHead=runtimeAsset(record.gateCheckpoint,equivalence.evidenceRoot,assetRoot);
  const rankExport=runtimeAsset(record.v5Export,equivalence.evidenceRoot,assetRoot);
  const manifestRelative=path.relative(equivalence.evidenceRoot,record.integrationManifest.file);
  need(!manifestRelative.startsWith('..')&&!path.isAbsolute(manifestRelative)&&manifestRelative.startsWith('experiments/ranker/pretrained/'),
    'explicit current-code integration manifest mapping required');
  const integrationManifest={file:path.join(root,manifestRelative),sha256:record.integrationManifest.sha256};
  const serving=JSON.parse(await (deps.read??readFile)(servingManifest.file));
  need(serving.adapter?.file==='fit/adapter.safetensors'&&serving.publicCache?.file==='public-features/index.json'&&serving.inferenceParity?.file==='inference-parity.json',
    'reviewed co-located adapter/index/certificate layout required');
  const profile={version:VERSION,kind:'explicit-development-demo',qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,
    personaHead:{file:path.join(path.dirname(servingManifest.file),serving.adapter.file),sha256:selected.expected.adapterSha256},
    servingManifest,queryIntentHead,colorHead:runtimeAsset(colorHead,equivalence.evidenceRoot,assetRoot),
    colorCache:runtimeAsset(colorCache,equivalence.evidenceRoot,assetRoot),rankExport,
    integrationManifest,catalogFeatureRoot,queryIntentSourceManifestSha256:selected.expected.gateTRAINManifestSha256};
  const localCheck=deps.checked??checked;const assets=[];
  for(const [file,checksum,executable] of [[python,null,true],[path.join(root,...'experiments.ranker.pretrained.store_persona_adapter_engine_v2'.split('.'))+'.py'],
    [profile.personaHead.file,profile.personaHead.sha256],[profile.colorHead.file,profile.colorHead.sha256],[profile.colorCache.file,profile.colorCache.sha256],
    [profile.servingManifest.file,profile.servingManifest.sha256],[profile.queryIntentHead.file,profile.queryIntentHead.sha256],
    [profile.rankExport.file,profile.rankExport.sha256],[profile.integrationManifest.file,profile.integrationManifest.sha256],
    [path.join(root,'apps/storefront/node_modules/.bin/vp'),null,true],
    [path.join(candidate,'head.pt'),'074169e31bf5f6b99cc5bc35c9b2e64d82396d5225b3a67cc2cf5f613fcc4889'],
    [path.join(candidate,'serving-manifest.json'),record.copyManifest.sha256],
    [path.join(candidate,'public-runtime-features.json'),'bbd2400e0c6e2907c783f01b7c157714cae00ec5b76fc619790a939a0f9d07f3'],
    [path.join(assetRoot,'data/ranker/pretrained/store-parallel-text-v1-attempt-1/features.pt'),'2624a947f715e489431abf652a16e6872677fd0d4b824a76df694e0f8fae6ee8'],
    [path.join(assetRoot,'data/ranker/pretrained/actual-store-copy-fit-v2-attempt-1/adapter.safetensors'),'56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8'],
    ...['report.json','packet.json','RAW_FEATURES_FROZEN.json'].map(name=>[path.join(catalogFeatureRoot,name)])])assets.push(await localCheck(file,checksum,{executable:!!executable}));
  const model=path.join(assetRoot,'.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53');
  for(const [name,bytes] of [['model-00001-of-00002.safetensors',5349771222],['model-00002-of-00002.safetensors',600449850]]) {
    const asset=await localCheck(path.join(model,name));need(asset.bytes===bytes,'original local model shard size differs');assets.push(asset);
  }
  for(const name of ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'])assets.push(await localCheck(path.join(model,name)));
  const indexFile=path.join(path.dirname(servingManifest.file),serving.publicCache.file);
  assets.push(await localCheck(indexFile,serving.publicCache.sha256));
  assets.push(await localCheck(path.join(path.dirname(servingManifest.file),serving.inferenceParity.file),serving.inferenceParity.sha256));
  const index=JSON.parse(await (deps.read??readFile)(indexFile));
  for(const entry of Object.values(index.entries??{})) {
    need(typeof entry.archive==='string'&&!path.isAbsolute(entry.archive)&&!entry.archive.split('/').includes('..'),'co-located public feature archive required');
    assets.push(await localCheck(path.join(path.dirname(indexFile),entry.archive),entry.archiveSha256));
  }
  need(Object.keys(index.entries??{}).length===20,'all20 public-only feature archives required before spawn');
  const chosenSession=session??path.join(root,'data/personalized-development',`${new Date().toISOString().replaceAll(':','-')}-${randomUUID()}`);
  need(path.isAbsolute(chosenSession)&&!path.relative(root,chosenSession).startsWith('..'),'unique absolute session inside current code root required');
  const launcherSources=[path.join(root,'scripts/dev-personalized-development.mjs'),path.join(root,'scripts/dev-personalized.mjs'),path.join(root,'scripts/personalized-adapter-profile.mjs')];
  const launcherSourceSha256=Object.fromEntries(await Promise.all(launcherSources.map(async file=>[file,sha(await (deps.read??readFile)(file))])));
  return {version:VERSION,root,assetRoot,python,lock:path.join(assetRoot,'data/ranker/pretrained/native-owner.lock'),
    session:chosenSession,
    profile,assets,nativeArgs:adapterArguments({assetRoot,profile,candidate}),developmentEvidence:selected,
    receipt:{file:receipt,sha256:receiptSha256},...equivalence,launcherSourceSha256,
    qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,
    scope:'known public profile technical demo; quality gate failed17/32 versus frozen base20/32; no learned adaptation advantage',
    noDownloads:true,noEnvironmentFileLoaded:true};
}

export async function preflight(options,deps={}) {
  await (deps.ports??assertFreePorts)(); // Fail before metadata reads or processes when occupied.
  const plan=await (deps.build??buildPlan)(options,deps);
  (deps.lock??assertNativeLockFree)(plan.python,plan.lock);
  const checkout=await (deps.checkout??chooseCheckout)();let go;
  if(!checkout.reused) {
    go=path.join(plan.assetRoot,'.tools/go/bin/go');await (deps.checked??checked)(go,null,{executable:true});
    await (deps.checked??checked)(path.join(plan.root,'apps/api/cmd/demo-checkout-server/main.go'));
  }
  return {...plan,checkout,...(go?{go}:{})};
}

export async function execute(plan,{run=runChildren,read=readFile}={}) {
  need(plan.version===VERSION&&plan.qualityGatePassed===false&&plan.modelPromotion===false&&plan.acceptedProfile===false,
    'only explicit non-promoted development plan may run');
  for(const [filename,checksum] of Object.entries({...plan.currentSourceSha256,...plan.launcherSourceSha256}))
    need(sha(await read(filename))===checksum,`source changed before spawn: ${filename}`);
  console.log('Development demo: quality gate FAILED (17/32; frozen base20/32). Model promotion: false.');
  // Generic owned-process runner is reused; acceptedProfile/verifyAdapterArtifacts
  // are never called with a relabelled profile. This separate development policy
  // and native engine's own checksum/927/guard checks control this explicit entry.
  await run(plan);
}

export async function main(argv=process.argv.slice(2),deps={}) {
  const options={};let run=false;
  const names={'--receipt':'receipt','--receipt-sha256':'receiptSha256','--asset-root':'assetRoot','--session':'session'};
  for(let i=0;i<argv.length;i++) {
    if(argv[i]==='--run')run=true;
    else if(names[argv[i]]&&argv[i+1])options[names[argv[i]]]=argv[++i];
    else throw Error(`Unknown development-demo argument: ${argv[i]}`);
  }
  const plan=await (deps.preflight??preflight)(options,deps);
  if(!run){console.log(JSON.stringify({...plan,execution:'CPU preflight only; no services/session created'},null,2));return plan;}
  process.umask(0o077);await (deps.execute??execute)(plan);return plan;
}
if(process.argv[1]&&pathToFileURL(path.resolve(process.argv[1])).href===import.meta.url)
  main().catch(error=>{console.error(error.message);process.exitCode=1;});
