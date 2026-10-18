import { createRequire } from 'node:module';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { QUERIES, TIMEOUT_MS, installObserver, matchDomToCopy, parsedSourceFieldsMatch } from './store_candidate_browser_probe_v1.mjs';
import { installFetchCapture, parseResponseBytes, selectResponseEvidence } from './store_candidate_collection_probe_v1.mjs';
import { CONTEXT_OPTIONS, settleNavigationCaptures } from './store_candidate_browser_probe_v2.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const NATIVE = 'http://127.0.0.1:8766/personalize';
export const BOOTSTRAP_QUERY = 'Browse the store';
export const DEFAULT_MANIFEST = 'experiments/ranker/pretrained/store_demo_integration_v11/manifest.json';
const sha = value => createHash('sha256').update(value).digest('hex');
const save = (path, value) => writeFile(path, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });

export function fixedPairs() {
  return Array.from({ length: 5 }, (_, repeat) => QUERIES.map((query, queryIndex) => ({
    pairIndex: repeat * QUERIES.length + queryIndex, repeat, queryIndex, query,
    order: (repeat + queryIndex) % 2 === 0 ? ['generic', 'local'] : ['local', 'generic'],
  }))).flat();
}
export function fixedPlan() {
  return { version: 'paired-loaded-collection-incremental.v1', pairs: fixedPairs(), fixedPairs: 20,
    fixedArms: 40, perArmDenominator: 20, viewport: CONTEXT_OPTIONS.viewport, deviceScaleFactor: 1,
    bootstrapQuery: BOOTSTRAP_QUERY, scoredQueries: QUERIES, origin: 'http://127.0.0.1:4312',
    workload: 'Every arm gets a NEW isolated context and one unscored Browse bootstrap, then collection pre-scroll and two rAF BEFORE one assigned hash query. No post-dispatch scroll. Twenty paired queries; 10 generic-first and 10 local-first; all first/outlier observations retained.',
    browserCache: 'Context begins HTTP-cache cold at unscored bootstrap; scored hash dispatch is LOADED and cache-eligible from that own bootstrap. Not cold-navigation latency. OS/filesystem/native caches uncontrolled; resource timings retained.',
    nativeCache: 'Resident native model and catalog/static fact features warm. Local bootstrap and scored query each require one fresh native query POST and fresh neural copy; no response cache. Generic requires zero POSTs.',
    nativeRequestsPlanned: { unscoredBootstrap: 20, scored: 20, total: 40 },
    primaryEndpoint: 'Hash dispatch performance.now to current-query first min(3, actual DOM count) complete cards/title/copy fully visible and primary images complete+decoded. Both arms use the same observer two-rAF barrier after current-query DOM, decoded images and full visibility. Generic proof is status+input; local proof begins at the existing current-request firstThree React-commit performance mark, then requires its app mirror and actual response10 fields. App mirror frames overlap the common observer frames; no extra local-only frame barrier. Passive clone overhead applies to local; raw-body settlement is outside endpoint timing.',
    localFullEndpoint: 'Same dispatch clock to current complete mirror, all20 accepted fields across six cards plus six primary images decoded; DOM copy/raw response verified after capture. Six-card visibility is not required or claimed.',
    genericFields: 'Existing static product titles/prices/variants; generic has no generated badges/body and no invented model field counts.',
    zeroResults: 'Actual zero DOM cards plus current query response recorded separately; never a three-card success or included as a nonempty useful-content latency.',
    estimator: 'Nearest-rank p50/p95. Headline percentiles only with all20 valid nonempty samples per arm; paired local-minus-generic percentiles only with all20 valid pairs. Missing/failure remains null; no replacement/drop/success-only distribution.',
    physicalPaintMeasured: false, timeoutMs: TIMEOUT_MS, bodySettlementMs: 5000 };
}

export function nearestRank(values, p) {
  if (!values.length || values.some(v => !Number.isFinite(v)) || !(p > 0 && p <= 1)) return null;
  const sorted = [...values].sort((a,b) => a-b);
  return sorted[Math.max(0, Math.ceil(p * sorted.length) - 1)];
}
export function distribution(values, expected = 20) {
  const valid = values.filter(Number.isFinite);
  return { fixedDenominator: expected, valid: valid.length, missing: expected-valid.length, rawMs: values,
    p50Ms: values.length === expected && valid.length === expected ? nearestRank(valid,.5) : null,
    p95Ms: values.length === expected && valid.length === expected ? nearestRank(valid,.95) : null,
    minimumMs: valid.length === expected ? Math.min(...valid) : null,
    maximumMs: valid.length === expected ? Math.max(...valid) : null,
    estimator: 'nearest-rank; no percentile when fixed complete denominator is missing' };
}
export function summarizePairs(rows) {
  const values = mode => fixedPairs().map(pair => {
    const row = rows.find(r => r.pairIndex === pair.pairIndex && r.mode === mode);
    return row?.protocol?.passed && row.observation?.firstUseful?.cardCount > 0
      ? row.observation.firstUseful.fromDispatchMs : null;
  });
  const generic=values('generic'),local=values('local');
  const full=fixedPairs().map(pair => {
    const row=rows.find(r=>r.pairIndex===pair.pairIndex&&r.mode==='local');
    return row?.protocol?.passed && row.localFullValidated ? row.observation?.fullSix?.fromDispatchMs ?? null : null;
  });
  return { fixedPairs:20,fixedArms:40,recorded:rows.length,
    unattempted:fixedPairs().flatMap(p=>p.order.filter(mode=>!rows.some(r=>r.pairIndex===p.pairIndex&&r.mode===mode)).map(mode=>({pairIndex:p.pairIndex,mode}))),
    genericFirstUseful:distribution(generic),localFirstUseful:distribution(local),localFullSix:distribution(full),
    pairedIncrementalLocalMinusGeneric:distribution(local.map((v,i)=>Number.isFinite(v)&&Number.isFinite(generic[i])?v-generic[i]:null)),
    protocolPassed:rows.filter(r=>r.protocol?.passed).length,
    zeroResults:rows.filter(r=>r.observation?.zeroResult).map(r=>({pairIndex:r.pairIndex,mode:r.mode,receipt:r.observation.zeroResult})),
    failures:rows.filter(r=>r.error||!r.protocol?.passed).map(r=>({pairIndex:r.pairIndex,mode:r.mode,error:r.error??null,protocol:r.protocol??null})),
    scope:'Paired loaded/pre-scrolled collection requests, not cold navigation or generic generated20fields; no physical paint claim' };
}

// Self-contained passive page observer. Reuses v1 geometry/decode snapshots and
// never changes scrolling, image loading, responses, source text or model data.
export function installPairedObserver(settings) {
  const state=window.__pairedIncremental={...settings,firstUseful:null,fullSix:null,zeroResult:null,
    genericCommit:null,genericFramesReadyAt:null,commonCandidate:null,commonFramesReadyAt:null,stopped:false,last:null};
  if(settings.dispatchQuery){
    state.dispatchedAt=performance.now();
    state.dispatch={dispatchedAt:state.dispatchedAt,previousHash:location.hash,scrollY};
    location.hash=`#collection?q=${encodeURIComponent(state.query)}`;
    state.dispatch.targetHash=location.hash;
  }
  const visibleCards=(snap,n,copy)=>snap.cards.slice(0,n).length===n&&snap.cards.slice(0,n).every(card=>
    card.rect?.fullyWithinViewport&&card.titleRect?.fullyWithinViewport&&Boolean(card.title)
    &&(!copy||(card.statusParagraphs?.length>=2&&card.statusParagraphs.every(p=>p.rect?.fullyWithinViewport)))
    &&card.image?.clippedPrimaryRect?.fullyWithinViewport&&card.image.complete&&card.image.naturalWidth>0&&card.image.decodeStatus==='decoded');
  const currentMirror=stage=>window.__candidateProbe?.stages?.[stage]?.mirror;
  const localMirrorValid=(mirror,fields)=>mirror?.startTime>=state.dispatchedAt
    &&mirror.detail?.requestId&& !state.oldRequestIds.includes(mirror.detail.requestId)
    &&mirror.detail.requiredFields===fields&&mirror.detail.acceptedFields===fields&&mirror.detail.withheldFields===0;
  const poll=()=>{
    if(state.stopped)return;
    const snap=window.__candidateProbe?.last;const now=performance.now();
    if(!snap){requestAnimationFrame(poll);return;}
    const input=document.getElementById('shop-search')?.value??null;
    const status=document.querySelector('#collection .section-heading [role=status]')?.textContent?.trim()??'';
    const queryDom=status.includes(`“${state.query}”`)&&input===state.query;
    state.last={at:now,devicePixelRatio:window.devicePixelRatio,input,status,snapshot:snap};
    if(state.mode==='generic'&&queryDom&&!state.genericCommit){
      state.genericCommit={observedAt:now,input,status};
    }
    const first=currentMirror('firstThree'),complete=currentMirror('complete');
    const localReady=localMirrorValid(first,10);
    const commitMarks=state.mode==='local'?performance.getEntriesByType('mark').filter(e=>
      e.name.endsWith('.firstThree.react-commit')&&e.startTime>=state.dispatchedAt
      &&e.detail?.stage==='firstThree'&&e.detail?.requestId&&!state.oldRequestIds.includes(e.detail.requestId)
      &&e.detail.requiredFields===10&&e.detail.acceptedFields===10&&e.detail.withheldFields===0):[];
    const localCommit=commitMarks[commitMarks.length-1];
    const genericReady=state.mode==='generic'&&queryDom&&state.genericFramesReadyAt!==null;
    const count=snap.domCardCount;const n=Math.min(3,count);
    const currentDom=state.mode==='generic'?queryDom:Boolean(localCommit)&&input===state.query;
    const predicate=n>0&&currentDom&&visibleCards(snap,n,state.mode==='local');
    if(!state.commonCandidate&&(predicate||state.mode==='generic'&&queryDom&&count===0)){
      state.commonCandidate={observedAt:now,requestId:localCommit?.detail?.requestId??null,
        commitMark:localCommit?{name:localCommit.name,startTime:localCommit.startTime,detail:localCommit.detail}:null};
      requestAnimationFrame(()=>requestAnimationFrame(()=>{state.commonFramesReadyAt=performance.now();
        if(state.mode==='generic')state.genericFramesReadyAt=state.commonFramesReadyAt;}));
    }
    if(!state.firstUseful&&predicate&&state.commonFramesReadyAt!==null
      &&(state.mode==='generic'||localReady&&first.detail.requestId===state.commonCandidate.requestId)){
      state.firstUseful={observedAt:now,fromDispatchMs:now-state.dispatchedAt,cardCount:n,actualDomCount:count,
        requestId:state.mode==='local'?first.detail.requestId:null,snapshot:snap,
        genericCommit:state.genericCommit,genericFramesReadyAt:state.genericFramesReadyAt,
        commonCandidate:state.commonCandidate,commonFramesReadyAt:state.commonFramesReadyAt};
    }
    if(!state.zeroResult&&count===0&&(genericReady||state.mode==='local'&&localMirrorValid(complete,0))){
      state.zeroResult={observedAt:now,fromDispatchMs:now-state.dispatchedAt,actualDomCount:0,
        status,input,requestId:state.mode==='local'?complete.detail.requestId:null};
    }
    if(state.mode==='local'&&!state.fullSix&&localMirrorValid(complete,20)&&snap.cards.length>=6
        &&snap.cards.slice(0,6).every(c=>c.image?.complete&&c.image.naturalWidth>0&&c.image.decodeStatus==='decoded')){
      state.fullSix={observedAt:now,fromDispatchMs:now-state.dispatchedAt,requestId:complete.detail.requestId,
        actualDomCount:count,snapshot:snap,allSixVisible:snap.cards.slice(0,6).every(c=>c.rect?.fullyWithinViewport)};
    }
    requestAnimationFrame(poll);
  };
  requestAnimationFrame(poll);
  return state.dispatch??null;
}

export function protocolForArm(row,seenIds=new Set()) {
  const deviations=[];const count=row.actualPostRequests?.length??0;
  if(row.error||row.captureError)deviations.push('Arm has a recorded workload/capture error');
  if(!Number.isFinite(row.dispatch?.dispatchedAt))deviations.push('Missing browser dispatch clock');
  if(row.observation?.firstUseful){const candidate=row.observation.commonCandidate,frames=row.observation.commonFramesReadyAt;
    if(!candidate||!Number.isFinite(candidate.observedAt)||candidate.observedAt<row.dispatch?.dispatchedAt
      ||!Number.isFinite(frames)||frames<candidate.observedAt
      ||!Number.isFinite(row.observation.firstUseful.fromDispatchMs))deviations.push('Missing common current-DOM/visible/decoded two-rAF proof');}
  if(row.observation?.last?.devicePixelRatio!==1)deviations.push('Actual DPR1 not observed');
  if(row.mode==='generic'){
    if(count!==0||row.allPostRequests?.length!==0)deviations.push('Generic arm must make zero POST requests');
    const commit=row.observation?.genericCommit,frames=row.observation?.genericFramesReadyAt;
    if(!commit||commit.input!==row.query||!commit.status?.includes(`“${row.query}”`)
      ||!Number.isFinite(commit.observedAt)||commit.observedAt<row.dispatch?.dispatchedAt
      ||!Number.isFinite(frames)||frames<commit.observedAt)deviations.push('Missing current-query DOM plus two-rAF proof');
    if(!row.observation?.firstUseful&&!row.observation?.zeroResult)deviations.push('Missing useful generic cards or explicit zero-result receipt');
  }else{
    if(count!==1||row.actualPostRequests[0]?.query!==row.query)deviations.push('Local arm requires one assigned fresh query POST');
    if(!row.bodySettlement?.settled||!row.cloneCaptureSettlement?.settled)deviations.push('Native response capture unsettled');
    const id=row.observation?.firstUseful?.requestId??row.observation?.zeroResult?.requestId;
    if(!id||seenIds.has(id))deviations.push('Missing or reused local request ID');
    if(!row.responseEvidence?.selectedSource||row.responseEvidence.pathsAgree===false)deviations.push('No agreed actual native response evidence');
    const zero=Boolean(row.observation?.zeroResult);
    for(const stage of zero?['rank','complete']:['rank','firstThree','complete'])if(!row.responseEvidence?.events?.some(e=>e.type===stage&&e.requestId===id))deviations.push(`Missing actual ${stage} response for current request`);
    const first=row.responseEvidence?.events?.find(e=>e.type==='firstThree'&&e.requestId===id);
    const copy=first?.copySource;
    if(zero){
      const complete=row.responseEvidence?.events?.find(e=>e.type==='complete'&&e.requestId===id);
      if(complete?.requiredFields!==0||complete?.sections?.flatMap(s=>s.cards).length!==0||complete?.copySource?.kind!=='no-generation')deviations.push('Zero-result receipt does not match an actual no-generation response');
    }else{
      if(!['fresh-parallel-head','fresh-native'].includes(copy?.kind)||copy?.generatedEarlier!==false||copy?.freshPerRequestGeneration!==true)deviations.push('Expected fresh neural copy, not earlier response cache');
      if(first?.requiredFields!==10||first?.acceptedFields!==10||first?.withheldFields!==0)deviations.push('FirstThree actual response must contain all10 accepted fields');
      if(!row.firstDomMatch?.passed||!row.firstSourceMatch)deviations.push('First cards do not match actual response and parsed provenance');
    }
  }
  return{passed:deviations.length===0,deviations,requestId:row.observation?.firstUseful?.requestId??row.observation?.zeroResult?.requestId??null};
}

export function selectArmEvidence(cdp,clones,id,zero=false){
  if(!zero)return selectResponseEvidence(cdp,clones,id);
  const valid=r=>r.parseErrors?.length===0&&r.events?.some(e=>e.type==='rank'&&e.requestId===id)
    &&r.events?.some(e=>e.type==='complete'&&e.requestId===id&&e.requiredFields===0
      &&e.sections?.flatMap(s=>s.cards).length===0&&e.copySource?.kind==='no-generation');
  const a=cdp.find(valid),b=clones.find(valid),agree=a&&b?a.bodySha256===b.bodySha256:null;
  const selected=agree===false?null:a??b;
  return{selectedSource:selected?(selected===a?'playwright-response-body':'in-page-actual-response-clone'):null,
    pathsAgree:agree,selectedBodySha256:selected?.bodySha256??null,events:selected?.events??[],
    scope:'Actual zero-result rank/complete bytes only; no fabricated firstThree or copy fields'};
}

function captureBucket(){return{actualPostRequests:[],allPostRequests:[],responses:[],pending:[],open:true};}
async function settleBucket(page,bucket,dispatchedAt,needsNative){
  if(needsNative||bucket.actualPostRequests.length)Object.assign(bucket,await settleNavigationCaptures(bucket.pending,timeout=>page.waitForFunction(start=>{
    const entries=(window.__nativeFetchCapture?.requests??[]).filter(r=>r.startedAt>=start);
    return entries.length>0&&entries.every(r=>r.status!=='pending');
  },dispatchedAt,{timeout})));
  else{bucket.bodySettlement={settled:true,started:bucket.pending.length};bucket.cloneCaptureSettlement={settled:true,expectedNativeRequests:0};}
  bucket.open=false;
  const captures=await page.evaluate(start=>(window.__nativeFetchCapture?.requests??[]).filter(r=>r.startedAt>=start),dispatchedAt);
  bucket.cloneResponses=captures.map(c=>c.status==='complete'?parseResponseBytes(c.bodyBytes,{...c,source:'in-page-actual-response-clone'}):c);
}
async function health(){const response=await fetch('http://127.0.0.1:8766/health',{signal:AbortSignal.timeout(5000)});const raw=await response.text();return{status:response.status,body:JSON.parse(raw),bodySha256:sha(Buffer.from(raw))};}

export async function runPaired(output,manifestPath=DEFAULT_MANIFEST){
  await mkdir(output);
  const manifest=JSON.parse(await readFile(join(ROOT,manifestPath),'utf8'));
  const files=[...new Set(['experiments/ranker/pretrained/store_paired_incremental_probe_v1.mjs',
    'experiments/ranker/pretrained/test_store_paired_incremental_probe_v1.mjs',
    'experiments/ranker/pretrained/store_paired_incremental_probe_v1_PLAN.json',
    'experiments/ranker/pretrained/store_candidate_browser_probe_v1.mjs',
    'experiments/ranker/pretrained/store_candidate_browser_probe_v2.mjs',
    'experiments/ranker/pretrained/store_candidate_collection_probe_v1.mjs',
    'experiments/ranker/pretrained/store_candidate_image_service_v1.py',
    'experiments/ranker/pretrained/store_candidate_engine_v1.py','experiments/ranker/pretrained/store_demo_engine.py',
    'experiments/ranker/pretrained/store_rank_session_v5.py',manifestPath,
    ...Object.keys(manifest.frontendSha256??{}),...Object.keys(manifest.verificationSourcesSha256??{})])];
  const hashes=async()=>Object.fromEntries(await Promise.all(files.map(async p=>[p,sha(await readFile(join(ROOT,p)))])));
  const report={...fixedPlan(),startedAt:new Date().toISOString(),sourceHashes:await hashes(),
    integrationManifest:{path:manifestPath,sha256:sha(await readFile(join(ROOT,manifestPath))),version:manifest.version},attempts:[]};
  await save(join(output,'plan.json'),report);let browser;const seenIds=new Set();
  try{
    report.healthBefore=await health();if(!report.healthBefore.body.ready)throw Error('Native service not ready');
    const require=createRequire(join(ROOT,'apps/runner/package.json'));const{chromium}=require('playwright');
    browser=await chromium.launch({channel:'chrome',headless:false});report.browserVersion=browser.version();
    arms:for(const pair of report.pairs)for(const mode of pair.order){
      const row={pairIndex:pair.pairIndex,repeat:pair.repeat,query:pair.query,mode,order:pair.order,
        startedAt:new Date().toISOString(),requestFailures:[],pageErrors:[]};
      const context=await browser.newContext({...CONTEXT_OPTIONS,viewport:{...CONTEXT_OPTIONS.viewport}});
      let page,scored;
      try{
        await context.addInitScript(installFetchCapture);await context.addInitScript(installObserver);
        page=await context.newPage();let bucket=captureBucket();const bootstrap=bucket;
        page.on('requestfailed',r=>row.requestFailures.push({url:r.url(),method:r.method(),error:r.failure()}));
        page.on('pageerror',e=>row.pageErrors.push({message:e.message,stack:e.stack??null}));
        page.on('request',r=>{if(r.method()==='POST')bucket.allPostRequests.push({url:r.url(),method:r.method()});
          if(r.url()===NATIVE&&r.method()==='POST'){let query=null;try{query=JSON.parse(r.postData()??'{}').query??null;}catch{}
            bucket.actualPostRequests.push({url:r.url(),query,request:r.postData()});}});
        page.on('response',res=>{if(res.url()!==NATIVE||res.request().method()!=='POST')return;const target=bucket;
          const receipt={status:res.status(),captureStatus:'pending'};target.responses.push(receipt);
          target.pending.push((async()=>{try{const error=await res.finished();if(error)throw error;
            const parsed=parseResponseBytes(await res.body(),{source:'playwright-response-body'});
            if(target.open)Object.assign(receipt,parsed,{captureStatus:'complete'});
          }catch(error){if(target.open)Object.assign(receipt,{captureStatus:'error',error:String(error)});}})());});
        await page.goto(`http://127.0.0.1:4312/${mode==='local'?'?model=local':''}#collection?q=${encodeURIComponent(BOOTSTRAP_QUERY)}`,
          {waitUntil:'domcontentloaded',timeout:TIMEOUT_MS});
        if(mode==='local')await page.waitForFunction(()=>Boolean(window.__candidateProbe?.imageEndpoints?.complete),null,{timeout:TIMEOUT_MS});
        else await page.waitForFunction(query=>document.querySelector('#collection .section-heading [role=status]')?.textContent?.includes(`“${query}”`)
          &&document.getElementById('shop-search')?.value===query,BOOTSTRAP_QUERY,{timeout:TIMEOUT_MS});
        await settleBucket(page,bootstrap,0,mode==='local');
        row.bootstrap={...bootstrap,pending:undefined};
        const bootState=await page.evaluate(()=>window.__candidateProbe);
        row.bootstrapObserver=bootState;
        const oldIds=Object.values(bootState.stages).map(s=>s.mirror.detail.requestId);
        if(mode==='local'){
          const id=bootState.stages.complete?.mirror?.detail?.requestId;
          row.bootstrap.responseEvidence=selectResponseEvidence(bootstrap.responses,bootstrap.cloneResponses,id);
          if(bootstrap.actualPostRequests.length!==1||bootstrap.actualPostRequests[0].query!==BOOTSTRAP_QUERY
            ||!bootstrap.bodySettlement.settled||!bootstrap.cloneCaptureSettlement.settled||!row.bootstrap.responseEvidence.selectedSource)
            throw Error('Unscored native bootstrap did not fully settle');
          for(const id of oldIds)seenIds.add(id);
        }else if(bootstrap.allPostRequests.length)throw Error('Generic bootstrap unexpectedly sent POST');
        await page.evaluate(()=>document.getElementById('collection').scrollIntoView({block:'start',behavior:'instant'}));
        await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
        bucket=captureBucket();scored=bucket;
        row.preDispatch=await page.evaluate(settings=>{
          const before={hash:location.hash,scrollY,cardCount:document.querySelectorAll('#collection article.product').length};
          document.documentElement.removeAttribute('data-local-model-readiness');window.__candidateProbe.stopped=true;
          return{before,query:settings.query,mode:settings.mode,oldRequestIds:settings.oldRequestIds};
        },{query:pair.query,mode,oldRequestIds:oldIds});
        await page.evaluate(installObserver);
        // Observer arming, in-page clock and hash assignment share one turn.
        row.dispatch=await page.evaluate(installPairedObserver,
          {mode,query:pair.query,oldRequestIds:oldIds,dispatchQuery:true});
        await page.waitForFunction(mode=>Boolean((window.__pairedIncremental?.firstUseful||window.__pairedIncremental?.zeroResult)
          &&(mode==='generic'||window.__pairedIncremental?.fullSix||window.__pairedIncremental?.zeroResult)),mode,{timeout:TIMEOUT_MS});
      }catch(error){row.error=String(error);}
      finally{
        if(scored&&page)try{
        await settleBucket(page,scored,row.dispatch?.dispatchedAt??Infinity,mode==='local');
        Object.assign(row,{...scored,pending:undefined});
        row.observation=await page.evaluate(()=>{window.__pairedIncremental.stopped=true;window.__candidateProbe.stopped=true;
          return{...window.__pairedIncremental,browser:window.__candidateProbe,navigation:performance.getEntriesByType('navigation').map(e=>e.toJSON()),
            resources:performance.getEntriesByType('resource').map(e=>e.toJSON())};});
        if(mode==='local'){
          const id=row.observation.firstUseful?.requestId??row.observation.zeroResult?.requestId;
          row.responseEvidence=selectArmEvidence(row.responses,row.cloneResponses,id,Boolean(row.observation.zeroResult));
          const first=row.responseEvidence.events.find(e=>e.type==='firstThree'&&e.requestId===id);
          const complete=row.responseEvidence.events.find(e=>e.type==='complete'&&e.requestId===id);
          row.firstDomMatch=matchDomToCopy(row.observation.firstUseful?.snapshot,first);
          row.firstSourceMatch=parsedSourceFieldsMatch(row.observation.browser.stages.firstThree?.mirror?.detail?.copySource,first?.copySource);
          row.localFullValidated=Boolean(row.observation.fullSix&&row.observation.fullSix.requestId===id&&matchDomToCopy(row.observation.fullSix.snapshot,complete).passed
            &&complete?.requiredFields===20&&complete?.acceptedFields===20&&complete?.withheldFields===0);
        }
        const screenshot=join(output,`pair-${pair.pairIndex+1}-${mode}.png`);const bytes=await page.screenshot({path:screenshot,fullPage:false,timeout:5000});
        row.screenshot={path:screenshot,sha256:sha(bytes),bytes:bytes.length};
        }catch(error){row.captureError=String(error);Object.assign(row,{...scored,pending:undefined});}
        row.protocol=protocolForArm(row,seenIds);if(row.protocol.requestId)seenIds.add(row.protocol.requestId);
        report.attempts.push(row);await save(join(output,`pair-${pair.pairIndex+1}-${mode}.json`),row);await context.close();}
      if(!row.protocol.passed){report.stoppedBeforeNextArm={pairIndex:pair.pairIndex,mode,reason:'Failed protocol, endpoint or unsettled evidence; no replacement attempts'};break arms;}
    }
    report.healthAfter=await health();report.sourceHashesAfter=await hashes();
    report.sourcesUnchanged=Object.entries(report.sourceHashes).every(([p,h])=>report.sourceHashesAfter[p]===h);
    report.startupUnchanged=JSON.stringify(report.healthBefore.body.startup)===JSON.stringify(report.healthAfter.body.startup);
    if(!report.sourcesUnchanged||!report.startupUnchanged)report.error='Source or service startup/checkpoint lifetime changed';
  }catch(error){report.error=String(error);}
  finally{if(browser)await browser.close();report.finishedAt=new Date().toISOString();report.summary=summarizePairs(report.attempts);await save(join(output,'report.json'),report);}
  return report;
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  if(process.argv.length<3||process.argv.length>4)throw Error('Usage: node store_paired_incremental_probe_v1.mjs NEW_OUTPUT [INTEGRATION_MANIFEST]');
  const report=await runPaired(resolve(process.argv[2]),process.argv[3]??DEFAULT_MANIFEST);console.log(JSON.stringify(report.summary,null,2));
  if(report.error||report.summary.protocolPassed!==40||report.summary.pairedIncrementalLocalMinusGeneric.valid!==20)process.exitCode=1;
}
