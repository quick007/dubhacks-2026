import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { fixedPairs, fixedPlan, nearestRank, distribution, summarizePairs, installPairedObserver,
  protocolForArm, selectArmEvidence, BOOTSTRAP_QUERY } from './store_paired_incremental_probe_v1.mjs';
import { parseResponseBytes } from './store_candidate_collection_probe_v1.mjs';

const query='a light for my bedside table';
const source={kind:'fresh-parallel-head',generatedEarlier:false,freshPerRequestGeneration:true};
function snapshot(count=16){return{domCardCount:count,cards:Array.from({length:Math.min(6,count)},(_,i)=>({
  title:`Static title ${i}`,rect:{fullyWithinViewport:i<3},titleRect:{fullyWithinViewport:i<3},
  statusParagraphs:[{text:'Actual badge',rect:{fullyWithinViewport:i<3}},{text:'Actual body',rect:{fullyWithinViewport:i<3}}],
  image:{clippedPrimaryRect:{fullyWithinViewport:i<3},complete:true,naturalWidth:100,decodeStatus:'decoded'},
}))};}
function mirror(stage,id='new',startTime=110){return{startTime,duration:20,detail:{requestId:id,stage,
  requiredFields:stage==='firstThree'?10:20,acceptedFields:stage==='firstThree'?10:20,withheldFields:0}};}
function observer(mode='generic',count=16){
  let now=100,status=`No close matches for “${BOOTSTRAP_QUERY}”. Browse the collection below.`,input=BOOTSTRAP_QUERY;
  const queue=[],marks=[];const window={devicePixelRatio:1,__candidateProbe:{last:snapshot(count),stages:{}}};
  const location={hash:'#collection?q=Browse%20the%20store'};
  const document={getElementById:()=>({value:input}),querySelector:()=>({textContent:status})};
  const result=vm.runInNewContext(`(${installPairedObserver.toString()})(${JSON.stringify({mode,query,oldRequestIds:['old'],dispatchQuery:true})})`,
    {window,document,location,scrollY:600,performance:{now:()=>now,getEntriesByType:()=>marks},requestAnimationFrame:fn=>queue.push(fn),encodeURIComponent});
  return{window,queue,result,location,marks,tick(){now+=10;queue.shift()?.();},current(){input=query;status=`Suggestions for “${query}”`;}};
}
function genericRow(index=0){return{pairIndex:index,query,mode:'generic',actualPostRequests:[],allPostRequests:[],dispatch:{dispatchedAt:100},
  observation:{commonCandidate:{observedAt:110},commonFramesReadyAt:140,last:{devicePixelRatio:1},genericCommit:{observedAt:110,input:query,status:`Suggestions for “${query}”`},
    genericFramesReadyAt:140,firstUseful:{cardCount:3,fromDispatchMs:60,actualDomCount:16}}};}
function localRow(index=0){return{pairIndex:index,query,mode:'local',dispatch:{dispatchedAt:100},actualPostRequests:[{query}],allPostRequests:[{url:'native'}],
  bodySettlement:{settled:true},cloneCaptureSettlement:{settled:true},firstDomMatch:{passed:true},firstSourceMatch:true,
  observation:{commonCandidate:{observedAt:110},commonFramesReadyAt:140,last:{devicePixelRatio:1},firstUseful:{cardCount:3,fromDispatchMs:160,requestId:`new-${index}`,actualDomCount:16},
    fullSix:{fromDispatchMs:180,requestId:`new-${index}`}},localFullValidated:true,
  responseEvidence:{selectedSource:'in-page-actual-response-clone',pathsAgree:null,
    events:['rank','firstThree','complete'].map(type=>({type,requestId:`new-${index}`,copySource:source,
      requiredFields:type==='firstThree'?10:20,acceptedFields:type==='firstThree'?10:20,withheldFields:0}))}};}

test('fixed20pairs cover4queries five times with10AB/10BA and both orders for every query',()=>{
  const pairs=fixedPairs();assert.equal(pairs.length,20);assert.equal(new Set(pairs.map(p=>p.pairIndex)).size,20);
  assert.equal(pairs.filter(p=>p.order[0]==='generic').length,10);
  for(const q of new Set(pairs.map(p=>p.query))){const rows=pairs.filter(p=>p.query===q);assert.equal(rows.length,5);
    assert.equal(new Set(rows.map(p=>p.order[0])).size,2);assert.notEqual(q,BOOTSTRAP_QUERY);}
  const plan=fixedPlan();assert.equal(plan.fixedArms,40);assert.equal(plan.nativeRequestsPlanned.total,40);
  assert.match(plan.browserCache,/not cold-navigation/i);assert.equal(plan.deviceScaleFactor,1);
});

test('predeclared nearest-rank p50/p95 retain first outliers and never summarize missing samples',()=>{
  const values=Array.from({length:20},(_,i)=>i+1);assert.equal(nearestRank(values,.5),10);assert.equal(nearestRank(values,.95),19);
  values[0]=9000;const d=distribution(values);assert.equal(d.maximumMs,9000);assert.equal(d.rawMs[0],9000);
  assert.equal(d.p95Ms,20);const missing=[...values];missing[3]=null;
  assert.equal(distribution(missing).p50Ms,null);assert.equal(distribution(missing).missing,1);
});

test('fixed summaries report20raw slots perarm and paired increments without dropping bad rows',()=>{
  const rows=fixedPairs().flatMap(p=>{const a=genericRow(p.pairIndex),b=localRow(p.pairIndex);
    a.protocol=protocolForArm(a);b.protocol=protocolForArm(b);return[a,b];});
  const all=summarizePairs(rows);assert.equal(all.genericFirstUseful.valid,20);assert.equal(all.localFirstUseful.valid,20);
  assert.equal(all.pairedIncrementalLocalMinusGeneric.p50Ms,100);assert.equal(all.localFullSix.valid,20);
  rows[0].protocol.passed=false;const bad=summarizePairs(rows);
  assert.equal(bad.genericFirstUseful.p95Ms,null);assert.equal(bad.pairedIncrementalLocalMinusGeneric.p50Ms,null);
  assert.equal(bad.genericFirstUseful.rawMs.length,20);assert.equal(bad.genericFirstUseful.rawMs[0],null);
  assert.equal(summarizePairs([]).unattempted.length,40);
});

test('generic current hash alone cannot pass; actual current DOM and twoRAF are required',()=>{
  const o=observer();assert.equal(o.result.dispatchedAt,100);assert.match(o.location.hash,/bedside/);
  o.tick();assert.equal(o.window.__pairedIncremental.genericCommit,null);assert.equal(o.window.__pairedIncremental.firstUseful,null);
  o.current();o.tick();assert.ok(o.window.__pairedIncremental.genericCommit);
  assert.equal(o.window.__pairedIncremental.firstUseful,null);
  for(let i=0;i<5;i++)o.tick();
  const state=o.window.__pairedIncremental;assert.ok(state.genericFramesReadyAt>=state.genericCommit.observedAt);
  assert.equal(state.firstUseful.cardCount,3);assert.equal(state.firstUseful.actualDomCount,16);
  assert.ok(state.firstUseful.fromDispatchMs>0);
});

test('generic uses actualmin3 count and records genuine zero separately without fabricated fields',()=>{
  for(const count of [0,1,2,16]){const o=observer('generic',count);o.current();for(let i=0;i<8;i++)o.tick();
    const state=o.window.__pairedIncremental;
    if(count===0){assert.ok(state.zeroResult);assert.equal(state.firstUseful,null);}
    else assert.equal(state.firstUseful.cardCount,Math.min(3,count));
    assert.equal(Object.hasOwn(state,'requiredFields'),false);}
});

test('generic protocol requires zeroPOST, queryDOM proof and finite frame proof, not native model fields',()=>{
  const good=genericRow();assert.equal(protocolForArm(good).passed,true);
  for(const mutate of [r=>{r.actualPostRequests=[{query}];},r=>{r.allPostRequests=[{url:'unexpected'}];},
    r=>{r.observation.genericCommit.input='old';},r=>{r.observation.genericFramesReadyAt=undefined;},
    r=>{r.observation.genericCommit.observedAt=90;},r=>{r.observation.last.devicePixelRatio=2;}]){
    const bad=structuredClone(good);mutate(bad);assert.equal(protocolForArm(bad).passed,false);}
});

test('local observer rejects oldrequests; common3visible and separate6decoded are different predicates',()=>{
  const o=observer('local');o.window.__candidateProbe.stages.firstThree={mirror:mirror('firstThree','old',90)};
  o.tick();assert.equal(o.window.__pairedIncremental.firstUseful,null);
  o.current();o.marks.push({name:'fieldwork.local-model.request.2.firstThree.react-commit',startTime:110,detail:mirror('firstThree').detail});
  o.window.__candidateProbe.stages.firstThree={mirror:mirror('firstThree')};o.tick();
  assert.equal(o.window.__pairedIncremental.firstUseful,null);
  for(let i=0;i<7;i++)o.tick();
  assert.equal(o.window.__pairedIncremental.firstUseful.cardCount,3);
  assert.equal(o.window.__pairedIncremental.fullSix,null);
  o.window.__candidateProbe.stages.complete={mirror:mirror('complete')};o.tick();
  assert.ok(o.window.__pairedIncremental.fullSix);assert.equal(o.window.__pairedIncremental.fullSix.allSixVisible,false);
});

test('decodefailure or clippedcard cannot become common useful content',()=>{
  for(const corrupt of [s=>{s.cards[0].image.decodeStatus='error';},s=>{s.cards[1].rect.fullyWithinViewport=false;}]){
    const o=observer();corrupt(o.window.__candidateProbe.last);o.current();for(let i=0;i<8;i++)o.tick();
    assert.equal(o.window.__pairedIncremental.firstUseful,null);}
});

test('local protocol requires one current fresh neural POST, matching fields, IDs and settled rawbytes',()=>{
  const good=localRow();assert.equal(protocolForArm(good).passed,true);
  assert.equal(protocolForArm(good,new Set(['new-0'])).passed,false);
  for(const mutate of [r=>{r.actualPostRequests.push({query});},r=>{r.actualPostRequests[0].query='old';},
    r=>{r.bodySettlement.settled=false;},r=>{r.cloneCaptureSettlement.settled=false;},
    r=>{r.responseEvidence.pathsAgree=false;},r=>{r.firstDomMatch.passed=false;},
    r=>{r.responseEvidence.events[1].acceptedFields=9;},r=>{r.responseEvidence.events[1].copySource={kind:'generated-earlier',generatedEarlier:true,freshPerRequestGeneration:false};}]){
    const bad=structuredClone(good);mutate(bad);assert.equal(protocolForArm(bad).passed,false);}
});

test('explicit native zero uses actual rank/complete bytes without fake firstThree or generatedfields',()=>{
  const events=[{type:'rank',requestId:'zero'},{type:'complete',requestId:'zero',requiredFields:0,sections:[],
    copySource:{kind:'no-generation',generatedEarlier:false,freshPerRequestGeneration:false}}];
  const bytes=Buffer.from(events.map(e=>JSON.stringify(e)).join('\n'));
  const receipt=parseResponseBytes(bytes);const evidence=selectArmEvidence([], [receipt],'zero',true);
  assert.equal(evidence.selectedSource,'in-page-actual-response-clone');assert.equal(evidence.events.length,2);
  const base=localRow();base.observation.firstUseful=null;base.observation.zeroResult={requestId:'zero',actualDomCount:0};
  base.responseEvidence=evidence;assert.equal(protocolForArm(base).passed,true);
  const mismatch=selectArmEvidence([receipt],[parseResponseBytes(Buffer.concat([bytes,Buffer.from('\n')]))],'zero',true);
  assert.equal(mismatch.pathsAgree,false);assert.equal(mismatch.selectedSource,null);
});

test('both modes start identical twoRAF only after decoded full-visible current DOM; local mirror overlaps',()=>{
  for(const mode of ['generic','local']){
    const o=observer(mode);o.current();
    if(mode==='local')o.marks.push({name:'fieldwork.local-model.request.2.firstThree.react-commit',startTime:110,detail:mirror('firstThree').detail});
    o.window.__candidateProbe.last.cards[0].image.decodeStatus='pending';
    for(let i=0;i<6;i++)o.tick();assert.equal(o.window.__pairedIncremental.commonCandidate,null);
    o.window.__candidateProbe.last.cards[0].image.decodeStatus='decoded';o.tick();
    const candidate=o.window.__pairedIncremental.commonCandidate;assert.ok(candidate);
    assert.equal(o.window.__pairedIncremental.commonFramesReadyAt,null);
    for(let i=0;i<6;i++)o.tick();
    const ready=o.window.__pairedIncremental.commonFramesReadyAt;assert.ok(ready>candidate.observedAt);
    if(mode==='local'){
      assert.equal(o.window.__pairedIncremental.firstUseful,null);
      o.window.__candidateProbe.stages.firstThree={mirror:mirror('firstThree')};o.tick();
    }
    assert.ok(o.window.__pairedIncremental.firstUseful);
    assert.equal(o.window.__pairedIncremental.firstUseful.commonFramesReadyAt,ready);
    assert.equal(o.window.__pairedIncremental.commonFramesReadyAt,ready);
  }
});
