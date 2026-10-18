import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { spawnSync } from 'node:child_process';
import { fixedSchedule,plan,fixedDistribution,summary,parseArgs,installPersonaFetchCapture,installPersonaReadiness,
  verifyFacts,imageFacts,domPriceFacts,verifyProtocol,selectPersonaEvidence,loadFacts,ENDPOINT } from './store_persona_browser_probe_v1.mjs';
import { parseResponseBytes } from './store_candidate_collection_probe_v1.mjs';

const {facts}=await loadFacts();
const source={kind:'fresh-parallel-head',generatedEarlier:false,freshPerRequestGeneration:true};
const request={query:'Browse the store',persona:{background:['Public reading habit'],colorPreferences:['Sage']},publicHistory:[]};
function events(){const cards=facts.slice(0,6).map(p=>({productId:p.id,variantId:p.variants[0].id,originalName:p.name,itemPrice:p.price,
  selectedColor:p.variants[0].colorName,title:`${p.name} title`,badge:'Actual fixture badge',bodyCopy:'Actual fixture copy'}));
 return[{type:'rank',requestId:'new',productIds:facts.map(p=>p.id),selectedProductIds:cards.map(c=>c.productId),shortfall:0,
  variants:Object.fromEntries(cards.map(c=>[c.productId,c.variantId])),provenance:{featureEncoding:{queryFeatureCacheHit:false,contextFeatureCacheHit:true,queryNativeForwards:1,contextNativeForwards:0}}},
  ...['firstThree','complete'].map((type,i)=>({type,requestId:'new',sections:Array.from({length:i+1},(_,j)=>({heading:'Actual fixture heading',cards:cards.slice(j*3,j*3+3)})),
   requiredFields:i?20:10,acceptedFields:i?20:10,withheldFields:0,copySource:source}))];}
function geometry(){return{left:0,right:100,top:0,bottom:100,fullyWithinViewport:true};}
function snapshot(){return{domCardCount:16,scrollY:500,cards:facts.slice(0,6).map((p,i)=>({originalName:p.name,title:`${p.name} title`,selectedColor:p.variants[0].colorName,
 displayPrice:`$${p.price}`,rect:geometry(),titleRect:geometry(),statusParagraphs:[{text:'Actual fixture badge',rect:geometry()},{text:'Actual fixture copy',rect:geometry()}],
 image:{src:`http://127.0.0.1:4313${p.variants[0].imageUrls[0]}`,complete:true,naturalWidth:100,decodeStatus:'decoded',clippedPrimaryRect:geometry(),alt:p.variants[0].imageAlts[0]}}))};}
function observer(settings={navigation:true,query:'Browse the store'}){
 let now=100,popup=false,triggerRect={left:1000,right:1100,top:900,bottom:950};const queue=[],listeners={};const mirrors=[];const snap=snapshot();
 const window={devicePixelRatio:1,__candidateProbe:{last:snap}};
 const document={addEventListener:(name,fn)=>listeners[name]=fn,documentElement:{getAttribute:()=>JSON.stringify(mirrors)},
  querySelectorAll:()=>snap.cards.map(c=>({querySelector:selector=>selector==='.product-heading > span'?{textContent:c.displayPrice}:selector==='.add-button'?(c.addButtonRect?{getBoundingClientRect:()=>c.addButtonRect}:null):{getAttribute:name=>name==='alt'?c.image.alt:c.image.src}})),
  querySelector:selector=>selector==='.person-popover'?(popup?{getClientRects:()=>[{}]}:null):{getBoundingClientRect:()=>triggerRect},
  getElementById:()=>({value:''})};
 vm.runInNewContext(`(${installPersonaReadiness.toString()})(${JSON.stringify(settings)})`,
  {window,document,location:{hash:'#collection'},scrollY:500,URLSearchParams,performance:{now:()=>now},requestAnimationFrame:fn=>queue.push(fn)});
 return{window,mirrors,listeners,snap,popup(value){popup=value;},trigger(value){triggerRect=value;},tick(){now+=10;queue.shift()?.();},fresh(){
  for(const event of events())mirrors.push({startTime:110,duration:20,detail:{stage:event.type,requestId:'new',requiredFields:event.requiredFields,acceptedFields:event.acceptedFields,withheldFields:event.withheldFields}});
 }};
}
function goodRow(){const ev=events();return{query:request.query,actualPostRequests:[{payload:request}],bodySettlement:{settled:true},cloneCaptureSettlement:{settled:true},
 responseEvidence:{selectedSource:'in-page-actual-response-clone',pathsAgree:null,events:ev},sourceFacts:verifyFacts(ev,facts),domPrices:domPriceFacts(snapshot(),ev[2],facts),
 observation:{last:{devicePixelRatio:1,input:''},stages:Object.fromEntries(['rank','firstThree','complete'].map(stage=>[stage,{mirror:{detail:{requestId:'new'}}}]))}};}

test('fixed13 schedule never counts same-person no-op and isolates10switches/query/reload',()=>{
 const schedule=fixedSchedule();assert.equal(schedule.length,13);assert.equal(schedule[0].navigation,'goto');
 const switches=schedule.filter(s=>s.kind==='loaded-profile-switch');assert.equal(switches.length,10);assert.equal(switches.at(-1).personId,'bob');
 for(let i=1;i<=10;i++)assert.notEqual(schedule[i].personId,schedule[i-1].personId);
 assert.equal(schedule[11].query,'work bag');assert.equal(schedule[12].query,'work bag');assert.equal(schedule[12].navigation,'reload');
 assert.match(plan().cache,/may already be precomputed/);assert.match(plan().visibility,/Initial viewport/);
});

test('plan is CPUdefault and --run requires new explicit output, rejects unknown/missing arguments',async()=>{
 assert.equal(parseArgs([]).run,false);assert.throws(()=>parseArgs(['--run']),/unique --out/);
 assert.throws(()=>parseArgs(['--run','--out']),/Missing/);assert.throws(()=>parseArgs(['--gpu']),/Unknown/);
 const result=spawnSync(process.execPath,['experiments/ranker/pretrained/store_persona_browser_probe_v1.mjs'],{cwd:new URL('../../../',import.meta.url),encoding:'utf8'});
 assert.equal(result.status,0);assert.equal(JSON.parse(result.stdout).fixedRequests,13);
});

test('completefixed10 nearest-rank p50/p95 retain outlier and missing slots',()=>{
 const raw=[999,2,3,4,5,6,7,8,9,10];assert.equal(fixedDistribution(raw).p50Ms,6);assert.equal(fixedDistribution(raw).p95Ms,999);
 const missing=[...raw];missing[1]=null;assert.equal(fixedDistribution(missing).p50Ms,null);
 const rows=fixedSchedule().filter(r=>r.kind==='loaded-profile-switch').map(r=>({...r,protocol:{passed:true},usefulContract:{firstThree:true,fullSix:true},timings:{firstThree:{fromNativeFetchMs:r.index},fullSix:{fromNativeFetchMs:r.index+1}}}));
 assert.equal(summary(rows).loadedSwitchFirst3.valid,10);rows[2].protocol.passed=false;
 assert.equal(summary(rows).loadedSwitchFirst3.p95Ms,null);assert.equal(summary(rows).loadedSwitchFirst3.rawMs[2],null);
 assert.equal(summary([]).unattempted.length,13);
});

test('passive8767 clone preserves original fetch args/Response and ignores8766 without extraPOST',async()=>{
 const calls=[];const originalResponse=new Response('real bytes');const window={fetch:async(...args)=>{calls.push(args);return originalResponse;}};
 vm.runInNewContext(`(${installPersonaFetchCapture.toString()})()`,{window,URL,location:{href:'http://127.0.0.1:4313/'},performance:{now:()=>1},Uint8Array});
 const options={method:'POST',body:JSON.stringify(request)};
 assert.equal(await window.fetch(ENDPOINT,options),originalResponse);await new Promise(resolve=>setTimeout(resolve,5));
 assert.equal(calls.length,1);assert.equal(calls[0][1],options);assert.equal(window.__personaFetch.requests.length,1);
 assert.equal(Buffer.from(window.__personaFetch.requests[0].bodyBytes).toString(),'real bytes');
 await window.fetch('http://127.0.0.1:8766/personalize',options);assert.equal(window.__personaFetch.requests.length,1);assert.equal(calls.length,2);
});

test('geometry observer separates initial visible3, full6decoded and popup occlusion',()=>{
 const o=observer();o.fresh();o.popup(true);o.tick();const state=o.window.__personaReadiness;
 assert.equal(state.firstThree,null);assert.ok(state.fullSix);assert.equal(state.last.actualUsefulCards,0);
 o.popup(false);o.tick();assert.equal(state.firstThree.cardCount,3);assert.equal(state.last.actualUsefulCards,6);
 const hidden=observer();hidden.fresh();for(const card of hidden.snap.cards)card.rect.fullyWithinViewport=false;hidden.tick();
 assert.equal(hidden.window.__personaReadiness.firstThree,null);assert.ok(hidden.window.__personaReadiness.fullSix);
 assert.equal(hidden.window.__personaReadiness.last.actualUsefulCards,0);
});

test('loaded profile clock starts at real option click; stale mirrors and decode failures cannot pass',()=>{
 const o=observer({navigation:false,displayName:'Maya',query:'Browse the store'});o.fresh();o.tick();assert.equal(o.window.__personaReadiness.dispatchedAt,null);
 o.listeners.click({target:{closest:()=>({textContent:'Maya Reading at home'})}});assert.ok(o.window.__personaReadiness.dispatchedAt>0);
 o.mirrors.forEach(m=>m.startTime=1);o.tick();assert.equal(o.window.__personaReadiness.stages.complete,undefined);
 o.mirrors.forEach(m=>m.startTime=200);o.snap.cards[0].image.decodeStatus='error';o.tick();
 assert.equal(o.window.__personaReadiness.firstThree,null);assert.equal(o.window.__personaReadiness.fullSix,null);
});

test('every canonical copy name/price/color/variant/order and20acceptedfields is checked',()=>{
 assert.equal(facts.length,16);assert.equal(facts.flatMap(p=>p.variants).length,48);
 const ev=events();assert.equal(verifyFacts(ev,facts).passed,true);
 assert.equal(verifyFacts(ev,facts,{itemBudget:1}).passed,false);
 assert.equal(verifyFacts(ev,facts,{hardRequirements:[{attribute:'color',value:'impossible'}]}).passed,false);
 for(const mutate of [e=>{e[2].sections[0].cards[0].itemPrice=999;},e=>{e[2].sections[0].cards[0].selectedColor='invented';},
  e=>{e[2].acceptedFields=19;},e=>{e[2].sections[0].cards.reverse();},e=>{e[0].shortfall=1;}]){
  const bad=structuredClone(ev);mutate(bad);assert.equal(verifyFacts(bad,facts).passed,false);}
 const snap=snapshot();assert.equal(imageFacts(snap,ev[2],facts).passed,true);
 snap.cards[0].image.src='http://remote.invalid'+facts[0].variants[0].imageUrls[0];assert.equal(imageFacts(snap,ev[2],facts).passed,false);
});

test('actualprotocol requires exactpubliccontext, freshID, nativecache receipts and agreed settledbytes',()=>{
 const row=goodRow();assert.equal(verifyProtocol(row,request).passed,true);
 for(const mutate of [r=>{r.actualPostRequests[0].payload={query:'different'};},r=>{r.actualPostRequests.push({payload:request});},
  r=>{r.bodySettlement.settled=false;},r=>{r.responseEvidence.pathsAgree=false;},r=>{r.observation.stages.complete.mirror.detail.requestId='old';},
  r=>{r.responseEvidence.events[0].provenance.featureEncoding={};},r=>{r.responseEvidence.events[2].copySource={kind:'generated-earlier',generatedEarlier:true,freshPerRequestGeneration:false};}]){
  const bad=structuredClone(row);mutate(bad);assert.equal(verifyProtocol(bad,request).passed,false);}
 assert.equal(verifyProtocol(row,request,new Set(['new'])).passed,false);
});

test('actualzero byte receipt needs no fake firstThree; dual raw paths must agree by SHA',()=>{
 const zero=[{type:'rank',requestId:'zero',selectedProductIds:[]},{type:'complete',requestId:'zero',sections:[],requiredFields:0,copySource:{kind:'no-generation'}}];
 const bytes=Buffer.from(zero.map(e=>JSON.stringify(e)).join('\n')),parsed=parseResponseBytes(bytes);
 const evidence=selectPersonaEvidence([], [parsed],'zero');assert.equal(evidence.events.length,2);assert.ok(evidence.selectedSource);
 const mismatch=selectPersonaEvidence([parsed],[parseResponseBytes(Buffer.concat([bytes,Buffer.from('\n')]))],'zero');
 assert.equal(mismatch.pathsAgree,false);assert.equal(mismatch.selectedSource,null);
});


test('visible DOM price must match both canonical catalog and actual payload; missing price cannot pass',()=>{
 const complete=events()[2],snap=snapshot();
 const valid=imageFacts(snap,complete,facts);assert.equal(valid.passed,true);
 assert.equal(valid.checks[0].visiblePrice,`$${facts[0].price}`);
 assert.equal(valid.checks[0].canonicalPrice,complete.sections[0].cards[0].itemPrice);
 snap.cards[0].displayPrice='$999';const wrongDom=imageFacts(snap,complete,facts);
 assert.equal(wrongDom.passed,false);assert.equal(wrongDom.checks[0].priceMatches,false);
 const row=goodRow();row.domPrices=domPriceFacts(snap,complete,facts);assert.equal(verifyProtocol(row,request).passed,false);
 assert.equal(wrongDom.checks[0].payloadItemPrice,facts[0].price);
 snap.cards[0].displayPrice=undefined;assert.equal(imageFacts(snap,complete,facts).passed,false);
 snap.cards[0].displayPrice=`$${facts[0].price}`;
 const wrongPayload=structuredClone(complete);wrongPayload.sections[0].cards[0].itemPrice=999;
 assert.equal(imageFacts(snap,wrongPayload,facts).passed,false);
 const o=observer();o.fresh();o.tick();
 assert.equal(o.window.__personaReadiness.firstThree.snapshot.cards[0].displayPrice,`$${facts[0].price}`);
});


test('closed popup cannot hide fixed-trigger overlap: third-card intersection blocks useful3',()=>{
 const o=observer();o.fresh();
 const third=o.snap.cards[2];const thirdRect={...geometry(),left:220,right:320};
 third.rect=thirdRect;third.titleRect=thirdRect;third.image.clippedPrimaryRect=thirdRect;
 third.statusParagraphs.forEach(p=>p.rect=thirdRect);
 third.addButtonRect={left:220,right:320,top:70,bottom:100};
 o.trigger({left:240,right:280,top:80,bottom:95});
 for(let i=0;i<6;i++)o.tick();
 const state=o.window.__personaReadiness;
 assert.equal(state.last.popupVisible,false);assert.equal(state.firstThree,null);assert.ok(state.fullSix);
 const receipt=state.last.snapshot.cards[2].switcherOcclusion;
 assert.equal(receipt.overlaps.card,true);assert.equal(receipt.overlaps.title,true);
 assert.equal(receipt.overlaps.primaryImage,true);assert.deepEqual([...receipt.overlaps.body],[true,true]);
 assert.equal(receipt.overlaps.addButton,true);assert.equal(receipt.noObscuringOverlap,false);
 assert.equal(state.last.snapshot.cards[0].switcherOcclusion.noObscuringOverlap,true);
 assert.equal(state.last.switcherRect.left,240);
 const clear=observer();clear.fresh();clear.tick();assert.ok(clear.window.__personaReadiness.firstThree);
});

test('strict whole-card space and an available Add button also reject overlap independently',()=>{
 for(const part of ['card-space','add-button']){
  const o=observer();o.fresh();const third=o.snap.cards[2];
  third.rect={...geometry(),left:220,right:320};
  if(part==='card-space')o.trigger({left:240,right:280,top:80,bottom:95});
  else{third.addButtonRect={left:220,right:320,top:110,bottom:140};o.trigger({left:240,right:280,top:115,bottom:130});}
  o.tick();const state=o.window.__personaReadiness,receipt=state.last.snapshot.cards[2].switcherOcclusion;
  assert.equal(receipt.overlaps.title,false);assert.equal(receipt.overlaps.primaryImage,false);
  assert.equal(receipt.overlaps.card,part==='card-space');
  assert.equal(receipt.overlaps.addButton,part==='add-button'?true:null);
  assert.equal(receipt.noObscuringOverlap,false);assert.equal(state.firstThree,null);
 }
});
