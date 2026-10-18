import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
import {spawnSync} from 'node:child_process';
import {plan,parseArgs,fixedSchedule,fixedDistribution,summary,adapterReceipt,cacheReceipts,assertSelection,assertHealth,
 renderedGeneratedFields,assertWarmStartupSummary,settleCurrentFunctionalCaptures,assessRapidArm,assessCartArm,installCheckoutSwitchObservation,verifyProfilePanel,installPersonaFetchCapture,loadFacts,verifyFacts,domPriceFacts,imageFacts,verifyProtocol,installPersonaReadiness,ENDPOINT,INTEGRATION_SHA} from './store_persona_warm_browser_probe_v1.mjs';
import {installPersonaReadiness as frozenV1Observer} from './store_persona_browser_probe_v1.mjs';
const sha=x=>createHash('sha256').update(x).digest('hex');
const {facts}=await loadFacts();
const request={query:'Browse the store',persona:{background:['Public need'],colorPreferences:['Sage']},publicHistory:[]};
function rankReceipt(){return{queryFeature:{cacheHit:false,nativeBackboneForwardCalls:1,forceFreshQuery:true},colorFeature:{cacheHit:true,nativeBackboneForwardCalls:0},
 learnedQuerySpecificProbability:.4,phraseWhitelistUsed:false,failedOldProductRankingUsed:false,rankAdapterAndCopyAdapterMixed:false,hiddenProfileInputs:false,
 rankDecision:{cacheHit:true,kind:'public-XHY-feature-reuse',nativeBodyForwards:0,nativeLanguageModelProjectionCalls:1,gradientEvaluationCalls:0,
 freshLearnedDecision:true,copyAdapterParticipated:false,generatedContinuationTokens:0,optimizerUpdates:0,targetWeightsProvided:false,promptTokenIdsSha256:'a'.repeat(64)}};}
function events(){const cards=facts.slice(0,6).map(p=>({productId:p.id,variantId:p.variants[0].id,originalName:p.name,itemPrice:p.price,selectedColor:p.variants[0].colorName,
 title:`${p.name} fixture title`,badge:'Fixture badge',bodyCopy:'Fixture copy'}));
 const provenance={hiddenProfileInputs:false,personaRanking:rankReceipt()};
 return[{type:'rank',requestId:'fresh',productIds:facts.map(p=>p.id),selectedProductIds:cards.map(c=>c.productId),shortfall:0,
 variants:Object.fromEntries(cards.map(c=>[c.productId,c.variantId])),provenance},...['firstThree','complete'].map((type,i)=>({type,requestId:'fresh',provenance,
 sections:Array.from({length:i+1},(_,j)=>({heading:'Fixture heading',cards:cards.slice(j*3,j*3+3)})),requiredFields:i?20:10,acceptedFields:i?20:10,withheldFields:0,
 copySource:{kind:'fresh-parallel-head',generatedEarlier:false,freshPerRequestGeneration:true}}))];}
function row(){const ev=events(),snapshot={cards:facts.slice(0,6).map(p=>({displayPrice:`$${p.price}`}))};return{query:request.query,actualPostRequests:[{payload:request}],
 bodySettlement:{settled:true},cloneCaptureSettlement:{settled:true},responseEvidence:{selectedSource:'actual-clone',pathsAgree:null,events:ev},
 sourceFacts:verifyFacts(ev,facts),domPrices:domPriceFacts(snapshot,ev[2],facts),
 observation:{last:{devicePixelRatio:1,input:''},stages:Object.fromEntries(['rank','firstThree','complete'].map(stage=>[stage,{mirror:{detail:{requestId:'fresh'}}}]))}};}
function healthFixture(){const expected={adapterSha256:'adapter',adapterManifestSha256:'manifest',gateSha256:'gate',gateSourceSha256:'gate-source',gateTRAINManifestSha256:'train',
 colorHeadSha256:'color',v5HeadSha256:'v5',copyHeadSha256:'copy',rankExportSha256:'export'};
 const startup={persona:{kind:'persona-adapter-engine.v2',checkpointSha256:'adapter',servingManifestSha256:'manifest',
 queryIntent:{checkpointSha256:'gate',sourceSha256:'gate-source',sourceManifestSha256:'train'},color:{checkpointSha256:'color',failedOldProductOrderingUsed:false}},
 rankCheckpointSha256:'v5',copyCheckpointSha256:'56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8',copyCandidate:{checkpointSha256:'copy'},
 rankSessionProvenance:{rankHead:{servingExportSha256:'export'}},sourceTensorsHashed:927,memory:{processBudgetBytes:7*2**30},imageDeliveryManifestSha256:INTEGRATION_SHA};
 Object.assign(startup.persona,{measurementEngine:'persona-adapter-engine.v3-measurement',forceFreshQuery:true,wholePromptRankCacheIncludesQuery:true,noResponseWarmupOrTextCache:true,featurePreencoding:{version:'persona-adapter-runtime.v3',enabled:true,rows:Array(10).fill({}),nativeBackboneForwardCalls:10,encodingMs:100,rankingWarmResponses:0,copyWarmResponses:0,responseWordsCached:false,rankWholePromptFeaturesPrecomputedHere:false}});
 expected.startup=structuredClone(startup);return{expected,health:{ready:true,status:'ready',startup}};}
function goodRapid(){return{actualPostRequests:[{},{}],exactMiddleAndFinalInputs:true,middleRequest:{abortObserved:true,statusAtAbort:'pending'},finalDispatchedAt:200,
 final:{protocol:{passed:true,requestId:'final-A'},domVariantImages:{passed:true},adapterHttpAudit:{protocolPassed:true,qualityPassed:true}},observation:{observations:[{at:250,cards:Array.from({length:3},()=>({originalName:'Canonical item',title:'Canonical item'}))}],newMirrors:[{completedAt:100,detail:{requestId:'old-A'}},{completedAt:300,detail:{requestId:'final-A'}}]}};}
function goodCart(){return{actualPostRequests:[{},{}],exactMayaAndBobInputs:true,addedCanonicalItem:{productId:'mug',variantId:'mug-sage',originalName:'Canonical source item',selectedColor:'Sage',itemPrice:24},
 bobCartBefore:{count:1,text:'Canonical source item Sage',lines:[{price:'$24'}]},mayaCart:{count:0,text:'Your cart is empty.',checkoutReceipt:null},bobRestored:{count:1,checkoutReceipt:null},
 checkoutRequests:[{}],checkoutResponses:[{status:202,body:{receipt:{realPurchase:false,items:[{productId:'mug',variantId:'mug-sage',unitPrice:24,quantity:1}]}}}],checkoutPendingAtSwitch:false,
 finals:[{protocol:{passed:true},domVariantImages:{passed:true},adapterHttpAudit:{protocolPassed:true,qualityPassed:true}},{protocol:{passed:true},domVariantImages:{passed:true},adapterHttpAudit:{protocolPassed:true,qualityPassed:true}}]};}

test('v1 exact bytes preserved; v2 CPU plan retains fixed13 and separate functional arms',async()=>{
 assert.equal(sha(await readFile(new URL('./store_persona_browser_probe_v1.mjs',import.meta.url))),'8a15441a7e0f5c94e241c383d341217645cc73cf1df2107b1e8135ccd4581f63');
 assert.equal(plan().version,'store-persona-warm-browser-probe.v1');assert.equal(fixedSchedule().length,13);assert.equal(fixedSchedule()[12].navigation,'reload');
 assert.match(plan().functional,/excluded/);assert.match(plan().sourceBindingLimit,/encodingMeasurement text SHA/);
 const r=spawnSync(process.execPath,['experiments/ranker/pretrained/store_persona_warm_browser_probe_v1.mjs'],{cwd:new URL('../../../',import.meta.url),encoding:'utf8'});
 assert.equal(r.status,0);assert.equal(JSON.parse(r.stdout).fixedRequests,13);
});
test('real run requires exact explicit candidate and output; accepted/default flags rejected',()=>{
 assert.equal(parseArgs([]).run,false);assert.throws(()=>parseArgs(['--run']),/unique/);
 assert.throws(()=>parseArgs(['--run','--out','new']),/development-demo/);
 assert.throws(()=>parseArgs(['--accepted-profile','whatever']),/Unknown/);
 assert.throws(()=>parseArgs(['--run','--out','new','--candidate-receipt','r','--candidate-sha256','bad']),/exact/);
 assert.equal(parseArgs(['--run','--out','new','--candidate-receipt','r','--candidate-sha256','a'.repeat(64)]).run,true);
});
test('serving-pending admission rejects accepted state or incomplete selection',()=>{
 const selection={mode:'technical-development-demo',developmentDemo:true,qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,overallPromotionPass:false,acceptanceStateWritten:false,expected:{startup:{}},pins:{source:'sha'}};
 assert.equal(assertSelection(selection),true);
 for(const mutate of [x=>{x.mode='accepted';},x=>{x.overallPromotionPass=true;},x=>{x.acceptanceStateWritten=true;},x=>{x.pins={};},x=>{x.qualityGatePassed=true;},x=>{x.modelPromotion=true;},x=>{x.acceptedProfile=true;}]){
 const bad=structuredClone(selection);mutate(bad);assert.throws(()=>assertSelection(bad));}
});
test('health binds actual adapter/query/color/v5/copy/927/memory/image lifetime identity',()=>{
 const {health,expected}=healthFixture();assert.equal(assertHealth(health,expected),true);
 for(const mutate of [h=>{h.startup.persona.queryIntent.checkpointSha256='wrong';},h=>{h.startup.sourceTensorsHashed=926;},h=>{h.startup.copyCandidate.checkpointSha256='wrong';},h=>{h.startup.imageDeliveryManifestSha256='wrong';}]){
 const bad=structuredClone(health);mutate(bad);assert.throws(()=>assertHealth(bad,expected));}
 const forged=healthFixture();forged.health.startup.sourceTensorsHashed=forged.expected.startup.sourceTensorsHashed=926;
 assert.throws(()=>assertHealth(forged.health,forged.expected),/927/);
});
test('new actual counters accept XHY but never fabricate old featureEncoding receipt',()=>{
 const ev=events();assert.equal(adapterReceipt(ev).passed,true);assert.equal(verifyProtocol(row(),request).passed,true);
 const cache=cacheReceipts(ev[0],ev[2]);assert.equal(cache.rankDecision.nativeBodyForwards,0);assert.equal(cache.queryFeature.nativeBackboneForwardCalls,1);
 assert.equal(cache.featureEncoding,null); // absent real receipt stays unknown, never a fabricated zero
 for(const mutate of [r=>{delete r.queryFeature;},r=>{r.colorFeature.nativeBackboneForwardCalls=1;},r=>{r.rankDecision.nativeBodyForwards=1;},r=>{r.rankDecision.copyAdapterParticipated=true;},r=>{r.phraseWhitelistUsed=true;}]){
 const bad=events();mutate(bad[0].provenance.personaRanking);assert.equal(adapterReceipt(bad).passed,false);}
});
test('fresh dynamic rank path needs actual suffix/body/projection/gradient receipts',()=>{
 const ev=events(),n=ev[0].provenance.personaRanking.rankDecision;
 Object.assign(n,{cacheHit:false,kind:'fresh-native-last16-rank-adapter',nativeBodyForwards:3,nativeLanguageModelProjectionCalls:2,gradientEvaluationCalls:1,nativeRankSuffixTokens:16});
 assert.equal(adapterReceipt(ev).passed,true);n.gradientEvaluationCalls=0;assert.equal(adapterReceipt(ev).passed,false);
});
test('changed provenance, stale ID, unagreed/unsettled bytes, DOM price errors cannot pass',()=>{
 for(const mutate of [r=>{r.responseEvidence.events[2].provenance=structuredClone(r.responseEvidence.events[2].provenance);r.responseEvidence.events[2].provenance.hiddenProfileInputs=true;},
 r=>{r.observation.stages.complete.mirror.detail.requestId='stale';},r=>{r.responseEvidence.pathsAgree=false;},r=>{r.cloneCaptureSettlement.settled=false;},r=>{r.domPrices.passed=false;},r=>{r.pageErrors=[{message:'real failure'}];}]){
 const bad=row();mutate(bad);assert.equal(verifyProtocol(bad,request).passed,false);}
});
test('passive fetch clone records real AbortSignal without a second POST or changed Response',async()=>{
 const calls=[],controller=new AbortController(),original=new Response('actual fixture bytes');const window={fetch:async(...args)=>{calls.push(args);return original;}};
 vm.runInNewContext(`(${installPersonaFetchCapture.toString()})()`,{window,URL,location:{href:'http://127.0.0.1:4313/'},performance:{now:()=>7},Uint8Array});
 const options={method:'POST',body:JSON.stringify(request),signal:controller.signal};assert.equal(await window.fetch(ENDPOINT,options),original);
 controller.abort();await new Promise(resolve=>setTimeout(resolve,5));const captured=window.__personaFetch.requests[0];
 assert.equal(calls.length,1);assert.equal(calls[0][1],options);assert.equal(captured.abortObserved,true);assert.equal(captured.abortedAt,7);
 assert.equal(Buffer.from(captured.bodyBytes).toString(),'actual fixture bytes');
});
test('rapid arm requires actual pending abort and rejects stale mirror after final A dispatch',()=>{
 const arm=goodRapid();assert.equal(assessRapidArm(arm).passed,true);
 for(const mutate of [a=>{a.middleRequest.statusAtAbort='complete';},a=>{a.middleRequest.abortObserved=false;},a=>{a.actualPostRequests.pop();},
 a=>{a.observation.newMirrors.push({completedAt:201,detail:{requestId:'B'}});},a=>{a.final.adapterHttpAudit.qualityPassed=false;},a=>{a.final.domVariantImages.passed=false;},a=>{a.observation.observations[0].cards[0].title='Stale B generated title';}]){
 const bad=structuredClone(arm);mutate(bad);assert.equal(assessRapidArm(bad).passed,false);}
});
test('cart arm verifies actual per-person baskets and receipt invalidation without inventing race success',()=>{
 const arm=goodCart(),result=assessCartArm(arm);assert.equal(result.passed,true);assert.equal(result.checkoutRaceExercised,false);assert.match(result.checkoutRaceScope,/No live in-flight/);
 for(const mutate of [a=>{a.mayaCart.count=1;},a=>{a.bobRestored.count=0;},a=>{a.bobRestored.checkoutReceipt='old';},a=>{a.checkoutResponses[0].body.receipt.realPurchase=true;},a=>{a.bobCartBefore.text='wrong item';}]){
 const bad=structuredClone(arm);mutate(bad);assert.equal(assessCartArm(bad).passed,false);}
});
test('visible public profile/history checks newest3 type/item/color and summary without labels',()=>{
 const person={summary:'Public summary',request:{persona:{colorPreferences:['Sage']},publicHistory:[
 {eventId:'earlier',eventType:'view',itemName:'Older item',occurredAt:'2026-01-01',selectedColor:'Ivory'},
 {eventId:'later',eventType:'purchase',itemName:'Later item',occurredAt:'2026-01-02',selectedColor:'Sage'}]}};
 const panel={text:'Public summary Favorite colors: Sage',history:['Purchased Later item Sage','Viewed Older item Ivory']};
 assert.equal(verifyProfilePanel(person,panel).passed,true);
 assert.equal(verifyProfilePanel(person,{...panel,history:[...panel.history].reverse()}).passed,false);
 assert.equal(verifyProfilePanel(person,{...panel,text:'different'}).passed,false);
});
test('functional results never enter fixed10 timing denominator; missed slots remain null',()=>{
 const rows=fixedSchedule().filter(r=>r.kind==='loaded-profile-switch').map(r=>({...r,protocol:{passed:true},usefulContract:{firstThree:true,fullSix:true},timings:{firstThree:{fromNativeFetchMs:r.index},fullSix:{fromNativeFetchMs:r.index+1}}}));
 rows.push({kind:'functional-only',index:99,protocol:{passed:true},timings:{firstThree:{fromNativeFetchMs:0}}});
 assert.equal(summary(rows).loadedSwitchFirst3.fixedDenominator,10);assert.equal(summary(rows).loadedSwitchFirst3.p50Ms,5);
 rows[1].usefulContract.firstThree=false;assert.equal(summary(rows).loadedSwitchFirst3.p50Ms,null);
 assert.equal(fixedDistribution([999,2,3,4,5,6,7,8,9,10]).p95Ms,999);
});

test('checkout race receipt reads pending at actual option click, not before opening popover',()=>{
 let listener;const window={};let pending=false;
 const document={addEventListener:(_name,fn)=>listener=fn,removeEventListener:()=>{},querySelector:selector=>selector.includes('primary-button')?{disabled:pending}:{getAttribute:()=>null}};
 vm.runInNewContext(`(${installCheckoutSwitchObservation.toString()})({displayName:'Maya'})`,{window,document,performance:{now:()=>200}});
 pending=true;listener({target:{closest:()=>({textContent:'MMayaPublic role',querySelector:selector=>selector==='strong'?{textContent:'Maya'}:null})}});
 assert.equal(window.__personaCheckoutSwitch.pendingAtActualClick,true);assert.equal(window.__personaCheckoutSwitch.at,200);
 pending=false;listener({target:{closest:()=>({textContent:'MMayaPublic role',querySelector:selector=>selector==='strong'?{textContent:'Maya'}:null})}});
 assert.equal(window.__personaCheckoutSwitch.pendingAtActualClick,false);
});

test('geometry/decode/price/occlusion observer differs only by actual strong-name click matcher',()=>{const old=frozenV1Observer.toString().replace("event.target.closest?.('.person-options button')?.textContent.trim().startsWith(settings.displayName)","event.target.closest?.('.person-options button')?.querySelector('strong')?.textContent?.trim()===settings.displayName");assert.equal(installPersonaReadiness.toString(),old);});

test('strict v2 source remains byte-exact and development plan never claims model acceptance',async()=>{assert.equal(sha(await readFile(new URL('./store_persona_browser_probe_v2.mjs',import.meta.url))),'e8583a9feaa4848522d8cabe91387b2112f683095bcc2426ee7c0a6d0df144d8');for(const key of ['qualityGatePassed','modelPromotion','acceptedProfile'])assert.equal(plan()[key],false);assert.equal(INTEGRATION_SHA,'16acf1713efcd3164be473b6f44c480cac4aa3cdfee760add21877c958aa8beb');});


test('actual public Maya button has avatar prefix: strong-name click starts exact clock',async()=>{
 const people=JSON.parse(await readFile(new URL('./synthetic_history_v1/demo-people.json',import.meta.url),'utf8')).people;
 const maya=people.find(p=>p.id==='maya');assert.equal(maya.displayName,'Maya');
 const listeners={},window={};let now=123;
 const document={addEventListener:(name,fn)=>listeners[name]=fn};
 vm.runInNewContext(`(${installPersonaReadiness.toString()})({navigation:false,displayName:'Maya',query:'Browse the store'})`,
  {window,document,scrollY:0,location:{hash:''},URLSearchParams,performance:{now:()=>now},requestAnimationFrame:()=>{}});
 const actualButton={textContent:`M${maya.displayName}${maya.role}`,querySelector:selector=>selector==='strong'?{textContent:maya.displayName}:null};
 assert.equal(actualButton.textContent.startsWith(maya.displayName),false);
 listeners.click({target:{closest:()=>actualButton}});
 assert.equal(window.__personaReadiness.dispatchedAt,123);assert.equal(window.__personaReadiness.action.kind,'actual-profile-option-click');
 now=150;listeners.click({target:{closest:()=>actualButton}});assert.equal(window.__personaReadiness.dispatchedAt,123);
});

test('similar names, absent strong or non-profile clicks cannot start the profile clock',()=>{
 for(const button of [null,{textContent:'MMaya role',querySelector:()=>null},{textContent:'MMaya role',querySelector:()=>({textContent:'Mayabelle'})}]){
  const listeners={},window={};const document={addEventListener:(name,fn)=>listeners[name]=fn};
  vm.runInNewContext(`(${installPersonaReadiness.toString()})({navigation:false,displayName:'Maya',query:'Browse the store'})`,
   {window,document,scrollY:0,location:{hash:''},URLSearchParams,performance:{now:()=>123},requestAnimationFrame:()=>{}});
  listeners.click({target:{closest:()=>button}});assert.equal(window.__personaReadiness.dispatchedAt,null);
 }
});



test('functional current barrier excludes still-pending deliberately aborted middle CDP body without fabricating its completion',async()=>{
 const never=new Promise(()=>{}),middle={headers:{'x-request-id':'middle'},captureStatus:'pending'},final={headers:{'x-request-id':'final'},captureStatus:'complete'};
 const capture={responses:[middle,final],pending:[never,Promise.resolve()]};let waited=false;
 const result=await settleCurrentFunctionalCaptures(capture,'final',async()=>{waited=true;},5);
 assert.equal(result.bodySettlement.settled,true);assert.equal(result.cloneCaptureSettlement.settled,true);assert.equal(waited,true);
 assert.deepEqual(result.currentCaptureScope.matchingResponseIndices,[1]);assert.equal(result.currentCaptureScope.noncurrentResponses[0].captureStatus,'pending');
 assert.equal(middle.captureStatus,'pending');assert.equal(result.bodySettlement.started,1);
});

test('unresolved current CDP response or clone still fails even when earlier aborted capture is excluded',async()=>{
 const never=new Promise(()=>{}),capture={responses:[{headers:{'x-request-id':'middle'},captureStatus:'pending'},{headers:{'x-request-id':'final'},captureStatus:'pending'}],pending:[never,never]};
 const result=await settleCurrentFunctionalCaptures(capture,'final',async()=>{},5);assert.equal(result.bodySettlement.settled,false);
 const complete={responses:capture.responses,pending:[never,Promise.resolve()]};
 const clone=await settleCurrentFunctionalCaptures(complete,'final',async()=>{throw Error('current clone remains pending');},5);assert.equal(clone.cloneCaptureSettlement.settled,false);
 for(const malformed of [{...complete,pending:[never]}, {...complete,responses:[...complete.responses,{headers:{'x-request-id':'final'}}],pending:[never,Promise.resolve(),Promise.resolve()]}]){
  const bad=await settleCurrentFunctionalCaptures(malformed,'final',async()=>{},5);assert.equal(bad.bodySettlement.settled,false);
 }
});


test('warm policy requires fresh query B1 on every request including same query reload',()=>{
 const ev=events();assert.equal(adapterReceipt(ev).passed,true);
 for(const mutate of [q=>{q.cacheHit=true;q.nativeBackboneForwardCalls=0;},q=>{q.forceFreshQuery=false;}]){
  const bad=structuredClone(ev);mutate(bad[0].provenance.personaRanking.queryFeature);assert.equal(adapterReceipt(bad).passed,false);
 }
 const {health}=healthFixture();const s=assertWarmStartupSummary(health.startup);assert.equal(s.nativeStartupColorEncodes,10);
 health.startup.persona.featurePreencoding.copyWarmResponses=1;assert.throws(()=>assertWarmStartupSummary(health.startup));
});

test('full6 generated20 has exactly19 rendered generated fields, second heading remains backend-only',()=>{
 const ev=events(),complete=ev[2];
 const snapshot={heading:'Fixture heading',cards:complete.sections.flatMap(s=>s.cards).map(c=>({title:c.title,statusParagraphs:[{text:c.originalName},{text:c.badge},{text:c.bodyCopy}]}))};
 const full=renderedGeneratedFields(snapshot,complete);assert.equal(full.passed,true);assert.equal(full.generatedAcceptedFields,20);
 assert.equal(full.expectedRenderedGeneratedFields,19);assert.equal(full.matchedRenderedGeneratedFields,19);assert.deepEqual(full.backendOnlyFields.map(x=>x.field),['S2']);
 const first=renderedGeneratedFields(snapshot,ev[1]);assert.equal(first.expectedRenderedGeneratedFields,10);assert.equal(first.matchedRenderedGeneratedFields,10);
 const second=structuredClone(complete);second.sections[1].heading='Distinct backend-only second heading';assert.equal(renderedGeneratedFields(snapshot,second).passed,true);
 snapshot.heading='Wrong current heading';assert.equal(renderedGeneratedFields(snapshot,complete).passed,false);
 snapshot.heading='Fixture heading';snapshot.cards[0].statusParagraphs=[];assert.equal(renderedGeneratedFields(snapshot,complete).passed,false);
 assert.match(plan().renderedFieldScope,/may be0/);
});

test('developmentV3 and new runtime/engine frozen sources stay preserved by warm browser prep',async()=>{
 for(const [name,expected] of [['store_persona_development_browser_probe_v3.mjs','4cddc34cc0866fab1cce0688e4340eb2f20a6ff9a367c6543795c9e25c1c5a9a'],
 ['store_persona_adapter_runtime_v3.py','6fb9626541a20d5ab9c55b7a1b930e7e57f107ff9dd70ac36493829621ecbc61'],
 ['store_persona_adapter_engine_v3.py','f1a4c0371745a9530f2d47b4bc3be732c718245d41c2c8a8fc61b45e5fc528d3']])assert.equal(sha(await readFile(new URL('./'+name,import.meta.url))),expected);
});
