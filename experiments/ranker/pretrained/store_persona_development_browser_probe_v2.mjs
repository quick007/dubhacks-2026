import { createRequire } from 'node:module';
import { mkdir, readFile, writeFile, access } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join, dirname, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { installObserver, parsedSourceFieldsMatch, matchDomToCopy } from './store_candidate_browser_probe_v1.mjs';
import { parseResponseBytes, selectResponseEvidence } from './store_candidate_collection_probe_v1.mjs';
import { CONTEXT_OPTIONS, settleNavigationCaptures } from './store_candidate_browser_probe_v2.mjs';

const ROOT=resolve(dirname(fileURLToPath(import.meta.url)),'../../..');
export const ORIGIN='http://127.0.0.1:4313';
export const ENDPOINT='http://127.0.0.1:8767/personalize';
export const PUBLIC_PEOPLE='experiments/ranker/pretrained/synthetic_history_v1/demo-people.json';
export const PROFILE_ORDER=Object.freeze(['maya','priya','lena','june','grace','sam','ana','rosa','felix','bob']);
export const TIMEOUT_MS=30000;
export const INTEGRATION_MANIFEST='experiments/ranker/pretrained/store_demo_integration_personalized_adapter_v2/manifest.json';
export const INTEGRATION_SHA='16acf1713efcd3164be473b6f44c480cac4aa3cdfee760add21877c958aa8beb';
export const PYTHON='/Users/lukas/Documents/Projects/dubhacks-2026/.tools/pretrained-venv/bin/python';
export const TOKENIZER='/Users/lukas/Documents/Projects/dubhacks-2026/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const canonical=value=>JSON.stringify(value,(_key,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))):v);
const save=(path,value)=>writeFile(path,JSON.stringify(value,null,2)+'\n',{flag:'wx'});

export function fixedSchedule(){return[
  {index:0,kind:'bare-navigation',personId:'bob',query:'Browse the store',navigation:'goto',preScrolled:false},
  ...PROFILE_ORDER.map((personId,i)=>({index:i+1,kind:'loaded-profile-switch',personId,query:'Browse the store',navigation:null,preScrolled:true})),
  {index:11,kind:'first-work-bag-query',personId:'bob',query:'work bag',navigation:null,preScrolled:true},
  {index:12,kind:'same-query-reload',personId:'bob',query:'work bag',navigation:'reload',preScrolled:false},
];}
export function plan(){return{version:'store-persona-development-browser-probe.v2',origin:ORIGIN,endpoint:ENDPOINT,
  developmentDemo:true,qualityGatePassed:false,modelPromotion:false,acceptedProfile:false,publicPeople:PUBLIC_PEOPLE,schedule:fixedSchedule(),fixedRequests:13,profileDenominator:10,
  viewport:CONTEXT_OPTIONS.viewport,deviceScaleFactor:1,headed:true,timeoutMs:TIMEOUT_MS,bodySettlementMs:5000,
  execution:'CPU plan by default. --run and a NEW --out directory required. Existing installed Chrome and ready explicit technically verified development-demo adapter only; actual quality failed and model promotion remains false; never starts servers or model workers.',
  workload:'One bare URL navigation at actual initial scroll. Then10 different public profile selections in one loaded collection, pre-scroll before each. Profile clock is real option click; actual public summary/recent3 history panel is read then real Escape dismisses popover before useful visibility (this capture/dismiss overhead is included). EndBob, submit work bag via real search form, then reload that route for another genuine POST. No route/fetch/text injection or forced same-hash request.',
  timing:'Native startup and browser launch excluded. Browser actual fetch→endpoint is common model-request clock; action→endpoint and navigation→endpoint separately retained. Existing React/two-rAF mirror plus actual geometry and image complete/decode; no physical paint claim. Passive clone overhead is part of this instrumented workload; body settlement occurs after UI endpoints.',
  cache:'NEW browser context starts HTTP cache cold for bare navigation. Loaded switches/search reuse same browser cache. Reload is HTTP-cache eligible with browser scroll restoration recorded, not a second cold navigation. Native query/color/rank-XHY/catalog-prefix cache state is actual raw provenance, never assumed cold. All10 public context features may already be precomputed; first work-bag means first query in this probe, not guaranteed native cache miss. Reload has identical public payload and a unique request ID; fresh LoRA scoring/learned color variants/neural copy required even on feature cache hits.',
  visibility:'Initial viewport reports actual useful-card count and first3 only if truly fully visible. Loaded first3 requires current request accepted10 fields, three whole cards/titles/copy and decoded primary images, with no open popover or intersecting fixed switcher. The actual .person-trigger rectangle is checked against the whole card bounding rectangle plus title, each copy paragraph, clipped primary image and Add button when present; any intersection rejects useful visibility. This strict whole-card policy also rejects overlap in otherwise unused card space. Full6 requires accepted20 fields and six decoded primary images separately; six-card visibility is never implied. Shortfalls/zero results remain explicit, never padded.',
  facts:'All actual copy fields and DOM strings are retained. Canonical public catalog validates original name, price, chosen variant/color and image-source identity; every accepted-field total and DOM copy string is checked. Each selected card visible DOM price must equal both its canonical price and actual payload itemPrice, with missing prices rejected. This is not a free-text semantic factuality or unseen-personalization-quality evaluation; no private labels or answers are read.',
  candidateAdmission:'Explicit --candidate-receipt plus --candidate-sha256; explicit development admission preserving the actual failed quality audit and passed mechanical evidence; same per-request public/factual/guard analyzer as HTTP probe v2. No accepted profile or default promotion is written. Source/backend pins and startup checked before/after. CPU public prompt tokenization/audit runs after UI/body endpoints, outside latency timers and between measured requests; no native model loaded by probe.',
  functional:'After all13 timing requests pass, separate actual Bob→Maya→Bob cancellation arm (2 model starts) plus Bob add/cart/simulated checkout→Maya empty cart→Bob restored cart arm (2 model starts,1 local simulated checkout). Their raw requests/failures are retained and excluded from fixed10 latency percentiles; no request interception or artificial service delays. If middle request/checkout finishes before switch, report race as not exercised, not a cancellation pass.',
  sourceBindingLimit:'Live base-query feature lacks its own text SHA; actual checked input and pinned runtime bind it. Actual LoRA prompt token SHA is independently recomputed by CPU tokenizer from public fields. Native rank/body/projection/gradient/query/color counters preserved, never derived from timing.',
  summary:'Nearest-rank p50/p95 only for the complete fixed10 loaded-switch denominator. First navigation/search/reload are separate single arms, no percentiles. Null failures/shortfalls retained; no replacements or success-only percentiles. Stop remaining requests on invalid/unsettled response protocol.'};}

export function fixedDistribution(values){const valid=values.filter(Number.isFinite);const complete=values.length===10&&valid.length===10;
 const sorted=[...valid].sort((a,b)=>a-b);return{fixedDenominator:10,valid:valid.length,rawMs:values,p50Ms:complete?sorted[4]:null,p95Ms:complete?sorted[9]:null,estimator:'nearest-rank; fixed10 complete only'};}
export function summary(rows){const switches=fixedSchedule().filter(x=>x.kind==='loaded-profile-switch');
 const values=endpoint=>switches.map(item=>{const r=rows.find(x=>x.index===item.index);return r?.protocol?.passed&&r.usefulContract?.[endpoint]?r.timings?.[endpoint]?.fromNativeFetchMs??null:null;});
 return{fixedRequests:13,recorded:rows.length,unattempted:fixedSchedule().filter(x=>!rows.some(r=>r.index===x.index)),
  protocolPassed:rows.filter(x=>x.protocol?.passed).length,endpointMisses:rows.filter(x=>x.kind==='loaded-profile-switch'&&(!x.usefulContract?.firstThree||!x.usefulContract?.fullSix)).map(x=>({index:x.index,personId:x.personId,imageEndpointMiss:x.imageEndpointMiss??null,endpointContractMiss:x.endpointContractMiss??null,usefulContract:x.usefulContract??null})),loadedSwitchFirst3:fixedDistribution(values('firstThree')),loadedSwitchFull6:fixedDistribution(values('fullSix')),
  separateArms:rows.filter(x=>x.kind!=='loaded-profile-switch').map(x=>({index:x.index,kind:x.kind,protocol:x.protocol,usefulContract:x.usefulContract,
   timings:Object.fromEntries(['firstThree','fullSix'].map(stage=>[stage,x.protocol?.passed&&x.usefulContract?.[stage]?x.timings?.[stage]??null:null])),
   rawTimings:x.timings,initialViewport:x.initialViewport,cache:x.cache,imageEndpointMiss:x.imageEndpointMiss??null,error:x.error??null})),
  failures:rows.filter(x=>x.error||!x.protocol?.passed).map(x=>({index:x.index,error:x.error??null,protocol:x.protocol??null})),scope:'Ten loaded public-profile switches only; no cold/public-context or physical-paint inference'};}

/** Passive actual8767 fetch clone. Original arguments and Response are unchanged. */
export function installPersonaFetchCapture(){
 const original=window.fetch;const capture=window.__personaFetch={requests:[]};
 window.fetch=function(...args){const[input,init]=args;const url=typeof input==='string'?new URL(input,location.href).href:input?.url??String(input);
  const method=String(init?.method??input?.method??'GET').toUpperCase();
  if(url!=='http://127.0.0.1:8767/personalize'||method!=='POST')return Reflect.apply(original,this,args);
  const entry={url,method,requestBody:typeof init?.body==='string'?init.body:null,startedAt:performance.now(),status:'pending',bodyBytes:null,abortObserved:false,abortedAt:null};capture.requests.push(entry);
  const signal=init?.signal??input?.signal;if(signal){entry.abortObserved=signal.aborted;signal.addEventListener('abort',()=>{entry.abortObserved=true;entry.abortedAt=performance.now();entry.statusAtAbort=entry.status;},{once:true});}
  return Reflect.apply(original,this,args).then(response=>{entry.responseStatus=response.status;
   try{response.clone().arrayBuffer().then(buffer=>{entry.bodyBytes=Array.from(new Uint8Array(buffer));entry.finishedAt=performance.now();entry.status='complete';},error=>{entry.error=String(error);entry.status='error';});}
   catch(error){entry.error=String(error);entry.status='error';}return response;
  },error=>{entry.error=String(error);entry.status='error';throw error;});
 };
}

/** Read-only DOM observer; profile/search action clock comes from real captured UI events. */
export function installPersonaReadiness(settings={navigation:true,query:null}){
 if(window.__personaReadiness)window.__personaReadiness.stopped=true;
 const state=window.__personaReadiness={...settings,query:settings.query??new URLSearchParams(location.hash.split('?')[1]??'').get('q')??'Browse the store',dispatchedAt:settings.navigation?0:null,stages:{},firstThree:null,fullSix:null,
  firstAnyUseful:null,zeroResult:null,last:null,stopped:false};
 const dispatch=(event,kind)=>{if(state.stopped||state.dispatchedAt!==null)return;
  if(kind==='profile'&&event.target.closest?.('.person-options button')?.querySelector('strong')?.textContent?.trim()===settings.displayName){
   state.dispatchedAt=performance.now();state.action={kind:'actual-profile-option-click',at:state.dispatchedAt,scrollY};}
  if(kind==='query'&&event.target.matches?.('.shop-search')){
   state.dispatchedAt=performance.now();state.action={kind:'actual-search-form-submit',at:state.dispatchedAt,scrollY};}
 };
 document.addEventListener('click',event=>dispatch(event,'profile'),{capture:true});
 document.addEventListener('submit',event=>dispatch(event,'query'),{capture:true});
 const overlap=(a,b)=>a&&b&&a.left<b.right&&a.right>b.left&&a.top<b.bottom&&a.bottom>b.top;
 const poll=()=>{if(state.stopped)return;const base=window.__candidateProbe?.last;
  if(!base||state.dispatchedAt===null){requestAnimationFrame(poll);return;}
  let mirrors=[];try{mirrors=JSON.parse(document.documentElement.getAttribute('data-local-model-readiness')||'[]');}catch{}
  const articles=[...document.querySelectorAll('#collection article.product')].slice(0,6);
  const snap={...base,cards:base.cards.map((card,i)=>({...card,
    displayPrice:articles[i]?.querySelector('.product-heading > span')?.textContent?.trim()??null,image:{...card.image,
    alt:articles[i]?.querySelector('.product-art img')?.getAttribute('alt')??null,
    sourceAttr:articles[i]?.querySelector('.product-art img')?.getAttribute('src')??null}}))};
  const popup=document.querySelector('.person-popover');const popupVisible=Boolean(popup&&popup.getClientRects().length);
  const trigger=document.querySelector('.person-trigger');const triggerRect=trigger?.getBoundingClientRect();
  for(const [i,card] of snap.cards.entries()){
   const add=articles[i]?.querySelector('.add-button');const addRect=add?.getBoundingClientRect?.()??null;
   const checks={card:Boolean(overlap(card.rect,triggerRect)),title:Boolean(overlap(card.titleRect,triggerRect)),
    body:(card.statusParagraphs??[]).map(p=>Boolean(overlap(p.rect,triggerRect))),
    primaryImage:Boolean(overlap(card.image?.clippedPrimaryRect,triggerRect)),addButton:addRect?Boolean(overlap(addRect,triggerRect)):null};
   card.switcherOcclusion={policy:'strict-whole-card-and-content-rectangle-intersection',overlaps:checks,
    addButtonRect:addRect?{left:addRect.left,right:addRect.right,top:addRect.top,bottom:addRect.bottom}:null,
    noObscuringOverlap:!checks.card&&!checks.title&&!checks.body.some(Boolean)&&!checks.primaryImage&&checks.addButton!==true};
  }
  const useful=card=>card.rect?.fullyWithinViewport&&card.titleRect?.fullyWithinViewport&&Boolean(card.title)
   &&card.statusParagraphs?.length>=2&&card.statusParagraphs.every(p=>p.rect?.fullyWithinViewport)
   &&card.image?.clippedPrimaryRect?.fullyWithinViewport&&card.image.complete&&card.image.naturalWidth>0&&card.image.decodeStatus==='decoded'
   &&!popupVisible&&card.switcherOcclusion.noObscuringOverlap;
  const count=snap.cards.filter(useful).length;const at=performance.now();
  state.last={at,devicePixelRatio:window.devicePixelRatio,input:document.getElementById('shop-search')?.value??null,
   popupVisible,actualUsefulCards:count,switcherRect:triggerRect?{left:triggerRect.left,right:triggerRect.right,top:triggerRect.top,bottom:triggerRect.bottom}:null,snapshot:snap};
  for(const stage of ['rank','firstThree','complete']){
   const mirror=mirrors.filter(e=>e.detail?.stage===stage&&e.startTime>=state.dispatchedAt).sort((a,b)=>b.startTime-a.startTime)[0];
   if(!mirror)continue;
   if(!state.stages[stage])state.stages[stage]={mirror,snapshot:snap,observedAt:at};
  }
  const first=state.stages.firstThree,complete=state.stages.complete;
  const valid=e=>e&&e.mirror.detail.acceptedFields===e.mirror.detail.requiredFields&&e.mirror.detail.withheldFields===0;
  if(valid(first)&&!state.firstAnyUseful&&count>0)state.firstAnyUseful={observedAt:at,cardCount:count,snapshot:snap,requestId:first.mirror.detail.requestId};
  if(valid(first)&&first.mirror.detail.requiredFields===10&&!state.firstThree&&snap.cards.length>=3&&snap.cards.slice(0,3).every(useful))
   state.firstThree={observedAt:at,cardCount:3,snapshot:snap,requestId:first.mirror.detail.requestId};
  if(valid(complete)&&complete.mirror.detail.requiredFields===20&&!state.fullSix&&snap.cards.length>=6
    &&snap.cards.slice(0,6).every(c=>c.image?.complete&&c.image.naturalWidth>0&&c.image.decodeStatus==='decoded'))
   state.fullSix={observedAt:at,cardCount:6,snapshot:snap,requestId:complete.mirror.detail.requestId,fullyVisibleCards:snap.cards.filter(useful).length};
  if(valid(complete)&&complete.mirror.detail.requiredFields===0&&!state.zeroResult)
   state.zeroResult={observedAt:at,actualDomCount:snap.domCardCount,requestId:complete.mirror.detail.requestId,snapshot:snap};
  requestAnimationFrame(poll);
 };
 requestAnimationFrame(poll);
}

export function verifyFacts(events,facts,request={}){const failures=[];const rank=events.find(e=>e.type==='rank');
 const copies=events.filter(e=>e.type==='firstThree'||e.type==='complete');const products=new Map(facts.map(p=>[p.id,p]));
 if(!rank||!Array.isArray(rank.productIds)||new Set(rank.productIds).size!==rank.productIds.length||rank.productIds.some(id=>!products.has(id))
  ||canonical(rank.selectedProductIds)!==canonical(rank.productIds.slice(0,rank.selectedProductIds?.length))||rank.shortfall!==6-rank.selectedProductIds?.length)failures.push('Actual rank selection/shortfall differs from source roster');
 for(const event of copies){const cards=event.sections?.flatMap(s=>s.cards)??[];let accepted=0;
  for(const section of event.sections??[])accepted+=Number(typeof section.heading==='string'&&Boolean(section.heading.trim()));
  for(const card of cards){const product=products.get(card.productId),variant=product?.variants.find(v=>v.id===card.variantId);
   if(Number.isFinite(request.itemBudget)&&card.itemPrice>request.itemBudget)failures.push(`Public item budget exceeded: ${card.productId}`);
   if(request.hardRequirements?.some(r=>r.attribute==='color'&&r.value!==card.selectedColor))failures.push(`Explicit color requirement differs: ${card.productId}`);
   if(!product||!variant||card.originalName!==product.name||card.itemPrice!==product.price||card.selectedColor!==variant.colorName||rank?.variants?.[card.productId]!==card.variantId)failures.push(`Source-bound facts/variant mismatch: ${card.productId}`);
   for(const field of ['title','badge','bodyCopy']){if(card[field]!==null&&(typeof card[field]!=='string'||!card[field].trim()))failures.push(`Invalid ${field}`);accepted+=Number(typeof card[field]==='string'&&Boolean(card[field].trim()));}
  }
  const selected=event.type==='firstThree'?rank?.selectedProductIds?.slice(0,3):rank?.selectedProductIds;
  if(canonical(cards.map(c=>c.productId))!==canonical(selected))failures.push('Copy order differs from actual selected IDs');
  if(event.requiredFields!==(event.sections?.length??0)+cards.length*3||event.acceptedFields!==accepted||event.acceptedFields+event.withheldFields!==event.requiredFields)failures.push('Actual accepted-field count mismatch');
 }
 return{passed:failures.length===0,failures,selectedCount:rank?.selectedProductIds?.length??null,
  firstFields:copies.find(e=>e.type==='firstThree')?.acceptedFields??null,completeFields:copies.find(e=>e.type==='complete')?.acceptedFields??null,
  scope:'Every copy payload count, real original name/price/color/variant and selected order; free-text semantic factuality not claimed'};}
export function domPriceFacts(snapshot,event,facts){
 const cards=event?.sections?.flatMap(s=>s.cards)??[];
 const checks=cards.map((card,i)=>{const product=facts.find(p=>p.id===card.productId),visiblePrice=snapshot?.cards?.[i]?.displayPrice??null;
  return{productId:card.productId,visiblePrice,canonicalPrice:product?.price??null,payloadItemPrice:card.itemPrice,
   priceMatches:Boolean(product&&card.itemPrice===product.price&&visiblePrice===`$${product.price}`)};
 });
 return{passed:Boolean(event)&&checks.length===cards.length&&checks.every(check=>check.priceMatches),checks,
  scope:'Actual rendered price text at complete React observation versus canonical catalog and actual payload; visibility separately checked at useful endpoints'};
}
export function imageFacts(snapshot,event,facts){const cards=event?.sections?.flatMap(s=>s.cards)??[];
 const checks=cards.map((card,i)=>{const product=facts.find(p=>p.id===card.productId),variant=product?.variants.find(v=>v.id===card.variantId),image=snapshot?.cards?.[i]?.image;
  const visiblePrice=snapshot?.cards?.[i]?.displayPrice??null;
  const priceMatches=Boolean(product&&card.itemPrice===product.price&&visiblePrice===`$${product.price}`);
  let path=null;try{const url=new URL(image?.src,ORIGIN);if(url.origin===ORIGIN)path=url.pathname;}catch{}
  return{productId:card.productId,variantId:card.variantId,currentSrc:image?.src??null,alt:image?.alt??null,
   visiblePrice,canonicalPrice:product?.price??null,payloadItemPrice:card.itemPrice,priceMatches,
   sourceAllowed:Boolean(variant?.imageUrls.includes(path)),altAllowed:Boolean(variant?.imageAlts.includes(image?.alt))};});
 return{passed:checks.length===cards.length&&cards.length>0&&checks.every(x=>x.sourceAllowed&&x.altAllowed&&x.priceMatches),checks};}
export function verifyProtocol(row,expectedRequest,seen=new Set()){
 const failures=[];if(row.error||row.captureError)failures.push('Workload/capture error');if(row.pageErrors?.length)failures.push('Actual storefront page error');
 if(row.actualPostRequests?.length!==1||canonical(row.actualPostRequests?.[0]?.payload)!==canonical(expectedRequest))failures.push('Requires one exact public UI request');
 if(!row.bodySettlement?.settled||!row.cloneCaptureSettlement?.settled)failures.push('Actual body capture unsettled');
 if(!row.responseEvidence?.selectedSource||row.responseEvidence.pathsAgree===false)failures.push('Missing or disagreeing actual response bytes');
 const events=row.responseEvidence?.events??[],rank=events.find(e=>e.type==='rank'),complete=events.find(e=>e.type==='complete');
 const id=rank?.requestId;if(events.some(e=>e.type==='error'))failures.push('Actual native stream contains an error');
 const adapter=adapterReceipt(events);if(!adapter.passed)failures.push(...adapter.failures);
 if(!id||seen.has(id))failures.push('Missing/reused current request ID');
 if(!complete||complete.requestId!==id||events.some(e=>e.requestId!==id))failures.push('Current response stages disagree');
 const selected=rank?.selectedProductIds?.length??0;
 if(selected>0&&!events.some(e=>e.type==='firstThree'))failures.push('Missing actual firstThree');
 for(const stage of selected>0?['rank','firstThree','complete']:['rank','complete'])if(row.observation?.stages?.[stage]?.mirror?.detail?.requestId!==id)failures.push(`Missing current ${stage} React mirror`);
 if(row.observation?.last?.devicePixelRatio!==1)failures.push('Actual DPR1 not recorded');
 if(row.observation?.last?.input!==(row.query==='Browse the store'?'':row.query))failures.push('Actual search input differs');
 if(!row.sourceFacts?.passed)failures.push('Canonical public source facts failed');
 if(!row.domPrices?.passed)failures.push('Actual rendered selected-card prices differ from canonical catalog/payload or are missing');
 if(selected>0&&(!['fresh-parallel-head','fresh-native'].includes(complete?.copySource?.kind)||complete?.copySource?.freshPerRequestGeneration!==true||complete?.copySource?.generatedEarlier!==false))failures.push('Fresh neural copy not proven');
 return{passed:failures.length===0,failures,requestId:id??null};
}

/** CPU admission/audit bridge only. No worker/model/browser launch; actual run only. */
const CPU_BRIDGE=String.raw`
import json,sys
from pathlib import Path
from experiments.ranker.pretrained import store_persona_http_semantic_probe_v2 as probe
from experiments.ranker.pretrained import store_persona_development_admission_v1 as development
payload=json.load(sys.stdin)
selected=development.selection(payload['candidateReceipt'],payload['candidateSha256'])
if sys.argv[1]=='selection': print(json.dumps(selected,allow_nan=False));sys.exit(0)
from transformers import AutoTokenizer
from experiments.ranker.pretrained import store_qwen_lexical_choice_inputs_v2 as formatter
from experiments.ranker.pretrained import store_persona_adapter_contract_v2 as contract
from experiments.ranker.pretrained import store_persona_combined_query_v1 as query
tokenizer=AutoTokenizer.from_pretrained(payload['tokenizer'],local_files_only=True,trust_remote_code=False)
request=payload['request'];catalog=formatter.load_public_catalog();products=probe.old.public_catalog()
effective,policy,_=contract.common_request(request,catalog,products)
row=formatter.build_row(request,catalog,tokenizer,'runtime-public-request','canonical')
public={key:row[key] for key in ('prompt','tokenIds','catalogPrefixTokenIds')}
public.update(tokenIdsSha256=contract.token_hash(row['tokenIds']),routeTokenIds=[c['tokenId'] for c in row['choices']])
checked=contract.suffix_contract(public)
expected={'promptTokenIdsSha256':checked['promptTokenIdsSha256'],'promptTokens':len(row['tokenIds']),
 'queryTextSha256':query.digest_text(query.runtime_text(request,products))}
case={'request':request,'effectiveRequest':effective,'effectivePolicy':policy}
result=probe.analyze_events(case,expected,payload['events'],payload['headers'],products,selected['expected'],set())
print(json.dumps(result,allow_nan=False))
`;
export function cpuBridge(command,payload){
 const result=spawnSync(PYTHON,['-B','-c',CPU_BRIDGE,command],{cwd:ROOT,input:JSON.stringify({...payload,tokenizer:TOKENIZER}),encoding:'utf8',timeout:60000,maxBuffer:16*1024*1024,
  env:{...process.env,TOKENIZERS_PARALLELISM:'false',HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1'}});
 if(result.error||result.status!==0)throw Error(`CPU ${command} admission/public audit failed: ${result.error??result.stderr}`);
 return JSON.parse(result.stdout);
}
export function assertSelection(selected){
 if(selected?.mode!=='technical-development-demo'||selected.developmentDemo!==true||selected.qualityGatePassed!==false||selected.modelPromotion!==false||selected.acceptedProfile!==false||selected.overallPromotionPass!==false||selected.acceptanceStateWritten!==false
  ||!selected.expected?.startup||!selected.pins||!Object.keys(selected.pins).length)throw Error('Explicit non-promoted development-demo selection required; actual quality failure cannot become accepted or promoted');
 return true;
}
export function assertHealth(health,expected){
 if(health?.ready!==true||health.status!=='ready'||canonical(health.startup)!==canonical(expected.startup))throw Error('Live ready startup differs from selected candidate lifetime');
 const s=health.startup,p=s.persona,q=p?.queryIntent;
 if(p?.kind!=='persona-adapter-engine.v2'||p.checkpointSha256!==expected.adapterSha256||p.servingManifestSha256!==expected.adapterManifestSha256
  ||q?.checkpointSha256!==expected.gateSha256||q?.sourceSha256!==expected.gateSourceSha256||q?.sourceManifestSha256!==expected.gateTRAINManifestSha256
  ||p.color?.checkpointSha256!==expected.colorHeadSha256||p.color?.failedOldProductOrderingUsed!==false
  ||s.rankCheckpointSha256!==expected.v5HeadSha256||s.copyCheckpointSha256!=='56c6c1f4199d8689f1920d871213a46d2d121fec4e5836ff70e33aaf98c7c7b8'
  ||s.copyCandidate?.checkpointSha256!==expected.copyHeadSha256||s.rankSessionProvenance?.rankHead?.servingExportSha256!==expected.rankExportSha256
  ||s.sourceTensorsHashed!==927||s.memory?.processBudgetBytes!==7*2**30||s.imageDeliveryManifestSha256!==INTEGRATION_SHA)throw Error('Live adapter/query/color/copy/927/memory/image identity differs');
 return true;
}
export function cacheReceipts(rank,complete){const r=rank?.provenance?.personaRanking;
 return{queryFeature:r?.queryFeature??null,colorFeature:r?.colorFeature??null,rankDecision:r?.rankDecision??null,
  learnedQuerySpecificProbability:r?.learnedQuerySpecificProbability??null,copySource:complete?.copySource??null,
  scope:'Actual base-query/color acquisition, LoRA rank XHY/body/projection/gradient/catalog counters and fresh neural word head are separate; no inferred zero or aggregate GPU-call claim.',
  queryTextBinding:'checked actual public input and pinned runtime source; live base-query feature has no separate text SHA'};}
export function adapterReceipt(events){const failures=[],rank=events.find(e=>e.type==='rank'),p=rank?.provenance,r=p?.personaRanking,n=r?.rankDecision;
 const bad=message=>failures.push(message);
 if(!r||r.phraseWhitelistUsed!==false||r.failedOldProductRankingUsed!==false||r.rankAdapterAndCopyAdapterMixed!==false||r.hiddenProfileInputs!==false||p.hiddenProfileInputs!==false)bad('Missing adapter isolation/learned query routing receipts');
 if(!Number.isFinite(r?.learnedQuerySpecificProbability)||r.learnedQuerySpecificProbability<0||r.learnedQuerySpecificProbability>1)bad('Missing learned query probability');
 for(const key of ['queryFeature','colorFeature']){const f=r?.[key];if(typeof f?.cacheHit!=='boolean'||!Number.isInteger(f.nativeBackboneForwardCalls)||f.nativeBackboneForwardCalls!==Number(!f.cacheHit))bad(`Missing/contradictory actual ${key} counters`);}
 if(!n||typeof n.cacheHit!=='boolean'||n.freshLearnedDecision!==true||n.copyAdapterParticipated!==false||n.generatedContinuationTokens!==0||n.optimizerUpdates!==0||n.targetWeightsProvided!==false
  ||!(/^[a-f0-9]{64}$/).test(n.promptTokenIdsSha256??''))bad('Missing fresh target-free LoRA prompt/decision receipt');
 for(const key of ['nativeBodyForwards','nativeLanguageModelProjectionCalls','gradientEvaluationCalls'])if(!Number.isInteger(n?.[key])||n[key]<0)bad(`Missing actual rank ${key}`);
 if(n?.cacheHit===true&&(n.kind!=='public-XHY-feature-reuse'||n.nativeBodyForwards!==0||n.nativeLanguageModelProjectionCalls!==1||n.gradientEvaluationCalls!==0))bad('XHY reuse falsely claims native body/gradient work');
 if(n?.cacheHit===false&&(n.kind!=='fresh-native-last16-rank-adapter'||n.nativeLanguageModelProjectionCalls!==2||n.gradientEvaluationCalls!==1||n.nativeRankSuffixTokens!==16))bad('Fresh rank suffix counters differ');
 if(events.some(e=>canonical(e.provenance)!==canonical(p)))bad('Actual adapter provenance changed between response stages');
 return{passed:!failures.length,failures,cache:cacheReceipts(rank,events.find(e=>e.type==='complete'))};
}

/** Separate functional observer: no DOM writes, requests, generated fields or gold. */
export function installFunctionalObservation(){
 if(window.__personaFunctional)window.__personaFunctional.stopped=true;
 const state=window.__personaFunctional={startedAt:performance.now(),observations:[],newMirrors:[],stopped:false};let previous=null;const seen=new Set();
 const poll=()=>{if(state.stopped)return;let mirrors=[];try{mirrors=JSON.parse(document.documentElement.getAttribute('data-local-model-readiness')||'[]');}catch{}
  for(const m of mirrors){const key=JSON.stringify([m.startTime,m.duration,m.detail?.requestId,m.detail?.stage]);if(!seen.has(key)){seen.add(key);state.newMirrors.push({...m,observedAt:performance.now(),completedAt:m.startTime+m.duration});}}
  const row={at:performance.now(),person:document.querySelector('.person-trigger strong')?.textContent?.trim()??null,
   cartCount:Number(document.querySelector('.cart-count')?.textContent??0),hash:location.hash,
   checkoutReceipt:document.querySelector('main')?.getAttribute('data-demo-receipt-id')??null,
   checkoutId:document.querySelector('main')?.getAttribute('data-demo-checkout-id')??null,
   heading:document.querySelector('main h1')?.textContent?.trim()??null,
   cards:[...document.querySelectorAll('#collection article.product')].slice(0,6).map(c=>({title:c.querySelector('h3')?.textContent?.trim(),
    originalName:c.querySelector('.product-colors')?.getAttribute('aria-label')?.replace(/ colors$/,''),
    selectedColor:c.querySelector('.selected-color')?.textContent?.trim(),copy:[...c.querySelectorAll(':scope > p.search-status')].map(p=>p.textContent?.trim())}))};
  const content=JSON.stringify({...row,at:null});if(content!==previous){state.observations.push(row);previous=content;}requestAnimationFrame(poll);};requestAnimationFrame(poll);
}
export function verifyProfilePanel(person,panel){const normalize=x=>String(x??'').replace(/\s+/g,' ').trim(),text=normalize(panel?.text);
 const recent=[...person.request.publicHistory].sort((a,b)=>b.occurredAt.localeCompare(a.occurredAt)).slice(0,3);
 const failures=[];if(!text.includes(normalize(person.summary)))failures.push('Visible public summary differs');
 if(person.request.persona?.colorPreferences.some(c=>!text.includes(normalize(c))))failures.push('Visible public color preferences differ');
 if(panel?.history?.length!==recent.length)failures.push('Visible recent history count differs');
 const labels={view:'Viewed',cart:'Added to cart',purchase:'Purchased'};
 for(const [i,event] of recent.entries()){const actual=normalize(panel?.history?.[i]);for(const fact of [labels[event.eventType],event.itemName,event.selectedColor].filter(Boolean))if(!actual.includes(normalize(fact)))failures.push(`Visible recent history fact differs at${i}`);}
 return{passed:!failures.length,failures,recentPublicEventIds:recent.map(e=>e.eventId),scope:'Actual rendered panel summary/colors/latest3 event type, canonical item and selected color; no hidden persona truth or recommendation labels'};
}
export function assessRapidArm(arm){const failures=[];
 if(arm.actualPostRequests?.length!==2||!arm.exactMiddleAndFinalInputs)failures.push('Rapid arm requires exactly two actual public UI POSTs');
 if(!arm.middleRequest?.abortObserved||arm.middleRequest.statusAtAbort!=='pending')failures.push('Middle actual fetch was not observed pending at UI abort; race not exercised');
 if(!arm.final?.protocol?.passed||!arm.final?.domVariantImages?.passed||!arm.final?.adapterHttpAudit?.protocolPassed||!arm.final?.adapterHttpAudit?.qualityPassed)failures.push('Final A current response/factual audit failed');
 const stale=(arm.observation?.newMirrors??[]).filter(m=>m.completedAt>=arm.finalDispatchedAt&&m.detail?.requestId!==arm.final?.protocol?.requestId);
 if(stale.length)failures.push('Noncurrent response mirror completed after final A dispatch');
 const finalCards=arm.final?.responseEvidence?.events?.find(e=>e.type==='complete')?.sections?.flatMap(s=>s.cards)??[];
 const finalTitles=new Map(finalCards.map(c=>[c.originalName,c.title]));
 const observations=arm.observation?.observations;const transientDomFailures=[];
 if(!Array.isArray(observations)||!observations.some(o=>o.cards?.length>=3))failures.push('Missing actual transient card DOM observations');
 for(const o of observations??[])if(o.at>=arm.finalDispatchedAt)for(const card of o.cards??[])if(card.title&&card.title!==card.originalName&&card.title!==finalTitles.get(card.originalName))transientDomFailures.push({at:o.at,...card});
 if(transientDomFailures.length)failures.push('Noncurrent generated title visible after final A dispatch');
 return{transientDomFailures,passed:!failures.length,failures,staleMirrors:stale,cancellationScope:'Actual browser AbortSignal observed; server may already have completed a native step. No assertion that GPU work was immediately interrupted.'};}
export function installCheckoutSwitchObservation({displayName}){
 const listener=event=>{if(event.target.closest?.('.person-options button')?.querySelector('strong')?.textContent?.trim()!==displayName)return;
  window.__personaCheckoutSwitch={at:performance.now(),pendingAtActualClick:document.querySelector('.cart-page button.primary-button')?.disabled===true,
   receiptAtActualClick:document.querySelector('main')?.getAttribute('data-demo-receipt-id')??null};document.removeEventListener('click',listener,true);};
 document.addEventListener('click',listener,true);
}
export function assessCartArm(arm){const failures=[];
 if(arm.actualPostRequests?.length!==2||!arm.exactMayaAndBobInputs)failures.push('Cart arm requires exactly two native public UI POSTs');
 if(!arm.addedCanonicalItem||arm.bobCartBefore?.count!==1||!arm.bobCartBefore?.text?.includes(arm.addedCanonicalItem.originalName)||!arm.bobCartBefore?.text?.includes(arm.addedCanonicalItem.selectedColor))failures.push('Bob actual selected variant cart binding missing');
 if(arm.bobCartBefore?.lines?.length!==1||arm.bobCartBefore.lines[0].price!==`$${arm.addedCanonicalItem?.itemPrice}`)failures.push('Actual Bob cart price differs from actual canonical selected-card price');
 if(arm.mayaCart?.count!==0||!arm.mayaCart?.text?.includes('Your cart is empty.'))failures.push('Maya actual cart inherited Bob items');
 if(arm.bobRestored?.count!==1||arm.bobRestored?.checkoutReceipt!==null||arm.mayaCart?.checkoutReceipt!==null)failures.push('Bob cart restoration or per-person checkout invalidation failed');
 if(arm.checkoutRequests?.length!==1||arm.checkoutResponses?.length!==1||arm.checkoutResponses[0].status!==202||arm.checkoutResponses[0].body?.receipt?.realPurchase!==false)failures.push('One actual local simulated checkout receipt required');
 const checkoutItem=arm.checkoutResponses?.[0]?.body?.receipt?.items?.[0];if(arm.checkoutResponses?.[0]?.body?.receipt?.items?.length!==1||checkoutItem?.productId!==arm.addedCanonicalItem?.productId||checkoutItem?.variantId!==arm.addedCanonicalItem?.variantId||checkoutItem?.unitPrice!==arm.addedCanonicalItem?.itemPrice||checkoutItem?.quantity!==1)failures.push('Actual checkout receipt item/variant/price differs from selected cart');
 if(arm.finals?.length!==2||arm.finals.some(f=>!f.protocol?.passed||!f.domVariantImages?.passed||!f.adapterHttpAudit?.protocolPassed||!f.adapterHttpAudit?.qualityPassed))failures.push('Current per-person response/factual audit failed');
 return{passed:!failures.length,failures,checkoutRaceExercised:arm.checkoutPendingAtSwitch===true,
  checkoutRaceScope:arm.checkoutPendingAtSwitch?'Observed local checkout pending before switch; final actual person state checked after response settlement':'Checkout had already settled before switch; completed-receipt invalidation checked. No live in-flight checkout race pass claimed.'};}

export function parseArgs(args){const options={run:false,out:null,manifest:INTEGRATION_MANIFEST,candidateReceipt:null,candidateSha256:null};
 const names={'--out':'out','--manifest':'manifest','--candidate-receipt':'candidateReceipt','--candidate-sha256':'candidateSha256'};
 for(let i=0;i<args.length;i++){if(args[i]==='--run')options.run=true;
  else if(names[args[i]]){if(!args[i+1]||args[i+1].startsWith('--'))throw Error(`Missing ${args[i]} value`);options[names[args[i]]]=args[++i];}
  else throw Error(`Unknown argument ${args[i]}`);}
 if(options.run&&!options.out)throw Error('--run requires a unique --out directory');
 if(options.run&&(!options.candidateReceipt||!/^([a-f0-9]{64})$/.test(options.candidateSha256??'')))throw Error('--run requires explicit development-demo --candidate-receipt and exact --candidate-sha256');
 if(options.manifest!==INTEGRATION_MANIFEST)throw Error('Only reviewed adapter integration manifest is supported');return options;}
export async function loadFacts(){
 const require=createRequire(join(ROOT,'apps/storefront/package.json'));
 const ts=require('tsx/cjs/api').require,from=join(ROOT,'apps/storefront/package.json');
 const{PRODUCTS}=ts('./src/config.ts',from);
 const{PRODUCT_VARIANTS,getVariantImages}=ts('./src/variants.ts',from);
 const{responsiveProductImage}=ts('./src/product-responsive-runtime.ts',from);
 const{buildLocalModelRequest}=ts('./src/local-model.ts',from);
 const facts=PRODUCTS.map(product=>({...product,variants:PRODUCT_VARIANTS[product.id].map(variant=>{
  const images=getVariantImages(variant);
  const imageUrls=images.flatMap(image=>{
   const delivery=responsiveProductImage(image.src);
   return[image.src,delivery.src,...(delivery.srcSet??'').split(',').map(x=>x.trim().split(' ')[0]).filter(Boolean)];
  });
  return{id:variant.id,colorName:variant.colorName,imageAlts:images.map(i=>i.alt),imageUrls:[...new Set(imageUrls)]};
 })}));
 return{requestFor:(person,query)=>buildLocalModelRequest(query,person.request),facts};
}
export function selectPersonaEvidence(cdp,clones,id){
 const normal=selectResponseEvidence(cdp,clones,id);if(normal.selectedSource||normal.pathsAgree===false)return normal;
 const valid=r=>r.parseErrors?.length===0&&r.events?.some(e=>e.type==='rank'&&e.requestId===id&&e.selectedProductIds?.length===0)
  &&r.events?.some(e=>e.type==='complete'&&e.requestId===id&&e.requiredFields===0&&e.sections?.length===0&&e.copySource?.kind==='no-generation');
 const a=cdp.find(valid),b=clones.find(valid),agree=a&&b?a.bodySha256===b.bodySha256:null,selected=agree===false?null:a??b;
 return{selectedSource:selected?(selected===a?'playwright-response-body':'in-page-actual-response-clone'):null,pathsAgree:agree,
  selectedBodySha256:selected?.bodySha256??null,events:selected?.events??[],scope:'Actual rank/empty complete bytes; no synthesized firstThree'};
}
async function health(){const response=await fetch('http://127.0.0.1:8767/health',{signal:AbortSignal.timeout(5000)});const raw=await response.text();return{status:response.status,body:JSON.parse(raw),sha256:sha(Buffer.from(raw))};}
async function hashes(paths){return Object.fromEntries(await Promise.all(paths.map(async p=>[p,sha(await readFile(isAbsolute(p)?p:join(ROOT,p)))])));}
async function inventory(manifest){const fixed=['experiments/ranker/pretrained/store_persona_development_browser_probe_v2.mjs','experiments/ranker/pretrained/test_store_persona_development_browser_probe_v2.mjs','experiments/ranker/pretrained/STORE_PERSONA_DEVELOPMENT_BROWSER_PROBE_V2_PLAN.json','experiments/ranker/pretrained/store_persona_development_admission_v1.py','experiments/ranker/pretrained/store_persona_browser_probe_v2.mjs','experiments/ranker/pretrained/test_store_persona_browser_probe_v2.mjs','experiments/ranker/pretrained/STORE_PERSONA_BROWSER_PROBE_V2_PLAN.json','experiments/ranker/pretrained/store_persona_browser_probe_v1.mjs','experiments/ranker/pretrained/test_store_persona_browser_probe_v1.mjs','experiments/ranker/pretrained/STORE_PERSONA_BROWSER_PROBE_V1_PLAN.json',PUBLIC_PEOPLE,
 'apps/storefront/package.json','apps/storefront/vite.config.ts',
 'experiments/ranker/pretrained/store_candidate_browser_probe_v1.mjs','experiments/ranker/pretrained/store_candidate_browser_probe_v2.mjs','experiments/ranker/pretrained/store_candidate_collection_probe_v1.mjs',
 'experiments/ranker/pretrained/store_persona_adapter_engine_v2.py','experiments/ranker/pretrained/store_persona_adapter_runtime_v2.py','experiments/ranker/pretrained/store_persona_adapter_contract_v2.py','experiments/ranker/pretrained/store_persona_http_semantic_probe_v2.py','experiments/ranker/pretrained/store_persona_context_v1.py',
 'experiments/ranker/pretrained/store_candidate_engine_v1.py','experiments/ranker/pretrained/store_candidate_image_service_v1.py','experiments/ranker/pretrained/store_demo_engine.py','experiments/ranker/pretrained/store_demo_service.py','experiments/ranker/pretrained/store_demo_image_binding_v1.py','experiments/ranker/pretrained/store_rank_session_v5.py',
 ...['main.tsx','ShopView.tsx','PersonaSwitcher.tsx','personas.ts','local-model.ts','local-model-mode.ts','local-model-mode.test.ts','local-model-client.ts','use-local-model.ts','local-model-performance.ts','ProductCard.tsx','ProductGallery.tsx','ProductPhoto.tsx','variants.ts','images.ts','config.ts','catalog-expansion.ts','styles.css','product-responsive-runtime.ts','product-image-runtime.json','color-gallery-layouts.json'].map(p=>`apps/storefront/src/${p}`)];
 if(manifest){const content=JSON.parse(await readFile(join(ROOT,manifest),'utf8'));fixed.push(manifest,...Object.keys(content.frontendSha256??{}),...Object.keys(content.verificationSourcesSha256??{}));}
 for(const path of ['experiments/ranker/pretrained/store_persona_qwen_v1.py','experiments/ranker/pretrained/store_persona_qwen_fit_v1.py'])try{await access(join(ROOT,path));fixed.push(path);}catch{}
 return[...new Set(fixed)].sort();}

async function choosePerson(page,person,{alreadyOpen=false,keepOpen=false}={}){
 if(!alreadyOpen)await page.getByRole('button',{name:/Switch person/}).click();
 const panel=page.getByRole('dialog',{name:'Who’s shopping?'});
 const escaped=person.displayName.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
 const option=panel.getByRole('button',{name:new RegExp(`^${escaped}(?:\\s|$)`)});
 if(await option.getAttribute('aria-pressed')==='true')throw Error('Functional same-person selection would be a no-op');
 await option.click();if(!keepOpen)await page.keyboard.press('Escape');
}
function newBucket(){return{actualPostRequests:[],responses:[],pending:[],requestFailures:[],pageErrors:[],open:true};}
async function functionalFinal(page,capture,expected,ctx,scopeStart){
 const row={query:expected.query};
 try{
  await page.waitForFunction(()=>Boolean(window.__personaReadiness?.stages?.complete),null,{timeout:TIMEOUT_MS});
  Object.assign(row,await settleNavigationCaptures(capture.pending,timeout=>page.waitForFunction(start=>{
   const requests=(window.__personaFetch?.requests??[]).filter(r=>r.startedAt>=start);return requests.length>0&&requests.every(r=>r.status!=='pending');
  },scopeStart,{timeout})));
  const clones=await page.evaluate(start=>(window.__personaFetch?.requests??[]).filter(r=>r.startedAt>=start),scopeStart);
  row.cloneResponses=clones.map(c=>c.status==='complete'?parseResponseBytes(c.bodyBytes,{...c,source:'in-page-actual-response-clone'}):c);
  row.observation=await page.evaluate(()=>{window.__personaReadiness.stopped=true;return window.__personaReadiness;});
  const id=row.observation.stages.complete?.mirror?.detail?.requestId;
  row.responseEvidence=selectPersonaEvidence(capture.responses,row.cloneResponses,id);
  const events=row.responseEvidence.events,complete=events.find(e=>e.type==='complete');
  row.actualPostRequests=capture.actualPostRequests.filter(r=>canonical(r.payload)===canonical(expected));
  row.sourceFacts=verifyFacts(events,ctx.facts,expected);
  row.domPrices=domPriceFacts(row.observation.stages.complete?.snapshot,complete,ctx.facts);
  row.domCopy=matchDomToCopy(row.observation.stages.complete?.snapshot,complete);
  row.domVariantImages=imageFacts(row.observation.stages.complete?.snapshot,complete,ctx.facts);
  row.parsedSourceMatch=parsedSourceFieldsMatch(row.observation.stages.complete?.mirror?.detail?.copySource,complete?.copySource);
  row.cache=cacheReceipts(events.find(e=>e.type==='rank'),complete);
  const headers=capture.responses.find(r=>r.headers?.['x-request-id']===id)?.headers??{};
  row.adapterHttpAudit=cpuBridge('audit',{candidateReceipt:ctx.options.candidateReceipt,candidateSha256:ctx.options.candidateSha256,request:expected,events,headers});
 }catch(error){row.error=String(error);}
 row.protocol=verifyProtocol(row,expected,ctx.seen);
 if(!row.domCopy?.passed||!row.domVariantImages?.passed||!row.parsedSourceMatch||!row.adapterHttpAudit?.protocolPassed||!row.adapterHttpAudit?.qualityPassed){row.protocol.passed=false;row.protocol.failures.push('Current final DOM copy/source/public adapter audit failed');}
 if(row.protocol.requestId)ctx.seen.add(row.protocol.requestId);
 return row;
}
async function actualUiState(page){return page.evaluate(()=>({count:Number(document.querySelector('.cart-count')?.textContent??0),
 text:document.querySelector('.cart-page')?.textContent??null,lines:[...document.querySelectorAll('.cart-items > div')].map(line=>({text:line.querySelector('span')?.textContent?.trim()??null,price:line.querySelector(':scope > span:last-child')?.textContent?.trim()??null})),person:document.querySelector('.person-trigger strong')?.textContent?.trim()??null,
 checkoutReceipt:document.querySelector('main')?.getAttribute('data-demo-receipt-id')??null,
 checkoutId:document.querySelector('main')?.getAttribute('data-demo-checkout-id')??null,
 hash:location.hash,at:performance.now()}));}
export async function runFunctionalArms(page,ctx){const arms=[];const bob=ctx.byId.get('bob'),maya=ctx.byId.get('maya');
 const requestBob=ctx.requestFor(bob,'work bag'),requestMaya=ctx.requestFor(maya,'work bag');
 let capture=newBucket();ctx.selectBucket(capture);
 const rapid={kind:'rapid-Bob-Maya-Bob',scope:'Functional only; excluded from fixed13 timing aggregate',expectedNativeRequests:2,startedAt:new Date().toISOString()};
 try{
  const start=await page.evaluate(()=>performance.now());rapid.startedAtBrowser=start;
  await page.evaluate(installFunctionalObservation);
  await page.evaluate(installPersonaReadiness,{navigation:false,query:'work bag',displayName:maya.displayName});
  await choosePerson(page,maya,{keepOpen:true});
  await page.waitForFunction(start=>(window.__personaFetch?.requests??[]).some(r=>r.startedAt>=start),start,{timeout:5000});
  rapid.middleAtFinalDispatch=await page.evaluate(start=>(window.__personaFetch?.requests??[]).find(r=>r.startedAt>=start),start);
  await page.evaluate(installPersonaReadiness,{navigation:false,query:'work bag',displayName:bob.displayName});
  await choosePerson(page,bob,{alreadyOpen:true});
  rapid.final=await functionalFinal(page,capture,requestBob,ctx,start);
  rapid.finalDispatchedAt=rapid.final.observation?.dispatchedAt??null;
  rapid.actualPostRequests=capture.actualPostRequests;rapid.responses=capture.responses;rapid.requestFailures=capture.requestFailures;rapid.pageErrors=capture.pageErrors;
  rapid.actualClones=await page.evaluate(start=>(window.__personaFetch?.requests??[]).filter(r=>r.startedAt>=start),start);
  rapid.middleRequest=rapid.actualClones[0]??null;
  rapid.exactMiddleAndFinalInputs=canonical(capture.actualPostRequests.map(r=>r.payload))===canonical([requestMaya,requestBob]);
  rapid.observation=await page.evaluate(()=>{window.__personaFunctional.stopped=true;return window.__personaFunctional;});
  rapid.currentUi=await actualUiState(page);
  Object.assign(rapid,assessRapidArm(rapid));
 }catch(error){rapid.error=String(error);rapid.passed=false;}
 finally{capture.open=false;rapid.finishedAt=new Date().toISOString();await save(join(resolve(ctx.options.out),'functional-rapid-switch.json'),{...rapid,pending:undefined});arms.push(rapid);}
 if(!rapid.passed)return arms;
 capture=newBucket();ctx.selectBucket(capture);
 const cart={kind:'per-person-cart-and-checkout-invalidation',scope:'Functional only; excluded from fixed13 timing aggregate',expectedNativeRequests:2,expectedLocalCheckoutRequests:1,
  checkoutRequests:[],checkoutResponses:[],finals:[],startedAt:new Date().toISOString()};
 const pendingCheckout=[];
 const checkoutRequest=request=>{if(request.method()==='POST'&&new URL(request.url()).pathname==='/api/demo-checkout'){let payload=null;try{payload=request.postDataJSON();}catch{}
  cart.checkoutRequests.push({url:request.url(),payload,postData:request.postData(),startedAt:Date.now()});}};
 const checkoutResponse=response=>{if(response.request().method()!=='POST'||new URL(response.url()).pathname!=='/api/demo-checkout')return;
  const record={status:response.status(),headers:response.headers(),captureStatus:'pending',receivedAt:Date.now()};cart.checkoutResponses.push(record);
  pendingCheckout.push((async()=>{try{const body=await response.body();record.rawUtf8=body.toString('utf8');record.rawSha256=sha(body);record.body=JSON.parse(record.rawUtf8);record.captureStatus='complete';}catch(error){record.error=String(error);record.captureStatus='error';}})());};
 page.on('request',checkoutRequest);page.on('response',checkoutResponse);
 try{
  const previous=rapid.final.responseEvidence.events.find(e=>e.type==='complete'),card=previous?.sections?.[0]?.cards?.[0];
  if(!card)throw Error('No actual selected card available to test cart; not padded');
  cart.addedCanonicalItem={productId:card.productId,variantId:card.variantId,originalName:card.originalName,itemPrice:card.itemPrice,selectedColor:card.selectedColor};
  await page.locator('#collection article.product').first().getByRole('button',{name:'Add to cart',exact:true}).click();
  await page.getByRole('button',{name:'View cart',exact:true}).click();
  await page.waitForSelector('.cart-page');cart.bobCartBefore=await actualUiState(page);
  await page.getByRole('button',{name:'Confirm order',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('.cart-page button.primary-button')?.textContent?.includes('Confirming')||document.querySelector('main')?.hasAttribute('data-demo-receipt-id'),null,{timeout:5000}).catch(()=>{});
  await page.evaluate(installCheckoutSwitchObservation,{displayName:maya.displayName});
  const start=await page.evaluate(()=>performance.now());
  await page.evaluate(installPersonaReadiness,{navigation:false,query:'work bag',displayName:maya.displayName});
  await choosePerson(page,maya);
  const finalMaya=await functionalFinal(page,capture,requestMaya,ctx,start);cart.finals.push(finalMaya);
  cart.checkoutSwitchObservation=await page.evaluate(()=>window.__personaCheckoutSwitch??null);cart.checkoutPendingAtSwitch=cart.checkoutSwitchObservation?.pendingAtActualClick===true;
  await page.getByRole('button',{name:'View cart',exact:true}).click();await page.waitForSelector('.cart-page');cart.mayaCart=await actualUiState(page);
  const bobStart=await page.evaluate(()=>performance.now());
  await page.evaluate(installPersonaReadiness,{navigation:false,query:'work bag',displayName:bob.displayName});
  await choosePerson(page,bob);
  const finalBob=await functionalFinal(page,capture,requestBob,ctx,bobStart);cart.finals.push(finalBob);
  await Promise.all(pendingCheckout);await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  cart.bobRestored=await actualUiState(page);
  cart.actualPostRequests=capture.actualPostRequests;cart.responses=capture.responses;
  cart.requestFailures=capture.requestFailures;cart.pageErrors=capture.pageErrors;
  cart.exactMayaAndBobInputs=canonical(capture.actualPostRequests.map(r=>r.payload))===canonical([requestMaya,requestBob]);
  Object.assign(cart,assessCartArm(cart));
 }catch(error){cart.error=String(error);cart.passed=false;}
 finally{capture.open=false;page.off('request',checkoutRequest);page.off('response',checkoutResponse);cart.finishedAt=new Date().toISOString();await save(join(resolve(ctx.options.out),'functional-cart-checkout.json'),cart);arms.push(cart);}
 return arms;
}

export async function run(options){const output=resolve(options.out);await mkdir(output);
 const report={...plan(),version:'store-persona-development-browser-probe.v2',startedAt:new Date().toISOString(),attempts:[],functionalArms:[]};let browser,context;let selected;
 try{const people=JSON.parse(await readFile(join(ROOT,PUBLIC_PEOPLE),'utf8')).people;const byId=new Map(people.map(p=>[p.id,p]));
  if(people[0]?.id!=='bob'||PROFILE_ORDER.some(id=>!byId.has(id)))throw Error('Fixed public roster changed');
  selected=cpuBridge('selection',{candidateReceipt:options.candidateReceipt,candidateSha256:options.candidateSha256});assertSelection(selected);report.candidateSelection=selected;
  if(sha(await readFile(join(ROOT,options.manifest)))!==INTEGRATION_SHA)throw Error('Reviewed adapter UI/image integration manifest changed');
  const{facts,requestFor}=await loadFacts();report.publicCatalog=facts;report.sourceHashes=await hashes([...new Set([...(await inventory(options.manifest)),...Object.keys(selected.pins)])]);
  report.healthBefore=await health();assertHealth(report.healthBefore.body,selected.expected);
  await save(join(output,'plan.json'),report);
  const require=createRequire(join(ROOT,'package.json'));const{chromium}=require('@playwright/test');
  browser=await chromium.launch({channel:'chrome',headless:false});report.browserVersion=browser.version();
  context=await browser.newContext({...CONTEXT_OPTIONS,viewport:{...CONTEXT_OPTIONS.viewport}});
  await context.addInitScript(installPersonaFetchCapture);await context.addInitScript(installObserver);await context.addInitScript(installPersonaReadiness);
  const page=await context.newPage();const seen=new Set();let bucket;
  page.on('request',request=>{if(bucket&&request.method()==='POST'&&request.url()===ENDPOINT){let payload=null;try{payload=request.postDataJSON();}catch{}
   bucket.actualPostRequests.push({url:request.url(),payload,postData:request.postData()});}});
  page.on('requestfailed',request=>bucket?.requestFailures.push({url:request.url(),error:request.failure()}));
  page.on('pageerror',error=>bucket?.pageErrors.push({message:error.message,stack:error.stack??null}));
  page.on('response',response=>{if(!bucket||response.url()!==ENDPOINT||response.request().method()!=='POST')return;const target=bucket;const receipt={status:response.status(),headers:response.headers(),captureStatus:'pending'};target.responses.push(receipt);
   target.pending.push((async()=>{try{const error=await response.finished();if(error)throw error;const parsed=parseResponseBytes(await response.body(),{source:'playwright-response-body'});if(target.open)Object.assign(receipt,parsed,{captureStatus:'complete'});}
    catch(error){if(target.open)Object.assign(receipt,{error:String(error),captureStatus:'error'});}})());});
  for(const item of fixedSchedule()){
   const person=byId.get(item.personId);const expected=requestFor(person,item.query);
   const row={...item,expectedPublicRequest:expected,publicRequestSha256:sha(Buffer.from(canonical(expected))),startedAt:new Date().toISOString()};
   bucket={actualPostRequests:[],responses:[],pending:[],requestFailures:[],pageErrors:[],open:true};const capture=bucket;
   try{
    if(item.navigation){if(item.navigation==='goto')await page.goto(`${ORIGIN}/`,{waitUntil:'domcontentloaded',timeout:TIMEOUT_MS});
     else await page.reload({waitUntil:'domcontentloaded',timeout:TIMEOUT_MS});}
    else{
     await page.evaluate(()=>document.getElementById('collection').scrollIntoView({block:'start',behavior:'instant'}));
     await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
     row.preAction=await page.evaluate(()=>({scrollY,input:document.getElementById('shop-search')?.value,hash:location.hash}));
     await page.evaluate(installPersonaReadiness,{navigation:false,query:item.query,displayName:person.displayName});
     if(item.kind==='loaded-profile-switch'){
      await page.getByRole('button',{name:/Switch person/}).click();
      const panel=page.getByRole('dialog',{name:'Who’s shopping?'});
      const option=panel.getByRole('button',{name:new RegExp(`^${person.displayName.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}(?:\\s|$)`)});
      if(await option.getAttribute('aria-pressed')==='true')throw Error('Same-person selection would not create a fresh request');
      await option.click();
      await panel.getByRole('heading',{name:`${person.displayName}’s day to day`,exact:true}).waitFor();
      row.profilePanel={text:await panel.innerText(),history:await panel.locator('.person-history li').allTextContents()};
      row.profileHistory=verifyProfilePanel(person,row.profilePanel);await page.keyboard.press('Escape');
     }else{await page.locator('#shop-search').fill(item.query);await page.locator('.shop-search button[type=submit]').click();}
    }
    await page.waitForFunction(()=>Boolean(window.__personaReadiness?.stages?.complete),null,{timeout:TIMEOUT_MS});
    if(item.kind==='bare-navigation'){
     await page.waitForFunction(()=>Boolean(window.__personaReadiness?.fullSix||window.__personaReadiness?.zeroResult),null,{timeout:3000}).catch(error=>{row.imageEndpointMiss=String(error);});
    }else{
     const fields=await page.evaluate(()=>window.__personaReadiness?.stages?.complete?.mirror?.detail?.requiredFields);
     if(fields===20)await page.waitForFunction(()=>Boolean(window.__personaReadiness?.firstThree&&window.__personaReadiness?.fullSix),null,{timeout:3000}).catch(error=>{row.imageEndpointMiss=String(error);});
     else row.endpointContractMiss={actualRequiredFields:fields,expectedForSix:20,scope:'Real shortfall/zero retained; never padded to fixed useful3/full6 contract'};
    }
   }catch(error){row.error=String(error);}
   try{
    const dispatchedAt=await page.evaluate(()=>window.__personaReadiness?.dispatchedAt??Infinity);
    Object.assign(row,await settleNavigationCaptures(capture.pending,timeout=>page.waitForFunction(start=>{
     const requests=(window.__personaFetch?.requests??[]).filter(r=>r.startedAt>=start);return requests.length>0&&requests.every(r=>r.status!=='pending');
    },dispatchedAt,{timeout})));
    capture.open=false;Object.assign(row,{...capture,pending:undefined});
    const clones=await page.evaluate(start=>(window.__personaFetch?.requests??[]).filter(r=>r.startedAt>=start),dispatchedAt);
    row.cloneResponses=clones.map(c=>c.status==='complete'?parseResponseBytes(c.bodyBytes,{...c,source:'in-page-actual-response-clone'}):c);
    row.observation=await page.evaluate(()=>{window.__personaReadiness.stopped=true;return{...window.__personaReadiness,navigation:performance.getEntriesByType('navigation').map(e=>e.toJSON()),resources:performance.getEntriesByType('resource').map(e=>e.toJSON())};});
    const id=row.observation.stages.complete?.mirror?.detail?.requestId;
    row.responseEvidence=selectPersonaEvidence(row.responses,row.cloneResponses,id);
    const events=row.responseEvidence.events,rank=events.find(e=>e.type==='rank'),first=events.find(e=>e.type==='firstThree'),complete=events.find(e=>e.type==='complete');
    row.sourceFacts=verifyFacts(events,facts,expected);
    row.domPrices=domPriceFacts(row.observation.stages.complete?.snapshot,complete,facts);
    row.cache=cacheReceipts(rank,complete);
    row.domCopy={firstThree:matchDomToCopy(row.observation.firstThree?.snapshot,first),fullSix:matchDomToCopy(row.observation.fullSix?.snapshot,complete),
     completeAtObservation:matchDomToCopy(row.observation.stages.complete?.snapshot,complete),
     parsedSourceMatch:parsedSourceFieldsMatch(row.observation.stages.complete?.mirror?.detail?.copySource,complete?.copySource)};
    row.imageFacts={firstThree:imageFacts(row.observation.firstThree?.snapshot,first,facts),fullSix:imageFacts(row.observation.fullSix?.snapshot,complete,facts)};
    row.usefulContract={firstThree:Boolean(row.sourceFacts.passed&&first?.acceptedFields===10&&first?.withheldFields===0&&row.domCopy.firstThree.passed&&row.imageFacts.firstThree.passed&&row.domCopy.parsedSourceMatch),
     fullSix:Boolean(row.sourceFacts.passed&&complete?.acceptedFields===20&&complete?.withheldFields===0&&row.domCopy.fullSix.passed&&row.imageFacts.fullSix.passed&&row.domCopy.parsedSourceMatch)};
    row.timings=Object.fromEntries(['firstThree','fullSix'].map(stage=>{const endpoint=row.observation[stage];return[stage,endpoint?{observedAt:endpoint.observedAt,fromNativeFetchMs:endpoint.observedAt-(clones[0]?.startedAt??NaN),fromActionMs:endpoint.observedAt-row.observation.dispatchedAt,navigationStartMs:item.navigation?endpoint.observedAt:null}:null];}));
    row.initialViewport=item.kind==='bare-navigation'?{actualUsefulCardsAtCapture:row.observation.last.actualUsefulCards,firstAnyUseful:row.observation.firstAnyUseful,
     firstThreeActuallyReady:Boolean(row.usefulContract.firstThree),finalScrollY:row.observation.last.snapshot.scrollY,snapshot:row.observation.last.snapshot}:null;
    if(item.kind==='same-query-reload'){const prior=report.attempts.find(r=>r.kind==='first-work-bag-query');row.sameQueryRepeat={identicalInput:canonical(prior?.actualPostRequests?.[0]?.payload)===canonical(row.actualPostRequests[0]?.payload),priorRequestId:prior?.protocol?.requestId??null,currentRequestId:id};}
    const path=join(output,`attempt-${item.index+1}.png`),bytes=await page.screenshot({path,fullPage:false,timeout:5000});row.screenshot={path,sha256:sha(bytes)};
   }catch(error){row.captureError=String(error);capture.open=false;Object.assign(row,{...capture,pending:undefined});}
   try{row.adapterHttpAudit=cpuBridge('audit',{candidateReceipt:options.candidateReceipt,candidateSha256:options.candidateSha256,request:expected,events:row.responseEvidence?.events??[],headers:row.responses.find(r=>r.headers?.['x-request-id'])?.headers??{}});}catch(error){row.adapterAuditError=String(error);}
   row.rawUsefulContract={...row.usefulContract};if(!row.adapterHttpAudit?.qualityPassed)row.usefulContract={firstThree:false,fullSix:false};
   row.protocol=verifyProtocol(row,expected,seen);if(item.kind==='loaded-profile-switch'&&!row.profileHistory?.passed){row.protocol.passed=false;row.protocol.failures.push('Actual visible profile/history panel differs from public seed');}if(!row.adapterHttpAudit?.protocolPassed){row.protocol.passed=false;row.protocol.failures.push('Independent current public-prompt/adapter/query/guard audit failed');}if(item.kind==='same-query-reload'&&!row.sameQueryRepeat?.identicalInput){row.protocol.passed=false;row.protocol.failures.push('Reload input differs from first work-bag public request');}
   if(row.protocol.requestId)seen.add(row.protocol.requestId);report.attempts.push(row);await save(join(output,`attempt-${item.index+1}.json`),row);
   if(!row.protocol.passed){report.stoppedBeforeNextRequest={index:item.index,reason:'Invalid/unsettled actual request; no replacement or continued contamination'};break;}
  }
  if(report.attempts.length===13&&report.attempts.every(r=>r.protocol.passed))report.functionalArms=await runFunctionalArms(page,{facts,requestFor,byId,seen,options,selectBucket:value=>{bucket=value;}});
  report.healthAfter=await health();assertHealth(report.healthAfter.body,selected.expected);report.sourceHashesAfter=await hashes(Object.keys(report.sourceHashes));
  report.sourcesUnchanged=canonical(report.sourceHashes)===canonical(report.sourceHashesAfter);
  report.startupUnchanged=canonical(report.healthBefore.body.startup)===canonical(report.healthAfter.body.startup);
  if(!report.sourcesUnchanged||!report.startupUnchanged)report.error='Source or native startup lifetime changed';
 }catch(error){report.error=String(error);}
 finally{for(const resource of [context,browser])if(resource)try{await resource.close();}catch(error){(report.teardownErrors??=[]).push(String(error));report.error??='Browser cleanup failed';}report.finishedAt=new Date().toISOString();report.summary={...summary(report.attempts),functional:report.functionalArms??[],functionalArmsExpected:2};await save(join(output,'report.json'),report);}
 return report;
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const options=parseArgs(process.argv.slice(2));if(!options.run)console.log(JSON.stringify(plan(),null,2));
 else{const report=await run(options);console.log(JSON.stringify(report.summary,null,2));if(report.error||report.summary.protocolPassed!==13||report.summary.loadedSwitchFirst3.valid!==10||report.summary.loadedSwitchFull6.valid!==10||report.functionalArms.length!==2||report.functionalArms.some(a=>!a.passed))process.exitCode=1;}
}
