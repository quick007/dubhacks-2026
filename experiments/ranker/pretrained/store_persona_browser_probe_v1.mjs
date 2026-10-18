import { createRequire } from 'node:module';
import { mkdir, readFile, writeFile, access } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { installObserver, parsedSourceFieldsMatch, matchDomToCopy } from './store_candidate_browser_probe_v1.mjs';
import { parseResponseBytes, selectResponseEvidence } from './store_candidate_collection_probe_v1.mjs';
import { CONTEXT_OPTIONS, settleNavigationCaptures } from './store_candidate_browser_probe_v2.mjs';

const ROOT=resolve(dirname(fileURLToPath(import.meta.url)),'../../..');
export const ORIGIN='http://127.0.0.1:4313';
export const ENDPOINT='http://127.0.0.1:8767/personalize';
export const PUBLIC_PEOPLE='experiments/ranker/pretrained/synthetic_history_v1/demo-people.json';
export const PROFILE_ORDER=Object.freeze(['maya','priya','lena','june','grace','sam','ana','rosa','felix','bob']);
export const TIMEOUT_MS=30000;
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const canonical=value=>JSON.stringify(value,(_key,v)=>v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.entries(v).sort(([a],[b])=>a.localeCompare(b))):v);
const save=(path,value)=>writeFile(path,JSON.stringify(value,null,2)+'\n',{flag:'wx'});

export function fixedSchedule(){return[
  {index:0,kind:'bare-navigation',personId:'bob',query:'Browse the store',navigation:'goto',preScrolled:false},
  ...PROFILE_ORDER.map((personId,i)=>({index:i+1,kind:'loaded-profile-switch',personId,query:'Browse the store',navigation:null,preScrolled:true})),
  {index:11,kind:'first-work-bag-query',personId:'bob',query:'work bag',navigation:null,preScrolled:true},
  {index:12,kind:'same-query-reload',personId:'bob',query:'work bag',navigation:'reload',preScrolled:false},
];}
export function plan(){return{version:'store-persona-browser-probe.v1',origin:ORIGIN,endpoint:ENDPOINT,
  publicPeople:PUBLIC_PEOPLE,schedule:fixedSchedule(),fixedRequests:13,profileDenominator:10,
  viewport:CONTEXT_OPTIONS.viewport,deviceScaleFactor:1,headed:true,timeoutMs:TIMEOUT_MS,bodySettlementMs:5000,
  execution:'CPU plan by default. --run and a NEW --out directory required. Existing installed Chrome and ready accepted persona backend only; never starts servers or model workers.',
  workload:'One bare URL navigation at actual initial scroll. Then10 different public profile selections in one loaded collection, pre-scroll before each. Profile clock is real option click; real Escape dismisses popover before useful visibility. EndBob, submit work bag via real search form, then reload that route for another genuine POST. No route/fetch/text injection or forced same-hash request.',
  timing:'Native startup and browser launch excluded. Browser actual fetch→endpoint is common model-request clock; action→endpoint and navigation→endpoint separately retained. Existing React/two-rAF mirror plus actual geometry and image complete/decode; no physical paint claim. Passive clone overhead is part of this instrumented workload; body settlement occurs after UI endpoints.',
  cache:'NEW browser context starts HTTP cache cold for bare navigation. Loaded switches/search reuse same browser cache. Reload is HTTP-cache eligible with browser scroll restoration recorded, not a second cold navigation. Native query/context cache state is actual raw provenance, never assumed cold. All10 public context features may already be precomputed; first work-bag means first query in this probe, not guaranteed native cache miss. Reload has identical public payload and a unique request ID; fresh rank/variants/neural copy required even on feature cache hits.',
  visibility:'Initial viewport reports actual useful-card count and first3 only if truly fully visible. Loaded first3 requires current request accepted10 fields, three whole cards/titles/copy and decoded primary images, with no open popover or intersecting fixed switcher. The actual .person-trigger rectangle is checked against the whole card bounding rectangle plus title, each copy paragraph, clipped primary image and Add button when present; any intersection rejects useful visibility. This strict whole-card policy also rejects overlap in otherwise unused card space. Full6 requires accepted20 fields and six decoded primary images separately; six-card visibility is never implied. Shortfalls/zero results remain explicit, never padded.',
  facts:'All actual copy fields and DOM strings are retained. Canonical public catalog validates original name, price, chosen variant/color and image-source identity; every accepted-field total and DOM copy string is checked. Each selected card visible DOM price must equal both its canonical price and actual payload itemPrice, with missing prices rejected. This is not a free-text semantic factuality or unseen-personalization-quality evaluation; no private labels or answers are read.',
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
  const entry={url,method,requestBody:typeof init?.body==='string'?init.body:null,startedAt:performance.now(),status:'pending',bodyBytes:null};capture.requests.push(entry);
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
  if(kind==='profile'&&event.target.closest?.('.person-options button')?.textContent.trim().startsWith(settings.displayName)){
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
 const failures=[];if(row.error||row.captureError)failures.push('Workload/capture error');
 if(row.actualPostRequests?.length!==1||canonical(row.actualPostRequests?.[0]?.payload)!==canonical(expectedRequest))failures.push('Requires one exact public UI request');
 if(!row.bodySettlement?.settled||!row.cloneCaptureSettlement?.settled)failures.push('Actual body capture unsettled');
 if(!row.responseEvidence?.selectedSource||row.responseEvidence.pathsAgree===false)failures.push('Missing or disagreeing actual response bytes');
 const events=row.responseEvidence?.events??[],rank=events.find(e=>e.type==='rank'),complete=events.find(e=>e.type==='complete');
 const id=rank?.requestId;if(events.some(e=>e.type==='error'))failures.push('Actual native stream contains an error');
 const encoding=rank?.provenance?.featureEncoding;
 if(typeof encoding?.queryFeatureCacheHit!=='boolean'||typeof encoding?.contextFeatureCacheHit!=='boolean'
  ||!Number.isInteger(encoding.queryNativeForwards)||!Number.isInteger(encoding.contextNativeForwards))failures.push('Missing actual query/context feature cache and native-forward receipts');
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

export function parseArgs(args){const options={run:false,out:null,manifest:null};
 for(let i=0;i<args.length;i++){if(args[i]==='--run')options.run=true;
  else if(args[i]==='--out'||args[i]==='--manifest'){const key=args[i].slice(2);if(!args[i+1]||args[i+1].startsWith('--'))throw Error(`Missing ${args[i]} value`);options[key]=args[++i];}
  else throw Error(`Unknown argument ${args[i]}`);}
 if(options.run&&!options.out)throw Error('--run requires a unique --out directory');return options;}
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
async function hashes(paths){return Object.fromEntries(await Promise.all(paths.map(async p=>[p,sha(await readFile(join(ROOT,p)))])));}
async function inventory(manifest){const fixed=['experiments/ranker/pretrained/store_persona_browser_probe_v1.mjs','experiments/ranker/pretrained/test_store_persona_browser_probe_v1.mjs','experiments/ranker/pretrained/STORE_PERSONA_BROWSER_PROBE_V1_PLAN.json',PUBLIC_PEOPLE,
 'apps/storefront/package.json','apps/storefront/vite.config.ts',
 'experiments/ranker/pretrained/store_candidate_browser_probe_v1.mjs','experiments/ranker/pretrained/store_candidate_browser_probe_v2.mjs','experiments/ranker/pretrained/store_candidate_collection_probe_v1.mjs',
 'experiments/ranker/pretrained/store_persona_engine_v1.py','experiments/ranker/pretrained/store_persona_runtime_v1.py','experiments/ranker/pretrained/store_persona_context_v1.py',
 'experiments/ranker/pretrained/store_candidate_engine_v1.py','experiments/ranker/pretrained/store_candidate_image_service_v1.py','experiments/ranker/pretrained/store_demo_engine.py','experiments/ranker/pretrained/store_demo_service.py','experiments/ranker/pretrained/store_demo_image_binding_v1.py','experiments/ranker/pretrained/store_rank_session_v5.py',
 ...['main.tsx','ShopView.tsx','PersonaSwitcher.tsx','personas.ts','local-model.ts','local-model-client.ts','use-local-model.ts','local-model-performance.ts','ProductCard.tsx','ProductGallery.tsx','ProductPhoto.tsx','variants.ts','images.ts','config.ts','catalog-expansion.ts','styles.css','product-responsive-runtime.ts','product-image-runtime.json','color-gallery-layouts.json'].map(p=>`apps/storefront/src/${p}`)];
 if(manifest){const content=JSON.parse(await readFile(join(ROOT,manifest),'utf8'));fixed.push(manifest,...Object.keys(content.frontendSha256??{}),...Object.keys(content.verificationSourcesSha256??{}));}
 for(const path of ['experiments/ranker/pretrained/store_persona_qwen_v1.py','experiments/ranker/pretrained/store_persona_qwen_fit_v1.py'])try{await access(join(ROOT,path));fixed.push(path);}catch{}
 return[...new Set(fixed)].sort();}

export async function run(options){const output=resolve(options.out);await mkdir(output);
 const report={...plan(),startedAt:new Date().toISOString(),attempts:[]};let browser,context;
 try{const people=JSON.parse(await readFile(join(ROOT,PUBLIC_PEOPLE),'utf8')).people;const byId=new Map(people.map(p=>[p.id,p]));
  if(people[0]?.id!=='bob'||PROFILE_ORDER.some(id=>!byId.has(id)))throw Error('Fixed public roster changed');
  const{facts,requestFor}=await loadFacts();report.publicCatalog=facts;report.sourceHashes=await hashes(await inventory(options.manifest));
  report.healthBefore=await health();if(!report.healthBefore.body.ready||!report.healthBefore.body.startup)throw Error('Accepted persona backend must already be ready');
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
  page.on('response',response=>{if(!bucket||response.url()!==ENDPOINT||response.request().method()!=='POST')return;const target=bucket;const receipt={status:response.status(),captureStatus:'pending'};target.responses.push(receipt);
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
      await option.click();await page.keyboard.press('Escape');
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
    row.cache={featureEncoding:rank?.provenance?.featureEncoding??null,contextSource:rank?.provenance?.personaRanking?.contextFeature??null,querySource:rank?.provenance?.personaRanking?.queryFeature??null,copySource:complete?.copySource??null};
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
   row.protocol=verifyProtocol(row,expected,seen);if(item.kind==='same-query-reload'&&!row.sameQueryRepeat?.identicalInput){row.protocol.passed=false;row.protocol.failures.push('Reload input differs from first work-bag public request');}
   if(row.protocol.requestId)seen.add(row.protocol.requestId);report.attempts.push(row);await save(join(output,`attempt-${item.index+1}.json`),row);
   if(!row.protocol.passed){report.stoppedBeforeNextRequest={index:item.index,reason:'Invalid/unsettled actual request; no replacement or continued contamination'};break;}
  }
  report.healthAfter=await health();report.sourceHashesAfter=await hashes(Object.keys(report.sourceHashes));
  report.sourcesUnchanged=canonical(report.sourceHashes)===canonical(report.sourceHashesAfter);
  report.startupUnchanged=canonical(report.healthBefore.body.startup)===canonical(report.healthAfter.body.startup);
  if(!report.sourcesUnchanged||!report.startupUnchanged)report.error='Source or native startup lifetime changed';
 }catch(error){report.error=String(error);}
 finally{for(const resource of [context,browser])if(resource)try{await resource.close();}catch(error){(report.teardownErrors??=[]).push(String(error));report.error??='Browser cleanup failed';}report.finishedAt=new Date().toISOString();report.summary=summary(report.attempts);await save(join(output,'report.json'),report);}
 return report;
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const options=parseArgs(process.argv.slice(2));if(!options.run)console.log(JSON.stringify(plan(),null,2));
 else{const report=await run(options);console.log(JSON.stringify(report.summary,null,2));if(report.error||report.summary.protocolPassed!==13||report.summary.loadedSwitchFirst3.valid!==10||report.summary.loadedSwitchFull6.valid!==10)process.exitCode=1;}
}
