import {chromium} from '@playwright/test';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import path from 'node:path';
const output=path.resolve(process.argv[2]);await mkdir(output,{recursive:false});
const report={version:'main-factorized-persona-launch-browser-smoke.v1',synthetic:true,qualityGatePassed:false,modelPromotion:false,v4ConditioningQualityValidated:false,checks:[],requests:[],pageErrors:[]};
let browser,context;
const need=(ok,name)=>{report.checks.push({name,passed:Boolean(ok)});if(!ok)throw Error(name);};
try{
 browser=await chromium.launch({channel:'chrome',headless:false});context=await browser.newContext({viewport:{width:1280,height:960},deviceScaleFactor:1});const page=await context.newPage();page.on('pageerror',e=>report.pageErrors.push(String(e)));
 async function capture(action){const promise=page.waitForResponse(r=>r.url()==='http://127.0.0.1:8767/personalize'&&r.request().method()==='POST'&&r.status()===200,{timeout:45000});await action();const response=await promise;const raw=await response.body();const events=raw.toString().trim().split('\n').map(JSON.parse);const rank=events.find(e=>e.type==='rank'),complete=events.find(e=>e.type==='complete');const request=response.request().postDataJSON();const cards=complete.sections.flatMap(s=>s.cards);const d=complete.provenance.personaRanking;
 need(cards.length===6&&complete.acceptedFields===20&&complete.withheldFields===0,'six cards and20 fresh accepted fields');
 need(d.queryFeature.cacheHit===false&&d.queryFeature.nativeBackboneForwardCalls===1,'actual fresh Qwen query forward');
 need(d.rankDecision.cacheHit===true&&d.rankDecision.nativeBodyForwards===0&&d.rankDecision.scoresOrRankingsOrResponseWordsCached===false,'pure profile reuse with fresh projection');
 await page.waitForFunction(cards=>cards.every((card,i)=>{const el=document.querySelectorAll('#collection article.product')[i];const paragraphs=[...el?.querySelectorAll('p')??[]].map(x=>x.textContent.trim());return el?.querySelector('h3')?.textContent.trim()===card.title&&paragraphs.includes(card.originalName)&&paragraphs.includes(card.badge)&&paragraphs.includes(card.bodyCopy)&&el.querySelector('.selected-color')?.textContent.trim()===card.selectedColor;}),cards,{timeout:20000});
 need(await page.locator('.person-trigger').count()===1,'actual persona switcher present');
 const i=report.requests.length;await writeFile(path.join(output,`request-${i+1}.ndjson`),raw);report.requests.push({request,requestId:rank.requestId,selectedProductIds:rank.selectedProductIds,acceptedFields:complete.acceptedFields,queryFeature:d.queryFeature,rankDecision:d.rankDecision,rawSha256:createHash('sha256').update(raw).digest('hex')});return request;
 }
 const first=await capture(()=>page.goto('http://127.0.0.1:4313/',{waitUntil:'domcontentloaded'}));
 need(first.query==='Browse the store'&&first.publicHistory.length>0,'bare main URL uses observed public history');
 const second=await capture(async()=>{await page.locator('#shop-search').fill('work bag');await page.locator('#shop-search').press('Enter');});
 need(second.query==='work bag'&&JSON.stringify(first.persona)===JSON.stringify(second.persona)&&JSON.stringify(first.publicHistory)===JSON.stringify(second.publicHistory),'new query keeps same public persona/history');
 need(report.requests[0].rankDecision.profileKeySha256===report.requests[1].rankDecision.profileKeySha256,'same profile key across different actual queries');
 need(JSON.stringify(report.requests[0].selectedProductIds)!==JSON.stringify(report.requests[1].selectedProductIds),'actual query changes selected order');
 await page.screenshot({path:path.join(output,'main-personalized.png'),fullPage:false});need(report.pageErrors.length===0,'zero browser page errors');
 report.sourceHashes={};for(const file of ['apps/storefront/src/main.tsx','experiments/ranker/pretrained/store_persona_adapter_runtime_v4.py','experiments/ranker/pretrained/store_persona_adapter_engine_v4.py'])report.sourceHashes[file]=createHash('sha256').update(await readFile(file)).digest('hex');
}catch(e){report.error=String(e);process.exitCode=1;}
finally{if(context)await context.close();if(browser)await browser.close();report.browserClosed=true;await writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:!report.error,checks:report.checks.length,requests:report.requests.length,error:report.error??null}));}
