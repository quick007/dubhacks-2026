import {chromium} from '@playwright/test';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {installPersonaFetchCapture,installPersonaReadiness,loadFacts,runDistinctQueries,querySummary,PUBLIC_PEOPLE,ENDPOINT} from '../experiments/ranker/pretrained/store_persona_factorized_browser_probe_v1.mjs';
import {installObserver} from '../experiments/ranker/pretrained/store_candidate_browser_probe_v1.mjs';
import {parseResponseBytes} from '../experiments/ranker/pretrained/store_candidate_collection_probe_v1.mjs';
const [out,receipt,digest,label]=process.argv.slice(2);if(!out||!receipt||!digest||!label)throw Error('Usage: OUTPUT RECEIPT SHA BEFORE_OR_AFTER');
const output=path.resolve(out);await mkdir(output);const sha=b=>createHash('sha256').update(b).digest('hex');
const files=['apps/storefront/src/use-route.ts','apps/storefront/src/main.tsx','apps/storefront/src/styles.css','experiments/ranker/pretrained/store_persona_adapter_engine_v4.py','experiments/ranker/pretrained/store_persona_adapter_runtime_v4.py','experiments/ranker/pretrained/store_persona_factorized_browser_probe_v1.mjs',path.relative(process.cwd(),import.meta.filename)];
const hashes=async()=>Object.fromEntries(await Promise.all(files.map(async f=>[f,sha(await readFile(f))])));
const health=async()=>{const response=await fetch('http://127.0.0.1:8767/health');if(!response.ok)throw Error('Model not ready');return response.json();};
const nativePid=()=>Number(execFileSync('lsof',['-tiTCP:8767','-sTCP:LISTEN'],{encoding:'utf8'}).trim());
const report={version:'search-results-navigation-browser.v1',label,synthetic:true,qualityGatePassed:false,modelPromotion:false,startedAt:new Date().toISOString(),scope:'Same frozen running model, ten fixed real submitted searches; copied original visibility/image/protocol observer, no injected results, no model restart or changes. Initial Browse request is declared untimed setup.'};let browser,context;
try{
 report.sourcesBefore=await hashes();report.healthBefore=await health();report.nativePidBefore=nativePid();
 const people=JSON.parse(await readFile(PUBLIC_PEOPLE,'utf8')).people;const {facts,requestFor}=await loadFacts();
 browser=await chromium.launch({channel:'chrome',headless:false});context=await browser.newContext({viewport:{width:1280,height:960},deviceScaleFactor:1});
 await context.addInitScript(installPersonaFetchCapture);await context.addInitScript(installObserver);await context.addInitScript(installPersonaReadiness);
 const page=await context.newPage();let bucket;
 page.on('request',request=>{if(bucket&&request.method()==='POST'&&request.url()===ENDPOINT)bucket.actualPostRequests.push({url:request.url(),payload:request.postDataJSON(),postData:request.postData()});});
 page.on('requestfailed',request=>bucket?.requestFailures.push({url:request.url(),error:request.failure()}));
 page.on('pageerror',error=>bucket?.pageErrors.push({message:error.message}));
 page.on('response',response=>{if(!bucket||response.url()!==ENDPOINT||response.request().method()!=='POST')return;const target=bucket;const receipt={status:response.status(),headers:response.headers(),captureStatus:'pending'};target.responses.push(receipt);target.pending.push((async()=>{try{const error=await response.finished();if(error)throw error;const parsed=parseResponseBytes(await response.body(),{source:'playwright-response-body'});if(target.open)Object.assign(receipt,parsed,{captureStatus:'complete'});}catch(error){if(target.open)Object.assign(receipt,{error:String(error),captureStatus:'error'});}})());});
 await page.goto('http://127.0.0.1:4313/',{waitUntil:'domcontentloaded'});await page.waitForFunction(()=>window.__personaReadiness?.stages?.complete&&window.__personaFetch.requests.every(r=>r.status!=='pending'),null,{timeout:30000});
 report.rows=await runDistinctQueries(page,{facts,requestFor,byId:new Map(people.map(p=>[p.id,p])),seen:new Set(),options:{out:output,candidateReceipt:path.resolve(receipt),candidateSha256:digest},selectBucket:value=>{bucket=value;}});
 report.summary=querySummary(report.rows);report.finalGeometry=await page.evaluate(()=>window.__personaReadiness.last);await page.screenshot({path:path.join(output,'submitted-search.png'),fullPage:false});
 report.sourcesAfter=await hashes();report.healthAfter=await health();report.nativePidAfter=nativePid();
 report.sourcesUnchanged=JSON.stringify(report.sourcesBefore)===JSON.stringify(report.sourcesAfter);report.nativeUnchanged=report.nativePidBefore===report.nativePidAfter&&JSON.stringify(report.healthBefore.startup)===JSON.stringify(report.healthAfter.startup);
 if(report.rows.length!==10||report.rows.some(r=>!r.protocol?.passed)||!report.sourcesUnchanged||!report.nativeUnchanged)throw Error('Incomplete protocol or source/native identity changed');
}catch(error){report.error=String(error);process.exitCode=1;}
finally{if(context)await context.close();if(browser)await browser.close();report.browserClosed=true;report.finishedAt=new Date().toISOString();await writeFile(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({label,error:report.error??null,summary:report.summary,nativeUnchanged:report.nativeUnchanged}));}
