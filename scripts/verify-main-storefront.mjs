import { chromium } from '@playwright/test';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';

// A local integration check with an actual simulated order; no request mocks.
const out=resolve(process.argv[3]??'');
if(process.argv[2]!=='--run'||!process.argv[3])throw Error('Explicit --run NEW_OUTPUT_DIRECTORY required');
await mkdir(out);
const report={version:'main-storefront-acceptance.v2',synthetic:true,realPurchase:false,
  startedAt:new Date().toISOString(),checks:[],pageErrors:[],modelRequests:[],checkoutResponses:[]};
const check=(name,passed)=>{report.checks.push({name,passed});if(!passed)throw Error(name);};
const sha=b=>createHash('sha256').update(b).digest('hex');
let browser,context;
try{
  const source='apps/storefront/src/main.tsx';report.source={file:source,sha256:sha(await readFile(source))};
  browser=await chromium.launch({channel:'chrome',headless:false});
  context=await browser.newContext({viewport:{width:1280,height:960},deviceScaleFactor:1});
  const page=await context.newPage();
  page.on('pageerror',e=>report.pageErrors.push(String(e)));
  page.on('request',r=>{if(/127\.0\.0\.1:876[67]/.test(r.url()))report.modelRequests.push({url:r.url(),method:r.method()});});
  page.on('response',r=>{if(r.url().includes('/api/demo-checkout'))report.checkoutResponses.push({url:r.url(),status:r.status()});});
  await page.goto('http://127.0.0.1:4312/',{waitUntil:'networkidle'});
  check('generic catalog renders all16 products',await page.locator('#collection article.product').count()===16);
  check('generic mode has no persona switcher',await page.locator('.person-trigger').count()===0);
  await page.screenshot({path:out+'/generic.png'});
  await page.locator('#shop-search').fill('bottle');
  await page.locator('.shop-search button[type=submit]').click();
  const card=page.locator('article.product').filter({has:page.getByRole('heading',{name:'Everyday steel bottle',exact:true})});
  await card.waitFor();check('search preserves real bottle identity',await card.count()===1);
  await card.getByRole('button',{name:'Add to cart',exact:true}).click();
  await page.getByRole('button',{name:'View cart',exact:true}).click();
  await page.getByRole('heading',{name:'Your cart',exact:true}).waitFor();
  check('cart contains one source-priced bottle',(await page.locator('.cart-items').innerText()).includes('$36'));
  await page.screenshot({path:out+'/cart.png'});
  const posted=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/api/demo-checkout'));
  await page.getByRole('button',{name:'Confirm order',exact:true}).click();
  const response=await posted;report.post=await response.json();
  check('actual local checkout accepted202',response.status()===202);
  await page.getByRole('heading',{name:'Order confirmed',exact:true}).waitFor();
  const main=page.locator('main');report.visible={checkoutId:await main.getAttribute('data-demo-checkout-id'),receiptId:await main.getAttribute('data-demo-receipt-id')};
  check('visible confirmation is bound to real receipt',report.visible.checkoutId===report.post.receipt.checkoutId&&report.visible.receiptId===report.post.receipt.receiptId);
  const fresh=await fetch('http://127.0.0.1:4331/api/demo-checkout/'+report.visible.checkoutId);
  report.freshGet={status:fresh.status,body:await fresh.json()};
  check('independent fresh server read matches receipt',fresh.status===200&&JSON.stringify(report.freshGet.body.receipt)===JSON.stringify(report.post.receipt));
  check('source amount and fake-order boundary preserved',report.post.receipt.items.length===1&&report.post.receipt.items[0].productId==='bottle'&&report.post.receipt.items[0].quantity===1&&report.post.receipt.subtotal===36&&report.post.receipt.realPurchase===false);
  await page.screenshot({path:out+'/confirmed.png'});
  await page.getByRole('button',{name:'Back to the collection',exact:true}).click();
  await page.locator('#collection').waitFor();
  report.returnUrl=page.url();
  check('return navigation works',await page.locator('#shop-search').isVisible()
    &&await page.locator('#shop-search').inputValue()==='bottle'
    &&await page.getByRole('heading',{name:'Order confirmed',exact:true}).count()===0);
  check('generic flow sends zero model requests',report.modelRequests.length===0);
  check('no browser application errors',report.pageErrors.length===0);
}catch(error){report.error=String(error);process.exitCode=1;}
finally{
  for(const value of [context,browser])if(value)await value.close();
  report.finishedAt=new Date().toISOString();report.passed=!report.error&&report.checks.every(c=>c.passed);
  await writeFile(out+'/report.json',JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({passed:report.passed,checks:report.checks,error:report.error},null,2));
}
