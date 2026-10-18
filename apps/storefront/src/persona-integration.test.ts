import { test } from "node:test";
import assert from "node:assert/strict";
import { buildLocalModelRequest, localModelEnabled, localModelEndpoint, preserveLocalModelSearch,
  beginLocalModelRequest, applyLocalModelEvent, INITIAL_LOCAL_MODEL_STATE,
  type LocalRankEvent, type PersonalizationContext } from "./local-model";
import { createLocalModelClient } from "./local-model-client";
import { switchShoppingPerson, checkoutStillCurrent, runCurrentCheckout } from "./shopping-session";
import { PRODUCTS, DEFAULT_CONFIG } from "./config";
import { PRODUCT_VARIANTS } from "./variants";
import { parseRoute, routeHash } from "./routes";
import { CartView } from "./CartView";
import { ShopView } from "./ShopView";
import { personalizeStorefront } from "./personalization";
import { SHOPPER_PERSONAS, personaRequest, recentHistory } from "./personas";

const tick = () => new Promise<void>(resolve => setTimeout(resolve,5));
const context = (name: string): PersonalizationContext => ({ persona: { background: [`${name} rides to work`], colorPreferences: ["Sage"] },
  publicHistory: [{ eventId: `${name}-purchase`, eventType: "purchase", itemName: "Trail mug", category: "drinkware", selectedColor: "Sage",
    occurredAt: "2026-10-15T12:00:00Z", source: "demo-seed", productId: "mug", variantId: "mug-sage" }] });
function rank(requestId: string): LocalRankEvent {
  return { type: "rank", requestId, productIds: PRODUCTS.map(p => p.id), selectedProductIds: PRODUCTS.slice(0,6).map(p => p.id),
    variants: Object.fromEntries(PRODUCTS.slice(0,6).map(p => [p.id,PRODUCT_VARIANTS[p.id][0].id])), shortfall: 0,
    timing: { rankMs: 100 }, provenance: { ranking: "Learned order", images: "Store gallery" } };
}

test("public payload projects observed fields, retaining actual query and explicit constraints", () => {
  const supplied = { ...context("A"), query: "seed query", displayName: "Name not sent", targetProductIds: ["mug"], hiddenTruth: "secret", itemBudget: 80,
    hardRequirements: [{ attribute: "color", value: "Black" }] };
  Object.assign(supplied.publicHistory![0],{ desiredProduct: "pack", rationale: "hidden" });
  const request=buildLocalModelRequest(" exact product name ",supplied);
  assert.equal(request.query,"exact product name");assert.equal(request.itemBudget,80);
  assert.deepEqual(request.hardRequirements,[{attribute:"color",value:"Black"}]);
  for(const key of ["displayName","targetProductIds","hiddenTruth","desiredProduct","rationale"])assert.equal(JSON.stringify(request).includes(key),false);
  assert.equal(request.publicHistory?.[0].productId,"mug");assert.equal(buildLocalModelRequest("",context("A")).query,"Browse the store");
  request.persona!.background.push("new");assert.equal(supplied.persona!.background.length,1);
});

test("trusted personalized build opens ordinary shop; generic baseline and endpoints are explicit and local", () => {
  assert.equal(localModelEnabled("",true),true);assert.equal(localModelEndpoint("",true),"http://127.0.0.1:8767/personalize");
  assert.equal(localModelEnabled("?model=generic",true),false);assert.equal(localModelEnabled("",false),false);
  assert.equal(localModelEndpoint("?candidate=persona"),"http://127.0.0.1:8767/personalize");
  assert.equal(localModelEndpoint("?endpoint=https://untrusted.example"),"http://127.0.0.1:8766/personalize");
  assert.equal(preserveLocalModelSearch("headline=Hello","?model=local&candidate=persona"),"headline=Hello&model=local&candidate=persona");
});

test("person context clears old rank, copy and variants at same query before accepting fresh results", () => {
  let state=beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE,"Browse the store","A");
  state=applyLocalModelEvent(state,rank("a"),"Browse the store","A");
  const b=beginLocalModelRequest(state,"Browse the store","B",true);
  assert.equal(b.rank,null);assert.equal(b.copy,null);assert.equal(b.firstThreeReadyMs,null);assert.equal(b.displayQuery,"");
  assert.equal(applyLocalModelEvent(b,rank("late-a"),"Browse the store","A"),b);
  const fresh=applyLocalModelEvent(b,rank("b"),"Browse the store","B");assert.equal(fresh.rank?.requestId,"b");
  assert.equal(applyLocalModelEvent(fresh,{type:"error",requestId:"a",message:"old"},"Browse the store","A"),fresh);
});

test("rapid A→B→A cancels detached work and fetches fresh A without old-rank replay", async () => {
  const calls: {body:string;signal:AbortSignal;stream:ReadableStreamDefaultController<Uint8Array>}[]=[];
  const client=createLocalModelClient((async (_url,init)=>{
    let stream!:ReadableStreamDefaultController<Uint8Array>;
    const body=new ReadableStream<Uint8Array>({start(controller){stream=controller;}});
    calls.push({body:String(init?.body),signal:init!.signal as AbortSignal,stream});return new Response(body);
  }) as typeof fetch);
  const a=buildLocalModelRequest("same query",context("A")),b=buildLocalModelRequest("same query",context("B"));
  const oldEvents:string[]=[],bEvents:string[]=[],newEvents:string[]=[],errors:string[]=[];
  const error=(e:Error)=>errors.push(e.message);
  const send=(index:number,id:string)=>calls[index].stream.enqueue(new TextEncoder().encode(JSON.stringify(rank(id))+"\n"));
  const stopA=client.subscribe(a,{event:e=>oldEvents.push(e.type==="error"?"error":e.requestId),error});
  await tick();send(0,"old-a");await tick();assert.deepEqual(oldEvents,["old-a"]);
  stopA();const stopB=client.subscribe(b,{event:e=>bEvents.push(e.type),error});
  assert.equal(calls[0].signal.aborted,true);await tick();
  stopB();const stopNewA=client.subscribe(a,{event:e=>newEvents.push(e.type==="error"?"error":e.requestId),error});
  assert.equal(calls[1].signal.aborted,true);assert.deepEqual(newEvents,[]);await tick();
  assert.equal(calls.length,3);assert.equal(calls[0].body,calls[2].body);assert.notEqual(calls[0].body,calls[1].body);
  send(0,"late-old-a");send(1,"late-b");send(2,"fresh-a");await tick();
  assert.deepEqual(newEvents,["fresh-a"]);assert.deepEqual(bEvents,[]);assert.deepEqual(errors,[]);
  stopNewA();await tick();assert.equal(calls[2].signal.aborted,true);for(const call of calls)call.stream.close();
});

test("per-person carts survive switching but stale checkout receipts and in-flight responses cannot follow", () => {
  const aCart=[{productId:"mug" as const,variantId:PRODUCT_VARIANTS.mug[0].id}],bCart=[{productId:"pack" as const,variantId:PRODUCT_VARIANTS.pack[1].id}];
  const a={personId:"A",cart:aCart,carts:{B:bCart},checkoutReceipt:{receiptId:"a-receipt"}};
  const b=switchShoppingPerson(a,"B");assert.deepEqual(b.cart,bCart);assert.equal(b.checkoutReceipt,null);
  const returned=switchShoppingPerson(b,"A");assert.deepEqual(returned.cart,aCart);assert.equal(returned.checkoutReceipt,null);
  assert.equal(checkoutStillCurrent("old-a","new-b"),false);assert.equal(checkoutStillCurrent("old-a","new-a"),false);
  assert.equal(checkoutStillCurrent("current","current"),true);assert.equal(switchShoppingPerson(a,"A"),a);
});

test("cart and return navigation preserve current search and collection", () => {
  const shop={page:"shop" as const,query:"exact name & blue",collectionId:"bags"};
  const cart=parseRoute(routeHash({...shop,page:"cart"}));assert.equal(cart.query,shop.query);assert.equal(cart.collectionId,"bags");
  assert.deepEqual(parseRoute(routeHash({...cart,page:"shop"})),shop);
});

test("shopping surfaces keep model telemetry internal and normal cart naming", () => {
  const model={...beginLocalModelRequest(INITIAL_LOCAL_MODEL_STATE,"work bag"),phase:"complete" as const,rank:rank("r")};
  const shop=ShopView({config:DEFAULT_CONFIG,experience:personalizeStorefront("work bag",DEFAULT_CONFIG),collectionId:"all",query:"work bag",
    searchInput:"work bag",visibleProducts:[...PRODUCTS],selectedVariants:{},added:"",localModel:model,personName:"Alex",onSearchInput(){},onSearchSubmit(){},onExampleSearch(){},onClearSearch(){},onCollectionChange(){},onVariantChange(){},onAdd(){}});
  const serialized=JSON.stringify(shop);assert.match(serialized,/Suggestions for/);assert.doesNotMatch(serialized,/Learned product order|acceptedFields|fresh learned fields/);
  const cart=CartView({lines:[],subtotal:0,quote:null,total:0,onPlaceOrder(){},onContinue(){}});
  assert.match(JSON.stringify(cart),/Your cart/);assert.doesNotMatch(JSON.stringify(cart),/Your demo cart/);
});

test("all ten public people preserve real query, give distinct observed contexts and recent dated history", () => {
  assert.equal(SHOPPER_PERSONAS.length,10);const requests=SHOPPER_PERSONAS.map(person=>personaRequest(person,"same query"));
  assert.equal(new Set(requests.map(value=>JSON.stringify(value))).size,10);
  for(const [i,person] of SHOPPER_PERSONAS.entries()){
    assert.equal(requests[i].query,"same query");assert.equal(Object.hasOwn(requests[i],"id"),false);
    const recent=recentHistory(person);assert.equal(recent.length,3);
    assert.ok(recent.every(event=>Number.isFinite(Date.parse(event.occurredAt))));
    assert.deepEqual(recent,[...recent].sort((a,b)=>b.occurredAt.localeCompare(a.occurredAt)));
  }
});


test("late checkout success and rejection after switching never confirm, alert or finish the new cart", async () => {
  for (const outcome of ["success", "failure"] as const) {
    let activeId = "A-old", receipt: string | null = null, pending = true;
    const callbacks: string[] = [];
    let resolveOld!: (value: string) => void, rejectOld!: (value: Error) => void;
    const old = new Promise<string>((resolve,reject) => {resolveOld=resolve;rejectOld=reject;});
    const running = runCurrentCheckout(() => old, () => checkoutStillCurrent("A-old", activeId), {
      confirmed: value => {receipt=value;callbacks.push("old-confirmed");},
      failed: () => {callbacks.push("old-failed");}, settled: () => {pending=false;callbacks.push("old-settled");},
    });
    activeId="A-new-after-B"; // A→B→A is a new authority even though the display identity matches.
    if(outcome==="success")resolveOld("old receipt");else rejectOld(new Error("old failure"));
    await running;assert.equal(receipt,null);assert.equal(pending,true);assert.deepEqual(callbacks,[]);
    await runCurrentCheckout(async () => "new receipt", () => checkoutStillCurrent("A-new-after-B",activeId), {
      confirmed: value => {receipt=value;}, failed: () => assert.fail("new checkout failed"), settled: () => {pending=false;},
    });
    assert.equal(receipt,"new receipt");assert.equal(pending,false);
  }
});
