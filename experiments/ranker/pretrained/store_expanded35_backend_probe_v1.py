"""Bounded16-request actual expanded35 backend verification. Default CPU plan.
Root-only execution; no native model/service launch. Local checkout is simulated.
"""
import argparse,json,time,urllib.request,uuid
from pathlib import Path
from . import store_expanded35_verification_v1 as audit
TOKENIZER='/Users/lukas/Documents/Projects/dubhacks-2026/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots/8b2b98c00a6b4d291155e4890773ca8f769aee53'
ENDPOINT='http://127.0.0.1:8767'
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def exchange(url,payload=None):
    data=None if payload is None else json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode();req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json','Origin':'http://127.0.0.1:4313'},method='GET' if data is None else 'POST');started=time.perf_counter()
    with urllib.request.urlopen(req,timeout=45) as response:raw=response.read();return {'status':response.status,'headers':{k.lower():v for k,v in response.headers.items()},'raw':raw,'clientExchangeMs':(time.perf_counter()-started)*1000}
def actual_receipt(server_output,headers,events,request):
    path=Path(server_output)/'requests'/headers['x-request-id']/'receipt.json'
    for _ in range(40):
        if path.exists() and json.loads(path.read_text()).get('status')=='completed':break
        time.sleep(.05)
    receipt=json.loads(path.read_text())
    if receipt.get('input')!=request or receipt.get('events')!=events or receipt.get('status')!='completed':raise ValueError('Actual completed server receipt/request/rawbody mismatch')
    return receipt,{'file':str(path),'sha256':audit.sha(path)}
def checkout_probe(api):
    checkout_id=str(uuid.uuid4());request={'schema':'fieldwork.demo-checkout.v1','checkoutId':checkout_id,'shippingVisible':True,'items':[{'productId':'work-tote','variantId':'work-tote-navy','quantity':2}]};post=exchange(api+'/api/demo-checkout',request);body=json.loads(post['raw']);get=exchange(api+'/api/demo-checkout/'+checkout_id);read=json.loads(get['raw']);receipt=body.get('receipt',{})
    checks={'post202':post['status']==202,'get200':get['status']==200,'newNonduplicate':body.get('duplicate') is False,'identicalDurableReceipt':read.get('receipt')==receipt,'newCatalogID':receipt.get('items')==[{'productId':'work-tote','variantId':'work-tote-navy','quantity':2,'unitPrice':74,'lineTotal':148}],'canonicalTotals':receipt.get('subtotal')==receipt.get('total')==148 and receipt.get('shipping')=={'status':'free','cost':0},'simulatedOnly':receipt.get('realPurchase') is False}
    return {'request':request,'post':{**post,'raw':post['raw'].decode()},'get':{**get,'raw':get['raw'].decode()},'checks':checks,'passed':all(checks.values()),'scope':'Separate actual local simulated checkoutPOST+durableGET, no model/browser timing claim'}
def summarize(rows):
    result={}
    for kind,n in [('warm-public-Browse',10),('category-guard',3),('novel-query-miss',3)]:
        chosen=[r for r in rows if r['kind']==kind];result[kind]={'recorded':len(chosen),'serverFullAvailable':audit.distribution([r.get('audit',{}).get('serverTimings',{}).get('complete',{}).get('readyMs') if r.get('audit',{}).get('passed') else None for r in chosen],n),'serverFullSix':audit.distribution([r.get('audit',{}).get('serverTimings',{}).get('complete',{}).get('readyMs') if r.get('audit',{}).get('passed') and r['audit'].get('all20Accepted') else None for r in chosen],n),'queryCache':[{'hit':r.get('audit',{}).get('queryCacheHit'),'nativeForwards':r.get('audit',{}).get('queryNativeForwards'),'actualCards':r.get('audit',{}).get('actualCards'),'shortfall':r.get('audit',{}).get('shortfall')} for r in chosen]}
    return result

def run(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);schedule=[r for r in audit.fixed_schedule() if args.workload=='all' or r['kind']!='novel-query-miss'];report={'version':'expanded35-backend-probe.v1','workload':args.workload,'startedAt':time.time(),'fixedSchedule':schedule,'attempts':[],'modelPromotion':False,'startupExcluded':True,'timingScope':'Complete backend rank/freshwordcopy/validation/serialization +HTTPexchange; no browser/image claim. Cacheactual0/1native disclosed.'}
    try:
        evidence=audit.startup_evidence(args.startup,args.startup_sha256,args.binding_manifest)
        from transformers import AutoTokenizer
        evidence['tokenizer']=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True,trust_remote_code=False)
        report['sourceHashesBefore']={**evidence['pins'],str(Path(__file__)):audit.sha(__file__),str(Path(audit.__file__)):audit.sha(audit.__file__)}
        report['healthBefore']=json.loads(exchange(ENDPOINT+'/health')['raw']);report['startup']=evidence['startup']
        if report['healthBefore'].get('ready') is not True or report['healthBefore'].get('startup')!=evidence['startup']:raise ValueError('Actual ready service must equal sealed35 startup')
        seen=set()
        for case in report['fixedSchedule']:
            row={**case}
            try:
                wire=exchange(ENDPOINT+'/personalize',case['request']);(out/f"request-{case['index']}.ndjson").write_bytes(wire['raw']);events=[json.loads(x) for x in wire['raw'].splitlines() if x.strip()];receipt,ref=actual_receipt(args.server_output,wire['headers'],events,case['request']);row.update(events=events,headers=wire['headers'],clientExchangeMs=wire['clientExchangeMs'],rawBodySha256=__import__('hashlib').sha256(wire['raw']).hexdigest(),serverReceipt=ref,audit=audit.audit_events(case['request'],events,wire['headers'],evidence,receipt,seen))
                if case['kind']=='warm-public-Browse' and row['audit']['queryCacheHit'] is not True:row['cacheExpectationFailure']='Known defaultBrowse expectedactualwarmhit'
                if case['kind']=='novel-query-miss' and row['audit']['queryCacheHit'] is not False:row['cacheExpectationFailure']='Distinct novelquery alreadycached; coldmissclaimnotestablished'
            except Exception as error:row['error']=repr(error)
            report['attempts'].append(row);save(out/f"request-{case['index']}.json",row)
            if row.get('error') or not row.get('audit',{}).get('passed'):break
        if args.checkout:report['checkout']=checkout_probe(args.api)
        report['healthAfter']=json.loads(exchange(ENDPOINT+'/health')['raw']);report['startupUnchanged']=report['healthAfter'].get('startup')==evidence['startup'];report['sourceHashesAfter']={p:audit.sha(p) for p in report['sourceHashesBefore']};report['sourcesUnchanged']=report['sourceHashesBefore']==report['sourceHashesAfter']
    except Exception as error:report['error']=repr(error)
    report['summary']=summarize(report['attempts']);report['completed']=len(report['attempts'])==len(schedule) and all(r.get('audit',{}).get('passed') and not r.get('cacheExpectationFailure') for r in report['attempts']) and report.get('startupUnchanged') and report.get('sourcesUnchanged') and (not args.checkout or report.get('checkout',{}).get('passed'));save(out/'report.json',report);print(json.dumps({'out':str(out),'completed':bool(report['completed']),'error':report.get('error'),'summary':report['summary']}));return report

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',action='store_true');p.add_argument('--out');p.add_argument('--startup');p.add_argument('--startup-sha256');p.add_argument('--binding-manifest');p.add_argument('--server-output');p.add_argument('--workload',choices=['core','all'],default='core');p.add_argument('--checkout',action='store_true');p.add_argument('--api',default='http://127.0.0.1:4331');a=p.parse_args()
    if not a.run:print(json.dumps({'version':'expanded35-backend-probe.v1','cpuOnlyPlan':True,'catalogSha256':audit.CATALOG_SHA,'schedule':audit.fixed_schedule(),'checkoutOptionalSeparate':True},indent=2));return
    if not all((a.out,a.startup,a.startup_sha256,a.binding_manifest,a.server_output)):p.error('Unique --out plusactual --startup/--startup-sha256/--binding-manifest/--server-output required')
    if a.api!='http://127.0.0.1:4331':p.error('Only explicitlydeclared localcheckout4331 supported')
    raise SystemExit(0 if run(a)['completed'] else 1)
if __name__=='__main__':main()
