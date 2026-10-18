"""CPU-only bridge joining actual expanded35 browser/HTTP receipts."""
import json,sys
from . import store_expanded35_verification_v1 as audit
from .store_expanded35_backend_probe_v1 import actual_receipt,TOKENIZER

def main():
 p=json.load(sys.stdin);e=audit.startup_evidence(p['startup'],p['startupSha256'],p['bindingManifest'])
 from transformers import AutoTokenizer
 e['tokenizer']=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True,trust_remote_code=False)
 if sys.argv[1]=='selection':print(json.dumps({k:v for k,v in e.items() if k not in ('vocabulary','tokenizer')},allow_nan=False));return
 receipt,reference=actual_receipt(p['serverOutput'],p['headers'],p['events'],p['request']);result=audit.audit_events(p['request'],p['events'],p['headers'],e,receipt)
 print(json.dumps({'audit':result,'serverReceipt':reference,'actualWordSlotFields':receipt.get('parallelGeneration',{}).get('fields',[])},allow_nan=False))
if __name__=='__main__':main()
