"""Pinned tokenizer only; never imports torch, transformers, MLX, model or private data."""
from pathlib import Path
import argparse,hashlib,json,os
os.environ['TOKENIZERS_PARALLELISM']='false'
from tokenizers import Tokenizer
from validate import HERE,read,public_context_text
p=argparse.ArgumentParser();p.add_argument('--tokenizer',type=Path,required=True);args=p.parse_args()
tokenizer=Tokenizer.from_file(str(args.tokenizer));profiles=read('profiles.train.public.json')['profiles'];rows=[]
for profile in profiles:
 text=public_context_text(profile['request']);ids=tokenizer.encode(text,add_special_tokens=False).ids
 if not 0<len(ids)<=512:raise ValueError(f'{profile["profileId"]}: {len(ids)} tokens exceeds predeclared compact context; no truncation')
 rows.append({'profileId':profile['profileId'],'tokens':len(ids),'textSha256':hashlib.sha256(text.encode()).hexdigest(),'tokenIdsSha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()})
report={'status':'cpu-whole-public-context-tokenizer-pass','profiles':len(rows),'tokenizerFile':str(args.tokenizer.resolve()),'tokenizerSha256':hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),'publicProfilesSha256':hashlib.sha256((HERE/'profiles.train.public.json').read_bytes()).hexdigest(),'formatterScope':'Exact public observations serialization, ageDays, no event/profile/session IDs or absolute dates; same public-context format as existing Qwen representation','maxTokens':512,'actualTokenRange':[min(r['tokens'] for r in rows),max(r['tokens'] for r in rows)],'truncatedRows':0,'gpuUsed':False,'modelLoaded':False,'privateLabelsRead':False,'rows':rows}
out=HERE/'TOKEN_PREFLIGHT.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
