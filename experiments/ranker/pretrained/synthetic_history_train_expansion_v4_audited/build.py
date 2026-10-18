"""Reproduce the bounded TRAIN-only repair from immutable v3 and independent audit."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,re
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'synthetic_history_train_expansion_v3'
AUDIT=HERE.parent/'SYNTHETIC_HISTORY_TRAIN_EXPANSION_V3_INDEPENDENT_AUDIT.json'
CATALOG=HERE.parent/'synthetic_history_v1/catalog-binding.json'
SCHEMA='authored-public-persona-train-expansion.v4-audited'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def verify_sources():
 plan=read(HERE/'CORRECTION_PLAN.json')
 assert sha(PARENT/'manifest.json')==plan['parentManifestSha256']
 assert sha(AUDIT)==plan['independentAuditSha256']
 assert sha(PARENT/'profiles.train.public.json')==plan['publicSha256Unchanged']
 for name,digest in read(HERE/'V3_TREE_FREEZE.json')['files'].items():assert sha(PARENT/name)==digest,(name,'v3 changed')
 manifest=read(PARENT/'manifest.json')
 for name,digest in manifest['files'].items():assert sha(PARENT/name)==digest
 assert sha(CATALOG)==manifest['sourceSha256']['experiments/ranker/pretrained/synthetic_history_v1/catalog-binding.json']
 return plan

def expressed_colors(request,products):
 """Exact canonical color phrases in public list order, longest name first.

 Longest matching avoids fabricating Sage from Sage stripe or Navy from Navy band.
 This parser is TRAIN annotation evidence, not a runtime recommendation function.
 """
 colors={v['colorName'] for p in products.values() for v in p['variants']}
 names=sorted(colors,key=lambda c:(-len(c),c))
 pattern=re.compile(r'(?<!\w)(?:'+ '|'.join(re.escape(c) for c in names)+r')(?!\w)',re.I)
 canonical={c.casefold():c for c in colors}
 return [canonical[m.group().casefold()] for text in request['persona']['colorPreferences'] for m in pattern.finditer(text)]

def evidence_for(request,pid,products):
 choices=[v for v in products[pid]['variants'] if all(c['value']==v['colorName'] for c in request['hardRequirements'])]
 assert choices
 mentions=expressed_colors(request,products)
 first=next((color for color in mentions if any(v['colorName']==color for v in choices)),None)
 preferred=[v['id'] for v in choices if first is None or v['colorName']==first]
 return {'policy':'no-offered-soft-preference-indifferent' if first is None else 'first-expressed-offered-soft-preference',
  'evidenceField':'request.persona.colorPreferences','publicColorText':deepcopy(request['persona']['colorPreferences']),
  'expressedColorOrder':mentions,'firstExpressedOfferedColor':first,
  'eligibleVariantIds':[v['id'] for v in choices],'preferredVariantIds':preferred,
  'supervisionWeights':{vid:1/len(preferred) for vid in preferred},
  'interpretation':'equal indifference among hard-eligible source variants; no unique human preference asserted' if first is None else 'exact first expressed color available among hard-eligible source variants; synthetic supervision, not observed choice'}

def color_contrast(a,b):
 changes=[]
 for key in ['query','itemBudget','hardRequirements','publicHistory']:
  if a[key]!=b[key]:changes.append(key)
 for key in ['background','colorPreferences']:
  if a['persona'][key]!=b['persona'][key]:changes.append('persona.'+key)
 budget=a['itemBudget']!=b['itemBudget']
 return {'actualChangedFields':changes,'budgetAlsoChanges':budget,
  'interpretation':'compound-budget-color-intervention' if budget else 'soft-color-only-intervention' if changes==['persona.colorPreferences'] else 'color-with-hard-requirement-or-background-intervention',
  'cleanSoftColorOnly':changes==['persona.colorPreferences']}

def build(target=None):
 plan=verify_sources();target=Path(target) if target else HERE;target.mkdir(exist_ok=True,parents=True)
 products={p['id']:p for p in read(CATALOG)['products']}
 public=read(PARENT/'profiles.train.public.json');profiles={p['profileId']:p for p in public['profiles']}
 labels=read(PARENT/'labels.train.json');labels['schemaVersion']=SCHEMA
 audit=read(AUDIT);corrections={(c['profileId'],c['productId']):c for c in audit['variantPreferenceCorrections']}
 assert len(corrections)==37
 observed=set();rows=[]
 for label in labels['labels']:
  profile=profiles[label['profileId']];preferred=[];evidence={}
  for pid,grade in label['relevanceGrades'].items():
   if not grade:continue
   e=evidence_for(profile['request'],pid,products);evidence[pid]=e;preferred.extend(e['preferredVariantIds'])
   productids={v['id'] for v in products[pid]['variants']};old=[v for v in label['preferredVariantIds'] if v in productids]
   key=(label['profileId'],pid)
   if e['policy']=='no-offered-soft-preference-indifferent':
    assert key in corrections
    c=corrections[key];assert old==[c['unsupportedSingletonPreferredVariantId']]
    assert e['preferredVariantIds']==c['eligibleIndifferentVariantIds']
    assert e['expressedColorOrder']==c['expressedColorOrder']
    observed.add(key);rows.append({'profileId':key[0],'productId':pid,'oldPreferredVariantIds':old,'newPreferredVariantIds':e['preferredVariantIds'],'policy':e['policy']})
   else:
    assert key not in corrections and old==e['preferredVariantIds'],key
  label['preferredVariantIds']=preferred;label['variantPreferenceEvidence']=evidence
 assert observed==set(corrections)
 pairs=read(PARENT/'pairs.train.json');pairs['schemaVersion']=SCHEMA
 for pair in pairs['pairs']:
  if pair['kind']=='color-preference-or-hard-color-release':
   pair['colorContrastAudit']=color_contrast(profiles[pair['baseProfileId']]['request'],profiles[pair['counterfactualProfileId']]['request'])
 report={'schemaVersion':SCHEMA,'authorId':'/root/runner_review','originatingIndependentAuditorId':'/root/copy_audit',
  'originatingIndependentAuditSha256':plan['independentAuditSha256'],'repairedBindings':len(rows),'unchangedSupportedBindings':235,
  'publicBytesUnchanged':True,'all2048ProductGradeCellsUnchanged':True,'acceptableVariantSetsUnchanged':True,
  'repairVerification':'AUTHOR_CPU_VALIDATION_ONLY; independent v4 recheck required before fit',
  'corrections':rows,'evaluationFilesRead':False,'modelOrGpuUsed':False}
 (target/'profiles.train.public.json').write_bytes((PARENT/'profiles.train.public.json').read_bytes())
 write(target/'labels.train.json',labels);write(target/'pairs.train.json',pairs);write(target/'CORRECTIONS.json',report)
 return report

if __name__=='__main__':print(json.dumps({k:v for k,v in build().items() if k!='corrections'},indent=2))
