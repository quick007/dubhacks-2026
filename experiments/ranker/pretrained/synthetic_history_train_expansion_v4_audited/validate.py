"""CPU-only validation of the v4 TRAIN repair. Never loads model or evaluation data."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import importlib.util,json,math
import build
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def read(name):return json.loads((HERE/name).read_text())
def old_validator():
 spec=importlib.util.spec_from_file_location('frozen_v3_validation',build.PARENT/'validate.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def check_label(profile,label,products,original):
 """Original grade/eligibility checks plus exact supported or indifferent variant targets."""
 assert label['profileId']==profile['profileId']==original['profileId']
 restored=deepcopy(label);restored.pop('variantPreferenceEvidence')
 restored['preferredVariantIds']=deepcopy(original['preferredVariantIds'])
 assert restored==original,'product grades, acceptable variants or other original annotations changed'
 old_validator().check_label(profile,original,products)
 positives={pid for pid,grade in label['relevanceGrades'].items() if grade}
 evidence=label['variantPreferenceEvidence'];assert set(evidence)==positives
 expectedflat=[]
 for pid in label['relevanceGrades']:
  if pid not in positives:continue
  expected=build.evidence_for(profile['request'],pid,products)
  actual=evidence[pid]
  assert actual==expected,(profile['profileId'],pid,'unsupported variant preference evidence')
  weights=actual['supervisionWeights']
  assert weights and all(math.isfinite(w) and w>0 for w in weights.values())
  assert math.isclose(sum(weights.values()),1,rel_tol=0,abs_tol=1e-12)
  assert len(set(weights.values()))==1,'unequal target mass'
  assert set(weights)==set(actual['preferredVariantIds'])
  assert set(weights)<=set(actual['eligibleVariantIds'])<=set(label['acceptableVariantIds'])
  expectedflat.extend(actual['preferredVariantIds'])
 assert label['preferredVariantIds']==expectedflat,'arbitrary singleton or wrong flattened preference set'
 assert len(expectedflat)==len(set(expectedflat))
 return positives

def audit():
 build.verify_sources()
 v3=old_validator().audit() # Entire original immutable TRAIN packet and its existing invariants.
 assert (HERE/'profiles.train.public.json').read_bytes()==(build.PARENT/'profiles.train.public.json').read_bytes()
 for name in ['PLAN.json','PLAN_FREEZE.json','SEMANTIC_COVERAGE.json','AUTHORSHIP.json','TOKEN_PREFLIGHT.json','PUBLIC_DEDUP.json']:
  assert (HERE/name).read_bytes()==(build.PARENT/name).read_bytes(),name
 products={p['id']:p for p in build.read(build.CATALOG)['products']}
 profiles=read('profiles.train.public.json')['profiles'];labels=read('labels.train.json')['labels']
 assert len(profiles)==len(labels)==128
 byid={p['profileId']:p for p in profiles};bylabel={l['profileId']:l for l in labels}
 original={l['profileId']:l for l in build.read(build.PARENT/'labels.train.json')['labels']}
 assert set(byid)==set(bylabel)==set(original)
 policies=Counter();corrected=set();preferred_count=0
 for label in labels:
  profile=byid[label['profileId']];old_validator().check_request(profile['request'],products)
  check_label(profile,label,products,original[label['profileId']]);preferred_count+=len(label['preferredVariantIds'])
  for pid,e in label['variantPreferenceEvidence'].items():
   policies[e['policy']]+=1
   if e['policy']=='no-offered-soft-preference-indifferent':corrected.add((label['profileId'],pid))
 audit=build.read(build.AUDIT)
 assert corrected=={(c['profileId'],c['productId']) for c in audit['variantPreferenceCorrections']}
 assert policies=={'no-offered-soft-preference-indifferent':37,'first-expressed-offered-soft-preference':235}
 oldpairs=build.read(build.PARENT/'pairs.train.json')['pairs'];pairs=read('pairs.train.json')['pairs']
 assert len(pairs)==len(oldpairs)==64
 contrast_counts=Counter()
 for pair,old in zip(pairs,oldpairs,strict=True):
  restored=deepcopy(pair);contrast=restored.pop('colorContrastAudit',None);assert restored==old
  if pair['kind']=='color-preference-or-hard-color-release':
   actual=build.color_contrast(byid[pair['baseProfileId']]['request'],byid[pair['counterfactualProfileId']]['request'])
   assert contrast==actual;contrast_counts[actual['interpretation']]+=1
  else:assert contrast is None
 assert contrast_counts=={'compound-budget-color-intervention':12,'color-with-hard-requirement-or-background-intervention':3,'soft-color-only-intervention':1}
 correction=read('CORRECTIONS.json')
 assert len(correction['corrections'])==37 and {(c['profileId'],c['productId']) for c in correction['corrections']}==corrected
 return {'status':'AUTHOR_CPU_VALIDATION_PASS_PENDING_INDEPENDENT_V4_RECHECK','schemaVersion':build.SCHEMA,
  'trainRows':128,'groups':64,'publicByteIdenticalToV3':True,'publicSha256':build.sha(HERE/'profiles.train.public.json'),
  'v3WholeTreeUnchanged':True,'allProductGradeCellsUnchanged':2048,'positiveProductBindings':272,
  'acceptableVariantListsUnchanged':128,'variantPolicyCounts':dict(policies),'flattenedPreferredVariantIds':preferred_count,
  'variantTargetMass':'Each positive product has unit mass distributed equally over its preferred set; indifference is not a unique human preference.',
  'colorPairContrastCounts':dict(contrast_counts),'originalCpuValidation':v3,
  'compactTokenPreflightReused':'Unchanged 205–294 compact context tokens; not actual full catalog choice LM prompt. Full prompt needs separate consumer preflight.',
  'evaluationFilesRead':False,'modelLoaded':False,'gpuUsed':False,
  'independentAuditorId':'/root/copy_audit','originatingIndependentAuditSha256':build.sha(build.AUDIT),
  'fitGate':'Independent v4 repair recheck required before fit; this report is author validation.'}

if __name__=='__main__':print(json.dumps(audit(),indent=2))
