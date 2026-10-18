"""Finalize six prewritten packets after the owner declares serving ready. CPU only.

Requires a saved owner-ready receipt, verifies every already-frozen scenario byte,
re-exports pure public TS configuration, and changes only the catalog receipt and
manifest. Never dispatches a shopper or requests the native model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'experiments/ranker/pretrained'
sys.path.insert(0,str(ROOT))
from experiments.ranker.pretrained.store_demo_catalog import runtime_export
from experiments.ranker.pretrained.store_catalog_v1 import digest, RAW_EXPORT_SHA256

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def require(value,message):
    if not value:raise ValueError(message)

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--owner-ready-receipt',type=Path,required=True)
    args=parser.parse_args()
    ready=args.owner_ready_receipt.resolve()
    require(ready.is_relative_to(BASE) and ready.is_file() and '.env' not in ready.parts,'Saved non-secret experiment owner READY receipt required')
    require(ready.stat().st_size>0,'Empty READY receipt is insufficient')
    cohort_path=BASE/'store_shopper_eval_v2/cohort-preregistration.json'
    cohort=read(cohort_path)
    require(cohort['version']=='synthetic-shopper-cohort-preregistration.v1' and cohort['experimentId']=='actual-store-heldout-shopper-v2','Fixed campaign cohort preregistration required')
    design=cohort['fixedDesign']
    require((design['pairs'],design['assignedSessions'],design['purchaseFeasiblePairs'],design['negativePairs'],design['negativeArmAssignments'])==(60,120,48,12,24),'Fixed60/120cohort composition required')
    require((design['globalRandomSeed'],design['maxActionsPerSession'],design['maxWallSecondsPerSession'])==(20261018,24,120),'Fixed cohort seed and agent budgets required')
    require(design['blocks']==['store_shopper_eval_v'+str(v) for v in range(2,8)] and design['execution']=='sequential','All six sequential blocks required')
    require(cohort['runtimePilot']['pairs']==5 and cohort['runtimePilot']['includedInMainAssignedDenominator'] is True and cohort['runtimePilot']['retuningReplacementOrOutcomeBasedResizing'] is False,'Unchanged included five-pair runtime pilot required')
    require(cohort['deadline']['utc']=='2026-10-18T13:00:00Z','Fixed6amPDT deadline required')
    require(len(cohort['negativeAbstentionAudit']['fixedRubric'])==4,'Fixed blind negative-abstention rubric required')
    source_names=list(read(BASE/'store_shopper_eval_v2/catalog-binding.json')['liveInputSourceHashes'])
    before={name:sha(ROOT/name) for name in source_names}
    actual=runtime_export()
    require(digest(actual)==RAW_EXPORT_SHA256,'Current public export differs from fixed original catalog/prices/colors/images/shipping')
    after={name:sha(ROOT/name) for name in source_names}
    require(before==after,'Catalog source changed during CPU export')
    variants={r['productId']:r['colorVariants'] for r in actual['variants']}
    cells=Counter();queries=[];frozen=[]
    for version in range(2,8):
        directory=BASE/f'store_shopper_eval_v{version}'
        manifest=read(directory/'manifest.json')
        require(manifest['sourceReceiptPhase']=='provisional-awaiting-model-owner-ready','Preserve finalized packet; refusing repeat mutation')
        for name,expected in manifest['files'].items():require(sha(directory/name)==expected,'Scenario packet changed before finalization: '+str(directory/name))
        require(manifest.get('experimentId')=='actual-store-heldout-shopper-v2' and read(directory/'pair-plan.json').get('experimentId')=='actual-store-heldout-shopper-v2' and read(directory/'preregistration.json').get('experimentId')=='actual-store-heldout-shopper-v2','Shared campaign experiment ID required in all six packet projections')
        overlap=read(directory/'overlap-check.json')
        require(sha(BASE/'store_function_v5/public-cases.json')==overlap['v5PublicCasesSha256'],'Current ranking query roster changed since wording-overlap check')
        require(sha(BASE/'store_shopper_eval_v1/public-observations.json')==overlap['comparisonSourceHashes']['oldPublic'],'Prior shopper queries changed since wording-overlap check')
        require(sha(BASE/'store_function_v5_eval_seed_v1.json')==overlap['comparisonSourceHashes']['v5IndependentEval'],'Independent ranking evaluation changed since wording-overlap check')
        binding=read(directory/'catalog-binding.json')
        snapshot=binding['catalogSnapshot']
        require(len(snapshot['products'])==16 and sum(len(p['variants']) for p in snapshot['products'])==48,'Exact16/48 binding required')
        require(snapshot['sha256']==digest({'products':snapshot['products']}),'Scorer catalog snapshot digest differs')
        for product,bound in zip(actual['products'],snapshot['products'],strict=True):
            require((product['id'],product['name'],product['price'],product['category'],product.get('searchTerms',[]))==(bound['productId'],bound['originalName'],bound['itemPrice'],bound['facts']['category'],bound['facts']['searchTerms']),'Public identity/price facts changed')
            require([(v['variant']['id'],v['variant']['colorName']) for v in variants[product['id']]]==[(v['variantId'],v['selectedColor']) for v in bound['variants']],'Public color names/IDs changed')
        public=read(directory/'public-observations.json');private=read(directory/'shopper-instructions.json');gold=read(directory/'evaluator-gold.json')
        queries.extend(row['observableContext']['query'].casefold().strip() for row in public)
        for p,g in zip(private[:8],gold[:8],strict=True):cells[(g['acceptableProductIds'][0],p['privateShopperNeeds']['preferences'][0]['value'])]+=1
        frozen.append((directory,manifest,binding))
    require(len(queries)==60 and len(set(queries))==60,'Sixty distinct frozen public queries required')
    require(len(cells)==48 and set(cells.values())=={1},'Each of16products×3privatecolors must appear once')
    receipt={'version':'shopper-current-catalog-cpu-freeze.v1','experimentId':'actual-store-heldout-shopper-v2','cohortPreregistrationSha256':sha(cohort_path),'globalRandomSeed':20261018,'verifiedAtUtc':datetime.now(timezone.utc).isoformat(),'ownerReadyReceipt':str(ready.relative_to(ROOT)),'ownerReadyReceiptSha256':sha(ready),'runtimePublicExportSha256':digest(actual),'originalPublicExportSha256':RAW_EXPORT_SHA256,'sourceHashes':after,'counts':{'blocks':6,'pairs':60,'assignedSessions':120,'purchaseFeasiblePairs':48,'abstentionPairs':12,'products':16,'colorVariants':48,'uniquePositiveProductColorCells':48},'nativeRequests':0,'browserRuns':0,'claim':'Scenario-held-out, catalog-known; exact original public export equality. No unseen-product or human-lift claim.'}
    for directory,manifest,binding in frozen:
        binding.update(liveInputSourceHashes=after,runtimePublicExportVerifiedCPU=True,rawExportSha256=digest(actual),sourceReceiptPhase='frozen-after-owner-ready',currentSourceVerification=receipt)
        write(directory/'catalog-binding.json',binding)
        manifest.update(sourceReceiptPhase='frozen-after-owner-ready',frozen=True,sourceFacts='Pure TS runtime export exactly equals original fixed16products/48colors/prices/images/shipping; current catalog source hashes independently verified after owner READY.')
        manifest['files']['catalog-binding.json']=sha(directory/'catalog-binding.json')
        write(directory/'manifest.json',manifest)
        print(json.dumps({'packet':directory.name,'manifestSha256':sha(directory/'manifest.json'),'catalogBindingSha256':sha(directory/'catalog-binding.json')}))
if __name__=='__main__':main()
