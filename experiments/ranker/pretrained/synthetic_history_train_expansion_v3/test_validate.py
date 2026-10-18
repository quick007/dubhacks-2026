from copy import deepcopy
import json
from pathlib import Path
import unittest
import validate

class ExpansionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.profiles=validate.read('profiles.train.public.json')['profiles']
  cls.labels={l['profileId']:l for l in validate.read('labels.train.json')['labels']}
  source=validate.ROOT/'experiments/ranker/pretrained/synthetic_history_v1/catalog-binding.json'
  cls.products={p['id']:p for p in json.loads(source.read_text())['products']}
 def test_complete_frozen_training_packet(self):
  audit=validate.audit();self.assertEqual(audit['trainRows'],128);self.assertEqual(audit['preferredVariantCoverage'],48)
  self.assertGreaterEqual(audit['uniqueBroadQueries'],61);self.assertEqual(audit['baseBackgroundBlocks'],64)
 def test_inference_visible_outcome_or_identity_fields_rejected(self):
  r=deepcopy(self.profiles[0]['request']);r['targetProductId']='mug'
  with self.assertRaises(AssertionError):validate.check_request(r,self.products)
  r=deepcopy(self.profiles[0]['request']);r['persona']['userGroupId']='opaque'
  with self.assertRaises(AssertionError):validate.check_request(r,self.products)
 def test_public_purchase_cannot_be_positive_without_replacement(self):
  p=next(p for p in self.profiles if not self.labels[p['profileId']]['replacementAllowedProductIds'])
  l=deepcopy(self.labels[p['profileId']]);owned=l['alreadyOwnedProductIds'][0];l['relevanceGrades'][owned]=2
  with self.assertRaises(AssertionError):validate.check_label(p,l,self.products)
 def test_ineligible_budget_or_color_is_never_positive(self):
  p=deepcopy(self.profiles[0]);l=self.labels[p['profileId']];p['request']['itemBudget']=1
  with self.assertRaises(AssertionError):validate.check_label(p,l,self.products)
  p=deepcopy(self.profiles[0]);p['request']['hardRequirements']=[{'attribute':'color','value':'not-a-store-color'}]
  with self.assertRaises(AssertionError):validate.check_label(p,l,self.products)
 def test_observed_store_identity_and_variant_are_source_conditioned(self):
  r=deepcopy(self.profiles[0]['request']);r['publicHistory'][0]['selectedColor']='fictional'
  with self.assertRaises(AssertionError):validate.check_request(r,self.products)
 def test_public_formatter_omits_identifiers_absolute_dates_and_labels(self):
  p=self.profiles[0];text=validate.check_request(p['request'],self.products)
  for forbidden in [p['profileId'],p['sessionId'],p['userGroupId'],'relevanceGrades','preferredVariantIds']:
   self.assertNotIn(forbidden,text)
 def test_explicit_replacement_is_required_for_owned_positive(self):
  p=deepcopy(next(p for p in self.profiles if self.labels[p['profileId']]['replacementAllowedProductIds']))
  l=self.labels[p['profileId']];p['request']['persona']['background']=['I do not have a new request.']
  with self.assertRaises(AssertionError):validate.check_label(p,l,self.products)

class ReproductionTests(unittest.TestCase):
 def test_authored_builder_reproduces_all_four_frozen_training_outputs(self):
  import importlib.util,tempfile,shutil
  spec=importlib.util.spec_from_file_location('train_expansion_builder',validate.HERE/'build.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  with tempfile.TemporaryDirectory() as temp:
   target=Path(temp)
   for name in ['PLAN.json','PLAN_FREEZE.json','SEMANTIC_COVERAGE.json']:shutil.copyfile(validate.HERE/name,target/name)
   module.HERE=target;module.build()
   for name in ['profiles.train.public.json','labels.train.json','pairs.train.json','AUTHORSHIP.json']:
    self.assertEqual((target/name).read_bytes(),(validate.HERE/name).read_bytes())

if __name__=='__main__':unittest.main()
