from copy import deepcopy
from pathlib import Path
import importlib.util,json,tempfile,unittest
# Avoid an unrelated already-imported validate module under unittest discovery.
_spec=importlib.util.spec_from_file_location('v4_train_repair_validate',Path(__file__).with_name('validate.py'))
validate=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(validate)
build=validate.build

class RepairTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.profiles={p['profileId']:p for p in validate.read('profiles.train.public.json')['profiles']}
  cls.labels={l['profileId']:l for l in validate.read('labels.train.json')['labels']}
  cls.original={l['profileId']:l for l in build.read(build.PARENT/'labels.train.json')['labels']}
  cls.products={p['id']:p for p in build.read(build.CATALOG)['products']}
  cls.corrected=next((p['profileId'],pid) for p in cls.profiles.values() for pid,e in cls.labels[p['profileId']]['variantPreferenceEvidence'].items() if e['policy']=='no-offered-soft-preference-indifferent')
 def check(self,pid,label):return validate.check_label(self.profiles[pid],label,self.products,self.original[pid])
 def test_full_packet_preserves_public_product_grades_and_old_tree(self):
  a=validate.audit();self.assertEqual(a['allProductGradeCellsUnchanged'],2048);self.assertEqual(a['positiveProductBindings'],272)
  self.assertTrue(a['v3WholeTreeUnchanged']);self.assertTrue(a['publicByteIdenticalToV3'])
 def test_all_source_eligible_variants_are_equal_when_no_offered_preference(self):
  key,pid=self.corrected;l=self.labels[key];e=l['variantPreferenceEvidence'][pid]
  self.assertEqual(e['preferredVariantIds'],e['eligibleVariantIds']);self.assertEqual(len(e['preferredVariantIds']),3)
  self.assertEqual(set(e['supervisionWeights'].values()),{1/3});self.check(key,l)
 def test_arbitrary_source_first_singleton_is_rejected(self):
  key,pid=self.corrected;l=deepcopy(self.labels[key]);e=l['variantPreferenceEvidence'][pid]
  l['preferredVariantIds']=[v for v in l['preferredVariantIds'] if v not in e['preferredVariantIds'][1:]]
  with self.assertRaises(AssertionError):self.check(key,l)
 def test_unequal_variant_supervision_rejected_even_if_sum_one(self):
  key,pid=self.corrected;l=deepcopy(self.labels[key]);e=l['variantPreferenceEvidence'][pid]
  e['supervisionWeights']=dict(zip(e['preferredVariantIds'],[.5,.25,.25]))
  with self.assertRaises(AssertionError):self.check(key,l)
 def test_first_expressed_available_color_wins_over_catalog_order(self):
  r=deepcopy(next(iter(self.profiles.values()))['request']);r['hardRequirements']=[]
  r['persona']['colorPreferences']=['I would choose Cobalt before Sage. Ivory is okay.']
  e=build.evidence_for(r,'mug',self.products)
  self.assertEqual(e['preferredVariantIds'],['mug-cobalt']);self.assertEqual(e['policy'],'first-expressed-offered-soft-preference')
 def test_longer_color_names_do_not_create_false_short_color_preference(self):
  r=deepcopy(next(iter(self.profiles.values()))['request']);r['hardRequirements']=[]
  r['persona']['colorPreferences']=['Sage stripe, then Navy band.']
  self.assertEqual(build.expressed_colors(r,self.products),['Sage stripe','Navy band'])
  self.assertEqual(build.evidence_for(r,'mug',self.products)['policy'],'no-offered-soft-preference-indifferent')
 def test_hard_color_limits_indifference_choices(self):
  r=deepcopy(next(iter(self.profiles.values()))['request']);r['hardRequirements']=[{'attribute':'color','value':'Ivory'}]
  r['persona']['colorPreferences']=['Cobalt']
  e=build.evidence_for(r,'mug',self.products);self.assertEqual(e['eligibleVariantIds'],['mug-ivory'])
  self.assertEqual(e['supervisionWeights'],{'mug-ivory':1.0})
 def test_grade_or_acceptable_variant_changes_rejected(self):
  key,_=self.corrected
  for field in ['relevanceGrades','acceptableVariantIds']:
   l=deepcopy(self.labels[key])
   if field=='relevanceGrades':l[field][next(iter(l[field]))]=99
   else:l[field].append('invented-variant')
   with self.assertRaises(AssertionError):self.check(key,l)
 def test_public_target_field_is_rejected(self):
  r=deepcopy(next(iter(self.profiles.values()))['request']);r['targetProductId']='mug'
  with self.assertRaises(AssertionError):validate.old_validator().check_request(r,self.products)
 def test_compound_pair_disclosure_uses_actual_fields(self):
  pairs=[p for p in validate.read('pairs.train.json')['pairs'] if 'colorContrastAudit' in p]
  self.assertEqual(sum(p['colorContrastAudit']['budgetAlsoChanges'] for p in pairs),12)
  self.assertEqual(sum(p['colorContrastAudit']['cleanSoftColorOnly'] for p in pairs),1)
  for p in pairs:
   a=self.profiles[p['baseProfileId']]['request'];b=self.profiles[p['counterfactualProfileId']]['request']
   self.assertEqual(p['colorContrastAudit']['budgetAlsoChanges'],a['itemBudget']!=b['itemBudget'])
 def test_exact_37_bindings_match_independent_train_audit(self):
  a=build.read(build.AUDIT);expected={(c['profileId'],c['productId']) for c in a['variantPreferenceCorrections']}
  actual={(c['profileId'],c['productId']) for c in validate.read('CORRECTIONS.json')['corrections']}
  self.assertEqual(len(actual),37);self.assertEqual(actual,expected)
 def test_builder_reproduces_all_four_outputs_byte_for_byte(self):
  with tempfile.TemporaryDirectory() as temp:
   build.build(temp)
   for name in ['profiles.train.public.json','labels.train.json','pairs.train.json','CORRECTIONS.json']:
    self.assertEqual((Path(temp)/name).read_bytes(),(validate.HERE/name).read_bytes())

if __name__=='__main__':unittest.main()
