"""CPU protocol fixtures only. These tests never establish native evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import store_expanded_parallel_copy_v1 as expanded
from . import store_parallel_text_head_v1 as old
from . import store_parallel_text_data_v1 as data

class ExpandedTests(unittest.TestCase):
 def catalog(self,count=3):
  return {'version':'fixture','products':[{'productId':f'p{i}','name':f'Canvas bag {chr(65+i)}','category':'Canvas carry bag','itemPrice':24+i,'variants':[{'variantId':f'p{i}-{c.lower()}','colorName':c,'images':[{'src':'/fixture.jpg'}]} for c in ('Navy','Sand','Olive')]} for i in range(count)]}
 def test_dynamic_public_packet_all_fields_sourcebound_no_inference_labels(self):
  p=expanded.build_packet(self.catalog(),'a'*64)
  self.assertEqual(p['bindingCount'],9);self.assertEqual(len(p['train']),9*15)
  for row in old.inference_rows(p['train']):self.assertEqual(set(row),{'input','featureKeys'})
  self.assertEqual(p['heldoutBindings'],0)
  for r in p['featureRecords']:self.assertEqual(r,data.public_record(r['facts']))
  self.assertIn('Canvas',p['vocabulary']);self.assertIn('$26.00',p['vocabulary'])
 def test_duplicate_and_slot_overflow_fail_closed(self):
  c=self.catalog();c['products'][1]['productId']='p0'
  with self.assertRaises(ValueError):expanded.build_packet(c,'a'*64)
  c=self.catalog();c['products'][0]['name']=' '.join(['Very']*15)
  with self.assertRaises(ValueError):expanded.build_packet(c,'a'*64)
 def test_prepare_refuses_partial_catalog_without_creating_output(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);f=root/'catalog.json';expanded.dump(f,self.catalog());out=root/'packet'
   with self.assertRaises(ValueError):expanded.prepare(f,out)
   self.assertFalse(out.exists())
 def native_fixture(self,root,p):
  expected={'evidenceSource':'native-frozen-qwen','modelRevision':old.MODEL_REVISION,'copyAdapterEnabled':False,'rankAdapterEnabled':False,'nativeForwards':p['bindingCount'],'truncatedRows':0,'catalogSha256':p['catalogSha256'],'frozenTensorCount':927,'frozen927HashesUnchanged':True,'shutdownVerified':True}
  f=root/'synthetic-native-protocol-receipt.json';expanded.dump(f,{'copyFeatureProvenance':expected,'fixtureOnly':True})
  prov={**expected,'nativeReceiptFile':str(f),'nativeReceiptSha256':expanded.sha(f)}
  x=torch.eye(p['bindingCount'],4096)
  return expanded.make_expanded_feature_bundle(p['featureRecords'],x,prov,catalog_sha=p['catalogSha256'])
 def test_bundle_order_tamper_and_receipt_integrity(self):
  with tempfile.TemporaryDirectory() as td:
   p=expanded.build_packet(self.catalog(),'a'*64);b=self.native_fixture(Path(td),p)
   self.assertEqual(len(expanded.validate_bundle(b,p)),9)
   b['featureKeys']=list(reversed(b['featureKeys']))
   with self.assertRaises(ValueError):expanded.validate_bundle(b,p)
   b=self.native_fixture(Path(td),p);Path(b['provenance']['nativeReceiptFile']).write_text('{}')
   with self.assertRaises(ValueError):expanded.validate_bundle(b,p)
 def test_old48_native_validator_unchanged_rejects_dynamic(self):
  with tempfile.TemporaryDirectory() as td:
   p=expanded.build_packet(self.catalog(),'a'*64);b=self.native_fixture(Path(td),p)
   with self.assertRaises(ValueError):old.make_feature_bundle(p['featureRecords'],b['features'],b['provenance'])
 def test_failed_completeness_preserves_updated_checkpoint_and_refuses_serving(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);cf=root/'catalog.json';expanded.dump(cf,self.catalog());packet=root/'packet';expanded.prepare(cf,packet,expected_products=3,expected_bindings=9);p=json.loads((packet/'packet.TRAIN.json').read_text());bundle=self.native_fixture(root,p);bf=root/'features.pt';torch.save(bundle,bf)
   config={**expanded.CONFIG,'maxSteps':1,'checkEvery':1}
   with patch.object(expanded,'CONFIG',config):report=expanded.fit_expanded(packet,bf,root/'fit')
   self.assertEqual(report['optimizerUpdates'],1);self.assertTrue(report['headWeightsChanged']);self.assertTrue((root/'fit/head.pt').exists());self.assertEqual(report['status'],'failed-supervised-completeness-gate')
   with self.assertRaises(ValueError):expanded.load_expanded_parallel_assets(root/'fit',expanded.sha(root/'fit/serving-manifest.json'))
 def test_loader_never_teacher_or_train_packet_and_manifest_hash_fail(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);catalog=self.catalog();cf=root/'catalog.json';expanded.dump(cf,catalog);p=expanded.build_packet(catalog,expanded.sha(cf));b=self.native_fixture(root,p);bf=root/'features.pt';torch.save(b,bf)
   public={k:p[k] for k in ('version','scope','catalogSha256','productCount','bindingCount','featureRecords','bindings')};public['teacherOrLabelContent']=False;expanded.dump(root/'public-runtime-features.json',public)
   h=old.ParallelTextHead(p['vocabulary'],width=8);torch.save({'version':expanded.VERSION,'vocabulary':p['vocabulary'],'width':8,'state_dict':h.state_dict(),'featureBundleSha256':expanded.sha(bf),'catalogSha256':p['catalogSha256']},root/'head.pt')
   manifest={'version':expanded.VERSION,'ready':True,'runtimeTeacherLookup':False,'heldoutQualityClaim':False,'allBindingsSupervised':True,'reloadExactSlotParity':True,'catalogFile':str(cf),'catalogSha256':expanded.sha(cf),'headFile':'head.pt','headSha256':expanded.sha(root/'head.pt'),'publicFile':'public-runtime-features.json','publicSha256':expanded.sha(root/'public-runtime-features.json'),'featureBundleFile':str(bf),'featureBundleSha256':expanded.sha(bf),'nativeReceiptFile':b['provenance']['nativeReceiptFile'],'nativeReceiptSha256':b['provenance']['nativeReceiptSha256'],'sourceSha256':expanded.sources()};expanded.dump(root/'serving-manifest.json',manifest);mh=expanded.sha(root/'serving-manifest.json')
   with patch.object(data,'teacher',side_effect=AssertionError('Serving must not call teacher')):
    loaded,fmap,records,metadata=expanded.load_expanded_parallel_assets(root,mh)
   self.assertEqual(len(fmap),9);self.assertFalse(metadata['teacherStringsReadAtServing']);self.assertEqual(metadata['featureBundleFile'],str(bf))
   with self.assertRaises(ValueError):expanded.load_expanded_parallel_assets(root,'0'*64)
   (root/'head.pt').write_bytes(b'changed')
   with self.assertRaises(ValueError):expanded.load_expanded_parallel_assets(root,mh)
if __name__=='__main__':unittest.main()
