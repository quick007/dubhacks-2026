"""CPU-only prospective fit contracts; never extract or fit native features."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from .semantic_copy_feature_fit import (CONFIG,dataset_packet,digest,dump,feature_bytes,
    load_feature,serving_rows,shuffled_order,summary)


class FeatureFitContracts(unittest.TestCase):
    def test_import_does_not_load_native_runtime(self):
        code=('import sys; import experiments.ranker.pretrained.semantic_copy_feature_fit; '
              'assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)')
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)

    def test_frozen_dataset_and_serving_renderer_agree(self):
        _,packet=dataset_packet()
        self.assertEqual(len(packet['train']['fields']),504)
        for split in ('train','dev','test'):
            for case in packet[split]['cases']:
                fields=[f for f in packet[split]['fields'] if f['caseId']==case['caseId']]
                common,rows,ordered=serving_rows(case,fields)
                self.assertEqual(len(rows),14)
                self.assertEqual(sum(r['maxTokens'] for r in rows),224)
                self.assertEqual(ordered[0]['messages'],[common]+rows[0]['messages'])
        case=packet['train']['cases'][0]
        fields=copy.deepcopy([f for f in packet['train']['fields'] if f['caseId']==case['caseId']])
        fields[0]['messages'][-1]['content']+=' changed'
        with self.assertRaisesRegex(ValueError,'serving renderer'):serving_rows(case,fields)

    def test_global_shuffle_has_one_update_per_row(self):
        order=shuffled_order()
        self.assertEqual(order,shuffled_order())
        self.assertEqual(sorted(order),list(range(504)))
        self.assertNotEqual(order,list(range(504)))
        self.assertGreater(len({i//14 for i in order[:14]}),8)
        with self.assertRaises(ValueError):shuffled_order(503)

    def test_full_suffix_feature_memory_bound(self):
        self.assertEqual(feature_bytes(39),39*20480*4)
        self.assertLessEqual(feature_bytes(256),CONFIG['singleHostFeatureBudgetBytes'])
        for count in (0,257,True,1.5):
            with self.subTest(count=count),self.assertRaises(ValueError):feature_bytes(count)

    def write_feature(self,out):
        import hashlib
        arrays={k:np.ones((1,39,width),dtype=np.float32) for k,width in (('x',12288),('h',4096),('y',4096))}
        metadata={'dtypes':{k:'mlx.core.bfloat16' for k in arrays},'predictionStartInBranch':35,
            'targets':[1,2,3,99],'gradientFeatureGatePassed':True}
        archive=out/'row.npz';meta=out/'row.json';np.savez(archive,**arrays);dump(meta,metadata)
        entry={'archive':archive.name,'metadata':meta.name,'archiveSha256':digest(archive),
            'metadataSha256':digest(meta),'featureTensors':{k:{'shape':list(v.shape),'dtype':metadata['dtypes'][k],
                'sha256':hashlib.sha256(metadata['dtypes'][k].encode()+str(tuple(v.shape)).encode()+v.tobytes()).hexdigest()}
                for k,v in arrays.items()}}
        return entry

    def test_one_row_archive_hash_dtype_and_gate_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);entry=self.write_feature(out);item=load_feature(entry,out)
            self.assertEqual(item['x'].shape,(1,39,12288))
            bad=copy.deepcopy(entry);bad['featureTensors']['x']['sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'receipt mismatch'):load_feature(bad,out)
            meta=out/entry['metadata'];value=json.loads(meta.read_text());value['gradientFeatureGatePassed']=False
            dump(meta,value)
            with self.assertRaisesRegex(ValueError,'artifact changed'):load_feature(entry,out)
            entry['metadataSha256']=digest(meta)
            with self.assertRaisesRegex(ValueError,'per-row gate'):load_feature(entry,out)

    def test_failure_and_unattempted_cases_keep_fixed_denominator(self):
        arms=[{'split':s,'arm':a,'status':'not-attempted'} for s in ('dev','test')
              for a in ('frozen','learned') for _ in range(3)]
        arms[0].update(status='completed-attempt',quality={'allFieldsFulfilled':True,
            'newTitleCount':6,'generatedHeadingCount':2,'generatedBadgeCount':4},lockedPlanCopyMs=12.)
        arms[1]['status']='failed';arms[3]['status']='running'
        result=summary(arms)
        self.assertEqual(result['dev/frozen']['validRateFixedCases'],1/3)
        self.assertEqual(result['dev/frozen']['failed'],1)
        self.assertEqual(result['dev/frozen']['unattempted'],1)
        self.assertEqual(result['dev/learned']['inProgress'],1)
        self.assertFalse(result['dev/frozen']['completeAboveFoldClaim'])


if __name__=='__main__':unittest.main()
