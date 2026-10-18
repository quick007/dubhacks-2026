"""CPU fit-boundary fixtures; synthetic arrays are never native evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from . import store_persona_choice_fit_v1 as fit

class FitBoundaryTests(unittest.TestCase):
    def rows(self):
        return [{'caseId':f'unit-{i}-{order}','profileId':str(i),'order':order,
                 'split':'train','targetPurpose':'maximum-grade-TRAIN-supervision'}
                for i in range(448) for order in ('canonical','reversed')]
    def test_exact896_roster_accepts_and_eval_poison_refuses(self):
        rows=self.rows();self.assertEqual(len(fit.checked_roster(rows)),896)
        rows[-1]['split']='independent-evaluation'
        with self.assertRaisesRegex(ValueError,'TRAIN'):fit.checked_roster(rows)
    def test_duplicate_profile_order_and_wrong_target_purpose_refuse(self):
        rows=self.rows();rows[1]['order']='canonical'
        with self.assertRaises(ValueError):fit.checked_roster(rows)
        rows=self.rows();rows[0]['targetPurpose']='arbitrary-evaluation-numerical-control'
        with self.assertRaisesRegex(ValueError,'TRAIN'):fit.checked_roster(rows)
    def test_exact_serialized_row_and_target_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);np.savez_compressed(root/'unit.npz',**{k:np.zeros((1,16,d),dtype='float32') for k,d in [('x',12288),('h',4096),('y',4096)]})
            metadata={'singleChoiceGradientGatePassed':True,'caseId':'unit-only','dtypes':{k:'mlx.core.bfloat16' for k in ('x','h','y')},'routeTokenIds':list(range(16)),'targetWeights':[1.]+[0.]*15}
            tensors={}
            with np.load(root/'unit.npz') as arrays:
                for k in ('x','h','y'):tensors[k]={'sha256':fit.hashlib.sha256(metadata['dtypes'][k].encode()+str(tuple(arrays[k].shape)).encode()+arrays[k].tobytes()).hexdigest()}
            (root/'unit.json').write_text(json.dumps(metadata));(root/'gate.json').write_text(json.dumps({'status':'exact-gate-passed','fixtureOnly':True,
                'sourceSha256':fit.pilot.lexical.sha(fit.features.__file__),'sourceStatePreserved':True,'adapterPreserved':True,
                'comparisons':{'unit-only':{'exact':True}},'featureTensors':tensors}))
            row={'caseId':'unit-only','archive':'unit.npz','metadata':'unit.json','gate':'gate.json',
                 'routeTokenIds':metadata['routeTokenIds'],'targetWeights':metadata['targetWeights']}
            for key in ('archive','metadata','gate'):row[key+'Sha256']=fit.pilot.lexical.sha(root/row[key])
            self.assertEqual(fit.load_feature(root,row)['x'].shape,(1,16,12288))
            row['targetWeights']=[0.,1.]+[0.]*14
            with self.assertRaisesRegex(ValueError,'targets'):fit.load_feature(root,row)
    def test_tampered_archive_refused_before_array_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);file=root/'bad.npz';file.write_bytes(b'original')
            row={'archive':'bad.npz','archiveSha256':fit.pilot.lexical.sha(file)}
            file.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'):fit.load_feature(root,row)

if __name__=='__main__':unittest.main()
