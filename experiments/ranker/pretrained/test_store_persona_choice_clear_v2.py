"""CPU objective/schedule/admission fixtures, never training/native evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import store_persona_choice_clear_fit_v2 as fit
from . import store_persona_choice_clear_train_v2 as driver

class ClearFitTests(unittest.TestCase):
    def test_single_fixed_schedule_uses_clear256_fourteen_times_and_teacher640_pool(self):
        ce,teacher=fit.schedules()
        self.assertEqual(len(ce),3584);self.assertEqual(len(teacher),3584)
        self.assertEqual(set(ce),set(range(256)));self.assertTrue(all(ce.count(i)==14 for i in range(256)))
        self.assertEqual(set(teacher),set(range(640)));self.assertTrue(all(teacher.count(i) in (5,6) for i in range(640)))
        self.assertEqual((ce,teacher),fit.schedules());self.assertFalse(fit.RECIPE['oldAuthoredPriorityLabelsInOptimizer'])
        self.assertEqual(fit.RECIPE['learningRate'],.0001);self.assertEqual(fit.RECIPE['teacherWeight'],.25)
    def test_teacher_targets_are_stable_source_logit_softmax_not_authored_grades(self):
        weights=fit.teacher_weights([0.]*16);self.assertEqual(weights,[1/16]*16)
        scores=list(range(16));a=fit.teacher_weights(scores);b=fit.teacher_weights(scores[::-1])
        self.assertAlmostEqual(sum(a),1);self.assertEqual(a,b[::-1]);self.assertGreater(a[-1],a[0])
        with self.assertRaises(ValueError):fit.teacher_weights([float('nan')]*16)
    def test_one_way_public_projection_only_attaches_explicit_new_objective_targets(self):
        pure={'x':'x','h':'h','y':'y','dtypes':{},'routeTokenIds':list(range(16))};teacher=[1/16]*16
        with patch.object(fit.diagnostic,'load_public_feature',return_value=pure):
            item=fit.objective_item('/original',{'targetWeights':['LEGACY_AUTHORED_TARGETS']},{'targetWeights':teacher})
        self.assertEqual(item['targetWeights'],teacher);self.assertNotIn('LEGACY_AUTHORED_TARGETS',repr(item))
        self.assertEqual(set(item),set(pure)|{'targetWeights','singleChoiceGradientGatePassed'})
    def test_pair_loss_adds_fixed_teacher_weight_without_hard_owned_mask(self):
        with patch.object(fit.original.features,'feature_value',side_effect=[(2.,'clear-scores'),(4.,'teacher-scores')]):
            loss,aux=fit.pair_value(None,None,{'owned-positive':'allowed-replacement'},{'teacher':'public-context'})
        self.assertEqual(loss,3.);self.assertEqual(aux['clearLoss'],2.);self.assertEqual(aux['teacherLoss'],4.)
    def test_default_validated_plan_does_not_initialize_native_or_write_output(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);p=root/'plan.json';p.write_text(json.dumps({'version':driver.VERSION,'recipe':driver.RECIPE,'expectedCounts':driver.COUNTS,'sha256':{}}))
            with patch.object(driver,'owned_model',side_effect=AssertionError('No native in CPU default')):
                result=driver.run(p,driver.pilot.lexical.sha(p),root/'out')
            self.assertEqual(result['nativeCalls'],0);self.assertFalse((root/'out').exists())
    def test_memory_deadline_update_and_export_contract_preserved(self):
        self.assertEqual(driver.RECIPE['maxNativeSeconds'],900);self.assertEqual(driver.RECIPE['memoryGiB'],7)
        self.assertEqual(driver.RECIPE['epochs']*driver.RECIPE['featuresExpected'],3584)
        self.assertEqual(sum(driver.COUNTS.values()),1240);self.assertEqual(driver.RECIPE['cachedScoringMode'],'plain')
        self.assertEqual(driver.RECIPE['dynamicMode'],'gradient-public-XHY-last16')

if __name__=='__main__':unittest.main()
