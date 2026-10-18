"""CPU-only lifecycle/preflight tests; mocked rows never constitute native evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import store_persona_choice_inference_pilot_v1 as pilot

class PilotTests(unittest.TestCase):
    def plan(self,directory):
        source=directory/'unit-source';source.write_text('pinned')
        plan=directory/'plan.json';plan.write_text(json.dumps({'version':pilot.VERSION,
            'recipe':pilot.RECIPE,'rows':[{},{}],'sha256':{str(source):pilot.lexical.sha(source)}}))
        return plan,source
    def test_default_preflight_never_opens_native_lock_or_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,_=self.plan(root)
            with patch.object(pilot.lexical,'listening',side_effect=AssertionError('No native listener inspection required')):
                result=pilot.run(plan,pilot.lexical.sha(plan),root/'never-created')
            self.assertEqual(result['optimizerUpdates'],0);self.assertFalse(result['nativeExecuted'])
            self.assertFalse((root/'never-created').exists())
    def test_changed_pins_and_wrong_plan_checksum_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,source=self.plan(root)
            with self.assertRaisesRegex(ValueError,'SHA'):pilot.run(plan,'wrong',root/'unused')
            source.write_text('changed')
            with self.assertRaises(ValueError):pilot.run(plan,pilot.lexical.sha(plan),root/'unused')
    def test_native_permission_requires_exact_root_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,_=self.plan(root)
            with self.assertRaisesRegex(ValueError,'Root'):pilot.run(plan,pilot.lexical.sha(plan),root/'unused',execute=True,owner=None)
    def test_model_listener_blocks_before_outdir_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan,_=self.plan(root)
            with patch.object(pilot.lexical,'listening',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'Retire'):pilot.run(plan,pilot.lexical.sha(plan),root/'unused',execute=True,owner='root')
            self.assertFalse((root/'unused').exists())
    def test_recipe_has_no_optimizer_pilot_and_fixed_training_steps(self):
        self.assertEqual(pilot.RECIPE['pilotOptimizerUpdates'],0)
        self.assertEqual(pilot.RECIPE['epochs']*pilot.RECIPE['featuresExpected'],pilot.RECIPE['updatesExpected'])

if __name__=='__main__':unittest.main()
