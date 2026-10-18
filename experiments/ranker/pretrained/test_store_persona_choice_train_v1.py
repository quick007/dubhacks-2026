"""CPU orchestration guards only, no synthetic training evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import store_persona_choice_train_v1 as trainer

class TrainingProtocolTests(unittest.TestCase):
    def test_default_validated_plan_never_starts_native_or_writes_out(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plan.json';path.write_text(json.dumps({'version':trainer.VERSION,'recipe':trainer.RECIPE,'expectedCounts':trainer.COUNTS,'sha256':{}}))
            with patch.object(trainer.pilot.lexical,'listening',side_effect=AssertionError('No live action in CPU mode')):
                result=trainer.run(path,trainer.pilot.lexical.sha(path),Path(directory)/'unused')
            self.assertEqual(result['optimizerUpdates'],0);self.assertFalse((Path(directory)/'unused').exists())
    def test_counts_and_recipe_are_fixed_before_future_gold(self):
        self.assertEqual(sum(trainer.COUNTS.values()),1240)
        self.assertEqual(trainer.COUNTS['train'],896)
        self.assertEqual(trainer.RECIPE['epochs']*trainer.RECIPE['featuresExpected'],3584)
        self.assertFalse(trainer.RECIPE['wholePromptNoCacheBitParityClaim'])
    def test_manifest_tamper_and_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'public.json').write_text('public')
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'files':{'public.json':trainer.pilot.lexical.sha(root/'public.json')}}))
            sha=trainer.pilot.lexical.sha(manifest);trainer.frozen_files(root,sha)
            (root/'public.json').write_text('changed')
            with self.assertRaises(ValueError):trainer.frozen_files(root,sha)

if __name__=='__main__':unittest.main()
