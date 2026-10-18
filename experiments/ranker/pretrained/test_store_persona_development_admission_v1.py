from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import store_persona_development_admission_v1 as demo


class DevelopmentAdmissionTests(unittest.TestCase):
    def test_actual_failed_quality_and_mechanical_passes_are_both_preserved_cpu_only(self):
        with patch.object(demo.probe.old,'exchange',side_effect=AssertionError('No HTTP in admission')):
            record=demo.build_record();result=demo.evidence(record,demo.ROOT,require_startup=False)
        self.assertEqual(result['mode'],'technical-development-demo')
        self.assertFalse(result['modelPromotion']);self.assertFalse(result['qualityGatePassed']);self.assertFalse(result['acceptedProfile'])
        self.assertEqual(result['qualityFailure'],{'rawPrimaryTop1':17,'rawFrozenBasePrimaryTop1':20,
            'combinedPrimaryTop1':17,'combinedFrozenBasePrimaryTop1':19,'denominator':32})
        self.assertEqual(result['expected']['adapterSha256'],demo.FIXED['adapter'])
        self.assertIsNone(result['expected']['startup'])

    def test_no_quality_pass_or_accepted_profile_alias(self):
        original=demo.build_record()
        for mutate in [lambda r:r.update(qualityGatePassed=True),lambda r:r.update(modelPromotion=True),
                       lambda r:r.update(acceptedProfile=True),lambda r:r.update(status='accepted'),
                       lambda r:r.update(version='persona-adapter-serving-candidate.v1')]:
            record=deepcopy(original);mutate(record)
            with self.assertRaisesRegex(ValueError,'non-promoted'):demo.evidence(record,demo.ROOT,require_startup=False)

    def test_changed_failure_receipt_or_gate_checksum_cannot_be_relabelled(self):
        record=demo.build_record();record['offlineAudit']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'changed'):demo.evidence(record,demo.ROOT,require_startup=False)
        record=demo.build_record();record['gateCheckpoint']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'changed'):demo.evidence(record,demo.ROOT,require_startup=False)

    def test_strict_offline_pass_selector_still_rejects_development_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'development.json';demo.probe.old.write(path,demo.build_record())
            with self.assertRaisesRegex(ValueError,'candidate'):demo.probe.selection(path,demo.probe.old.sha(path))
            with self.assertRaises((KeyError,ValueError)):demo.selection(path,demo.probe.old.sha(path))


if __name__=='__main__':unittest.main()
