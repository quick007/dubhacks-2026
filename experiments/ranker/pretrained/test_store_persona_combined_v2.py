import unittest
from unittest.mock import patch

from . import store_persona_combined_predictions_v1 as prior
from . import store_persona_combined_predictions_v2 as current
from . import test_store_persona_combined_v1 as fixtures
from .audit_persona_combined_query_replay_v1 import check_combined_rows


class ExactRawProvenanceTests(unittest.TestCase):
    setUp=fixtures.CombinedPredictionTests.setUp

    def test_original_fraction_json_values_preserved_without_blend_or_order_change(self):
        for name in self.raw:
            for row in self.raw[name]:row['scores16']=[i/97 for i in range(16)]
        before=prior.compose_rows(self.rows,self.raw,self.values,self.products,gate_sha='a'*64)
        after=current.compose_rows(self.rows,self.raw,self.values,self.products,gate_sha='a'*64)
        self.assertNotEqual(before['TRAIN-popularity'][0]['rawSourceScores16'],self.raw['TRAIN-popularity'][0]['scores16'])
        for name in self.raw:
            for source,old,new in zip(self.raw[name],before[name],after[name],strict=True):
                self.assertEqual(new['rawSourceScores16'],source['scores16'])
                self.assertEqual(new['scores16'],old['scores16']);self.assertEqual(new['order'],old['order'])
                self.assertEqual(new['rawLearnedOrder'],old['rawLearnedOrder'])
        # The full344 fixture below exercises the independently frozen checker.

    def test_full344_fake_writer_contract_uses_v2_and_independent_replay(self):
        original=fixtures.combined
        def checked_compose(rows,raw,values,products,*,gate_sha):
            result=current.compose_rows(rows,raw,values,products,gate_sha=gate_sha)
            for name,records in result.items():
                check_combined_rows(records,rows,values,products,gate_sha)
                source={r['caseId']:r for r in raw[name]}
                self.assertTrue(all(r['rawSourceScores16']==source[r['caseId']]['scores16'] for r in records))
            return result
        # Exercise original orchestration fixture against V2; the actual routine
        # and independent checker run, while only native/artifact loaders are fake.
        with patch.object(fixtures,'combined',current),patch.object(current,'compose_rows',wraps=current.compose_rows) as compose:
            fixtures.CombinedWriterFreezeTests.test_full344_writer_freezes_all_baseline_files_with_actual_trainer_key_names(self)
            self.assertEqual(compose.call_count,1)
            rows,raw,values,products=compose.call_args.args
            checked_compose(rows,raw,values,products,gate_sha=compose.call_args.kwargs['gate_sha'])
        self.assertIs(fixtures.combined,original)


if __name__=='__main__':unittest.main()
