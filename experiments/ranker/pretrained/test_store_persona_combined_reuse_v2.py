"""CPU exact-cache/source-contract tests; no fabricated native evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import store_persona_combined_query_reuse_v2 as q
from . import store_persona_choice_clear_combined_predictions_v1 as writer
from . import store_persona_combined_predictions_v2 as old_writer

def row(text):
    ids=[ord(c) for c in text]
    return {'text':text,'textSha256':q.digest_text(text),'tokenIds':ids,'tokens':len(ids),'tokenIdsSha256':q.original.contract.token_hash(ids)}
def vector(index):
    value=torch.zeros(1,4096);value[0,index]=1.;return value

class ReuseTests(unittest.TestCase):
    def test_only_exact_full_text_and_tokens_reuse_no_prefix_or_profile_substitution(self):
        a,b=row('exact query under35'),row('exact query under36')
        reused,fresh=q.select_reuse([a,b],[a]);self.assertEqual(reused,[a['textSha256']]);self.assertEqual(fresh,[b['textSha256']])
        changed=copy.deepcopy(a);changed['tokenIds']=[1]
        with self.assertRaisesRegex(ValueError,'different complete text/token'):q.select_reuse([changed],[a])
    def test_assembly_preserves_actual_source_tensor_bits_and_refuses_missing_fresh(self):
        a,b=row('a'),row('b');packet={'uniqueQueries':[a,b],'reusedKeys':[a['textSha256']],'freshKeys':[b['textSha256']]}
        prior={'uniqueQueries':[a]};fresh={'uniqueQueries':[b]};original={a['textSha256']:vector(0)};new={b['textSha256']:vector(1)}
        result=q.assemble_values(packet,prior,original,fresh,new)
        self.assertTrue(torch.equal(result[a['textSha256']],original[a['textSha256']]));self.assertNotEqual(result[a['textSha256']].data_ptr(),original[a['textSha256']].data_ptr())
        with self.assertRaisesRegex(ValueError,'actual fresh native'):q.assemble_values(packet,prior,original)
        with self.assertRaises(ValueError):q.assemble_values(packet,prior,original,{'uniqueQueries':[a]},new)
    def test_all_reuse_requires_no_new_native_feature(self):
        a=row('a');packet={'uniqueQueries':[a],'reusedKeys':[a['textSha256']],'freshKeys':[]}
        result=q.assemble_values(packet,{'uniqueQueries':[a]},{a['textSha256']:vector(3)})
        self.assertEqual(set(result),{a['textSha256']})
    def test_default_preflight_never_calls_native_or_creates_output(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);native=root/'fresh.json';native.write_text('{}')
            packet={'version':q.VERSION,'recipe':q.RECIPE,'sha256':{},'counts':{},'freshNativePlan':{'file':str(native),'sha256':q.sha(native)}}
            plan=root/'plan.json';plan.write_text(json.dumps(packet))
            with patch.object(q.original,'extract',side_effect=AssertionError('No native in CPU default')):
                result=q.extract(plan,q.sha(plan),root/'unused')
            self.assertFalse(result['nativeExecuted']);self.assertFalse((root/'unused').exists())
    def test_combined_formula_and_all_control_logic_reused_without_changes(self):
        self.assertEqual(writer.FUSION,old_writer.FUSION);self.assertEqual(writer.MODEL_FILES,old_writer.MODEL_FILES)
        self.assertEqual(writer.compose_rows.__code__.co_code,old_writer.compose_rows.__code__.co_code)

if __name__=='__main__':unittest.main()
