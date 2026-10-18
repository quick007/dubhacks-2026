"""CPU-only evidence binding and exact multi-turn cache-boundary checks."""
import copy
import unittest
from .shared_evidence_experiment import shared_layout,exact_messages_suffix,message_suffixes,same_outputs


class SharedEvidenceTests(unittest.TestCase):
    def test_all_fields_caps_and_card_lexicons_preserved(self):
        from .test_heading_experiment import HeadingExperimentTests
        plan,request,catalog=HeadingExperimentTests().fixture(6)
        before=copy.deepcopy((plan,request,catalog))
        common,rows,evidence=shared_layout(plan,request,catalog)
        self.assertEqual((plan,request,catalog),before)
        self.assertEqual(len(rows),14);self.assertEqual(sum(r['maxTokens'] for r in rows),224)
        self.assertEqual(len(evidence['cards']),6);self.assertEqual(len(evidence['sections']),2)
        for i,facts in enumerate(evidence['cards']):
            self.assertEqual(facts['identity'],catalog[str(i)]['title'])
            self.assertEqual(facts['approvedTitles'],[catalog[str(i)]['presentationVariants'][0]['title']])
        self.assertIn('requiredObservedTags',common['content'])
        self.assertNotIn('Selected public evidence:',rows[2]['messages'][0]['content'])

    def test_exact_multiple_turns_and_consumption(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                self.seen=messages;return [1,2,3,4]
        tok=Tokenizer();model=object();cache=object();state={'model':model,'consumed':False,'tokens':[1,2],
            'messages':[{'role':'user','content':'rank'}],'cache':cache}
        row={'messages':[{'role':'user','content':'common'},{'role':'user','content':'field'}]}
        suffix,returned=exact_messages_suffix(tok,row,state,model)
        self.assertEqual([m['content'] for m in tok.seen],['rank','common','field'])
        self.assertEqual(suffix,[3,4]);self.assertIs(returned,cache);self.assertTrue(state['consumed'])
        with self.assertRaises(ValueError):exact_messages_suffix(tok,row,state,model)

    def test_suffix_hook_restored_on_error(self):
        from . import parallel_decode
        original=parallel_decode.exact_row_suffix
        with self.assertRaises(RuntimeError):
            with message_suffixes():
                self.assertIs(parallel_decode.exact_row_suffix,exact_messages_suffix)
                raise RuntimeError('injected failure')
        self.assertIs(parallel_decode.exact_row_suffix,original)


if __name__=='__main__':unittest.main()
