"""CPU accounting checks; model parity is checked on every native profile pair."""
import unittest
from .manifest_profile import field_delivery


class ManifestProfileTests(unittest.TestCase):
    def test_buffered_multiline_delivery_and_unterminated_last_line(self):
        text='S1: Travel\nT1: New title\nB1: NONE'
        events=[{'relativeMs':10,'charEnd':0},
                {'relativeMs':30,'charEnd':len('S1: Travel\nT1: New title\n')},
                {'relativeMs':50,'charEnd':len(text)},
                {'relativeMs':60,'charEnd':len(text)}]
        rows=field_delivery(text,events,5)
        self.assertEqual([r['label'] for r in rows],['S1','T1','B1'])
        self.assertEqual([r['postPromptDeliveryWindowMs'] for r in rows],[25,0,20])
        self.assertEqual([r['lineDeliveredMs'] for r in rows],[30,30,50])
        self.assertEqual(rows[-1]['rawLine'],'B1: NONE')

    def test_empty_and_missing_events(self):
        self.assertEqual(field_delivery('',[],1),[])
        with self.assertRaises(ValueError):field_delivery('T1: text',[],1)


if __name__=='__main__':unittest.main()
