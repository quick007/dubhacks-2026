"""CPU orchestration/corruption tests; synthetic embeddings are not quality evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import focused_rank_transfer as m
from .catalog_fusion.test_runner import CpuEncoderStub, provenance, token_record


class FocusedTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        _, cls.catalog, cls.requests = m.public_inputs(m.DATA)

    def test_hard_masks_finite_cardinality_and_rotations(self):
        request = next(iter(self.requests.values())); order = request['candidateProductIds']
        scores = torch.arange(36, dtype=torch.float32)
        baseline = m.ranked(scores, order, request, self.catalog)
        for k in m.COUNTS:
            self.assertEqual(len(baseline['top'][str(k)]), k)
            self.assertFalse(any(m.guard_reasons(request,self.catalog[p]) for p in baseline['top'][str(k)]))
        for offset in m.OFFSETS:
            ids = order[offset:]+order[:offset]
            self.assertEqual(m.ranked(scores[[order.index(p) for p in ids]], ids, request, self.catalog),baseline)
        for corrupt in (torch.zeros(35),torch.full((36,),float('nan'))):
            with self.assertRaises(ValueError):m.ranked(corrupt,order,request,self.catalog)
        with self.assertRaises(ValueError):m.ranked(scores,order[:-1]+order[:1],request,self.catalog)

    def test_frozen_head_checksum_and_metadata_fail_closed(self):
        receipt, _, _ = m.heads(m.SOURCE)
        row=receipt['heads']['learned']; path=m.SOURCE/row['file']
        with self.assertRaises(ValueError):m.load_head(path,'0'*64,row['metadata'])
        metadata=copy.deepcopy(row['metadata']);metadata['arm']='corrupt'
        with self.assertRaises(ValueError):m.load_head(path,row['sha256'],metadata)

    def test_cpu_stub_exact60_freezes_before_labels_and_keeps_denominators(self):
        receipt, models, old = m.heads(m.SOURCE)
        rows=[{'role':'catalog','member':p,'text':m.catalog_text_v2(v,'focused')} for p,v in self.catalog.items()]
        rows += [{'role':'request','member':r,'text':m.query_text_v2(v,self.catalog,'focused')} for r,v in self.requests.items()]
        rows=[r|token_record(r) for r in rows]
        packet={'sourceHeads':receipt,'sourceHashes':m.sources(),'datasetDirectory':str(m.DATA),
            'dataFreezeSha256':m.file_digest(m.DATA/'FREEZE.json'),'limitations':['CPU-test-stub; no quality evidence']}
        encoder=CpuEncoderStub()
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);m.write_json(out/'packet.json',packet)
            real_verify=m.verify_dataset
            def checked_verify(dataset,private=False):
                if private:
                    self.assertEqual(len(encoder.calls),60)
                    self.assertTrue((out/'PREDICTIONS_FROZEN.json').exists())
                    self.assertTrue(encoder.closed)
                return real_verify(dataset,private=private)
            with patch.object(m,'heads',return_value=(receipt,models,old)),patch.object(m,'verify_dataset',side_effect=checked_verify):
                m.run(out,(packet,self.catalog,self.requests,rows,provenance()),lambda:encoder)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['nativeForwards'],60);self.assertEqual(report['fits'],0)
            self.assertEqual(report['evidenceSource'],'cpu-test-stub')
            self.assertEqual(len(report['observations']),24)
            self.assertEqual(report['quality']['public-tag-oracle']['uniqueGoldCorrect'],24)
            self.assertEqual(len(list((out/'features').glob('*/*'))),120)
            predictions=json.loads((out/'predictions.json').read_text())
            labels={r['recordId']:r for r in map(json.loads,(m.DATA/'private/truth.jsonl').read_text().splitlines())}
            controls=json.loads((m.DATA/'private/controls.json').read_text())
            predictions.pop(next(iter(predictions)))
            with self.assertRaises(ValueError):m.evaluate(predictions,labels,controls)

    def test_encoder_source_mismatch_rejected_before_forward(self):
        receipt, models, old = m.heads(m.SOURCE)
        encoder=CpuEncoderStub();encoder.evidence_source='wrong'
        packet={'sourceHeads':receipt,'sourceHashes':m.sources()}
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'heads',return_value=(receipt,models,old)):
            with self.assertRaisesRegex(ValueError,'provenance'):
                m.run(Path(directory),(packet,self.catalog,self.requests,[],provenance()),lambda:encoder)
        self.assertEqual(encoder.calls,[]);self.assertTrue(encoder.closed)

if __name__=='__main__':unittest.main()
