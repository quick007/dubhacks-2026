"""Injected CPU contract tests; no native model, service or speed claims."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import store_candidate_engine_v1 as candidate
from . import store_parallel_text_data_v1 as data
from .store_catalog_v1 import load_snapshot
from .store_demo_engine import RequestCancelled, checked_request
from .store_generated_copy_cache_v1 import warm_plan
from .store_rank_session import action_bindings


def fixture_generation(rows, *, unsafe=False):
    """Invented test strings, never a native/learned-model evidence receipt."""
    fields = []
    for row in rows:
        inp = row['input']; facts = inp.get('source', {}).get('facts', {})
        text = {'section': 'Your selected collection', 'title': f"{facts.get('name')} in {facts.get('selectedColor')}",
            'badge': f"{facts.get('selectedColor')} selected",
            'copy': f"Shown in {facts.get('selectedColor')} at ${facts.get('itemPrice')}."}[inp['kind']]
        if unsafe and inp['kind'] == 'copy':
            text = 'Shown in Ultraviolet at $999.'
        fields.append({'text': text, 'accepted': True, 'rejections': [], 'finishReason': 'eos', 'slotIds': [2, 0]})
    return {'fields': fields, 'fieldCount': len(fields), 'feedforwardCalls': 1, 'autoregressiveSteps': 0,
        'scope': 'invented CPU fixture only'}


class CandidateEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_snapshot(); cls.products = cls.snapshot['inputs']['products']
        request, _ = checked_request({'query': 'browse'})
        cls.cards = action_bindings(request, list(range(6)), cls.snapshot)

    def test_import_does_not_load_native_mlx(self):
        subprocess.run([sys.executable, '-c', 'import sys; import experiments.ranker.pretrained.store_candidate_engine_v1; '
            'assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)'], check=True, capture_output=True)

    def test_variable_public_roster_abstains_incomplete_headings_without_padding(self):
        for count in range(1, 7):
            cards = self.cards[:count]; rows = candidate.parallel_rows(cards)
            self.assertEqual(len(rows), count * 3 + count // 3)
            with patch.object(data, 'teacher', side_effect=AssertionError('serving teacher call')):
                sections, validation = candidate.parallel_payload(cards, rows, fixture_generation(rows), count)
            self.assertEqual(sum(len(s['cards']) for s in sections), count)
            self.assertEqual(validation['requiredFields'], count * 3 + (count + 2) // 3)
            self.assertEqual(validation['withheldFields'], int(count % 3 != 0))
            if count % 3:
                self.assertIsNone(sections[-1]['heading'])
            actual_ids = [c['productId'] for s in sections for c in s['cards']]
            self.assertEqual(actual_ids, [c['action']['productId'] for c in cards])
            self.assertNotIn('targetText', json.dumps(rows))
            self.assertNotIn('teacher', json.dumps(rows))

    def test_six_cards_20_fields_and_first_three_10_same_generation(self):
        rows = candidate.parallel_rows(self.cards); generated = fixture_generation(rows)
        first, first_validation = candidate.parallel_payload(self.cards, rows, generated, 3)
        full, full_validation = candidate.parallel_payload(self.cards, rows, generated, 6)
        self.assertEqual((len(rows), first_validation['requiredFields'], full_validation['requiredFields']), (20, 10, 20))
        self.assertEqual(first[0], full[0]); self.assertTrue(full_validation['allFieldsAccepted'])

    def test_unsafe_accepted_flag_does_not_bypass_factual_gate_or_repair_output(self):
        rows = candidate.parallel_rows(self.cards); generated = fixture_generation(rows, unsafe=True)
        sections, validation = candidate.parallel_payload(self.cards, rows, generated, 6)
        self.assertEqual(validation['withheldFields'], 6)
        self.assertTrue(all(c['bodyCopy'] is None for s in sections for c in s['cards']))
        self.assertIn('Ultraviolet', json.dumps(generated))

    def test_changed_public_binding_roster_rejected(self):
        rows = candidate.parallel_rows(self.cards); generated = fixture_generation(rows)
        rows[1]['featureKeys'] = ['wrong-public-key']
        with self.assertRaisesRegex(ValueError, 'source-conditioned'):
            candidate.parallel_payload(self.cards, rows, generated, 6)

    def test_cache_source_flags_are_dynamic_for_hit_miss_and_mixed(self):
        for hits, misses, kind in [(8, 0, 'generated-earlier'), (0, 8, 'fresh-native'), (5, 3, 'mixed-native-cache')]:
            raw = {'initialHits': hits, 'initialMisses': misses, 'entries': [{'ageMs': 1234, 'generatedEarlier': bool(hits)}],
                'nativeColdGenerationMsThisRequest': 5500 if misses else 0, 'bindingSha256': 'a' * 64,
                'disclosure': 'Earlier native model text'}
            source = candidate.source_for_cache(raw)
            self.assertEqual(source['kind'], kind)
            self.assertEqual(source['generatedEarlier'], bool(hits))
            self.assertEqual(source['freshPerRequestGeneration'], bool(misses))
            self.assertEqual(source.get('maxAgeMs'), 1234 if hits else None)

    def fake_engine(self, directory, mode='parallel'):
        engine = object.__new__(candidate.CandidateDemoEngine)
        engine.output = Path(directory); engine.mode = mode; engine.head = object(); engine.feature_map = {}
        engine.adapter = object(); engine.copy_sha = 'copy-test'; engine.copy_metadata = {'checkpointSha256': 'head-test'}
        engine._assert_candidate = lambda: None
        engine.session = SimpleNamespace(snapshot=self.snapshot, products=self.products, head_sha='rank-test',
            rank_request=lambda request, **kwargs: {'scores16': list(range(16)), 'fixture': True})
        engine.binding = SimpleNamespace(assert_current_sources=lambda receipt: None)
        engine.bb = SimpleNamespace()
        return engine

    def personalize(self, value, *, mode='parallel', generated=None, cancelled=lambda: False):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(directory, mode); events = []
            generate = generated or (lambda _head, rows, _features: fixture_generation([{'input': r['input'], 'featureKeys': r['featureKeys']} for r in rows]))
            with patch('experiments.ranker.pretrained.store_parallel_text_head_v1.generate', side_effect=generate), \
                    patch('experiments.ranker.pretrained.store_copy_wide_v2.guard_memory', return_value={'fixture': True}):
                engine.personalize(value, 'case', time.perf_counter(), events.append, cancelled)
            return events, json.loads((Path(directory) / 'requests/case/receipt.json').read_text())

    def test_parallel_stream_preserves_rank_actions_and_contiguous_endpoint_clock(self):
        events, receipt = self.personalize({'query': 'water for my day'})
        self.assertEqual([e['type'] for e in events], ['rank', 'firstThree', 'complete'])
        self.assertEqual(len(events[0]['productIds']), 16); self.assertEqual(len(events[0]['selectedProductIds']), 6)
        self.assertEqual(events[1]['requiredFields'], 10); self.assertEqual(events[2]['requiredFields'], 20)
        self.assertLessEqual(events[0]['timing']['readyMs'], events[1]['timing']['readyMs'])
        self.assertLessEqual(events[1]['timing']['readyMs'], events[2]['timing']['readyMs'])
        self.assertEqual(events[2]['copySource']['kind'], 'fresh-parallel-head')
        self.assertTrue(events[2]['copySource']['freshPerRequestGeneration']); self.assertFalse(events[2]['copySource']['generatedEarlier'])
        self.assertEqual(receipt['status'], 'completed'); self.assertFalse(receipt['copyShopperQueryUsed'])
        self.assertEqual(receipt['effectiveRequest']['query'], 'water for my day')

    def test_exact_name_ineligible_budget_has_zero_cards_no_copy_call(self):
        def never(*args):
            raise AssertionError('No copy generation for ineligible exact-name request')
        events, receipt = self.personalize({'query': 'Trail mug under $20'}, generated=never)
        self.assertEqual([e['type'] for e in events], ['rank', 'complete'])
        self.assertEqual(events[0]['selectedProductIds'], []); self.assertEqual(events[1]['requiredFields'], 0)
        self.assertEqual(events[1]['shortfall'], 6); self.assertEqual(receipt['cards'], [])
        self.assertFalse(events[1]['copySource']['freshPerRequestGeneration'])
        self.assertEqual(events[1]['copySource']['kind'], 'no-generation')

    def test_typed_color_binds_real_matching_variant_and_guard_unknown_stock_empty(self):
        events, _ = self.personalize({'query': 'a bottle', 'hardRequirements': [{'attribute': 'color', 'value': 'Sage'}]})
        cards = [c for s in events[-1]['sections'] for c in s['cards']]
        self.assertTrue(cards); self.assertTrue(all(c['selectedColor'] == 'Sage' for c in cards))
        stock, _ = self.personalize({'query': 'a bottle in stock'})
        self.assertEqual(stock[-1]['sections'], [])

    def test_cancelled_request_archived_and_copy_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(directory); events = []
            with self.assertRaises(RequestCancelled):
                engine.personalize({'query': 'browse'}, 'cancel', time.perf_counter(), events.append, lambda: True)
            receipt = json.loads((Path(directory) / 'requests/cancel/receipt.json').read_text())
            self.assertEqual(receipt['status'], 'cancelled'); self.assertEqual(events, [])
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(directory); events = []
            with patch('experiments.ranker.pretrained.store_parallel_text_head_v1.generate', side_effect=ValueError('head failure')):
                with self.assertRaisesRegex(ValueError, 'head failure'):
                    engine.personalize({'query': 'browse'}, 'fail', time.perf_counter(), events.append, lambda: False)
            receipt = json.loads((Path(directory) / 'requests/fail/receipt.json').read_text())
            self.assertEqual(receipt['status'], 'failed'); self.assertEqual([e['type'] for e in events], ['rank'])
            self.assertIn('head failure', receipt['error']['message'])

    def test_cache_stream_uses_resolved_native_rows_and_marks_earlier_text(self):
        from .store_demo_engine import demo_layout
        with tempfile.TemporaryDirectory() as directory:
            engine = self.fake_engine(directory, 'generated-cache'); events = []; calls = []
            def resolve(cards, *, request_started, native_generate):
                calls.append((len(cards), callable(native_generate), request_started))
                _, rows = demo_layout(cards)
                for row in rows:
                    field = row['fields'][0]
                    if field[0] == 'S':
                        text = 'Your selected collection'
                    else:
                        facts = cards[int(field[1:]) - 1]['facts']
                        text = (facts['selectedColor'] + ' selected' if field[0] == 'B' else
                            facts['name'] + ' in ' + facts['selectedColor'] + ('.' if field[0] == 'C' else ''))
                    row.update(text=text, tokenIds=[7], rawSampledTokenIds=[7, 99], stopTokenId=99,
                        arSteps=2, finishReason='stop')
                return {'rawRows': rows, 'validation': {}, 'cache': {'initialHits': 8, 'initialMisses': 0,
                    'entries': [{'ageMs': 123, 'generatedEarlier': True}], 'nativeColdGenerationMsThisRequest': 0,
                    'bindingSha256': 'fixture', 'disclosure': 'Previously generated native model copy'}}
            engine.copy_cache = SimpleNamespace(resolve=resolve)
            with patch('experiments.ranker.pretrained.store_copy_wide_v2.guard_memory', return_value={'fixture': True}), \
                    patch('experiments.ranker.pretrained.store_parallel_text_head_v1.generate', side_effect=AssertionError('cache must not call head')):
                engine.personalize({'query': 'water'}, 'cached', time.perf_counter(), events.append, lambda: False)
            self.assertEqual(len(calls), 1); self.assertEqual(calls[0][:2], (6, True))
            self.assertEqual([e['type'] for e in events], ['rank', 'firstThree', 'complete'])
            self.assertEqual(events[-1]['requiredFields'], 20)
            self.assertEqual(events[-1]['copySource']['kind'], 'generated-earlier')
            self.assertFalse(events[-1]['copySource']['freshPerRequestGeneration'])
            receipt = json.loads((Path(directory) / 'requests/cached/receipt.json').read_text())
            self.assertEqual(len(receipt['rawRows']), 20); self.assertEqual(receipt['status'], 'completed')

    def test_public_only_asset_loader_never_reads_labels_results_or_teachers(self):
        # All tensors/head calls injected; this verifies the serving file boundary.
        records = []
        for group in warm_plan(self.snapshot)['groups']:
            records.extend(data.public_record(c['facts']) for c in group['cards'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); head = root / 'head.pt'; features = root / 'features.pt'; public = root / 'public.json'
            head.write_bytes(b'fixture-not-model'); features.write_bytes(b'fixture-not-features')
            public.write_text(json.dumps({'schemaVersion': 'parallel-word-public-runtime-features.v1', 'records': records, 'teacherOrLabelContent': False}))
            manifest = {'scope': 'TRAIN-ALL public catalog demo; no heldout evaluation', 'all48BindingsSupervised': True,
                'runtimeTeacherLookup': False, 'heldoutQualityClaim': False,
                'headFile': str(head), 'headSha256': candidate.sha(head), 'featureBundleFile': str(features),
                'featureBundleSha256': candidate.sha(features), 'publicFeatureContractFile': str(public),
                'publicFeatureContractSha256': candidate.sha(public)}
            (root / 'serving-manifest.json').write_text(json.dumps(manifest))
            original_read = Path.read_text
            def guarded(path, *args, **kwargs):
                if any(w in path.name.lower() for w in ('labels', 'replay', 'results')):
                    raise AssertionError('Offline evaluation/training file read during serving')
                return original_read(path, *args, **kwargs)
            def validate(bundle, packet):
                self.assertEqual(set(packet), {'featureRecords'}); self.assertEqual(packet['featureRecords'], records)
                return {'fixture': True}
            with patch.object(Path, 'read_text', guarded), patch.object(data, 'teacher', side_effect=AssertionError('teacher call')), \
                    patch('experiments.ranker.pretrained.store_parallel_text_demo_all_v1.load_head', return_value=(SimpleNamespace(vocabulary=['eos']), {'featureBundleSha256': manifest['featureBundleSha256']})), \
                    patch('torch.load', return_value={}), patch('experiments.ranker.pretrained.store_parallel_text_head_v1.validate_bundle', side_effect=validate):
                _, _, loaded, metadata = candidate.load_parallel_assets(root)
            self.assertEqual(loaded, records); self.assertFalse(metadata['teacherStringsReadAtServing'])
            head.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                candidate.load_parallel_assets(root)


if __name__ == '__main__':
    unittest.main()
