"""CPU fixtures only; no real native output, model execution or speed claim."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest

from .store_catalog_v1 import load_snapshot
from .store_copy_data_v2 import target_values
from .store_generated_copy_cache_v1 import (GeneratedCopyCache, NativeGenerationRequired,
    catalog_card, hashed, personalize_with_cache, producer_spec, warm_plan)
from .store_demo_engine import demo_layout


class GeneratedCopyCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = load_snapshot()
        cls.spec = producer_spec('a' * 64, 'pinned-test-revision', 'b' * 64)
        cls.plan = warm_plan(cls.snapshot)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.now = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
        self.cache = GeneratedCopyCache(Path(self.temporary.name) / 'cache', self.snapshot, self.spec,
            clock=lambda: self.now.isoformat())

    def generated(self, cards, *, fresh=False, unsafe_lamp=False):
        common, rows = demo_layout(cards)
        values = target_values({'cards': cards}, 0)
        for row in rows:
            field = row['fields'][0]; text = values[field]
            if unsafe_lamp and field.startswith('C') and cards[int(field[1:]) - 1]['action']['productId'] == 'lamp':
                text = f"This {cards[int(field[1:]) - 1]['facts']['name']} has 12 hours of battery."
            row.update(text=text, tokenIds=[5], rawSampledTokenIds=[5, 99], stopTokenId=99,
                arSteps=2, finishReason='stop')
        at = self.now if fresh else self.now - timedelta(minutes=10)
        return {'rows': rows, 'startedAt': at.isoformat(), 'completedAt': at.isoformat(),
            'coldGenerationMs': 8000, 'provenance': {'nativeGeneration': True, 'fallback': False,
                'checkpointSha256': self.spec['checkpointSha256'], 'modelRevision': self.spec['modelRevision'],
                'tokenizerSha256': self.spec['tokenizerSha256'], 'nativeWidth': 10, 'sameWidthControlsSha256': 'c' * 64}}

    def test_import_never_loads_mlx(self):
        subprocess.run([sys.executable, '-c', 'import sys; import experiments.ranker.pretrained.store_generated_copy_cache_v1; '
            'assert not any(k == "mlx" or k.startswith("mlx.") for k in sys.modules)'], check=True, capture_output=True)

    def test_fixed_warm_plan_covers48_real_colors_with_eight_complete_six_card_requests(self):
        self.assertEqual(len(self.plan['groups']), 8)
        self.assertEqual({(g['fields'], g['sampledCap']) for g in self.plan['groups']}, {(20, 416)})
        cards = [card for group in self.plan['groups'] for card in group['cards']]
        self.assertEqual(len({(c['action']['productId'], c['action']['variantId']) for c in cards}), 48)
        self.assertFalse(self.plan['nativeExecution'])

    def test_hits_explicitly_generated_earlier_with_age_cost_and_all20_serialized_fields(self):
        cards = self.plan['groups'][0]['cards']
        self.cache.record_generation(cards, self.generated(cards))
        result = self.cache.resolve(cards, request_started=time.perf_counter() - .01)
        self.assertEqual(result['cache']['copySource'], 'generated-earlier')
        self.assertFalse(result['cache']['freshPerRequestGenerationClaim'])
        self.assertEqual(result['cache']['initialHits'], 8)
        self.assertEqual(result['cache']['initialMisses'], 0)
        self.assertEqual(result['cache']['nativeColdGenerationMsThisRequest'], 0)
        self.assertTrue(all(e['generatedEarlier'] and e['ageMs'] == 600000 for e in result['cache']['entries']))
        self.assertEqual(len(result['rawRows']), 20)
        self.assertEqual(result['validation']['requiredFields'], 20)
        self.assertEqual(sum(len(s['cards']) for s in result['payload']['sections']), 6)
        self.assertEqual(len(result['encodedPayload']), result['payloadBytes'])
        self.assertGreaterEqual(result['validatedAndSerializedAtMs'], 10)
        self.assertTrue(result['requestTimingIncludesCallerRanking'])

    def test_position_rebinding_preserves_exact_native_text_tokens_eos_and_prompt(self):
        cards = self.plan['groups'][0]['cards']; generated = self.generated(cards)
        self.cache.record_generation(cards, generated)
        result = self.cache.resolve(list(reversed(cards)))
        moved = next(row for row in result['rawRows'] if row['fields'] == ['T1'])
        source = next(row for row in generated['rows'] if row['fields'] == ['T6'])
        self.assertEqual(moved['cacheOrigin']['originalFields'], ['T6'])
        for key in ('text', 'tokenIds', 'rawSampledTokenIds', 'stopTokenId', 'messages', 'maxTokens'):
            self.assertEqual(moved[key], source[key])

    def test_bad_lamp_body_and_non_eos_rows_are_preserved_withheld_without_repair(self):
        cards = self.plan['groups'][1]['cards']; generated = self.generated(cards, unsafe_lamp=True)
        bad = next(row for row in generated['rows'] if row['fields'] == ['B1'])
        bad['finishReason'] = 'length'; bad['stopTokenId'] = None
        self.cache.record_generation(cards, generated)
        result = self.cache.resolve(cards)
        lamp = next(card for section in result['payload']['sections'] for card in section['cards'] if card['productId'] == 'lamp')
        self.assertIsNone(lamp['bodyCopy'])
        self.assertIsNone(result['payload']['sections'][0]['cards'][0]['badge'])
        self.assertGreaterEqual(result['validation']['withheldFields'], 2)
        self.assertTrue(any('12 hours of battery' in row['text'] for row in result['rawRows']))
        self.assertEqual(len(list((self.cache.root / 'attempts').glob('*.json'))), 1)

    def test_misses_require_real_native_callback_and_preserve_full_six_card_roster(self):
        cards = self.plan['groups'][0]['cards']
        with self.assertRaises(NativeGenerationRequired) as caught:
            self.cache.resolve(cards)
        self.assertEqual(len(caught.exception.misses), 8)
        observed = []
        def native(selected):
            observed.append(deepcopy(selected)); return self.generated(selected, fresh=True)
        result = self.cache.resolve(cards, native_generate=native)
        self.assertEqual(observed, [cards])
        self.assertEqual(result['cache']['copySource'], 'fresh-native-on-miss')
        self.assertTrue(result['cache']['nativeMissGeneration'])
        self.assertEqual(result['cache']['nativeColdGenerationMsThisRequest'], 8000)
        self.assertEqual(len(result['rawRows']), 20)
        self.assertEqual(self.cache.resolve(cards)['cache']['copySource'], 'generated-earlier')

    def test_old_archive_cannot_be_presented_as_fresh_miss_generation(self):
        cards = self.plan['groups'][0]['cards']
        with self.assertRaisesRegex(ValueError, 'earlier archive'):
            self.cache.resolve(cards, native_generate=lambda selected: self.generated(selected))

    def test_first_attempt_policy_keeps_failed_native_fields_and_archives_later_attempts(self):
        cards = self.plan['groups'][1]['cards']
        self.cache.record_generation(cards, self.generated(cards, unsafe_lamp=True))
        self.now += timedelta(minutes=1)
        second = self.cache.record_generation(cards, self.generated(cards, fresh=True))
        self.assertEqual(second['insertedKeys'], [])
        lamp = next(card for section in self.cache.resolve(cards)['payload']['sections'] for card in section['cards'] if card['productId'] == 'lamp')
        self.assertIsNone(lamp['bodyCopy'])
        self.assertEqual(len(list((self.cache.root / 'attempts').glob('*.json'))), 2)

    def test_changed_fact_variant_checkpoint_prompt_or_raw_archive_cannot_reuse_entry(self):
        cards = self.plan['groups'][0]['cards']; receipt = self.cache.record_generation(cards, self.generated(cards))
        changed = deepcopy(cards); changed[0]['facts']['itemPrice'] += 1
        with self.assertRaisesRegex(ValueError, 'bound public catalog'):
            self.cache.resolve(changed)
        other = producer_spec('d' * 64, 'pinned-test-revision', 'b' * 64)
        with self.assertRaisesRegex(ValueError, 'different public facts'):
            GeneratedCopyCache(self.cache.root, self.snapshot, other)
        Path(receipt['attemptFile']).write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'archive changed'):
            self.cache.resolve(cards)

    def test_native_provenance_and_complete_prompt_roster_are_required(self):
        cards = self.plan['groups'][0]['cards']; generated = self.generated(cards)
        generated['provenance']['fallback'] = True
        with self.assertRaisesRegex(ValueError, 'native producer'):
            self.cache.record_generation(cards, generated)
        generated = self.generated(cards); generated['rows'].pop()
        with self.assertRaisesRegex(ValueError, '20-field roster'):
            self.cache.record_generation(cards, generated)
        generated = self.generated(cards); generated['rows'][0]['messages'][0]['content'] += ' target leak'
        with self.assertRaisesRegex(ValueError, 'serving prompt'):
            self.cache.record_generation(cards, generated)

    def test_rehashed_entry_must_match_original_raw_attempt_cost_and_identity(self):
        cards = self.plan['groups'][0]['cards']
        self.cache.record_generation(cards, self.generated(cards))
        _, _, descriptors = self.cache.descriptors(cards)
        descriptor = next(d for d in descriptors if d['kind'] == 'card')
        path = self.cache.root / 'entries' / (descriptor['key'] + '.json')
        original = json.loads(path.read_text())
        mutations = {
            'cost': lambda entry: entry.__setitem__('coldGenerationMs', 1),
            'timestamp': lambda entry: entry.__setitem__('completedAt', self.now.isoformat()),
            'producer': lambda entry: entry['provenance'].__setitem__('nativeWidth', 20),
            'raw-text': lambda entry: entry['rawRows'][0].__setitem__('text', 'Forged text'),
            'raw-token': lambda entry: entry['rawRows'][0]['tokenIds'].append(6),
            'original-slot': lambda entry: entry['rawRows'][0].__setitem__('rowId', 'title-6'),
            'validation': lambda entry: entry.__setitem__('validationAtGeneration', []),
            'attempt-path': lambda entry: entry.__setitem__('attemptFile', '../manifest.json'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                envelope = deepcopy(original); mutate(envelope['entry'])
                envelope['sha256'] = hashed(envelope['entry'])
                path.write_text(json.dumps(envelope))
                with self.assertRaises(ValueError):
                    self.cache.resolve(cards)
        path.write_text(json.dumps(original))
        self.assertEqual(self.cache.resolve(cards)['cache']['initialHits'], 8)

    def test_learned_rank_composition_respects_public_guards_and_does_not_pad_shortfalls(self):
        cards = self.plan['groups'][0]['cards']; self.cache.record_generation(cards, self.generated(cards))
        observed = []
        def rank_request(request, *, copy_adapter):
            observed.append(request); return {'scores16': [float(16 - i) for i in range(16)]}
        session = types.SimpleNamespace(snapshot=self.snapshot, products=self.snapshot['inputs']['products'], rank_request=rank_request)
        result = personalize_with_cache(session, self.cache, {'query': 'a mug'})
        self.assertEqual(result['shortfall'], 0); self.assertEqual(result['validation']['requiredFields'], 20)
        self.assertEqual(observed[0]['preferences'], ['a mug'])
        self.assertNotIn('hiddenProfile', observed[0])
        empty = personalize_with_cache(session, self.cache, {'query': 'something in stock'})
        self.assertEqual(empty['payload']['sections'], []); self.assertEqual(empty['shortfall'], 6)


if __name__ == '__main__':
    unittest.main()
