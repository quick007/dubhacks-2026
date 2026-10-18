"""CPU-only branch binding, token work and fresh-root preparation controls."""
import copy
import json
import unittest

from .branch_local_copy import build_layout, pending_plan, token_inventory, prepare_root


class StubTokenizer:
    def encode(self, text):
        return [100 + sum(map(ord, text[i:i + 4])) for i in range(0, len(text), 4)]

    def apply_chat_template(self, messages, *, tokenize, return_dict, add_generation_prompt, enable_thinking):
        assert tokenize is True and return_dict is False and enable_thinking is False
        return [10] + [v for m in messages for v in [11, *self.encode(m['content']), 12]] + ([13, 14] if add_generation_prompt else [])


class FakeCache:
    def __init__(self):
        self.state = []


class FakeMx:
    def __init__(self):
        self.evaluated = []

    def array(self, values):
        return values

    def eval(self, values):
        self.evaluated.append(copy.deepcopy(values))

    def synchronize(self):
        pass


class FakeModel:
    def __init__(self):
        self.calls = []

    def model(self, inputs, *, cache):
        self.calls.append(copy.deepcopy(inputs[0]));cache[0].state.extend(inputs[0])
        return object()  # No LM projection or hidden evaluation API exists.


class FakeBackbone:
    def __init__(self):
        self.tokenizer = StubTokenizer();self.mx = FakeMx();self.model = FakeModel();self.checks = 0

    def check_memory(self):
        self.checks += 1


class BranchLocalCopyTests(unittest.TestCase):
    def fixture(self):
        from .page_plan import make_page_plan
        catalog = {}
        for i in range(6):
            identity = f'Aster{i} {18 + i}L Daypack'
            catalog[f'opaque-product-{i}'] = {'id': f'opaque-product-{i}', 'title': identity,
                'category': 'backpacks', 'description': 'Listed fabric and zip closure.',
                'price': 40 + i, 'shippingCost': 5, 'inStock': True, 'tags': ['compact', 'durable'],
                'presentationVariants': [{'id': f'opaque-image-{i}', 'title': identity,
                    'imageDescription': identity + ' shown in a detail view.', 'facts': [f'{18 + i}L', 'compact'],
                    'presentationStyle': 'detail-view', 'compatibleSurfaces': ['compact-card']},
                    {'id': f'opaque-alt-{i}', 'title': identity + ' — compact',
                     'imageDescription': identity + ' shown being carried.', 'facts': ['compact'],
                     'presentationStyle': 'in-use-view', 'compatibleSurfaces': ['compact-card']}]}
        request = {'recordId': 'locked-fixture', 'asOf': '2026-09-07T12:00:00Z', 'history': [],
            'candidateProductIds': list(catalog), 'context': {'task': 'Choose travel backpacks.', 'statedBudget': 200,
            'statedRequirements': [], 'statedPreferences': ['compact'], 'presentationSurface': 'compact-card',
            'presentationPreferences': ['detail-view']},
            'presentationCompatibility': {p: {v['id']: True for v in c['presentationVariants']} for p, c in catalog.items()}}
        decision = {'rankedProducts': [{'productId': p} for p in catalog],
            'presentationActions': {p: {'productId': p, 'variantId': c['presentationVariants'][0]['id']} for p, c in catalog.items()}}
        return make_page_plan(request, catalog, decision, 6), request, catalog

    def test_bound_facts_preserve_grid_and_duplicate_card_evidence_explicitly(self):
        plan, request, catalog = self.fixture();before = copy.deepcopy((plan, request, catalog))
        root, rows, evidence = build_layout(plan, request, catalog)
        self.assertEqual((plan, request, catalog), before)
        self.assertEqual(len(rows), 14);self.assertEqual(sum(r['maxTokens'] for r in rows), 224)
        self.assertFalse(evidence['cardFactCacheReuseImplemented'])
        heading = json.loads(rows[0]['messages'][0]['content'].split(': ', 1)[1])
        self.assertEqual([c['index'] for c in heading['cards']], [1, 2, 3])
        self.assertEqual(heading['sections'][0]['universalTags'], ['compact', 'durable'])
        title = next(r for r in rows if r['rowId'] == 'title-1');badge = next(r for r in rows if r['rowId'] == 'badge-1')
        self.assertEqual(title['messages'][0], badge['messages'][0])
        self.assertEqual(len(json.loads(title['messages'][0]['content'].split(': ', 1)[1])['cards']), 1)
        text = json.dumps([root, *[r['messages'] for r in rows]])
        self.assertNotIn('opaque-product-', text);self.assertNotIn('opaque-image-', text)
        card = evidence['cards'][0];product = next(iter(catalog.values()))
        self.assertEqual(card['identity'], product['title'])
        self.assertEqual(card['productFacts']['description'], product['description'])
        self.assertEqual([card['identity'] + s for s in card['approvedTitleSuffixes']], [v['title'] for v in product['presentationVariants']])
        self.assertEqual(card['selectedImage']['imageDescriptionTemplate'].replace('{identity}', card['identity']), product['presentationVariants'][0]['imageDescription'])

    def test_old_copy_never_enters_new_context_and_empty_intersection_is_exact(self):
        plan, request, catalog = self.fixture()
        plan['sections'][0]['heading'] = {'source': 'generated', 'text': 'STALE RESPONSE'}
        plan['sections'][0]['products'][0]['title'] = {'source': 'generated', 'text': 'SECRET GENERATION'}
        catalog['opaque-product-0']['tags'] = ['compact']
        catalog['opaque-product-1']['tags'] = ['durable']
        # Keep the recorded support valid while testing actual universal facts.
        card = plan['sections'][0]['products'][1];card['approvedFactIds'] = [];card['badgeDecision'] = 'NONE'
        root, rows, evidence = build_layout(plan, request, catalog)
        self.assertEqual(evidence['sections'][0]['universalTags'], [])
        self.assertNotIn('STALE RESPONSE', json.dumps([root, rows]));self.assertNotIn('SECRET GENERATION', json.dumps([root, rows]))
        self.assertIsNone(pending_plan(plan)['sections'][0]['products'][0]['title']['text'])

    def test_token_inventory_counts_every_fact_repetition_and_refuses_boundaries(self):
        plan, request, catalog = self.fixture();root, rows, _ = build_layout(plan, request, catalog)
        tok = StubTokenizer();result = token_inventory(tok, root, rows)
        self.assertEqual(len(result['rows']), 14)
        self.assertEqual(result['totalRootAndAllRowSuffixTokens'], result['rootTokens'] + sum(r['suffixTokens'] for r in result['rows']))
        self.assertFalse(result['cardFactCacheReuseImplemented'])
        class Broken(StubTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                ids = super().apply_chat_template(messages, **kwargs)
                if len(messages) > 1:ids[0] = 999
                return ids
        with self.assertRaisesRegex(ValueError, 'exact root'):token_inventory(Broken(), root, rows)
        with self.assertRaisesRegex(ValueError, 'context/append'):token_inventory(tok, root, [{**rows[0], 'maxTokens': 10000}])
        with self.assertRaisesRegex(ValueError, 'root exceeds'):token_inventory(tok, {'role': 'user', 'content': 'x' * 10000}, rows)

    def test_root_preparation_uses_fresh_empty_cache_and_only_chunked_hidden_model(self):
        plan, request, catalog = self.fixture();root, _, _ = build_layout(plan, request, catalog)
        bb = FakeBackbone();created = []
        def factory(model):
            self.assertIs(model, bb.model);value = [FakeCache()];created.append(value);return value
        first, receipt = prepare_root(bb, root, cache_factory=factory)
        second, _ = prepare_root(bb, root, cache_factory=factory)
        self.assertIsNot(first['cache'], second['cache']);self.assertEqual(first['cache'][0].state, first['tokens'])
        self.assertEqual(first['tokens'], second['tokens']);self.assertFalse(first['consumed'])
        self.assertTrue(all(1 <= len(c) <= 256 for c in bb.model.calls));self.assertEqual(bb.checks, len(bb.model.calls))
        self.assertTrue(receipt['freshEmptyCache']);self.assertFalse(receipt['rankingPrefixInCopyContext'])
        self.assertFalse(receipt['unusedLogitsOrHiddenForced'])

    def test_no_twelve_or_twentyfour_expansion(self):
        plan, request, catalog = self.fixture()
        for count in (12, 24):
            with self.assertRaisesRegex(ValueError, 'six complete'):build_layout({**plan, 'requestedCardCount': count}, request, catalog)


if __name__ == '__main__':
    unittest.main()
