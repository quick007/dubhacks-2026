"""Independent authored evaluation only. This file contains private gold; never encode/import at serving."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'persona_qwen_independent_eval_v1'
RECIPES = [
    ('mug', 'notebook', 'a set of cacao tasting tins', 'Pantry supplies',
     'Makes hot cacao at a small desk after dinner and currently has no dedicated drinking cup there.',
     'Also wants somewhere to write short tasting notes, with loose scraps currently used for that.'),
    ('pack', 'jacket', 'a slim laptop sleeve', 'Computer accessories',
     'Now takes a computer and a few books between two workspaces each day and needs a hands-free daily carry option.',
     'Walks from the station in changeable weather, so an outer rain layer is a useful second need.'),
    ('notebook', 'organizer', 'a packet of herb seeds', 'Gardening supplies',
     'Records plant growth and watering experiments by hand and needs a small place to keep those pages together.',
     'Pencils and little charging adapters are currently loose in the same bag and could use a separate case.'),
    ('jacket', 'sneakers', 'a neighborhood bus pass', 'Public transport',
     'Waits outside for a bus and walks the last part of the journey, often through drizzle without a hooded outer layer.',
     'Also needs ordinary casual footwear for that daily walk; the old pair is becoming worn.'),
    ('bottle', 'towel', 'a mixed fitness class pass', 'Exercise services',
     'Takes exercise classes and wants a reusable container for drinking water between activities.',
     'Sometimes finishes with a pool shower and also needs something for drying off afterwards.'),
    ('tote', 'lunchbox', 'a neighborhood bakery subscription', 'Food subscriptions',
     'Collects bread and a few loose vegetables on foot and needs an easy hand-carried bag for the purchases.',
     'Often stays out over lunch and would also like a container for a prepared meal.'),
    ('blanket', 'sweatshirt', 'a second-hand chaise lounge', 'Living room furniture',
     'Has arranged a reading seat by a drafty window and wants a shared fabric cover to put across the knees.',
     'Also spends cool evenings moving around the flat, when a wearable cotton layer would be useful.'),
    ('lamp', 'blanket', 'a stack of used novels', 'Books',
     'Reads printed books in a dim corner after sunset and first needs a small light on the side table.',
     'A fabric cover across the lap is another useful addition for staying in that corner on cooler evenings.'),
    ('lunchbox', 'mug', 'a small rice cooker', 'Kitchen equipment',
     'Has begun cooking an extra portion at night and needs a container to take that meal to the next day workspace.',
     'There is also no dedicated cup at that desk for a warm drink during the afternoon.'),
    ('headphones', 'lamp', 'an audio language course', 'Learning subscriptions',
     'Studies spoken lessons in a shared flat and needs private listening without playing audio through speakers.',
     'Also reads printed exercises late in the evening and could use a small table light.'),
    ('sneakers', 'bottle', 'a city walking route booklet', 'Local activity guides',
     'Has made long walks between errands a daily habit and first needs casual footwear for those trips.',
     'Would also like a reusable container to carry drinking water along the route.'),
    ('sweatshirt', 'sunglasses', 'a shared art studio membership', 'Workspace services',
     'Starts the day in a cool indoor studio and needs an easy wearable layer over a shirt.',
     'Later walks through bright open streets, where eyewear for the sun would be another useful addition.'),
    ('duffel', 'headphones', 'an overnight rail reservation', 'Travel bookings',
     'Needs to carry folded clothes and small essentials for one overnight visit, rather than a daily computer commute.',
     'Would also like private listening for spoken programs on the train.'),
    ('sunglasses', 'tote', 'tickets for afternoon outdoor matches', 'Sports outings',
     'Spends sunny afternoons watching outdoor matches and first wants eyewear for the bright conditions.',
     'Usually stops at small food stalls afterwards and would also use a simple hand-carried shopping bag.'),
    ('organizer', 'pack', 'a portable charging hub', 'Computer accessories',
     'Has loose plugs, leads and adapters tangled together and first needs a little zip case to keep them separate.',
     'Also carries a computer and papers between buildings each day and could use a hands-free daily bag.'),
    ('towel', 'duffel', 'a pool practice pass', 'Swimming services',
     'Has started regular pool sessions and first needs something for drying off after leaving the water.',
     'Also needs a larger hand-carried bag for a change of clothes and the rest of the pool kit.'),
]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')


def main():
    OUT.mkdir(exist_ok=False)
    binding_path = HERE / 'synthetic_history_v1/catalog-binding.json'
    binding = json.loads(binding_path.read_text()); products = {p['id']: p for p in binding['products']}
    recipe = {'version': 'independent-persona-qwen-eval.v1', 'authoredBeforeNativeFit': True,
        'fixedCandidateRecipe': {'train': '160 original TRAIN plus160 generic TRAIN-only ownership contrasts; same80TRAIN groups',
            'publicContextRepresentation': 'fresh frozenQwen4096; query/budget/hard requirements/public background/colors/observed item text+relative ages; no IDs/gold',
            'catalogRepresentation': 'original verified frozenQwen4096 known16 catalog',
            'rankWidth': 64, 'colorWidth': 32, 'colorCandidateHashDimensions': 256,
            'learnedOwnership': 'one shared candidate-purchased indicator coefficient initialized0; no fixed owned mask',
            'steps': 250, 'optimizer': 'AdamW', 'learningRate': .003, 'weightDecay': .0001, 'seed': 20261018,
            'loss': 'graded product crossentropy plus0.5 graded variant crossentropy', 'checkpoint': 'final fixed step only'},
        'evaluationDesign': '16 independent authored scenarios, two matched purchase-history variants each; primary and secondary catalog IDs form a permutation so each product is highest-grade twice',
        'primaryMetric': 'highest-grade top1 across32; report both-sides-correct per16pairs, raw learned vs hardguarded ranks separately',
        'proposedBeforeOutcomesGate': {'highestGradeTop1AtLeast': 24, 'bothPairSidesHighestGradeAtLeast': 12,
            'newlyOwnedTop1AtMost': 0, 'hardConstraintOrFactViolationsAtMost': 0},
        'scope': 'Authored known16 catalog and TRAIN-seen ownership-contrast mechanism, new public wording/combinations. No unseen human/causal task or real behavior claim.',
        'goldIsolation': 'Do not parse labels.private.json or this authoring source before checkpoint and all32 predictions frozen. Public encoding uses public.json request fields only.',
        'catalogBindingSha256': sha(binding_path), 'authoringSourceSha256': sha(__file__)}
    write('RECIPE_FROZEN.json', recipe)
    profiles = []; labels = []; pairs = []
    ids = list(products)
    for index, (primary, secondary, anchor, category, background, other_need) in enumerate(RECIPES):
        p, q = products[primary], products[secondary]
        others = [pid for pid in ids if pid not in (primary, secondary)]
        observed = [others[(index + offset) % len(others)] for offset in (0, 3, 7)]
        budget = 150 if max(p['price'], q['price']) > 100 else (90 if max(p['price'], q['price']) <= 90 and index % 2 else 100)
        shared_colors = sorted(set(v['colorName'] for v in p['variants']) & set(v['colorName'] for v in q['variants']))
        hard = [{'attribute': 'color', 'value': shared_colors[0]}] if shared_colors and index % 3 == 0 else []
        color = hard[0]['value'] if hard else p['variants'][index % len(p['variants'])]['colorName']
        request = {'query': 'Choose a useful next addition from the public context and purchase history.',
            'itemBudget': budget, 'hardRequirements': hard,
            'persona': {'background': [background, other_need, 'Prefers useful additions and does not want another copy of an item already purchased.'], 'colorPreferences': [color]},
            'publicHistory': [{'eventId': f'ind-eval-{index:02d}-event-0', 'eventType': 'purchase',
                'itemName': anchor, 'category': category, 'occurredAt': '2026-10-10T10:00:00.000Z', 'source': 'demo-seed', 'external': True}]}
        for offset, pid in enumerate(observed):
            item = products[pid]; variant = item['variants'][(index + offset) % len(item['variants'])]
            request['publicHistory'].append({'eventId': f'ind-eval-{index:02d}-event-{offset+1}',
                'eventType': ('purchase', 'view', 'cart')[offset], 'itemName': item['name'], 'category': item['category'],
                'selectedColor': variant['colorName'], 'occurredAt': f'2026-10-{12+offset:02d}T10:00:00.000Z',
                'source': 'demo-seed', 'productId': pid, 'variantId': variant['id']})
        pair_ids = []
        for side in ('a', 'b'):
            case_id = f'ind-qwen-eval-{index:02d}-{side}'; pair_ids.append(case_id); r = deepcopy(request)
            if side == 'b':
                selected = next((v for v in p['variants'] if v['colorName'] == color), p['variants'][0])
                r['publicHistory'][0] = {**{k:v for k,v in r['publicHistory'][0].items() if k != 'external'},
                    'itemName': p['name'], 'category': p['category'], 'selectedColor': selected['colorName'],
                    'productId': primary, 'variantId': selected['id']}
            profiles.append({'profileId': case_id, 'userGroupId': f'ind-qwen-eval-group-{index:02d}',
                'sessionId': f'ind-qwen-eval-session-{index:02d}-{side}', 'request': r})
            grades = {pid: 0 for pid in ids}; grades[primary] = 3 if side == 'a' else 0; grades[secondary] = 2 if side == 'a' else 3
            relevant = [products[pid] for pid, g in grades.items() if g > 0]
            acceptable = [v['id'] for item in relevant for v in item['variants'] if not hard or v['colorName'] == hard[0]['value']]
            preferred = [next((v['id'] for v in item['variants'] if v['colorName'] == color),
                              next(v['id'] for v in item['variants'] if not hard or v['colorName'] == hard[0]['value'])) for item in relevant]
            labels.append({'profileId': case_id, 'pairId': f'ind-qwen-eval-pair-{index:02d}',
                'relevanceGrades': grades, 'primaryProductId': primary if side == 'a' else secondary,
                'alreadyOwnedProductIds': [observed[0]] + ([primary] if side == 'b' else []),
                'preferredVariantIds': preferred, 'acceptableVariantIds': acceptable,
                'rationale': ('The first unmet public need has grade3; the second need is a useful complement.' if side == 'a' else
                    'The newly observed purchase satisfies the first need; the previously secondary public need becomes the useful next addition.'),
                'judgmentSource': 'independent-agent-authored-synthetic-not-observed-choice'})
        pairs.append({'pairId': f'ind-qwen-eval-pair-{index:02d}', 'profileIds': pair_ids,
            'publicDifference': 'only first observed purchase item identity changes; eventID/time/source/type and all other context identical'})
    write('public.json', {'schemaVersion': 'independent-persona-qwen-eval.v1',
        'provenance': {'kind': 'independent-agent-authored-synthetic', 'browserObserved': False, 'realCustomerData': False},
        'profiles': profiles, 'pairs': pairs})
    write('labels.private.json', {'schemaVersion': 'independent-persona-qwen-eval.v1',
        'scope': 'evaluation-only; parse only after checkpoint/all32predictions frozen; never serving or encoding input', 'labels': labels})
    write('MANIFEST.json', {'schemaVersion': 'independent-persona-qwen-eval.v1', 'profiles': 32, 'groups': 16,
        'pairs': 16, 'catalogProducts': 16, 'highestGradeCountPerProduct': 2,
        'trainAllowed': False, 'modelSelectionAllowed': False, 'predictionFreezeRequiredBeforePrivateLabels': True,
        'files': {name: sha(OUT / name) for name in ['RECIPE_FROZEN.json', 'public.json', 'labels.private.json']},
        'limitations': ['Synthetic relevance judgments, not human choices.', 'Known catalog and TRAIN-seen ownership-contrast mechanism.',
            'Two matched sessions per group are correlated; report16pair results.', 'Hardguards may reduce candidate sets; report raw and guarded quality separately.']})
    print(json.dumps({'publicPath': str(OUT / 'public.json'), 'publicSha256': sha(OUT / 'public.json'),
        'manifestSha256': sha(OUT / 'MANIFEST.json'), 'profiles': len(profiles)}))


if __name__ == '__main__': main()
