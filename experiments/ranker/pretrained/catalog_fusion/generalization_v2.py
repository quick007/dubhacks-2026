"""One reviewed full/focused input comparison; CPU freeze/prepare, root-only run.

192 native B1 forwards: (33 catalog products +63 requests) x2 representations.
Product-only seven-atom factual distillation, no image/copy/UI model or dummy
presentation variants. Private ranking targets are evaluator-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
from torch import nn

from ..boundary import clean_product, guard_reasons
from .inputs import canonical, digest, request_text, tensor_digest
from .paired_data import ATOMS
from .preflight import DEFAULT_MODEL, REVISION, file_digest
from .runner import execution_receipt, source_hashes
from .runtime import FeatureStore, NativeEncoder, assert_model_files, verify_model, write_json
from .semantic_factors import atom_metrics

VERSION = "public-atom-generalization-v2"
MODES, SPLITS = ("full", "focused"), ("train", "validation", "test")
SCHEDULE = {"seed": 728, "steps": 80, "learningRate": .03, "weightDecay": .001,
            "cpuThreads": 1, "epsilon": 1e-8, "maxSecondsPerArm": 60,
            "objective": "mean TRAIN query BCE + mean unique TRAIN product BCE, equal role weights"}
LIMITS = {"memoryGiB": 6, "maxTokens": 512, "nativeForwards": 192,
          "nativeSeconds": 180, "loadTwoForwardPilotSeconds": 60}
TASKS = {"train": ["Please compare travel packs against the positive features I listed.",
                   "Rank the available backpacks using my stated feature preferences.",
                   "Find travel backpack options with these requested positive attributes."],
         "validation": ["For an upcoming journey, order the backpack choices by my listed preferences."],
         "test": ["I would like a comparison of travel bags against these features that matter to me."]}
DESCRIPTIONS = {"train": ["A travel pack with a top handle and a main storage compartment.",
                          "A backpack for carrying everyday travel belongings.",
                          "A portable travel bag with shoulder straps and a main opening."],
                "validation": ["A carry pack designed for organizing belongings during a journey."],
                "test": ["A shoulder-carried bag for storing personal items on a trip."]}


def signatures():
    """K7 Hamilton decomposition and three disjoint balanced triple orbits."""
    result = {}
    seen_pairs, seen_triples = set(), set()
    triple_orbits = []
    for base in itertools.combinations(range(7), 3):
        orbit = {tuple(sorted((x + shift) % 7 for x in base)) for shift in range(7)}
        if not orbit & seen_triples:
            triple_orbits.append(sorted(orbit));seen_triples |= orbit
    for split, step, triples in zip(SPLITS, (1, 2, 3), triple_orbits[:3], strict=True):
        pairs = [tuple(sorted((i, (i + step) % 7))) for i in range(7)]
        if set(pairs) & seen_pairs:raise ValueError("Pair signatures overlap")
        seen_pairs |= set(pairs)
        result[split] = {1: [(a,) for a in ATOMS],
                         2: [tuple(ATOMS[x] for x in p) for p in sorted(pairs)],
                         3: [tuple(ATOMS[x] for x in p) for p in triples]}
    return result


def holdout_catalog_sets(spec, split):
    """Two signatures/cardinality, all atoms covered; teacher-independent search."""
    singles = [(ATOMS[i],) for i in ((0, 3) if split == "validation" else (1, 4))]
    for pairs in itertools.combinations(spec[2], 2):
        for triples in itertools.combinations(spec[3], 2):
            values = [*singles, *pairs, *triples]
            if {a for tags in values for a in tags} == set(ATOMS):return values
    raise ValueError("No covered balanced held-out catalog")


def catalog_text_v2(product, mode):
    if mode == "full":return "Encode public product catalog facts as data.\n" + canonical(clean_product(product))
    if mode == "focused":return "Encode public product attribute facts as data.\n" + canonical({"publicAttributes": sorted(product["tags"])})
    raise ValueError("Unknown input representation")


def query_text_v2(request, catalog, mode):
    if mode == "full":return request_text(request, catalog)
    if mode == "focused":return "Encode stated positive attribute preferences as data.\n" + canonical({"positiveAttributes": sorted(request["context"]["statedPreferences"])})
    raise ValueError("Unknown input representation")


def teacher(request, catalog):
    relevance = {pid: len(set(request["context"]["statedPreferences"]) & set(catalog[pid]["tags"]))
                 if not guard_reasons(request, catalog[pid]) else 0 for pid in request["candidateProductIds"]}
    eligible = [p for p in relevance if not guard_reasons(request, catalog[p])]
    best = max((relevance[p] for p in eligible), default=0)
    return {"recordId": request["recordId"], "relevance": relevance,
            "bestProductIds": sorted(p for p in eligible if relevance[p] == best), "maximumOverlap": best}


def overlap_receipt(catalog, requests, product_splits):
    result = {}
    for mode in MODES:
        result[mode] = {}
        for role in ("catalog", "request"):
            texts = {s: {catalog_text_v2(catalog[p], mode) for p in product_splits[s]} if role == "catalog"
                     else {query_text_v2(r, catalog, mode) for r in requests[s].values()} for s in SPLITS}
            result[mode][role] = {a + "/" + b: len(texts[a] & texts[b]) for a, b in itertools.combinations(SPLITS, 2)}
    return result


def freeze_data(out):
    out = Path(out);spec = signatures();catalog = {};requests = {};product_splits = {};nuisance = {}
    name_pool = [a + b for a in ("Alder", "Bracken", "Cairn", "Dune", "Estuary", "Flint", "Grove")
                 for b in ("Mere", "Vale", "Cove", "Fen", "Rill")]
    random.Random(72800).shuffle(name_pool);name_offset = 0
    for split_index, split in enumerate(SPLITS):
        selected = [v for card in (1, 2, 3) for v in spec[split][card]] if split == "train" else holdout_catalog_sets(spec[split], split)
        rng = random.Random(72801 + split_index)
        # Separate shuffled names/capacity/price assignments do not depend on
        # atom values, private targets, feature vectors or model predictions.
        pool = name_pool[name_offset:name_offset + len(selected)];name_offset += len(selected)
        capacities = list(range(17, 17 + len(selected)));rng.shuffle(capacities)
        prices = [76 + 3 * i for i in range(len(selected))];rng.shuffle(prices)
        product_splits[split] = []
        for i, tags in enumerate(selected):
            pid = "gv2-product-" + digest(canonical([VERSION, split, i]))[:18]
            title = f"{pool[i]} {capacities[i]}.7L Backpack"
            template = i % len(DESCRIPTIONS[split])
            catalog[pid] = {"id": pid, "title": title, "category": "backpacks", "description": DESCRIPTIONS[split][template],
                            "price": prices[i], "shippingCost": 5 + i % 4, "inStock": True, "tags": list(tags)}
            product_splits[split].append(pid)
            nuisance[pid] = {"catalogTemplateFamily": f"{split}-catalog-{template}", "capacity": capacities[i] + .7}
        requests[split] = {}
        for card in (1, 2, 3):
            for i, preferences in enumerate(spec[split][card]):
                rid = "gv2-request-" + digest(canonical([VERSION, split, card, i]))[:18]
                order = product_splits[split][:];random.Random(72802 + card * 100 + i + split_index * 1000).shuffle(order)
                q = {"recordId": rid, "asOf": "2026-10-18T02:00:00Z", "candidateProductIds": order, "history": [],
                     "context": {"task": TASKS[split][i % len(TASKS[split])], "statedBudget": 160,
                                 "statedRequirements": [], "statedPreferences": list(preferences),
                                 "presentationSurface": None, "presentationPreferences": []}}
                requests[split][rid] = q
                nuisance[rid] = {"queryTemplateFamily": f"{split}-query-{i % len(TASKS[split])}"}
    out.mkdir(parents=True, exist_ok=False);(out / "public").mkdir();(out / "private").mkdir()
    write_json(out / "catalog.json", {"synthetic": True, "products": list(catalog.values())})
    pairs = [];coverage = {};permutation = {}
    for split in SPLITS:
        labels = {rid: teacher(q, catalog) for rid, q in requests[split].items()}
        with (out / f"public/{split}.jsonl").open("x") as h:
            for q in requests[split].values():h.write(canonical(q) + "\n")
        with (out / f"private/{split}.jsonl").open("x") as h:
            for label in labels.values():h.write(canonical(label) + "\n")
        for a, b in itertools.combinations(requests[split], 2):
            if set(labels[a]["bestProductIds"]).isdisjoint(labels[b]["bestProductIds"]):
                pairs.append({"split": split, "recordIds": [a, b], "bestSetsDisjoint": True})
        permutation[split] = {}
        for card in (1, 2, 3):
            ids = [rid for rid, q in requests[split].items() if len(q["context"]["statedPreferences"]) == card]
            permutation[split].update({rid: ids[(i + 1) % len(ids)] for i, rid in enumerate(ids)})
        coverage[split] = {"queryRows": 21, "catalogRows": len(product_splits[split]),
            "queryCardinalityCounts": dict(Counter(len(q["context"]["statedPreferences"]) for q in requests[split].values())),
            "catalogCardinalityCounts": dict(Counter(len(catalog[p]["tags"]) for p in product_splits[split])),
            "queryAtomCountsByCardinality": {str(k): {a: sum(a in q["context"]["statedPreferences"] for q in requests[split].values() if len(q["context"]["statedPreferences"]) == k) for a in ATOMS} for k in (1, 2, 3)},
            "catalogAtomCounts": {a: sum(a in catalog[p]["tags"] for p in product_splits[split]) for a in ATOMS},
            "disjointBestSetPairs": sum(p["split"] == split for p in pairs)}
    write_json(out / "public/metadata.json", {"productSplits": product_splits, "nuisance": nuisance, "permutation": permutation})
    write_json(out / "private/pairs.json", pairs)
    files = ["catalog.json", "public/metadata.json", *[f"public/{s}.jsonl" for s in SPLITS], *[f"private/{s}.jsonl" for s in SPLITS], "private/pairs.json"]
    frozen = {"version": VERSION, "atUtc": datetime.now(timezone.utc).isoformat(), "generatorSha256": file_digest(__file__),
              "files": {f: file_digest(out / f) for f in files}, "coverage": coverage,
              "products": 33, "queries": 63, "representations": list(MODES), "nativeForwards": 192,
              "encoderTextOverlap": overlap_receipt(catalog, requests, product_splits),
              "catalogBalanceScope": "TRAIN atom frequencies exactly balanced; held-out catalogs have2 products per cardinality and cover all7 atoms, with unequal atom frequencies explicitly reported in coverage.",
              "signaturePolicy": "Singleton semantics necessarily shared; pair/triple signatures partitioned across all three splits. Catalog held-out sets are subsets of held-out query signatures.",
              "templatePolicy": "Train3/query3 catalog wording families; validation/test distinct held-out families. Focused text intentionally removes all name/capacity/description/task nuisance.",
              "teacher": "tie-aware unweighted public positive-atom overlap among eligible products; no ID tie target",
              "fit": SCHEDULE, "limits": LIMITS,
              "limitations": ["Synthetic backpack seven-atom role-decoding experiment, distinct from teammate jacket domain.",
                              "Focused singleton inputs overlap across split identities; this tests nuisance invariance, not unseen singleton semantics.",
                              "Pair/triple signatures are held out; no all-attribute-combination or real-user/conversion claim.",
                              "Catalog6 held-out candidates; no12/24 grid, image, copy, grouping or complete-page evidence.",
                              "Known price/stock/requirements guards remain deterministic outside factor inputs; all base cases are eligible.",
                              "Full/focused comparison shares supervision/schedule but jointly repairs data coverage relative to old runs; old-run gains cannot isolate either correction."]}
    write_json(out / "FREEZE.json", frozen)
    return frozen


def read_public(dataset):
    dataset = Path(dataset).resolve();frozen = json.loads((dataset / "FREEZE.json").read_text())
    if frozen["version"] != VERSION or frozen["generatorSha256"] != file_digest(__file__):raise ValueError("v2 source/data freeze changed")
    for name, sha in frozen["files"].items():
        if name.startswith("private/"):continue
        if (dataset / name).resolve() != dataset / name or file_digest(dataset / name) != sha:raise ValueError("v2 public data changed")
    products = json.loads((dataset / "catalog.json").read_text())["products"];catalog = {p["id"]: p for p in products}
    requests = {s: {q["recordId"]: q for q in map(json.loads, (dataset / f"public/{s}.jsonl").read_text().splitlines())} for s in SPLITS}
    metadata = json.loads((dataset / "public/metadata.json").read_text())
    if len(catalog) != 33 or any(len(requests[s]) != 21 for s in SPLITS):raise ValueError("Fixed v2 membership changed")
    if overlap_receipt(catalog, requests, metadata["productSplits"]) != frozen["encoderTextOverlap"]:raise ValueError("v2 overlap accounting changed")
    return frozen, catalog, requests, metadata


def prepare(dataset, model, prior, tokenizer):
    frozen, catalog, requests, metadata = read_public(dataset);model = Path(model).resolve();prior = Path(prior)
    provenance = json.loads((prior / "model-verification.json").read_text());assert_model_files(model, provenance)
    old = json.loads((prior / "preflight.json").read_text())
    if model.name != REVISION or str(model) != old["modelPath"]:raise ValueError("Pinned local model required")
    for name, key in (("tokenizer.json", "tokenizerSha256"), ("config.json", "configSha256"), ("chat_template.jinja", "chatTemplateSha256")):
        if file_digest(model / name) != old[key]:raise ValueError("Pinned model/tokenizer bytes changed")
    rows = []
    for mode in MODES:
        for split in SPLITS:
            for pid in metadata["productSplits"][split]:rows.append({"role": "catalog", "member": [mode, pid], "split": split, "mode": mode, "id": pid, "text": catalog_text_v2(catalog[pid], mode)})
            for rid, q in requests[split].items():rows.append({"role": "request", "member": [mode, rid], "split": split, "mode": mode, "id": rid, "text": query_text_v2(q, catalog, mode)})
    for row in rows:
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
        if not 0 < len(ids) <= 512:raise ValueError("v2 input exceeds512; no truncation")
        row.update(textSha256=digest(row["text"]), tokenIds=ids, tokenSha256=digest(canonical(ids)), tokens=len(ids), batchShape=[1, len(ids)])
    if len(rows) != 192:raise ValueError("Fixed192 v2 native encodes required")
    packet = {"version": VERSION, "dataFreezeSha256": file_digest(Path(dataset) / "FREEZE.json"), "sourceHashes": source_hashes(),
              "modelPath": str(model), "modelRevision": REVISION, "quantization": old["quantization"],
              "tokenizerSha256": old["tokenizerSha256"], "configSha256": old["configSha256"], "chatTemplateSha256": old["chatTemplateSha256"],
              "priorModelReceiptSha256": file_digest(prior / "model-verification.json"), "fit": SCHEDULE, "limits": LIMITS,
              "encoderTextOverlap": frozen["encoderTextOverlap"], "coverage": frozen["coverage"],
              "records": [{k: v for k, v in r.items() if k != "text"} for r in rows],
              "tokenRange": {m: [min(r["tokens"] for r in rows if r["mode"] == m), max(r["tokens"] for r in rows if r["mode"] == m)] for m in MODES}}
    return packet, frozen, catalog, requests, metadata, rows, provenance


class Factors(nn.Module):
    def __init__(self, means, frequency, arm="fusion"):
        super().__init__();self.arm = arm;generator = torch.Generator().manual_seed(728)
        for role in ("query", "catalog"):
            self.register_buffer(role + "_mean", means[role].clone())
            setattr(self, role + "_weight", nn.Parameter(torch.randn(7, 4096, generator=generator) * .05))
            setattr(self, role + "_bias", nn.Parameter(torch.zeros(7)))
        self.register_buffer("query_frequency", frequency.clone())
        if arm == "prior":self.query_weight.requires_grad_(False);self.query_bias.requires_grad_(False)
    def logits(self, query, catalog):
        outputs = []
        for role, values in (("query", query), ("catalog", catalog)):
            centered = values - getattr(self, role + "_mean")
            normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            outputs.append(normalized @ getattr(self, role + "_weight").T + getattr(self, role + "_bias"))
        return outputs
    def probabilities(self, query, catalog):
        q, c = self.logits(query, catalog)
        return (self.query_frequency.expand_as(q) if self.arm == "prior" else q.sigmoid()), c.sigmoid()


def save_head(model, path, metadata):
    with Path(path).open("xb") as h:torch.save({"version": VERSION, "arm": model.arm, "metadata": metadata,
        "metadataSha256": digest(canonical(metadata)), "state": model.state_dict(), "parameters": 57358}, h)
    return file_digest(path)


def load_head(path, sha, metadata):
    if file_digest(path) != sha:raise ValueError("v2 checkpoint bytes changed")
    z = torch.load(path, map_location="cpu", weights_only=True)
    if z.get("version") != VERSION or z["metadata"] != metadata or z["metadataSha256"] != digest(canonical(metadata)) or z["parameters"] != 57358:raise ValueError("Wrong v2 checkpoint contract")
    s = z["state"];means = {r: s[r + "_mean"] for r in ("query", "catalog")}
    if any(tensor_digest(means[r]) != metadata["meanSha256"][r] for r in means):raise ValueError("TRAIN role means changed")
    if not torch.equal(s["query_frequency"], torch.tensor(metadata["queryFrequency"])):raise ValueError("TRAIN query frequency changed")
    model = Factors(means, s["query_frequency"], z["arm"]);model.load_state_dict(s, strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError("Nonfinite v2 checkpoint")
    return model.eval()


def train_inputs(mode, catalog, requests, metadata, store, rows):
    # This function selects TRAIN before reading ANY numeric feature rows.
    qids = sorted(requests["train"]);pids = sorted(metadata["productSplits"]["train"])
    mapping = {(r["mode"], r["role"], r["id"]): r for r in rows}
    q = torch.stack([store.load(mapping[mode, "request", rid])[0][0] for rid in qids])
    c = torch.stack([store.load(mapping[mode, "catalog", pid])[0][0] for pid in pids])
    qt = torch.tensor([[float(a in requests["train"][rid]["context"]["statedPreferences"]) for a in ATOMS] for rid in qids])
    ct = torch.tensor([[float(a in catalog[pid]["tags"]) for a in ATOMS] for pid in pids])
    means = {"query": q.double().mean(0).float(), "catalog": c.double().mean(0).float()}
    contract = {"version": VERSION, "mode": mode, "vocabulary": list(ATOMS), "schedule": SCHEDULE,
                "queryMembers": qids, "catalogMembers": pids,
                "meanSha256": {r: tensor_digest(v) for r, v in means.items()},
                "rawTrainSha256": {"query": tensor_digest(q), "catalog": tensor_digest(c)},
                "publicTrainTargetSha256": {"query": tensor_digest(qt), "catalog": tensor_digest(ct)},
                "queryFrequency": qt.mean(0).tolist(), "usesPrivateRankingTargets": False}
    return q, c, qt, ct, means, contract


def fit(mode, arm, out, inputs):
    q, c, qt, ct, means, contract = inputs;model = Factors(means, qt.mean(0), arm)
    before = {n: p.detach().clone() for n, p in model.named_parameters()};start = time.perf_counter()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.03, weight_decay=.001)
    trace = []
    for step in range(80):
        if time.perf_counter() - start > 60:raise TimeoutError("v2 fixed fit budget exceeded")
        optimizer.zero_grad(set_to_none=True);ql, cl = model.logits(q, c)
        qb = nn.functional.binary_cross_entropy_with_logits(ql, qt);cb = nn.functional.binary_cross_entropy_with_logits(cl, ct)
        loss = cb if arm == "prior" else qb + cb
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite v2 BCE")
        loss.backward()
        grads = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        if len(grads) != sum(p.requires_grad for p in model.parameters()) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):raise FloatingPointError("Missing/nonfinite v2 gradient")
        if step == 0 and any(v <= 0 for v in grads.values()):raise FloatingPointError("No genuine v2 initial gradient")
        optimizer.step();trace.append({"step": step + 1, "lossBefore": loss.item(), "queryBce": qb.item(), "catalogBce": cb.item(), "gradientL2": grads})
    path = Path(out) / f"{mode}-{arm}.pt";sha = save_head(model, path, contract);restored = load_head(path, sha, contract)
    with torch.no_grad():
        if any(not torch.equal(a, b) for a, b in zip(model.probabilities(q, c), restored.probabilities(q, c), strict=True)):raise AssertionError("v2 exact training prediction reload failed")
        ql, cl = restored.logits(q, c)
        final_cb = nn.functional.binary_cross_entropy_with_logits(cl, ct)
        final_qb = nn.functional.binary_cross_entropy_with_logits(ql, qt)
    delta = {n: (p - before[n]).abs().sum().item() for n, p in model.named_parameters()}
    if any(delta[n] <= 0 for n, p in model.named_parameters() if p.requires_grad):raise AssertionError("v2 fit did not update all intended parameters")
    if arm == "prior" and (delta["query_weight"] or delta["query_bias"]):raise AssertionError("Prior acquired query conditioning")
    write_json(Path(out) / f"{mode}-{arm}-trace.json", trace)
    return restored, {"file": path.name, "sha256": sha, "metadata": contract, "steps": 80,
                      "lossBefore": trace[0]["lossBefore"], "lossAfter": (final_cb if arm == "prior" else final_qb + final_cb).item(),
                      "parameterAbsoluteChange": delta,
                      "trainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "seconds": time.perf_counter() - start, "exactTrainingPredictionReload": True}


def require_frozen(out, packet):
    z = json.loads((Path(out) / "HEADS_FROZEN.json").read_text())
    if z["version"] != VERSION or z["packetSha256"] != file_digest(Path(out) / "packet.json") or z["privateLabelsBeforeFreeze"] is not False or z["sourceHashes"] != source_hashes():raise ValueError("v2 immutable checkpoint freeze required")
    if z["dataFreezeSha256"] != packet["dataFreezeSha256"] or set(z["heads"]) != {m + "-" + a for m in MODES for a in ("fusion", "prior", "random")}:raise ValueError("v2 fixed model roster changed")
    raw_path = Path(out) / "RAW_FEATURES_FROZEN.json"
    if z["rawFeaturesFreezeSha256"] != file_digest(raw_path):raise ValueError("v2 raw feature freeze changed")
    raw = json.loads(raw_path.read_text())
    if raw["nativeForwards"] != 192 or raw["packetSha256"] != z["packetSha256"]:raise ValueError("v2 incomplete raw feature freeze")
    for name, sha in raw["files"].items():
        if file_digest(Path(out) / name) != sha:raise ValueError("v2 frozen raw feature bytes changed")
    return {k: load_head(Path(out) / v["file"], v["sha256"], v["metadata"]) for k, v in z["heads"].items()}


def read_labels(dataset, out, packet, requests):
    require_frozen(out, packet);frozen = read_public(dataset)[0];result = {}
    for split in SPLITS:
        name = f"private/{split}.jsonl";path = Path(dataset).resolve() / name
        if path.resolve() != path or file_digest(path) != frozen["files"][name]:raise ValueError("v2 private targets changed")
        result[split] = {z["recordId"]: z for z in map(json.loads, path.read_text().splitlines())}
        if set(result[split]) != set(requests[split]):raise ValueError("v2 evaluation target membership changed")
    path = Path(dataset).resolve() / "private/pairs.json"
    if file_digest(path) != frozen["files"]["private/pairs.json"]:raise ValueError("v2 pair targets changed")
    return result, json.loads(path.read_text())


def evaluate(mode, split, models, catalog, requests, metadata, store, rows, labels, pairs):
    mapping = {(r["mode"], r["role"], r["id"]): r for r in rows};qids = sorted(requests);pids = sorted(metadata["productSplits"][split])
    q = {rid: store.load(mapping[mode, "request", rid])[0][0] for rid in qids}
    c = torch.stack([store.load(mapping[mode, "catalog", pid])[0][0] for pid in pids])
    ct = torch.tensor([[float(a in catalog[pid]["tags"]) for a in ATOMS] for pid in pids]);qt = torch.tensor([[float(a in requests[rid]["context"]["statedPreferences"]) for a in ATOMS] for rid in qids])
    cases = {};summary = {};diagnostics = {}
    with torch.no_grad():
        for arm in ("fusion", "prior", "random", "permuted-query", "public-oracle"):
            cases[arm] = {};query_probs = [];catalog_probs = None
            for rid in qids:
                source = metadata["permutation"][split][rid] if arm == "permuted-query" else rid
                if arm == "public-oracle":qp = qt[qids.index(rid)];cp = ct
                else:qp, cp = models["fusion" if arm == "permuted-query" else arm].probabilities(q[source], c)
                query_probs.append(qp);catalog_probs = cp;scores = cp @ qp
                raw = sorted(pids, key=lambda p: (-scores[pids.index(p)].item(), p))
                eligible = [p for p in raw if not guard_reasons(requests[rid], catalog[p])];chosen = eligible[0] if eligible else None
                dcg = lambda values: sum((2 ** v - 1) / math.log2(i + 2) for i, v in enumerate(values))
                rel = labels[rid]["relevance"];ideal = dcg(sorted(rel.values(), reverse=True)[:3])
                cases[arm][rid] = {"querySource": source, "productIds": pids, "scores": scores.tolist(), "queryProbabilities": qp.tolist(),
                                  "catalogProbabilities": cp.tolist(), "rawRank": raw, "guardedRank": eligible + [p for p in raw if p not in eligible],
                                  "chosenProductId": chosen, "correctBestSet": chosen in labels[rid]["bestProductIds"],
                                  "ndcg3": dcg([rel[p] for p in eligible[:3]]) / ideal if ideal else 1.,
                                  "guardsApplied": True}
            paired = [p for p in pairs if p["split"] == split]
            summary[arm] = {"fixedRequests": 21, "bestSetCorrect": sum(z["correctBestSet"] for z in cases[arm].values()),
                            "meanNdcg3": sum(z["ndcg3"] for z in cases[arm].values()) / 21,
                            "disjointBestSetPairs": len(paired), "pairBothCorrect": sum(all(cases[arm][rid]["correctBestSet"] for rid in p["recordIds"]) for p in paired),
                            "pairChoiceChanges": sum(len({cases[arm][rid]["chosenProductId"] for rid in p["recordIds"]}) == 2 for p in paired),
                            "byPreferenceCardinality": {str(k): {"requests": 7, "correct": sum(z["correctBestSet"] for rid, z in cases[arm].items() if len(requests[rid]["context"]["statedPreferences"]) == k)} for k in (1, 2, 3)}}
            diagnostics[arm] = {"query": atom_metrics(torch.stack(query_probs), qt), "catalog": atom_metrics(catalog_probs, ct)}
    return {"summary": summary, "atomDiagnostics": diagnostics, "cases": cases}


@contextmanager
def native_owner_lock(path, owner):
    if owner != "root":raise ValueError("Native execution belongs to root; no alternative execution")
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError("Native owner lock busy; stop without fallback") from error
        try:yield
        finally:fcntl.flock(handle, fcntl.LOCK_UN)


def run(out, dataset, prepared, encoder_factory, *, started):
    packet, frozen, catalog, requests, metadata, rows, provenance = prepared;out = Path(out)
    write_json(out / "config.json", {"execution": execution_receipt(), "packet": packet, "scope": "one full/focused product-only comparison"})
    store = FeatureStore(out / "features", provenance);begin = time.perf_counter();encoder = encoder_factory();observations = []
    try:
        if encoder.evidence_source != provenance["evidenceSource"]:raise ValueError("Encoder provenance mismatch")
        with (out / "native-observations.jsonl").open("x") as log:
            for i, row in enumerate(rows):
                if time.perf_counter() - begin > 180 or (i < 2 and time.perf_counter() - begin > 60):raise TimeoutError("v2 native time guard; no automatic increase")
                values, timing = encoder.encode(row["text"], row)
                if timing["batchRows"] != 1 or timing["paddedTokensPerRow"] != row["tokens"] or timing["backboneForwardCalls"] != 1 or timing["truncatedRows"] != 0:raise ValueError("v2 exact B1 whole-input receipt required")
                if time.perf_counter() - begin > 180 or (i < 2 and time.perf_counter() - begin > 60):raise TimeoutError("v2 native pilot/budget exceeded")
                _, receipt = store.save(row, values, timing)
                observation = {"mode": row["mode"], "split": row["split"], "role": row["role"], "member": row["member"], "timing": timing,
                               "artifactSha256": receipt["artifactSha256"], "rawKey": receipt["rawKeys"][0], "memory": encoder.memory()}
                observations.append(observation);log.write(json.dumps(observation) + "\n");log.flush()
                print(json.dumps({"event": "v2-native-row", "completed": i + 1, "fixedForwards": 192, "mode": row["mode"], "split": row["split"], "role": row["role"]}), flush=True)
        memory = encoder.memory();model_load = encoder.load_seconds;native_seconds = time.perf_counter() - begin
    finally:encoder.close()
    if len(observations) != 192 or source_hashes() != packet["sourceHashes"]:raise ValueError("v2 fixed extraction/source receipt failed")
    artifacts = {}
    for row in rows:
        # Save already verifies serialization during extraction. Seal bytes here
        # without deserializing held-out numeric rows into the fitting phase.
        for path in store.paths(row["role"], row["member"]):artifacts[str(path.relative_to(out))] = file_digest(path)
    write_json(out / "RAW_FEATURES_FROZEN.json", {"version": VERSION, "nativeForwards": 192,
        "packetSha256": file_digest(out / "packet.json"), "files": artifacts, "privateTargetsRead": False})
    # Fit only TRAIN numeric features. Held-out values stay on disk until freeze.
    fit_started = time.perf_counter();models = {};heads = {};train_contracts = {}
    for mode in MODES:
        inputs = train_inputs(mode, catalog, requests, metadata, store, rows);models[mode] = {};train_contracts[mode] = inputs[-1]
        for arm in ("fusion", "prior"):
            models[mode][arm], heads[mode + "-" + arm] = fit(mode, arm, out, inputs)
        q, c, qt, _, means, contract = inputs;random_head = Factors(means, qt.mean(0), "random")
        name = mode + "-random.pt";sha = save_head(random_head, out / name, contract)
        heads[mode + "-random"] = {"file": name, "sha256": sha, "metadata": contract, "trained": False}
    write_json(out / "HEADS_FROZEN.json", {"version": VERSION, "atUtc": datetime.now(timezone.utc).isoformat(), "packetSha256": file_digest(out / "packet.json"),
        "dataFreezeSha256": packet["dataFreezeSha256"], "sourceHashes": packet["sourceHashes"], "heads": heads,
        "rawFeaturesFreezeSha256": file_digest(out / "RAW_FEATURES_FROZEN.json"), "privateLabelsBeforeFreeze": False})
    fit_seconds = time.perf_counter() - fit_started
    eval_started = time.perf_counter()
    restored = require_frozen(out, packet);labels, pairs = read_labels(dataset, out, packet, requests);results = {}
    for mode in MODES:
        results[mode] = {};mode_models = {a: restored[mode + "-" + a] for a in ("fusion", "prior", "random")}
        for split in SPLITS:
            require_frozen(out, packet)
            results[mode][split] = evaluate(mode, split, mode_models, catalog, requests[split], metadata, store, rows, labels[split], pairs)
            write_json(out / f"{mode}-{split}-evaluation.json", results[mode][split])
        # Exact post-evaluation training means/targets reload, no statistics update.
        if train_inputs(mode, catalog, requests, metadata, store, rows)[-1] != train_contracts[mode]:raise ValueError("v2 TRAIN statistics changed")
    require_frozen(out, packet);read_public(dataset)
    native_timing = {}
    for mode in MODES:
        native_timing[mode] = {}
        for split in SPLITS:
            native_timing[mode][split] = {}
            for role in ("catalog", "request"):
                values = [z["timing"]["encodeTotalMs"] for z in observations if (z["mode"], z["split"], z["role"]) == (mode, split, role)]
                native_timing[mode][split][role] = {"forwards": len(values), "nativeEncodeSumMs": sum(values),
                    "minimumMs": min(values), "medianMs": statistics.median(values), "maximumMs": max(values)}
    report = {"status": "completed", "version": VERSION, "execution": json.loads((out / "config.json").read_text())["execution"],
              "packetSha256": file_digest(out / "packet.json"), "dataFreezeSha256": packet["dataFreezeSha256"],
              "fixedNativeForwards": 192, "nativeForwards": len(observations), "nativeModelLoadSeconds": model_load,
              "nativeLifecycleSecondsIncludingLoad": native_seconds,
              "nativeEncodingPersistenceSeconds": native_seconds - model_load, "nativeMemoryAfterExtraction": memory,
              "actualNativeEncodeTimingByRepresentationSplitRole": native_timing,
              "cpuFitFreezeSeconds": fit_seconds, "cpuEvaluationAuditSeconds": time.perf_counter() - eval_started,
              "wallSecondsIncludingStartup": time.perf_counter() - started, "encoderTextOverlap": frozen["encoderTextOverlap"],
              "coverage": frozen["coverage"], "fit": SCHEDULE, "heads": heads,
              "quality": {m: {s: results[m][s]["summary"] for s in SPLITS} for m in MODES},
              "atomDiagnostics": {m: {s: results[m][s]["atomDiagnostics"] for s in SPLITS} for m in MODES},
              "privateRankingLabelsOpenedAfterAllHeadFreezes": True, "meansFromTrainOnly": True,
              "generatesCopy": False, "imageModel": "not evaluated; no dummy variants", "limitations": frozen["limitations"]}
    write_json(out / "report.json", report);return report


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument("action", choices=("freeze", "prepare", "run"))
    parser.add_argument("--dataset", type=Path, default=Path("experiments/ranker/pretrained/catalog_fusion/generalization_v2"))
    parser.add_argument("--out", type=Path, required=True);parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--prior", type=Path, default=Path("data/ranker/pretrained/catalog-user-fusion-v1-attempt-1"))
    parser.add_argument("--packet", type=Path);parser.add_argument("--reviewed-packet-sha256");parser.add_argument("--reviewed-data-sha256")
    parser.add_argument("--native-owner", choices=("root",));parser.add_argument("--confirm-serialized", action="store_true")
    parser.add_argument("--native-lock", type=Path, default=Path("data/ranker/pretrained/.native-generalization-v2.lock"))
    args = parser.parse_args();started = time.perf_counter();torch.set_num_threads(1)
    if args.action == "freeze":print(json.dumps(freeze_data(args.out), indent=2));return
    from tokenizers import Tokenizer
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        prepared = prepare(args.dataset, args.model, args.prior, Tokenizer.from_file(str(args.model / "tokenizer.json")));packet = prepared[0]
        if args.action == "run":
            if not args.confirm_serialized or args.native_owner != "root":raise ValueError("Root must confirm sole native ownership; no fallback execution")
            if args.packet is None or file_digest(args.packet) != args.reviewed_packet_sha256 or args.reviewed_data_sha256 != packet["dataFreezeSha256"] or json.loads(args.packet.read_text()) != packet:raise ValueError("Reviewed v2 data/source/token packet checksums required")
        write_json(args.out / "packet.json", packet)
        if args.action == "prepare":print(json.dumps({"status": "CPU packet only; no fitting/native", "packetSha256": file_digest(args.out / "packet.json"), "tokenRange": packet["tokenRange"], "nativeForwards": 192}, indent=2));return
        with native_owner_lock(args.native_lock, args.native_owner):
            verified = verify_model(args.model, packet)
            if verified["weightsSha256"] != prepared[-1]["weightsSha256"] or verified["tokenizerSha256"] != prepared[-1]["tokenizerSha256"]:raise ValueError("Full pinned shard/tokenizer hash mismatch")
            write_json(args.out / "model-verification.json", verified)
            def factory():assert_model_files(args.model, verified);return NativeEncoder(args.model)
            run(args.out, args.dataset, prepared, factory, started=started)
    except BaseException as error:
        write_json(args.out / "FAILED.json", {"status": "failed", "exception": type(error).__name__, "message": str(error),
            "fixedNativeForwards": 192, "fixedRequestsPerRepresentationPerSplit": 21,
            "partialSuccessNotPromoted": True, "wallSecondsIncludingStartup": time.perf_counter() - started});raise


if __name__ == "__main__":main()
