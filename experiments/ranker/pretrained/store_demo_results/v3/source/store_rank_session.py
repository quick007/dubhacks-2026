"""Resident actual-store rank adapter; model-side actions, no HTTP/UI changes.

Caller owns the native worker lock and a verified frozen Qwen backbone. Copy
adapters must be disabled for ranking: cached catalog features were frozen-base.
The current 16-class head is a bounded known-store candidate, not promoted.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch

from .catalog_fusion.inputs import digest
from .catalog_fusion.runtime import FeatureStore, assert_model_files
from .store_catalog_v1 import canonical, load_snapshot
from .store_model_bridge import assert_current_sources
from .store_function_adaptation import load_head, rank, request_text, validate_request


def action_bindings(request, product_indices, snapshot):
    """Bind selected products to real declared variants; no image learning claim."""
    validate_request(request)
    desired=[r['value'] for r in request['hardRequirements'] if r['attribute']=='color']
    cards=[]
    for index in product_indices:
        if type(index) is not int or not 0<=index<len(snapshot['inputs']['products']):
            raise ValueError('Actual catalog membership required')
        product=snapshot['inputs']['products'][index];external=snapshot['externalMapping']['products'][index]
        eligible=[i for i,c in enumerate(product['colorVariants']) if all(c['colorName']==v for v in desired)]
        if not eligible:raise ValueError('No single declared variant satisfies every hard color requirement')
        variant=eligible[0];color=product['colorVariants'][variant];route=external['colorVariants'][variant]
        cards.append({'productIndex':index,'facts':{'name':product['name'],'category':product['category'],
            'itemPrice':product['itemPrice'],'selectedColor':color['colorName']},
            'action':{'productId':external['productId'],'variantId':route['variantId'],'image':deepcopy(route['images'][0])}})
    return cards


def exact_name_order(request, products, guarded):
    """An explicit whole-query exact name is an identity instruction, after guards."""
    name=request['query'].strip().casefold()
    matches=[i for i,p in enumerate(products) if p['name'].casefold()==name]
    order=list(guarded['eligibleRank'])
    preferred=[i for i in matches if i in order]
    order=preferred+[i for i in order if i not in preferred]
    return order,{'applied':bool(preferred),'policy':'whole-query exact public product name, case insensitive; hard guards first',
                  'matchedProductIndices':matches,'learned':False}


class StoreRankSession:
    def __init__(self, backbone, run_directory, *, source_validator=assert_current_sources):
        self.bb=backbone;self.run=Path(run_directory);self.report=json.loads((self.run/'report.json').read_text())
        if self.report['status']!='completed' or self.report['nativeForwards']!=64:
            raise ValueError('Completed actual-store rank checkpoint required')
        self.packet=json.loads((self.run/'packet.json').read_text());self.provenance=self.report['featureProvenance']
        if backbone.revision!=self.provenance['modelRevision'] or backbone.hidden_size!=4096:
            raise ValueError('Resident backbone differs from trained feature contract')
        if hashlib.sha256((self.run/'packet.json').read_bytes()).hexdigest()!=self.report['packetSha256']:
            raise ValueError('Native rank packet changed')
        assert_model_files(self.packet['model']['modelPath'],self.provenance)
        self.source_validator=source_validator
        self.snapshot=load_snapshot();self.source_validator(self.snapshot['receipt'])
        self.products=self.snapshot['inputs']['products']
        self.catalog_rows=[r for r in self.packet['records'] if r['role']=='catalog']
        if len(self.catalog_rows)!=16:raise ValueError('Exact16 catalog feature membership required')
        self.store=FeatureStore.open_existing(self.run/'features',self.provenance)
        receipt=self.report['heads']['learned'];self.head=load_head(self.run/receipt['file'],receipt['sha256'],receipt['metadata'])
        self.head_sha=receipt['sha256'];self.forward_calls=0

    def rank_request(self, request, *, copy_adapter=None):
        """Fresh native query -> verified catalog -> learned head -> actual actions.

        Entire timer includes source freshness, tokenization, encoding, cached
        catalog verification, scoring, hard guards, exact-name policy, variant
        binding and serialized six-card decision payload. No generated copy.
        """
        started=time.perf_counter();validate_request(request);self.source_validator(self.snapshot['receipt'])
        text=request_text(request);tokens=self.bb.tokenizer.encode(text,add_special_tokens=False)
        if not 0<len(tokens)<=512:raise ValueError('Whole public query must fit512tokens; no truncation')
        active=self.bb.model.layers[-1].mlp.down_proj
        if hasattr(active,'lora_a') and copy_adapter is None:
            raise ValueError('Active copy adapter must be explicitly disabled for ranking')
        if copy_adapter is not None:
            if active is not copy_adapter:raise ValueError('Copy adapter must belong to the resident model')
            from .semantic_copy_training import disabled
            context=disabled(self.bb.model,copy_adapter)
        else:context=nullcontext()
        with context:
            values,timing=self.bb.encode([text])
        self.forward_calls+=1
        if timing['batchRows']!=1 or timing['backboneForwardCalls']!=1 or timing['truncatedRows']!=0 or timing['paddedTokensPerRow']!=len(tokens):
            raise ValueError('Exact whole-query B1 feature receipt required')
        catalog=torch.cat([self.store.load(row)[0] for row in self.catalog_rows])
        with torch.no_grad():scores=(self.head.probabilities('catalog',catalog)@self.head.probabilities('query',values)[0]).tolist()
        guarded=rank(scores,request,self.products,6);order,identity=exact_name_order(request,self.products,guarded)
        indices=order[:6];cards=action_bindings(request,indices,self.snapshot)
        result={'schemaVersion':'actual-store-ranked-actions.v1','request':deepcopy(request),'scores16':scores,
            'learnedRank':guarded,'productIndices':indices,'cards':cards,'eligibleCount':len(order),
            'shortfall':max(0,6-len(order)),'exactNamePolicy':identity,'rankCheckpointSha256':self.head_sha,
            'queryTextSha256':digest(text),'queryTokens':len(tokens),'native':timing,
            'imageColorPolicy':'first declared satisfying color; first declared image, deterministic and untrained',
            'stock':None,'sizes':None,'delivery':'cart-subtotal policy excluded from item budget',
            'scope':'model-side learned rank/action binding; no generated copy, browser paint or live UI integration'}
        self.source_validator(self.snapshot['receipt']);encoded=canonical(result).encode()
        return {**result,'payloadSha256':hashlib.sha256(encoded).hexdigest(),'payloadBytes':len(encoded),
                'freshRankToSerializedActionsMs':(time.perf_counter()-started)*1000}
