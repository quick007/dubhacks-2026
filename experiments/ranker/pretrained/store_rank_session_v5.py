"""Explicit v5 head on the verified original v3 feature/catalog session.

No new forward-count metadata is disguised as the old64-forward run. The caller
selects a reviewed export by checksum after evaluating quality. Serving reads
only that export/head and original v3 verification/catalog assets, never v5
reports, evaluation data, training labels or teacher text. No native load here.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import time

import torch

from .store_catalog_v1 import canonical
from .store_function_adaptation import load_head, product_text, VOCAB
from .store_model_bridge import assert_current_sources
from .store_rank_session import StoreRankSession

VERSION = 'store-v5-rank-session.v1'
EXPORT_VERSION = 'store-rank-head-v5-export.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_value(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Exact lowercase SHA256 required')
    return value


def read_export(path, expected_sha256):
    path = Path(path)
    if path.is_symlink() or sha(path) != digest_value(expected_sha256):
        raise ValueError('Reviewed v5 serving export checksum mismatch')
    value = json.loads(path.read_text())
    if (value.get('version') != EXPORT_VERSION or value.get('frozen927HashesUnchanged') is not True or
            type(value.get('newNativeForwards')) is not int or value['newNativeForwards'] != 96 or
            type(value.get('totalFeatureVectors')) is not int or value['totalFeatureVectors'] != 144 or
            value.get('requestEncodingUnchanged') is not True or value.get('teacherLookupAtServing') is not False):
        raise ValueError('Completed96-new/144-total native proof and927 integrity export required')
    head = value['head']; metadata = head['metadata']; config = metadata['config']
    if (head['file'] != 'learned.pt' or metadata['catalogMembers'] != list(range(16)) or
            len(metadata['queryMembers']) != 96 or len(set(metadata['queryMembers'])) != 96 or
            metadata['vocabulary'] != list(VOCAB) or config.get('steps') != 80 or
            config.get('dimensions') != 4096 or config.get('classes') != 16 or
            config.get('newNativeForwards') != 96 or config.get('totalFeatureVectors') != 144 or
            config.get('newTrainForwards') != 64 or config.get('newEvaluationForwards') != 32 or
            config.get('reusedFeatures') != 48):
        raise ValueError('Exact known16 FunctionHead /96 TRAIN /80-step serving contract required')
    if not all(type(rid) is str for rid in metadata['queryMembers']):
        raise ValueError('Only external TRAIN member identifiers allowed in head metadata')
    digest_value(head['sha256']); digest_value(value['sourceBaseReportSha256'])
    head_path = path.parent / head['file']
    if head_path.is_symlink() or sha(head_path) != head['sha256']:
        raise ValueError('V5 head weights checksum mismatch')
    return value


class V5RankSession(StoreRankSession):
    """Factory-compatible extension; the original session/report stay intact."""
    def __init__(self, backbone, run_directory, *, head_export_path, head_export_sha256,
                 source_validator=assert_current_sources):
        self.head_export_path = Path(head_export_path)
        self.head_export_sha = digest_value(head_export_sha256)
        export = read_export(self.head_export_path, self.head_export_sha)
        base_run = Path(run_directory)
        if base_run.resolve() != Path(export['sourceBaseRun']).resolve():
            raise ValueError('V5 head requires the exact original v3 catalog feature base')
        if sha(base_run / 'report.json') != export['sourceBaseReportSha256']:
            raise ValueError('Original v3 feature-base report changed')
        # This legitimately loads the OLD64-forward report/head/catalog first.
        # No v5 run report is ever presented to the original loader.
        super().__init__(backbone, base_run, source_validator=source_validator)
        if self.provenance != export['featureProvenance']:
            raise ValueError('V5 training and resident original catalog provenance differ')
        self.feature_base_head_sha = self.head_sha
        if [r['caseId'] for r in self.catalog_rows] != [str(i) for i in range(16)]:
            raise ValueError('Original16 public catalog order required')
        for row, product in zip(self.catalog_rows, self.products, strict=True):
            if row['text'] != product_text(product):
                raise ValueError('V5 public catalog facts differ from verified source vectors')
        expected_files = {path.resolve() for row in self.catalog_rows for path in self.store.paths(row['role'], row['member'])}
        exported_files = {Path(name).resolve(): checksum for name, checksum in export['originalCatalogFeatureFilesSha256'].items()}
        if len(exported_files) != 32 or set(exported_files) != expected_files:
            raise ValueError('Exactly32 original16 catalog tensor/receipt pins required')
        if any(sha(path) != digest_value(checksum) for path, checksum in exported_files.items()):
            raise ValueError('Original v3 catalog tensor/receipt changed')
        catalog = torch.cat([self.store.load(row)[0] for row in self.catalog_rows])
        self.head_path = self.head_export_path.parent / export['head']['file']
        head = load_head(self.head_path, export['head']['sha256'], export['head']['metadata'])
        if not torch.equal(head.catalog_mean, catalog.double().mean(0).float()):
            raise ValueError('V5 catalog centering mean differs from the verified16 source vectors')
        self.head = head.requires_grad_(False).eval(); self.head_sha = export['head']['sha256']
        self.feature_base_provenance = {'kind': 'original-v3-frozen-feature-base', 'run': str(base_run),
            'reportSha256': export['sourceBaseReportSha256'], 'packetSha256': self.report['packetSha256'],
            'originalRunNativeForwards': self.report['nativeForwards'], 'catalogFeatureVectors': 16,
            'catalogTensorAndReceiptPins': 32, 'modelRevision': self.provenance['modelRevision'],
            'originalV3HeadSha256': self.feature_base_head_sha}
        self.rank_head_provenance = {'kind': 'v5-known-catalog-function-head', 'version': VERSION,
            'servingExportSha256': self.head_export_sha, 'headCheckpointSha256': self.head_sha,
            'newRunNativeForwards': 96, 'totalFeatureVectors': 144, 'reusedTrainAndCatalogVectors': 48,
            'trainQueries': 96, 'freshEvaluationQueries': 32, 'trainSteps': 80,
            'frozen927TrainingHashesUnchanged': True, 'teacherOrEvaluationFilesReadAtServing': False,
            'requestEncodingUnchanged': True,
            'selection': 'explicit caller-selected serving export checksum; adapter performs no quality promotion',
            'scope': export['scope']}
        self.startup_provenance = {'featureBase': deepcopy(self.feature_base_provenance),
            'rankHead': deepcopy(self.rank_head_provenance)}
        self._assert_head_assets()

    def _assert_head_assets(self):
        if sha(self.head_export_path) != self.head_export_sha or sha(self.head_path) != self.head_sha:
            raise ValueError('Loaded v5 serving export/head changed')
        return True

    def integrity_receipt(self):
        self._assert_head_assets()
        return {'v5ServingExportUnchanged': True, 'v5HeadCheckpointUnchanged': True,
            'servingExportSha256': self.head_export_sha, 'headCheckpointSha256': self.head_sha}

    def rank_request(self, request, *, copy_adapter=None):
        started = time.perf_counter(); self._assert_head_assets()
        result = super().rank_request(request, copy_adapter=copy_adapter)
        payload = {key: value for key, value in result.items() if key not in
            ('payloadSha256', 'payloadBytes', 'freshRankToSerializedActionsMs')}
        payload.update(featureBaseProvenance=deepcopy(self.feature_base_provenance),
            rankHeadProvenance=deepcopy(self.rank_head_provenance),
            baseSessionRankToSerializedActionsMs=result['freshRankToSerializedActionsMs'])
        encoded = canonical(payload).encode()
        return {**payload, 'payloadSha256': hashlib.sha256(encoded).hexdigest(), 'payloadBytes': len(encoded),
            'freshRankToSerializedActionsMs': (time.perf_counter() - started) * 1000}


def make_v5_rank_session_factory(head_export_path, head_export_sha256):
    """Use as rank_session_factory after the owner adds that optional hook.

    No evaluation/teacher/report path for the new run is inferred or opened.
    Capturing an explicit export/checksum does not start a worker or model.
    """
    path = Path(head_export_path); expected = digest_value(head_export_sha256)
    def factory(backbone, run_directory, *, source_validator=assert_current_sources):
        return V5RankSession(backbone, run_directory, head_export_path=path,
            head_export_sha256=expected, source_validator=source_validator)
    return factory
