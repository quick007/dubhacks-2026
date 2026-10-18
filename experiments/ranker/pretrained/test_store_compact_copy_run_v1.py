"""Meaningful lifecycle mocks only; no native execution or fabricated receipts."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import signal
import tempfile
from types import SimpleNamespace
import unittest

from .store_compact_copy_run_v1 import (PREFLIGHT_SHA, CHECKPOINT_SHA, run_compact, shutdown_engine)

HASHES = {str(i): 'hash-' + str(i) for i in range(927)}

class CompactRunTests(unittest.TestCase):
    def dependencies(self, *, partial_failure=False, runner_failure=False, listening=False):
        calls = []; owned = [False]
        @contextmanager
        def lock():
            self.assertFalse(owned[0]); owned[0] = True; calls.append('lock-enter')
            try: yield
            finally: calls.append('lock-exit'); owned[0] = False
        class Engine:
            def __init__(inner, **kwargs):
                self.assertTrue(owned[0]); calls.append('initialize')
                self.assertEqual(kwargs['width'], 20); self.assertEqual(kwargs['memory_gib'], 7)
                inner.bb = object(); inner.adapter = object(); inner.before_hashes = HASHES.copy()
                inner.startup = {'sourceTensorsHashed': 927}
                if partial_failure: raise RuntimeError('constructor failed after backbone')
        def runner(bb, adapter, checkpoint, checkpoint_sha, out, preflight_path, **kwargs):
            self.assertFalse(owned[0], 'API-owned lock must never be nested in wrapper lock'); calls.append('api')
            self.assertEqual(checkpoint_sha, CHECKPOINT_SHA); self.assertEqual(kwargs['reviewed_preflight_sha256'], PREFLIGHT_SHA)
            self.assertEqual(kwargs['plan'], 'diagnostic8'); self.assertEqual(kwargs['native_owner'], 'root')
            if runner_failure: raise RuntimeError('API failed after initialization')
            return {'status': 'completed', 'summary': {'mockOnly': True}}
        def shutdown(engine, preflight):
            self.assertTrue(owned[0]); self.assertTrue(hasattr(engine, 'bb')); calls.append('shutdown')
            return {'status': 'verified', 'frozen927HashesUnchanged': True, 'tensorsHashed': 927, 'mockOnly': True}
        def preflight(manifest, **kwargs):
            calls.append('cpu-preflight')
            return {'sourceSha256': {'test': 'bytes'}, 'rankRun': 'test-rank', 'checkpoint': 'test-checkpoint',
                'checkpointSha256': CHECKPOINT_SHA, 'expectedFrozen927Hashes': HASHES.copy()}
        return {'preflight': preflight, 'service_listening': lambda: listening, 'engine_type': lambda: Engine,
            'native_lock': lock, 'runner': runner, 'shutdown': shutdown, 'source_inventory': lambda: {'test': 'bytes'}}, calls

    def test_module_import_and_default_prepare_never_load_native_dependencies(self):
        code = ('import sys; import experiments.ranker.pretrained.store_compact_copy_run_v1; '
            'assert not any(n=="mlx" or n.startswith("mlx.") for n in sys.modules)')
        subprocess.run([sys.executable, '-c', code], check=True)
        with tempfile.TemporaryDirectory() as t:
            deps, calls = self.dependencies(listening=True)
            report = run_compact(Path(t) / 'cpu', manifest='test', _dependencies=deps)
            self.assertEqual(report['status'], 'cpu-preflight-only'); self.assertFalse(report['nativeExecution'])
            self.assertEqual(calls, ['cpu-preflight'])

    def test_run_refuses_service_and_missing_root_with_durable_outer_receipts(self):
        with tempfile.TemporaryDirectory() as t:
            for index, owner in enumerate((None, 'root')):
                deps, calls = self.dependencies(listening=True); out = Path(t) / str(index)
                report = run_compact(out, manifest='test', run=True, native_owner=owner, _dependencies=deps)
                self.assertEqual(report['status'], 'failed'); self.assertEqual(calls, [])
                self.assertTrue((out / 'lifecycle.json').is_file()); self.assertTrue((out / 'startup.json').is_file())
                self.assertTrue((out / 'shutdown.json').is_file())

    def test_initialization_api_shutdown_use_separate_non_nested_lock_scopes(self):
        with tempfile.TemporaryDirectory() as t:
            deps, calls = self.dependencies()
            report = run_compact(Path(t) / 'native', manifest='test', run=True, native_owner='root', _dependencies=deps)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(calls, ['cpu-preflight', 'lock-enter', 'initialize', 'lock-exit', 'api', 'lock-enter', 'shutdown', 'lock-exit'])

    def test_partial_constructor_and_api_failures_still_hash_shutdown_and_preserve_failure(self):
        with tempfile.TemporaryDirectory() as t:
            for index, options in enumerate(({'partial_failure': True}, {'runner_failure': True})):
                deps, calls = self.dependencies(**options); out = Path(t) / str(index)
                report = run_compact(out, manifest='test', run=True, native_owner='root', _dependencies=deps)
                self.assertEqual(report['status'], 'failed'); self.assertEqual(calls.count('shutdown'), 1)
                self.assertEqual(report['shutdown']['tensorsHashed'], 927)
                self.assertIn('failed after', report['error']['message'])
                self.assertEqual(json.loads((out / 'lifecycle.json').read_text())['status'], 'failed')

    def test_busy_initialization_lock_never_constructs_model(self):
        with tempfile.TemporaryDirectory() as t:
            deps, calls = self.dependencies()
            @contextmanager
            def busy():
                raise RuntimeError('Native owner lock busy'); yield
            deps['native_lock'] = busy
            report = run_compact(Path(t) / 'busy', manifest='test', run=True, native_owner='root', _dependencies=deps)
            self.assertEqual(report['status'], 'failed'); self.assertEqual(calls, ['cpu-preflight'])

    def test_shutdown_releases_stream_even_if_hash_or_checkpoint_read_fails(self):
        calls = []
        class Stream:
            def __exit__(self, *_): calls.append('stream-exit')
        engine = SimpleNamespace(bb=SimpleNamespace(mx=SimpleNamespace(synchronize=lambda: calls.append('sync')),
            runtime_memory=lambda: {}), stream_context=Stream(), before_hashes=HASHES.copy())
        receipt = shutdown_engine(engine, {'checkpoint': '/nonexistent/checkpoint', 'checkpointSha256': 'missing', 'expectedFrozen927Hashes': HASHES},
            hash_reader=lambda _: HASHES.copy())
        self.assertEqual(receipt['status'], 'partial-unavailable'); self.assertTrue(receipt['streamReleased'])
        self.assertEqual(calls, ['sync', 'stream-exit']); self.assertIn('checkpointError', receipt)
        self.assertEqual(receipt['tensorsHashed'], 927)

    def test_outer_monotonic_budget_fails_after_api_and_still_shuts_down_without_alarm(self):
        with tempfile.TemporaryDirectory() as t:
            deps, calls = self.dependencies(); now = [0.]; original = deps['runner']
            deps['clock'] = lambda: now[0]
            def overrun(*args, **kwargs):
                result = original(*args, **kwargs); now[0] = 601.; return result
            deps['runner'] = overrun
            previous = signal.getitimer(signal.ITIMER_REAL)
            report = run_compact(Path(t) / 'overrun', manifest='test', run=True, native_owner='root', _dependencies=deps)
            self.assertEqual(report['status'], 'failed'); self.assertTrue(report['outerBudgetExceeded'])
            self.assertEqual(calls.count('shutdown'), 1); self.assertEqual(report['shutdown']['status'], 'verified')
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), previous)

    def test_unique_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileExistsError): run_compact(Path(t), manifest='test')

if __name__ == '__main__': unittest.main()
