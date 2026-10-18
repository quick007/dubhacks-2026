"""CPU failure-path fixtures only: HTTP sockets, fake workers and fake native state.

No browser, native model, or generated-copy quality result is claimed here.
"""
from concurrent.futures import Future
from contextlib import contextmanager
import hashlib
import http.client
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import store_persona_engine_v1 as engine
from . import store_candidate_image_service_v1 as image
from . import store_persona_runtime_v1 as runtime
from . import store_demo_engine as resident


class HandlerTests(unittest.TestCase):
    def response(self,error):
        class Worker:
            def submit(self,value,request_id,started):
                self.cancel=threading.Event();events=queue.Queue();future=Future()
                future.set_exception(error);events.put(None)
                return events,self.cancel,future
        worker=Worker();server=engine.ThreadingHTTPServer(('127.0.0.1',0),engine.persona_handler_for(worker));server.daemon_threads=True
        thread=threading.Thread(target=lambda:server.serve_forever(poll_interval=.01),daemon=True);thread.start()
        connection=http.client.HTTPConnection('127.0.0.1',server.server_address[1],timeout=3)
        try:
            connection.request('POST','/personalize',json.dumps({'query':'Browse the store','persona':{'background':[],'colorPreferences':[]},'publicHistory':[]}),{'Content-Type':'application/json','Origin':'http://127.0.0.1:4313'})
            response=connection.getresponse();body=response.read().decode()
            return response.status,body,worker.cancel.is_set()
        finally:
            connection.close();server.shutdown();server.server_close();thread.join(timeout=3)
    def test_real_http_fake_worker_failure_serializes_error_ndjson(self):
        status,body,cancelled=self.response(ValueError('fixture-only worker failure'))
        self.assertEqual(status,200);message=json.loads(body)
        self.assertEqual(message['type'],'error');self.assertEqual(message['message'],'ValueError: fixture-only worker failure')
        self.assertTrue(message['requestId']);self.assertTrue(cancelled)
    def test_cancellation_class_is_available_to_handler_and_is_not_error_event(self):
        status,body,cancelled=self.response(engine.RequestCancelled('fixture cancelled'))
        self.assertEqual(status,200);self.assertEqual(body,'');self.assertTrue(cancelled)


class ForwardReceiptTests(unittest.TestCase):
    def test_warm_feature_hits_and_fresh_feature_calls_have_distinct_actual_counts(self):
        warm=engine.feature_forward_receipt({'queryFeature':{'nativeBackboneForwardCalls':0,'cacheHit':True},'contextFeature':{'nativeBackboneForwardCalls':0,'cacheHit':True}})
        cold=engine.feature_forward_receipt({'queryFeature':{'nativeBackboneForwardCalls':1,'cacheHit':False},'contextFeature':{'nativeBackboneForwardCalls':1,'cacheHit':False}})
        self.assertEqual(warm['totalNativeForwards'],0);self.assertEqual(cold['totalNativeForwards'],2)
        self.assertTrue(warm['contextFeatureCacheHit']);self.assertFalse(cold['contextFeatureCacheHit'])
        self.assertIsNone(engine.feature_forward_receipt({})['totalNativeForwards'])


class ConstructorAndSourceTests(unittest.TestCase):
    def fake_base(self,directory,events):
        class Stream:
            def __enter__(self):events.append('stream-enter')
            def __exit__(self,*args):events.append('stream-exit')
        class Base:
            def __init__(self,**kwargs):
                self.output=Path(directory);self.startup={};self.mode='parallel';self.copy_metadata={}
                self.copy_run=Path(directory)/'copy';self.copy_run.mkdir();(self.copy_run/'adapter.safetensors').write_bytes(b'fixture')
                self.copy_sha=hashlib.sha256(b'fixture').hexdigest();self.before_hashes={'fixture-layer':'unchanged'}
                self.stream_context=Stream();self.stream_context.__enter__();self._native_stream_exited=False
                self.bb=SimpleNamespace(model=SimpleNamespace(language_model=object()),mx=SimpleNamespace(synchronize=lambda *args:events.append('synchronize')),runtime_memory=lambda:{'fixtureOnly':True})
                self.session=SimpleNamespace(head_sha='fixture-head',products=[],snapshot={'externalMapping':{'products':[]}})
            def _assert_candidate(self):pass
            def close(self):
                try:self._assert_candidate()
                finally:events.append('base-close')
                return {'fixtureOnly':True}
        return Base
    def fake_runtime(self):
        return SimpleNamespace(catalog={'products':[]},assert_current=lambda:None)
    def test_startup_reports_actual_loaded_feature_entry_count_without_fresh_encode_claim(self):
        events=[];rt=self.fake_runtime();rt.context_cache={'features':{'first':{},'second':{}},'sourceExtractionReportSha256':'fixture-report-sha'}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(image,'ImageCandidateDemoEngine',self.fake_base(directory,events)),patch.object(runtime,'PersonaRuntime',return_value=rt):
                obj=engine.engine_class()(persona_head='unused',persona_head_sha256='chosen-checkpoint',persona_feature_root='unused')
                metadata=obj.startup['persona'];self.assertEqual(metadata['checkpointSha256'],'chosen-checkpoint')
                self.assertEqual(metadata['featureCaches']['precomputedPublicContextEntriesAtStartup'],2)
                self.assertEqual(metadata['featureCaches']['startupFeatureEncodingNativeForwards'],0)
                self.assertEqual(metadata['featureCaches']['queryEntriesAtStartup'],0)
                self.assertEqual(metadata['featureCaches']['contextCacheSourceExtractionReportSha256'],'fixture-report-sha')
    def test_post_super_startup_write_failure_cleans_real_base_protocol_and_preserves_original(self):
        events=[];failure=OSError('fixture startup write failed');original_write=Path.write_text
        def write(path,*args,**kwargs):
            if path.name=='startup.json':raise failure
            return original_write(path,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(image,'ImageCandidateDemoEngine',self.fake_base(directory,events)),patch.object(runtime,'PersonaRuntime',return_value=self.fake_runtime()),patch.object(Path,'write_text',write),patch('experiments.ranker.pretrained.semantic_copy_training.parameter_hashes',return_value={'fixture-layer':'unchanged'}),patch.dict('sys.modules',{'mlx_lm.generate':SimpleNamespace(generation_stream='fixture-stream')}):
                klass=engine.engine_class()
                with self.assertRaises(OSError) as caught:klass(persona_head='unused',persona_head_sha256='unused',persona_feature_root='unused')
            self.assertIs(caught.exception,failure);self.assertEqual(events.count('stream-exit'),1)
            receipt=json.loads((Path(directory)/'initialization-failure.json').read_text())
            self.assertTrue(receipt['frozenHashCleanupConfirmed']);self.assertEqual(receipt['error'],'fixture startup write failed')
            self.assertEqual(events.count('synchronize'),1)
    def test_base_constructor_failure_is_not_cleaned_twice_by_wrapper(self):
        failure=RuntimeError('fixture base already cleaned')
        class Base:
            def __init__(self,**kwargs):raise failure
        with patch.object(image,'ImageCandidateDemoEngine',Base),patch.object(runtime,'PersonaRuntime',return_value=self.fake_runtime()),patch.object(image,'cleanup_failed_start') as cleanup:
            with self.assertRaises(RuntimeError) as caught:engine.engine_class()(persona_head='unused',persona_head_sha256='unused',persona_feature_root='unused')
            self.assertIs(caught.exception,failure);cleanup.assert_not_called()
    def test_cleanup_error_does_not_replace_original_startup_exception(self):
        events=[];failure=OSError('original fixture failure')
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(image,'ImageCandidateDemoEngine',self.fake_base(directory,events)),patch.object(runtime,'PersonaRuntime',return_value=self.fake_runtime()),patch.object(Path,'write_text',side_effect=failure),patch.object(image,'cleanup_failed_start',side_effect=RuntimeError('cleanup fixture failed')):
                with self.assertRaises(OSError) as caught:engine.engine_class()(persona_head='unused',persona_head_sha256='unused',persona_feature_root='unused')
            self.assertIs(caught.exception,failure);self.assertTrue(any('cleanup fixture failed' in note for note in failure.__notes__))
    def test_new_engine_source_drift_is_rejected_and_close_still_tears_down(self):
        events=[]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(image,'ImageCandidateDemoEngine',self.fake_base(directory,events)),patch.object(runtime,'PersonaRuntime',return_value=self.fake_runtime()):
                obj=engine.engine_class()(persona_head='unused',persona_head_sha256='unused',persona_feature_root='unused')
                self.assertIn(str(Path(engine.__file__).resolve()),obj.startup['persona']['engineSourcesSha256'])
                with patch.object(engine,'sha',return_value='drift'):
                    with self.assertRaisesRegex(ValueError,'engine/handler source changed'):obj._assert_candidate()
                    with self.assertRaisesRegex(ValueError,'engine/handler source changed'):obj.close()
                self.assertEqual(events.count('base-close'),1)
    def test_source_drift_between_ranking_and_event_prevents_event_emit(self):
        events=[];drift={'active':False};real_sha=engine.sha;rt=self.fake_runtime()
        def rank(*args):
            drift['active']=True
            return {'query':'Browse the store','itemBudget':None,'hardRequirements':[]},{},[],{},[],{}
        rt.rank=rank
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(image,'ImageCandidateDemoEngine',self.fake_base(directory,events)),patch.object(runtime,'PersonaRuntime',return_value=rt):
                obj=engine.engine_class()(persona_head='unused',persona_head_sha256='unused',persona_feature_root='unused');obj.adapter=object()
                emitted=[]
                with patch.object(engine,'sha',side_effect=lambda path:'drift' if drift['active'] else real_sha(path)):
                    with self.assertRaisesRegex(ValueError,'engine/handler source changed'):
                        obj.personalize({'query':'Browse the store'},'fixture-request',time.perf_counter(),emitted.append,lambda:False)
                self.assertEqual(emitted,[])
                receipt=json.loads((Path(directory)/'requests/fixture-request/receipt.json').read_text());self.assertEqual(receipt['status'],'failed')

if __name__=='__main__':unittest.main()
