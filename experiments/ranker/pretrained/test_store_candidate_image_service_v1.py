"""Injected CPU lifetime tests: no model, native stream, browser, service or Git."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from . import store_candidate_image_service_v1 as image
from . import store_candidate_engine_v1 as candidate
from . import store_demo_image_binding_v1 as binding
from .store_demo_engine import ResidentDemoEngine
from .store_catalog_v1 import StoreCatalogError


IMAGE_SOURCES = ('experiments/ranker/pretrained/store_candidate_image_service_v1.py',
    'experiments/ranker/pretrained/store_demo_image_binding_v1.py',
    'experiments/ranker/pretrained/store_rank_session_v5.py',
    'scripts/prepare-product-webp.mjs', 'scripts/prepare-product-responsive-webp.mjs',
    'scripts/prepare-product-runtime-projection.mjs')


class ImageCandidateServiceTests(unittest.TestCase):
    @contextmanager
    def fake_lifetime(self, *, parent_error=None, image_manifest_error=False,
                      stream_only=False, native_close_error=None):
        with tempfile.TemporaryDirectory() as td, ExitStack() as stack:
            root=Path(td);output=root/'lifetime';output.mkdir()
            for name in IMAGE_SOURCES:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fake pinned '+name)
            parent_source=root/'candidate-code.py';parent_source.write_text('fake parent code')
            asset=root/'candidate.pt';asset.write_bytes(b'not model weights; CPU pin fixture')
            receipt={'sources':[{'path':'fake-catalog-source','sha256':'a'*64}]}
            state=SimpleNamespace(root=root,output=output,native_closes=[],objects=[],parent_calls=[])
            def fake_parent(engine,**kwargs):
                state.objects.append(engine);state.parent_calls.append(kwargs)
                engine.output=output;engine.startup={'fakeCpuLifetime':True}
                engine.stream_context=MagicMock()
                if not stream_only:
                    engine.bb=object();engine.before_hashes={'fake-tensor':'b'*64}
                engine.binding=SimpleNamespace(manifest={} if image_manifest_error else
                    {'imageDelivery':{'version':'fake-v3-runtime-projection','learnedImageFeatures':False}},
                    manifest_sha='c'*64,assert_current_sources=Mock(return_value=True))
                engine.session=SimpleNamespace(snapshot={'receipt':receipt},integrity_receipt=Mock(return_value={
                    'v5ServingExportUnchanged':True,'v5HeadCheckpointUnchanged':True}))
                engine.candidate_sources={parent_source.name:candidate.sha(parent_source)}
                engine.copy_metadata={'assetSha256':{str(asset):candidate.sha(asset)}}
                if parent_error is not None:raise parent_error
            def fake_native_close(engine):
                if hasattr(engine,'_native_closed_receipt'):
                    return deepcopy(engine._native_closed_receipt)
                state.native_closes.append(engine)
                if native_close_error is not None:raise native_close_error
                result={'frozen927HashesUnchanged':True,'copyCheckpointUnchanged':True,'fakeNativeClosed':True}
                engine.stream_context.__exit__(None,None,None)
                engine._native_closed_receipt=deepcopy(result)
                (output/'shutdown.json').write_text(json.dumps(result))
                return result
            stack.enter_context(patch.object(image,'ROOT',root))
            stack.enter_context(patch.object(candidate,'HERE',root))
            stack.enter_context(patch.object(candidate.CandidateDemoEngine,'__init__',fake_parent))
            stack.enter_context(patch.object(ResidentDemoEngine,'close',fake_native_close))
            yield state

    def test_success_pins_all_image_sources_and_preserves_parent_checks_and_v5_integrity(self):
        with self.fake_lifetime() as fixture:
            rank_factory=object()
            engine=image.ImageCandidateDemoEngine(mode='parallel',rank_session_factory=rank_factory)
            self.assertEqual(set(engine.image_sources),set(IMAGE_SOURCES))
            self.assertIs(fixture.parent_calls[0]['catalog_binding_factory'],image.ImageDeliveryCatalogBinding)
            self.assertIs(fixture.parent_calls[0]['rank_session_factory'],rank_factory)
            startup=json.loads((fixture.output/'startup.json').read_text())
            self.assertEqual(startup['imageServingSourcesSha256'],engine.image_sources)
            self.assertEqual(startup['imageDeliveryManifestSha256'],'c'*64)
            engine._assert_candidate()
            engine.binding.assert_current_sources.assert_called_with(engine.session.snapshot['receipt'])
            self.assertTrue(engine.session.integrity_receipt.called)
            result=engine.close()
            self.assertEqual(len(fixture.native_closes),1)
            self.assertTrue(result['candidateAssetsUnchanged'])
            self.assertTrue(result['imageServingSourcesUnchanged'])
            self.assertTrue(result['v5HeadCheckpointUnchanged'])
            self.assertEqual(json.loads((fixture.output/'shutdown.json').read_text()),result)

    def test_helper_entry_v5_session_and_every_asset_script_drift_rejected_and_native_close_runs(self):
        for name in IMAGE_SOURCES:
            with self.subTest(source=name),self.fake_lifetime() as fixture:
                engine=image.ImageCandidateDemoEngine(mode='parallel')
                (fixture.root/name).write_text('changed after initialization')
                with self.assertRaisesRegex(ValueError,'Image serving source changed'):
                    engine._assert_candidate()
                with self.assertRaisesRegex(ValueError,'Image serving source changed'):
                    engine.close()
                self.assertEqual(fixture.native_closes,[engine])
                engine.stream_context.__exit__.assert_called_once()
                self.assertNotIn('imageServingSourcesUnchanged',json.loads((fixture.output/'shutdown.json').read_text()))

    def test_missing_lifetime_pin_fails_closed_but_native_close_still_runs(self):
        with self.fake_lifetime() as fixture:
            engine=image.ImageCandidateDemoEngine(mode='parallel')
            (fixture.root/IMAGE_SOURCES[0]).unlink()
            with self.assertRaises(FileNotFoundError):engine.close()
            self.assertEqual(fixture.native_closes,[engine])

    def test_parent_candidate_asset_and_code_checks_are_not_bypassed(self):
        for name in ('candidate.pt','candidate-code.py'):
            with self.subTest(source=name),self.fake_lifetime() as fixture:
                engine=image.ImageCandidateDemoEngine(mode='parallel')
                (fixture.root/name).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Candidate .* changed'):
                    engine._assert_candidate()
                with self.assertRaises(ValueError):engine.close()
                self.assertEqual(fixture.native_closes,[engine])

    def test_manifest_binding_and_v5_export_head_failure_never_prevent_native_teardown(self):
        for failure in ('binding','v5'):
            with self.subTest(failure=failure),self.fake_lifetime() as fixture:
                engine=image.ImageCandidateDemoEngine(mode='parallel')
                check=engine.binding.assert_current_sources if failure=='binding' else engine.session.integrity_receipt
                check.side_effect=ValueError('drift in '+failure)
                with self.assertRaisesRegex(ValueError,'drift in '+failure):engine.close()
                self.assertEqual(fixture.native_closes,[engine])
                engine.stream_context.__exit__.assert_called_once()

    def test_post_parent_image_initialization_failure_closes_native_and_keeps_original_error(self):
        with self.fake_lifetime(image_manifest_error=True) as fixture:
            with self.assertRaises(KeyError) as caught:image.ImageCandidateDemoEngine(mode='parallel')
            self.assertEqual(caught.exception.args,('imageDelivery',))
            self.assertEqual(fixture.native_closes,fixture.objects)
            receipt=json.loads((fixture.output/'initialization-failure.json').read_text())
            self.assertTrue(receipt['frozenHashCleanupConfirmed'])
            self.assertEqual(receipt['errorType'],'KeyError')

    def test_parent_partial_start_error_is_preserved_and_native_cleanup_bypasses_bad_pins(self):
        original=ValueError('rank session could not initialize')
        with self.fake_lifetime(parent_error=original) as fixture:
            with self.assertRaises(ValueError) as caught:image.ImageCandidateDemoEngine(mode='parallel')
            self.assertIs(caught.exception,original)
            self.assertEqual(fixture.native_closes,fixture.objects)
            receipt=json.loads((fixture.output/'initialization-failure.json').read_text())
            self.assertTrue(receipt['frozenHashCleanupConfirmed'])

    def test_stream_only_partial_start_exits_stream_without_frozen_hash_claim(self):
        original=ValueError('backbone construction failed before initial hash receipt')
        with self.fake_lifetime(parent_error=original,stream_only=True) as fixture:
            with self.assertRaises(ValueError) as caught:image.ImageCandidateDemoEngine(mode='parallel')
            self.assertIs(caught.exception,original)
            self.assertEqual(fixture.native_closes,[])
            engine=fixture.objects[0];engine.stream_context.__exit__.assert_called_once()
            args=engine.stream_context.__exit__.call_args.args
            self.assertIs(args[0],ValueError);self.assertIs(args[1],original)
            receipt=json.loads((fixture.output/'initialization-failure.json').read_text())
            self.assertFalse(receipt['frozenHashCleanupConfirmed'])

    def test_failed_native_start_cleanup_also_exits_stream_and_preserves_original_error(self):
        original=ValueError('original startup failure')
        with self.fake_lifetime(parent_error=original,native_close_error=OSError('hash read failed')) as fixture:
            with self.assertRaises(ValueError) as caught:image.ImageCandidateDemoEngine(mode='parallel')
            self.assertIs(caught.exception,original)
            engine=fixture.objects[0]
            engine.stream_context.__exit__.assert_called_once()
            receipt=json.loads((fixture.output/'initialization-failure.json').read_text())
            self.assertFalse(receipt['frozenHashCleanupConfirmed'])
            self.assertEqual(receipt['error'],'original startup failure')
            self.assertEqual(receipt['cleanupError']['type'],'OSError')

    def test_successful_native_close_receipt_is_idempotent_and_returned_as_copy(self):
        engine=object.__new__(ResidentDemoEngine)
        engine._native_closed_receipt={'frozen927HashesUnchanged':True,'nested':{'value':1}}
        first=ResidentDemoEngine.close(engine);first['nested']['value']=9
        self.assertEqual(ResidentDemoEngine.close(engine)['nested']['value'],1)

    def test_real_native_close_exits_fake_stream_on_hash_checkpoint_write_and_sync_errors(self):
        from . import semantic_copy_training as training
        from . import semantic_copy_feature_fit as fit
        generation=ModuleType('mlx_lm.generate');generation.generation_stream=object()
        for failure in ('hash','checkpoint','write','sync'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as td,ExitStack() as stack:
                engine=object.__new__(ResidentDemoEngine);engine.output=Path(td);engine.copy_run=Path(td)
                engine.copy_sha='copy-hash';engine.before_hashes={'layer':'hash'};engine.stream_context=MagicMock()
                mx=SimpleNamespace(synchronize=Mock(side_effect=OSError('sync failed') if failure=='sync' else None))
                engine.bb=SimpleNamespace(model=SimpleNamespace(language_model=object()),mx=mx,runtime_memory=lambda:{'fixture':True})
                stack.enter_context(patch.dict(sys.modules,{'mlx_lm':ModuleType('mlx_lm'),'mlx_lm.generate':generation}))
                stack.enter_context(patch.object(training,'parameter_hashes',return_value={'layer':'hash'},
                    side_effect=OSError('hash failed') if failure=='hash' else None))
                stack.enter_context(patch.object(fit,'digest',return_value='copy-hash',
                    side_effect=OSError('checkpoint failed') if failure=='checkpoint' else None))
                if failure=='write':stack.enter_context(patch.object(Path,'write_text',side_effect=OSError('write failed')))
                with self.assertRaises(OSError):ResidentDemoEngine.close(engine)
                mx.synchronize.assert_called_once_with(generation.generation_stream)
                engine.stream_context.__exit__.assert_called_once()
                self.assertTrue(engine._native_stream_exited)
                self.assertFalse(hasattr(engine,'_native_closed_receipt'))

    def test_image_entry_drift_during_generation_is_rejected_before_completed_events(self):
        from .store_catalog_v1 import DEFAULT_OUT
        from .test_store_candidate_engine_v1 import fixture_generation
        snapshot={'inputs':json.loads((DEFAULT_OUT/'model_inputs.json').read_text()),
            'externalMapping':json.loads((DEFAULT_OUT/'external_mapping.json').read_text()),
            'receipt':json.loads((DEFAULT_OUT/'IMPORT_RECEIPT.json').read_text())}
        with self.fake_lifetime() as fixture:
            engine=image.ImageCandidateDemoEngine(mode='parallel');engine.mode='parallel'
            engine.head=object();engine.feature_map={};engine.adapter=object();engine.copy_sha='fake-copy'
            engine.session.snapshot=snapshot;engine.session.products=snapshot['inputs']['products'];engine.session.head_sha='fake-rank'
            engine.session.rank_request=lambda *args,**kwargs:{'scores16':list(range(16))}
            def generate(_head,rows,_features):
                (fixture.root/IMAGE_SOURCES[0]).write_text('changed during generation')
                return fixture_generation(rows)
            events=[]
            with patch('experiments.ranker.pretrained.store_parallel_text_head_v1.generate',side_effect=generate), \
                    patch('experiments.ranker.pretrained.store_copy_wide_v2.guard_memory',return_value={'fixture':True}):
                with self.assertRaisesRegex(ValueError,'Image serving source changed'):
                    engine.personalize({'query':'browse'},'drift',0.,events.append,lambda:False)
            self.assertEqual([e['type'] for e in events],['rank'])
            receipt=json.loads((fixture.output/'requests/drift/receipt.json').read_text())
            self.assertEqual(receipt['status'],'failed')
            with self.assertRaises(ValueError):engine.close()
            self.assertEqual(fixture.native_closes,[engine])

    def test_startup_receipt_write_failure_still_cleans_native_and_keeps_error(self):
        original=Path.write_text;failure=OSError('startup receipt disk failure')
        def write(path,*args,**kwargs):
            if path.name=='startup.json':raise failure
            return original(path,*args,**kwargs)
        with self.fake_lifetime() as fixture,patch.object(Path,'write_text',write):
            with self.assertRaises(OSError) as caught:image.ImageCandidateDemoEngine(mode='parallel')
            self.assertIs(caught.exception,failure)
            self.assertEqual(fixture.native_closes,fixture.objects)
            self.assertTrue((fixture.output/'initialization-failure.json').is_file())

    def test_explicit_head_export_and_checksum_must_be_paired_before_any_server_or_worker(self):
        for optional in (['--rank-head-export','head-export.json'],['--rank-head-export-sha256','a'*64]):
            args=['image-service','--candidate-run','candidate','--integration-manifest','manifest',
                  '--output','unused-output',*optional]
            with self.subTest(optional=optional),patch.object(sys,'argv',args), \
                    patch.object(image,'ThreadingHTTPServer') as server,patch.object(image,'ModelWorker') as worker, \
                    patch('sys.stderr',new=io.StringIO()):
                with self.assertRaises(SystemExit) as caught:image.main()
                self.assertEqual(caught.exception.code,2)
                server.assert_not_called();worker.assert_not_called()

    def test_paired_explicit_v5_factory_reaches_engine_without_loading_worker_or_binding_socket(self):
        from . import store_rank_session_v5 as v5
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            output=Path(td)/'unused';factory=object();server=Mock();worker=Mock();worker.close.return_value=True
            args=['image-service','--candidate-run','candidate','--integration-manifest','manifest',
                  '--output',str(output),'--rank-head-export','head-export.json',
                  '--rank-head-export-sha256','a'*64]
            stack.enter_context(patch.object(sys,'argv',args))
            make=stack.enter_context(patch.object(v5,'make_v5_rank_session_factory',return_value=factory))
            stack.enter_context(patch.object(image,'ThreadingHTTPServer',return_value=server))
            make_worker=stack.enter_context(patch.object(image,'ModelWorker',return_value=worker))
            construct=stack.enter_context(patch.object(image,'ImageCandidateDemoEngine',return_value=object()))
            stack.enter_context(patch.object(image.signal,'signal'))
            stack.enter_context(patch('sys.stdout',new=io.StringIO()))
            image.main();make.assert_called_once_with(Path('head-export.json'),'a'*64)
            make_worker.call_args.args[0]()
            self.assertIs(construct.call_args.kwargs['rank_session_factory'],factory)
            self.assertEqual(construct.call_args.kwargs['width'],10)
            server.server_close.assert_called_once();worker.close.assert_called_once()

    def test_malformed_paired_v5_checksum_rejected_before_server_or_worker(self):
        args=['image-service','--candidate-run','candidate','--integration-manifest','manifest',
            '--output','unused-output','--rank-head-export','head-export.json',
            '--rank-head-export-sha256','BAD']
        with patch.object(sys,'argv',args),patch.object(image,'ThreadingHTTPServer') as server, \
                patch.object(image,'ModelWorker') as worker:
            with self.assertRaisesRegex(ValueError,'SHA256'):image.main()
            server.assert_not_called();worker.assert_not_called()


class CompactRuntimeBindingCpuTests(unittest.TestCase):
    @contextmanager
    def fake_binding(self):
        with tempfile.TemporaryDirectory() as td,ExitStack() as stack:
            root=Path(td)
            for name in set(binding.SOURCES)|set(binding.FRONTEND)|set(binding.EXTRA)|set(binding.VERIFICATION_SOURCES):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('cpu fixture '+name)
            summary={'sourceCount':87};assets={'version':'fake-assets','encoder':'fake','summary':summary}
            (root/binding.FULL_MANIFEST).write_text(json.dumps(assets))
            sources=[{'path':name,'sha256':binding.sha(root/name)} for name in binding.SOURCES]
            snapshot={'receipt':{'sources':sources,'outputs':{'model_inputs.json':{'canonicalSha256':'b'*64}}}}
            value={'version':binding.VERSION,'rendererOnlyExceptions':{},'auxiliarySourceExceptions':{},
                'apiFixtureAudit':{'currentFixturePersonaTruthPassedToInference':False,
                    'publicProductExportAndModelFactsExactlyUnchanged':True,'currentSha256':binding.sha(root/binding.API_FIXTURE)},
                'frontendSha256':{n:binding.sha(root/n) for n in set(binding.FRONTEND)|set(binding.EXTRA)},
                'verificationSourcesSha256':{n:binding.sha(root/n) for n in binding.VERIFICATION_SOURCES},
                'rawExportSha256':binding.RAW_EXPORT_SHA256,'modelInputSha256':'b'*64,
                'imageDelivery':{'summary':summary}}
            manifest=root/'image-manifest.json';manifest.write_text(json.dumps(value))
            stack.enter_context(patch.object(binding,'ROOT',root))
            stack.enter_context(patch.object(binding,'runtime_export',return_value={'fakePublicExport':True}))
            stack.enter_context(patch.object(binding,'digest',return_value=binding.RAW_EXPORT_SHA256))
            stack.enter_context(patch.object(binding,'load_snapshot',return_value=snapshot))
            stack.enter_context(patch.object(binding,'checked_assets',return_value=assets))
            yield SimpleNamespace(root=root,manifest=manifest,value=value,snapshot=snapshot,assets=assets)

    def test_v3_compact_runtime_sources_and_verification_inventory_are_required(self):
        with self.fake_binding() as fixture:
            active=binding.ImageDeliveryCatalogBinding(fixture.manifest)
            self.assertEqual(binding.VERSION,'actual-store-render-image-delivery.v3')
            self.assertIn('apps/storefront/src/product-image-runtime.json',binding.EXTRA)
            self.assertIn('apps/storefront/src/product-responsive-runtime.ts',binding.EXTRA)
            self.assertIn('scripts/prepare-product-runtime-projection.mjs',binding.VERIFICATION_SOURCES)
            self.assertTrue(active.assert_current_sources(fixture.snapshot['receipt']))
            for inventory,name in (('frontendSha256','apps/storefront/src/product-image-runtime.json'),
                                   ('verificationSourcesSha256','scripts/prepare-product-runtime-projection.mjs')):
                bad=deepcopy(fixture.value);del bad[inventory][name]
                fixture.manifest.write_text(json.dumps(bad))
                with self.assertRaises(StoreCatalogError):binding.ImageDeliveryCatalogBinding(fixture.manifest)

    def test_runtime_projection_helper_script_and_manifest_drift_fail_closed(self):
        sources=('apps/storefront/src/product-image-runtime.json',
            'apps/storefront/src/product-responsive-runtime.ts',*binding.VERIFICATION_SOURCES)
        for name in sources:
            with self.subTest(source=name),self.fake_binding() as fixture:
                active=binding.ImageDeliveryCatalogBinding(fixture.manifest)
                (fixture.root/name).write_text('changed since initialization')
                with self.assertRaisesRegex(StoreCatalogError,'source changed'):
                    active.assert_current_sources(fixture.snapshot['receipt'])
        with self.fake_binding() as fixture:
            active=binding.ImageDeliveryCatalogBinding(fixture.manifest)
            fixture.manifest.write_text(fixture.manifest.read_text()+'\n')
            with self.assertRaisesRegex(StoreCatalogError,'receipt changed'):
                active.assert_current_sources(fixture.snapshot['receipt'])

    def test_symlink_runtime_projection_rejected_even_when_bytes_match(self):
        with self.fake_binding() as fixture:
            active=binding.ImageDeliveryCatalogBinding(fixture.manifest)
            path=fixture.root/'apps/storefront/src/product-image-runtime.json'
            other=fixture.root/'same-bytes.json';other.write_bytes(path.read_bytes());path.unlink();path.symlink_to(other)
            with self.assertRaisesRegex(StoreCatalogError,'source changed'):
                active.assert_current_sources(fixture.snapshot['receipt'])

    def test_checked_assets_requires_compact_projection_verification_without_executing_scripts(self):
        with self.fake_binding() as fixture:
            # Call the original helper with every subprocess mocked; no Node/Git.
            for projection in ({'status':'verified','assets':87},{'status':'verified','assets':86}):
                with patch.object(binding.subprocess,'check_output',side_effect=[
                        json.dumps({'status':'verified'}),json.dumps({'status':'verified'}),json.dumps(projection)]) as checks:
                    if projection['assets']==87:
                        self.assertEqual(REAL_CHECKED_ASSETS(),fixture.assets)
                    else:
                        with self.assertRaisesRegex(ValueError,'compact runtime projection'):REAL_CHECKED_ASSETS()
                    self.assertEqual(checks.call_args_list[-1].args[0],
                        ['node','scripts/prepare-product-runtime-projection.mjs','--check'])


REAL_CHECKED_ASSETS=binding.checked_assets


if __name__=='__main__':
    unittest.main()
