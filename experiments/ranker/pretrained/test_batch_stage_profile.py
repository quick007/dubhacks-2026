"""CPU-only profiler accounting, restoration and fixed-denominator gates."""
import types
import unittest
from .batch_stage_profile import StageTrace,fixed_cases,summarize,CASE_IDS


class BatchStageTests(unittest.TestCase):
    def test_nested_exclusive_times_reconcile(self):
        trace=StageTrace(lambda stream:None,lambda:None)
        with trace.span('outer'):
            with trace.span('inner'):pass
        outer=next(e for e in trace.events if e['category']=='outer')
        self.assertAlmostEqual(sum(trace.totals.values()),outer['inclusiveMs'])
        self.assertTrue(all(e['exclusiveMs']>=0 for e in trace.events))

    def test_all_native_hooks_restore_after_exception(self):
        def operation(self,*args,**kwargs):return None
        batch=type('Batch',(),{n:operation for n in ('insert_segments','_next','_make_batch')})
        prompt=type('Prompt',(),{n:operation for n in ('extend','split','prompt','generate')})
        generation=type('Generation',(),{n:operation for n in ('_step','next','extend','filter','extract_cache')})
        native=types.SimpleNamespace(BatchGenerator=batch,PromptProcessingBatch=prompt,GenerationBatch=generation)
        saved={(cls,n):getattr(cls,n) for cls in (batch,prompt,generation) for n in cls.__dict__ if not n.startswith('__')}
        trace=StageTrace(lambda stream:None,lambda:None)
        with self.assertRaises(RuntimeError):
            with trace.install(native):raise RuntimeError('injected exception')
        for (cls,n),original in saved.items():self.assertIs(getattr(cls,n),original)

    def test_fixed_cases_and_failed_denominator(self):
        cases=[{'caseId':cid,'cards':n} for cid,n in zip(CASE_IDS,(1,3,6),strict=True)]
        self.assertEqual(fixed_cases(cases[::-1]),cases)
        with self.assertRaises(ValueError):fixed_cases(cases[:-1])
        with self.assertRaises(ValueError):fixed_cases(cases+[cases[0]])
        records=[{'arms':{name:{'status':'failed','quality':{'allFieldsFulfilled':True}} for name in ('plain256','trace256','plain512')}}]
        self.assertEqual(summarize(records)['plain256'],{'fixedCases':3,'completed':0,'failed':1,'unattempted':2,'validPages':0})


if __name__=='__main__':unittest.main()
