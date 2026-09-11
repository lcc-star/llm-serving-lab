import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec=importlib.util.spec_from_file_location('policy',Path(__file__).parents[2]/'python/minisgl/scheduler/policy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
BatchPolicy=module.BatchPolicy


class Manager:
    def __init__(self,phase):
        self.phase=phase;self.available=True;self.calls=[]
    def schedule_next_batch(self,*args):
        self.calls.append(args)
        return SimpleNamespace(phase=self.phase) if self.available else None


class PolicyTests(unittest.TestCase):
    def setUp(self): self.p=Manager('prefill');self.d=Manager('decode')
    def test_default_matches_original_priority_and_budget(self):
        p=BatchPolicy()
        self.assertEqual(p.select(self.p,self.d,1024).phase,'prefill')
        self.assertEqual(self.p.calls,[(1024,)])
        self.assertEqual(self.d.calls,[])
    def test_decode_first_short_circuits(self):
        self.assertEqual(BatchPolicy('decode_first').select(self.p,self.d,10).phase,'decode')
        self.assertEqual(self.p.calls,[])
    def test_alternates_under_continuous_dual_demand(self):
        p=BatchPolicy('alternating')
        self.assertEqual([p.select(self.p,self.d,256).phase for _ in range(20)],['prefill','decode']*10)
    def test_blocked_prefill_falls_back_to_decode(self):
        self.p.available=False
        for name in module.SUPPORTED_POLICIES:
            self.assertEqual(BatchPolicy(name).select(self.p,self.d,256).phase,'decode')
    def test_finished_or_cancelled_decode_does_not_block_prefill(self):
        self.d.available=False
        for name in module.SUPPORTED_POLICIES:
            self.assertEqual(BatchPolicy(name).select(self.p,self.d,256).phase,'prefill')
    def test_idle_resets_and_no_batch_is_fabricated(self):
        p=BatchPolicy('alternating');p.select(self.p,self.d,1)
        self.p.available=self.d.available=False
        self.assertIsNone(p.select(self.p,self.d,1))
        self.p.available=self.d.available=True
        self.assertEqual(p.select(self.p,self.d,1).phase,'prefill')
    def test_one_phase_only_makes_progress(self):
        p=BatchPolicy('alternating');self.d.available=False
        self.assertEqual([p.select(self.p,self.d,1).phase for _ in range(4)],['prefill']*4)
    def test_invalid_policy(self):
        with self.assertRaises(ValueError): BatchPolicy('typo')
