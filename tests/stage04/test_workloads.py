import importlib.util
from pathlib import Path
import unittest
import subprocess
import sys
import os

spec=importlib.util.spec_from_file_location('workloads',Path(__file__).parents[2]/'study/stage04_system_evaluation/workloads.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.pool=[dict(bucket=kind,input_ids=[i]*length) for kind,length in [('short',64),('long',2048)] for i in range(100,140)]
    def test_six_scenarios_repeatable_and_sorted(self):
        for s in m.SCENARIOS:
            a=m.make_workload(self.pool,s,24,2,42);b=m.make_workload(self.pool,s,24,2,42)
            self.assertEqual(a,b)
            self.assertEqual(len(a),24)
            self.assertEqual(len({r['uid'] for r in a}),24)
            self.assertEqual(a,sorted(a,key=lambda r:(r['arrival'],r['uid'])))
    def test_rate_changes_arrivals_not_prompts(self):
        for s in m.SCENARIOS:
            a=m.make_workload(self.pool,s,24,1,42);b=m.make_workload(self.pool,s,24,2,42)
            self.assertEqual({r['uid']:r['input_ids'] for r in a},{r['uid']:r['input_ids'] for r in b})
    def test_seed_changes_workload(self):
        a=m.make_workload(self.pool,'mixed_steady',24,2,42);b=m.make_workload(self.pool,'mixed_steady',24,2,43)
        self.assertNotEqual(m.workload_hash(a),m.workload_hash(b))
    def test_background_decode_output_and_long_bursts(self):
        a=m.make_workload(self.pool,'decode_new',24,2,42)
        self.assertTrue(all(r['arrival']==0 and r['max_tokens']==512 for r in a if r['uid']<4))
        b=m.make_workload(self.pool,'long_burst',24,2,42)
        arrivals=[r['arrival'] for r in b if r['kind']=='long']
        self.assertEqual(arrivals[:4],[.3]*4)
    def test_capacity_and_invalid_input(self):
        self.assertEqual(m.page_capacity('kv_tight'),4608)
        with self.assertRaises(ValueError):m.make_workload(self.pool,'short_only',7,1,0)

    def test_reproducible_across_python_hash_seeds(self):
        code = """
import importlib.util
from pathlib import Path
p=Path('study/stage04_system_evaluation/workloads.py')
s=importlib.util.spec_from_file_location('w',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
pool=[dict(bucket=k,input_ids=[i]*n) for k,n in [('short',64),('long',2048)] for i in range(100,140)]
print([m.workload_hash(m.make_workload(pool,scenario,24,2,42)) for scenario in m.SCENARIOS])
"""
        results=[subprocess.check_output([sys.executable,'-c',code],
                 cwd=Path(__file__).parents[2],env=dict(os.environ,PYTHONHASHSEED=str(seed)))
                 for seed in range(4)]
        self.assertTrue(all(result==results[0] for result in results))
