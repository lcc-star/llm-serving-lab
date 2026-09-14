import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('stage4_analysis',Path(__file__).parents[2]/'study/stage04_system_evaluation/analysis.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class AnalysisTests(unittest.TestCase):
    def test_group_by_kind_not_uid_and_count_wait(self):
        rows=[dict(uid=10,kind='short',arrival=0,input_ids=[1]*64,max_tokens=2),
              dict(uid=0,kind='long',arrival=0,input_ids=[1]*2048,max_tokens=2)]
        events=[dict(event='request',uid=r['uid'],arrival=0,received=0) for r in rows]
        events += [dict(event='batch',phase='prefill',uids=[10,0],t=.1,pending=1,decode_running=0,kv_free_pages=100,kv_allocated_pages=2112)]
        events += [dict(event='token',uid=10,t=.2,finished=False),dict(event='token',uid=0,t=.3,finished=False),
                   dict(event='token',uid=10,t=.4,finished=True),dict(event='token',uid=0,t=.8,finished=True)]
        result=m.analyze(events,rows)
        self.assertAlmostEqual(result['short_itl_s']['p99'],.2)
        self.assertAlmostEqual(result['long_itl_s']['p99'],.5)
        self.assertEqual(result['errors'],0)
        self.assertEqual(result['max_pending'],1)
        self.assertAlmostEqual(result['max_first_schedule_wait_s'],.1)
