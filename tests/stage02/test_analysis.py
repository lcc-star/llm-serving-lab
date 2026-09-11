import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'study/stage02_prefill_interference'))
from analysis import analyze


class AnalysisTests(unittest.TestCase):
    def test_gap_explained_by_prefill(self):
        events=[dict(event='request',uid=0,arrival=0,received=0),
                dict(event='batch',phase='prefill',t=.01,uids=[0]),
                dict(event='token',uid=0,t=.1,finished=False),
                dict(event='execution',phase='prefill',start=.2,end=.5,input_tokens=4096),
                dict(event='batch',phase='decode',t=.55,uids=[0]),
                dict(event='token',uid=0,t=.6,finished=True)]
        r=analyze(events)['worst_short_gap']
        self.assertAlmostEqual(r['seconds'],.5)
        self.assertAlmostEqual(r['prefill_cpu_span_s'],.3)
        self.assertEqual(r['prefill_batches'],1)
        self.assertEqual(r['prefill_tokens'],4096)
        self.assertEqual(r['decode_batches_in_gap'],1)

    def test_no_prefill_inside_gap(self):
        events=[dict(event='request',uid=0,arrival=0,received=0),
                dict(event='batch',phase='prefill',t=0,uids=[0]),
                dict(event='execution',phase='prefill',start=0,end=.05,input_tokens=128),
                dict(event='token',uid=0,t=.1,finished=False),
                dict(event='token',uid=0,t=.2,finished=True)]
        self.assertEqual(analyze(events)['worst_short_gap']['prefill_batches'],0)
