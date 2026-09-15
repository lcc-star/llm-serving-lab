import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).resolve().parents[2]/'study/stage05_correctness_attribution/summarize_profiles.py'
spec=importlib.util.spec_from_file_location('profiles',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ChainTests(unittest.TestCase):
    def test_only_consecutive_prefill_with_waiting_decode_counts(self):
        def b(phase,start,end,waiting=1):return dict(phase=phase,start_ms=start,end_ms=end,decode_waiting=waiting)
        rows=[b('prefill',0,100,0),b('prefill',100,110),b('prefill',115,125),b('decode',125,130),b('prefill',130,145)]
        self.assertEqual(m.blocking_chain(rows),dict(count=2,start_ms=100,end_ms=125,span_ms=25))
    def test_no_blocking_prefill(self):
        self.assertEqual(m.blocking_chain([])['count'],0)
