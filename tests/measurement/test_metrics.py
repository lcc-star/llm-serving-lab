import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('metrics', Path(__file__).parents[2]/'study/stage01_measurement/metrics.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class MetricsTests(unittest.TestCase):
    def test_known_timeline(self):
        events = [dict(event='request',uid=0,arrival=0.,received=.1),
                  dict(event='batch',t=.2,uids=[0]),
                  dict(event='token',uid=0,t=.3,finished=False),
                  dict(event='token',uid=0,t=.5,finished=True)]
        result = metrics.summarize(events)
        assert result['ttft_s']['p50'] == .3
        self.assertAlmostEqual(result['itl_s']['p50'], .2)
        assert result['output_tokens_per_s'] == 4
        assert result['arrival_to_first_schedule_s']['p50'] == .2
        assert result['completed'] == 1


    def test_empty_and_incomplete(self):
        assert metrics.summarize([])['output_tokens_per_s'] == 0
        result = metrics.summarize([dict(event='request',uid=0,arrival=0,received=0)])
        assert result['incomplete'] == 1
        assert result['ttft_s']['p99'] is None


    def test_invalid_time_order(self):
        with self.assertRaises(AssertionError):
            metrics.summarize([dict(event='request',uid=0,arrival=1,received=0)])


    def test_single_token_no_itl_and_duplicate_finish(self):
        events = [dict(event='request',uid=0,arrival=0,received=0),
                  dict(event='batch',t=0,uids=[0]), dict(event='token',uid=0,t=1,finished=True)]
        assert metrics.summarize(events)['itl_s']['count'] == 0
        with self.assertRaises(AssertionError):
            metrics.summarize(events + [dict(event='token',uid=0,t=2,finished=True)])
