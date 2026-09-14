import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'audit', Path(__file__).parents[2] / 'study/stage03_scheduling/audit_decisions.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AuditTests(unittest.TestCase):
    def events(self):
        return [dict(event='decision', policy='wait_time', wait_ms={'prefill': 0, 'decode': 60},
                     thresholds_ms={'prefill': 200, 'decode': 50},
                     overdue={'prefill': False, 'decode': True},
                     tracked_requests={'prefill': 1, 'decode': 1}, preferred='decode',
                     selected='decode', reason='decode_overdue', fallback=False, token_budget=256),
                dict(event='batch', phase='decode', input_tokens=1)]

    def test_matching_trace(self):
        self.assertEqual(module.audit(self.events())['selections_with_both_phases_present'],
                         {'decode': 1})

    def test_reject_incorrect_reason_and_selected_phase(self):
        for field, value in [('reason', 'below_thresholds'), ('selected', 'prefill')]:
            events = self.events()
            events[0][field] = value
            with self.assertRaises(AssertionError):
                module.audit(events)

    def test_reject_missing_batch(self):
        with self.assertRaises(AssertionError):
            module.audit(self.events()[:1])
