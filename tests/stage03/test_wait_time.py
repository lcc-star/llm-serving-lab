"""Deterministic clock tests; no model, GPU, wall-clock sleeps or private admissions."""
from types import SimpleNamespace
import unittest
from test_policy import BatchPolicy, Manager


class WaitTimeTests(unittest.TestCase):
    def setUp(self):
        self.t = 0.0
        self.p, self.d = Manager('prefill'), Manager('decode')
        self.policy = BatchPolicy('wait_time', decode_wait_ms=125, prefill_wait_ms=250,
                                  clock=lambda: self.t)
        self.policy.observe(self.p, self.d)

    def select(self):
        return self.policy.select(self.p, self.d, 256)

    def test_below_threshold_retains_baseline(self):
        self.t = .124
        self.assertEqual(self.select().phase, 'prefill')
        self.assertEqual(self.policy.last_decision['reason'], 'below_thresholds')
        self.assertEqual(self.p.calls, [(256,)])

    def test_exact_decode_threshold(self):
        self.t = .125
        self.assertEqual(self.select().phase, 'decode')
        self.assertEqual(self.policy.last_decision['reason'], 'decode_overdue')

    def test_prefill_threshold_with_younger_decode(self):
        self.d.running_reqs = []
        self.policy.observe(self.p, self.d)
        self.t = .25
        self.d.running_reqs = [SimpleNamespace(uid=2)]
        self.assertEqual(self.select().phase, 'prefill')
        self.assertEqual(self.policy.last_decision['reason'], 'prefill_overdue')

    def test_both_overdue_choose_earlier_deadline(self):
        self.t = .5
        self.assertEqual(self.select().phase, 'decode')
        self.assertEqual(self.policy.last_decision['reason'], 'both_overdue')
        self.t = .75
        self.assertEqual(self.select().phase, 'prefill')
        self.assertEqual(self.policy.last_decision['reason'], 'both_overdue')

    def test_equal_deadline_prefers_decode(self):
        self.d.running_reqs = []
        self.policy.observe(self.p, self.d)
        self.t = .125
        self.d.running_reqs = [SimpleNamespace(uid=2)]
        self.policy.observe(self.p, self.d)
        self.t = .5
        self.assertEqual(self.select().phase, 'decode')

    def test_only_served_prefill_requests_reset(self):
        self.p.pending_list.append(SimpleNamespace(uid=3))
        self.policy.observe(self.p, self.d)
        # Emulate a budget that fits one chunk of the head request only.
        self.p.schedule_next_batch = lambda budget: SimpleNamespace(
            phase='prefill', reqs=self.p.pending_list[:1])
        self.t = .0625
        self.select()
        self.assertEqual(self.policy.wait_since['prefill'], {0: .0625, 3: 0})

    def test_new_arrivals_do_not_reset_existing_age(self):
        self.t = .0625
        self.p.pending_list.append(SimpleNamespace(uid=3))
        self.policy.observe(self.p, self.d)
        self.assertEqual(self.policy.wait_since['prefill'], {0: 0, 3: .0625})

    def test_resource_block_falls_back_without_resetting_blocked_wait(self):
        self.d.running_reqs = []
        self.policy.observe(self.p, self.d)
        self.t = .5
        self.d.running_reqs = [SimpleNamespace(uid=2)]
        self.p.available = False
        self.assertEqual(self.select().phase, 'decode')
        self.assertTrue(self.policy.last_decision['fallback'])
        self.assertEqual(self.policy.wait_since['prefill'][0], 0)
        self.p.available = True
        self.assertEqual(self.select().phase, 'prefill')

    def test_both_blocked_preserve_wait_until_actual_service(self):
        self.p.available = self.d.available = False
        self.t = .5
        self.assertIsNone(self.select())
        self.assertEqual(self.policy.wait_since, {'prefill': {0: 0}, 'decode': {1: 0}})

    def test_chunk_transition_completion_abort_and_uid_reuse(self):
        self.select()  # chunk remains in pending
        self.assertIn(0, self.policy.wait_since['prefill'])
        self.t = .0625
        self.p.pending_list = []
        self.d.running_reqs.append(SimpleNamespace(uid=0))
        self.policy.observe(self.p, self.d)
        self.assertEqual(self.policy.wait_since['prefill'], {})
        self.assertEqual(self.policy.wait_since['decode'][0], .0625)
        self.d.running_reqs = []
        self.policy.observe(self.p, self.d)
        self.assertEqual(self.policy.wait_since, {'prefill': {}, 'decode': {}})
        self.t = 100
        self.p.pending_list = [SimpleNamespace(uid=0)]
        self.policy.observe(self.p, self.d)
        self.assertEqual(self.policy.wait_since['prefill'][0], 100)

    def test_both_phases_receive_repeated_opportunities(self):
        # Keep both phases runnable, vary simulated non-preemptible batch duration.
        phases = []
        for i in range(200):
            self.t += (.03125, .0625, .25)[i % 3]
            phases.append(self.select().phase)
        for start in range(0, 200, 10):
            self.assertEqual(set(phases[start:start+10]), {'prefill', 'decode'})

    def test_empty_phase_never_becomes_overdue(self):
        self.p.pending_list = []
        self.p.available = False
        self.t = 100
        self.assertEqual(self.select().phase, 'decode')
        self.assertIsNone(self.policy.last_decision['wait_ms']['prefill'])
        self.assertFalse(self.policy.last_decision['overdue']['prefill'])

    def test_invalid_thresholds(self):
        for value in (0, -1, float('nan'), float('inf')):
            for field in ('decode_wait_ms', 'prefill_wait_ms'):
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    BatchPolicy('wait_time', **{field: value})
