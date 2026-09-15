"""CPU regressions for real scheduler result handling with controlled in-flight batches."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import torch
from minisgl.core import Req, SamplingParams
from minisgl.message import AbortBackendMsg
from minisgl.scheduler.scheduler import Scheduler


def fixture(ignore_eos=True):
    scheduler=Scheduler.__new__(Scheduler)
    req=Req(input_ids=torch.tensor([10,11],dtype=torch.int32),table_idx=0,cached_len=1,
            output_len=3,uid=0,sampling_params=SamplingParams(ignore_eos=ignore_eos),cache_handle=None)
    scheduler._inflight_reqs={req:2};scheduler._terminal_reqs=set();scheduler.eos_token_id=99
    scheduler.prefill_manager=SimpleNamespace(abort_req=Mock(return_value=None))
    scheduler.decode_manager=SimpleNamespace(remove_req=Mock(),abort_req=Mock(return_value=None))
    scheduler.cache_manager=SimpleNamespace(lazy_free_region=nullcontext,cache_req=Mock())
    scheduler.batch_policy=SimpleNamespace(observe=Mock())
    scheduler._free_req_resources=Mock()
    scheduler.replies=[]
    scheduler.send_result=lambda replies:scheduler.replies.extend(replies)
    return scheduler,req


def result(req,token):
    batch=SimpleNamespace(reqs=[req],is_prefill=False)
    return (SimpleNamespace(batch=batch),(None,torch.tensor([token]),SimpleNamespace(synchronize=lambda:None)))


class OverlapLifecycleTests(unittest.TestCase):
    def setUp(self):
        logger_patch=patch("minisgl.scheduler.scheduler.logger")
        logger_patch.start()
        self.addCleanup(logger_patch.stop)

    def test_eos_discards_speculative_output_and_defers_free(self):
        s,r=fixture(False)
        s._process_last_data(result(r,99))
        self.assertEqual(len(s.replies),1);self.assertTrue(s.replies[0].finished)
        s._free_req_resources.assert_not_called()
        s._process_last_data(result(r,42))
        self.assertEqual(len(s.replies),1)
        s._free_req_resources.assert_called_once_with(r)
        self.assertFalse(s._inflight_reqs);self.assertFalse(s._terminal_reqs)

    def test_length_uses_delivered_host_tokens_not_advanced_device_length(self):
        s,r=fixture()
        r.append_host(torch.tensor([12]));r.device_len=r.max_device_len
        self.assertFalse(r.can_decode)
        s._process_last_data(result(r,13))
        self.assertFalse(s.replies[-1].finished)
        s._free_req_resources.assert_not_called()
        s._process_last_data(result(r,14))
        self.assertTrue(s.replies[-1].finished)
        self.assertEqual(len(r.input_ids),r.max_device_len)
        s._free_req_resources.assert_called_once_with(r)

    def test_abort_finds_final_batch_outside_decode_manager(self):
        s,r=fixture();s._inflight_reqs={r:1}
        s._process_one_msg(AbortBackendMsg(uid=0))
        s._free_req_resources.assert_not_called()
        s._process_last_data(result(r,42))
        self.assertFalse(s.replies)
        s._free_req_resources.assert_called_once_with(r)
        self.assertFalse(s._terminal_reqs)

    def test_duplicate_abort_does_not_double_release(self):
        s,r=fixture();s._inflight_reqs={r:1}
        s._process_one_msg(AbortBackendMsg(uid=0));s._process_one_msg(AbortBackendMsg(uid=0))
        s._process_last_data(result(r,42))
        s._free_req_resources.assert_called_once_with(r)
