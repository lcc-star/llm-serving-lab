"""A running request must stop referencing duplicate pages returned to the allocator."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch
from minisgl.scheduler.cache import CacheManager


class LiveRemapTests(unittest.TestCase):
    def test_running_prefix_points_to_canonical_pages_and_frees_original_duplicate(self):
        cm=CacheManager.__new__(CacheManager)
        cm.page_table=torch.tensor([[4,5,6,7]],dtype=torch.int32)
        old=SimpleNamespace(cached_len=2)
        new=SimpleNamespace(cached_len=4,get_matched_indices=lambda:torch.tensor([4,5,8,7],dtype=torch.int32))
        cm.prefix_cache=SimpleNamespace(insert_prefix=lambda ids,pages:(3,new))
        cm.unlock=Mock();cm.lock=Mock();freed=[]
        cm._free=lambda pages:freed.append(pages) # retain views as lazy_free_region does
        req=SimpleNamespace(input_ids=torch.tensor([10,11,12,13]),cached_len=4,table_idx=0,cache_handle=old)
        cm.cache_req(req,finished=False)
        self.assertTrue(torch.equal(cm.page_table[0],torch.tensor([4,5,8,7])))
        self.assertEqual(freed[0].tolist(),[6])
        self.assertIs(req.cache_handle,new)
        cm.lock.assert_called_once_with(new)
