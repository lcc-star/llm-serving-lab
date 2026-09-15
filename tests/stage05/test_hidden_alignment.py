import importlib.util
from pathlib import Path
import unittest
import torch

path = Path(__file__).resolve().parents[2] / 'study/stage05_correctness_attribution/summarize_hidden.py'
spec = importlib.util.spec_from_file_location('summarize_hidden', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def chunk(start, values):
    return dict(phase='prefill', start=start, end=start+len(values), tensors={'x':torch.tensor(values)})


class AlignmentTests(unittest.TestCase):
    def test_different_chunk_boundaries_same_logical_tensor(self):
        a = module.prefill_tensor([chunk(0,[1,2]),chunk(2,[3])], 'x')
        b = module.prefill_tensor([chunk(0,[1]),chunk(1,[2,3])], 'x')
        self.assertTrue(module.delta(a,b)['equal'])

    def test_gap_and_overlap_rejected(self):
        for start in (1,3):
            with self.assertRaises(AssertionError):
                module.prefill_tensor([chunk(0,[1,2]),chunk(start,[3])], 'x')

    def test_numeric_difference_count(self):
        result = module.delta(torch.tensor([1.,2.]),torch.tensor([1.,2.125]))
        self.assertEqual(result['different_elements'],1)
        self.assertEqual(result['max_abs'],.125)


class ReferenceAttentionTests(unittest.TestCase):
    def test_uniform_attention_and_gqa_head_groups(self):
        q = torch.zeros(1,4,1)
        k = torch.zeros(2,2,1)
        v = torch.tensor([[[2.],[10.]],[[4.],[14.]]])
        result = module.reference_attention(q,k,v)
        self.assertTrue(torch.equal(result,torch.tensor([[3.,3.,12.,12.]],dtype=torch.float64)))
