import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('prepare',Path(__file__).parents[2]/'study/stage04_system_evaluation/prepare.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class PrepareTests(unittest.TestCase):
    def test_skip_empty_and_continuation(self):
        self.assertIsNone(m.first_prompt({'conversations':[]}))
        self.assertIsNone(m.first_prompt({'conversations':[{'from':'gpt','value':'answer'}]}))
        self.assertIsNone(m.first_prompt({'conversations':[{'from':'human','value':' '}]}))
    def test_preserve_system_and_first_user_only(self):
        r={'conversations':[{'from':'system','value':'instruction'}, {'from':'human','value':'question'}, {'from':'gpt','value':'answer'}]}
        self.assertEqual(m.first_prompt(r),[{'role':'system','content':'instruction'},{'role':'user','content':'question'}])
    def test_split_siblings_together(self):
        self.assertEqual(m.group_id({'id':'conversation_0'}),m.group_id({'id':'conversation_3'}))
        self.assertEqual(m.split_for_group(m.group_id({'id':'conversation_0'})),m.split_for_group(m.group_id({'id':'conversation_3'})))
    def test_length_bucket_boundaries(self):
        self.assertEqual([m.bucket(n) for n in [512,513,2047,2048]],['short','medium','medium','long'])
