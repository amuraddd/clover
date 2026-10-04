import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from clover.evaluate.b2_evals import b2_evaluation as b2
from clover.evaluate.b2_run_evaluation import run_b2_evaluation


class B2NotebookFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Notebooks now import the production runner instead of defining a copy.
        cls.run_evaluation = staticmethod(run_b2_evaluation)

    def test_arguments_immediate_json_saves_and_no_return(self):
        self.assertEqual(list(inspect.signature(self.run_evaluation).parameters),
                         ['baselines','seeds','b2_run_name','images_per_prompt','fraction','rollout_chunk_size'])
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / 'plan.json'; plan.write_text('{}')
            folder = root / 'clover/evaluate/metrics/example'
            def stage(previous, result):
                def call(*args, **kwargs):
                    self.assertTrue((folder / f'{previous}.json').is_file())
                    return result
                return Mock(side_effect=call)
            frame = pd.DataFrame([{'baseline': 'sd15', 'score': .5}])
            prepare = Mock(return_value=plan)
            with patch.object(b2,'ROOT',root), patch.multiple(b2,
                prepare_b2_run=prepare,
                generate_b2_images=stage('b2_plan',{'sd15':'images.json'}),
                score_b2_fid=stage('b2_image_manifests',[{'fid':0.}]),
                score_b2_clip=stage('b2_fid',[{'clip':.3}]),
                caption_b2_images=stage('b2_clip',[{'caption':'a dog'}]),
                score_b2_bert=stage('b2_captions',[{'bert_f1':.6}]),
                score_b2_imagereward=stage('b2_bert',[{'imagereward':1.2}]),
                score_b2_dreamsim=stage('b2_imagereward',[{'dreamsim_diversity':.4}]),
                summarize_b2_scores=stage('b2_dreamsim',(frame,frame,frame))):
                result = self.run_evaluation(['sd15','ddpo'],[123,124,126],'example',10,.8)
            self.assertIsNone(result)
            self.assertEqual(prepare.call_args.args[0],['sd15','ddpo'])
            self.assertEqual(prepare.call_args.kwargs['seeds'],[123,124,126])
            self.assertEqual(prepare.call_args.kwargs['images_per_prompt'],10)
            self.assertEqual(prepare.call_args.kwargs['fraction'],.8)
            expected={'b2_plan','b2_image_manifests','b2_fid','b2_clip','b2_captions',
                      'b2_bert','b2_imagereward','b2_dreamsim','b2_per_image','b2_per_seed','b2_summary'}
            self.assertEqual({p.stem for p in folder.glob('*.json')},expected)
            self.assertEqual(json.loads((folder/'b2_fid.json').read_text()),[{'fid':0.}])

    def test_completed_scores_survive_later_stage_failure(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); plan=root/'plan.json'; plan.write_text('{}')
            with patch.object(b2,'ROOT',root), patch.multiple(b2,
                prepare_b2_run=Mock(return_value=plan),generate_b2_images=Mock(return_value={}),
                score_b2_fid=Mock(return_value=[{'fid':2.}]),
                score_b2_clip=Mock(side_effect=RuntimeError('test scoring failure'))):
                with self.assertRaisesRegex(RuntimeError,'test scoring failure'):
                    self.run_evaluation(['sd15','ddpo'],[123],'partial',10,.8)
            folder=root/'clover/evaluate/metrics/partial'
            self.assertEqual(json.loads((folder/'b2_fid.json').read_text()),[{'fid':2.}])
            self.assertFalse((folder/'b2_clip.json').exists())


if __name__ == '__main__':
    unittest.main()
