"""CPU-only checks for the CompBench CUDA build preflight."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from clover.evaluate import t21_comp_bench as compbench


class CompBenchSetupTests(unittest.TestCase):
    def test_toolkit_version_check(self):
        for toolkit_version, torch_version in [('12.4', '12.4'), ('12.4', '11.7')]:
            with self.subTest(toolkit=toolkit_version, torch=torch_version), TemporaryDirectory() as tmp:
                nvcc = Path(tmp) / 'bin/nvcc'
                nvcc.parent.mkdir()
                nvcc.touch()
                responses = [
                    SimpleNamespace(stdout=json.dumps({'cuda_home': tmp, 'torch_cuda': torch_version})),
                    SimpleNamespace(stdout=f'Cuda compilation tools, release {toolkit_version}, V12.4.131'),
                ]
                with patch.object(compbench.subprocess, 'run', side_effect=responses) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    if toolkit_version == torch_version:
                        compbench._check_compbench_cuda(Path('/backend/python'), {})
                    else:
                        with self.assertRaisesRegex(RuntimeError, '12.4.*11.7'):
                            compbench._check_compbench_cuda(Path('/backend/python'), {})
                self.assertEqual(run.call_args.args[0], [str(nvcc), '--version'])

    def test_missing_cuda_or_cpu_torch(self):
        for runtime in ({'cuda_home': None, 'torch_cuda': '12.4'},
                        {'cuda_home': '/cuda', 'torch_cuda': None}):
            with self.subTest(runtime=runtime), patch.object(compbench.subprocess, 'run',
                    return_value=SimpleNamespace(stdout=json.dumps(runtime))) as run:
                with self.assertRaisesRegex(RuntimeError, 'CUDA-enabled PyTorch and a CUDA toolkit'):
                    compbench._check_compbench_cuda(Path('/backend/python'), {})
                self.assertEqual(run.call_count, 1)

    def test_missing_compiler(self):
        with TemporaryDirectory() as tmp, patch.object(compbench.subprocess, 'run',
                return_value=SimpleNamespace(stdout=json.dumps({'cuda_home': tmp, 'torch_cuda': '12.4'}))):
            with self.assertRaisesRegex(RuntimeError, 'compiler is missing'):
                compbench._check_compbench_cuda(Path('/backend/python'), {})

    def test_mismatch_stops_before_detectron_build_or_weights_download(self):
        with patch.object(compbench, 'prepare_source'), \
             patch.object(compbench, '_check_compbench_cuda', side_effect=RuntimeError('mismatch')), \
             patch.object(compbench.subprocess, 'run') as run, \
             patch.object(compbench, 'fetch') as fetch:
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                compbench.setup_compbench(install_runtime=True)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][:3], ['uv', 'sync', '--locked'])
        fetch.assert_not_called()

    def test_setup_rebuilds_detectron_against_synced_torch(self):
        with patch.object(compbench, 'prepare_source', return_value=Path('/upstream')), \
             patch.object(compbench, '_check_compbench_cuda') as check, \
             patch.object(compbench.subprocess, 'run') as run, \
             patch.object(compbench, 'fetch'), contextlib.redirect_stdout(io.StringIO()):
            compbench.setup_compbench(install_runtime=True)
        check.assert_called_once()
        build = run.call_args_list[1]
        self.assertIn('--no-cache', build.args[0])
        self.assertIn('--no-build-isolation', build.args[0])
        index = build.args[0].index('--reinstall-package')
        self.assertEqual(build.args[0][index + 1], 'detectron2')
        self.assertEqual(build.kwargs['env']['FORCE_CUDA'], '1')
        self.assertEqual(build.kwargs['env']['MAX_JOBS'], '2')


if __name__ == '__main__':
    unittest.main()
