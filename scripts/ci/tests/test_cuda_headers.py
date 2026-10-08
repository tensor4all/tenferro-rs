import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.ci.check_cuda_headers import HEADERS, check


class HeaderTests(unittest.TestCase):
    @patch('scripts.ci.check_cuda_headers.ctypes.CDLL')
    def test_missing_headers_fail_before_loading_libraries(self, load):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'cuda_runtime.h'):
                check(Path(directory))
        load.assert_not_called()

    @patch('scripts.ci.check_cuda_headers.ctypes.CDLL')
    def test_compile_uses_bundled_include_paths_and_releases_program(self, load):
        nvrtc = MagicMock()
        nvrtc.nvrtcCreateProgram.return_value = 0
        nvrtc.nvrtcCompileProgram.return_value = 0
        load.side_effect = [MagicMock(), nvrtc]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for header in HEADERS:
                path = root / 'include' / ('cccl' if header.startswith('cuda/') else '') / header
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / 'lib64').mkdir()
            (root / 'lib64/libnvrtc-builtins.so.12.8').touch()
            check(root)
            options = list(nvrtc.nvrtcCompileProgram.call_args.args[2])
            self.assertIn(f'--include-path={root / "include"}'.encode(), options)
            self.assertIn(f'--include-path={root / "include/cccl"}'.encode(), options)
            nvrtc.nvrtcDestroyProgram.assert_called_once()
            nvrtc.nvrtcCompileProgram.return_value = 6
            load.side_effect = [MagicMock(), nvrtc]
            with self.assertRaisesRegex(ValueError, 'compilation failed'):
                check(root)
            self.assertEqual(nvrtc.nvrtcDestroyProgram.call_count, 2)


if __name__ == '__main__':
    unittest.main()
