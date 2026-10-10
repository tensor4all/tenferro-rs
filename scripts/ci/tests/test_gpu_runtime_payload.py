"""Exercise hosted packaging and GPU reconstruction without network or a GPU."""
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.ci.tests.test_runpod_cost_contracts import step_script

ROOT = Path(__file__).resolve().parents[3]


class RuntimePayloadTests(unittest.TestCase):
    def test_packaged_dependencies_and_selected_sdk_reconstruct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / 'scripts/ci'
            scripts.mkdir(parents=True)
            (scripts / 'install_cutensor.sh').write_text('''#!/bin/bash
set -eu
printf "cutensor\\n" >> "$FIXTURE_INSTALL_LOG"
mkdir -p "$2/lib"
printf cutensor > "$2/lib/libcutensor.so.2"
printf unused-static > "$2/lib/libcutensor_static.a"
printf unused-multi-gpu > "$2/lib/libcutensorMg.so.2"
printf unused-mpi > "$2/lib/libcutensorMp.so.2"
''')
            (scripts / 'install_cuda_runtime_tree.sh').write_text('''#!/bin/bash
set -eu
printf "cuda-%s\\n" "$1" >> "$FIXTURE_INSTALL_LOG"
rm -rf "$2"
mkdir -p "$2/targets/x86_64-linux/lib" "$2/targets/x86_64-linux/include"
ln -s targets/x86_64-linux/lib "$2/lib64"
ln -s targets/x86_64-linux/include "$2/include"
for lib in nvrtc cublas cusolver cusparse; do
  printf '%s' "$1" > "$2/lib64/lib${lib}.so.12"
done
printf '%s' "$1" > "$2/include/cuda_runtime.h"
touch "$2/.seed-complete"
''')
            binaries = root / 'bin'
            binaries.mkdir()
            for name, body in {
                'cargo-nextest': '#!/bin/sh\nexit 0\n',
                'cargo': '#!/bin/sh\nexit 0\n',
                'rustup': f'#!/bin/sh\nprintf "%s\\n" "{binaries}/cargo"\n',
                'python3': f'''#!{os.sys.executable}
import os,pathlib,sys,zipfile
args=sys.argv[1:]
if args[:3]==['-m','pip','download']:
    dest=pathlib.Path(args[args.index('--dest')+1])
    for package in args[-3:]:
        with zipfile.ZipFile(dest/(package.split('==')[0]+'.whl'), 'w') as archive:
            archive.writestr('all-members.txt', package)
            archive.writestr('nvidia/cuda_nvcc/bin/ptxas', b'fixture executable')
elif args[0]=='scripts/ci/check_cuda_headers.py':
    root=pathlib.Path(args[args.index('--cuda-root')+1])
    assert (root/'.seed-complete').is_file()
    assert (root/'include/cuda_runtime.h').is_file()
elif args[0]=='-':
    os.execv(sys.executable, [sys.executable, *args])
else:
    raise SystemExit('unexpected command')
''',
            }.items():
                path = binaries / name
                path.write_text(body)
                path.chmod(0o755)
            environment = dict(os.environ, PATH=f'{binaries}:{os.environ["PATH"]}',
                               TENFERRO_CI_CACHE_ROOT=str(root / 'cache'),
                               CUTENSOR_VERSION='2.6.0.4', FIXTURE_INSTALL_LOG=str(root / 'install.log'),
                               JAX_CUDA12_PJRT_VERSION='0.10.2',
                               NVIDIA_CUDNN_CU12_VERSION='9.23.2.1',
                               NVIDIA_CUDA_NVCC_CU12_VERSION='12.9.86')
            result = subprocess.run(['bash', str(ROOT / 'scripts/ci/prepare_gpu_execution_payload.sh')],
                                    cwd=root, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for bundle, stem in [('runtime-common', 'runtime'), ('runtime-sdk-12.6', 'sdk'),
                                 ('runtime-sdk-12.8', 'sdk')]:
                directory = root / bundle
                pieces = [directory / f'{part:02}' / f'{stem}.part{part:02}' for part in range(5)]
                self.assertTrue(all(piece.stat().st_size > 0 for piece in pieces))
                data = b''.join(piece.read_bytes() for piece in pieces)
                checksum = (directory / '00' / f'{stem}.sha256').read_text().split()[0]
                self.assertEqual(hashlib.sha256(data).hexdigest(), checksum)
            common = root / 'common-unpacked'
            common.mkdir()
            archive = root / 'runtime-common/runtime.tar.zst'
            archive.write_bytes(b''.join((root / 'runtime-common' / f'{part:02}' /
                                         f'runtime.part{part:02}').read_bytes() for part in range(5)))
            subprocess.run(['tar', '--zstd', '-xf', str(archive), '-C', str(common)], check=True)
            self.assertEqual((common / 'bin/cargo').read_bytes(), (binaries / 'cargo').read_bytes())
            self.assertEqual((common / 'bin/cargo-nextest').read_bytes(),
                             (binaries / 'cargo-nextest').read_bytes())
            self.assertEqual(len(list((common / 'wheels').glob('*.whl'))), 3)
            self.assertEqual((common / 'opt/tenferro-ci/cutensor-2.6.0.4/lib/libcutensor.so.2').read_text(),
                             'cutensor')
            self.assertEqual(sorted(p.name for p in
                (common / 'opt/tenferro-ci/cutensor-2.6.0.4/lib').iterdir()), ['libcutensor.so.2'])
            self.assertTrue((root / 'cache/cutensor-2.6.0.4/lib/libcutensor_static.a').is_file())
            self.assertFalse((common / 'cuda-12.6').exists())
            self.assertFalse((common / 'cuda-12.8').exists())
            receiver = root / 'receiver'
            (receiver / 'sdk-transfer').mkdir(parents=True)
            for part in range(5):
                for file in (root / 'runtime-sdk-12.8' / f'{part:02}').iterdir():
                    shutil.copy(file, receiver / 'sdk-transfer' / file.name)
            child = (ROOT / '.github/workflows/runpod-gpu-execute.yml').read_text()
            install = step_script(child, 'Install selected CUDA SDK').replace(
                '/opt/ci-cost-runtime', str(receiver / 'installed'))
            result = subprocess.run(['bash', '-c', install], cwd=receiver,
                                    env=dict(environment, SELECTED_RUNTIME='12.8'),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            installed = receiver / 'installed/cuda-12.8'
            self.assertEqual((installed / 'include/cuda_runtime.h').read_text(), '12.8')
            for library in ('nvrtc', 'cublas', 'cusolver', 'cusparse'):
                self.assertEqual((installed / f'lib64/lib{library}.so').read_text(), '12.8')
            self.assertFalse((receiver / 'installed/cuda-12.6').exists())
            shutil.rmtree(receiver / 'installed')
            (receiver / 'sdk-transfer/sdk.part02').write_bytes(b'corrupt transfer')
            result = subprocess.run(['bash', '-c', install], cwd=receiver,
                                    env=dict(environment, SELECTED_RUNTIME='12.8'),
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((receiver / 'installed').exists())

            install_log = root / 'install.log'
            self.assertEqual(install_log.read_text().splitlines(),
                             ['cutensor', 'cuda-12.6', 'cuda-12.8'])
            # Reusing a complete restored tree must not invoke either installer.
            # Missing markers, libraries, and headers rebuild only the bad tier.
            scenarios = [
                ('prepared-wheels', None, []),
                ('warm', None, []),
                ('cutensor-miss', 'cutensor-2.6.0.4/lib/libcutensor.so.2', ['cutensor']),
                ('marker-miss', 'cuda-runtime-12.6/.seed-complete', ['cuda-12.6']),
                ('library-miss', 'cuda-runtime-12.8/lib64/libcublas.so.12', ['cuda-12.8']),
                ('header-miss', 'cuda-runtime-12.8/include/cuda_runtime.h', ['cuda-12.8']),
            ]
            for name, missing, expected_installs in scenarios:
                with self.subTest(cache_state=name):
                    install_log.write_text('')
                    if missing:
                        (root / 'cache' / missing).unlink()
                    work = root / name
                    work.mkdir()
                    shutil.copytree(scripts, work / 'scripts/ci')
                    result = subprocess.run(
                        ['bash', str(ROOT / 'scripts/ci/prepare_gpu_execution_payload.sh'),
                         *(['--unpack-pjrt-wheels'] if name == 'prepared-wheels' else [])],
                        cwd=work, env=environment, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(install_log.read_text().splitlines(), expected_installs)
                    if name == 'prepared-wheels':
                        payload = work / 'runtime-payload'
                        self.assertFalse((payload / 'wheels').exists())
                        for package, version in [('jax-cuda12-pjrt', '0.10.2'),
                                                 ('nvidia-cudnn-cu12', '9.23.2.1'),
                                                 ('nvidia-cuda-nvcc-cu12', '12.9.86')]:
                            unpacked = payload / 'wheels-unpacked' / package
                            self.assertEqual((unpacked / 'all-members.txt').read_text(),
                                             f'{package}=={version}')
                            executable = unpacked / 'nvidia/cuda_nvcc/bin/ptxas'
                            self.assertEqual(executable.read_bytes(), b'fixture executable')
                            self.assertTrue(os.access(executable, os.X_OK))


    def test_pjrt_consumer_accepts_prepared_and_legacy_payloads(self):
        workflow = (ROOT / '.github/workflows/runpod-gpu-execute.yml').read_text()
        script = step_script(workflow, 'Run OpenXLA PJRT E2E tests from archive')
        # Execute the real dependency preparation, stopping before GPU execution.
        script = script[:script.index('# Runtime-only plugin setup;')]
        for prepared in (False, True):
            with self.subTest(prepared=prepared), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payload = root / 'payload'
                wheels = payload / 'wheels'
                wheels.mkdir(parents=True)
                members = {'jax_plugins/xla_cuda12/xla_cuda_plugin.so': b'plugin',
                           'nvidia/cudnn/lib/libcudnn.so.9': b'cudnn',
                           'nvidia/cuda_nvcc/bin/ptxas': b'ptxas',
                           'nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc': b'bitcode'}
                wheel = wheels / 'fixture.whl'
                with zipfile.ZipFile(wheel, 'w') as archive:
                    for name, data in members.items():
                        archive.writestr(name, data)
                if prepared:
                    with zipfile.ZipFile(wheel) as archive:
                        archive.extractall(payload / 'wheels-unpacked/fixture')
                    shutil.rmtree(wheels)
                binary = root / 'bin'
                binary.mkdir()
                nm = binary / 'nm'
                nm.write_text('#!/bin/sh\nprintf "GetPjrtApi\\n"\n')
                nm.chmod(0o755)
                (root / 'pjrt-tests.tar.zst').touch()
                result = subprocess.run(['bash', '-c', script.replace(
                    '/opt/ci-cost-runtime', str(payload))], cwd=root,
                    env=dict(os.environ, RUNNER_TEMP=str(root / 'temp'),
                             PJRT_ARCHIVE='pjrt-tests.tar.zst',
                             PATH=f'{binary}:{os.environ["PATH"]}'),
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                target = (payload / 'wheels-unpacked' if prepared else
                          root / 'temp/openxla-pjrt') / 'fixture'
                for name, data in members.items():
                    self.assertEqual((target / name).read_bytes(), data)
                self.assertTrue(os.access(target / 'nvidia/cuda_nvcc/bin/ptxas', os.X_OK))
                if prepared:
                    self.assertFalse((root / 'temp').exists())

    def test_preparation_is_a_required_read_only_hosted_prerequisite(self):
        parent = (ROOT / '.github/workflows/runpod-gpu-test.yml').read_text()
        prep = (ROOT / '.github/workflows/runpod-gpu-runtime.yml').read_text()
        child = (ROOT / '.github/workflows/runpod-gpu-execute.yml').read_text()
        self.assertIn('cuda-archive, gpu-runtime]', parent)
        self.assertIn('record_result "gpu-runtime" "${RUNTIME_RESULT}"', parent)
        self.assertIn('runs-on: ubuntu-24.04', prep)
        self.assertIn('ref: ${{ inputs.tenferro_ref }}', prep)
        self.assertNotIn('github.workflow_sha', prep)
        self.assertIn('repository: ${{ env.TENFERRO_REPO }}', prep)
        self.assertIn('tenferro_ref: ${{ needs.authorize.outputs.tenferro_ref }}',
                      parent.split('  gpu-runtime:', 1)[1].split('  gpu-execution:', 1)[0])
        self.assertIn('persist-credentials: false', prep)
        self.assertIn('artifact_prefix: ${{ steps.runtime_ready.outputs.artifact_prefix }}', prep)
        self.assertGreater(prep.index('name: Publish prepared runtime identity'),
                           prep.index('name: Upload runtime cuda12.8 part 04'))
        self.assertNotIn('secrets.', prep)
        self.assertNotIn('actions/cache/save', prep)
        self.assertIn('test -n "${RUNTIME_ARTIFACT_PREFIX}"', child)
        gpu = child.split('  run-gpu-tests:', 1)[1].split('  cleanup-runpod:', 1)[0]
        self.assertNotIn('rust-toolchain@', gpu)
        self.assertNotIn('pip download', gpu)
        self.assertNotIn('install_cuda_runtime_tree.sh', gpu)
        self.assertIn('steps.select_cuda_runtime.outputs.runtime_version }}-part*', gpu)
        for bundle in ('common', 'cuda12.6', 'cuda12.8'):
            for part in range(5):
                self.assertIn(f'-{bundle}-part{part:02}', prep)

    def test_image_runner_registration_follows_cuda_launch_proof(self):
        child = (ROOT / '.github/workflows/runpod-gpu-execute.yml').read_text()
        startup = step_script(child, 'Provision cheapest compatible RunPod pod')
        self.assertLess(startup.index('env -u RUNNER_JIT_CONFIG python3'),
                        startup.index('exec ./run.sh --jitconfig'))
        self.assertIn('cd /home/runner', startup)
        self.assertNotIn('./bin/installdependencies.sh', startup)
