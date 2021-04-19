# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import io
import os
import sys
import unittest
import warnings
from shutil import copy

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import tempdir
from horovod.runner.common.util import safe_shell_exec

# NOTE: when this file is moved, adjust this path to `.buildkite`
BUILDKITE_ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, '.buildkite')
sys.path.append(BUILDKITE_ROOT)

from get_changed_code_files import is_code_file

GEN_PIPELINE_FNAME = os.path.join(BUILDKITE_ROOT, 'gen-pipeline.sh')


class BuildKiteTests(unittest.TestCase):
    """
    Tests for .buildkite directory
    """

    def __init__(self, *args, **kwargs):
        super(BuildKiteTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def _run(self, cmd, env):
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            exit_code = safe_shell_exec.execute(cmd, env=env, stdout=stdout, stderr=stderr)
            return exit_code, stdout.getvalue(), stderr.getvalue()
        finally:
            stdout.close()
            stderr.close()

    """
    Tests the generated buildkite pipeline.
    
    Compares output of .buildkite/gen-pipeline.sh with test/single/data/expected_buildkite_pipeline.yaml.
    To see the changes in the output, run the following in your Horovod project root:
    
        BUILDKITE_PIPELINE_SLUG=SLUG BUILDKITE_BRANCH=BRANCH PIPELINE_MODE=FULL .buildkite/gen-pipeline.sh > test/single/data/expected_buildkite_pipeline.yaml
    
    Then run `git diff` to see the changes in the generated pipeline YAML.
    Commit `test/single/data/expected_buildkite_pipeline.yaml` to get those changes into your PR.
    """
    def test_gen_full_pipeline(self):
        expected_filename = os.path.join(os.path.dirname(__file__), 'data/expected_buildkite_pipeline.yaml')
        with open(expected_filename, 'r') as f:
            lines = f.readlines()
            expected_pipeline = ''.join(lines)

        gen_pipeline_env = dict(BUILDKITE_PIPELINE_SLUG='SLUG', BUILDKITE_BRANCH='BRANCH', PIPELINE_MODE='FULL')
        gen_pipeline_cmd = GEN_PIPELINE_FNAME

        exit_code, actual_pipeline, gen_pipeline_log = self._run(gen_pipeline_cmd, gen_pipeline_env)

        self.assertEqual(0, exit_code)
        self.assertEqual('WARNING:root:no commit (None) or default branch (None) given\n', gen_pipeline_log)
        self.assertEqual(expected_pipeline, actual_pipeline)

    def test_gen_gpu_pipeline(self):
        expected_filename = os.path.join(os.path.dirname(__file__), 'data/expected_buildkite_gpu_pipeline.yaml')
        with open(expected_filename, 'r') as f:
            lines = f.readlines()
            expected_pipeline = ''.join(lines)

        gen_pipeline_env = dict(BUILDKITE_PIPELINE_SLUG='SLUG', BUILDKITE_BRANCH='BRANCH', PIPELINE_MODE='GPU FULL')
        gen_pipeline_cmd = GEN_PIPELINE_FNAME

        exit_code, actual_pipeline, gen_pipeline_log = self._run(gen_pipeline_cmd, gen_pipeline_env)

        self.assertEqual(0, exit_code)
        self.assertEqual('WARNING:root:no commit (None) or default branch (None) given\n', gen_pipeline_log)
        self.assertEqual(expected_pipeline, actual_pipeline)

    """
    Tests the given command produces the full pipeline.
    """
    def do_test_gen_full_pipeline(self, cmd, env=dict()):
        expected_filename = os.path.join(os.path.dirname(__file__), 'data/expected_buildkite_pipeline.yaml')
        with open(expected_filename, 'r') as f:
            lines = f.readlines()
            expected_pipeline = ''.join(lines)

        cmd_env = dict(BUILDKITE_PIPELINE_SLUG='SLUG', BUILDKITE_BRANCH='BRANCH', PIPELINE_MODE='FULL')
        cmd_env.update(env)
        exit_code, pipeline, log = self._run(cmd, cmd_env)

        self.assertEqual(0, exit_code)
        self.assertEqual('', log)
        self.assertEqual(expected_pipeline, pipeline)

    """
    Tests code changes produces the full pipeline.

    The .buildkite/gen-pipeline.sh script calls the co-located get_changed_code_files.py script.
    So we can mock the output of that Python script by copying gen-pipeline.sh into a temp
    directory and providing our own mock Python script.
    """
    def test_gen_pipeline_with_code_changes(self):
        with tempdir() as dir:
            tmp_gen_pipeline_sh = os.path.join(dir, 'gen-pipeline.sh')
            copy(GEN_PIPELINE_FNAME, tmp_gen_pipeline_sh)

            for filename in ['.buildkite/gen-pipeline.sh',
                             'cmake/file',
                             'examples/file',
                             'horovod/file',
                             'test/file',
                             'Dockerfile.cpu']:
                with open(os.path.join(dir, 'get_changed_code_files.py'), 'w') as py:
                    py.write("print('{}')".format(filename))

                self.do_test_gen_full_pipeline(tmp_gen_pipeline_sh)

    """
    Tests non-code changes produces the short pipeline.

    The .buildkite/gen-pipeline.sh script calls the co-located get_changed_code_files.py script.
    So we can mock the output of that Python script by copying gen-pipeline.sh into a temp
    directory and providing our own mock Python script.
    """
    def test_gen_pipeline_with_non_code_changes(self):
        with tempdir() as dir:
            tmp_gen_pipeline_sh = os.path.join(dir, 'gen-pipeline.sh')
            copy(GEN_PIPELINE_FNAME, tmp_gen_pipeline_sh)

            with open(os.path.join(dir, 'get_changed_code_files.py'), 'w') as py:
                py.write("pass")

            gen_pipeline_env = dict(BUILDKITE_PIPELINE_SLUG='SLUG', BUILDKITE_BRANCH='BRANCH')
            exit_code, actual_pipeline, gen_pipeline_log = self._run(tmp_gen_pipeline_sh, gen_pipeline_env)

            self.assertEqual(0, exit_code)
            self.assertEqual('', gen_pipeline_log)
            self.assertEqual('steps:\n'
                             '- wait\n'
                             '- wait\n'
                             '- wait\n', actual_pipeline)

    """
    Tests no changed code files on master produces the full pipeline.
    """
    def test_gen_pipeline_on_default_branch(self):
        with tempdir() as dir:
            tmp_gen_pipeline_sh = os.path.join(dir, 'gen-pipeline.sh')
            copy(GEN_PIPELINE_FNAME, tmp_gen_pipeline_sh)

            with open(os.path.join(dir, 'get_changed_code_files.py'), 'w') as py:
                py.write("pass")

            env = dict(BUILDKITE_BRANCH='default', BUILDKITE_PIPELINE_DEFAULT_BRANCH='default')
            self.do_test_gen_full_pipeline(tmp_gen_pipeline_sh, env)

    """
    Tests a failing get_changed_code_files.py script produces the full pipeline.

    The .buildkite/gen-pipeline.sh script calls the co-located get_changed_code_files.py script.
    So we can mock the output of that Python script by copying gen-pipeline.sh into a temp
    directory and providing our own mock Python script.
    """
    def test_gen_pipeline_with_failing_py(self):
        with tempdir() as dir:
            tmp_gen_pipeline_sh = os.path.join(dir, 'gen-pipeline.sh')
            copy(GEN_PIPELINE_FNAME, tmp_gen_pipeline_sh)

            with open(os.path.join(dir, 'get_changed_code_files.py'), 'w') as py:
                py.write('import sys\n')
                py.write('sys.exit(1)')

            self.do_test_gen_full_pipeline(tmp_gen_pipeline_sh)

    """
    Tests .buildkite/get_changed_code_files.py identifies files as non-code files.
    """
    def test_get_changed_code_files_with_non_code_files(self):
        for file in ['.buildkite/get_changed_code_files.py',
                     '.github/new_file',
                     'docs/new_file',
                     'new_file.md',
                     'new_file.rst']:
            self.assertFalse(is_code_file(file), file)

    """
    Tests .buildkite/get_changed_code_files.py identifies files as code files.
    """
    def test_get_changed_code_files_with_code_files(self):
        for file in ['.buildkite/gen-pipeline.sh',
                     'cmake/file',
                     'examples/file',
                     'horovod/file',
                     'test/file',
                     'Dockerfile.cpu']:
            self.assertTrue(is_code_file(file), file)

    def test_empty_pipeline(self):
        expected_pipeline = ('steps:\n'
                             '- wait\n'
                             '- wait\n'
                             '- wait\n')

        gen_pipeline_env = dict(BUILDKITE_PIPELINE_SLUG='SLUG', BUILDKITE_BRANCH='BRANCH', PIPELINE_MODE='')
        gen_pipeline_cmd = GEN_PIPELINE_FNAME

        exit_code, actual_pipeline, gen_pipeline_log = self._run(gen_pipeline_cmd, gen_pipeline_env)

        self.assertEqual(0, exit_code)
        self.assertEqual('WARNING:root:no commit (None) or default branch (None) given\n', gen_pipeline_log)
        self.assertEqual(expected_pipeline, actual_pipeline)
