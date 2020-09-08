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

import os
import io
import unittest
import warnings
from shutil import copy

from horovod.runner.common.util import safe_shell_exec

from common import tempdir

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
    
    Compares output of .buildkite/gen_pipeline.sh with test/data/expected_buildkite_pipeline.yaml.
    To see the changes in the output, run the following in your Horovod project root:
    
        BUILDKITE_PIPELINE_SLUG=SLUG BUILDKITE_BRANCH=BRANCH .buildkite/gen-pipeline.sh > test/data/expected_buildkite_pipeline.yaml
    
    Then run `git diff` to see the changes in the generated pipeline YAML.
    Commit `test/data/expected_buildkite_pipeline.yaml` to get those changes into your PR.
    """
    def test_gen_pipeline(self):
        with open('data/expected_buildkite_pipeline.yaml', 'r') as f:
            lines = f.readlines()
            expected_pipeline = ''.join(lines)

        gen_pipeline_env = dict(BUILDKITE_PIPELINE_SLUG='SLUG', BUILDKITE_BRANCH='BRANCH')
        gen_pipeline_cmd = '../.buildkite/gen-pipeline.sh'

        exit_code, actual_pipeline, gen_pipeline_log = self._run(gen_pipeline_cmd, gen_pipeline_env)

        self.assertEqual(0, exit_code)
        self.assertEqual(expected_pipeline, actual_pipeline)
        self.assertEqual('DEBUG:root:commit = None\n'
                         'DEBUG:root:pr number = None\n'
                         'DEBUG:root:branch = BRANCH\n'
                         'DEBUG:root:default = None\n', gen_pipeline_log)

    """
    Tests .buildkite/gen-pipeline.sh with a commit that has no code changes.

    The .buildkite/gen-pipeline.sh script calls the co-located get_commit_files.py script.
    So we can mock the output of that Python script by copying gen-pipeline.sh into a temp
    directory and providing our own mock Python script.
    """
    def test_gen_pipeline_with_non_code_changes(self):
        with tempdir() as dir:
            tmp_gen_pipeline_sh = os.path.join(dir, 'gen-pipeline.sh')
            copy('../.buildkite/gen-pipeline.sh', tmp_gen_pipeline_sh)
            with open(os.path.join(dir, 'get_commit_files.py'), 'w') as py:
                py.write("print('.buildkite/get_commit_files.py')\n")
                py.write("print('.github/new_file')\n")
                py.write("print('docs/new_file')\n")
                py.write("print('new_file.md')\n")
                py.write("print('new_file.rst')\n")

            gen_pipeline_env = dict(BUILDKITE_PIPELINE_SLUG='SLUG', BUILDKITE_BRANCH='BRANCH')
            exit_code, actual_pipeline, gen_pipeline_log = self._run(tmp_gen_pipeline_sh, gen_pipeline_env)

            self.assertEqual(0, exit_code)
            self.assertEqual("steps:\n"
                             "- label: \':book: Build Docs\'\n"
                             "  command: 'cd /workdir/docs && pip install -r requirements.txt && make html'\n"
                             "  plugins:\n"
                             "  - docker#v3.1.0:\n"
                             "      image: 'python:3.7'\n"
                             "  timeout_in_minutes: 5\n"
                             "  retry:\n"
                             "    automatic: true\n"
                             "  agents:\n"
                             "    queue: cpu\n"
                             "- wait\n"
                             "- wait\n"
                             "- wait\n", actual_pipeline)
            self.assertEqual('', gen_pipeline_log)

    def do_test_gen_full_pipeline(self, cmd, env=dict()):
        with open('data/expected_buildkite_pipeline.yaml', 'r') as f:
            lines = f.readlines()
            expected_pipeline = ''.join(lines)

        cmd_env = dict(BUILDKITE_PIPELINE_SLUG='SLUG', BUILDKITE_BRANCH='BRANCH')
        cmd_env.update(env)
        exit_code, pipeline, log = self._run(cmd, cmd_env)

        self.assertEqual(0, exit_code)
        self.assertEqual(expected_pipeline, pipeline)
        self.assertEqual('', log)

    def test_gen_pipeline_with_code_changes(self):
        with tempdir() as dir:
            tmp_gen_pipeline_sh = os.path.join(dir, 'gen-pipeline.sh')
            copy('../.buildkite/gen-pipeline.sh', tmp_gen_pipeline_sh)

            for filename in ['.buildkite/gen-pipeline.sh',
                             'cmake/file',
                             'examples/file',
                             'horovod/file',
                             'test/file',
                             'Dockerfile.cpu',
                             '']:
                with open(os.path.join(dir, 'get_commit_files.py'), 'w') as py:
                    py.write("print('{}')".format(filename))

                self.do_test_gen_full_pipeline(tmp_gen_pipeline_sh)

    """
    Tests gen-pipeline.sh with no commit changes. Should generate the full pipeline.
    """
    def test_gen_pipeline_with_empty_changes(self):
        with tempdir() as dir:
            tmp_gen_pipeline_sh = os.path.join(dir, 'gen-pipeline.sh')
            copy('../.buildkite/gen-pipeline.sh', tmp_gen_pipeline_sh)

            with open(os.path.join(dir, 'get_commit_files.py'), 'w') as py:
                py.write("pass")

            self.do_test_gen_full_pipeline(tmp_gen_pipeline_sh)

    """
    Tests gen-pipeline.sh with no commit changes. Should generate the full pipeline.
    """
    def test_gen_pipeline_on_default_branch(self):
        with tempdir() as dir:
            tmp_gen_pipeline_sh = os.path.join(dir, 'gen-pipeline.sh')
            copy('../.buildkite/gen-pipeline.sh', tmp_gen_pipeline_sh)

            with open(os.path.join(dir, 'get_commit_files.py'), 'w') as py:
                py.write("print('.github/new_file')")

            env = dict(BUILDKITE_BRANCH='default', BUILDKITE_PIPELINE_DEFAULT_BRANCH='default')
            self.do_test_gen_full_pipeline(tmp_gen_pipeline_sh, env)
