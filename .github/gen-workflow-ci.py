# Copyright 2021 Uber Technologies, Inc. All Rights Reserved.
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

import re
from typing import List, Dict, Set

import yaml
from yaml import Loader


def main():
    import subprocess
    import pathlib
    from collections import Counter, defaultdict

    # run gen-pipeline.sh to get full Buildkite pipeline
    path = pathlib.Path(__file__).parent
    script = path.joinpath('..', '.buildkite', 'gen-pipeline.sh').absolute()
    env = dict(
        PIPELINE_MODE='FULL',
        BUILDKITE_PIPELINE_SLUG='horovod',
        BUILDKITE_PIPELINE_DEFAULT_BRANCH='master',
        BUILDKITE_BRANCH='master'
    )
    proc = subprocess.run([script], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'Script exited with code {proc.returncode}: {script}')

    # parse the pipeline
    pipeline = yaml.load(proc.stdout, Loader=Loader)
    steps = pipeline.get('steps', [])

    images = [plugin['docker-compose#v3.5.0']['build']
              for step in steps if isinstance(step, dict) and 'label' in step
                                and step['label'].startswith(':docker: Build ')
              for plugin in step['plugins'] if 'docker-compose#v3.5.0' in plugin]

    cpu_tests = [(re.sub(r' \(test-.*', '', re.sub(':[^:]*: ', '', step['label'])),
                  step['command'],
                  step['timeout_in_minutes'],
                  plugin['docker-compose#v3.5.0']['run'])
                 for step in steps if isinstance(step, dict) and 'label' in step and 'command' in step
                 and not step['label'].startswith(':docker: Build ') and '-cpu-' in step['label']
                 for plugin in step['plugins'] if 'docker-compose#v3.5.0' in plugin]

    # we need to distinguish the two oneccl variants of some tests
    cpu_tests = [(label + (' [ONECCL OFI]' if 'mpirun_command_ofi' in command else (' [ONECCL MPI]' if 'mpirun_command_mpi' in command else '')),
                  command,
                  timeout,
                  image)
                 for label, command, timeout, image in cpu_tests]

    # check that labels are unique per image
    cardinalities = Counter([(label, image) for label, command, timeout, image in cpu_tests])
    conflicts = [(label, image, card) for (label, image), card in cardinalities.items() if card > 1]
    if conflicts:
        summary = '\n'.join([f'"{label}" for image "{image}"' for label, image, card in conflicts])
        raise RuntimeError(f'There are {len(conflicts)} duplicate test labels for images:\n{summary}')

    # commands for some labels may differ
    # we make their labels unique here
    label_commands = defaultdict(Counter)
    for label, command, timeout, image in cpu_tests:
        label_commands[label][command] += 1

    labels_with_multiple_commands = {label: c for label, c in label_commands.items() if len(c) > 1}
    new_labels_per_label_command = {(label, command): f'{label} {index+1}'
                                    for label, commands in labels_with_multiple_commands.items()
                                    for index, command in enumerate(commands)}

    cpu_tests = [(new_labels_per_label_command[(label, command)] if (label, command) in new_labels_per_label_command else label,
                  command,
                  timeout,
                  image)
                 for label, command, timeout, image in cpu_tests]

    # come up with test ids from test labels
    test_labels = {label for label, command, timeout, image in cpu_tests}
    test_id_per_label = [(label, re.sub('[^a-zA-Z0-9_]', '', re.sub('[ .]', '_', label)))
                         for label in test_labels]
    if len({id for label, id in test_id_per_label}) != len(test_labels):
        raise RuntimeError('Some test ids are not unique')
    test_id_per_label = dict(test_id_per_label)

    # collect tests per image
    tests_per_image = {image: {test_id_per_label[label]
                               for label, command, timeout, test_image in cpu_tests
                               if test_image == image}
                       for image in sorted(images)}

    # index tests by id
    tests = {test_id_per_label[label]: dict(label=label, command=command, timeout=timeout)
             for label, command, timeout, image in cpu_tests}

    def workflow_header() -> str:
        return (f'# Do not edit this file! It has been generated by .github/gen-workflow-ci.py\n'
                f'\n'
                f'name: CI\n'
                f'on:\n'
                f'  schedule:\n'
                f'    - cron: "0 12 * * *"\n'
                f'  push:\n'
                f'    branches:\n'
                f'      - master\n'
                f'    tags:\n'
                f'      - \'*\'\n'
                f'  pull_request:\n'
                f'\n')

    def jobs(*jobs: str) -> str:
        return 'jobs:\n' + '\n'.join(jobs)

    def validate_workflow_job() -> str:
        return (f'  validate-workflow:\n'
                f'    name: "Verify GitHub Workflow"\n'
                f'    runs-on: ubuntu-latest\n'
                f'    steps:\n'
                f'      - name: Checkout\n'
                f'        uses: actions/checkout@v2\n'
                f'      - name: Setup Python\n'
                f'        uses: actions/setup-python@v2\n'
                f'        with:\n'
                f'          python-version: 3.8\n'
                f'      - name: Check ci.yaml is up-to-date\n'
                f'        run: |\n'
                f'          echo "::group::pip install -r .github/requirements.txt"\n'
                f'          pip install -r .github/requirements.txt\n'
                f'          echo "::endgroup::"\n'
                f'          python .github/gen-workflow-ci.py\n'
                f'          if [[ $(git diff .github/workflows/ci.yaml | wc -l) -gt 0 ]]\n'
                f'          then\n'
                f'            echo "::error::Workflow file .github/workflows/ci.yaml is out-dated, please run .github/gen-workflow-ci.py and commit changes"\n'
                f'            exit 1\n'
                f'          fi\n'
                f'        shell: bash\n')

    def build_and_test_images(name: str,
                              images: List[str],
                              tests_per_image: Dict[str, Set[str]],
                              tests: Dict[str, Dict]) -> str:
        return (f'  {name}:\n'
                f'    name: "Build and Test (${{{{ matrix.image }}}})"\n'
                f'    needs: validate-workflow\n'
                f'    runs-on: ubuntu-latest\n'
                f'    strategy:\n'
                f'      max-parallel: {len(images)}\n'
                f'      fail-fast: false\n'
                f'      matrix:\n'
                f'        include:\n' +
                '\n'.join([f'          - image: {image}\n' +
                           f''.join([f'            {test}: true\n'
                                     for test in sorted(list(tests_per_image[image]))]) +
                           f'            build_timeout: {30 if "-cpu-" in image else 40}\n'
                           for image in sorted(images)
                           # oneccl does not compile on GitHub Workflows:
                           # https://github.com/horovod/horovod/issues/2846
                           if '-oneccl-' not in image]) +
                f'\n'
                f'    steps:\n'
                f'      - name: Checkout\n'
                f'        uses: actions/checkout@v2\n'
                f'        with:\n'
                f'          submodules: recursive\n'
                f'\n'
                f'      - name: Setup Python\n'
                f'        uses: actions/setup-python@v2\n'
                f'        with:\n'
                f'          python-version: 3.8\n'
                f'\n'
                f'      - name: Setup docker-compose\n'
                f'        run: pip install docker-compose\n'
                f'\n'
                f'      - name: Configure AWS credentials\n'
                f'        uses: aws-actions/configure-aws-credentials@v1\n'
                f'        continue-on-error: true\n'
                f'        with:\n'
                f'          aws-access-key-id: ${{{{ secrets.AWS_ACCESS_KEY_ID }}}}\n'
                f'          aws-secret-access-key: ${{{{ secrets.AWS_SECRET_ACCESS_KEY }}}}\n'
                f'          aws-region: us-east-1\n'
                f'\n'
                f'      - name: Login to Amazon ECR\n'
                f'        id: ecr\n'
                f'        continue-on-error: true\n'
                f'        uses: aws-actions/amazon-ecr-login@v1\n'
                f'\n'
                f'      - name: Add cache_from to docker-compose YAML\n'
                f'        # disabled as 1) the cache image is not been picked up and 2) image is too large for github workers\n'
                f'        if: false\n'
                f'        run: |\n'
                f'          mv docker-compose.test.yml docker-compose.test.yml.bak\n'
                f'          .github/add-cache-to-docker-compose.sh "${{{{ steps.ecr.outputs.registry }}}}" < docker-compose.test.yml.bak > docker-compose.test.yml\n'
                f'          git diff\n'
                f'        shell: bash\n'
                f'\n'
                f'      - name: Pull latest image\n'
                f'        if: false\n'
                f'        continue-on-error: true\n'
                f'        run: |\n'
                f'          docker pull ${{{{ steps.ecr.outputs.registry }}}}/buildkite:horovod-${{{{ matrix.image }}}}-latest\n'
                f'\n'
                f'      - name: Build\n'
                f'        id: build\n'
                f'        run: |\n'
                f'          .github/timeout-and-retry.sh ${{{{ matrix.build_timeout }}}}m 3 10 docker-compose -f docker-compose.test.yml build ${{{{ matrix.image }}}}\n'
                f'\n' +
                '\n'.join([f'      - name: "{test["label"]}"\n'
                           f'        if: always() && steps.build.outcome == \'success\' && matrix.{test_id}\n'
                           f'        run: |\n'
                           f'          mkdir -p artifacts/${{{{ matrix.image }}}}/{test_id}\n'
                           f'          docker-compose -f docker-compose.test.yml run -e GITHUB_ACTIONS --rm --volume "$(pwd)/artifacts/${{{{ matrix.image }}}}/{test_id}:/artifacts" ${{{{ matrix.image }}}} /bin/bash /horovod/.github/timeout-and-retry.sh {test["timeout"]}m 3 10 {test["command"]}\n'
                           f'        shell: bash\n'
                           for test_id, test in sorted(tests.items(), key=lambda test: test[0])]) +
                f'\n'
                f'      - name: Upload Test Results\n'
                f'        uses: actions/upload-artifact@v2\n'
                f'        if: always() && contains(matrix.image, \'-cpu-\')\n'
                f'        with:\n'
                f'          name: Unit Test Results - ${{{{ matrix.image }}}}\n'
                f'          path: artifacts/${{{{ matrix.image }}}}/**/*.xml\n')

    def trigger_buildkite_job(name: str, needs: List[str]) -> str:
        return (f'  {name}:\n'
                f'    name: "Build and Test (GPUs on Builtkite)"\n'
                f'    needs: [{",".join(needs)}]\n'
                f'    runs-on: ubuntu-latest\n'
                f'    # disabled for further testing workflow without triggering buildkite work\n'
                f'    if: >\n'
                f'      false &&\n'
                f'      ( github.event_name == \'push\' || github.event.pull_request.head.repo.full_name == github.repository )\n'
                f'\n'
                f'    steps:\n'
                f'      - name: Trigger Buildkite Pipeline\n'
                f'        id: build\n'
                f'        uses: EnricoMi/trigger-pipeline-action@master\n'
                f'        env:\n'
                f'          PIPELINE: "horovod/horovod"\n'
                # on "push" event, github.event.pull_request.head.ref will be empty
                # and trigger-pipeline-action falls back to github.ref
                f'          BRANCH: "${{{{ github.event.pull_request.head.ref }}}}"\n'
                f'          MESSAGE: "GPU Tests triggered by GitHub"\n'
                f'          BUILDKITE_API_ACCESS_TOKEN: ${{{{ secrets.BUILDKITE_TOKEN }}}}\n'
                f'          BUILD_ENV_VARS: "{{\\"PIPELINE_MODE\\": \\"GPU FULL\\"}}"\n'
                f'\n'
                f'      - name: Download Buildkite Artifacts\n'
                f'        uses: docker://ghcr.io/enricomi/download-buildkite-artifact-action:v1\n'
                f'        with:\n'
                f'          github_token: ${{{{ github.token }}}}\n'
                f'          buildkite_token: ${{{{ secrets.BUILDKITE_TOKEN }}}}\n'
                f'          buildkite_build_url: ${{{{ steps.build.outputs.url }}}}\n'
                f'          ignore_build_states: blocked,canceled,skipped,not_run\n'
                f'          ignore_job_states: timed_out\n'
                f'          output_path: artifacts/Unit Test Results - GPUs on Buildkite\n'
                f'\n'
                f'      - name: Upload Test Results\n'
                f'        uses: actions/upload-artifact@v2\n'
                f'        if: always()\n'
                f'        with:\n'
                f'          name: Unit Test Results - GPUs on Builtkite\n'
                f'          path: artifacts/Unit Test Results - GPUs on Buildkite/**/*.xml\n')

    def publish_unit_test_results(name: str, needs: List[str]) -> str:
        return (f'  {name}:\n'
                f'    name: "Publish Unit Tests Results"\n'
                f'    needs: [{",".join(needs)}]\n'
                f'    runs-on: ubuntu-latest\n'
                f'    if: >\n'
                f'      always() &&\n'
                f'      ( github.event_name == \'push\' || github.event.pull_request.head.repo.full_name == github.repository )\n'
                f'\n'
                f'    steps:\n'
                f'      - name: Download GitHub Artifacts\n'
                f'        uses: actions/download-artifact@v2\n'
                f'        with:\n'
                f'          path: artifacts\n'
                f'\n'
                f'      - name: Publish Unit Test Results\n'
                f'        uses: docker://ghcr.io/enricomi/publish-unit-test-result-action:v1\n'
                f'        if: always()\n'
                f'        with:\n'
                f'          github_token: ${{{{ github.token }}}}\n'
                f'          files: "artifacts/Unit Test Results */**/*.xml"\n')

    with open(path.joinpath('workflows', 'ci.yaml').absolute(), 'wt') as w:
        heads = ['tfhead', 'torchhead', 'mxnethead']
        release_images = [image for image in images if not all(head in image for head in heads)]
        allhead_images = [image for image in images if all(head in image for head in heads)]
        workflow = workflow_header() + jobs(
            validate_workflow_job(),
            build_and_test_images(name='build-and-test', images=release_images, tests_per_image=tests_per_image, tests=tests),
            build_and_test_images(name='build-and-test-heads', images=allhead_images, tests_per_image=tests_per_image, tests=tests),
            trigger_buildkite_job(name='buildkite', needs=['build-and-test']),
            publish_unit_test_results(name='publish-test-results', needs=['build-and-test, build-and-test-heads, buildkite'])
        )
        print(workflow, file=w, end='')


if __name__ == "__main__":
    main()
