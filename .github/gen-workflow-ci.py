import re

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
        BUILDKITE_PIPELINE_SLUG='horovod',
        BUILDKITE_PIPELINE_DEFAULT_BRANCH='master',
        BUILDKITE_BRANCH='master'
    )
    proc = subprocess.run([script], env=env, stdout=subprocess.PIPE, encoding='utf-8')
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
                  plugin['docker-compose#v3.5.0']['run'])
                 for step in steps if isinstance(step, dict) and 'label' in step and 'command' in step
                 and not step['label'].startswith(':docker: Build ') and '-cpu-' in step['label']
                 for plugin in step['plugins'] if 'docker-compose#v3.5.0' in plugin]

    # we need to distinguish the two oneccl variants of some tests
    cpu_tests = [(label + (' [ONECCL OFI]' if 'mpirun_command_ofi' in command else (' [ONECCL MPI]' if 'mpirun_command_mpi' in command else '')),
                  command,
                  image)
                 for label, command, image in cpu_tests]

    # check that labels are unique per image
    cardinalities = Counter([(label, image) for label, command, image in cpu_tests])
    conflicts = [(label, image, card) for (label, image), card in cardinalities.items() if card > 1]
    if conflicts:
        summary = '\n'.join([f'"{label}" for image "{image}"' for label, image, card in conflicts])
        raise RuntimeError(f'There are {len(conflicts)} duplicate test labels for images:\n{summary}')

    # commands for some labels may differ
    # we make their labels unique here
    label_commands = defaultdict(Counter)
    for label, command, image in cpu_tests:
        label_commands[label][command] += 1

    labels_with_multiple_commands = {label: c for label, c in label_commands.items() if len(c) > 1}
    new_labels_per_label_command = {(label, command): f'{label} {index+1}'
                                    for label, commands in labels_with_multiple_commands.items()
                                    for index, command in enumerate(commands)}

    cpu_tests = [(new_labels_per_label_command[(label, command)] if (label, command) in new_labels_per_label_command else label,
                  command,
                  image)
                 for label, command, image in cpu_tests]

    # come up with test ids from test labels
    test_labels = {label for label, command, image in cpu_tests}
    test_id_per_label = [(label, re.sub('[^a-zA-Z0-9_]', '', re.sub('[ .]', '_', label)))
                         for label in test_labels]
    if len({id for label, id in test_id_per_label}) != len(test_labels):
        raise RuntimeError('Some test ids are not unique')
    test_id_per_label = dict(test_id_per_label)

    # collect tests per image
    tests_per_image = {image: {test_id_per_label[label]
                               for label, command, test_image in cpu_tests
                               if test_image == image}
                       for image in sorted(images)}

    # index tests by id
    tests = {test_id_per_label[label]: dict(label=label, command=command)
             for label, command, image in cpu_tests}

    with open('workflows/ci.yaml', 'wt') as w:
        print(f'name: CI\n'
              f'on:\n'
              f'  push:\n'
              f'    branches:\n'
              f'      - master\n'
              f'    tags:\n'
              f'      - \'*\'\n'
              f'  pull_request:\n'
              f'\n'
              f'jobs:\n'
              f'  build-and-test:\n'
              f'    name: "Build and Test (${{{{ matrix.image }}}})"\n'
              f'    runs-on: ubuntu-latest\n'
              f'    continue-on-error: true\n'
              f'    strategy:\n'
              f'      max-parallel: {len(images)}\n'
              f'      fail-fast: false\n'
              f'      matrix:\n'
              f'        include:\n' +
              '\n'.join([f'          - image: {image}\n' +
                         '\n'.join([f'            {test}: true'
                                    for test in sorted(list(tests))]) + '\n'
                         for image, tests in tests_per_image.items()]) +
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
              f'      - name: Build\n'
              f'        run: docker-compose -f docker-compose.test.yml build ${{{{ matrix.image }}}}\n'
              f'\n'
              f'      - name: Docker ls\n'
              f'        run: docker image ls\n'
              f'\n' +
              '\n'.join([f'      - name: "{test["label"]}"\n'
                         f'        if: matrix.{test_id}\n'
                         f'        run: |\n'
                         f'          mkdir -p artifacts/${{{{ matrix.image }}}}/{test_id}\n'
                         f'          docker-compose -f docker-compose.test.yml run --rm --volume "$(pwd)/artifacts/${{{{ matrix.image }}}}/{test_id}:/artifacts" ${{{{ matrix.image }}}} {test["command"]}\n'
                         f'        shell: bash\n'
                         for test_id, test in tests.items()]) +
              f'\n'
              f'      - name: List artifacts\n'
              f'        run: ls -lahR artifacts\n'
              f'\n'
              f'      - name: Upload Test Results\n'
              f'        uses: actions/upload-artifact@v2\n'
              f'        if: always()\n'
              f'        with:\n'
              f'          name: Unit Test Results\n'
              f'          path: artifacts/**/*.xml\n'
              f'\n'
              f'  publish-test-results:\n'
              f'    name: "Publish Unit Tests Results"\n'
              f'    needs: build-and-test\n'
              f'    runs-on: ubuntu-latest\n'
              f'    if: ( github.event_name == \'push\' || github.event.pull_request.head.repo.full_name == github.repository )\n'
              f'\n'
              f'    steps:\n'
              f'      - name: Download Artifacts\n'
              f'        uses: actions/download-artifact@v2\n'
              f'        with:\n'
              f'          path: artifacts\n'
              f'\n'
              f'      - name: Publish Unit Test Results\n'
              f'        uses: docker://ghcr.io/enricomi/publish-unit-test-result-action:v1\n'
              f'        with:\n'
              f'          github_token: ${{ github.token }}\n'
              f'          files: "artifacts/Unit Test Results */**/*.xml"\n'
              f'', file=w, end='')


if __name__ == "__main__":
    main()
