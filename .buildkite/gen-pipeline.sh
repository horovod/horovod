#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eu

# our repository in AWS
repository=823773083436.dkr.ecr.us-east-1.amazonaws.com/buildkite

# list of all the tests
tests=( \
       test-cpu-mlsl-py3_6-tf1_12_0-keras2_2_2-torch1_0_0-mxnet1_4_0-pyspark2_4_0 \
)

build_test() {
  local test=$1

  echo "- label: ':docker: Build ${test}'"
  echo "  plugins:"
  echo "  - docker-compose#6b0df8a98ff97f42f4944dbb745b5b8cbf04b78c:"
  echo "      build: ${test}"
  echo "      image-repository: ${repository}"
  echo "      cache-from: ${test}:${repository}:${BUILDKITE_PIPELINE_SLUG}-${test}-latest"
  echo "      config: docker-compose.test.yml"
  echo "      push-retries: 5"
  echo "  - ecr#v1.2.0:"
  echo "      login: true"
  echo "  timeout_in_minutes: 30"
  echo "  retry:"
  echo "    automatic: true"
  echo "  agents:"
  echo "    queue: cpu"
}

cache_test() {
  local test=$1

  echo "- label: ':docker: Update ${BUILDKITE_PIPELINE_SLUG}-${test}-latest'"
  echo "  plugins:"
  echo "  - docker-compose#v2.6.0:"
  echo "      push: ${test}:${repository}:${BUILDKITE_PIPELINE_SLUG}-${test}-latest"
  echo "      config: docker-compose.test.yml"
  echo "      push-retries: 3"
  echo "  - ecr#v1.2.0:"
  echo "      login: true"
  echo "  timeout_in_minutes: 5"
  echo "  retry:"
  echo "    automatic: true"
  echo "  agents:"
  echo "    queue: cpu"
}

run_test() {
  local test=$1
  local queue=$2
  local label=$3
  local command=$4

  echo "- label: '${label}'"
  echo "  command: ${command}"
  echo "  plugins:"
  echo "  - docker-compose#v2.6.0:"
  echo "      run: ${test}"
  echo "      config: docker-compose.test.yml"
  echo "      pull-retries: 3"
  echo "  - ecr#v1.2.0:"
  echo "      login: true"
  echo "  timeout_in_minutes: 5"
  echo "  retry:"
  echo "    automatic: true"
  echo "  agents:"
  echo "    queue: ${queue}"
}

run_all() {
  local test=$1
  local queue=$2

  run_test "${test}" "${queue}" \
    ":pytest: Run PyTests (${test})" \
    "bash -c \"cd /horovod/test && (echo test_*.py | xargs -n 1 \\\$(cat /mpirun_command) pytest -v --capture=no)\""

  run_test "${test}" "${queue}" \
    ":muscle: Test TensorFlow MNIST (${test})" \
    "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/tensorflow_mnist.py\""
}

build_docs() {
  echo "- label: ':book: Build Docs'"
  echo "  command: 'cd /workdir/docs && pip install -r requirements.txt && make html'"
  echo "  plugins:"
  echo "  - docker#v3.1.0:"
  echo "      image: 'python:3.7'"
  echo "  timeout_in_minutes: 5"
  echo "  retry:"
  echo "    automatic: true"
  echo "  agents:"
  echo "    queue: cpu"
}

# begin the pipeline.yml file
echo "steps:"

# build every test container
for test in ${tests[@]}; do
  build_test "${test}"
done

# build documentation
#build_docs

# wait for all builds to finish
echo "- wait"

# cache test containers if built from master
if [[ "${BUILDKITE_BRANCH}" == "master" ]]; then
  for test in ${tests[@]}; do
    cache_test "${test}"
  done
fi

# run all the cpu tests
for test in ${tests[@]}; do
  if [[ ${test} == *-cpu-* ]]; then
    run_all ${test} "cpu"
  fi
done

# wait for all builds to finish
echo "- wait"

# run all the gpu tests
for test in ${tests[@]}; do
  if [[ ${test} == *-gpu-* ]]; then
    run_all ${test} "gpu"
  fi
done
