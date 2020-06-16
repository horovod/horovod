#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eu

# our repository in AWS
repository=823773083436.dkr.ecr.us-east-1.amazonaws.com/buildkite

# list of all the tests
tests=( \
       #test-cpu-openmpi-py3_6-tf1_6_0-keras2_1_2-torch0_4_1-mxnet1_4_1-pyspark2_3_2 \
       test-cpu-gloo-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
       #test-cpu-gloo-py3_7-tf2_2_0-keras2_3_1-torch1_5_0-mxnet1_5_0-pyspark2_4_0 \
       #test-cpu-gloo-py3_8-tf2_2_0-keras2_3_1-torch1_5_0-mxnet1_5_0-pyspark3_0_0 \
       #test-cpu-openmpi-py3_6-tf1_14_0-keras2_2_4-torch1_2_0-mxnet1_4_1-pyspark2_4_0 \
       #test-cpu-openmpi-py3_6-tf2_0_0-keras2_3_1-torch1_5_0-mxnet1_5_0-pyspark2_4_0 \
       #test-cpu-openmpi-py3_6-tfhead-kerashead-torchhead-mxnethead-pyspark2_4_0 \
       #test-cpu-mpich-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
       #test-cpu-oneccl-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
       #test-cpu-oneccl-ofi-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
       #test-gpu-openmpi-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_4_1-pyspark2_4_0 \
       #test-gpu-gloo-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_4_1-pyspark2_4_0 \
       #test-gpu-openmpi-gloo-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_4_1-pyspark2_4_0 \
       #test-gpu-openmpi-py3_6-tf2_2_0-keras2_3_1-torch1_5_0-mxnet1_6_0-pyspark2_4_0 \
       #test-gpu-openmpi-py3_6-tfhead-kerashead-torchhead-mxnethead-pyspark2_4_0 \
       #test-mixed-openmpi-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
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
  local timeout=${5-5}

  echo "- label: '${label}'"
  echo "  command: ${command}"
  echo "  plugins:"
  echo "  - docker-compose#v2.6.0:"
  echo "      run: ${test}"
  echo "      config: docker-compose.test.yml"
  echo "      pull-retries: 3"
  echo "  - ecr#v1.2.0:"
  echo "      login: true"
  echo "  timeout_in_minutes: ${timeout}"
  echo "  retry:"
  echo "    automatic: true"
  echo "  agents:"
  echo "    queue: ${queue}"
}

run_gloo_integration() {
  local test=$1
  local queue=$2

  # Elastic
  local elastic_tensorflow="test_elastic_tensorflow.py"
  local elastic_spark_tensorflow="test_elastic_spark_tensorflow.py"
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
      elastic_tensorflow="test_elastic_tensorflow2.py"
  fi

  run_test "${test}" "${queue}" \
    ":factory: Elastic Tests (${test})" \
    "bash -c \"cd /horovod/test/integration && HOROVOD_LOG_LEVEL=DEBUG pytest --forked -v --log-cli-level 10 --log-cli-format '[%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d %(funcName)s()] %(message)s' --capture=no test_elastic_torch.py ${elastic_tensorflow}\""

}

run_gloo() {
  local test=$1
  local queue=$2

  run_gloo_integration ${test} ${queue}
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
build_docs

# wait for all builds to finish
echo "- wait"

# cache test containers if built from master
if [[ "${BUILDKITE_BRANCH}" == "master" ]]; then
  for test in ${tests[@]}; do
    cache_test "${test}"
  done
fi

oneccl_env=""

# run all the cpu unit tests and integration tests
for test in ${tests[@]}; do
  if [[ ${test} == *-cpu-* ]]; then
    # if gloo is specified, run gloo cpu unit tests and integration tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo ${test} "cpu"
    fi
  fi
done

# wait for all cpu unit and integration tests to finish
echo "- wait"

# run 2x gpu integration tests
for test in ${tests[@]}; do
  if [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]]; then
    # if gloo is specified, run gloo gpu integration tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo_integration ${test} "2x-gpu-g4"
    fi
  fi
done
