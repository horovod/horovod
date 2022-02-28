#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eu

# our repository in AWS
repository=823773083436.dkr.ecr.us-east-1.amazonaws.com/buildkite

# our queues
cpu_queue="cpu-v572"
gpux2_queue="2x-gpu-v572"
gpux4_queue="4x-gpu-v572"

# our baseline test is
baseline="test-cpu-gloo-py3_8-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark3_2_0"
# in run_gloo_integration we run 'Elastic Spark * Tests' for this baseline
# so it has to have Gloo mpi kind

# skip tests when there are no code changes
dir="$(dirname "$0")"
code_files=$(python "$dir/get_changed_code_files.py" || echo failure)
tests=$(if [[ -n "${PIPELINE_MODE:-}" ]] && ( [[ "${BUILDKITE_BRANCH:-}" == "${BUILDKITE_PIPELINE_DEFAULT_BRANCH:-}" ]] || [[ -n "$code_files" ]] ); then
  # we vary the baseline along the Python dimension and PySpark together
  # run_gloo_integration expects these to have Gloo mpi kind to run 'Elastic Spark * Tests'
  printf "test-cpu-gloo-py3_7-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark2_4_8 "
  printf "test-cpu-gloo-py3_8-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark3_1_2 "
  # our baseline
  printf "$baseline "

  # then we vary the baseline along mpi kinds dimension
  # our baseline again
# printf "test-cpu-gloo-py3_8-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark3_2_0 "
  printf "test-cpu-mpich-py3_8-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark3_2_0 "
  printf "test-cpu-oneccl-py3_8-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark3_2_0 "
  printf "test-cpu-openmpi-py3_8-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark3_2_0 "
  # note: we test openmpi-gloo mpi kind in this variation in each of [cpu, gpu, mixed]
  printf "test-cpu-openmpi-gloo-py3_8-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark3_2_0 "

  # then we vary the baseline along the framework dimensions all together
  # some frameworks are not available for our baseline Python version 3.8, so we use Python 3.7
  # run_gloo_integration expects tf1 to have Gloo mpi kind to run 'Elastic Spark * Tests'
  # there is no mxnet-1.6.0.post0 and mxnet-1.6.0 does not work with horovod
  # https://github.com/apache/incubator-mxnet/issues/16193
  # so we test with mxnet 1.5.1
  printf "test-cpu-gloo-py3_7-tf1_15_5-keras2_2_4-torch1_7_1-mxnet1_5_1_p0-pyspark3_2_0 "
  printf "test-cpu-gloo-py3_8-tf2_5_2-keras2_5_0rc0-torch1_8_1-mxnet1_7_0_p2-pyspark3_2_0 "
  printf "test-cpu-gloo-py3_8-tf2_6_2-keras2_6_0-torch1_9_1-mxnet1_8_0_p0-pyspark3_2_0 "
  # our baseline again
# printf "test-cpu-gloo-py3_8-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark3_2_0 "
  printf "test-cpu-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_2_0 "

  # then we vary the frameworks for gpu
  # there is no mxnet-1.6.0.post0 and mxnet-1.6.0 does not work with horovod
  # https://github.com/apache/incubator-mxnet/issues/16193
  # so we test with mxnet 1.5.1
  printf "test-gpu-gloo-py3_7-tf1_15_5-keras2_2_4-torch1_7_1-mxnet1_5_1_p0-pyspark3_2_0 "
  # here we deviate from mxnet==1.7.0.post2 as there is none for cu101, only post1
  printf "test-gpu-gloo-py3_8-tf2_5_2-keras2_5_0rc0-torch1_8_1-mxnet1_7_0_p1-pyspark3_2_0 "
  printf "test-gpu-gloo-py3_8-tf2_6_2-keras2_6_0-torch1_9_1-mxnet1_8_0_p0-pyspark3_2_0 "
  printf "test-gpu-openmpi-gloo-py3_8-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark3_2_0 "
  printf "test-gpu-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_2_0 "

  # and one final test with mixed cpu+gpu
  printf "test-mixed-openmpi-gloo-py3_8-tf2_7_0-keras2_7_0-torch1_10_1-mxnet1_9_0-pyspark3_2_0 "
fi | if [[ "${PIPELINE_MODE:-}" == "GPU"* ]]; then sed -E "s/[^ ]*-cpu-[^ ]*//g"; else cat; fi \
   | if [[ "${PIPELINE_MODE:-}" == "GPU HEADS" ]]; then sed -E "s/ /\n/g" | grep -e "-tfhead-keras_none-torchhead-mxnethead-" | paste -s -d " " -; else cat; fi \
   | if [[ "${PIPELINE_MODE:-}" == "GPU NON HEADS" ]]; then sed -E "s/[^ ]*-tfhead-keras_none-torchhead-mxnethead-[^ ]*//g"; else cat; fi)
read -r -a tests <<< "$tests"


build_test() {
  local test=$1

  echo "- label: ':docker: Build ${test}'"
  echo "  env:"
  echo "    COMPOSE_HTTP_TIMEOUT: 300"
  echo "  plugins:"
  echo "  - docker-compose#v3.5.0:"
  echo "      build: ${test}"
  echo "      image-repository: ${repository}"
  echo "      cache-from: ${test}:${repository}:${BUILDKITE_PIPELINE_SLUG}-${test}-latest"
  echo "      config: docker-compose.test.yml"
  echo "      push-retries: 5"
  echo "  - ecr#v1.2.0:"
  echo "      login: true"
  echo "  timeout_in_minutes: 40"
  echo "  retry:"
  echo "    automatic: true"
  echo "  agents:"
  echo "    queue: ${cpu_queue}"
}

run_test() {
  local test=$1
  local queue=$2
  local label=$3
  local command=$4
  local timeout=${5-10}

  echo "- label: '${label}'"
  echo "  command: ${command}"
  echo "  artifact_paths: \"artifacts/**\""
  echo "  env:"
  echo "    COMPOSE_HTTP_TIMEOUT: 300"
  echo "  plugins:"
  echo "  - docker-compose#v3.5.0:"
  echo "      run: ${test}"
  echo "      volumes: \"./artifacts:/artifacts\""
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

run_mpi_pytest() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}
  oneccl_env=$(echo ${oneccl_env//:/ })

  test_env=""
  if [[ ${queue} == *"gpu"* ]]; then
    test_env="HOROVOD_TEST_GPU=1"
  fi

  run_test "${test}" "${queue}" \
    ":pytest: MPI Single PyTests (${test})" \
    "bash -c \"${oneccl_env} ${test_env} cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh mpi)\"" \
    10
}

run_mpi() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}

  run_mpi_pytest ${test} ${queue} ${oneccl_env}
}

run_gloo_pytest() {
  local test=$1
  local queue=$2

  test_env=""
  if [[ ${queue} == *"gpu"* ]]; then
    test_env="HOROVOD_TEST_GPU=1"
  fi

  run_test "${test}" "${queue}" \
    ":pytest: Gloo Single PyTests (${test})" \
    "bash -c \"${test_env} cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh gloo)\"" \
    15
}

run_gloo() {
  local test=$1
  local queue=$2

  run_gloo_pytest ${test} ${queue}
}

# begin the pipeline.yml file
echo "steps:"

# build every test container
for test in ${tests[@]-}; do
  build_test "${test}"
done

# wait for all builds to finish
echo "- wait"

oneccl_env="\\\$(cat:/oneccl_env):&&"
oneccl_cmd_ofi="${oneccl_env}:echo:'/mpirun_command_ofi':>:/mpirun_command:&&"
oneccl_cmd_mpi="${oneccl_env}:echo:'/mpirun_command_mpi':>:/mpirun_command:&&"

# run all the cpu unit tests and integration tests
for test in ${tests[@]-}; do
  if [[ ${test} == *-cpu-* ]]; then
    # if gloo is specified, run gloo cpu unit tests and integration tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo ${test} ${cpu_queue}
    fi

    # if oneCCL is specified, run some tests twice,
    # once with mpirun_command_ofi, and once with mpirun_command_mpi
    if [[ ${test} == *oneccl* ]]; then
      # run mpi cpu unit tests and integration tests
      run_mpi ${test} ${cpu_queue} ${oneccl_cmd_mpi}
      run_mpi ${test} ${cpu_queue} ${oneccl_cmd_ofi}
    else
      # run mpi cpu unit tests and integration tests
      if [[ ${test} == *mpi* ]]; then
        run_mpi ${test} ${cpu_queue}
      fi
    fi
  fi
done

# wait for all cpu unit and integration tests to finish
echo "- wait"

# run 4x gpu unit tests
for test in ${tests[@]-}; do
  if [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]]; then
    # if gloo is specified, run gloo gpu unit tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo_pytest ${test} ${gpux4_queue}
    fi

    # if mpi is specified, run mpi gpu unit tests
    if [[ ${test} == *mpi* ]]; then
      run_mpi_pytest ${test} ${gpux4_queue}
    fi
  fi
done

