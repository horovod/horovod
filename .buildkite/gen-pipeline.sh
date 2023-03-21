#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eu

# our repository in AWS
repository=823773083436.dkr.ecr.us-east-1.amazonaws.com/buildkite

# our queues
cpu_queue="cpu-v5111"
gpux2_queue="2x-gpu-v5111"
gpux4_queue="4x-gpu-v5111"

# our baseline test is
baseline="test-cpu-gloo-py3_8-tf2_11_0-keras2_11_0-torch1_13_0-mxnet1_9_1-pyspark3_3_1"
# in run_gloo_integration we run 'Elastic Spark * Tests' for this baseline
# so it has to have Gloo mpi kind

# skip tests when there are no code changes
dir="$(dirname "$0")"
code_files=$(python "$dir/get_changed_code_files.py" || echo failure)
tests=$(if [[ -n "${PIPELINE_MODE:-}" ]] && ( [[ "${BUILDKITE_BRANCH:-}" == "${BUILDKITE_PIPELINE_DEFAULT_BRANCH:-}" ]] || [[ -n "$code_files" ]] ); then
  # we vary the baseline along the Python dimension and PySpark together
  # run_gloo_integration expects these to have Gloo mpi kind to run 'Elastic Spark * Tests'
  # our baseline
  printf "$baseline "

  printf "test-cpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_3_1 "
  # these are the lowest framework versions that Horovod compiles with, but they are not tested
  printf "test-cpu-openmpi-gloo-py3_7-tfmin-kerasmin-torchmin-mxnetmin-pysparkmin "

  printf "test-gpu-openmpi-gloo-py3_8-tf2_11_0-keras2_11_0-torch1_13_0-mxnet1_9_1-pyspark3_3_1 "
  printf "test-gpu-openmpi-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_3_1 "
  # these are the lowest framework versions that Horovod compiles with, but they are not tested
  printf "test-gpu-openmpi-gloo-py3_7-tfmin-kerasmin-torchmin-mxnetmin-pysparkmin "

  # and one final test with mixed cpu+gpu
  printf "test-mixed-openmpi-gloo-py3_8-tf2_11_0-keras2_11_0-torch1_13_0-mxnet1_9_1-pyspark3_3_1 "
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
  echo "  - docker-compose#v3.10.0:"
  echo "      build: ${test}"
  echo "      image-repository: ${repository}"
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
  echo "  - docker-compose#v3.10.0:"
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
    ":pytest: MPI Cluster PyTests (${test})" \
    "bash -c \"${oneccl_env} ${test_env} /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.mpi.static.xml test_static_run.py\""
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
    ":pytest: Gloo Cluster PyTests (${test})" \
    "bash -c \"${test_env} /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.static.xml test_static_run.py\""
}

run_gloo() {
  local test=$1
  local queue=$2

  run_gloo_pytest ${test} ${queue}
}

run_single_integration() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}
  oneccl_env=$(echo ${oneccl_env//:/ })

  # Only in TensorFlow 1.X
  if [[ ${test} != *"tf2_"* ]] && [[ ${test} != *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: Single Keras MNIST (${test})" \
      "bash -c \"${oneccl_env} python /horovod/examples/keras/keras_mnist_advanced.py --epochs 3 --batch-size 64\""
  fi
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
  if [[ ${test} == *-cpu-* && ${test} != *min-* ]]; then
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

      # no runner application, world size = 1
      run_single_integration ${test} ${cpu_queue} ${oneccl_cmd_mpi}
      run_single_integration ${test} ${cpu_queue} ${oneccl_cmd_ofi}
    else
      # run mpi cpu unit tests and integration tests
      if [[ ${test} == *mpi* ]]; then
        run_mpi ${test} ${cpu_queue}
      fi

      # no runner application, world size = 1
      run_single_integration ${test} ${cpu_queue}
    fi
  fi
done

# wait for all cpu unit and integration tests to finish
echo "- wait"

# run 4x gpu unit tests
for test in ${tests[@]-}; do
  if ( [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]] ) && [[ ${test} != *min-* ]]; then
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

# wait for all gpu unit tests to finish
echo "- wait"

