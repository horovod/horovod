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
baseline="test-cpu-gloo-py3_8-tf2_8_0-keras2_8_0-torch1_10_2-mxnet1_9_0-pyspark3_2_1"
# in run_gloo_integration we run 'Elastic Spark * Tests' for this baseline
# so it has to have Gloo mpi kind

# skip tests when there are no code changes
dir="$(dirname "$0")"
code_files=$(python "$dir/get_changed_code_files.py" || echo failure)
tests=$(if [[ -n "${PIPELINE_MODE:-}" ]] && ( [[ "${BUILDKITE_BRANCH:-}" == "${BUILDKITE_PIPELINE_DEFAULT_BRANCH:-}" ]] || [[ -n "$code_files" ]] ); then
  printf "test-cpu-gloo-py3_8-tf2_7_1-keras2_7_0-torch1_9_1-mxnet1_8_0_p0-pyspark3_2_1 "
  printf "test-cpu-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_2_1 "
  printf "test-gpu-openmpi-gloo-py3_8-tf2_8_0-keras2_8_0-torch1_10_2-mxnet1_9_0-pyspark3_2_1 "
  printf "test-gpu-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_2_1 "
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

run_spark_integration() {
  local test=$1
  local queue=$2

  # Horovod Spark Estimator tests
  if [[ ${test} != *"mpich"* && ${test} != *"oneccl"* ]]; then
    if [[ ${queue} != *gpu* ]]; then
      run_test "${test}" "${queue}" \
        ":spark: Spark PyTests (${test})" \
        "bash -c \"cd /horovod/test/integration && (ls -1 test_spark*.py | xargs -n 1 /bin/bash /pytest_standalone.sh spark)\"" \
        30
    fi

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
  if [[ ${test} == *-cpu-* ]]; then
    # if oneCCL is specified, run some tests twice,
    # once with mpirun_command_ofi, and once with mpirun_command_mpi
    if [[ ${test} == *oneccl* ]]; then
      # always run spark tests which use MPI and Gloo
      run_spark_integration ${test} ${cpu_queue}
    else
      # always run spark tests which use MPI and Gloo
      run_spark_integration ${test} ${cpu_queue}
    fi
  fi
done

# wait for all cpu unit and integration tests to finish
echo "- wait"

# wait for all gpu unit tests to finish
echo "- wait"

# run 2x gpu integration tests
for test in ${tests[@]-}; do
  if [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]]; then
    run_spark_integration ${test} ${gpux2_queue}
  fi
done
