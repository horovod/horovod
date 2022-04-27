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
baseline="test-cpu-gloo-py3_8-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark3_2_1"
# in run_gloo_integration we run 'Elastic Spark * Tests' for this baseline
# so it has to have Gloo mpi kind

# skip tests when there are no code changes
dir="$(dirname "$0")"
code_files=$(python "$dir/get_changed_code_files.py" || echo failure)
tests=$(if [[ -n "${PIPELINE_MODE:-}" ]] && ( [[ "${BUILDKITE_BRANCH:-}" == "${BUILDKITE_PIPELINE_DEFAULT_BRANCH:-}" ]] || [[ -n "$code_files" ]] ); then
  # we vary the baseline along the Python dimension and PySpark together
  # run_gloo_integration expects these to have Gloo mpi kind to run 'Elastic Spark * Tests'
  printf "test-cpu-gloo-py3_7-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark2_4_8 "
  printf "test-cpu-gloo-py3_8-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark3_1_3 "
  # our baseline
  printf "$baseline "

  # then we vary the baseline along mpi kinds dimension
  # our baseline again
# printf "test-cpu-gloo-py3_8-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark3_2_1 "
  printf "test-cpu-mpich-py3_8-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark3_2_1 "
  printf "test-cpu-oneccl-py3_8-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark3_2_1 "
  printf "test-cpu-openmpi-py3_8-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark3_2_1 "
  # note: we test openmpi-gloo mpi kind in this variation in each of [cpu, gpu, mixed]
  printf "test-cpu-openmpi-gloo-py3_8-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark3_2_1 "

  # then we vary the baseline along the framework dimensions all together
  # run_gloo_integration expects tf1 to have Gloo mpi kind to run 'Elastic Spark * Tests'
  # Tensorflow 1.15.5 is only available for Python 3.7
  # Python 3.7 is only available on Ubuntu 18.04
  # see test-gpu-gloo-py3_7-tf1_15_5-... below why we have to test with mxnet 1.5.1 here
  printf "test-cpu-gloo-py3_7-tf1_15_5-keras2_2_4-torch1_8_1-mxnet1_5_1_p0-pyspark3_2_1 "
  printf "test-cpu-gloo-py3_8-tf2_6_3-keras2_6_0-torch1_9_1-mxnet1_7_0_p2-pyspark3_2_1 "
  printf "test-cpu-gloo-py3_8-tf2_7_1-keras2_7_0-torch1_10_2-mxnet1_8_0_p0-pyspark3_2_1 "
  # our baseline again
# printf "test-cpu-gloo-py3_8-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark3_2_1 "
  printf "test-cpu-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_2_1 "
  # these are the lowest framework versions that Horovod compiles with, but they are not tested
  printf "test-cpu-gloo-py3_7-tfmin-kerasmin-torchmin-mxnetmin-pysparkmin "

  # then we vary the frameworks for gpu
  # we need CUDA 10.0 as tensorflow-gpu==1.15.5 is compiled against and linked to CUDA 10.0
  # mxnet-cu100==1.7.0 does not contain mkldnn headers, and there is no 1.7.0.postx that would have them
  # there is no mxnet-1.6.0.post0 and mxnet-1.6.0 does not work with horovod
  # https://github.com/apache/incubator-mxnet/issues/16193
  # so we test with mxnet 1.5.1
  printf "test-gpu-gloo-py3_7-tf1_15_5-keras2_2_4-torch1_8_1-mxnet1_5_1_p0-pyspark3_2_1 "
  # here we deviate from mxnet==1.7.0.post2 as there is none for cu101, only post1
  printf "test-gpu-gloo-py3_8-tf2_6_3-keras2_6_0-torch1_9_1-mxnet1_7_0_p1-pyspark3_2_1 "
  printf "test-gpu-gloo-py3_8-tf2_7_1-keras2_7_0-torch1_10_2-mxnet1_8_0_p0-pyspark3_2_1 "
  printf "test-gpu-openmpi-gloo-py3_8-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark3_2_1 "
  printf "test-gpu-gloo-py3_8-tfhead-keras_none-torchhead-mxnethead-pyspark3_2_1 "
  # these are the lowest framework versions that Horovod compiles with, but they are not tested
  printf "test-gpu-gloo-py3_7-tfmin-kerasmin-torchmin-mxnetmin-pysparkmin "

  # and one final test with mixed cpu+gpu
  printf "test-mixed-openmpi-gloo-py3_8-tf2_8_0-keras2_8_0-torch1_11_0-mxnet1_9_0-pyspark3_2_1 "
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

run_mpi_integration() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}
  oneccl_env=$(echo ${oneccl_env//:/ })

  # TensorFlow 2.0 tests
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: MPI TensorFlow 2.0 MNIST Data Service (${test})" \
      "bash -c \"(which nvidia-smi && nvidia-smi || true); ${oneccl_env} horovodrun -np 2 python -m horovod.tensorflow.data.compute_worker /tmp/compute.json & horovodrun -np 2 --mpi python /horovod/examples/tensorflow2/tensorflow2_mnist_data_service.py /tmp/compute.json\""
  fi
}

run_mpi() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}

  run_mpi_integration ${test} ${queue} ${oneccl_env}
}

run_gloo_integration() {
  local test=$1
  local queue=$2

  # TensorFlow 2.0 tests
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: Gloo TensorFlow 2.0 MNIST Data Service (${test})" \
      "bash -c \"(which nvidia-smi && nvidia-smi || true); horovodrun -np 2 python -m horovod.tensorflow.data.compute_worker /tmp/compute.json & horovodrun -np 2 --gloo python /horovod/examples/tensorflow2/tensorflow2_mnist_data_service.py /tmp/compute.json\""
  fi
}

run_gloo() {
  local test=$1
  local queue=$2

  run_gloo_integration ${test} ${queue}
}

run_spark_integration() {
  local test=$1
  local queue=$2

  # Horovod Spark Estimator tests
  if [[ ${test} != *"mpich"* && ${test} != *"oneccl"* ]]; then
    if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
      run_test "${test}" "${queue}" \
        ":spark: Spark TensorFlow 2.0 MNIST Data Service (${test})" \
        "bash -c \"(which nvidia-smi && nvidia-smi || true); cd /horovod/examples/spark/tensorflow2; spark-submit --master \\\"local[2]\\\" \\\"/horovod/horovod/spark/tensorflow/compute_worker.py\\\" /tmp/compute.json & OMP_NUM_THREADS=1 /spark_env.sh spark-submit --master \\\"local[2]\\\" --py-files tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py,tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py tensorflow2_mnist_data_service.py /tmp/compute.json\""
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
  if [[ ${test} == *-cpu-* && ${test} != *min-* ]]; then
    # if gloo is specified, run gloo cpu unit tests and integration tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo ${test} ${cpu_queue}
    fi

      # run mpi cpu unit tests and integration tests
      if [[ ${test} == *mpi* ]]; then
        run_mpi ${test} ${cpu_queue}
      fi

      # always run spark tests which use MPI and Gloo
      run_spark_integration ${test} ${cpu_queue}

  fi
done

# wait for all cpu unit and integration tests to finish
echo "- wait"

# wait for all gpu unit tests to finish
echo "- wait"

# run 2x gpu integration tests
for test in ${tests[@]-}; do
  if ( [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]] ) && [[ ${test} != *min-* ]]; then
    # if gloo is specified, run gloo gpu integration tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo_integration ${test} ${gpux2_queue}
    fi

    # if mpi is specified, run mpi gpu integration tests
    if [[ ${test} == *mpi* ]]; then
      run_mpi_integration ${test} ${gpux2_queue}
    fi

    run_spark_integration ${test} ${gpux2_queue}
  fi
done
