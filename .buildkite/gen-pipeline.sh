#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eu

# our repository in AWS
repository=823773083436.dkr.ecr.us-east-1.amazonaws.com/buildkite

# list of all the tests
tests=( \
       test-cpu-openmpi-py2_7-tf1_1_0-keras2_0_0-torch0_4_0-mxnet1_4_1-pyspark2_1_2 \
       test-cpu-openmpi-py3_5-tf1_1_0-keras2_0_0-torch0_4_0-mxnet1_4_1-pyspark2_1_2 \
       test-cpu-openmpi-py3_6-tf1_1_0-keras2_0_0-torch0_4_0-mxnet1_4_1-pyspark2_1_2 \
       test-cpu-openmpi-py2_7-tf1_6_0-keras2_1_2-torch0_4_1-mxnet1_4_1-pyspark2_3_2 \
       test-cpu-openmpi-py3_5-tf1_6_0-keras2_1_2-torch0_4_1-mxnet1_4_1-pyspark2_3_2 \
       test-cpu-openmpi-py3_6-tf1_6_0-keras2_1_2-torch0_4_1-mxnet1_4_1-pyspark2_3_2 \
       test-cpu-openmpi-py2_7-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-cpu-openmpi-py3_5-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-cpu-openmpi-py3_6-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-cpu-gloo-py2_7-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-cpu-gloo-py3_5-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-cpu-gloo-py3_6-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-cpu-openmpi-gloo-py2_7-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-cpu-openmpi-gloo-py3_5-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-cpu-openmpi-gloo-py3_6-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-cpu-openmpi-py2_7-tf2_0_0-keras2_2_4-torch1_2_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-openmpi-py3_5-tf2_0_0-keras2_2_4-torch1_2_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-openmpi-py3_6-tf2_0_0-keras2_2_4-torch1_2_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-openmpi-py2_7-tfhead-kerashead-torchhead-mxnethead-pyspark2_4_0 \
       test-cpu-openmpi-py3_6-tfhead-kerashead-torchhead-mxnethead-pyspark2_4_0 \
       test-cpu-mpich-py3_6-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-mlsl-py3_6-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_5_0-pyspark2_4_0 \
       test-gpu-openmpi-py3_6-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-gpu-gloo-py3_6-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-gpu-openmpi-gloo-py3_6-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_4_1-pyspark2_4_0 \
       test-gpu-openmpi-py3_6-tf2_0_0-keras2_2_4-torch1_2_0-mxnet1_5_0-pyspark2_4_0 \
       test-gpu-openmpi-py3_6-tfhead-kerashead-torchhead-mxnethead-pyspark2_4_0 \
       test-mixed-openmpi-py3_6-tf1_14_0-keras2_2_4-torch1_1_0-mxnet1_5_0-pyspark2_4_0 \
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
  local pytest_queue=$3

  local exclude_keras_if_needed=""
  if [[ ${test} == *"tf2_"* ]]; then
    # TODO: support for Keras + TF 2.0 and TF-Keras 2.0
    exclude_keras_if_needed="| sed 's/[a-z_]*keras[a-z_.]*//g'"
  fi

  # pytests have 4x GPU use cases and require a separate queue
  run_test "${test}" "${pytest_queue}" \
    ":pytest: Run PyTests (${test})" \
    "bash -c \"cd /horovod/test && (echo test_*.py ${exclude_keras_if_needed} | xargs -n 1 \\\$(cat /mpirun_command) pytest -v --capture=no)\""

  # Legacy TensorFlow tests
  if [[ ${test} != *"tf2_"* ]]; then
    run_test "${test}" "${queue}" \
      ":muscle: Test TensorFlow MNIST (${test})" \
      "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/tensorflow_mnist.py\""

    if [[ ${test} != *"tf1_1_0"* && ${test} != *"tf1_6_0"* ]]; then
      run_test "${test}" "${queue}" \
        ":muscle: Test TensorFlow Eager MNIST (${test})" \
        "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/tensorflow_mnist_eager.py\""
    fi

    run_test "${test}" "${queue}" \
      ":muscle: Test Keras MNIST (${test})" \
      "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/keras_mnist_advanced.py\""
  fi

  run_test "${test}" "${queue}" \
    ":muscle: Test PyTorch MNIST (${test})" \
    "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/pytorch_mnist.py\""

  run_test "${test}" "${queue}" \
    ":muscle: Test MXNet MNIST (${test})" \
    "bash -c \"OMP_NUM_THREADS=1 \\\$(cat /mpirun_command) python /horovod/examples/mxnet_mnist.py\""

  # tests that should be executed only with the latest release since they don't test
  # a framework-specific functionality
  if [[ ${test} == *"tf1_14_0"* ]]; then
    run_test "${test}" "${queue}" \
      ":muscle: Test Stall (${test})" \
      "bash -c \"\\\$(cat /mpirun_command) python /horovod/test/test_stall.py\""

    if [[ ${test} == *"openmpi"* ]]; then
      run_test "${test}" "${queue}" \
        ":muscle: Test Horovodrun (${test})" \
        "horovodrun -np 2 -H localhost:2 python /horovod/examples/tensorflow_mnist.py"
      run_test "${test}" "${queue}" \
        ":muscle: Test Horovodrun (${test})" \
        "echo 'localhost slots=2' > hostfile" \
        "horovodrun -np 2 -hostfile hostfile python /horovod/examples/mxnet_mnist.py"
    fi
  fi

  # TensorFlow 2.0 tests
  if [[ ${test} == *"tf2_"* ]]; then
    run_test "${test}" "${queue}" \
      ":muscle: Test TensorFlow 2.0 MNIST (${test})" \
      "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/tensorflow2_mnist.py\""

    run_test "${test}" "${queue}" \
      ":muscle: Test TensorFlow 2.0 Keras MNIST (${test})" \
      "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/tensorflow2_keras_mnist.py\""
  fi
}

run_gloo() {
  local test=$1
  local queue=$2
  local pytest_queue=$3

  # Seems that spark tests depend on MPI, do not test those when mpi is not available
  local exclude_spark_if_needed=""
  if [[ ${test} != *"mpi"* ]]; then
    exclude_spark_if_needed="| sed 's/[a-z_]*spark[a-z_.]*//g'"
  fi

  run_test "${test}" "${pytest_queue}" \
    ":pytest: Run PyTests (${test})" \
    "bash -c \"cd /horovod/test && (echo test_*.py ${exclude_spark_if_needed} | xargs -n 1 horovodrun -np 2 -H localhost:2 --gloo pytest -v --capture=no)\""

  run_test "${test}" "${queue}" \
    ":muscle: Test Keras MNIST (${test})" \
    "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/keras_mnist_advanced.py"

  run_test "${test}" "${queue}" \
    ":muscle: Test PyTorch MNIST (${test})" \
    "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/pytorch_mnist.py"

  run_test "${test}" "${queue}" \
    ":muscle: Test MXNet MNIST (${test})" \
    "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/mxnet_mnist.py"
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

# run all the cpu tests
for test in ${tests[@]}; do
  if [[ ${test} == *-cpu-* ]]; then
    # if gloo is specified, run gloo_test
    if [[ ${test} == *-gloo* ]]; then
      run_gloo ${test} "cpu" "cpu"
    fi
    # if mpi is specified, run mpi cpu_test
    if [[ ${test} == *mpi* ]]; then
      run_all ${test} "cpu" "cpu"
    fi
  fi
done

# wait for all builds to finish
echo "- wait"

# run all the gpu tests
for test in ${tests[@]}; do
  if [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]]; then
    # if gloo is specified, run gloo_test
    if [[ ${test} == *-gloo* ]]; then
      run_gloo ${test} "gpu" "4x-gpu"
    fi
    # if mpi is specified, run mpi gpu_test
    if [[ ${test} == *mpi* ]]; then
      run_all ${test} "gpu" "4x-gpu"
    fi
  fi
done
