#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eu

# our repository in AWS
repository=823773083436.dkr.ecr.us-east-1.amazonaws.com/buildkite

# skip tests when there are no code changes
dir="$(dirname "$0")"
code_files=$(python "$dir/get_changed_code_files.py" || echo failure)
tests=$(if [[ "${BUILDKITE_BRANCH:-}" == "${BUILDKITE_PIPELINE_DEFAULT_BRANCH:-}" ]] || [[ -n "$code_files" ]]; then
  printf "test-cpu-openmpi-gloo-py3_6-tf1_15_0-keras2_2_4-torch1_2_0-mxnet1_4_1-pyspark2_3_2 "
  printf "test-cpu-gloo-py3_6-tf2_0_0-keras2_2_4-torch1_3_0-mxnet1_4_1-pyspark2_3_2 "
  printf "test-cpu-openmpi-py3_6-tf2_1_0-keras2_2_4-torch1_4_0-mxnet1_4_1-pyspark2_4_7 "
  printf "test-cpu-openmpi-py3_6-tf2_2_0-keras2_3_1-torch1_5_0-mxnet1_5_0-pyspark2_4_7 "
  printf "test-cpu-gloo-py3_7-tf2_3_1-keras2_3_1-torch1_6_0-mxnet1_5_0-pyspark2_4_7 "
  printf "test-cpu-gloo-py3_8-tf2_4_0-keras2_3_1-torch1_7_1-mxnet1_5_0-pyspark3_0_1 "
  printf "test-cpu-openmpi-py3_6-tfhead-kerashead-torchhead-mxnethead-pyspark2_4_7 "
  printf "test-cpu-mpich-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_7 "
  printf "test-cpu-oneccl-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_7 "
  printf "test-cpu-oneccl-ofi-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_7 "
  printf "test-gpu-openmpi-py3_6-tf1_15_0-keras2_2_4-torch1_3_0-mxnet1_4_1-pyspark2_4_7 "
  printf "test-gpu-gloo-py3_6-tf2_0_0-keras2_3_1-torch1_4_0-mxnet1_4_1-pyspark2_4_7 "
  printf "test-gpu-openmpi-py3_6-tf2_3_1-keras2_3_1-torch1_6_0-mxnet1_6_0-pyspark2_4_7 "
  printf "test-gpu-openmpi-gloo-py3_6-tf2_4_0-keras2_3_1-torch1_7_1-mxnethead-pyspark2_4_7 "
  printf "test-gpu-openmpi-py3_6-tfhead-kerashead-torchhead-mxnethead-pyspark2_4_7 "
  printf "test-mixed-openmpi-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_7 "
fi)
read -r -a tests <<< "$tests"


build_test() {
  local test=$1

  echo "- label: ':docker: Build ${test}'"
  echo "  plugins:"
  echo "  - docker-compose#v3.5.0:"
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
  echo "  - docker-compose#v3.5.0:"
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
  echo "  artifact_paths: \"artifacts/**\""
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

  # pytests have 4x GPU use cases and require a separate queue
  run_test "${test}" "${queue}" \
    ":pytest: MPI Parallel PyTests (${test})" \
    "bash -c \"${oneccl_env} ${test_env} cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 \\\$(cat /mpirun_command) /bin/bash /pytest.sh mpi)\"" \
    5
  run_test "${test}" "${queue}" \
    ":pytest: MPI Single PyTests (${test})" \
    "bash -c \"${oneccl_env} ${test_env} cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh mpi)\"" \
    10

  run_test "${test}" "${queue}" \
    ":pytest: MPI Cluster PyTests (${test})" \
    "bash -c \"${oneccl_env} ${test_env} /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.mpi.static.xml test_static_run.py\""
}

run_mpi_integration() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}
  oneccl_env=$(echo ${oneccl_env//:/ })

  # Run test_interactiverun.py
  if [[ ${test} != *"mpich"* ]] && [[ ${test} != *"oneccl"* ]]; then
    # TODO: support mpich
    run_test "${test}" "${queue}" \
      ":jupyter: Run PyTests test_interactiverun (${test})" \
      "bash -c \"cd /horovod/test && pytest -v --capture=no --continue-on-collection-errors --junit-xml=/artifacts/junit.mpi.integration.xml integration/test_interactiverun.py\""
  fi

  # Legacy TensorFlow tests
  if [[ ${test} != *"tf2_"* ]] && [[ ${test} != *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: MPI TensorFlow MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/tensorflow/tensorflow_mnist.py\""

    run_test "${test}" "${queue}" \
      ":tensorflow: MPI TensorFlow Eager MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/tensorflow/tensorflow_mnist_eager.py\""

    run_test "${test}" "${queue}" \
      ":tensorflow: MPI Keras MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/keras/keras_mnist_advanced.py\""

    run_test "${test}" "${queue}" \
      ":fire: MPI PyTorch MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/pytorch/pytorch_mnist.py\""
  fi

  if [[ ${test} == *"mxnet2_"* ]] || [[ ${test} == *"mxnethead"* ]]; then
      run_test "${test}" "${queue}" \
               ":muscle: MPI MXNet2 MNIST (${test})" \
               "bash -c \"${oneccl_env} OMP_NUM_THREADS=1 \\\$(cat /mpirun_command) python /horovod/examples/mxnet/mxnet2_mnist.py\""
  else
      run_test "${test}" "${queue}" \
               ":muscle: MPI MXNet MNIST (${test})" \
               "bash -c \"${oneccl_env} OMP_NUM_THREADS=1 \\\$(cat /mpirun_command) python /horovod/examples/mxnet/mxnet_mnist.py\""
  fi

  # tests that should be executed only with the latest release since they don't test
  # a framework-specific functionality
  if [[ ${test} == *"tf1_15_0"* ]]; then
    run_test "${test}" "${queue}" \
      ":muscle: MPI Stall (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/test/integration/test_stall.py\""

    if [[ ${test} == *"openmpi"* ]]; then
      run_test "${test}" "${queue}" \
        ":terminal: MPI Horovodrun (${test})" \
        "horovodrun -np 2 -H localhost:2 python /horovod/examples/tensorflow/tensorflow_mnist.py"
      run_test "${test}" "${queue}" \
        ":terminal: MPI Horovodrun (${test})" \
        "bash -c \"echo 'localhost slots=2' > hostfile && horovodrun -np 2 -hostfile hostfile python /horovod/examples/mxnet/mxnet_mnist.py\""
    fi
  fi

  # TensorFlow 2.0 tests
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: MPI TensorFlow 2.0 MNIST (${test})" \
      "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/tensorflow2/tensorflow2_mnist.py\""

    run_test "${test}" "${queue}" \
      ":tensorflow: MPI TensorFlow 2.0 Keras MNIST (${test})" \
      "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/tensorflow2/tensorflow2_keras_mnist.py\""
  fi
}

run_mpi() {
  local test=$1
  local queue=$2
  local oneccl_env=${3:-}

  run_mpi_pytest ${test} ${queue} ${oneccl_env}
  run_mpi_integration ${test} ${queue} ${oneccl_env}
}

run_gloo_pytest() {
  local test=$1
  local queue=$2

  test_env=""
  if [[ ${queue} == *"gpu"* ]]; then
    test_env="HOROVOD_TEST_GPU=1"
  fi

  run_test "${test}" "${queue}" \
    ":pytest: Gloo Parallel PyTests (${test})" \
    "bash -c \"${test_env} cd /horovod/test/parallel && (ls -1 test_*.py | xargs -n 1 horovodrun -np 2 -H localhost:2 --gloo /bin/bash /pytest.sh gloo)\"" \
    5
  run_test "${test}" "${queue}" \
    ":pytest: Gloo Single PyTests (${test})" \
    "bash -c \"${test_env} cd /horovod/test/single && (ls -1 test_*.py | xargs -n 1 /bin/bash /pytest_standalone.sh gloo)\"" \
    10

  run_test "${test}" "${queue}" \
    ":pytest: Gloo Cluster PyTests (${test})" \
    "bash -c \"${test_env} /etc/init.d/ssh start && cd /horovod/test/integration && pytest --forked -v --capture=fd --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.static.xml test_static_run.py\""
}

run_gloo_integration() {
  local test=$1
  local queue=$2

  # TensorFlow 2.0 tests
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: Gloo TensorFlow 2.0 MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/tensorflow2/tensorflow2_mnist.py"

    run_test "${test}" "${queue}" \
      ":tensorflow: Gloo TensorFlow 2.0 Keras MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/tensorflow2/tensorflow2_keras_mnist.py"
  else
    run_test "${test}" "${queue}" \
      ":tensorflow: Gloo TensorFlow MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/tensorflow/tensorflow_mnist.py"

    run_test "${test}" "${queue}" \
      ":tensorflow: Gloo Keras MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/keras/keras_mnist_advanced.py"
  fi

  run_test "${test}" "${queue}" \
    ":fire: Gloo PyTorch MNIST (${test})" \
    "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/pytorch/pytorch_mnist.py"

  if [[ ${test} == *"mxnet2_"* ]] || [[ ${test} == *"mxnethead"* ]]; then
      run_test "${test}" "${queue}" \
               ":muscle: Gloo MXNet2 MNIST (${test})" \
               "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/mxnet/mxnet2_mnist.py"
  else
      run_test "${test}" "${queue}" \
               ":muscle: Gloo MXNet MNIST (${test})" \
               "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/mxnet/mxnet_mnist.py"
  fi

  # Elastic
  local elastic_tensorflow="test_elastic_tensorflow.py test_elastic_tensorflow_keras.py"
  local elastic_spark_tensorflow="test_elastic_spark_tensorflow.py"
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
      elastic_tensorflow="test_elastic_tensorflow2.py"
      elastic_spark_tensorflow="test_elastic_spark_tensorflow2.py"
  fi

  run_test "${test}" "${queue}" \
    ":factory: Elastic Tests (${test})" \
    "bash -c \"cd /horovod/test/integration && HOROVOD_LOG_LEVEL=DEBUG pytest --forked -v --log-cli-level 10 --log-cli-format '[%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d %(funcName)s()] %(message)s' --capture=no --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.elastic.xml test_elastic_torch.py ${elastic_tensorflow}\""

  run_test "${test}" "${queue}" \
    ":factory: Elastic Spark TensorFlow Tests (${test})" \
    "bash -c \"cd /horovod/test/integration && SPARK_HOME=/spark SPARK_DRIVER_MEM=512m HOROVOD_LOG_LEVEL=DEBUG pytest --forked -v --log-cli-level 10 --log-cli-format '[%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d %(funcName)s()] %(message)s' --capture=no --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.elastic.spark.tf.xml ${elastic_spark_tensorflow}\"" \
    15

  run_test "${test}" "${queue}" \
    ":factory: Elastic Spark Torch Tests (${test})" \
    "bash -c \"cd /horovod/test/integration && SPARK_HOME=/spark SPARK_DRIVER_MEM=512m HOROVOD_LOG_LEVEL=DEBUG pytest --forked -v --log-cli-level 10 --log-cli-format '[%(asctime)-15s %(levelname)s %(filename)s:%(lineno)d %(funcName)s()] %(message)s' --capture=no --continue-on-collection-errors --junit-xml=/artifacts/junit.gloo.elastic.spark.torch.xml test_elastic_spark_torch.py\"" \
    15

}

run_gloo() {
  local test=$1
  local queue=$2

  run_gloo_pytest ${test} ${queue}
  run_gloo_integration ${test} ${queue}
}

run_spark_integration() {
  local test=$1
  local queue=$2

  # Horovod Spark Estimator tests
  if [[ ${test} != *"mpich"* && ${test} != *"oneccl"* ]]; then
    if [[ ${queue} != *gpu* ]]; then
      run_test "${test}" "${queue}" \
        ":spark: Spark PyTests (${test})" \
        "bash -c \"cd /horovod/test/integration && (ls -1 test_spark*.py | xargs -n 1 /bin/bash /pytest_standalone.sh spark)\""
    fi

    if [[ ${test} != *"tf2"* && ${test} != *"tfhead"* ]]; then
      run_test "${test}" "${queue}" \
        ":spark: Spark Keras Rossmann Run (${test})" \
        "bash -c \"OMP_NUM_THREADS=1 python /horovod/examples/spark/keras/keras_spark_rossmann_run.py --num-proc 2 --data-dir file:///data --epochs 3 --sample-rate 0.01\""

      run_test "${test}" "${queue}" \
        ":spark: Spark Keras Rossmann Estimator (${test})" \
        "bash -c \"OMP_NUM_THREADS=1 python /horovod/examples/spark/keras/keras_spark_rossmann_estimator.py --num-proc 2 --work-dir /work --data-dir file:///data --epochs 3 --sample-rate 0.01\""

      run_test "${test}" "${queue}" \
        ":spark: Spark Keras MNIST (${test})" \
        "bash -c \"OMP_NUM_THREADS=1 python /horovod/examples/spark/keras/keras_spark_mnist.py --num-proc 2 --work-dir /work --data-dir /data --epochs 3\""
    fi

    run_test "${test}" "${queue}" \
      ":spark: Spark Torch MNIST (${test})" \
      "bash -c \"OMP_NUM_THREADS=1 python /horovod/examples/spark/pytorch/pytorch_spark_mnist.py --num-proc 2 --work-dir /work --data-dir /data --epochs 3\""
  fi
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

  run_test "${test}" "${queue}" \
    ":fire: Single PyTorch MNIST (${test})" \
    "bash -c \"${oneccl_env} python /horovod/examples/pytorch/pytorch_mnist.py --epochs 3\""

  if [[ ${test} == *"mxnet2_"* ]] || [[ ${test} == *"mxnethead"* ]]; then
      run_test "${test}" "${queue}" \
               ":muscle: Single MXNet2 MNIST (${test})" \
               "bash -c \"${oneccl_env} python /horovod/examples/mxnet/mxnet2_mnist.py --epochs 3\""
  else
      run_test "${test}" "${queue}" \
               ":muscle: Single MXNet MNIST (${test})" \
               "bash -c \"${oneccl_env} python /horovod/examples/mxnet/mxnet_mnist.py --epochs 3\""
  fi
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
for test in ${tests[@]-}; do
  build_test "${test}"
done

# build documentation
build_docs

# wait for all builds to finish
echo "- wait"

# cache test containers if built from master
if [[ "${BUILDKITE_BRANCH}" == "master" ]]; then
  for test in ${tests[@]-}; do
    cache_test "${test}"
  done
fi

oneccl_env=""

# run all the cpu unit tests and integration tests
for test in ${tests[@]-}; do
  if [[ ${test} == *-cpu-* ]]; then
    # if gloo is specified, run gloo cpu unit tests and integration tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo ${test} "cpu"
    fi

    #if oneCCL is specified, prepare oneCCL environment
    if [[ ${test} == *oneccl* ]]; then
       oneccl_env="\\\$(cat:/oneccl_env):&&"
       if [[ ${test} == *ofi* ]]; then
          oneccl_env="${oneccl_env}:echo:'/mpirun_command_ofi':>:/mpirun_command:&&"
       else
          oneccl_env="${oneccl_env}:echo:'/mpirun_command_mpi':>:/mpirun_command:&&"
       fi
    else
       oneccl_env=""
    fi

    # if mpi is specified, run mpi cpu unit tests and integration tests
    # if oneccl is specified, run those tests, too
    if [[ ${test} == *mpi* || ${test} == *oneccl* ]]; then
      run_mpi ${test} "cpu" ${oneccl_env}
    fi

    # always run spark tests which use MPI and Gloo
    run_spark_integration ${test} "cpu"

    # no runner application, world size = 1
    run_single_integration ${test} "cpu" ${oneccl_env}
  fi
done

# wait for all cpu unit and integration tests to finish
echo "- wait"

# run 4x gpu unit tests
for test in ${tests[@]-}; do
  if [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]]; then
    # if gloo is specified, run gloo gpu unit tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo_pytest ${test} "4x-gpu-v510"
    fi

    # if mpi is specified, run mpi gpu unit tests
    if [[ ${test} == *mpi* ]]; then
      run_mpi_pytest ${test} "4x-gpu-v510"
    fi
  fi
done

# wait for all gpu unit tests to finish
echo "- wait"

# run 2x gpu integration tests
for test in ${tests[@]-}; do
  if [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]]; then
    # if gloo is specified, run gloo gpu integration tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo_integration ${test} "2x-gpu-v510"
    fi

    # if mpi is specified, run mpi gpu integration tests
    if [[ ${test} == *mpi* ]]; then
      run_mpi_integration ${test} "2x-gpu-v510"
    fi

    run_spark_integration ${test} "2x-gpu-v510"
  fi
done
