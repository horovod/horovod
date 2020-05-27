#!/bin/bash

# exit immediately on failure, or if an undefined variable is used
set -eu

# our repository in AWS
repository=823773083436.dkr.ecr.us-east-1.amazonaws.com/buildkite

# list of all the tests
tests=( \
       test-cpu-openmpi-py2_7-tf1_1_0-keras2_0_0-torch0_4_0-mxnet1_4_1-pyspark2_3_2 \
       test-cpu-openmpi-py3_6-tf1_1_0-keras2_0_0-torch0_4_0-mxnet1_4_1-pyspark2_3_2 \
       test-cpu-openmpi-py2_7-tf1_6_0-keras2_1_2-torch0_4_1-mxnet1_4_1-pyspark2_3_2 \
       test-cpu-openmpi-py3_6-tf1_6_0-keras2_1_2-torch0_4_1-mxnet1_4_1-pyspark2_3_2 \
       test-cpu-gloo-py2_7-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-gloo-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-gloo-py3_7-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-gloo-py3_8-tf2_2_0-keras2_3_1-torch1_5_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-openmpi-py3_6-tf1_14_0-keras2_2_4-torch1_2_0-mxnet1_4_1-pyspark2_4_0 \
       test-cpu-openmpi-py3_6-tf2_0_0-keras2_3_1-torch1_5_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-openmpi-py3_6-tfhead-kerashead-torchhead-mxnethead-pyspark2_4_0 \
       test-cpu-mpich-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-oneccl-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
       test-cpu-oneccl-ofi-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
       test-gpu-openmpi-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_4_1-pyspark2_4_0 \
       test-gpu-gloo-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_4_1-pyspark2_4_0 \
       test-gpu-openmpi-gloo-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_4_1-pyspark2_4_0 \
       test-gpu-openmpi-py3_6-tf2_2_0-keras2_3_1-torch1_5_0-mxnet1_6_0-pyspark2_4_0 \
       test-gpu-openmpi-py3_6-tfhead-kerashead-torchhead-mxnethead-pyspark2_4_0 \
       test-mixed-openmpi-py3_6-tf1_15_0-keras2_3_1-torch1_4_0-mxnet1_5_0-pyspark2_4_0 \
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
  echo "  timeout_in_minutes: 10"
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

  local exclude_keras_if_needed=""
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
    # TODO: support for Keras + TF 2.0 and TF-Keras 2.0
    exclude_keras_if_needed="| sed 's/test_keras.py//g' | sed 's/test_tensorflow_keras.py//g'"
  else
    exclude_keras_if_needed="| sed 's/[a-z_]*tensorflow2[a-z_.]*//g'"
  fi

  local exclude_interactiverun="| sed 's/test_interactiverun.py//g' | sed 's/test_spark_keras.py//g' | sed 's/test_spark_torch.py//g'"

  # Spark and Run test does not need to be executed with horovodrun, but we still run it below.
  local exclude_standalone_test="| sed 's/test_spark.py//g' | sed 's/test_run.py//g'"
  local standalone_tests="test_spark.py test_run.py"

  # TODO(travis): enable for Python 3.8 when Spark 3.0 released
  #  see: https://issues.apache.org/jira/browse/SPARK-29536
  if [[ ${test} == *"-py3_8-"* ]]; then
      standalone_tests="test_run.py"
  fi

  # pytests have 4x GPU use cases and require a separate queue
  run_test "${test}" "${queue}" \
    ":pytest: Run PyTests (${test})" \
    "bash -c \"${oneccl_env} cd /horovod/test && (echo test_*.py ${exclude_keras_if_needed} ${exclude_interactiverun} ${exclude_standalone_test} | xargs -n 1 \\\$(cat /mpirun_command) pytest -v --capture=no) && pytest --forked -v --capture=no ${standalone_tests}\""
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
      "bash -c \"cd /horovod/test && pytest -v --capture=no test_interactiverun.py\""
  fi

  # Legacy TensorFlow tests
  if [[ ${test} != *"tf2_"* ]] && [[ ${test} != *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: Test TensorFlow MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/tensorflow_mnist.py\""

    if [[ ${test} != *"tf1_1_0"* && ${test} != *"tf1_6_0"* ]]; then
      run_test "${test}" "${queue}" \
        ":tensorflow: Test TensorFlow Eager MNIST (${test})" \
        "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/tensorflow_mnist_eager.py\""
    fi

    run_test "${test}" "${queue}" \
      ":tensorflow: Test Keras MNIST (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/keras_mnist_advanced.py\""
  fi

  run_test "${test}" "${queue}" \
    ":python: Test PyTorch MNIST (${test})" \
    "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/examples/pytorch_mnist.py\""

  run_test "${test}" "${queue}" \
    ":muscle: Test MXNet MNIST (${test})" \
    "bash -c \"${oneccl_env} OMP_NUM_THREADS=1 \\\$(cat /mpirun_command) python /horovod/examples/mxnet_mnist.py\""

  # tests that should be executed only with the latest release since they don't test
  # a framework-specific functionality
  if [[ ${test} == *"tf1_14_0"* ]]; then
    run_test "${test}" "${queue}" \
      ":muscle: Test Stall (${test})" \
      "bash -c \"${oneccl_env} \\\$(cat /mpirun_command) python /horovod/test/test_stall.py\""

    if [[ ${test} == *"openmpi"* ]]; then
      run_test "${test}" "${queue}" \
        ":terminal: Test Horovodrun (${test})" \
        "horovodrun -np 2 -H localhost:2 python /horovod/examples/tensorflow_mnist.py"
      run_test "${test}" "${queue}" \
        ":terminal: Test Horovodrun (${test})" \
        "echo 'localhost slots=2' > hostfile" \
        "horovodrun -np 2 -hostfile hostfile python /horovod/examples/mxnet_mnist.py"
    fi
  fi

  # TensorFlow 2.0 tests
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: Test TensorFlow 2.0 MNIST (${test})" \
      "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/tensorflow2_mnist.py\""

    run_test "${test}" "${queue}" \
      ":tensorflow: Test TensorFlow 2.0 Keras MNIST (${test})" \
      "bash -c \"\\\$(cat /mpirun_command) python /horovod/examples/tensorflow2_keras_mnist.py\""
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

  local exclude_keras_if_needed=""
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
    # TODO: support for Keras + TF 2.0 and TF-Keras 2.0
    exclude_keras_if_needed="| sed 's/test_keras.py//g' | sed 's/test_tensorflow_keras.py//g'"
  else
    exclude_keras_if_needed="| sed 's/[a-z_]*tensorflow2[a-z_.]*//g'"
  fi

  # These are tested as integration style tests.
  local excluded_tests="| sed 's/test_interactiverun.py//g' | sed 's/test_spark_keras.py//g' | sed 's/test_spark_torch.py//g'"

  # Spark and Run test does not need to be executed with horovodrun, but we still run it below.
  local exclude_standalone_test="| sed 's/test_spark.py//g' | sed 's/test_run.py//g'"
  local standalone_tests="test_spark.py test_run.py"

  # TODO(travis): enable for Python 3.8 when Spark 3.0 released
  #  see: https://issues.apache.org/jira/browse/SPARK-29536
  if [[ ${test} == *"-py3_8-"* ]]; then
      standalone_tests="test_run.py"
  fi

  run_test "${test}" "${queue}" \
    ":pytest: Run PyTests (${test})" \
    "bash -c \"cd /horovod/test && (echo test_*.py ${exclude_keras_if_needed} ${excluded_tests} ${exclude_standalone_test} | xargs -n 1 horovodrun -np 2 -H localhost:2 --gloo pytest -v --capture=no) && pytest --forked -v --capture=no ${standalone_tests}\""
}

run_gloo_integration() {
  local test=$1
  local queue=$2

  # TensorFlow 2.0 tests
  if [[ ${test} == *"tf2_"* ]] || [[ ${test} == *"tfhead"* ]]; then
    run_test "${test}" "${queue}" \
      ":tensorflow: Test TensorFlow 2.0 MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/tensorflow2_mnist.py"

    run_test "${test}" "${queue}" \
      ":tensorflow: Test TensorFlow 2.0 Keras MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/tensorflow2_keras_mnist.py"
  else
    run_test "${test}" "${queue}" \
      ":tensorflow: Test TensorFlow MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/tensorflow_mnist.py"

    run_test "${test}" "${queue}" \
      ":tensorflow: Test Keras MNIST (${test})" \
      "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/keras_mnist_advanced.py"
  fi

  run_test "${test}" "${queue}" \
    ":python: Test PyTorch MNIST (${test})" \
    "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/pytorch_mnist.py"

  run_test "${test}" "${queue}" \
    ":muscle: Test MXNet MNIST (${test})" \
    "horovodrun -np 2 -H localhost:2 --gloo python /horovod/examples/mxnet_mnist.py"
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
  if [[ ${test} != *"tf1_1_0"* && ${test} != *"tf1_6_0"* && ${test} != *"torch0_"* && ${test} != *"mpich"* && ${test} != *"oneccl"* ]]; then
    if [[ ${test} != *"tf2"* && ${test} != *"tfhead"* ]]; then
      if [[ ! (  ${test} == *"py2"* && ${test} == *"gloo"* && ${test} != *"openmpi-gloo"* ) ]]; then
        run_test "${test}" "${queue}" \
          ":spark: Spark Keras Rossmann Run (${test})" \
          "bash -c \"OMP_NUM_THREADS=1 python /horovod/examples/keras_spark_rossmann_run.py --num-proc 2 --data-dir file:///data --epochs 3 --sample-rate 0.01\""

        run_test "${test}" "${queue}" \
          ":spark: Spark Keras Rossmann Estimator (${test})" \
          "bash -c \"OMP_NUM_THREADS=1 python /horovod/examples/keras_spark_rossmann_estimator.py --num-proc 2 --work-dir /work --data-dir file:///data --epochs 3 --sample-rate 0.01\""

        run_test "${test}" "${queue}" \
          ":spark: Spark Keras MNIST (${test})" \
          "bash -c \"OMP_NUM_THREADS=1 python /horovod/examples/keras_spark_mnist.py --num-proc 2 --work-dir /work --data-dir /data --epochs 3\""
      fi

      if [[ ${queue} != *gpu* ]]; then
        run_test "${test}" "${queue}" \
          ":spark: PyTests Spark Estimators (${test})" \
          "bash -c \"cd /horovod/test && pytest --forked -v --capture=no test_spark_keras.py test_spark_torch.py\""
      fi
    fi

    if [[ ! (  ${test} == *"py2"* && ${test} == *"gloo"* && ${test} != *"openmpi-gloo"* ) ]]; then
      run_test "${test}" "${queue}" \
        ":spark: Spark Torch MNIST (${test})" \
        "bash -c \"OMP_NUM_THREADS=1 python /horovod/examples/pytorch_spark_mnist.py --num-proc 2 --work-dir /work --data-dir /data --epochs 3\""
    fi
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
      "bash -c \"${oneccl_env} python /horovod/examples/keras_mnist_advanced.py --epochs 3 --batch-size 64\""
  fi

  run_test "${test}" "${queue}" \
    ":python: Single PyTorch MNIST (${test})" \
    "bash -c \"${oneccl_env} python /horovod/examples/pytorch_mnist.py --epochs 3\""

  run_test "${test}" "${queue}" \
    ":muscle: Single MXNet MNIST (${test})" \
    "bash -c \"${oneccl_env} python /horovod/examples/mxnet_mnist.py --epochs 3\""
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
    # TODO(travis): enable for Python 3.8 when Spark 3.0 released
    #  see: https://issues.apache.org/jira/browse/SPARK-29536
    if [[ ${test} != *"-py3_8-"* ]]; then
        run_spark_integration ${test} "cpu"
    fi

    # no runner application, world size = 1
    run_single_integration ${test} "cpu" ${oneccl_env}
  fi
done

# wait for all cpu unit and integration tests to finish
echo "- wait"

# run 4x gpu unit tests
for test in ${tests[@]}; do
  if [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]]; then
    # if gloo is specified, run gloo gpu unit tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo_pytest ${test} "4x-gpu-g4"
    fi

    # if mpi is specified, run mpi gpu unit tests
    if [[ ${test} == *mpi* ]]; then
      run_mpi_pytest ${test} "4x-gpu-g4"
    fi
  fi
done

# wait for all gpu unit tests to finish
echo "- wait"

# run 2x gpu integration tests
for test in ${tests[@]}; do
  if [[ ${test} == *-gpu-* ]] || [[ ${test} == *-mixed-* ]]; then
    # if gloo is specified, run gloo gpu integration tests
    if [[ ${test} == *-gloo* ]]; then
      run_gloo_integration ${test} "2x-gpu-g4"
    fi

    # if mpi is specified, run mpi gpu integration tests
    if [[ ${test} == *mpi* ]]; then
      run_mpi_integration ${test} "2x-gpu-g4"
    fi

    run_spark_integration ${test} "2x-gpu-g4"
  fi
done
