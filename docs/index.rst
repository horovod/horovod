Horovod documentation
=====================
Horovod improves the speed, scale, and resource utilization of deep learning training.

Get started
-----------
Choose your deep learning framework to learn how to get started with Horovod.

.. raw:: html

    <button class="accordion">TensorFlow</button>
    <div class="panel">
      <p>To use Horovod with TensorFlow on your laptop:
         <ol>
            <li><a href="https://www.open-mpi.org/faq/?category=building#easy-build">Install Open MPI 3.1.2 or 4.0.0</a>, or another MPI implementation. </li>
            <li>
               If you've installed TensorFlow from <a href="https://pypi.org/project/tensorflow">PyPI</a>, make sure that the <code>g++-4.8.5</code> or <code>g++-4.9</code> is installed.<br/>
               If you've installed TensorFlow from <a href="https://conda.io">Conda</a>, make sure that the <code>gxx_linux-64</code> Conda package is installed.
            </li>
            <li>Install the Horovod pip package: <code>pip install horovod</code></li>
            <li>Read <a href="https://horovod.readthedocs.io/en/latest/tensorflow.html">Horovod with TensorFlow</a> for best practices and examples. </li>
         </ol>
         Or, use <a href="https://horovod.readthedocs.io/en/latest/gpus_include.html">Horovod on GPUs</a>, in <a href="https://horovod.readthedocs.io/en/latest/spark_include.html">Spark</a>, <a href="https://horovod.readthedocs.io/en/latest/docker_include.html">Docker</a>, <a href="https://github.com/sylabs/examples/tree/master/machinelearning/horovod">Singularity</a>, or Kubernetes (<a href="https://github.com/kubeflow/examples/tree/master/demos/yelp_demo/ks_app/vendor/kubeflow/mpi-job">Kubeflow</a>, <a href="https://github.com/kubeflow/mpi-operator/">MPI Operator</a>, <a href="https://github.com/helm/charts/tree/master/stable/horovod">Helm Chart</a>, and <a href="https://github.com/IBM/FfDL/tree/master/etc/examples/horovod/">FfDL</a>).
      </p>
    </div>

    <button class="accordion">Keras</button>
    <div class="panel">
      <p>To use Horovod with Keras on your laptop:
         <ol>
            <li><a href="https://www.open-mpi.org/faq/?category=building#easy-build">Install Open MPI 3.1.2 or 4.0.0</a>, or another MPI implementation. </li>
            <li>
               If you've installed TensorFlow from <a href="https://pypi.org/project/tensorflow">PyPI</a>, make sure that the <code>g++-4.8.5</code> or <code>g++-4.9</code> is installed.<br/>
               If you've installed TensorFlow from <a href="https://conda.io">Conda</a>, make sure that the <code>gxx_linux-64</code> Conda package is installed.
            </li>
            <li>Install the Horovod pip package: <code>pip install horovod</code></li>
            <li>Read <a href="https://horovod.readthedocs.io/en/latest/keras.html">Horovod with Keras</a> for best practices and examples. </li>
         </ol>
         Or, use <a href="https://horovod.readthedocs.io/en/latest/gpus_include.html">Horovod on GPUs</a>, in <a href="https://horovod.readthedocs.io/en/latest/spark_include.html">Spark</a>, <a href="https://horovod.readthedocs.io/en/latest/docker_include.html">Docker</a>, <a href="https://github.com/sylabs/examples/tree/master/machinelearning/horovod">Singularity</a>, or Kubernetes (<a href="https://github.com/kubeflow/examples/tree/master/demos/yelp_demo/ks_app/vendor/kubeflow/mpi-job">Kubeflow</a>, <a href="https://github.com/kubeflow/mpi-operator/">MPI Operator</a>, <a href="https://github.com/helm/charts/tree/master/stable/horovod">Helm Chart</a>, and <a href="https://github.com/IBM/FfDL/tree/master/etc/examples/horovod/">FfDL</a>).
      </p>
    </div>

    <button class="accordion">PyTorch</button>
    <div class="panel">
      <p>To use Horovod with PyTorch on your laptop:
         <ol>
            <li><a href="https://www.open-mpi.org/faq/?category=building#easy-build">Install Open MPI 3.1.2 or 4.0.0</a>, or another MPI implementation. </li>
            <li>
               If you've installed PyTorch from <a href="https://pypi.org/project/torch">PyPI</a>, make sure that the <code>g++-4.9</code> or above is installed.<br/>
               If you've installed PyTorch from <a href="https://conda.io">Conda</a>, make sure that the <code>gxx_linux-64</code> Conda package is installed.
            </li>
            <li>Install the Horovod pip package: <code>pip install horovod</code></li>
            <li>Read <a href="https://horovod.readthedocs.io/en/latest/pytorch.html">Horovod with PyTorch</a> for best practices and examples. </li>
         </ol>
         Or, use <a href="https://horovod.readthedocs.io/en/latest/gpus_include.html">Horovod on GPUs</a>, in <a href="https://horovod.readthedocs.io/en/latest/spark_include.html">Spark</a>, <a href="https://horovod.readthedocs.io/en/latest/docker_include.html">Docker</a>, <a href="https://github.com/sylabs/examples/tree/master/machinelearning/horovod">Singularity</a>, or Kubernetes (<a href="https://github.com/kubeflow/examples/tree/master/demos/yelp_demo/ks_app/vendor/kubeflow/mpi-job">Kubeflow</a>, <a href="https://github.com/kubeflow/mpi-operator/">MPI Operator</a>, <a href="https://github.com/helm/charts/tree/master/stable/horovod">Helm Chart</a>, and <a href="https://github.com/IBM/FfDL/tree/master/etc/examples/horovod/">FfDL</a>).
      </p>
    </div>

    <button class="accordion">Apache MXNet</button>
    <div class="panel">
      <p>To use Horovod with Apache MXNet on your laptop:
         <ol>
            <li><a href="https://www.open-mpi.org/faq/?category=building#easy-build">Install Open MPI 3.1.2 or 4.0.0</a>, or another MPI implementation. </li>
            <li>Install the Horovod pip package: <code>pip install horovod</code></li>
            <li>Read <a href="https://horovod.readthedocs.io/en/latest/mxnet.html">Horovod with MXNet</a> for best practices and examples. </li>
         </ol>
         Or, use <a href="https://horovod.readthedocs.io/en/latest/gpus_include.html">Horovod on GPUs</a>, in <a href="https://horovod.readthedocs.io/en/latest/spark_include.html">Spark</a>, <a href="https://horovod.readthedocs.io/en/latest/docker_include.html">Docker</a>, <a href="https://github.com/sylabs/examples/tree/master/machinelearning/horovod">Singularity</a>, or Kubernetes (<a href="https://github.com/kubeflow/examples/tree/master/demos/yelp_demo/ks_app/vendor/kubeflow/mpi-job">Kubeflow</a>, <a href="https://github.com/kubeflow/mpi-operator/">MPI Operator</a>, <a href="https://github.com/helm/charts/tree/master/stable/horovod">Helm Chart</a>, and <a href="https://github.com/IBM/FfDL/tree/master/etc/examples/horovod/">FfDL</a>).
      </p>
    </div>

    <script>
        var acc = document.getElementsByClassName("accordion");
        var i;

        for (i = 0; i < acc.length; i++) {
          acc[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var panel = this.nextElementSibling;
            if (panel.style.maxHeight){
              panel.style.maxHeight = null;
            } else {
              panel.style.maxHeight = panel.scrollHeight + "px";
            }
          });
        }
     </script>

Guides
------

.. toctree::
   :maxdepth: 2

   summary_include

   concepts_include

   install_include

   api

   tensorflow

   keras

   pytorch

   mxnet

   running_include

   benchmarks_include

   inference_include

   gpus_include

   docker_include

   spark_include

   lsf_include

   tensor-fusion_include
   
   adasum_user_guide_include

   timeline_include

   autotune_include

   troubleshooting_include

   contributors_include



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
