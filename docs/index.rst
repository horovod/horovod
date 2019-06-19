Horovod documentation
=====================
Horovod improves the speed, scale, and resource utilization of deep learning training.

Get started
-----------
Choose your deep learning framework to learn how to get started with Horovod.

.. raw:: html

    <button class="accordion">Keras</button>
    <div class="panel">
      <p>To install Horovod on your laptop:
         <ol>
            <li><a href="https://www.open-mpi.org/faq/?category=building#easy-build">Install Open MPI</a> or another MPI implementation. </li>
            <li>Install the Horovod pip package: <code>pip install horovod</code></li>
            <li>Read how to use Horovod with Keras (link to Using Horovod with Keras). </li>
         </ol>
         Or, use <a href="https://horovod.readthedocs.io/en/latest/gpus_include.html">Horovod on GPUs</a>, in <a href="https://horovod.readthedocs.io/en/latest/spark_include.html">Spark</a>, in <a href="https://horovod.readthedocs.io/en/latest/docker_include.html">Docker</a>, <a href="https://github.com/sylabs/examples/tree/master/machinelearning/horovod">Singularity</a>, or in Kubernetes.
      </p>
    </div>

    <button class="accordion">MXNet</button>
    <div class="panel">
      <p>Lorem ipsum...</p>
    </div>

    <button class="accordion">PyTorch</button>
    <div class="panel">
      <p>Lorem ipsum...</p>
    </div>

    <button class="accordion">TensorFlow</button>
    <div class="panel">
      <p>Lorem ipsum...</p>
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

   mpirun

   api

   concepts_include

   running_include

   benchmarks_include

   docker_include

   gpus_include

   inference_include

   spark_include

   tensor-fusion_include

   timeline_include

   troubleshooting_include


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
