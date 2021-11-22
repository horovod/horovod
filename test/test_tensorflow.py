

# TODO: Move reducescatter tests to proper file

    def test_horovod_reducescatter_cpu(self):
        """Test on CPU that the reducescatter correctly sums and scatters 1D, 2D, 3D tensors."""
        if not hvd.mpi_enabled():
            return # Reducescatter is not yet implemented in gloo/mlsl

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensor = self.random_uniform(
                    [size * 4] * dim, -100, 100, dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)
            expected = tensor[rank * 4:(rank + 1) * 4] * size
            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_cpu_fused(self):
        """Test on CPU that the reducescatter correctly sums and scatters 1D, 2D, 3D tensors
        with Tensor Fusion."""
        if not hvd.mpi_enabled():
            return # Reducescatter is not yet implemented in gloo/mlsl

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensor = self.random_uniform(
                    [size * 4] * dim, -100, 100, dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)
            expected = tensor[rank * 4:(rank + 1) * 4] * size
            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            test = max_difference <= threshold
            tests.append(test)
        self.assertTrue(self.evaluate(tf.reduce_all(tests)),
                        "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_gpu(self):
        """Test that the reducescatter works on GPUs."""
        if not hvd.mpi_enabled():
            return # Reducescatter is not yet implemented in gloo/mlsl

        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            return

        if os.environ.get('HOROVOD_MIXED_INSTALL'):
            # Skip if compiled with CUDA but without HOROVOD_GPU_REDUCESCATTER.
            return

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [size * 4] * dim, -100, 100, dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)
            expected = tensor[rank * 4:(rank + 1) * 4] * size
            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                return

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter on GPU produces incorrect results")

    def test_horovod_reducescatter_gpu_fused(self):
        """Test that the reducescatter works on GPUs with Tensor Fusion.

        This test will crash badly if used with an MPI implementation that does
        not support GPU memory transfers directly, as it will call MPI_Send on
        a GPU data pointer."""
        if not hvd.mpi_enabled():
            return # Reducescatter is not yet implemented in gloo/mlsl

        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            return

        if os.environ.get('HOROVOD_MIXED_INSTALL'):
            # Skip if compiled with CUDA but without HOROVOD_GPU_REDUCESCATTER.
            return

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [size * 4] * dim, -100, 100, dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)
            expected = tensor[rank * 4:(rank + 1) * 4] * size
            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                return

            test = max_difference <= threshold
            tests.append(test)
        self.assertTrue(self.evaluate(tf.reduce_all(tests)),
                        "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_error(self):
        """Test that the reducescatter raises an error if different ranks try to
        send tensors of different rank or dimension."""
        if not hvd.mpi_enabled():
            return # Reducescatter is not yet implemented in gloo/mlsl

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        # Same rank, different dimension
        dims = [17 + rank] * 3
        tensor = self.random_uniform(dims, -1.0, 1.0)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.reducescatter(tensor))

        # Same number of elements, different rank
        if rank == 0:
            dims = [17, 23 * 57]
        else:
            dims = [17, 23, 57]
        tensor = self.random_uniform(dims, -1.0, 1.0)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.reducescatter(tensor))

    def test_horovod_reducescatter_type_error(self):
        """Test that the reducescatter raises an error if different ranks try to
        send tensors of different type."""
        if not hvd.mpi_enabled():
            return # Reducescatter is not yet implemented in gloo/mlsl

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        # Same rank, different dimension
        dims = [17] * 3
        tensor = tf.ones(dims,
                         dtype=tf.int32 if rank % 2 == 0 else tf.float32)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.reducescatter(tensor))

    def test_horovod_reducescatter_grad_cpu(self):
        """Test the correctness of the reducescatter gradient on CPU."""
        if not hvd.mpi_enabled():
            return # Reducescatter is not yet implemented in gloo/mlsl

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(self.random_uniform(
                        [size * 4] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        summed = hvd.reducescatter(tensor, op=hvd.Sum)
                else:
                    tensor = self.random_uniform(
                        [size * 4] * dim, -100, 100, dtype=dtype)
                    summed = hvd.reducescatter(tensor, op=hvd.Sum)

                grad_ys = tf.ones([4] + [size * 4] * (dim - 1))
                if _executing_eagerly():
                    grad_out = tape.gradient(summed, tensor, grad_ys)
                else:
                    grad = tf.gradients(summed, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([size * 4] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_reducescatter_grad_gpu(self):
        """Test the correctness of the reducescatter gradient on GPU."""
        if not hvd.mpi_enabled():
            return # Reducescatter is not yet implemented in gloo/mlsl

        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            return

        if os.environ.get('HOROVOD_MIXED_INSTALL'):
            # Skip if compiled with CUDA but without HOROVOD_GPU_REDUCESCATTER.
            return

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(
                        self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        summed = hvd.reducescatter(tensor, op=hvd.Sum)
                else:
                    tensor = self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype)
                    summed = hvd.reducescatter(tensor, op=hvd.Sum)

                grad_ys = tf.ones([4] + [size * 4] * (dim - 1))
                if _executing_eagerly():
                    grad_out = tape.gradient(summed, tensor, grad_ys)
                else:
                    grad = tf.gradients(summed, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([size * 4] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))


