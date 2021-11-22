# TODO: reducescatter -- migrate to proper file

def test_horovod_reducescatter(self):
    """Test that the allreduce correctly sums and scatters 1D, 2D, 3D tensors."""
    if not hvd.mpi_enabled():
        return  # Reducescatter is not yet implemented in gloo/mlsl

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                          torch.FloatTensor, torch.DoubleTensor])
    if _fp16_supported:
        dtypes += self.filter_supported_types([torch.HalfTensor])
    if torch.cuda.is_available():
        dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.cuda.HalfTensor]
    dims = [1, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([size * 4] * dim)).random_(-100, 100)
        tensor = self.cast_and_place(tensor, dtype)
        summed = hvd.reducescatter(tensor, op=hvd.Sum)
        tensor, summed = self.convert_cpu_fp16_to_fp32(tensor, summed)
        expected = tensor[rank * 4:(rank + 1) * 4] * size

        # Threshold for floating point equality depends on number of
        # ranks, since we're comparing against precise multiplication.
        if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                  torch.cuda.IntTensor, torch.cuda.LongTensor]:
            threshold = 0
        elif size < 10:
            threshold = 1e-4
        elif size < 15:
            threshold = 5e-4
        else:
            break

        assert list(summed.shape) == list(expected.shape)
        max_difference = summed.data.sub(expected).max()
        assert max_difference <= threshold, 'hvd.reducescatter produces incorrect results'


def test_horovod_reducescatter_average(self):
    """Test that the allreduce correctly averages and scatters 1D, 2D, 3D tensors."""
    if not hvd.mpi_enabled():
        return  # Reducescatter is not yet implemented in gloo/mlsl

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                          torch.FloatTensor, torch.DoubleTensor])
    if torch.cuda.is_available():
        dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.cuda.HalfTensor]
    dims = [1, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([size * 4] * dim)).random_(-100, 100)
        tensor = self.cast_and_place(tensor, dtype)
        averaged = hvd.reducescatter(tensor, op=hvd.Average)
        expected = tensor[rank * 4:(rank + 1) * 4]

        # Threshold for floating point equality depends on number of
        # ranks, since we're comparing against precise multiplication.
        if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                  torch.cuda.IntTensor, torch.cuda.LongTensor]:
            threshold = 0
        elif size < 10:
            threshold = 1e-4
        elif size < 15:
            threshold = 5e-4
        else:
            break

        assert list(averaged.shape) == list(expected.shape)
        max_difference = averaged.data.sub(expected).max()
        assert max_difference <= threshold, 'hvd.reducescatter produces incorrect results'


def test_horovod_reducescatter_adasum(self):
    """Test that the reducescatter raises an error if we use Adasum operation."""
    if not hvd.mpi_enabled():
        return  # Reducescatter is not yet implemented in gloo/mlsl

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                          torch.FloatTensor, torch.DoubleTensor])
    if torch.cuda.is_available():
        dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.cuda.HalfTensor]
    dims = [1, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([size * 4] * dim)).random_(-100, 100)
        tensor = self.cast_and_place(tensor, dtype)

        try:
            hvd.reducescatter(tensor, op=hvd.Adasum)
            assert False, 'hvd.reducescatter did not throw error'
        except (torch.FatalError, RuntimeError):
            pass


def test_horovod_reducescatter_async_fused(self):
    """Test that the reducescatter correctly sums 1D, 2D, 3D tensors
    with Tensor Fusion."""
    if not hvd.mpi_enabled():
        return  # Reducescatter is not yet implemented in gloo/mlsl

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()
    dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                                          torch.FloatTensor, torch.DoubleTensor])
    if _fp16_supported:
        dtypes += self.filter_supported_types([torch.HalfTensor])
    if torch.cuda.is_available():
        dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.cuda.HalfTensor]
    dims = [1, 2, 3]
    tests = []
    is_hvd_poll_false_once = False
    for dtype, dim in itertools.product(dtypes, dims):
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([size * 4] * dim)).random_(-100, 100)
        tensor = self.cast_and_place(tensor, dtype)
        handle = hvd.reducescatter_async(tensor, op=hvd.Sum)
        if not hvd.poll(handle):
            is_hvd_poll_false_once = True
        tensor, = self.convert_cpu_fp16_to_fp32(tensor)
        expected = tensor[rank * 4:(rank + 1) * 4] * size
        tests.append((dtype, expected, handle))

    # Make sure it's an asynchronous operation.
    assert is_hvd_poll_false_once, 'hvd.poll() always returns True, not an async op?'

    for dtype, expected, handle in tests:
        summed = hvd.synchronize(handle)
        summed, = self.convert_cpu_fp16_to_fp32(summed)
        assert list(summed.shape) == list(expected.shape)

        # Threshold for floating point equality depends on number of
        # ranks, since we're comparing against precise multiplication.
        if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                  torch.cuda.IntTensor, torch.cuda.LongTensor]:
            threshold = 0
        elif size < 10:
            threshold = 1e-4
        elif size < 15:
            threshold = 5e-4
        else:
            break

        max_difference = summed.sub(expected).max()
        assert max_difference <= threshold, 'hvd.allreduce produces incorrect results'


def test_horovod_reducescatter_error(self):
    """Test that the reducescatter raises an error if different ranks try to
    send tensors of different rank or dimension."""
    if not hvd.mpi_enabled():
        return  # Reducescatter is not yet implemented in gloo/mlsl

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    # This test does not apply if there is only one worker.
    if size == 1:
        return

    # Same rank, different dimension
    torch.manual_seed(1234)
    dims = [17 + rank] * 3
    tensor = torch.FloatTensor(*dims).random_(-100, 100)
    try:
        hvd.reducescatter(tensor)
        assert False, 'hvd.reducescatter did not throw error'
    except (torch.FatalError, RuntimeError):
        pass

    # Same number of elements, different rank
    torch.manual_seed(1234)
    if rank == 0:
        dims = [17, 23 * 57]
    else:
        dims = [17, 23, 57]
    tensor = torch.FloatTensor(*dims).random_(-100, 100)
    try:
        hvd.reducescatter(tensor)
        assert False, 'hvd.reducescatter did not throw error'
    except (torch.FatalError, RuntimeError):
        pass


def test_horovod_reducescatter_type_error(self):
    """Test that the reducescatter raises an error if different ranks try to
    send tensors of different type."""
    if not hvd.mpi_enabled():
        return  # Reducescatter is not yet implemented in gloo/mlsl

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    # This test does not apply if there is only one worker.
    if size == 1:
        return

    # Same rank, different dimension
    dims = [17] * 3
    if rank % 2 == 0:
        tensor = torch.IntTensor(*dims)
    else:
        tensor = torch.FloatTensor(*dims)

    try:
        hvd.reducescatter(tensor)
        assert False, 'hvd.reducescatter did not throw error'
    except (torch.FatalError, RuntimeError):
        pass


def test_horovod_reducescatter_duplicate_name_error(self):
    """Test that the reducescatter raises an error if there are
    two concurrent operations with the same name."""
    if not hvd.mpi_enabled():
        return  # Reducescatter is not yet implemented in gloo/mlsl

    hvd.init()
    size = hvd.size()

    # This test does not apply if there is only one worker.
    if size == 1:
        return

    dims = [17] * 3
    tensor = torch.FloatTensor(*dims)

    hvd.reducescatter_async(tensor, name='duplicate_name')
    try:
        for i in range(10):
            hvd.reducescatter_async(tensor, name='duplicate_name')
        assert False, 'hvd.reducescatter_async did not throw error'
    except (torch.FatalError, ValueError):
        pass


def test_horovod_reducescatter_grad(self):
    """Test the correctness of the reducescatter gradient."""
    if not hvd.mpi_enabled():
        return  # Reducescatter is not yet implemented in gloo/mlsl

    hvd.init()
    size = hvd.size()
    # Only Tensors of floating point dtype can require gradients
    dtypes = [torch.FloatTensor, torch.DoubleTensor]
    if torch.cuda.is_available():
        dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.cuda.HalfTensor]
    dims = [1, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([size * 4] * dim)).random_(-100, 100)
        tensor = self.cast_and_place(tensor, dtype)
        tensor.requires_grad_()
        summed = hvd.reducescatter(tensor, op=hvd.Sum)

        grad_shape = [4] + [size * 4] * (dim - 1)
        summed.backward(self.cast_and_place(torch.ones(grad_shape), dtype))
        grad_out = tensor.grad.data.cpu().numpy()

        expected = np.ones([size * 4] * dim) * size
        err = np.linalg.norm(expected - grad_out)
        self.assertLess(err, 0.00000001,
                        "gradient %s differs from expected %s, "
                        "error: %s" % (grad_out, expected, str(err)))


def test_horovod_reducescatter_grad_average(self):
    """Test the correctness of the reducescatter averaged gradient."""
    if not hvd.mpi_enabled():
        return  # Reducescatter is not yet implemented in gloo/mlsl

    hvd.init()
    size = hvd.size()
    # Only Tensors of floating point dtype can require gradients
    dtypes = [torch.FloatTensor, torch.DoubleTensor]
    if torch.cuda.is_available():
        dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.cuda.HalfTensor]
    dims = [1, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([size * 4] * dim)).random_(-100, 100)
        tensor = self.cast_and_place(tensor, dtype)
        tensor.requires_grad_()
        summed = hvd.reducescatter(tensor, op=hvd.Average)

        grad_shape = [4] + [size * 4] * (dim - 1)
        summed.backward(self.cast_and_place(torch.ones(grad_shape), dtype))
        grad_out = tensor.grad.data.cpu().numpy()

        expected = np.ones([size * 4] * dim)
        err = np.linalg.norm(expected - grad_out)
        self.assertLess(err, 0.00000001,
                        "gradient %s differs from expected %s, "
                        "error: %s" % (grad_out, expected, str(err)))

