class AdaptiveBatchSize:
    def __init__(self, base_batch_size, scale_factor=2):
        self.base_batch_size = base_batch_size
        self.scale_factor = scale_factor

    def get_scaled_batch_size(self, gpu_utilization):
        return int(self.base_batch_size * (1 + self.scale_factor * gpu_utilization))
