#include <stdint.h>

void CudaDotProductImpl(int count, const double* device_a, const double* device_b, 
						double* device_normsq_a, double* device_normsq_b, double* device_dot, double& host_normsq_a, double& host_normsq_b, double& host_dot);

void CudaDotProductImpl(int count, const float* device_a, const float* device_b, 
						double* device_normsq_a, double* device_normsq_b, double* device_dot, double& host_normsq_a, double& host_normsq_b, double& host_dot);

void CudaDotProductImpl(int count, const uint16_t* device_a, const uint16_t* device_b, 
						double* device_normsq_a, double* device_normsq_b, double* device_dot, double& host_normsq_a, double& host_normsq_b, double& host_dot);

void CudaScaleAddImpl(int count, double* a_device, const double* b_device, double host_a_coeff, double host_b_coeff);

void CudaScaleAddImpl(int count, float* a_device, const float* b_device, double host_a_coeff, double host_b_coeff);

void CudaScaleAddImpl(int count, uint16_t* a_device, const uint16_t* b_device, double host_a_coeff, double host_b_coeff);

template<typename T>
void MsCudaPairwiseReduce(int count, T* device_a, T* device_b){
		double normsq_a = 0.f;
		double normsq_b = 0.f;
		double dot = 0.f;

		double* device_normsq_a, * device_normsq_b, * device_dot;
		cudaMalloc(&device_normsq_a, sizeof(double));
		cudaMalloc(&device_normsq_b, sizeof(double));
		cudaMalloc(&device_dot, sizeof(double));

		CudaDotProductImpl(count, device_a, device_b, device_normsq_a, device_normsq_b, device_dot, normsq_a, normsq_b, dot);

		cudaFree(device_normsq_a);
		cudaFree(device_normsq_b);
		cudaFree(device_dot);

		double a_coeff = 1;
		double b_coeff = 1;           
		if (normsq_a != 0) 
			a_coeff = 1.0 - dot / normsq_a * 0.5;                                                                                                                                                                                                                      
		if (normsq_b != 0)
			b_coeff = 1.0 - dot / normsq_b * 0.5;

		CudaScaleAddImpl(count, device_a, device_b, a_coeff, b_coeff);
}


