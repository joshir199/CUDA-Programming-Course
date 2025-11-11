#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define N 999999
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


__device__ float atomicAddFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        // convert assumed bits to float, compare with val, choose greater
        float assumed_f = __int_as_float(assumed);
        float new_f = val + assumed_f;
        int new_i = __float_as_int(new_f);
        // try to CAS swap
        old = atomicCAS(address_as_i, assumed, new_i);
    } while (assumed != old);
    return __int_as_float(old);
}


// Calculate mean squared for each elements.
__global__ void rmsKernel(float* a, float* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float cache[];

    if(tid<N) {
        cache[threadIdx.x] = a[tid] * a[tid] / N;
    } else {
        cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // calculate per block sum
    int i = blockDim.x/2;
    while(i>0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i=i/2;
    }
    // accumulate total mean squared sum for all elements
    if(threadIdx.x == 0) {
        atomicAddFloat(c, cache[0]);
    }
}

// calculate the rms normalisation per element using the formula
// rms = sqrt(mean_square + epsilon)
// rms_normalised = gamma * x[i] / rms   +  beta;
__global__ void rmsNormalisation(float* a, float* b, float* c, float gamma, float beta, float eps) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < N) {
        c[tid] = rsqrtf(*b + eps) * gamma * a[tid] + beta;
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int blockPerGrid = (N + threadsPerBlock - 1)/ threadsPerBlock;
    float gamma = 0.9;
    float beta = 0.05;
    float eps = 0.00001f;

    float h_a[N], h_c[N];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(float)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = (rand() % 90) * 0.018f - (rand() % 90) * 0.05f;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));

    size_t shmem = (threadsPerBlock) * sizeof(float);
    rmsKernel<<<blockPerGrid, threadsPerBlock, shmem>>>(d_a, d_b);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    rmsNormalisation<<<blockPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, gamma, beta, eps);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 2.63

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"RMS Normalisation result at i:"<<i<<", is: "<<h_c[i]<<", original array: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}