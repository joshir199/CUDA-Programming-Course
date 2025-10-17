#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_fp16.h>  // to handle Floating Point 16 (FP16) of half precision
#include <cuda_runtime.h>
using namespace std;


#define n 999999

#define threadsPerBlock 256  // keeping the dimension 16 x 16 as multiple of warps 32

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

__global__ void dotproduct_kernel(__half* a, float* c,  int N) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float cache[];

    float temp = 0.0f; // per thread value storage

    // get dot product in FP32 precision
    if(tid<N) {
        temp = __half2float(a[tid]) * __half2float(a[tid]);
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    int i = blockDim.x/2;
    while(i>0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i = i/2;
    }

    if(threadIdx.x == 0) {
        atomicAddFloat(c, cache[0]);
    }
    __syncthreads();
}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;

    __half h_a[N]; // define FP16 data
    __half *d_a;
    float *d_c;
    float h_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(float)));
    cudaMemset(d_c, 0, sizeof(float));  // initialize the d_c with 0 for atomic operation

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = __float2half(1.0f);
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(__half), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    size_t shmem = threadsPerBlock * sizeof(float);
    dotproduct_kernel<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, d_c, N);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.95

    cout<<"Final output h_c : "<< h_c<<endl;


    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}