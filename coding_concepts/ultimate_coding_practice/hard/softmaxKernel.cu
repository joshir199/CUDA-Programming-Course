#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cfloat>
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


__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        // convert assumed bits to float, compare with val, choose greater
        float assumed_f = __int_as_float(assumed);
        float new_f = fmaxf(val, assumed_f);
        int new_i = __float_as_int(new_f);
        // try to CAS swap
        old = atomicCAS(address_as_i, assumed, new_i);
    } while (assumed != old);
    return __int_as_float(old);
}

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

// Get Max value to handle potential overflow issues
__global__ void getMaxValue(float* a, float* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float cache[threadsPerBlock];

    if(tid<N) {
        cache[threadIdx.x] = a[tid];
    } else {
        cache[threadIdx.x] = -FLT_MAX;
    }
    __syncthreads();  // shared memory is getting allocated

    int i = threadsPerBlock/2;
    while(i>0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
        }
        __syncthreads();
        i = i/2;
    }

    // Now get the max from each block through atomicMax operation
    if(threadIdx.x == 0) {
        atomicMaxFloat(c, cache[0]);  // use custom Max function for float values
    }
}


__global__ void softmaxFunction(float* a, float* b, float* c, float* totalSum) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float cache[threadsPerBlock];

    float temp = 0.0f; // per thread value storage

    // load the exponential of each term with handled overflow
    // condition using maximum of each element
    if(tid<N) {
        temp = expf(a[tid] - *b);
        c[tid] = temp;  // store temporary exponential values per element
    }
    cache[threadIdx.x] = temp;
    __syncthreads();

    int i = threadsPerBlock/2;
    while(i>0) {
        if(threadIdx.x < i) {
            cache[threadIdx.x] += cache[threadIdx.x + i];
        }
        __syncthreads();
        i = i/2;
    }

    // get the total sum of each exponential term from each block
    // into single variable (variable should be global rather than local
    // thread because it will reset to zero every iterations)
    if(threadIdx.x == 0) {
        atomicAddFloat(totalSum, cache[0]); //use custom Max function for float values
    }
}

__global__ void normalizeSoftmax(float totalSum, float* c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<N) {
        c[tid] = c[tid]/totalSum;
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    float h_a[N], h_c[N], h_max, h_totalSum;
    float *d_a, *d_b, *d_c, *totalSum;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&totalSum, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = (i % 7) * 0.8f;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));

    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    // Get the max value to handle the overflow of large numbers in exponential function
    getMaxValue<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_max, d_b, sizeof(float), cudaMemcpyDeviceToHost));

    softmaxFunction<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, totalSum);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_totalSum, totalSum, sizeof(float), cudaMemcpyDeviceToHost));

    cout<<"Max Value : "<< h_max<< ", Total Sum: "<< h_totalSum <<endl; // 4.8   && 258469

    // most important condition while doing any division
    if(h_totalSum == 0.0f) {
        h_totalSum += 0.000001; // add epsilon = 10^-5 to avoid division be zero
    }
    normalizeSoftmax<<<blocksPerGrid, threadsPerBlock>>>(h_totalSum, d_c);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;  // 2.7

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"Softmax function result at i:"<<i<<", is: "<<h_c[i]<<", original array: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(totalSum));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}