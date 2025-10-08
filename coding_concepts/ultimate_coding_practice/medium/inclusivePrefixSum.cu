#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define N 999999
#define threadsPerBlock 32

#define CHECK_CUDA(call) do {            \
    cudaError_t e = (call);                 \
    if(e != cudaSuccess) {                          \
        cerr<<"CUDA Error:"<<cudaGetErrorString(e)  \
        <<" in "<<__FILE__<<", at "<<__LINE__<<endl;    \
        exit(1);                                        \
    }                                               \
} while(0)


// First SCAN: for calculating per block prefix sum
// First Part of Kogge stone Algorithm
// First scan is done on original input array
__global__ void prefixSumKernelScan1(int* a, int* c, int* b) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int cache[threadsPerBlock];
    if(tid<N) {
        cache[threadIdx.x] = a[tid];
    }
    __syncthreads();

    // incremental thread scan with  2^(i-1) threads idle at i-th step starting at index 0.
    int i = 1;
    while(i<blockDim.x) {
        if(threadIdx.x >= i) {
            cache[threadIdx.x] += cache[threadIdx.x - i];
        }
        __syncthreads();

        i = i*2;
    }


    // copy the inclusive prefix sum per block
    if(tid<N) {
        c[tid] = cache[threadIdx.x];
    }

    // copy the accumulated total sum per block
    if( threadIdx.x == blockDim.x - 1) {
        b[blockIdx.x] = cache[threadIdx.x];
    }
}

// Second SCAN: for calculating prefix sum for per block sum
// Second Part of Kogge stone Algorithm
// Second scan is done on total sum per block array with size blocksPerGrid
__global__ void prefixSumKernelScan2(int* b, int size) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int cache[threadsPerBlock];

    if(tid<size) {
        cache[threadIdx.x] = b[tid];
    } else {
        cache[threadIdx.x] = 0;
    }
    __syncthreads();

    // incremental thread scan with  2^(i-1) threads idle at i-th step starting at index 0.
    int i = 1;
    while(i<blockDim.x) {
        if(threadIdx.x >= i) {
            cache[threadIdx.x] += cache[threadIdx.x - i];
        }
        __syncthreads();

        i = i*2;
    }

    // copy the inclusive prefix sum for total block sum
    if(tid<size) {
        b[tid] = cache[threadIdx.x];
    }
}


// Third Add: for adding the block sum with each element to get final prefix sum
// Third Part of Kogge stone Algorithm
// Here, Add is done on each element by summing the total sum values from previous blocks
__global__ void prefixSumKernelAdd(int* a, int* c, int* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // a : per block prefix sum
    // b : per block total cummulative sum
    // for all blocks from block Id = 1, add cache[threadIdx.x] value to elements
    if(tid<N) {
        if(blockIdx.x == 0) {
            c[tid] = a[tid];
        } else {
            c[tid] = a[tid] + b[blockIdx.x - 1];
        }
    }
}



// Algorithm uses SSA (Scan-Scan-Add) steps to calculate inclusive prefix sum
// This is based on Kogge-Stone Parallel Scan
int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int blockPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    int h_a[N], h_c[N], h_b[blockPerGrid];
    int *d_a, *d_c, *d_b, *d_d;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_b, blockPerGrid*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_d, N*sizeof(int)));

    for(int i=0; i<N; i++) {
        h_a[i] = 1;
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    //Scan 1 kernel
    prefixSumKernelScan1<<<blockPerGrid, threadsPerBlock>>>(d_a, d_c, d_b);

    // Scan 2 kernel
    prefixSumKernelScan2<<<(blockPerGrid + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(d_b, blockPerGrid);

    CHECK_CUDA(cudaMemcpy(h_b, d_b, blockPerGrid*sizeof(int), cudaMemcpyDeviceToHost));

    // Add kernel
    prefixSumKernelAdd<<<blockPerGrid, threadsPerBlock>>>(d_c, d_d, d_b);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_d, N*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(im ms) for kernel execution is "<<elapsed_time<<endl;  // 1.6

    for(int i=0; i< 5; i++) {
        cout<<"Scanned block sum array at i="<<i<<", is "<<h_b[i]<<endl;
    }

    for(int i=0; i< 50 && i<N; i++) {
        cout<<"Inclusive Prefix sum array at i="<<i<<", is "<<h_c[i]<<", before: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}