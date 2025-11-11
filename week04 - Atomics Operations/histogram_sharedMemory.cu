#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define N 256   // histogram bins
#define bufferSize 999999 //  data

#define threadsPerBlock 256 // same as number of bins in histogram for efficient computations

#define CHECK_CUDA(call) do {                       \
    cudaError_t e = (call);                         \
    if(e!=cudaSuccess) {                                \
        cerr<<"CUDA Error : "<<cudaGetErrorString(e)    \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl;   \
        exit(1);                                    \
    }                                                   \
} while(0)

// This histogram computation kernel uses shared memory for per block histogram and
// global memory for combining those histogram into final result
// Therefore, read/write to shared memory is very fast
__global__ void kernelHistogramShared(int* a, int* c){

    extern __shared__ int cache[]; // shared memory for each block assigned dynamically during kernel launch
    int cacheIdx = threadIdx.x;   // threadId in each block

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x; // Since we are using fewer number of grids and threads
                                         // we require each thread process multiple datapoints.
    cache[cacheIdx] = 0; // initialize each thread with 0
    __syncthreads(); // ensure all operations done

    // While loop to traverse through the multiple grids
    while(tid<bufferSize) {
        atomicAdd(&cache[a[tid]], 1);
        tid += offset;
    }
    __syncthreads();

    // Once all the blocks completed their histogram computation
    // We can sum the values at corresponding cacheIdx of each block.
    if(cacheIdx<threadsPerBlock) {
        atomicAdd(&c[cacheIdx], cache[cacheIdx]);
    }
}

int main() {

    // initialize and start recording time using CUDA Events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    // define the vector for histogram computation for host and device
    int h_a[bufferSize], h_c[N];
    int *d_a, *d_c;

    // allocate memory for device variable
    CHECK_CUDA(cudaMalloc(&d_a, bufferSize*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(int)));

    // Initialize the vector with data between 0 and 255
    for(int i =0;i<bufferSize;i++) {
        h_a[i] = rand() % 256;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    // transfer data to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bufferSize*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_c, 0, N*sizeof(int)));

    int blocksPerGrid = 48 * 2;  // 48 is number of Multiprocessor in this GPU
                                // And, this number of grids are for best performance

    // assigning dynamic memory size to shared memory
    size_t shmem = (threadsPerBlock)*sizeof(int);

    kernelHistogramShared<<<blocksPerGrid, threadsPerBlock, shmem>>>(d_a, d_c);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy histogram result from device to Host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<<elapsed_time<<endl;  // Elapsed time(in ms) : 0.38

    long sum = 0;
    for(int i = 0 ; i<N;i++){
        sum += h_c[i];
        if(i<50) {
            cout<<"frequency at index i = "<<i<<" is "<<h_c[i]<<endl;
        }
    }
    cout<<"Is total count matching : "<<(999999 == sum)<<endl;

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}