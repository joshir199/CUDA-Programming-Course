#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define chunk 32768
#define threadPerBlock 256

#define CHECK_CUDA(call) do {           \
    cudaError_t e = (call);                 \
    if(e != cudaSuccess) {                      \
        cerr<<"CUDA Error : "<<cudaGetErrorString(e)<<  \
        " in "<<__FILE__<<" at "<<__LINE__<<endl;       \
        exit(1);                                    \
    }                                                   \
} while(0)

// simple kernelA
__global__ void kernelA(int* a, int* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<chunk) {
        c[tid] = a[tid] - tid;
    }
}

// Simple KernelB
__global__ void kernelB(int* c, int* d) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<chunk) {
        d[tid] = c[tid] + 2*tid;
    }
}


int main() {

    // define and check device Overlap properties
    cudaDeviceProp prop;
    int deviceNumber;
    CHECK_CUDA(cudaGetDevice(&deviceNumber));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceNumber));
    if(!prop.deviceOverlap) {
        cout<<"Existing GPU does not support device Overlap";
    }

    // define and create global Event to record timings
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Create memory variable for host and device
    int *h_a = (int*)malloc(chunk*sizeof(int));
    int *h_d = (int*)malloc(chunk*sizeof(int));
    int *d_a, *d_c, *d_d;

    // allocate global memory for device variable
    CHECK_CUDA(cudaMalloc(&d_a, chunk*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, chunk*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_d, chunk*sizeof(int)));

    // allocate pinned memory for host variable
    CHECK_CUDA(cudaMallocHost(&h_a, chunk*sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&h_d, chunk*sizeof(int)));

    // fill some values in input variable
    for(int i=0; i<chunk; i++) {
        h_a[i] = 2*i - 1;
    }

    // define and create 2 streams
    cudaStream_t stream0, stream1;
    CHECK_CUDA(cudaStreamCreate(&stream0));
    CHECK_CUDA(cudaStreamCreate(&stream1));

    // Create events for copy and kernel execution operations to handle dependency
    // only when we change stream between kernelA and kernelB execution
    cudaEvent_t kernelComputeA;
    CHECK_CUDA(cudaEventCreateWithFlags(&kernelComputeA, cudaEventDisableTiming));

    int blockPerGrid = (chunk + threadPerBlock -1)/threadPerBlock;

    CHECK_CUDA(cudaEventRecord(start));

    // Stream0: Do host to device copy of data for kernel execution
    CHECK_CUDA(cudaMemcpyAsync(d_a, h_a, chunk*sizeof(int), cudaMemcpyHostToDevice, stream0));

    // stream0: Do the kernelA execution after the data is copied in device
    // kernelA is dependent on data d_a copy to be finished (run on same stream)
    kernelA<<<blockPerGrid, threadPerBlock, 0, stream0>>>(d_a, d_c);
    CHECK_CUDA(cudaEventRecord(kernelComputeA, stream0)); // start kernelA execution event recording

    // stream1: Do the kernelB execution after kernelA finishes
    // kernelB is dependent on output from kernelA execution (changing streams)
    CHECK_CUDA(cudaStreamWaitEvent(stream1, kernelComputeA, 0)); // wait for kernelA to finish
    kernelB<<<blockPerGrid, threadPerBlock, 0, stream1>>>(d_c, d_d);

    // stream1: Do the result copy from device to Host (same stream)
    CHECK_CUDA(cudaMemcpyAsync(h_d, d_d, chunk*sizeof(int), cudaMemcpyDeviceToHost, stream1));

    // Synchronize the streams
    CHECK_CUDA(cudaStreamSynchronize(stream0));
    CHECK_CUDA(cudaStreamSynchronize(stream1));

    // Record the overall time
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<<elapsed_time<<endl; // 0.098

    for(int i = 0; i< 100; i++) {
        cout<<"Final output at index i = "<<i<<", is "<<h_d[i]<<endl;
    }

    // free the pinned and global memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_d));

    // Destroy the events and streams
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(kernelComputeA));
    CHECK_CUDA(cudaStreamDestroy(stream0));
    CHECK_CUDA(cudaStreamDestroy(stream1));

    return 0;

}