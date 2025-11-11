#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define N 99999
#define Size 99999*10

#define CUDA_CHECK(call) do {         \
    cudaError_t e = (call);             \
    if(e!=cudaSuccess){                     \
        cerr<<"CUDA Error: "<<cudaGetErrorString(e)     \
        <<" in "<<__FILE__<<" at "<<__FILE__<<endl;     \
        exit(1);                                           \
    }                                           \
} while(0)


__global__ void kernelChunkSum(int *a, int *b, int *c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {

    // Check if the device supports the device overlap or not
    cudaDeviceProp prop;
    int whichDevice;
    CUDA_CHECK(cudaGetDevice(&whichDevice));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, whichDevice));
    if( !prop.deviceOverlap) {
        cout<<"The Existing Device does not support overlap feature required for streams"<<endl;
        return 0;
    }
    cout<<"The GPU supports overlap feature"<<endl;

    //define the Event variables for recording time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    // define the host and device variables for memory allocation
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    cudaStream_t stream;  // define the stream
    CUDA_CHECK(cudaStreamCreate(&stream)); // create the stream variable

    // allocate the memory to the device variable for only Chunk Size (N)
    CUDA_CHECK(cudaMalloc(&d_a, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c, N*sizeof(int)));

    // allocate the pinned memory to the Host variable to support Asynchronous copy
    CUDA_CHECK(cudaHostAlloc(&h_a, Size*sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_b, Size*sizeof(int), cudaHostAllocDefault));
    CUDA_CHECK(cudaMallocHost(&h_c, Size*sizeof(int)));

    // fill the random data
    for(int i=0;i<Size;i++){
        h_a[i] = i;
        h_b[i] = 2*i - 1;
    }

    // Now, create multiple operations to be passed into the stream
    // for the execution to take place in asynchronous manner
    // Here, we divide the overall array into multiple chunks
    for(int i=0; i<Size; i+=N) {
        // copy 1 for variable h_a starting from "i-th" index
        CUDA_CHECK(cudaMemcpyAsync(d_a, h_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream));

        // copy 1 for variable h_b starting from "i-th" index
        CUDA_CHECK(cudaMemcpyAsync(d_b, h_b+i, N*sizeof(int), cudaMemcpyHostToDevice, stream));

        // call the kernel to do the required task and make output
        kernelChunkSum<<<(N+255)/256, 256, 0, stream>>>(d_a, d_b, d_c);

        // copy 1 for variable h_c (output) starting from "i-th" index
        CUDA_CHECK(cudaMemcpyAsync(h_c+i, d_c, N*sizeof(int), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    cout<<"Elapsed time(in ms) : "<<elapsed_time<<endl; // Elapsed time(in ms) : 4.06733

    for(int i = 0; i<100;i++){
        cout<<" Vector Sum result: "<< h_c[i] <<endl;
    }

    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}