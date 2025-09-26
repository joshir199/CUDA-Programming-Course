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
    int *d_a0, *d_b0, *d_c0; // memory buffer for stream0
    int *d_a1, *d_b1, *d_c1; // memory buffer for stream1

    cudaStream_t stream0, stream1;  // define the streams
    CUDA_CHECK(cudaStreamCreate(&stream0)); // create the stream0 variable
    CUDA_CHECK(cudaStreamCreate(&stream1)); // create the stream1 variable

    // allocate the memory to the device variable for only Chunk Size (N)
    CUDA_CHECK(cudaMalloc(&d_a0, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b0, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c0, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_a1, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_b1, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c1, N*sizeof(int)));

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
    // Now, In each iteration of loop, we can process 2*N values by 2 streams combined
    for(int i=0; i<Size; i+= 2*N) {
        // ************************   stream0 calls   *************************
        // copy 1 for variable h_a starting from "i-th" index
        CUDA_CHECK(cudaMemcpyAsync(d_a0, h_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0));

        // copy 1 for variable h_b starting from "i-th" index
        CUDA_CHECK(cudaMemcpyAsync(d_b0, h_b+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0));

        // stream0 call the kernel to do the required task and make output
        kernelChunkSum<<<(N+255)/256, 256, 0, stream0>>>(d_a0, d_b0, d_c0);

        // copy 1 for variable h_c (output) starting from "i-th" index
        CUDA_CHECK(cudaMemcpyAsync(h_c+i, d_c0, N*sizeof(int), cudaMemcpyDeviceToHost, stream0));

        // ****************  stream1 calls   *******************************
        // copy 1 for variable h_a starting from "(i + N)-th" index
        CUDA_CHECK(cudaMemcpyAsync(d_a1, h_a+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1));

        // copy 1 for variable h_b starting from "(i + N)-th" index
        CUDA_CHECK(cudaMemcpyAsync(d_b1, h_b+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1));

        // stream1 call the kernel to do the required task and make output
        kernelChunkSum<<<(N+255)/256, 256, 0, stream1>>>(d_a1, d_b1, d_c1);

        // copy 1 for variable h_c (output) starting from "(i + N)-th" index
        CUDA_CHECK(cudaMemcpyAsync(h_c+i+N, d_c1, N*sizeof(int), cudaMemcpyDeviceToHost, stream1));
    }

    // call to synchronize the GPU computation before doing any CPU Host operation
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaStreamSynchronize(stream1));

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    cout<<"Elapsed time(in ms) : "<<elapsed_time<<endl; // Elapsed time(in ms) : 3.92611

    for(int i = 0; i<100;i++){
        cout<<" Vector Sum result: "<< h_c[i] <<endl;
    }

    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));
    CUDA_CHECK(cudaFree(d_a0));
    CUDA_CHECK(cudaFree(d_b0));
    CUDA_CHECK(cudaFree(d_c0));
    CUDA_CHECK(cudaFree(d_a1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_c1));
    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}