#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define N 1000

#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                       \
    cudaError_t e = (call);                         \
    if(e!=cudaSuccess) {                                \
        cerr<<"CUDA Error : "<<cudaGetErrorString(e)    \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl;   \
        exit(1);                                    \
    }                                                   \
} while(0)

// this uses atomicCAS operation function to find the winner thread doing the first operation
__global__ void kernelGetWinnerThread(int *winner) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Attempt to set winner only if it’s still 5
    int old = atomicCAS(winner, 5, tid);

    // Only the winning thread will see old == 5
    if (old == 5) {
        printf("Thread %d (Block %d, Thread %d) won!\n",
               tid, blockIdx.x, threadIdx.x);
    }
}


int main() {

    // initialize and start recording time using CUDA Events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    int blockPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    // define the vector for histogram computation for host and device
    int h_a[N];
    int *d_a;

    // allocate memory for device variable
    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(int)));

    // Initialize the vector
    for(int i =0;i<N;i++) {
        h_a[i] = 5+i; // first element is 5 which will be access & modified in CUDA device
    }

    // transfer data to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));

    kernelGetWinnerThread<<<blockPerGrid, threadsPerBlock>>>(d_a);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<<elapsed_time<<endl;  // Elapsed time(in ms) : 11.8983


    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}