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


// Reverse the array in place
__global__ void reverseArrayInPlace(float* a) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float temp;
    
    if(tid<N) {
        temp = a[tid]; // store values per thread
    }
    a[N-1-tid] = temp;
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    float h_a[N], h_c[N];
    float *d_a;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = (rand() % 90) * 0.018f;
    }
    

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
    
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
   
    convolution<<<blocksPerGrid, threadsPerBlock>>>(d_a);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_a, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"Reversed Array result at i:"<<i<<", is: "<<h_c[i]<<", original array: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}