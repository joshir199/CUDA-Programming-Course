#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define n 768
#define m 768
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// matrix addition which adds elements of two matrix at same index
__global__ void matrix_addition(float* a, float* b, float* c, int N, int M) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while(tid < N*M) {
        c[tid] = a[tid] + b[tid];
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    int N = n;
    int M = m;
    float h_a[N*M], h_b[N*M], h_c[N*M];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*M*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N*M*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*M*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N*M ;i++) {
        h_a[i] = ((rand() + i*i - 3*i + 46)%999)*1.0f;
        h_b[i] = ((rand() + i*i - 3*i + 99)%99)*0.08f;
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*M*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*M*sizeof(float), cudaMemcpyHostToDevice));

    int blocksPerGrid = (N*M + threadsPerBlock - 1)/threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N, M);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*M*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;

    for(int i = 0; i< 20 && i<N*M; i++) {
        cout<<"Matrix addition at i:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}