#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define n 999999

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


__global__ void value_clipping(float* a, float* c, float low, float high, int N){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid<N){
        float temp = a[tid] < low ? low : a[tid];
        c[tid] = temp > high ? high : temp;
    }
}


int main() {

    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    int N=n;
    float h_a[N], h_c[N], h_low, h_high;
    float *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(float)));

    for(int i=0;i<N;i++){
        h_a[i] = (rand()%15 + 4.0f)*0.3f;
    }
    h_low = 2.5f;
    h_high = 4.0f;

    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));

    int threadPerBlock = 256;
    int blockPerGrid = (N + threadPerBlock - 1)/ threadPerBlock;

    value_clipping<<<blockPerGrid, threadPerBlock>>>(d_a, d_c, h_low, h_high, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(end, 0));
    CHECK_CUDA(cudaEventSynchronize(end));
    float time_elapsed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time_elapsed, start, end));

    cout<<"Elapsed time(in ms): "<<time_elapsed<<endl;    // 1.99

    for(int i=0;i<30 && i<N;i++){
        cout<<"Clipped value for i: "<<h_c[i]<<" for input: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}