#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define n 499999

#define CHECK_CUDA(call) do {                       \
    cudaError_t e = (call);                        \
    if(e != cudaSuccess) {                         \
        cout<<"CUDA Error:"<<cudaGetErrorString(e)  \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                                \
} while(0)


__global__ void interleave_array(float* a, float* b, float* c, int N){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid<N){
        c[2*tid] = a[tid];
        c[2*tid + 1] = b[tid];
    }
}


int main() {

    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    int N=n;
    float h_a[N], h_b[N], h_c[2*N];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, 2*N*sizeof(float)));

    for(int i=0;i<N;i++){
        h_a[i] = (rand()%15 + 4.0f)*0.3f;
        h_b[i] = rand()%10;
    }


    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice));

    int threadPerBlock = 256;
    int blockPerGrid = (N + threadPerBlock - 1)/ threadPerBlock;

    interleave_array<<<blockPerGrid, threadPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, 2*N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(end, 0));
    CHECK_CUDA(cudaEventSynchronize(end));
    float time_elapsed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time_elapsed, start, end));

    cout<<"Elapsed time(in ms): "<<time_elapsed<<endl;    // 1.67

    for(int i=0;i<30 && i<N;i++){
        cout<<"Interleaved value of A & B for i: "<<h_c[i]<<" for input: "<<h_a[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}