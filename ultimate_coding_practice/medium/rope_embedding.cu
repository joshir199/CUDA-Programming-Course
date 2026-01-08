#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define threadsPerBlock 256
#define n 999
#define d 512  // Always even

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Rotary Positional Embedding (RoPE) for a batch of query vectors
// RoPE (X) = cos (.) X  +  rotate_half_negative_first_half(X) (.) sin
// (.) = element-wise multiplication,
// rotate_half_negative_first_half(X) = swap first half with second half and negate the first half
// The feature dimension is always given even number.
__global__ void rope_kernel(float* a, float* sin, float* cos, float* c, int N, int D) {

    int offset = blockIdx.x * D;    // per block handles one data point (one row)
    // Since, dimension D can go as high as 10K, using Shared memory will need 40KB which is almost limit (48KB)
    // Also, RoPE computation is single access computation, it does not require shared memory.
    int half_dim = D >> 1;

    for(int i = threadIdx.x; i<D; i+=blockDim.x) {
        int gIdx = i + offset;
        if(i < half_dim) {
            c[gIdx] = a[gIdx] * cos[gIdx] - sin[gIdx] * a[i + half_dim + offset];
        } else {
            c[gIdx] = a[gIdx] * cos[gIdx] + sin[gIdx] * a[i - half_dim + offset];
        }
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N = n;  // number of samples
    int D = d;  // feature dimension (always Even)

    float h_a[N*D], h_sin[N*D], h_cos[N*D], h_c[N*D];
    float *d_a, *d_sin, *d_cos, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sin, N*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cos, N*D*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*D*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N*D ;i++) {
        h_a[i] = (i%109 + 1) * 0.01f;
        h_sin[i] = (rand()%100)*0.01f;
        h_cos[i] = 1.0f - h_sin[i];
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*D*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sin, h_sin, N*D*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cos, h_cos, N*D*sizeof(float), cudaMemcpyHostToDevice));

    int blocksPerGrid = N;
    rope_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_sin, d_cos, d_c, N, D);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*D*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;    // 1.23

    for(int i = 0; i< 30 && i<N*D; i++) {
        cout<<"RoPE computation result at i:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_sin));
    CHECK_CUDA(cudaFree(d_cos));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}