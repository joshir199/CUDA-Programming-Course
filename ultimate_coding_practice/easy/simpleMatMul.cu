#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
using namespace std;


#define N 512
#define M 256
#define K 768
#define threadsDim 16  // keeping the dimension 16 x 16 as multiple of warps 32

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// CPU reference (simple triple loop)
void malMul_CPU(float* A, float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            double s = 0.0;
            for (int x = 0; x < N; ++x) {
                s += double(A[i*N + x]) * double(B[x*K + j]);
            }
            C[i*K + j] = (float)s;
        }
    }
}


__global__ void simpleMatMul(float* a, float* b, float* c) {

    int x_c = threadIdx.x + blockIdx.x * blockDim.x;  // column index
    int y_r = threadIdx.y + blockIdx.y * blockDim.y;  // row index

    if(x_c < K && y_r <M) {

        float tmp_sum = 0.0f;  // per thread variable to store sum details;

        // sum intermediate product as threadIndex x_c & y_r changes for each tmp_sum threads.
        for(int x = 0; x<N; x++) {
            tmp_sum += a[y_r * N + x] * b[x * K + x_c];
        }

        c[y_r * K + x_c] = tmp_sum; // copy the accumulated sum in output matrix of size [M x K]

    }
}




int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float h_a[M*N], h_b[N*K], h_c[M*K];
    float hcpu_a[M*N], hcpu_b[N*K], hcpu_c[M*K];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M*K*sizeof(float)));

    // fill data in host device
    for(int i=0; i<M*N ;i++) {
        h_a[i] = (rand() % 90) * 0.018f;
        hcpu_a[i] =(rand() % 90) * 0.018f;
    }
    for(int i = 0; i<N*K; i++){
        h_b[i] = (rand() % 19) * 0.018f;
        hcpu_b[i] = (rand() % 19) * 0.018f;
    }

    // CPU reference (for correctness)
    auto t0 = std::chrono::high_resolution_clock::now();
    malMul_CPU(hcpu_a, hcpu_b, hcpu_c);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();


    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*K*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    dim3 block(threadsDim, threadsDim);
    dim3 grid((N+threadsDim-1)/threadsDim, (M+threadsDim-1)/threadsDim);

    simpleMatMul<<<grid, block>>>(d_a, d_b, d_c);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*K*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"CPU Elapsed time(in ms) : "<< cpu_ms<<endl;  // 181
    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.5

    for(int i = 0; i< 50 && i<M*K; i++) {
        cout<<"Matrix Multiplication result at col + K*Row:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}