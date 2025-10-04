#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
using namespace std;

#define M 3
#define N 3

#define threadsDim 16  // In 2D grid, 16 x 16

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
        for (int j = 0; j < M; ++j) {
            double s = 0.0;
            for (int k = 0; k < N; ++k)
                s += double(A[i*N + k]) * double(B[k*N + j]);
            C[i*N + j] = (float)s;
        }
    }
}

__global__ void matMul_GPU(float* a, float* b, float* c) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x<M && y<N) {
        //int idx_a = x + y*N;
        //int idx_b = y + x*M;
        float sum = 0.0f;
        for(int k=0;k<N;k++) {
            sum += a[k + y*N]*b[y + k*M];
        }
        c[x + y*N] = sum;
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float h_a[M*N], h_b[N*M], h_c[M*M];
    float hcpu_a[M*N], hcpu_b[N*M], hcpu_c[M*M];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*M*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_a, N*M*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M*M*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N*M ;i++) {
        h_a[i] = 1.0f; //(rand() % 90) * 0.018f;
        hcpu_a[i] = 1.0f; //(rand() % 90) * 0.018f;
        h_b[i] = 2.0f; //(rand() % 90) * 0.018f;
        hcpu_b[i] = 2.0f; //(rand() % 90) * 0.018f;
    }

    // CPU reference (for correctness)
    auto t0 = std::chrono::high_resolution_clock::now();
    malMul_CPU(hcpu_a, hcpu_b, hcpu_c);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();


    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*M*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*M*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    dim3 block(threadsDim, threadsDim);
    dim3 grid((N+threadsDim-1)/threadsDim, (M+threadsDim-1)/threadsDim);

    matMul_GPU<<<grid, block>>>(d_a, d_b, d_c);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*M*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"CPU Elapsed time(in ms) : "<< cpu_ms<<endl;
    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"Matrix Multiplication result at col + N*Row:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}