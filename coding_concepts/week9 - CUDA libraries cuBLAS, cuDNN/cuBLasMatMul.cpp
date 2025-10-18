#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cublas_v2.h>  // for cuBLAS library
#include <cuda_runtime.h>
using namespace std;


#define N 768
#define M 512 // With tiling, it can handle the size of 768 for each
#define K 512

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// nvcc -arch=sm_86 -lcublas cuBLasMatMul.cu -o cuBLasMatMul

int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // for cuBLAS event handling
    cublasHandle_t handle;
    cublasCreate(&handle);

    float h_a[M*N], h_b[N*K], h_c[M*K];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M*K*sizeof(float)));

    // fill data in host device
    for(int i=0; i<M*N ;i++) {
        h_a[i] = (rand() % 90) * 0.018f;
    }
    for(int i = 0; i<N*K; i++){
        h_b[i] = (rand() % 19) * 0.018f;
    }
    float alpha = 1.0f, beta = 0.0f;


    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*K*sizeof(float), cudaMemcpyHostToDevice));
    //CHECK_CUDA(cudaMemset(d_c, 0, M*K*sizeof(float)));

    CHECK_CUDA(cudaEventRecord(start, 0));
    // A[MxN], B[NxK], C[MxK]
    // cuBLAS expects column-major, so transpose A and B for row-major input
    // Use CUBLAS_OP_N for no transpose, CUBLAS_OP_T for transpose
    // Incase of CUBLAS_OP_T -> change the order of A and B with transpose => B_T * A_T, C_T = (AB)_T)_T
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, K, M, N, &alpha, d_b, N, d_a, N, &beta, d_c, K); // careful for d_a -> N

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*K*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.72  (for 384 compared to simple) //1.76 for all 768

    for(int i = 0; i< 50 && i<M*K; i++) {
        cout<<"Matrix Multiplication result at col + K*Row:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    cublasDestroy(handle);

    return 0;
}