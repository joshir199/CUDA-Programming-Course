#include <iostream>
#include <cublas_v2.h>  // for cuBLAS library
#include <cuda_runtime.h>
using namespace std;

/*
// cuBLAS library Intro.
// It is a high-performance GPU library provided by NVIDIA that implements standard
// linear algebra operations, such as:
// Level 1 BLAS: Vector–Vector operations (e.g., dot products, scaling)
// Level 2 BLAS: Matrix–Vector operations (e.g., GEMV)
// Level 3 BLAS: Matrix–Matrix operations (e.g., GEMM)

// cuBLAS internally uses:
// - Register tiling, shared-memory blocking, Tensor Cores (on modern GPUs),
// - Streamed execution, and
// - Fused multiply–add (FMA) instructions.

*/

/* cuBLAS Usage:
// To use cuBLAS, you typically follow a fixed structure:
// (a) Initialize cuBLAS handle : Before any cuBLAS calls,
//     you must create a handle that represents the cuBLAS context
// (b) Allocate and transfer memory (as usual in CUDA -> host, device)
// (c) Perform operations via cuBLAS functions:
//  1. cublasSaxpy  :  Vector addition ( y = alpha*x + y)
//  2. cublasSdot  :  dot product ( x_T.y)
//  3. cublasSgemv  :  Matrix-Vector ( y = alpha * Ax + beta*Y)
//  4. cublasSgemm  :  Matrix-Matrix ( y = alpha * AxB + beta*Y)

// cuBLAS expects column-major, so transpose A and B for row-major input
// CUBLAS_OP_N - no transpose
// CUBLAS_OP_T - transpose
*/

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


int main() {

    cublasHandle_t handle; // define handle for cuBLAS context (similar to event)
    cublasCreate(&handle);

    float h_a[N], h_b[N], h_c[N];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(float)));

    for(int i = 0; i<N; i++) {
        h_a[i] = i*1.0f;
        h_b[i] = i*1.0f;
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    // void cublasSaxpy (int n, float alpha, const float *x, int incx, float *y, int incy)
    cublasSaxpy(N, alpha, d_a, 1, d_b, 1);

    CHECK_CUDA(cudaMemcpy(h_c, d_b, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cublasDestroy(handle));

    return 0;
}