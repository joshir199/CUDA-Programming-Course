#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_fp16.h>  // to handle Floating Point 16 (FP16) of half precision
#include <cuda_runtime.h>
using namespace std;


#define n 768
#define m 768 // With tiling, it can handle the size of 768 for each
#define k 768
#define Tile 16  // keep it same as block size

#define threadsDim 16  // keeping the dimension 16 x 16 as multiple of warps 32

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)



__global__ void tiledMatMul(const __half* a, const __half* b, float* c, int M, int N, int K) {

    int x_c = threadIdx.x + blockIdx.x * blockDim.x;  // column index
    int y_r = threadIdx.y + blockIdx.y * blockDim.y;  // row index

    // define two 2D sub tile of shape [Tile x Tile] for submatrix A & B
    __shared__ float cachetileA[Tile][Tile + 1]; // added extra padding for safe
    __shared__ float cachetileB[Tile][Tile + 1]; // added extra padding for safe

    // Now, for each tile we will calculate the partial product sum along common dimension
    int numTiles = (N + Tile - 1)/Tile;

    float partialSum = 0.0f; // matrix multiplication in FP32 precision for accuracy

    for(int i = 0; i< numTiles; i++) {

        int localx_c = threadIdx.x;
        int localy_r = threadIdx.y;

        // For matrix A
        int globalx_c = i*Tile + localx_c; // a[y_r, globalx_c] vary along horizontal per thread
        // For matrix B
        int globaly_r = i*Tile + localy_r; // b[globaly_r, x_c] vary along vertical per thread

        // load the data into shared memory per tileA for matrix A [M x N]
        // First, convert the FP16 to FP32 to avoid any loss in precision while doing Sum/Mul
        if(globalx_c < N && y_r < M) {
            cachetileA[localy_r][localx_c] = __half2float(a[y_r * N + globalx_c]);
        } else {
            cachetileA[localy_r][localx_c] = 0.0f;
        }

        // load the data into shared memory per tileB for matrix B [N x K]
        // First, convert the FP16 to FP32 to avoid any loss in precision while doing Sum/Mul
        if(x_c < K && globaly_r < N) {
            cachetileB[localy_r][localx_c] = __half2float(b[globaly_r * K + x_c]);
        } else {
            cachetileB[localy_r][localx_c] = 0.0f;
        }

        __syncthreads();

        // Now, do the matrix multiplication for corresponding tiles
        // unroll the matrix for element wise multiplication
        for(int j=0; j< Tile; j++) {
            partialSum += cachetileA[localy_r][j] * cachetileB[j][localx_c];
        }
        __syncthreads();
    }

    // copy the data per thread into the output matrix C [MxK]
    if(x_c < K && y_r < M) {
        c[y_r * K + x_c] = partialSum;
    }
}

// General Mixed Precision Matrix Multiplication : C = alpha*(AxB) + beta*C
// AxB is calculated in FP32 and later added with C as FP32 and finally converted back to FP16.
//  __half2float : FP16 to FP32  (half to full precision)
//  __float2half : FP32 to FP16  (full to half precision)
__global__ void multiplierKernel(float* a, __half* c, int M, int N, float alpha, float beta) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x ;
    if(tid<M*N) {
        // do operation in FP32 and quantize back to FP16 as output
        c[tid] = __float2half(beta * __half2float(c[tid])  +  alpha * a[tid]);
    }

}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int M = m;
    int N = n;
    int K = k;
    float alpha = 0.8f;
    float beta = 0.3f;

    __half h_a[M*N], h_b[N*K], h_c[M*K]; // define FP16 data
    __half *d_a, *d_b, *d_c;
    float *d_ab;

    CHECK_CUDA(cudaMalloc(&d_a, M*N*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_b, N*K*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_c, M*K*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_ab, M*K*sizeof(float)));

    // fill data in host device
    for(int i=0; i<M*N ;i++) {
        h_a[i] = __float2half((rand() % 90) * 0.018f);
    }
    for(int i = 0; i<N*K; i++){
        h_b[i] = __float2half((rand() % 19) * 0.018f);
    }
    for(int i=0; i<M*K ;i++) {
        h_c[i] = __float2half((rand() % 90) * 0.018f);
    }


    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*N*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*K*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, M*K*sizeof(__half), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    dim3 block(threadsDim, threadsDim);
    dim3 grid((K+threadsDim-1)/threadsDim, (M+threadsDim-1)/threadsDim);

    tiledMatMul<<<grid, block>>>(d_a, d_b, d_ab, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int threadsPerBlock = threadsDim * threadsDim;
    int blocksPerGrid = (M*K + threadsPerBlock - 1)/threadsPerBlock;
    multiplierKernel<<<blocksPerGrid, threadsPerBlock>>>(d_ab, d_c, M, K, alpha, beta);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*K*sizeof(__half), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 1.09

    // Cannot print FP16 directly, so convert it into FP32 first
    for(int i = 0; i< 50 && i<M*K; i++) {
        cout<<"Mixed Precision Matrix Multiplication at col + K*Row:"<<i<<", is: "<<__half2float(h_c[i])<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_ab));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}