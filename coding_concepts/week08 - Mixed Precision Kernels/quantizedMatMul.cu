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

// Matrix A [M x N] , B[N x K] to get C[M x K]
__global__ void quantizedtiledMatMul(const int8_t* a, const int8_t* b, int8_t* c, int M, int N, int K, int zA, int zB, int zC, float sA, float sB, float sC) {

    int x_c = threadIdx.x + blockIdx.x * blockDim.x;  // column index
    int y_r = threadIdx.y + blockIdx.y * blockDim.y;  // row index

    // define two 2D sub tile of shape [Tile x Tile] for sub-matrix A & B
    // Convert to int32_t or float only when multiplying.
    __shared__ int8_t cachetileA[Tile][Tile + 1]; // added extra padding for safe
    __shared__ int8_t cachetileB[Tile][Tile + 1]; // added extra padding for safe

    // Now, for each tile we will calculate the partial product sum along common dimension
    int numTiles = (N + Tile - 1)/Tile;

    int32_t partialSum = 0; // matrix multiplication in INT32 precision for accuracy

    for(int i = 0; i< numTiles; i++) {

        int localx_c = threadIdx.x;
        int localy_r = threadIdx.y;

        // For matrix A
        int globalx_c = i*Tile + localx_c; // a[y_r, globalx_c] vary along horizontal per thread
        // For matrix B
        int globaly_r = i*Tile + localy_r; // b[globaly_r, x_c] vary along vertical per thread

        // load the data into shared memory per tileA for matrix A [M x N]
        if(globalx_c < N && y_r < M) {
            cachetileA[localy_r][localx_c] = a[y_r * N + globalx_c];
        } else {
            cachetileA[localy_r][localx_c] = 0;
        }

        // load the data into shared memory per tileB for matrix B [N x K]
        if(x_c < K && globaly_r < N) {
            cachetileB[localy_r][localx_c] = b[globaly_r * K + x_c];
        } else {
            cachetileB[localy_r][localx_c] = 0;
        }

        __syncthreads();

        // Now, do the matrix multiplication for corresponding tiles
        // unroll the matrix for element wise multiplication
        for(int j=0; j< Tile; j++) {
            // Accumulate the raw int32 product
            partialSum += (int32_t(cachetileA[localy_r][j] - zA)  * int32_t(cachetileB[j][localx_c] - zB) );
        }
        __syncthreads();
    }

    // copy the data per thread into the output matrix C [MxK]

    if(x_c < K && y_r < M) {
        // This avoids both precision loss and redundant floating-point multiplications inside the loop.
        // scaling in float32
        float scaled_sum = partialSum * sA * sB / sC;
        // clamp(x, a, b) :  clamps the value x into the interval [a, b]
        // finally converting back to INT8 precision
        int quant = max(-128, min(__float2int_rn(scaled_sum) + zC, 127));
        c[y_r * K + x_c] = static_cast<int8_t>(quant);
    }
}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int M = m;
    int N = n;
    int K = k;
    int zA = 0;
    int zB = 0;
    int zC = 0;

    float sA = 0.8f;
    float sB = 0.3f;
    float sC = 0.3f;

    int8_t h_a[M*N], h_b[N*K], h_c[M*K]; // define INT8 data
    int8_t *d_a, *d_b, *d_c;


    CHECK_CUDA(cudaMalloc(&d_a, M*N*sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_b, N*K*sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_c, M*K*sizeof(int8_t)));

    // fill data in host device
    for(int i=0; i<M*N ;i++) {
        h_a[i] = 1.0; //((rand() % 90) * 0.018f);
    }
    for(int i = 0; i<N*K; i++){
        h_b[i] = 1.0; //((rand() % 19) * 0.018f);
    }


    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*N*sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*K*sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c, M*K*sizeof(int8_t), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    dim3 block(threadsDim, threadsDim);
    dim3 grid((K+threadsDim-1)/threadsDim, (M+threadsDim-1)/threadsDim);

    quantizedtiledMatMul<<<grid, block>>>(d_a, d_b, d_c, M, N, K, zA, zB, zC, sA, sB, sC);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*K*sizeof(__half), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 1.09

    // Cannot print FP16 directly, so convert it into FP32 first
    for(int i = 0; i< 30 && i<M*K; i++) {
        cout<<"INT8 Quantized Matrix Multiplication at col + K*Row:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}