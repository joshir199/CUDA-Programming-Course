#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define n 256
#define m 64
#define k 128
#define bch 32   // batch size
#define Tile 16  // keep it similar to warp-size

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)

// Matrix Multiplication using tiled sub-matrix for each sub matrix in a batch.
// Matrix multiplication of A[MxN], B[NxK] to get C[MxK]
__global__ void tiledMatMul(float* a, float* b, float* c, int batch, int M, int N, int K) {

    int x_c = threadIdx.x + blockIdx.x * blockDim.x;  // column index
    int y_r = threadIdx.y + blockIdx.y * blockDim.y;  // row index

    // index offset for each sub-matrix based on their size
    int batch_Offset_A = batch * M * N;   // For A
    int batch_Offset_B = batch * N * K;   // For B
    int batch_Offset_C = batch * M * K;   // For C

    // define two 2D sub tile of shape [Tile x Tile] for submatrix A & B
    __shared__ float cachetileA[Tile][Tile + 1]; // added extra padding for safe
    __shared__ float cachetileB[Tile][Tile + 1]; // added extra padding for safe

    // Now, for each tile we will calculate the partial product sum along common dimension
    int numTiles = (N + Tile - 1)/Tile;

    float partialSum = 0.0f;

    for(int i = 0; i< numTiles; i++) {

        int localx_c = threadIdx.x;
        int localy_r = threadIdx.y;

        // For matrix A
        int globalx_c = i*Tile + localx_c; // a[y_r, globalx_c] vary along horizontal per thread
        // For matrix B
        int globaly_r = i*Tile + localy_r; // b[globaly_r, x_c] vary along vertical per thread

        // load the data into shared memory per tileA for matrix A [M x N]
        if(globalx_c < N && y_r < M) {
            cachetileA[localy_r][localx_c] = a[batch_Offset_A + y_r * N + globalx_c]; // adding batch offset
        } else {
            cachetileA[localy_r][localx_c] = 0.0f;
        }

        // load the data into shared memory per tileB for matrix B [N x K]
        if(x_c < K && globaly_r < N) {
            cachetileB[localy_r][localx_c] = b[batch_Offset_B + globaly_r * K + x_c]; // adding batch offset
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

    // copy the data per thread into the output sub-matrix C [M x K]
    if(x_c < K && y_r < M) {
        c[batch_Offset_C + y_r * K + x_c] = partialSum; // adding batch offset
    }
}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int M=m;
    int N=n;
    int K=k;
    int B=bch;

    /*
    Input:
    B = 2, M = 2, K = 3, N = 2
    A = [
         [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
         [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        ]
    B = [
         [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
         [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]
        ]
    Output:
    Per sub-matrix multiplication : C_b = A_b x B_b
    C = [
         [[22.0, 28.0], [49.0, 64.0]],
         [[92.0, 68.0], [128.0, 95.0]]
        ]
    */

    float h_a[B*M*N], h_b[B*N*K], h_c[B*M*K];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, B*M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, B*N*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, B*M*K*sizeof(float)));

    // fill data in host device
    for(int i=0; i<B*M*N ;i++) {
        h_a[i] = (rand() % 90) * 0.018f;
        //cout<<"h_a: "<<h_a[i]<<endl;
    }
    for(int i = 0; i<B*N*K; i++){
        h_b[i] = (rand() % 19) * 0.018f;
        //cout<<"h_b: "<<h_b[i]<<endl;
    }

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, B*M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, B*N*K*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(Tile, Tile);
    dim3 grid((K+Tile-1)/Tile, (M+Tile-1)/Tile);

    for(int i=0; i<B; i++) {
        tiledMatMul<<<grid, block>>>(d_a, d_b, d_c, i, M, N, K);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaMemcpy(h_c, d_c, B*M*K*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 1.56

    for(int i = 0; i< 30 && i<B*M*K; i++) {
        cout<<"Matrix Multiplication result at col + K*Row:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}