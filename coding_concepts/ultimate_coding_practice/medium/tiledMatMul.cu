#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define N 768
#define M 768 // With tiling, it can handle the size of 768 for each
#define K 768
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


__global__ void tiledMatMul(float* a, float* b, float* c) {

    int x_c = threadIdx.x + blockIdx.x * blockDim.x;  // column index
    int y_r = threadIdx.y + blockIdx.y * blockDim.y;  // row index

    // define two 2D sub tile of shape [Tile x Tile] for submatrix A & B
    __shared__ float cachetileA[Tile][Tile + 1]; // added extra padding for safe
    __shared__ float cachetileB[Tile][Tile + 1]; // added extra padding for safe

    // Now, for each tile we will calculate the partial product sum along common dimension
    int numTiles = (N + Tile - 1)/Tile;

    float partialSum = 0.0f;

    for(int i = 0; i< numTiles; i++) {

        int localx_c = threadIdx.x;
        int localy_r = threadIdx.y;

        int globalx_c = i*Tile + localx_c; // a[y_r, globalx_c] vary along horizontal per thread
        int globaly_r = i*Tile + localy_r; // b[globaly_r, x_c] vary along vertical per thread

        // load the data into shared memory per tileA for matrix B [M x N]
        if(localx_c < Tile && localy_r < Tile) {
            cachetileA[localx_c][localy_r] = a[y_r * N + globalx_c];
        } else {
            cachetileA[localx_c][localy_r] = 0.0f;
        }

        // load the data into shared memory per tileB for matrix B [N x K]
        if(localx_c < Tile && localy_r < Tile) {
            cachetileB[localx_c][localy_r] = b[globaly_r * K + x_c];
        } else {
            cachetileB[localx_c][localy_r] = 0.0f;
        }

        __syncthreads();

        // Now, do the matrix multiplication for corresponding tiles
        // unroll the matrix for element wise multiplication
        for(int j=0; j< Tile; j++) {
            partialSum += cachetileA[localy_r][j] * cachetileB[j][localx_c];
        }
        __syncthreads();
    }

    // copy the data per thread into the output matrix
    if(x_c < K && y_r < M) {
        c[y_r * K + x_c] = partialSum;
    }
}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

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


    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*K*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    dim3 block(threadsDim, threadsDim);
    dim3 grid((N+threadsDim-1)/threadsDim, (M+threadsDim-1)/threadsDim);

    tiledMatMul<<<grid, block>>>(d_a, d_b, d_c);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

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

    return 0;
}