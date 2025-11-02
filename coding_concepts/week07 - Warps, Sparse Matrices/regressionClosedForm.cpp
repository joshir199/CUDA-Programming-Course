#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define n 768
#define m 768 // With tiling, it can handle the size of 768 for each
#define Tile 16  // keep it same as block size


#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Get matrix transpose
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x<cols && y<rows) {
        output[y + x*rows] = input[x + y *cols];
    }
}

// Multiply A[M,N], B[N, K] to get C[M, K]
__global__ void matMul(float* a, float* b, float* c, int M, int N, int K) {

    int x_c = threadIdx.x + blockIdx.x * blockDim.x;
    int y_r = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ float cachetileA[Tile][Tile + 1];
    __shared__ float cachetileB[Tile][Tile + 1];

    float partialSum = 0.0f;

    int numTiles = (N + Tile -1)/Tile;

    for(int i = 0; i<numTiles; i++) {

        int localx = threadIdx.x;
        int localy = threadIdx.y;

        // get global idx for Matrix A & B
        int gIdx_c = i * Tile + localx; // A[y_r, gIdx_c]
        int gIdy_r = i * Tile + localy; // B[gIdy_r, x_c]

        // For Matrix A[M, N]
        if(y_r <M && gIdx_c<N) {
            cachetileA[localy][localx] = a[y_r * N + gIdx_c];
        } else {
            cachetileA[localy][localx] = 0.0f;
        }

        // For Matrix B[N, K]
        if(gIdy_r<N && x_c<K) {
            cachetileB[localy][localx] = b[gIdy_r * K + x_c];
        } else {
            cachetileB[localy][localx] = 0.0f;
        }
        __syncthreads();

        // get per tile multiplication
        for(int k = 0; k<Tile; k++) {
            partialSum += cachetileA[localy][k] * cachetileB[k][localx];
        }
        __syncthreads();
    }

    if(y_r < M && x_c < K) {
        c[y_r * K + x_c] = partialSum;
    }

}

// Get Matrix-Vector multiplication using per block dot product for each row
// A[M, N], B[N]  => C[M]
__global__ void matVecMul(float* a, float* b, float* c, int M, int N) {
    int localx = threadIdx.x;
    float threadSum = 0.0f;

    // each block stores single row of A[M,N]
    for(int i = threadIdx.x; i<N; i+= blockDim.x) {

        int gIdx = blockIdx.x * N + i;
        threadSum += a[gIdx] * b[i];  // element-wise dot product
    }

    // initialize the block Sum
    __shared__ float blockSum[256];
    if(threadIdx.x < blockDim.x) {
        blockSum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // copy the perthread sum
    blockSum[threadIdx.x] = threadSum;
    __syncthreads();

    // parallel reduction for sum
    for(int i=blockDim.x/2; i>0; i/=2) {

        if(threadIdx.x < i) {
            blockSum[threadIdx.x] += blockSum[threadIdx.x + i];
        }
        __syncthreads();
    }

    // get the per row dot product as final output C[M]
    if(threadIdx.x == 0) {
        c[blockIdx.x] = blockSum[0];
    }
}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float h_a[M*N], h_b[M], h_c[N];
    float *d_a, *d_at, *d_ata, *d_inv, *d_d, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, M*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_at, N*M*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ata, N*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_inv, N*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_d, N*sizeof(float)));

    // fill data in host device
    for(int i=0; i<M*N ;i++) {
        h_a[i] = (rand() % 90) * 0.018f;
    }
    for(int i = 0; i<M; i++){
        h_b[i] = (rand() % 19) * 0.018f;
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, M*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    //Step 1: matrix transpose = X_t
    dim3 block(Tile, Tile);
    dim3 grid((N+Tile-1)/Tile, (M+Tile-1)/Tile);
    matrix_transpose_kernel<<<grid, block>>>(d_a, d_at, M, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 2: matrix multiplication = X_t.X  => [N, N]
    dim3 grid2((N+Tile-1)/Tile, (N+Tile-1)/Tile);
    matMul<<<grid2, block>>>(d_at, d_a, d_ata, N, M, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Step 3: matrix inverse = (X_t.X)^-1  => [N, N]


    // Step 4: matrix vector multiplication = X_t.y
    // A[N, M], B[M]  => C[N]
    int threadsPerBlock = 256;
    int blocksPerGrid = N;
    matVecMul<<<blocksPerGrid, threadsPerBlock>>>(d_at, d_b, d_d, N, M);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    // Step 5: Final matrix vector multiplication = [ (X_t.X)^-1 ] [ X_t.y ]
    // A[N, N], B[N] => C[N]
    int blocksPerGrid1 = M;
    matVecMul<<<blocksPerGrid1, threadsPerBlock>>>(d_inv, d_d, d_c, N, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.72  (for 384 compared to simple) //1.76 for all 768

    for(int i = 0; i< 50 && i<N; i++) {
        cout<<"Linear Regression Closed Form trainable parameter result at i:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFree(d_at));
    CHECK_CUDA(cudaFree(d_ata));
    CHECK_CUDA(cudaFree(d_inv));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}