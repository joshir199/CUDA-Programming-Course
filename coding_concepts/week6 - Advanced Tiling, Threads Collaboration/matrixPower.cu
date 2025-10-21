#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define n 768
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


// Calculate power of matrix using formula:
//  For P as some positive integer:
//  C = A^P = A^(P-1) x A , A = Input Square matrix
__global__ void matMul(float* a, float* b, float* c, int N) {
    int x_c = threadIdx.x + blockDim.x * blockIdx.x;
    int y_r = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float cacheTileA[Tile][Tile+1];
    __shared__ float cacheTileB[Tile][Tile+1];

    float partialsum = 0.0f;
    int numtiles = (N + Tile -1)/Tile;

    for(int i = 0; i<numtiles; i++) {
        int localx_c = threadIdx.x;
        int localy_r = threadIdx.y;

        int globalx_c = i*Tile + localx_c; // For Matrix A[y_r, globalx_c]
        int globaly_r = i*Tile + localy_r; // For Matrix B[globaly_r, x_c]

        //Load data from matrix C to tileA
        if(y_r<N && globalx_c< N){
            cacheTileA[localy_r][localx_c] = a[y_r * N + globalx_c];
        } else {
            cacheTileA[localy_r][localx_c] = 0.0f;
        }

        //Load data from matrix A to tileB
        if(globaly_r<N && x_c< N){
            cacheTileB[localy_r][localx_c] = b[globaly_r * N + x_c];
        } else {
            cacheTileB[localy_r][localx_c] = 0.0f;
        }
        __syncthreads();

        for(int k=0; k< Tile; k++) {
            partialsum += cacheTileA[localy_r][k] * cacheTileB[k][localx_c];
        }
        __syncthreads();
    }

    if(x_c< N && y_r < N) {
        c[y_r * N + x_c] = partialsum;
    }
}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int N=n;
    int P;
    float h_a[N*N], h_c[N*N];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*N*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N;i++) {
        for(int j=0; j<N;j++) {
            h_a[i*N + j] = (rand() % 9) * 0.018f;
        }
    }
    P = 7;

    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_a, N*N*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    dim3 block(threadsDim, threadsDim);
    dim3 grid((N+threadsDim-1)/threadsDim, (N+threadsDim-1)/threadsDim);
    for(int i = 1; i< P; i++) {
        matMul<<<grid, block>>>(d_a, d_c, d_b, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        float* temp = d_c;
        d_c = d_b;
        d_b = temp;
    }

    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 6.16 for all 768, P = 7

    for(int i = 0; i< 50 && i<N*N; i++) {
        cout<<"Matrix Power result at col + N*Row:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}