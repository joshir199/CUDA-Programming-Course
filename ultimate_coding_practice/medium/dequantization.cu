#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <cuda_runtime.h>
using namespace std;


#define m 768     // rows of input tensor
#define n 768     // cols
#define TILE_SIZE 64     // Tile size (<=128)


#define threadsDim 16  // keeping the dimension 16 x 16 as multiple of warps 32


#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)



// De-quantisation of weights using scaling matrix
__global__ void dequantization(float* a, float* b, float* c, int M, int N, int S_cols, int S_rows)
{
    int x_c = threadIdx.x + blockDim.x * blockIdx.x;
    int y_r = threadIdx.y + blockDim.y * blockIdx.y;

    extern __shared__ float scaleCache[];

    if(x_c >= N || y_r >= M) { return; }

    for(int i = threadIdx.y; i<S_rows; i+=blockDim.y) {
        for(int j = threadIdx.x; j<S_cols; j+=blockDim.x) {
            scaleCache[i * S_cols + j] = b[i * S_cols + j];
        }
    }
    __syncthreads();

    int scale_x = x_c / TILE_SIZE;
    int scale_y = y_r / TILE_SIZE;

    c[y_r * N + x_c] = a[y_r * N + x_c] * scaleCache[scale_y * S_cols + scale_x];
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int M = m;      // rows of inputs
    int N = n;      // cols of inputs
    int Tile = TILE_SIZE;      // Tile size for scaling

    int s_rows = (M + Tile - 1) / Tile;
    int s_cols = (N + Tile - 1) / Tile;


    float h_a[N*M], h_s[s_rows*s_cols], h_c[N*M];
    float *d_a, *d_s, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*M*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*M*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_s, s_rows*s_cols*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N*M ;i++) {
        h_a[i] = (rand() % 89) * 0.018f;
        //cout<<"h_a: "<<h_a[i]<<endl;
    }

    for(int i=0; i<s_cols * s_rows ;i++) {
        h_s[i] = (rand() % 5) * 1.0f;
        //cout<<"h_s: "<<h_s[i]<<endl;
    }


    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*M*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_s, h_s, s_rows*s_cols*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    dim3 block(threadsDim, threadsDim);
    dim3 grid((N + threadsDim-1)/threadsDim, (M + threadsDim-1)/threadsDim);
    size_t shmem = s_rows * s_cols * sizeof(float);

    dequantization<<<grid, block, shmem>>>(d_a, d_s, d_c, M, N, s_cols, s_rows);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());



    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*M*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.82

    for(int i = 0; i< 50 && i<N*M; i++) {
        cout<<"Weight de-quantization result at col + K*Row:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_s));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}