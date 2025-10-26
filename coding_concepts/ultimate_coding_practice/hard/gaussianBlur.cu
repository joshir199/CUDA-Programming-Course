#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define n 768
#define m 768 // With tiling, it can handle the size of 768 for each

#define Tile 16  // same as Block size

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// General padded Gaussian Blur convolution : Matrix A[MxN]
// each block compute Tile x Tile output
__global__ void gaussianBlurKernel(float* a, float* b, float* c, int M, int N, int k_r, int k_c) {

    extern __shared__ float cache[];   // shared memory to stored all the required inputs for [Tile x Tile] output

    // in case of padding, use k_r/2, k_c/2
    int halo_r = k_r/2;
    int halo_c = k_c/2;
    int sharedWidth = Tile + k_c - 1;  // total Input elements in each block for Tile x Tile output
    int sharedHeight = Tile + k_r - 1;

    for(int i = threadIdx.y; i<sharedHeight; i+=blockDim.y) {
        for(int j = threadIdx.x; j<sharedWidth; j+=blockDim.x) {

            int gIdx_c = blockIdx.x * Tile + j - halo_c;
            int gIdy_r = blockIdx.y * Tile + i - halo_r;

            if(gIdx_c >= 0 && gIdx_c < N && gIdy_r >= 0 && gIdy_r < M) {
                cache[i * sharedWidth + j] = a[gIdy_r * N + gIdx_c];
            } else {
                cache[i * sharedWidth + j] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Per block only Tile x Tile output
    int outx_c = threadIdx.x + blockIdx.x * Tile;  // output column index
    int outy_r = threadIdx.y + blockIdx.y * Tile;  // output row index

    // get 2D Gaussian blur convolution using 2D kernel
    if(threadIdx.x < Tile && threadIdx.y < Tile && outx_c < N && outy_r < M) {
        float sum = 0.0f;

        for(int i = 0; i< k_r; i++) {
            for(int j = 0; j < k_c; j++) {

                sum += cache[(threadIdx.y + i) * sharedWidth + (threadIdx.x + j)] * b[i * k_c + j];
            }
        }

        c[outy_r * N + outx_c] = sum;

    }

}



int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int M = m;
    int N = n;
    int K_R = 21;  // maximum kernel size 17 x 17
    int K_C = 21;  // maximum kernel size 17 x 17


    float h_a[M*N], h_b[K_R*K_C], h_c[M*N];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, K_R*K_C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, M*N*sizeof(float)));

    // fill data in host device
    for(int i=0; i<M*N ;i++) {
        h_a[i] = (rand() % 90) * 0.018f;
        //cout<<"h_a: "<<h_a[i]<<endl;
    }
    for(int i = 0; i<K_R*K_C; i++){
        h_b[i] = (rand() % 19) * 0.018f;
        //cout<<"h_b: "<<h_b[i]<<endl;
    }


    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K_R*K_C*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    int threadsDimx = Tile + K_C - 1;  // Due to padding, we need more threads to load data
    int threadsDimy = Tile + K_R - 1;
    dim3 block(Tile, Tile);
    dim3 grid((N + Tile-1)/Tile, (M + Tile-1)/Tile);  // per block [Tile x Tile] output
    size_t shmem = (threadsDimx) * (threadsDimy) * sizeof(float);  // for output size of Tile x Tile
    gaussianBlurKernel<<<grid, block, shmem>>>(d_a, d_b, d_c, M, N, K_R, K_C);


    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 1.32 for all 768

    for(int i = 0; i< 30 && i<M*N; i++) {
        cout<<"Gaussian Blur result at col + K*Row:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}