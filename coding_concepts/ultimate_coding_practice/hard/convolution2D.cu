#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define n 768
#define m 768 // With tiling, it can handle the size of 768 for each

#define threadsDim 16  // keeping the dimension 16 x 16 as multiple of warps 32

#define Tile 16  // same as Block size

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// General padded convolution : Matrix A[MxN]
__global__ void convolution2DKernel(float* a, float* b, float* c, int M, int N, int k_r, int k_c) {
    int x_c = threadIdx.x + blockIdx.x * blockDim.x;  // column index
    int y_r = threadIdx.y + blockIdx.y * blockDim.y;  // row index
    // valid output size
    int output_c = N - k_c + 1;
    int output_r = M - k_r + 1;

    // in case of padding, use k_r/2, k_c/2
    int halo_r = 0;
    int halo_c = 0;

    // starting input index for given output index
    int i_c = x_c - halo_c;
    int i_r = y_r - halo_r;

    extern __shared__ float cache[];  // shared memory of size k_r x k_c (same as kernel)

    int localx_c = threadIdx.x;
    int localy_r = threadIdx.y;
    int shared_dim = Tile + 2 * halo_r;

    // Load data into shared memory tile
    if(i_c >= 0 && i_c<N && i_r>=0 && i_r<M) {
        cache[localy_r * shared_dim + localx_c] = a[i_r * N + i_c];
    } else {
        cache[localy_r * shared_dim + localx_c] = 0.0f;
    }
    __syncthreads();

    float partialSum = 0.0f;

    // get 2D convolution using kernel
    if(x_c < output_c && y_r < output_r && (localx_c < Tile && localy_r < Tile)) {
        for(int i=0; i<k_r; i++) {
            for(int j = 0; j<k_c; j++) {
                partialSum += b[i * k_c + j] * cache[(localy_r + i) * shared_dim + (localx_c + j)];
            }
        }
        c[y_r * output_c + x_c] = partialSum;
    }

}


// Basic Non-padded 2D convolution
__global__ void conv2d_kernel(float* input, float* kernel, float* output, int input_rows, int input_cols, int k_rows, int k_cols)
{
    // valid output size
    int out_rows = input_rows - k_rows + 1;
    int out_cols = input_cols - k_cols + 1;

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= out_cols || out_y >= out_rows)  // check for index limit
        return;

    float val = 0.0f;
    // loop over kernel size and accumulate the values
    // repeated memory access per output
    for (int i = 0; i < k_rows; i++) {
        for (int j = 0; j < k_cols; j++) {
            float a = input[(out_y + i) * input_cols + (out_x + j)];
            float b = kernel[i * k_cols + j];
            val += a * b;
        }
    }
    output[out_y * out_cols + out_x] = val;
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int M = m;
    int N = n;
    int K_R = 28;  // maximum kernel size 31 x 31
    int K_C = 30;  // maximum kernel size 31 x 31
    int o_r = M - K_R + 1;
    int o_c = N - K_C + 1;

    float h_a[M*N], h_b[K_R*K_C], h_c[o_r * o_c];
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, K_R*K_C*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, o_r * o_c*sizeof(float)));

    // fill data in host device
    for(int i=0; i<M*N ;i++) {
        h_a[i] = i*1.0f;//(rand() % 90) * 0.018f;
        //cout<<"h_a: "<<h_a[i]<<endl;
    }
    for(int i = 0; i<K_R*K_C; i++){
        h_b[i] = ((i + 2)%2)*1.0f; //(rand() % 19) * 0.018f;
        //cout<<"h_b: "<<h_b[i]<<endl;
    }


    CHECK_CUDA(cudaMemcpy(d_a, h_a, M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, K_R*K_C*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    dim3 block(threadsDim, threadsDim);
    dim3 grid((N + threadsDim-1)/threadsDim, (M + threadsDim-1)/threadsDim);
    size_t shmem = (Tile + K_R - 1) * (Tile + K_C - 1) * sizeof(float);  // for output size of Tile x Tile
    convolution2DKernel<<<grid, block, shmem>>>(d_a, d_b, d_c, M, N, K_R, K_C);

    // For basic 2D convolution (without shared memory)
    // dim3 block(threadsDim, threadsDim);
    // dim3 grid((o_c + threadsDim-1)/threadsDim, (o_r + threadsDim-1)/threadsDim);
    // conv2d_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K_R, K_C);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, o_r * o_c*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 1.66 for all 768

    for(int i = 0; i< 50 && i<o_r * o_c; i++) {
        cout<<"2D Convolution result at col + K*Row:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}