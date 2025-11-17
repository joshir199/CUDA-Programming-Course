#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define n 128
#define m 128
#define depth 64

#define threadsDim 8  // keeping the dimension 8 x 8 x 8 as multiple of warps 32

__constant__ float filter[125];

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Basic Non-padded 3D convolution
__global__ void conv3d_kernel(float* input, float* output, int input_deps, int input_rows, int input_cols, int k_deps, int k_rows, int k_cols)
{
    // valid output size
    int out_rows = input_rows - k_rows + 1;
    int out_cols = input_cols - k_cols + 1;
    int out_deps = input_deps - k_deps + 1;

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_z >= out_deps || out_x >= out_cols || out_y >= out_rows){ return;} // check for size limit

    // per thread handles one output index and its value
    float convVal = 0.0f;

    // loop over kernel size and accumulate the values
    // repeated memory access per output
    for (int d = 0; d < k_deps; d++) {
        for (int i = 0; i < k_rows; i++) {
            for (int j = 0; j < k_cols; j++) {
                // global Index for input array with stride 1 and No padding
                int gIdx = (out_z + d) * input_cols * input_rows + (out_y + i) * input_cols + (out_x + j);
                // Index for kernel array
                int kIdx = d * k_rows * k_cols + i * k_cols + j;
                convVal += input[gIdx] * filter[kIdx];
            }
        }
    }
    output[out_z * out_rows * out_cols   +  out_y * out_cols  +  out_x] = convVal;
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int D = depth;
    int M = m;
    int N = n;
    int K_D = 5;
    int K_R = 5;  // maximum kernel size 5 x 5
    int K_C = 5;  // maximum kernel size 5 x 5
    int o_d = D - K_D + 1;
    int o_r = M - K_R + 1;
    int o_c = N - K_C + 1;

    float h_a[D*M*N], h_b[K_D*K_R*K_C], h_c[o_d*o_r * o_c];
    float *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, D*M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, o_d*o_r * o_c*sizeof(float)));

    // fill data in host device
    for(int i=0; i<D*M*N ;i++) {
        h_a[i] = (rand() % 90) * 0.018f;
        //cout<<"h_a: "<<h_a[i]<<endl;
    }

    //filter
    for(int i = 0; i<125; i++){
        if(i<K_C*K_D*K_R) {
            h_b[i] = (rand() % 19) * 0.018f;
        } else {
            h_b[i] = 0.0f;
        }
        //cout<<"h_b: "<<h_b[i]<<endl;
    }


    CHECK_CUDA(cudaMemcpy(d_a, h_a, D*M*N*sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(filter, h_b, 125*sizeof(float)));

    CHECK_CUDA(cudaEventRecord(start, 0));

    // For basic 3D convolution (without shared memory)
    // Per thread compute convolution for one output index
    dim3 block(threadsDim, threadsDim, threadsDim);
    dim3 grid((o_c + threadsDim-1)/threadsDim, (o_r + threadsDim-1)/threadsDim, (o_d + threadsDim-1)/threadsDim);
    conv3d_kernel<<<grid, block>>>(d_a, d_c, D, M, N, K_D, K_R, K_C);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, o_d * o_r * o_c*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 1.52 for all 64x128x128

    for(int i = 0; i< 50 && i<o_d * o_r * o_c; i++) {
        cout<<"3D Convolution result at indexi:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}