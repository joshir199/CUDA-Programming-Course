#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <cuda_runtime.h>
using namespace std;


#define n 16     // Number of input tensor
#define ch 3    // number of channels in input tensor (<=16)
#define h 128     // height of image (<= 768)
#define w 128     // width of image  (<= 768)
#define p 2     // padding to be applied (<= 16)
#define s 2     // stride for the kernel (<=16)
#define k 5     // square max-pool kernel (<=16)

#define threadsDim 16  // keeping the dimension 16 x 16 as multiple of warps 32


#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)



// General Max-Pooling 2D with strides and padding
__global__ void maxPooling2d_kernel(float* input, float* output, int H, int W, int K, int pd, int st, int secId)
{
    int o_r = (H - K + 2*pd )/st + 1;   // output grid size
    int o_c = (W - K + 2*pd )/st + 1;

    // Per thread compute the max-pooling for per output index
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    int partOffset = H * W * secId;     // Offset for input values
    int partOffsetOutput = o_r * o_c * secId;     // Offset for output values

    if (out_x >= o_c || out_y >= o_r) {return;}

    float maxVal = -FLT_MAX;    // Per thread maxValue corresponds to per output index
    int gIdx = out_x*st - pd;   // starting global Idx in input including padding and stride
    int gIdy = out_y*st - pd;   // starting global Idy in input including padding and stride

    // loop over kernel size and accumulate the maxValue
    // repeated memory access per output
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            if(gIdx + j < 0 || gIdy + i < 0 || gIdx + j >= W || gIdy + i >= H) { continue;}

            int gId = (gIdy + i) * W + (out_x + j) + partOffset;
            maxVal = fmaxf(maxVal, input[gId]); // Single thread, no race condition
        }
    }
    // update to output array
    output[out_y * o_r + out_x + partOffsetOutput] = maxVal;
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int C = ch;    // channel
    int N = n;      // Number of inputs
    int H = h;      // Height of grid
    int W = w;      // Width of grid
    int K = k;      // maximum kernel size 16 x 16
    int Pd = p;    // padding
    int St = s;    // stride
    int o_r = (H - K + 2*Pd )/St + 1;
    int o_c = (W - K + 2*Pd )/St + 1;

    float h_a[N*C*H*W], h_c[N*C*o_r * o_c];
    float *d_a, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, N*C*H*W*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N*C*o_r * o_c*sizeof(float)));

    // fill data in host device
    for(int i=0; i<N*C*H*W ;i++) {
        h_a[i] = (rand() % 90) * 0.018f;
        //cout<<"h_a: "<<h_a[i]<<endl;
    }



    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*C*H*W*sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(start, 0));

    dim3 block(threadsDim, threadsDim);
    dim3 grid((o_c + threadsDim-1)/threadsDim, (o_r + threadsDim-1)/threadsDim);
    for(int i =0; i<N*C; i++) {
        maxPooling2d_kernel<<<grid, block>>>(d_a, d_c, H, W, K, Pd, St, i);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }


    CHECK_CUDA(cudaMemcpy(h_c, d_c, N*C*o_r * o_c*sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"GPU Elapsed time(in ms) : "<< elapsed_time<<endl;  // 0.72

    cout<<"Output matrix shape, row: "<<o_r<<", and col: "<<o_c<<endl;  // 64 x 64

    for(int i = 0; i< 50 && i<N*C*o_r * o_c; i++) {
        cout<<"Max Pooling 2D result at col + K*Row:"<<i<<", is: "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}