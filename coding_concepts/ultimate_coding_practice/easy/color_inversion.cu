#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define Width 512
#define Height 256
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)


// Color inversion is performed by subtracting each color component (R, G, B) from 255.
// The Alpha component should remain unchanged.
__global__ void invert_kernel(int* image, int img_size) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while(tid < img_size) {
        if(tid%4 != 3) {
            // Change only RGB values not alpha values
            image[tid] = 255 - image[tid];
        }
        tid += blockDim.x * gridDim.x;
    }
}


int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    int img_size = Width * Height * 4; // Image which contains RGBA values for each pixel
    int h_image[img_size], h_image_out[img_size];
    int* d_image;

    CHECK_CUDA(cudaMalloc(&d_image, img_size*sizeof(int)));

    // fill data in host device
    for(int i=0; i<img_size ;i++) {
        h_image[i] = (i*i - 3*i + 46)%256;
    }

    CHECK_CUDA(cudaMemcpy(d_image, h_image, img_size*sizeof(int), cudaMemcpyHostToDevice));

    int blocksPerGrid = (img_size + threadsPerBlock - 1)/threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, img_size);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_image_out, d_image, img_size*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"Elapsed time(in ms) : "<< elapsed_time<<endl;

    for(int i = 0; i< 100 && i<img_size; i++) {
        cout<<"Image Inversion result at i:"<<i<<", is: "<<h_image_out[i]<<" and before:"<<h_image[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_image));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}