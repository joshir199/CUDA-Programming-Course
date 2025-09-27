#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define chunk 1024
#define threadPerBlock 256
#define numBuffers 3   // Triple buffering

#define CHECK_CUDA(call) do {           \
    cudaError_t e = (call);                 \
    if(e != cudaSuccess) {                      \
        cerr<<"CUDA Error : "<<cudaGetErrorString(e)<<  \
        " in "<<__FILE__<<" at "<<__LINE__<<endl;       \
        exit(1);                                    \
    }                                                   \
} while(0)


__global__ void kernelVectorShift(int* a, int* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<chunk) {
        c[tid] = a[tid] + 1;
    }
}

int main() {

    cudaDeviceProp prop;
    int deviceNumber;
    CHECK_CUDA(cudaGetDevice(&deviceNumber));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceNumber));
    if(!prop.deviceOverlap){
        cout<<"GPU does not support device overlap"<<endl;
        return 0;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int numBatches = 32;
    int blockPerGrid = (chunk + threadPerBlock - 1)/threadPerBlock;
    cout<<" Batch Size: "<<chunk<<", num. of batches: "<<numBatches<<endl;

    int **h_a = (int**)malloc(numBuffers*sizeof(int*));
    int **h_c = (int**)malloc(numBuffers*sizeof(int*));
    for(int i = 0; i<numBuffers ; i++) {
        CHECK_CUDA(cudaMallocHost(&h_a[i], chunk*sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&h_c[i], chunk*sizeof(int)));
    }

    int *d_a[numBuffers];
    int *d_c[numBuffers];
    for(int i = 0; i<numBuffers; i++) {
        CHECK_CUDA(cudaMalloc(&d_a[i], chunk*sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_c[i], chunk*sizeof(int)));
    }

    cudaEvent_t *h2dcopy = (cudaEvent_t*)malloc(numBuffers*sizeof(cudaEvent_t));
    cudaEvent_t *kernelcompute = (cudaEvent_t*)malloc(numBuffers*sizeof(cudaEvent_t));
    for(int i=0; i<numBuffers; i++) {
        CHECK_CUDA(cudaEventCreateWithFlags(&h2dcopy[i], cudaEventDisableTiming));
        CHECK_CUDA(cudaEventCreateWithFlags(&kernelcompute[i], cudaEventDisableTiming));
    }

    int h_data[numBatches * chunk];
    for(int i = 0; i<numBatches*chunk; i++) {
        h_data[i] = 2*i + rand() % (i+5) ;
    }

    cudaStream_t stream0, stream1, stream2;

    CHECK_CUDA(cudaStreamCreate(&stream0));
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    CHECK_CUDA(cudaEventRecord(start));
    //
    for(int i = 0; i<numBatches+2; i++) {

        if(i<numBatches) {
            int buf = i % numBuffers;
            for(int b = 0; b<chunk; b++) {
                h_a[buf][b] = h_data[b + chunk*i];
            }
            CHECK_CUDA(cudaMemcpyAsync(d_a[buf], h_a[buf], chunk*sizeof(int), cudaMemcpyHostToDevice, stream0));
            CHECK_CUDA(cudaEventRecord(h2dcopy[buf], stream0));
        }


        if(i>0 && i<numBatches-1) {
            int batchId = i-1;
            int buf = batchId % numBuffers;
            CHECK_CUDA(cudaStreamWaitEvent(stream1, h2dcopy[buf], 0));

            kernelVectorShift<<<blockPerGrid, threadPerBlock, 0, stream1>>>(d_a[buf], d_c[buf]);
            CHECK_CUDA(cudaGetLastError());

            CHECK_CUDA(cudaEventRecord(kernelcompute[buf], stream1));
        }

        if(i>1 && i< numBatches - 2) {
            int batchId = i-2;
            int buf = batchId % numBuffers;

            CHECK_CUDA(cudaStreamWaitEvent(stream2, kernelcompute[buf], 0));

            CHECK_CUDA(cudaMemcpyAsync(h_c[buf], d_c[buf], chunk*sizeof(int), cudaMemcpyDeviceToHost, stream2));

        }

    }

    CHECK_CUDA(cudaStreamSynchronize(stream0));
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<" Elpased time(in ms) : "<<elapsed_time<<endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    for(int i =0; i<numBatches ; i++) {
        cout<<"First element per batch i "<<i<<", is :"<<h_c[i%numBuffers]<<endl;
    }

    for(int i =0; i<numBuffers; i++) {
        CHECK_CUDA(cudaFree(d_a[i]));
        CHECK_CUDA(cudaFree(d_c[i]));
        CHECK_CUDA(cudaFreeHost(h_a[i]));
        CHECK_CUDA(cudaFreeHost(h_c[i]));
        CHECK_CUDA(cudaEventDestroy(h2dcopy[i]));
        CHECK_CUDA(cudaEventDestroy(kernelcompute[i]));
    }

    CHECK_CUDA(cudaStreamDestroy(stream0));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));

    free(h_a);
    free(h_c);
    free(h2dcopy);
    free(kernelcompute);


    return 0;
}