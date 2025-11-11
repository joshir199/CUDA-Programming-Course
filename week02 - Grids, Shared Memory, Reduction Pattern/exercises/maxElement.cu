#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;

#define N 1101
#define threadsPerBlock 256
#define max(a,b) (a>b)?a:b

#define CHECK_CUDA(call) do {                         \
    cudaError_t e = (call);                           \
    if(e != cudaSuccess) {                            \
        cout<<"Cuda Error: "<<cudaGetErrorString(e)   \
        <<", in:"<<__FILE__<<",at "<<__LINE__<<endl;  \
        exit(1);                                      \
    }                                              \
} while(0)



__global__ void kernelMaxElement(int* a, int* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // it will has memory for thread in one block
    __shared__ int cache[threadsPerBlock];
    int cacheIdx = threadIdx.x; // shared memory Id per block

    if(tid<N){
        cache[cacheIdx] = a[tid];
    } else {
        cache[cacheIdx] = 0; // if numbers are positive, otherwise use -INF.
    }
    __syncthreads();  // make sure all elements are copied

    // Using parallel reduction technique, calculate the max element per block
    int i = blockDim.x/2;
    while(i>0){
        if(cacheIdx < i) {
            cache[cacheIdx] = max(cache[cacheIdx], cache[cacheIdx + i]);
        }
        __syncthreads();
        i = i/2;
    }

    // after parallel reduction each block is done,
    // maximum for each block will be at its 0th index
    // So, copy those into the output array with blockIdx as index
    if(cacheIdx == 0) {
        c[blockIdx.x] = cache[cacheIdx];
    }

}



int main() {

    srand((unsigned)time(NULL));

    // Define event variables to record program time
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));


    // define the variables for host and device
    int h_a[N], h_c[N];
    int *d_a, *d_c;

    int blockPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;

    // allocate memory for device variables
    CHECK_CUDA(cudaMalloc(&d_a, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, blockPerGrid*sizeof(int)));

    // initialize the array
    for(int i=0;i<N;i++){
        h_a[i] = rand() % 9874654;
    }

    // send data to device variable
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice));

    // call the kernel for computation
    kernelMaxElement<<<blockPerGrid, threadsPerBlock>>>(d_a, d_c);

    // check for any error
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // get the result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, blockPerGrid*sizeof(int), cudaMemcpyDeviceToHost));

    // check and stop the recording
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    // print the computation time
    cout<<" Time elapsed(in ms): "<<elapsed_time<<endl;

    // get the maximum element by comparing the results from each block
    int maxElement = -1;
    for(int i =0; i< blockPerGrid; i++){
        cout<<" Maximum in block i = "<<i<<", is: "<<h_c[i]<<endl;
        maxElement = max(maxElement, h_c[i]);
    }
    cout<<" Maximum Element found is: "<<maxElement<<endl;

    return 0;
}