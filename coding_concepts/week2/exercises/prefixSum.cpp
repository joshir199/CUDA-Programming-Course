#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 500
#define threadsPerBlock 512

//Only when the N is less than totals threads per block
__global__ void kernelPrefixSumSimple(int* a, int* c) {
    int tid = threadIdx.x;
    __shared__ int cache[threadsPerBlock];
    int cacheId = threadIdx.x;

    // copy the array data into shared memory
    if(tid<N) {
        cache[tid] = a[tid];
    } else {
        cache[tid] = 0;
    }
    __syncthreads();

    // Up-sweep phase
    //Do the offset doubling pattern sum for each element
    int i = 1;
    while(i<N) {
        int temp = 0;
        if(tid>=i){
            temp = cache[tid - i];
        }
        __syncthreads();
        cache[tid] += temp;
        __syncthreads();
        i = i*2;
    }

    // The resultant sum will have inclusive prefix sum for each element
    // down-sweep phase
    // Shift each item to right to get exclusive sum
    int totalSumTilltid = cache[tid];
    if(tid == 0) {
        cache[tid] = 0;
    } else {
        cache[tid] = totalSumTilltid - a[tid];
    }
    __syncthreads();

    if(tid<N){
        c[tid] = cache[tid];
    }

}

int main() {
    int h_a[N], h_c[N]; // define variable to host

    int *d_a, *d_c;   // define variable for device

    // allocate memory to variable for device
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));

    // initialize he vector A
    for(int i =0; i<N; i++) {
        h_a[i] = 2*i - 101;
    }

    // transfer data from host to Device
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

    int blockPerGrid = min(32, (N + threadsPerBlock-1)/threadsPerBlock);
    // call the kernel to execute on GPU
    // Keep in mind that threadsPerBlock > N
    kernelPrefixSumSimple<<< blockPerGrid, threadsPerBlock>>>(d_a, d_c);

    // collect the result from Device
    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0 ; i<N;i++) {
        // print the result
        cout<<"Exclusive Prefix Sum at i= " << i << " : " << h_c[i] <<endl;
    }

    // free the global memory
    cudaFree(d_a);
    cudaFree(d_c);

    return 0;
}