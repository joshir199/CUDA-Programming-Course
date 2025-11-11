#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 1000

__global__ void addVectorThread(int* a, int* b, int* c) {
    int tid = threadIdx.x;
    if(tid<N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    
    int h_a[N], h_b[N], h_c[N]; //define array on host
    int *d_a, *d_b, *d_c;       // define array pointer on device
    
    // allocate memory for pointer on device
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_b, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));
    
    // fill data in host device
    for(int i=0; i<N ;i++) {
        h_a[i] = i*i + 1;
        h_b[i] = i*N - 99;
    }
    
    //copy data from host to device
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);
    
    // call kernel function
    addVectorThread<<<1, N>>>(d_a, d_b, d_c);
    
    //copy result from device to host
    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    //print the result
    for(int i=0;i<N;i++){
        cout<<h_a[i] <<" " << h_b[i] <<" " <<h_c[i]<<endl;
    }
    
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}