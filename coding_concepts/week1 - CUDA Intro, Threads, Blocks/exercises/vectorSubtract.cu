#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define N 1100

__global__ void kernelSubtract(int* a, int* b, int* c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid<n) {
        c[tid] = a[tid] - b[tid];
    }
}

int main() {
    int h_a[N], h_b[N], h_c[N]; // define variable to host

    int *d_a, *d_b, *d_c;   // define variable for device

    // allocate memory to variable for device
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_b, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));

    // initialize he matrix A and B
    for(int i =0; i<N; i++) {
        h_a[i] = 2*i + 9;
        h_b[i] = i*i;
    }

    // transfer data from host to Device
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N*sizeof(int), cudaMemcpyHostToDevice);

    // call the kernel to execute on GPU
    kernelSubtract<<< (N + 127)/128, 128>>>(d_a, d_b, d_c, N);

    // collect the result from Device
    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    // print the result
    for(int i=0; i<N; i++) {
        cout<<"matrix A : "<<h_a[i]<<", matrix B : " << h_b[i] <<", matrix C = A-B : "<<h_c[i]<<endl;
    }

    // free the global memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}