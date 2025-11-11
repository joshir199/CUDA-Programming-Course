#include <iostream>
#include <cuda_runtime.h>
using namespace std;
#define N 1000

__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int tid = blockIdx.x;
    if(tid<n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {

    int h_a[N], h_b[N], h_c[N]; // variable on host
    int *d_a, *d_b, *d_c;       // variable on device

    // allocate memory on device
    cudaMalloc(&d_a, N*sizeof(int));
    cudaMalloc(&d_b, N*sizeof(int));
    cudaMalloc(&d_c, N*sizeof(int));

    // assigning values to the arrays on host
    for(int i=0;i<N;i++) {
        h_a[i] = 2*i + 9;
        h_b[i] = 2*(i+1) + 19;
    }

    // copy data from host to device for execution
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice);

    // call the kernel to execute the code for desired operation
    vectorAdd<<<N, 1>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }

    // copy resultant vector from device to host
    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    // print the output
    for(int i=0;i<N; i++) {
        cout<< h_a[i] << " " << h_b[i] << " " << h_c[i] << endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}