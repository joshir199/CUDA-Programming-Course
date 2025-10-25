#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;

#define N 1024 // column
#define M 1024 // row
#define threadsPerBlock 256

#define CHECK_CUDA(call) do {                    \
    cudaError_t e = (call);                      \
    if(e != cudaSuccess) {                        \
        cout<<"CUDA Error: "<<cudaGetErrorString(e) \
        <<" in "<<__FILE__<<" at "<<__LINE__<<endl; \
        exit(1);                                     \
    }                                               \
} while(0)

// Calculate Sparse Matrix - Vector Multiplication with matrix in COO format
// Also called SpMV (Sparse Matrix-Vector multiplication)
__global__ void sparseMatVecMulUsingCOO(int* values, int* colInd, int* rowInd, int* vec, int* c, int nnz) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<nnz) { // thread per non-zero elements
        // atomically add to certain row of output vector with product of element in matrix with vector
        atomicAdd(&c[rowInd[tid]], values[tid]*vec[colInd[tid]]);
    }
}

// Convert Sparse matrix into its COO (Coordinate) format with Column, Row and Values arrays
void getCOOarrays(int* mat, vector<int> &values, vector<int> &colIndex, vector<int> &rowIndex) {

    int temp = 0;

    for(int i = 0; i<M; i++) {
        for(int j = 0; j<N; j++) {
            temp = mat[i*M + j];
            if(temp != 0) {
                values.push_back(temp);
                colIndex.push_back(j);
                rowIndex.push_back(i);
            }
        }
    }
}

int main() {
    // Define and create Events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    // define variables for device and host
    int h_a[M*N], h_b[N], h_c[M];
    int *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_b, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, M*sizeof(int)));

    // create dummy sparse matrix
    for(int i=0;i<M;i++) {
        h_c[i] = 0;
        for(int j =0; j<N; j++) {
            h_a[i*M + j] = ((i + j)/2) % 2;
        }
    }

    // create dummy dense vector
    for(int i=0;i<N;i++) {
        h_b[i] = (2*i + 1)%999;
    }

    //all non-zero elements, row by row ? length = nnz
    vector<int> h_values;
    // the column index of each value ? length = nnz
    vector<int> h_colInd;
    //cumulative count of non-zeros up to each row ? length = M + 1
    vector<int> h_rowInd;

    // get COO format arrays
    getCOOarrays(h_a, h_values, h_colInd, h_rowInd);

    // Number of non-zero elements in Sparse matrix
    int nnz = h_values.size();

    int *d_val, *d_colInd, *d_rowInd;

    CHECK_CUDA(cudaMalloc(&d_val, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colInd, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowInd, nnz*sizeof(int)));

    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val, h_values.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowInd, h_rowInd.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    // Initialize the output vector with zero and pass it to device
    CHECK_CUDA(cudaMemcpy(d_c, h_c, M*sizeof(int), cudaMemcpyHostToDevice));

    // Launch NNZ (non-zero elements) size of threads
    int blocksPerGrid = (nnz + threadsPerBlock - 1)/threadsPerBlock;

    sparseMatVecMulUsingCOO<<<blocksPerGrid, threadsPerBlock>>>(d_val, d_colInd, d_rowInd, d_b, d_c, nnz);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"ELapsed time(in ms) is "<<elapsed_time<<endl; //0.77

    for(int i = 0; i< nnz && i<20; i++) {
        cout<<"COO indexing format at i:"<<i<<", colId:"<<h_colInd[i]<<", Value:"<<h_values[i]<<endl;
    }
    for(int i = 0; i< N && i<20; i++) {
        cout<<"Vector at i:"<<i<<", is "<<h_b[i]<<endl;
    }
    for(int i = 0; i< M && i<20; i++) {
        cout<<"Sparse Matrix-Vector multiplication using at i:"<<i<<" is "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_rowInd));
    CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}