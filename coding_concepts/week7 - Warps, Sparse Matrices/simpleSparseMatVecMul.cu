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


__global__ void sparseMatVecMul(int* values, int* colInd, int* rowptr, int* vec, int* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<M) { // thread per row (M= number of rows)
        int sum = 0;  // sum variable for result per Row
        int start = rowptr[tid];  // inclusive
        int end = rowptr[tid + 1];  // exclusive

        for(int i = start; i<end; i++) {
            sum += values[i] * vec[colInd[i]];  // values * value at same index in vector
        }
        c[tid] = sum;
    }
}

void getCSRpointers(int* mat, vector<int> &values, vector<int> &colIndex, vector<int> &rowptrs) {

    int x_val = 0;
    rowptrs.clear();
    rowptrs.push_back(0);
    int temp = 0;

    for(int i = 0; i<M; i++) {
        for(int j = 0; j<N; j++) {
            temp = mat[i*M + j];
            if(temp != 0) {
                values.push_back(temp);
                colIndex.push_back(j);
                x_val++;
            }
        }
        rowptrs.push_back(x_val);
    }
}

int main() {

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));


    int h_a[M*N], h_b[N], h_c[M];
    int *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_b, N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, M*sizeof(int)));

    for(int i=0;i<M;i++) {
        for(int j =0; j<N; j++) {
            h_a[i*M + j] = ((i + j)/2) % 2;
        }
    }

    for(int i=0;i<N;i++) {
        h_b[i] = (2*i + 1)%999;
    }

    //all non-zero elements, row by row ? length = nnz
    vector<int> h_values;
    // the column index of each value ? length = nnz
    vector<int> h_colInd;
    //cumulative count of non-zeros up to each row ? length = M + 1
    vector<int> h_rowPtr;

    getCSRpointers(h_a, h_values, h_colInd, h_rowPtr);

    int nnz = h_values.size();

    int *d_val, *d_colInd, *d_rowptr;

    CHECK_CUDA(cudaMalloc(&d_val, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colInd, nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_rowptr, (M+1)*sizeof(int)));


    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val, h_values.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowptr, h_rowPtr.data(), (M+1)*sizeof(int), cudaMemcpyHostToDevice));

    int blocksPerGrid = (M + threadsPerBlock - 1)/threadsPerBlock;

    sparseMatVecMul<<<blocksPerGrid, threadsPerBlock>>>(d_val, d_colInd, d_rowptr, d_b, d_c);


    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"ELapsed time(in ms) is "<<elapsed_time<<endl; //0.59

    for(int i = 0; i< nnz && i<20; i++) {
        cout<<"CSR indexing format at i:"<<i<<", colId:"<<h_colInd[i]<<", value:"<<h_values[i]<<endl;
    }
    for(int i = 0; i< N && i<20; i++) {
        cout<<"Vector at i:"<<i<<", is "<<h_b[i]<<endl;
    }
    for(int i = 0; i< M && i<20; i++) {
        cout<<"Sparse Matrix-Vector multiplication using at i:"<<i<<" is "<<h_c[i]<<endl;
    }

    CHECK_CUDA(cudaFree(d_rowptr));
    CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}