#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
using namespace std;


#define N 768 // column
#define K 512 // common dimension
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

// Sparse Matrix-Dense Matrix (SpMM)
__global__ void sparseMatDenseMatMul(int* values, int* colInd, int* rowptr, int* d_mat, int* c) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int acc[N];

    if(tid<M) { // one thread per row (M= number of rows)

        // per thread calculates values for each row of final matrix
        // initialize the accumulated sum for elements of one row
        for(int i =0;i<N;i++) {
            acc[i] = 0;
        }

        int start = rowptr[tid];  // inclusive
        int end = rowptr[tid + 1];  // exclusive

        for(int i = start; i<end; i++) {
            // go over all the elements of one row of Final matrix
            for(int j =0; j<N; j++) {
                acc[j] += values[i] * d_mat[colInd[i] * N  + j];  // values * [i, j]th value in dense Mat[KxN]
            }
        }

        // get the result
        for(int i =0; i<N; i++) {
            c[tid*N + i] = acc[i];
        }

    }
}

void getCSRpointers(int* mat, vector<int> &values, vector<int> &colIndex, vector<int> &rowptrs) {

    int x_val = 0;
    rowptrs.clear();
    rowptrs.push_back(0);
    int temp = 0;

    for(int i = 0; i<M; i++) {
        for(int j = 0; j<K; j++) {
            temp = mat[i*K + j];
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


    int h_a[M*K], h_b[N*K], h_c[M*N];
    int *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_b, K*N*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, M*N*sizeof(int)));

    for(int i=0;i<M;i++) {
        for(int j =0; j<K; j++) {
            h_a[i*K + j] = (i+j)%2;
        }
    }

    for(int i=0;i<K;i++) {
        for(int j=0;j<N;j++) {
            h_b[i*N + j] = (i+j) % 99;
        }
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

    CHECK_CUDA(cudaMemcpy(d_b, h_b, N*K*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val, h_values.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowptr, h_rowPtr.data(), (M+1)*sizeof(int), cudaMemcpyHostToDevice));

    int blocksPerGrid = (M + threadsPerBlock - 1)/threadsPerBlock;

    sparseMatDenseMatMul<<<blocksPerGrid, threadsPerBlock>>>(d_val, d_colInd, d_rowptr, d_b, d_c);


    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M*N*sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

    cout<<"ELapsed time(in ms) is "<<elapsed_time<<endl; //8.10

    for(int i = 0; i< nnz && i<20; i++) {
        cout<<"CSR indexing format at i:"<<i<<", colId:"<<h_colInd[i]<<", value:"<<h_values[i]<<endl;
    }
    for(int i = 0; i< N*K && i<20; i++) {
        cout<<"Dense Matrix at i:"<<i<<", is "<<h_b[i]<<endl;
    }
    for(int i = 0; i< M*N && i<20; i++) {
        cout<<"Sparse Matrix-Dense Matrix (SpMM) multiplication using at i:"<<i<<" is "<<h_c[i]<<endl;
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