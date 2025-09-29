#include <iostream>
#include <cuda_runtime.h>
using namespace std;


int main() {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout<<"Number of CUDA-capable processor available: "<<deviceCount<<endl;  // 1

    if(deviceCount<2){
        cout<<"Existing System does not support multi-GPU programming"<<endl;
    }

    return 0;
}