#include <iostream>
#include <cuda_runtime.h>
using namespace std;

int main() {
    
    cout<<"Hello World!"<<endl;
    int count;
    cudaGetDeviceCount(&count);
    cout<<"GPU device count: %d\n" << count;
    return 0;
}