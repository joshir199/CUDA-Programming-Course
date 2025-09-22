#include <iostream>
#include <cuda_runtime.h>
using namespace std;

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    cout << "Number of CUDA devices: " << deviceCount << endl; // 1

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        cout << "\n===== Device " << dev << " =====" << endl;
        cout << "Name: " << prop.name << endl;
        cout << "Total Global Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << endl;
        cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << endl;
        cout << "Registers per Block: " << prop.regsPerBlock << endl;
        cout << "Warp Size: " << prop.warpSize << endl;
        cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;

        cout << "Max Threads Dimension: "
                  << prop.maxThreadsDim[0] << " x "
                  << prop.maxThreadsDim[1] << " x "
                  << prop.maxThreadsDim[2] << endl;

        cout << "Max Grid Size: "
                  << prop.maxGridSize[0] << " x "
                  << prop.maxGridSize[1] << " x "
                  << prop.maxGridSize[2] << endl;

        cout << "Clock Rate: " << (prop.clockRate / 1000) << " MHz" << endl;
        cout << "Multiprocessor Count: " << prop.multiProcessorCount << endl;
        cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    }
    
    /*
    Number of CUDA devices: 1

    ===== Device 0 =====
    Name: NVIDIA RTX A4000
    Total Global Memory: 16093 MB
    Shared Memory per Block: 49152 bytes
    Registers per Block: 65536
    Warp Size: 32
    Max Threads per Block: 1024
    Max Threads Dimension: 1024 x 1024 x 64
    Max Grid Size: 2147483647 x 65535 x 65535
    Clock Rate: 1560 MHz
    Multiprocessor Count: 48
    Compute Capability: 8.6
    */


    return 0;
}
