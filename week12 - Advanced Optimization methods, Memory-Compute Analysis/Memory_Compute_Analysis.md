## Warp Scheduling and latency tolerance
When threads are assigned to SMs, there are usually more threads (1024) assigned to an SM than there are cores(128) in the SM. Let’s clarify SM’s execution capability:
* Threads are grouped into warps (32 threads)
* An SM issues instructions per warp, not per thread

That is, each SM has only enough execution units to execute a subset of all the threads assigned to it at any point in time. Thus, the hardware can execute instructions only for a subset of all warps (1024/32 = 32) in the SM. 

<b>"Why do we have so many warps if only few are executed at any instant ?"</b>

Because GPUs needs to tolerate long-latency operations such as global memory access which takes 400-800 cycles. While an instruction is executed by warp in around 4-10 cycles. Thus, in order to avoid waiting long for global memory to be loaded, warp scheduler executes any available warps which is ready. This mechanism of filling the latency time of operations from some threads with work from other threads is often called <b>"latency tolerance" or "latency hiding".</b>

For latency tolerance to be effective, it is desirable for an SM to have many more threads assigned to it than can be simultaneously supported with its execution resources to maximize the chance of finding a warp that is ready to execute at any point in time. For example, in an Ampere A100 GPU, an SM has 64 cores but can have up to 2048 threads assigned to it at the same time. Thus the SM can have up to 32 times more threads. This oversubscription of threads to SMs is essential for latency tolerance. 

## GPU Resource Partitioning and Occupancy
Occupancy : The ratio of the number of warps assigned to an SM to the maximum number it supports is referred to as occupancy.
The execution resources in an SM include registers, shared memory, thread block slots, and thread slots.

<b>"How SM resources are partitioned ?"</b>
    
The SM resources are dynamically partitioned across threads to support their execution. For example, an Ampere A100 GPU can support a maximum of 32 blocks per SM, 64 warps (2048 threads) per SM, and 1024 threads per block. If a grid is launched with a block size of 1024 threads (the maximum allowed), the 2048 thread slots in each SM are partitioned and assigned to 2 blocks. 
Similarly, if a grid is launched with a block size of 512, 256, 128, or 64 threads, the 2048 thread slots are partitioned and assigned to 4, 8, 16, or 32 blocks, respectively.

But dynamic partitioning of resources can sometimes <b>cause underutilization of resources due to device constraints.</b> 
* For example, some SMs can only support 32 blocks slots at once. This means that only 1024 of the thread slots will be utilized, that is, 32 blocks with 32 threads each. <b>The occupancy in this case is (1024 assigned threads)/(2048 maximum threads) 50%.</b> Therefore to fully utilize the thread slots and achieve maximum occupancy, one needs at <b>least 64 threads</b> in each block.
* When the maximum number of threads per block is <b>not divisible </b> by the block size. If a block size of 768 is selected, the SM will be able to accommodate only 2 thread blocks (1536 threads), leaving 512 thread slots unutilized. <b>The occupancy in this case is (1536 assigned threads)/(2,048 maximum threads) 75%.</b>
* The impact of other resource constraints, such as registers and shared memory. <b>Automatic variables declared in a CUDA kernel are placed into registers.</b>
For example, the Ampere A100 GPU allows a maximum of 65,536 registers per SM. To run at full occupancy, each SM needs enough registers for 2048 threads, which means that each thread should not use more than (65,536 registers)/(2048 threads) - 32 registers per thread. If a kernel uses 64 registers per thread, the maximum mumber of threads that can be supported with 65,536 registers is 1024 threads. <b>The occupancy will be at most 50%.</b> Thus, based on required registers per thread, SM will dynamicall <b>adjust to support the possible number of blocks.</b>

# Memory Compute Analysis
The compute to global memory access ratio, defined as the number of FLOPs performed for each byte access from the global memory within a region of a program. This ratio is sometimes also referred to as <b>arithmetic intensity or computational intensity</b>. It reflects the amount of computation done by an application for every byte of data loaded.
    
For example, the Ampere A100 GPU has a peak global memory bandwidth of 1555 GB/second. Since, simple matrix multiplication kernel performs 0.25 FLOP/B (2 FLOP (mul + add) to 8B (floating point access from A & B)). The throughput of single-precision FLOPs that can be performed by kernel = <b>0.25 FLOP/B * (1555 GB/S) => 389 GFlOPS</b> which is just 2% of the peak throughput of the A100 GPU, which is 19,500 GFLOPS. This is due to high cost of rate of memory access from global memory to CUDA cores. This type of programs/kernels are <b>memory-bound programs</b>.

Applications with higher computational intensity are compute-bound and are not limited by memory bandwidth. 
    
<b>"How can we increase the computational intensity of the program?"</b>

We need to reduce the number of global memory accesses it performs for same amount of floating point operations. For example, to achieve maximum throughput in the Ampere A100 GPU, we need to have (19,500 GFLOP/S peak throughput / 1555 GB/S peak memory) = 12.5 FLOP/B. This ratio means that for every 4-byte floating point value accessed, there must be about 50 floating-point operations performed.

### Impact of Tiling in MatMul
Here, we load a tile of A and B once and reuse them many times from shared memory. Let's say tile size is TILE_WIDTH * TILE_WIDTH.
    
For tile A: TILE_WIDTH × TILE_WIDTH = TILE_WIDTH² elements
    
For tile B: TILE_WIDTH × TILE_WIDTH = TILE_WIDTH² elements
    
Total global loads per tile: <b>2 * TILE_WIDTH² floats</b> = <b>8 * TILE_WIDTH² B</b>

Total Computation: TILE_WIDTH x TILE_WIDTH x TILE_WIDTH x (1 Mul + 1 Add Ops) = <b>2 * TILE_WIDTH * TILE_WIDTH²</b>

<b>Computational Intensity per tile = Total Compute (FLOP) / Total global loads per tile (B) = TILE_WIDTH/4 </b>

For example, if TILE_WIDTH = 16, Computational Intensity = 4 FLOP/B, which is 16 times higher than previous 0.25 FLOP/B. This improvement allows the device to achieve <b>(1555 GB/second)*(4 OP/B)=6220 GFLOPS throughput which is now 32% os the peak throughput.</b> Thus, tiling converts a memory-bound kernel into a much more compute-heavy kernel.

### Limitations to improve computational intensity by just increasing TILE_WIDTH
The very resources that increase computational intensity (registers and shared memory) reduce occupancy if overused because occupancy is limited by per-thread and per-block resource usage. As we know that SM has finite Registers, Shared memory, Thread slots and Blocks per SM, for example <b>fixed shared memory size (48KB for RTX A4000, 164KB for A100).</b> If you want all thread slots occupied, the average shared memory per thread must be 164KB/2048 = 82B per thread for A100.

In tiled MatMul, While Tile size is limited by Shared Memory size per SM, It does not hinder the kernel occupancy. Threads per block is TILE_WIDTH² and shared memory usage per block is 8 * TILE_WIDTH² B. Therefore, <b>average shared memory usage per thread is 8B/thread (<< 82B).</b>

In counter example, if 256 threads in a block uses 32KB shared memory, then per thread usage is 132B (>82B). With such shared memory usage, the kernel cannot achieve full occupancy. Each SM can host a maximum of only (164 KB)/(132 B/thread)-1272 threads to avoid exceedng the limit. Therefore the <b>maximum achievable occupancy of this kernel will be (1272 assigned threads)/(2048 maximum threads)-62%.</b>

Therefore, Increasing data reuse does not automatically increase performance if it destroys occupancy. Thus, maximum occupancy is not always optimal.


  
