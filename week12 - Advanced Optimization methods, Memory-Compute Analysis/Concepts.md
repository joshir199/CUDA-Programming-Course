# Optimisations methods
In order to apply any optimization, we need to understand what kind of resource is limiting the performance (bottleneck). Bottlenecks can be arised due multiple access to global memory(speed & memory bandwidth), mutliple atomic operations (speed) etc.

<div align="center">
    <img src="https://github.com/joshir199/CUDA-Programming-Course/blob/main/other%20resources/optimisation_techniques.png" alt="optimisation techniques">
    <p><i>Fig: Different Optimisation techniques</i></p>
</div>

<b>** coalesced</b> : Threads in the same warp access adjacent memory locations. (CUDA hardware can access all those memory in each warp using single request)

**************************************
## Problem decomposition
In parallel computing, the problem must be formulated in such a way that it can be decomposed into subproblems that can be safely solved at the same time.
The two most common strategies for decomposing a problem for parallel execution are follows:
* output-centric - assigns threads to process different units of the <b>output data</b> în parallel
* input-centric - assigns threads to process different units of the <b>input data</b> in parallel


<div align="center">
    <img src="https://github.com/joshir199/CUDA-Programming-Course/blob/main/other%20resources/problem_decomposition.png" alt="problem decomposition">
    <p><i>Fig: Problem decomposition methods</i></p>
</div>

      
While both decomposition strategies lead to the same execution results, they can exhibit very different performance in a given hardware system.
The output-centric decomposition usually exhibits the gather memory access behavior. <b>Gather-based access</b> patterns are usually <b>more desirable</b> in CUDA devices because the threads can accumulate their results in their private registers. For example, Scan part of Prefix sum, Reduction, Convolution etc are output-centric problems.


The input-centric decomposition, by contrast, usually exhibits the scatter memory access behavior, in which each thread scatters or distributes the effect of an input value into the output values. <b>Scatter-based access</b> patterns are usually undesirable in CUDA devices because multiple threads can update the same grid point at the same time, thus requiring atomic updates. These atomic operations are significantly slower than the register accesses that are used in the output-centric decomposition. For example, Histogram count,  Addition part of Prefix sum etc are input centric problems.

## Avoid Control divergence
When threads in the same warp follow different execution paths, we say that these threads exhibit control divergence, that is, they diverge in their execution. 
The <b>cost of divergence</b>, however, is the extra passes the hardware needs to take to allow different threads in a warp to make their own decisions as well as the execution resources that are consumed by the inactive threads in each pass. 
<b>For example</b>, for an if-else construct, the execution works well when either all threads in a warp execute the if-path or all execute the else-path. However, when threads within a warp take different control flow paths, the SIMD hardware will <b>take multiple passes</b> through these paths, one pass for each path. 



If the decision condition is based on threadIdx values, the control statement can potentially cause <b>thread divergence</b>. For example, the statement if(threadIdx.x > 2) {...} causes the threads in the first warp of a block to follow two divergent control flow paths. Threads 0, 1, and 2 follow a different path than that of threads 3, 4, 5, and so on. A prevalent reason for using a control construct with thread control divergence is handling boundary conditions when mapping threads to data. This is usually because the total number of threads needs to be a multiple of the thread block size, whereas the size of the data can be an arbitrary number. 
For example, let's assume VectorAddition example where the vector length is 1003 and we picked 64 as the block size. we have an if(i<n) statement and we need to disable the last 21 threads in thread block 15 from doing work that is not expected. Only the last warp will have control divergence. That is, control divergence will affect only about 3% of the execution time. 
