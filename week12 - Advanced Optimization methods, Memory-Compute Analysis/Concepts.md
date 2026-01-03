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
The output-centric decomposition usually exhibits the gather memory access behavior. Gather-based access patterns are usually <b>more desirable</b> in CUDA devices because the threads can accumulate their results in their private registers. For example, Scan part of Prefix sum, Reduction, Convolution etc are output-centric problems.


The input-centric decomposition, by contrast, usually exhibits the scatter memory access behavior, in which each thread scatters or distributes the effect of an input value into the output values. Scatter-based access patterns are usually undesirable in CUDA devices because multiple threads can update the same grid point at the same time, thus requiring atomic updates. These atomic operations are significantly slower than the register accesses that are used in the output-centric decomposition. For example, Histogram count,  Addition part of Prefix sum etc are input centric problems.
