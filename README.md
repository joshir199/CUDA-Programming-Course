# CUDA Programming Course
This repository is built for systematic learning of CUDA Programming for beginner/intermediate. 
   
CUDA (Compute Unified Device Architecture) is NVIDIA’s platform for parallel computing on GPUs. CUDA is particularly relevant for AI because it enables massive parallelization for tasks like neural network training, inference, and data processing, which are computationally intensive.


The repository contains solved examples along with major concepts planned across various weeks.
<div align="center">
    <img src="https://github.com/joshir199/CUDA-Programming-Course/blob/main/week01%20-%20CUDA%20Intro%2C%20Threads%2C%20Blocks/architecture_diagram.png" alt="Memory hierarchy of a GPU">
    <p><i>Figure: Memory hierarchy of a GPU</i></p>
</div>


Key concepts about CUDA architecture:
   
<b>Streaming Multiprocessors (SMs)</b> : SMs are the core processing units in a GPU, each containing CUDA cores, registers, and shared memory. Example: An NVIDIA RTX 3080 has 68 SMs with 128 CUDA cores per SM.
  
<b>CUDA Cores</b> : These are the basic processing units for parallel computation. More CUDA cores generally mean higher parallel throughput.

<b>Memory Hierarchy</b>:
* Global Memory: Large but slow, accessible by all threads (e.g., 16 GB on RTX A4000).
* Shared Memory: Fast, on-chip memory shared within a thread block (e.g., 48 KB per SM on RTX A4000).
* Registers: Fastest, private to each thread.
* Constant/Texture Memory: Specialized for specific use cases (e.g., read-only data, max 10KB for constant memory on RTX A4000).

_________________________________________________________________________________________________________
     
The whole course has been divided into multiple weeks, each covering important topics in details. 

* For beginners, I recommend to start with book "CUDA by Example" mentioned below and practice the coding questions along.
* For intermediate and above, start with coding practices and learn important concepts from book "Programming Massively Parallel Processors" and lecture videos mentioned in reference section.
* Concepts & Notes for each week are added within each week's folder

### Prerequisites
* Hardware: Ensure you have access to an NVIDIA GPU
* Software: Install the CUDA Toolkit and a compatible NVIDIA driver. Use an IDE like for C/C++ coding.





*************************************************
# Following are the weekly topic covered in this course:
* Week 1 - CUDA Intro, Threads, Blocks
* Week 2 - Grids, Shared Memory, Reduction Pattern
* Week 3 - 2D Block, Grids, Matrices
* Week 5 - Atomics Operations, Privatisation
* Week 6 - Streams, Integrated Memory
* Week 7 - Advanced Tiling, Threads Collaboration
* Week 8 - Warp Concepts, Sparse Matrices
* Week 9 - CUDA libraries cuBLAS, cuDNN
* Week 10 - Sorting, Parallel Merge
* Week 11 - Graph algorithms, BFS, Graph traversal
* Week 12 - Advanced Optimization methods, Memory-Compute Analysis

*********************************************
# For coding CUDA kernels for different problems:
* ultimate_coding_practice
  * Easy
  * Medium
  * Hard 

**********************************************
# References:
* Books
  * "CUDA by Example: An Introduction to General-Purpose GPU Programming" by Jason Sanders and Edward Kandrot
  * "Programming Massively Parallel Processors" (4th Edition) by Wen-mei Hwu and David Kirk
* Youtube links:
  * Onur Mutlu Lectures : ![Programming Heterogeneous Computing Systems with GPUs (Fall 2022)](https://www.youtube.com/watch?v=Yzn8pxq-gtI&list=PL5Q2soXY2Zi8J0QbZ0c9ERLdnFRnO5U8C)
* Coding Practice Link:
  * LeetGPU : https://leetgpu.com/
*******************************************
# Fun Facts:
In CUDA programming, Floating-point arithmetic is not associative because of the rounding of intermediate results. {(A + B) + C != A + (B + C)} 
