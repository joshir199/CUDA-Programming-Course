# CUDA Programming Course
This repository is for systematic way of learning CUDA Programming on GPU. 
The repository contains solved examples along with major concepts planned across various weeks.

![Memory hierarchy of a GPU](https://github.com/joshir199/CUDA-Programming-Course/blob/main/week01%20-%20CUDA%20Intro%2C%20Threads%2C%20Blocks/architecture_diagram.png)
 Figure : Memory hierarchy of a GPU

     
The whole course has been divided into multiple weeks, each covering important topics in details. 

* For beginners, I recommend to start with book "CUDA by Example" mentioned below and practice the coding questions along.
* For intermediate and above, start with coding practices and learn important concepts from book "Programming Massively Parallel Processors" and lecture videos mentioned in reference section.

## [TODO]
* Notes for each week will be added soon....
*************************************************
# Following are the weekly topic covered in this course:
* Week 1 - CUDA Intro, Threads, Blocks
* Week 2 - Grids, Shared Memory, Reduction Pattern
* Week 3 - 2D Block, Grids, Matrices
* Week 5 - Atomics Operations
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
