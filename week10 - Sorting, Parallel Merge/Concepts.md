# Sorting
Sorting is one of the earliest and important applications for computers. 
A sorting algorithm arranges the elements of a list into a certain order.
More formally, any sorting algorithm must satisfy the following two conditions:
* The output is in either nondecreasing or nonincreasing order.
* The output is a permutation of the input.

## Sorting methods
* Comparison based - Merge Sort, Odd-Even Sort, Bitonic Sort
* Non-comparison based - Radix Sort

Comparison-based sorting algorithms <b>cannot achieve</b> better than O(N·logN) complexity when sorting a list of N elements 
because they must perform a minimal number of comparisons among the elements. 
In contrast, some of the noncomparison-based algorithms can achieve <b>better than</b> O(N·logN) complexity, 
but they may not generalize to arbitrary types of keys.

        
Radix sort can achieve lower time complexity than other sorting algorithms like merge sorts, bitonic sorts because it is noncomparison sorting method. 
However, radix sort can only be applied to sorting with certain types of keys. Other comparison based sorting methods like merge sort and bitonic 
sort can be used with any types of keys that has well-defined comparison operator.

# Radix Sort
It is a noncomparison-based sorting algorithm that works by distributing the keys that are being sorted into buckets on 
the basis of a radix value (or base in a positional numeral system). If the keys consist of multiple digits, the distribution of the keys 
is repeated for each digit until all digits are covered.
        
Each iteration is stable, preserving the order of the keys within each bucket from the previous iteration.
