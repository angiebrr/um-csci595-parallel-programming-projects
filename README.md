# Parallel Programming Projects

This repository is a collection of my optimized C and CUDA code snippets from my parallel programming class.

## Optimized C Code

This folder has two projects: Square Dgemm and N Body. 

Square Dgemm performs an optimized matrix-matrix multiply by using SSE2/SSE3 intrinsics and blocking data into the cache. A write-up is provided alongside the code (CSCI-595_Square-Dgemm_Writeup.pdf) explaining my methods and my struggles.

N Body performs a simple method of binning to efficiently apply forces to an N-body simulation, or the method that divides the world into bins and uses only the particles in surrounding bins to apply forces to each particle in a given bin. Like the last, a write-up is provided alongside the code (CSCI-595_N-Body_Writeup.pdf) explaining my methods and my struggles.

## Udacity Assignments

This folder has a variety of assignments that were given by the Udacity course for CUDA programming. Unlike the Optimized C Code, there isn't a write-up *per-se*. Mostly the code is documented via comments and a screenshot is provided of the output of the code on the Udacity servers.

