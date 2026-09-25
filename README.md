# Parallel Programming Projects

> [!WARNING]
> Archived and no longer maintained; kept for reference. Written in 2015 for the course's clusters (SDSC Trestles and TACC Stampede, both since retired) and Udacity's CUDA grading servers. The course-provided harness code isn't included, so none of it builds on its own.

- [Parallel Programming Projects](#parallel-programming-projects)
  - [Overview](#overview)
    - [Projects](#projects)
      - [Square DGEMM](#square-dgemm)
      - [N-body](#n-body)
    - [CUDA](#cuda)
  - [What's in here](#whats-in-here)

## Overview

My C, C++, and CUDA homework for CSCI 595, Parallel Programming, at the University of Montana in spring 2015. They're toy problems, but they were hard, and each of the two big ones comes with a write-up of what I tried and what went wrong.

**Tech:** C, C++, SSE2/SSE3 intrinsics, OpenMP, MPI, CUDA, SLURM

### Projects

#### Square DGEMM
 
A matrix-matrix multiply tuned with SSE2/SSE3 intrinsics, loop unrolling, and cache blocking. It averages 69.84% of peak CPU performance, up from about 10% for the naive version.

It took 36 versions of `dgemm-optimized.c` to reach an average of 69.84% of peak CPU performance, up from about 10% for the naive version. Unrolling into 4x4 SSE2 blocks got it to about 40%, and swapping one SSE2 instruction for an SSE3 one took it from 54% to 68%. Handling matrix sizes that weren't multiples of 4 took the most time, and one blocking bug that cost me more than five hours turned out to be `=` where I needed `+=`.

#### N-body

A particle simulation in serial, OpenMP, and MPI versions using spatial binning, run on TACC Stampede. 

I started with a Barnes-Hut quadtree, but it had to be rebuilt every step and didn't split well across MPI processes, so I started over with binning: each particle only looks at particles in its own and neighboring bins. The quadtree is still in `src/bhtree/`.


### CUDA

Five assignments from Udacity's CS344 course, each in its own `student_func.cu`: 

- a Gaussian blur,
- tone mapping (reduce, histogram, prefix sum),
- red-eye removal with a GPU radix sort,
- a fast histogram (8 ms → 1.6 ms with per-block shared-memory histograms), and 
- seamless image blending with Jacobi iterations

## What's in here

| Path | What it is |
|---|---|
| `Optimized C Code/Square Dgemm/` | `dgemm-optimized.c` and its write-up |
| `Optimized C Code/N Body/src/` | Serial, OpenMP, and MPI versions, the Barnes-Hut quadtree, and the Stampede Makefile |
| `Optimized C Code/N Body/stdout/` | SLURM job scripts and output from the Stampede runs |
| `Optimized C Code/N Body/CSCI-595_N-Body_Writeup.pdf` | The N-body write-up |
| `Udacity Assignments/` | One folder per CUDA assignment, with a screenshot of the passing run (or a write-up, for the histogram) |
