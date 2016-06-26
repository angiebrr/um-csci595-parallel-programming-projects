#include "utils.h"
#include "reference.cpp"

// ===================================================================================================================
// student_func.cu
// -------------------------------------------------------------------------------------------------------------------
// Angela Gross
// CSCI-595: Parallel Programming
// Homework 5
// Spring 2015
// -------------------------------------------------------------------------------------------------------------------
// This assignment is dedicated to producing an efficient histogram CUDA kernel. I have three kernels that are slow,
// fast, and faster in order to show the progression of optimization in this assignment.
//
// Methods of optimization included:
// - Using shared memory
// - Using local histograms in shared memory
// - Using efficient block sizes according to the number of SMs on the GPU (since we can fit 2 blocks in each SM, 
//   the best block size is 2 * SM_COUNT)
// - Since we're using an efficient block size (and with the thread per block needing to be equal to the number of 
//   bins) we have threads*blocks < numElems. Thus, we have to add as many elements as one thread can handle. This
//   has the added bonus of increasing the amount of work done for every __syncthreads() call.
// ===================================================================================================================

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// HELPER METHODS

// ===================================================================================================================
// SLOW_HISTO() 
// -------------------------------------------------------------------------------------------------------------------
// A CUDA kernel that uses a naive implementation of a histogram with atomic adding.
// Averages about 8 ms with 256 threads and n/156 blocks.
// ===================================================================================================================
__global__ void slow_histo(const unsigned int* const g_vals, unsigned int* const g_histo, unsigned int n)
{
  // retrieve the thread's global id and id and check it isn't outside the range
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid >= n) return;

  // atomic add by 1 into histogram
  atomicAdd(&(g_histo[g_vals[gid]]), 1);
}

// ===================================================================================================================
// FAST_HISTO() 
// -------------------------------------------------------------------------------------------------------------------
// A CUDA kernel that uses a ~4x faster version of slow_histo() by using a shared memory local histogram.
// Averages about 2 ms with 1024 threads (one for each bin) and n / 1024 blocks.
// ===================================================================================================================
__global__ void fast_histo(const unsigned int* const g_vals, unsigned int* const g_histo, unsigned int n)
{
  // retrieve the thread's global id and id and check it isn't outside the range
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid >= n) return;

  // declare external shared memory
  extern __shared__ unsigned int s_histo[];

  // zero out histogram and sync
  s_histo[threadIdx.x] = 0;
  __syncthreads();

  // atomically add value in local histogram and sync
  atomicAdd(&(s_histo[g_vals[gid]]), 1);
  __syncthreads();

  // atomically add local histogram into global histogram
  atomicAdd(&(g_histo[threadIdx.x]), s_histo[threadIdx.x]);
}

// ===================================================================================================================
// FASTER_HISTO() 
// -------------------------------------------------------------------------------------------------------------------
// A CUDA kernel that uses a ~5x faster version of slow_histo() by using a shared memory local histogram and doing 
// more work per thread call (i.e. atomically add more elements per thread before syncing threads)
// Averages about 1.5 ms with 1024 threads (one for each bin) and with 2 * multiprocessor blocks.
// ===================================================================================================================
__global__ void faster_histo(const unsigned int* const g_vals, unsigned int* const g_histo, unsigned int n)
{
  // retrieve the thread's global id -> don't need to check since we have less threads*blocks than elements
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // declare external shared memory
  extern __shared__ unsigned int s_histo[];

  // zero out histogram and sync
  s_histo[threadIdx.x] = 0;
  __syncthreads();

  // iterate over data in chunks for each thread and atomically add to local histogram and sync
  for(int i = gid; i < n; i += blockDim.x * gridDim.x) atomicAdd(&(s_histo[g_vals[i]]), 1);
  __syncthreads();

  // atomically add local histogram into global histogram
  atomicAdd(&(g_histo[threadIdx.x]), s_histo[threadIdx.x]);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// COMPUTEHISTOGRAM()
// -------------------------------------------------------------------------------------------------------------------
// This method does as it says: it computes a histogram. There are a few different attempts at optimizing a histogram,
// and I kept the different attempts at optimization around as separate kernels and commented them out. 
// ===================================================================================================================
void computeHistogram(const unsigned int* const d_vals,
                      unsigned int* const d_histo,
                      const unsigned int numBins,
                      const unsigned int numElems)
  {
    // SET UP
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // slow histogram variables
    // const unsigned int SLOW_NUM_THREADS = 256;
    // const unsigned int SLOW_NUM_BLOCKS = (int)ceil( (float)numElems/(float)SLOW_NUM_THREADS ) + 1;

    // fast histogram variables
    // const unsigned int FAST_NUM_THREADS = 1024;
    // const unsigned int FAST_NUM_BLOCKS = (int)ceil( (float)numElems/(float)FAST_NUM_THREADS ) + 1;

    // faster histogram variables
    // dynamically get number of multiprocessors since best block size is SM_COUNT * 2 (two blocks per SM)
    const unsigned int FASTER_NUM_THREADS = 1024;
    cudaDeviceProp properties;
    checkCudaErrors(cudaGetDeviceProperties(&properties, 0));
    const unsigned int FASTER_NUM_BLOCKS = 2 * (properties.multiProcessorCount);

    // local shared histogram array size (same number as bins)
    const unsigned int LOCAL_HISTO_SIZE = numBins;
    const size_t LOCAL_HISTO_ARR_SIZE = LOCAL_HISTO_SIZE * sizeof(unsigned int);

    // HISTOGRAM
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////   

    //slow_histo <<< SLOW_NUM_BLOCKS, SLOW_NUM_THREADS >>> (d_vals, d_histo, numElems);

    //fast_histo <<< FAST_NUM_BLOCKS, FAST_NUM_THREADS, LOCAL_HISTO_ARR_SIZE >>> (d_vals, d_histo, numElems);

    faster_histo <<< FASTER_NUM_BLOCKS, FASTER_NUM_THREADS, LOCAL_HISTO_ARR_SIZE >>> (d_vals, d_histo, numElems);
  }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
