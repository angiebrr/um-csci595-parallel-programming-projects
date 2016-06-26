#include "utils.h"

// ===================================================================================================================
// student_func.cu
// -------------------------------------------------------------------------------------------------------------------
// Angela Gross
// CSCI-595: Parallel Programming
// Homework 3
// Spring 2015
// -------------------------------------------------------------------------------------------------------------------
// This transforms the log of the luminance channel of an image by compressing its range to [0, 1], and this is done 
// by making a cumulative distribution of these luminance values. This is done in parallel by implementing a reduce, 
// histogram, and finally an inclusive prefix sum on those values (all using CUDA).
//
// In addition to the forums on the Udacity course, the following websites assisted in my implementation:
// - http://www.cuvilib.com/Reduction.pdf
// - http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
// ===================================================================================================================

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// TYPEDEF AND OPERATIONS FOR REDUCTIONS
typedef float (*op_func) (const float &, const float&);
__device__ float min_op(const float &x, const float& y) { return x<y?x:y; }
__device__ float max_op(const float &x, const float& y) { return x>y?x:y; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// REDUCE()
// -------------------------------------------------------------------------------------------------------------------
// A CUDA kernel that accepts a reduce operation (with return type float and arguments (float, float)) and performs 
// them on the global input array of length maxSize. 
//
// The two operations that have been created for this method are max and min. (See above)
// ===================================================================================================================
template<op_func op>
__global__ void reduce(const size_t maxSharedSize, const size_t maxGlobalSize, const float* const g_in, float* const g_out)
{
  // retrieve the thread's global id and id and check it isn't outside the range
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;
  if(gid > maxGlobalSize) return;

  // make shared memory
  const size_t localMaxSharedSize = maxSharedSize;
  extern __shared__ float s_data[];

  // put value in shared array and sync threads
  s_data[tid] = g_in[gid];
  __syncthreads();

  int oldS = blockDim.x;

  // perform reduction operations
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s && (tid + s) < localMaxSharedSize)
      s_data[tid] = op(s_data[tid + s], s_data[tid]);

    // if currently used s does not cover all the numbers computed by previous s
    if(tid == 0 && 2*s < oldS)
      s_data[tid] = op(s_data[tid + 2*s], s_data[tid]);
    
    __syncthreads();
    oldS = s;
  }

  // write result to global memory when finished
  if (tid == 0) g_out[blockIdx.x] = s_data[0];
}

// ===================================================================================================================
// LUM_HISTOGRAM()
// -------------------------------------------------------------------------------------------------------------------
// A CUDA kernel that creates a histogram using the luminance data into the specified bins.
// ===================================================================================================================
__global__ void lum_histogram(const size_t n, const float* g_in, int* g_bins, const size_t numBins, 
                              const float lumMin, const float lumRange)
{
  // retrieve the thread's global id and id and check it isn't outside the range
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid > n) return;

  // get item and put it into a bin
  int bin = (g_in[gid] - lumMin) / lumRange * numBins;

  // atomic add
  atomicAdd(&(g_bins[bin]), 1);
}

// ===================================================================================================================
// INCLUSIVE_PREFIX_SUM
// -------------------------------------------------------------------------------------------------------------------
// A CUDA kernel that uses a Hillis-Steele scan algorithm to perform an inclusive prefix sum on the input data.
//
// Note: Only supports one block of threads.
// ===================================================================================================================
__global__ void inclusive_prefix_sum(const size_t n, int* g_in, unsigned int* g_out)
{
  // retrieve id and check it isn't outside the range
  const int gid = threadIdx.x;
  if(gid > n) return;

  // make shared memory
  extern __shared__ float s_data[];

  // put global value in shared memory
  s_data[gid] = g_in[gid];
  __syncthreads();

  float value = 0.0;
  for(int offset = 1; offset < n; offset *= 2)
  {
    if(gid >= offset)
    {
      __syncthreads();
      value = s_data[gid - offset]; // prevent race condition
      __syncthreads();
      s_data[gid] += value;
    }
  }

  // write result to global memory when finished
  g_out[gid] = s_data[gid];
} 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// YOUR_HISTOGRAM_AND_PREFIXSUM()
// -------------------------------------------------------------------------------------------------------------------
// As the name suggests, this creates a histogram and creates a cumulative distribution function using the log of the
// luminance values of the image after finding the maximum and minimum log luminance values using CUDA.
// ===================================================================================================================
void your_histogram_and_prefixsum(const float* const d_logLuminance, unsigned int* const d_cdf, float &min_logLum, 
                                  float &max_logLum, const size_t numRows, const size_t numCols, const size_t numBins)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  float* d_out, * d_intermediate;

  const int LOG_LUMINANCE_SIZE = numRows * numCols;
  const int LOG_LUMINANCE_ARR_SIZE = LOG_LUMINANCE_SIZE * sizeof(float);

  const int THREADS = 1024;
  const int BLOCKS = LOG_LUMINANCE_SIZE / THREADS;
  const int SHARED_SIZE = THREADS;
  const int SHARED_ARR_SIZE = THREADS * sizeof(float);

  const int THREADS_REDUCE = BLOCKS;
  const int BLOCKS_REDUCE = 1;
  const int REDUCE_SHARED_ARR_SIZE = THREADS_REDUCE * sizeof(float);

  // MIN AND MAX
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // allocate and copy over the log lumincance into a device input array (for min)
  checkCudaErrors(cudaMalloc((void**)&d_out, LOG_LUMINANCE_ARR_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_intermediate, LOG_LUMINANCE_ARR_SIZE));

  // compute min via reduction
  reduce<min_op> <<<BLOCKS, THREADS, SHARED_ARR_SIZE>>> (SHARED_SIZE, LOG_LUMINANCE_SIZE, d_logLuminance, d_intermediate);
  reduce<min_op> <<<BLOCKS_REDUCE, THREADS_REDUCE, REDUCE_SHARED_ARR_SIZE>>> (SHARED_SIZE, LOG_LUMINANCE_SIZE, d_intermediate, d_out);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
 
  // copy reduced value to min variable from device
  checkCudaErrors(cudaMemcpy(&min_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));

  // compute max via reduction
  reduce<max_op> <<<BLOCKS, THREADS, SHARED_ARR_SIZE>>> (SHARED_SIZE, LOG_LUMINANCE_SIZE, d_logLuminance, d_intermediate);
  reduce<max_op> <<<BLOCKS_REDUCE, THREADS_REDUCE, REDUCE_SHARED_ARR_SIZE>>> (THREADS_REDUCE, LOG_LUMINANCE_SIZE, d_intermediate, d_out);

  // copy over reduced value to max variable form device
  checkCudaErrors(cudaMemcpy(&max_logLum, d_out, sizeof(float), cudaMemcpyDeviceToHost));
   
  // can print out to ensure accuracy
  //std::cout << "min = " << min_logLum << " max = " << max_logLum << "\n";

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  int* d_bins;
  const float BINS_ARR_SIZE = numBins * sizeof(int);

  const int THREADS_HISTO = 64;
  const int BLOCKS_HISTO = LOG_LUMINANCE_SIZE / THREADS_HISTO;

  const float LUM_RANGE = abs(max_logLum - min_logLum);

  // HISTOGRAM 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // allocate bins and initialize bins to a value of 0
  checkCudaErrors(cudaMalloc((void**)&d_bins, BINS_ARR_SIZE));
  checkCudaErrors(cudaMemset(d_bins, 0, BINS_ARR_SIZE));

  // generate a histogram
  lum_histogram<<<BLOCKS_HISTO, THREADS_HISTO>>> (LOG_LUMINANCE_SIZE, d_logLuminance, d_bins, numBins, min_logLum, LUM_RANGE);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  const int THREADS_SCAN = numBins;
  const int BLOCKS_SCAN = 1; // can't easily sync between blocks

  // has to be float since it's the same name as other shared
  const int SCAN_SHARED_ARR_SIZE = THREADS_SCAN * sizeof(float); 

  // PREFIX SUM
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // finally, compute the CDF
  inclusive_prefix_sum<<<BLOCKS_SCAN, THREADS_SCAN, SCAN_SHARED_ARR_SIZE>>> (THREADS_SCAN, d_bins, d_cdf);

  // CLEANUP
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  checkCudaErrors(cudaFree(d_out));
  checkCudaErrors(cudaFree(d_intermediate));
  checkCudaErrors(cudaFree(d_bins));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////