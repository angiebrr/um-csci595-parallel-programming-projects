#include "reference_calc.cpp"
#include "utils.h"
#include <math.h>

// ===================================================================================================================
// student_func.cu
// -------------------------------------------------------------------------------------------------------------------
// Angela Gross
// CSCI-595: Parallel Programming
// Homework 4
// Spring 2015
// -------------------------------------------------------------------------------------------------------------------
// This Udacity assignment implements red eye removal, and this is accomplished by first creating a score for every 
// pixel that tells us how likely it is to be a red eye pixel. This is already done in the grading framework code,
// and the function "your_sort" is receiving scores and pixel positions and the scores need to be sorted in 
// ascending order so the framework knows what pixels to alter to remove the red eye.
// 
// In order to sort these pixels, this code implements a 1-bit radix sort inspired by the following nvidia code:
// http://www.compsci.hunter.cuny.edu/~sweiss/course_materials/csci360/lecture_notes/radix_sort_cuda.cc
//
// Since it is only using 1-bit, we only need to scan the items and scatter them (and not use a histogram).
//
// The somewhat efficient multi-block scan was inspired from the following website:
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
// And it was performed with 3 kernel passes (instead of launching a kernel for every block)
//
// The code was optimized in the following ways:
// - A histogram was not used
// - A 3 kernel multi-block scan was developed
// - Pointers were passed around for the input and output so we didn't have to copy memory every iteration
// - Shared memory was utilized within the scan kernels
// - A block size (thread number) of 256 was the optimal size from test runs
// - Sychronizes and error checks were removed
// ===================================================================================================================

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_BITS 32
#define predicate(value, bit) ((value >> bit) & 1)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// HELPER METHODS

// ===================================================================================================================
// SWAP()
// -------------------------------------------------------------------------------------------------------------------
// Swaps unsigned integer pointers.
// ===================================================================================================================
void swap(unsigned int** a, unsigned int** b)
{
	unsigned int* temp = *a;
	*a = *b;
	*b = temp;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// CUDA KERNELS

// ===================================================================================================================
// PRESCAN()
// -------------------------------------------------------------------------------------------------------------------
// The first step in performing a scan in order to help find the scatter addresses and perform a radix sort. 
//
// This step performs a segmented inclusive scan where the size of a segment is the size of a block. The last thread
// in each block puts its value into a blockSums array so we can prepare for the next step in the mult-block scan.
// ===================================================================================================================
__global__ void prescan( unsigned int* g_inputVals, 
						 unsigned int* g_scan, 
						 unsigned int* g_blockSums,
						 unsigned int bit, 
						 size_t inputN )
{
	// make sure we don't walk off the edge
	unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int tid = threadIdx.x;
	if(gid >= inputN) return;

	// declare external shared memory
	extern __shared__ unsigned int s_data[];
	unsigned int sharedN = blockDim.x; // only go as far as the block dim
	
	// put predicate value in shared array and sync threads
	s_data[tid] = predicate(g_inputVals[gid], bit);
	__syncthreads();

	// perform inclusive scan
	unsigned int value = 0;
	for(unsigned int offset = 1; offset < sharedN; offset *= 2) 
	{
		if(tid >= offset)
		{
		  __syncthreads();
		  value = s_data[tid - offset];
		  __syncthreads();
		  s_data[tid] += value;
		}
	}

	// put entry in global memory
	g_scan[gid] = s_data[tid];

	// have last thread put its entry in the block sums to have the total sum of the block
	if(tid == blockDim.x - 1) g_blockSums[blockIdx.x] = s_data[tid];
}

// ===================================================================================================================
// SCAN_BLOCKSUMS()
// -------------------------------------------------------------------------------------------------------------------
// The second step in performing a scan in order to help find the scatter addresses and perform a radix sort. 
//
// This step performs an inclusive scan of the block sums with a single block of threads in order to prepare for 
// the next and final step in the multi-block scan.
// ===================================================================================================================
__global__ void scan_blocksums(unsigned int* g_blockSums, unsigned int n)
{
	// make sure we don't walk off the edge
	unsigned int tid = threadIdx.x;
	if(tid >= n) return;

	// declare external shared memory
	extern __shared__ unsigned int s_data[];
	
	// put block sum value in shared array and sync threads
	s_data[tid] = g_blockSums[tid];
	__syncthreads();

	// perform inclusive scan
	unsigned int value = 0;
	for(unsigned int offset = 1; offset < n; offset *= 2) 
	{
		if(tid >= offset)
		{
		  __syncthreads();
		  value = s_data[tid - offset];
		  __syncthreads();
		  s_data[tid] += value;
		}
	}

	// put entry in global memory
	g_blockSums[tid] = s_data[tid];
}

// ===================================================================================================================
// SCAN_ALL()
// -------------------------------------------------------------------------------------------------------------------
// The third and final step in performing a scan in order to help find the scatter addresses and perform a radix sort.
//
// This step adds all of the scanned block sums to all of the segments (except for the first segment) in order to 
// make the segmented inclusive scan an array-wide inclusive scan.
// ===================================================================================================================
__global__ void scan_all(unsigned int* g_scan, unsigned int* g_blockSums, size_t n)
{
	// make sure we don't walk off the array
	unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;
	if(gid >= n) return;

	// update segments with block sum values 
	if(blockIdx.x > 0) g_scan[gid] += g_blockSums[blockIdx.x - 1];
}

// ===================================================================================================================
// SCATTER()
// -------------------------------------------------------------------------------------------------------------------
// Uses the offsets to scatter the input into their correct positions. Note that the scanned entries are inclusive 
// due to the approach of the multi-block scan, and so we have to modify the scatter addresses slightly in order to
// accomodate for that.
// ===================================================================================================================
__global__ void scatter( unsigned int* g_inputVals, 
						 unsigned int* g_inputPos,
						 unsigned int* g_outputVals,
						 unsigned int* g_outputPos,
						 unsigned int* g_scan, 
						 unsigned int bit,
						 const size_t n )
{
	unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;
	if(gid >= n) return;

	// retrive the number of 1s (or trues) before this value and the total number of 1s (or trues)
	unsigned int trueBefore = g_scan[gid];
	unsigned int trueTotal = g_scan[n - 1]; 

	// calculate scatter address based on predicate and rearrange values and positions
	unsigned int sid = predicate(g_inputVals[gid], bit) ? (trueBefore-1 + n - trueTotal) : (gid - trueBefore);
	g_outputVals[sid] = g_inputVals[gid];
	g_outputPos[sid] = g_inputPos[gid];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// YOUR_SORT()
// -------------------------------------------------------------------------------------------------------------------
// Performs a radix sort on the input values (and modifies the corresponding indices of the input positions) on CUDA
// kernels. This particular radix sort only examines one bit at a time.
// ===================================================================================================================
void your_sort( unsigned int* const d_inputVals, 
				unsigned int* const d_inputPos, 
				unsigned int* const d_outputVals,
				unsigned int* const d_outputPos, 
				const size_t numElems )
{
	// SET UP
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// kernel threads and blocks (256 resulted in the best runtime)
	const unsigned int NUM_THREADS = 256;
    const unsigned int NUM_BLOCKS = (int)ceil( (float)numElems/(float)NUM_THREADS ) + 1;

    // scan shared arrays information and allocation
	unsigned int* d_scan, * d_blockSums;
	const unsigned int SCAN_SHARED_SIZE = NUM_THREADS;
	const size_t SCAN_ARR_SIZE = numElems * sizeof(unsigned int);
    const size_t SCAN_SHARED_ARR_SIZE = SCAN_SHARED_SIZE * sizeof(unsigned int);
    const size_t PRESCAN_SHARED_ARR_SIZE = NUM_BLOCKS * sizeof(unsigned int);
    checkCudaErrors(cudaMalloc(&d_scan, SCAN_ARR_SIZE));
    checkCudaErrors(cudaMalloc(&d_blockSums, PRESCAN_SHARED_ARR_SIZE));

    // ensure passed pointers still point to correct location when we're swapping
    unsigned int* d_inputValsAlias = d_inputVals; 
    unsigned int* d_outputValsAlias = d_outputVals;
    unsigned int* d_inputPosAlias = d_inputPos;
    unsigned int* d_outputPosAlias = d_outputPos;
    
    // RADIX SORT
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    for(unsigned int bit = 0; bit < MAX_BITS; bit++)
    {
    	// do a segmented inclusive scan (where a segment is a block) and put the sum of each block in blockSums
    	prescan <<< NUM_BLOCKS, NUM_THREADS, SCAN_SHARED_ARR_SIZE >>> (d_inputValsAlias, d_scan, d_blockSums, bit, numElems);

    	// do an inclusive scan on one block of threads on the block sums
    	scan_blocksums <<< 1, NUM_BLOCKS, PRESCAN_SHARED_ARR_SIZE >>> (d_blockSums, NUM_BLOCKS);

    	// finish off the scan by adding the block sums to the segmented scan
    	scan_all <<< NUM_BLOCKS, NUM_THREADS >>> (d_scan, d_blockSums, numElems);
        
    	// scatter
    	scatter <<< NUM_BLOCKS, NUM_THREADS >>> 
    			(d_inputValsAlias, d_inputPosAlias, d_outputValsAlias, d_outputPosAlias, d_scan, bit, numElems);
        
    	// exchange pointers to use the output as the input for the next iteration
    	swap(&d_inputValsAlias, &d_outputValsAlias);
        swap(&d_inputPosAlias, &d_outputPosAlias);
    }

    // copy input positions to output positions (they were just swapped when coming out of the loop)
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPosAlias, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    	
	// CLEAN UP
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	checkCudaErrors(cudaFree(d_scan));
	checkCudaErrors(cudaFree(d_blockSums));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////