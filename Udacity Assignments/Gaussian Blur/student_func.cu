#include "utils.h"

// ===================================================================================================================
// student_func.cu
// -------------------------------------------------------------------------------------------------------------------
// Angela Gross
// CSCI-595: Parallel Programming
// Homework 2
// Spring 2015
// -------------------------------------------------------------------------------------------------------------------
// This blurs an image using CUDA and a square array of weights (see gaussian_blur() below for more details). 
//
// In order to optimize:
// - The channels weren't separated, (as suggested), and so only one call to a kernel was made
// - Using this URL as a guide (http://cuda-programming.blogspot.com/2013/01/what-is-constant-memory-in-cuda.html), 
//   the filter was put into constant memory since it isn't ever changed
// - Making sure that as much of the global memory is in local memory in the kernel
// - Removing synchronization and error checks since it's done in the autograder (main.cpp) after it calls this
//   method.
//
// Since the methods were prototyped in other source files used by the autograder, many functions have unnecessary 
// arguments or, like in the case of cleanup(), the function itself is unnecessary. (I wasn't able to actually change
// these files to match on the Udacity assignment submission)
// ===================================================================================================================

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// CONST FILTER INFORMATION
// filter size and width won't change since we're using the same filter
const int MAX_FILTER_SIZE = 81;
const int FILTER_WIDTH = 9;
__constant__ float const_filter[MAX_FILTER_SIZE];

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// GAUSSIAN_BLUR()
// -------------------------------------------------------------------------------------------------------------------
// A CUDA kernel that uses a square array of weight values that are overlayed and applied to the current pixel and its
// neighboring pixels to blur the pixel. To do this, the lined-up weights are multiplied by the corresponding pixels.
// ===================================================================================================================
__global__ void gaussian_blur(const uchar4* const inputChannel, uchar4* const outputChannel, int numRows, int numCols)
{
  // retrieve the thread's assigned pixel and make sure the pixel isn't outside of the image
  const int2 xy_2D = make_int2( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if((xy_2D.x >= numCols) || (xy_2D.y >= numRows)) return;

  // declare/initialize variables used / put global memory in local memory
  float3 result = make_float3(0.0f, 0.0f, 0.0f);
  float3 image_value = make_float3(0.0f, 0.0f, 0.0f);
  int2 image_idx = make_int2(0, 0);
  int index = 0;
  int local_rows = numRows;
  int local_cols = numCols;
  float filter_value = 0.0;

  // for every value in the filter around the pixel (x, y), we apply the filter to the pixel and its neighbors
  for (int filter_y = -FILTER_WIDTH/2; filter_y <= FILTER_WIDTH/2; ++filter_y) 
  {
    for (int filter_x = -FILTER_WIDTH/2; filter_x <= FILTER_WIDTH/2; ++filter_x) 
    {
      // in order to avoid "walking over the image", we find the global image position for this filter position and 
      // clamp to boundary of the image. 
      image_idx.y = min(max(xy_2D.y + filter_y, 0), numRows - 1);
      image_idx.x = min(max(xy_2D.x + filter_x, 0), numCols - 1);

      // retrieve image/filter pair for each color channel
      index = image_idx.y * local_cols + image_idx.x;
      image_value.x = (float)inputChannel[index].x;
      image_value.y = (float)inputChannel[index].y;
      image_value.z = (float)inputChannel[index].z;
      filter_value = const_filter[(filter_y + FILTER_WIDTH/2) * FILTER_WIDTH + filter_x + FILTER_WIDTH/2];

      // compute result for each color channel and filter value
      result.x += image_value.x * filter_value;
      result.y += image_value.y * filter_value;
      result.z += image_value.z * filter_value;
    }
  }

  outputChannel[(xy_2D.x) + local_cols * (xy_2D.y)] = make_uchar4(result.x, result.y, result.z, 255);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// ALLOCATEMEMORYANDCOPYTOGPU()
// -------------------------------------------------------------------------------------------------------------------
// Allocates and copies the weighted filter to constant GPU memory. Since the prototype for this function is used in
// the autograder, unnecessary arguments are included.
// ===================================================================================================================
void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage, const float* const h_filter, 
                                const size_t filter_width)
{
  // copy the filter on the host (h_filter) to constant memory (see )
  checkCudaErrors(cudaMemcpyToSymbol(const_filter, h_filter,  sizeof(float) * MAX_FILTER_SIZE));
}

// ===================================================================================================================
// YOUR_GAUSSIAN_BLUR()
// -------------------------------------------------------------------------------------------------------------------
// Sets up and launches the gaussian blur kernel. Since the prototype for this function is used in the autograder,
// unnecessary arguments are included.
// ===================================================================================================================
void your_gaussian_blur(const uchar4* const h_inputImageRGBA, uchar4* const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, unsigned char* d_greenBlurred,  unsigned char* d_blueBlurred,
                        const int filter_width)
{
  // set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(32, 32, 1);

  // compute correct grid size (i.e., number of blocks per kernel launch) from the image size and and block size.
  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1, 1);

  // blur image
  gaussian_blur<<<gridSize, blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numRows, numCols);

  // synchronization (and error checking) removed because it is done after it calls this method in the autograder
}

// ===================================================================================================================
// CLEANUP()
// -------------------------------------------------------------------------------------------------------------------
// Frees memory allocated via CUDA (Nothing to free with this implementation). We have to include it since it's
// used in the autograder.
// ===================================================================================================================
void cleanup() 
{
  // Since we're using constant memory and haven't allocated anything in this file, nothing needs to be freed.
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////