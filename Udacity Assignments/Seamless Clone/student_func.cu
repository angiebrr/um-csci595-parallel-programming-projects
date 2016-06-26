
#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"

// ===================================================================================================================
// student_func.cu
// -------------------------------------------------------------------------------------------------------------------
// Angela Gross
// CSCI-595: Parallel Programming
// Homework 6
// Spring 2015
// -------------------------------------------------------------------------------------------------------------------
// The goal for this assignment is to take one image (the source) and paste it into another image (the destination) 
// attempting to match the two images so that the pasting is non-obvious. This is known as a "seamless clone".
//
// The basic ideas are as follows:
//  1) Figure out the interior and border of the source image
//  2) Use the values of the border pixels in the destination image  as boundary conditions for solving a Poisson 
//     equation that tells us how to blend the images.
//
// No pixels from the destination except pixels on the border are used to compute the match.
// ===================================================================================================================

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define ITERATIONS 800

// GLOBAL ATTRIBUTES
int* d_interior, * d_border;
float* d_redSource, * d_greenSource, * d_blueSource;
float* d_redDest, * d_greenDest, * d_blueDest;
float* d_redGuess1, * d_redGuess2, * d_greenGuess1, * d_greenGuess2, * d_blueGuess1, * d_blueGuess2;
uchar4* d_sourceImg, * d_destImg, * d_blendedImg;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ bool is_masked(uchar4 pixel) { return (pixel.x + pixel.y + pixel.z < 3 * 255); }

// ===================================================================================================================
// COMPUTE_BORDER_REGIONS()
// -------------------------------------------------------------------------------------------------------------------
// Instead of creating another array for the mask, we simply ask whether or not it is "masked" when computing the
// interior and border regions of the mask. An interior pixel has all 4 neighbors also inside the mask. A border pixel 
// is in the mask itself, but has at least one neighbor that isn't.
// ===================================================================================================================
__global__ void compute_border_regions(const size_t numRows, 
                                      const size_t numCols, 
                                      int* g_interior, 
                                      int* g_border,
                                      const uchar4* const g_sourceImg)
{
  // retrieve the thread's assigned pixel and make sure the pixel isn't outside of the image
  const int2 xy_2D = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if((xy_2D.x >= numCols) || (xy_2D.y >= numRows)) return;
  int xy_1D = (xy_2D.x) + numCols * (xy_2D.y);

  // memset for interior and border masks
  g_interior[xy_1D] = 0;
  g_border[xy_1D] = 0;
  
  // iterate through the masked pixels to determine whether it is a pixel on the border or the interior border
  if(is_masked(g_sourceImg[xy_1D]))
  {
    int masked = 0, neighbors = 0;
    
    // check to see (1) how many neighbors we have and (2) how many neighbors are masked
    // RIGHT NEIGHBOR
    if( (xy_2D.x + 1 < numCols) && (xy_2D.y < numRows) )
    {
      neighbors++;
      if(is_masked(g_sourceImg[(xy_2D.x + 1) + numCols * (xy_2D.y)])) masked++;
    }
    // LEFT NEIGHBOR
    if( (xy_2D.x - 1 < numCols) && (xy_2D.y < numRows) )
    {
      neighbors++;
      if(is_masked(g_sourceImg[(xy_2D.x - 1) + numCols * (xy_2D.y)])) masked++;
    }
    // TOP NEIGHBOR
    if( (xy_2D.x < numCols) && (xy_2D.y + 1 < numRows) )
    {
      neighbors++;
      if(is_masked(g_sourceImg[(xy_2D.x) + numCols * (xy_2D.y + 1)])) masked++;
    }
    // BOTTOM NEIGHBOR
    if( (xy_2D.x < numCols) && (xy_2D.y - 1 < numRows) )
    {
      neighbors++;
      if(is_masked(g_sourceImg[(xy_2D.x) + numCols * (xy_2D.y - 1)])) masked++;
    }

    // if all neighbors are masked, then it's an interior pixel
    if(masked == neighbors) g_interior[xy_1D] = 1;
    // if some neighbors are masked but not all, then it's a border pixel
    else if(masked > 0) g_border[xy_1D] = 1;
  }
}

// ===================================================================================================================
// SEPARATE_CHANNELS()
// -------------------------------------------------------------------------------------------------------------------
// A CUDA kernel that takes in an image represented as a uchar4 and splits it into three images consisting of only one 
// color channel each.
// ===================================================================================================================
__global__ void separate_channels(const size_t numRows, 
                                const size_t numCols, 
                                const uchar4* const inputImageRGBA, 
                                float* const redChannel, 
                                float* const greenChannel, 
                                float* const blueChannel)
{
  // retrieve the thread's assigned pixel and make sure the pixel isn't outside of the image
  const int2 xy_2D = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if((xy_2D.x >= numCols) || (xy_2D.y >= numRows)) return;
  int xy_1D = (xy_2D.x) + numCols * (xy_2D.y);

  // separate channels
  redChannel[xy_1D] = (float)inputImageRGBA[xy_1D].x;
  greenChannel[xy_1D] = (float)inputImageRGBA[xy_1D].y;
  blueChannel[xy_1D] = (float)inputImageRGBA[xy_1D].z;
}

// ===================================================================================================================
// RECOMBINE_CHANNELS()
// -------------------------------------------------------------------------------------------------------------------
// A CUDA kernel that takes in three color channels and recombines them into one image.  The alpha channel is set to 
// 255 to represent that this image has no transparency.
// ===================================================================================================================
__global__ void recombine_channels(const size_t numRows, 
                                  const size_t numCols, 
                                  uchar4* const outputImageRGBA,
                                  float* const redChannel, 
                                  float* const greenChannel,
                                  float* const blueChannel)
{
  // retrieve the thread's assigned pixel and make sure the pixel isn't outside of the image
  const int2 xy_2D = make_int2( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if (xy_2D.x >= numCols || xy_2D.y >= numRows) return;
  const int xy_1D = (xy_2D.x) + numCols * (xy_2D.y);

  // get and combine channels
  outputImageRGBA[xy_1D].x = (unsigned char)redChannel[xy_1D];
  outputImageRGBA[xy_1D].y = (unsigned char)greenChannel[xy_1D];
  outputImageRGBA[xy_1D].z = (unsigned char)blueChannel[xy_1D];
}

// ===================================================================================================================
// JACOBI_ITERATION()
// -------------------------------------------------------------------------------------------------------------------
// Our initial guess is going to be the source image itself.  This is a pretty good guess for what the blended image
// will look like and it means that we won't have to do as many iterations compared to if we had started far from the
// final solution.
//
// The following steps will be followed for one Jacobi Iteration:
//
//  1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
//      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
//          else if the neighbor in on the border then += DestinationImg[neighbor]
//      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)
//
//  2) Calculate the new pixel value:
//      float newVal= (Sum1 + Sum2) / 4.f 
//      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]
// ===================================================================================================================
__global__ void jacobi_iteration(const size_t numRows,
                                const size_t numCols,
                                float* g_guess1,
                                float* g_guess2,
                                float* g_sourceChannel,
                                float* g_destChannel,
                                const int* g_border,
                                const int* g_interior)
{
   // retrieve the thread's assigned pixel and make sure the pixel isn't outside of the image
  const int2 xy_2D = make_int2( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
  if (xy_2D.x >= numCols || xy_2D.y >= numRows) return;
  const int xy_1D = (xy_2D.x) + numCols * (xy_2D.y);

  // for every pixel in the interior, compute two sums over the 4 neighboring pixels
  if(g_interior[xy_1D] == 1)
  {
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, numNeighbors = 0.0f;
    int neighbor = 0;
    
    // RIGHT NEIGHBOR SUMS
    if( (xy_2D.x + 1 < numCols) && (xy_2D.y < numRows) )
    {
      numNeighbors++;

      neighbor = (xy_2D.x + 1) + numCols * (xy_2D.y);

      if(g_interior[neighbor] == 1) sum1 += g_guess1[neighbor];
      else if(g_border[neighbor] == 1) sum2 += g_destChannel[neighbor];

      sum3 += (g_sourceChannel[xy_1D] - g_sourceChannel[neighbor]);
    }
    // LEFT NEIGHBOR SUMS
    if( (xy_2D.x - 1 < numCols) && (xy_2D.y < numRows) )
    {
      numNeighbors++;

      neighbor = (xy_2D.x - 1) + numCols * (xy_2D.y);

      if(g_interior[neighbor] == 1) sum1 += g_guess1[neighbor];
      else if(g_border[neighbor] == 1) sum2 += g_destChannel[neighbor];

      sum3 += (g_sourceChannel[xy_1D] - g_sourceChannel[neighbor]);
    }
    // TOP NEIGHBOR SUMS
    if( (xy_2D.x < numCols) && (xy_2D.y + 1 < numRows) )
    {
      numNeighbors++;

      neighbor = (xy_2D.x) + numCols * (xy_2D.y + 1);

      if(g_interior[neighbor] == 1) sum1 += g_guess1[neighbor];
      else if(g_border[neighbor] == 1) sum2 += g_destChannel[neighbor];

      sum3 += (g_sourceChannel[xy_1D] - g_sourceChannel[neighbor]);
    }
    // BOTTOM NEIGHBOR SUMS
    if( (xy_2D.x < numCols) && (xy_2D.y - 1 < numRows) )
    {
      numNeighbors++;

      neighbor = (xy_2D.x) + numCols * (xy_2D.y - 1);

      if(g_interior[neighbor] == 1) sum1 += g_guess1[neighbor];
      else if(g_border[neighbor] == 1) sum2 += g_destChannel[neighbor];

      sum3 += (g_sourceChannel[xy_1D] - g_sourceChannel[neighbor]);
    }

    // clamp our next guess to [0, 255]
    g_guess2[xy_1D] = min( 255.f, max(0.0, (sum1 +  sum2 + sum3)/numNeighbors ) );
  }
  else
  {
    g_guess2[xy_1D] = g_destChannel[xy_1D];
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// ALLOCATEMEMORYANDCOPYTOGPU()
// -------------------------------------------------------------------------------------------------------------------
// Allocate all of the memory for the GPU (there is quite a bit!) and copy over the source and destination images to 
// the GPU.
// ===================================================================================================================
void allocateMemoryAndCopyToGPU(size_t IMG_SIZE,
                                const uchar4* const h_sourceImg,
                                const uchar4* const h_destImg,
                                uchar4* const h_blendedImg)
{
  // allocate memory for the border and interior pixel masks
  checkCudaErrors(cudaMalloc(&d_border, sizeof(int) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_interior, sizeof(int) * IMG_SIZE));

  // allocate memory for the three different channels
  checkCudaErrors(cudaMalloc(&d_redSource, sizeof(float) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_greenSource, sizeof(float) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_blueSource, sizeof(float) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_redDest, sizeof(float) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_greenDest, sizeof(float) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_blueDest, sizeof(float) * IMG_SIZE));

  // allocate memory for the three channels we'll flip-flop between
  checkCudaErrors(cudaMalloc(&d_redGuess1, sizeof(float) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_redGuess2, sizeof(float) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_greenGuess1, sizeof(float) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_greenGuess2, sizeof(float) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_blueGuess1, sizeof(float) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_blueGuess2, sizeof(float) * IMG_SIZE));

  // allocate memory for the source/dest images and the blended image
  checkCudaErrors(cudaMalloc(&d_sourceImg,  sizeof(uchar4) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_destImg,  sizeof(uchar4) * IMG_SIZE));
  checkCudaErrors(cudaMalloc(&d_blendedImg,  sizeof(uchar4) * IMG_SIZE));

  // copy the source and dest image over to the GPU
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg,  sizeof(uchar4) * IMG_SIZE, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_destImg, h_destImg,  sizeof(uchar4) * IMG_SIZE, cudaMemcpyHostToDevice));
}

// ===================================================================================================================
// COPYTOCPUANDDEALLOCATEMEMORY()
// -------------------------------------------------------------------------------------------------------------------
// Copy final output to CPU and deallocate everything else.
// ===================================================================================================================
void copyToCPUAndDeallocateMemory(size_t IMG_SIZE, uchar4* const h_blendedImg)
{
  // copy final output to host memory and deallocate
  checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * IMG_SIZE, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_blendedImg));

  // free allocated memory
  checkCudaErrors(cudaFree(d_border));
  checkCudaErrors(cudaFree(d_interior));
  checkCudaErrors(cudaFree(d_redSource));
  checkCudaErrors(cudaFree(d_greenSource));
  checkCudaErrors(cudaFree(d_blueSource));
  checkCudaErrors(cudaFree(d_redDest));
  checkCudaErrors(cudaFree(d_greenDest));
  checkCudaErrors(cudaFree(d_blueDest));
  checkCudaErrors(cudaFree(d_redGuess1));
  checkCudaErrors(cudaFree(d_redGuess2));
  checkCudaErrors(cudaFree(d_greenGuess1));
  checkCudaErrors(cudaFree(d_greenGuess2));
  checkCudaErrors(cudaFree(d_blueGuess1));
  checkCudaErrors(cudaFree(d_blueGuess2));
  checkCudaErrors(cudaFree(d_sourceImg));
  checkCudaErrors(cudaFree(d_destImg));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// YOUR_BLEND()
// -------------------------------------------------------------------------------------------------------------------
// Perform a "seamless clone" using the above CUDA kernels.
// ===================================================================================================================
void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
  // SET UP
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  const dim3 blockSize(32, 32, 1);
  const dim3 gridSize(numColsSource/blockSize.x + 1, numRowsSource/blockSize.y + 1, 1);
  size_t IMG_SIZE = numRowsSource * numColsSource;

  allocateMemoryAndCopyToGPU(IMG_SIZE, h_sourceImg, h_destImg, h_blendedImg);

  // SEAMLESS CLONE
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // computer interior and border regions
  compute_border_regions <<<gridSize, blockSize>>> (numRowsSource, numColsSource, d_interior, d_border, d_sourceImg);

  // separate channels in destination and source images
  separate_channels <<<gridSize, blockSize>>> (numRowsSource, numColsSource, d_destImg, d_redDest, d_greenDest, d_blueDest);
  separate_channels <<<gridSize, blockSize>>> (numRowsSource, numColsSource, d_sourceImg, d_redSource, d_greenSource, d_blueSource);

  // copy source channels as initial guess
  checkCudaErrors(cudaMemcpy(d_redGuess1, d_redSource, IMG_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_greenGuess1, d_greenSource, IMG_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_blueGuess1, d_blueSource, IMG_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));

  // perform jacobi iteration on all three channels
  for(int i = 0; i < ITERATIONS; i++)
  {
    // RED CHANNEL ITERATION AND SWAP
    jacobi_iteration <<<gridSize, blockSize>>> (numRowsSource, numColsSource, d_redGuess1, d_redGuess2, d_redSource, 
                                                d_redDest, d_border, d_interior);
    std::swap(d_redGuess1, d_redGuess2);

    // GREEN CHANNEL ITERATION AND SWAP
    jacobi_iteration <<<gridSize, blockSize>>> (numRowsSource, numColsSource, d_greenGuess1, d_greenGuess2, d_greenSource, 
                                                d_greenDest, d_border, d_interior);
    std::swap(d_greenGuess1, d_greenGuess2);

    // BLUE CHANNEL ITERATION AND SWAP
    jacobi_iteration <<<gridSize, blockSize>>> (numRowsSource, numColsSource, d_blueGuess1, d_blueGuess2, d_blueSource, 
                                                d_blueDest, d_border, d_interior);
    std::swap(d_blueGuess1, d_blueGuess2);
  }

  // recombine channels in final blended image
  recombine_channels <<<gridSize, blockSize>>> (numRowsSource, numColsSource, d_blendedImg, d_redGuess1, d_greenGuess1, d_blueGuess1);

  // CLEANUP
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  copyToCPUAndDeallocateMemory(IMG_SIZE, h_blendedImg);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////