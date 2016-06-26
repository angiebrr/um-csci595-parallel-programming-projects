/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3 -funroll-loops -ffast-math -msse3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <mmintrin.h> // SSE
#include <xmmintrin.h> // SSE
#include <pmmintrin.h> // SSE2
#include <emmintrin.h> // SSE3
#include <string.h> // memcpy and memset

// ===================================================================================================================
// dgemm-optimized.c
// -------------------------------------------------------------------------------------------------------------------
// Angela Gross
// CSCI-595: Parallel Programming
// Spring 2015
// -------------------------------------------------------------------------------------------------------------------
// Performs an optimized matrix-matrix multiply by using SSE2/SSE3 intrinsics and blocking data into the cache. 
// ===================================================================================================================

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// MACROS

#if !defined(MC)
#define MC 300
#define KC 180
#endif

#define min( i, j ) ( (i)<(j) ? (i): (j) )
#define A(i, j) A[j*lda + i]
#define B(i, j) B[j*ldb + i]
#define C(i, j) C[j*ldc + i]

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// GLOBAL VARIABLES / TYPEDEFS

const char* dgemm_desc = "Optimized, three-loop dgemm.";

typedef union
{
	__m128d v;
	double d[2];
} vreg_2d; 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// METHOD PROTOTYPES

static void dot_product_4x4(int K, double* A, int lda, double* B, int ldb, double* restrict C, int ldc);
static void pack_matrix_A(int K, double* A, int lda, double* packedA);
static void do_block(int m, int n, int k, double* A, double* B, double* restrict C);
static void square_matrix_multiply(int n, double* A, double* B, double* restrict C);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ==================================================================================================================
// SQUARE_DGEMM()
// ------------------------------------------------------------------------------------------------------------------
// This routine performs a dgemm operation C := C + A * B where A, B, and C are n-by-n matrices stored in 
// column-major format. On exit, A and B maintain their input values. 
// ==================================================================================================================
void square_dgemm (int n, double* A, double* B, double* restrict C)
{
	int nmod4 = n % 4;
	
	// If n isn't divisible by 4, then we move the array to a padded array that is divisible by 4
	if(nmod4 != 0)
	{
		// Get the next multiple of 4 and use that as n
		int n4 = n - nmod4 + 4;
		
		// Compute useful values for padding and initializing the array
		int pad = n4 - n;
		int arrPaddedSize = n4 * n4;
		int arrSize = n * n;
		size_t nBytes = sizeof(double) * n;
		
		double paddedA[arrPaddedSize] __attribute__((aligned(16))); 
		double paddedB[arrPaddedSize] __attribute__((aligned(16))); 
		double paddedC[arrPaddedSize] __attribute__((aligned(16))); 
		
		memset(paddedA, 0, sizeof(paddedA));
		memset(paddedB, 0, sizeof(paddedB));
		memset(paddedC, 0, sizeof(paddedC));
	
		// Due to matrix structure, we must put in padded zeroes (which are in an backwards-L shape) every n 
		// entries.
		for(int i = 0, padJump = 0; i < arrSize; i += n, padJump += n + pad)
		{
			memcpy(paddedA + padJump, A + i, nBytes); 
			memcpy(paddedB + padJump, B + i, nBytes); 
			memcpy(paddedC + padJump, C + i, nBytes);
		}
	
		// Multiply with new padded matrices
		square_matrix_multiply(n4, paddedA, paddedB, paddedC);
		
		// Copy back values (only C changed) into given memory (Have to skip padded zeroes every n entries)
		for(int i = 0, padJump = 0; i < arrSize; i += n, padJump += pad)
			memcpy(C + i, paddedC + i + padJump, nBytes);
	}
	else
	{
		square_matrix_multiply(n, A, B, C);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// HELPER METHODS FOR SQUARE_DGEMM

// ==================================================================================================================
// SQUARE_MATRIX_MULTIPLY()
// ------------------------------------------------------------------------------------------------------------------
// Performs a square dgemm operation in blocks.
// ==================================================================================================================
static void square_matrix_multiply(int n, double* A, double* B, double* restrict C)
{
	int Kn = 0, Mn = 0; // Used to help correct block dimensions for edge cases
	int lda = n, ldb = n, ldc = n; // Used in macros-> only different than n when packing
	
	for (int k = 0; k < n; k += KC)
	{
		Kn = min(KC, n - k);
		for (int i = 0; i < n; i += MC) 
		{
			Mn = min(MC, n - i);
			do_block( Mn, n, Kn, &A(i, k), &B(k, 0), &C(i, 0) );
		}
	}
}

// ==================================================================================================================
// DO_BLOCK()
// ------------------------------------------------------------------------------------------------------------------
// Performs an unrolled matrix multiply, but not before packing A into a contiguous block of memory.
// ==================================================================================================================
static void do_block(int m, int n, int k, double* A, double* B, double* restrict C)
{
	double packedA[m*k];
	int lda = n, ldb = n, ldc = n; // Used in macros-> only different than n when packing
	
	// For each jth column of B (unrolled by 4)
	for (int j = 0; j < n; j += 4)
	{	
		// For each ith row of A (unrolled by 4)
		for (int i = 0; i < m; i += 4) 
		{
			// Pack a contiguous block of A since we're using rows of A (not contiguous)
			if(j == 0) 
				pack_matrix_A( k, &A(i, 0), lda, &packedA[i*k] );
			
			// Update next 16 values of C(i, j)
			dot_product_4x4( k, &packedA[i*k], 4, &B(0, j), ldb, &C(i, j), ldc );
		}
	}
}

// ==================================================================================================================
// PACK_MATRIX_A()
// ------------------------------------------------------------------------------------------------------------------
// Puts a block of A in contiguous memory for quicker accesses.
// ==================================================================================================================
static void pack_matrix_A(int K, double* A, int lda, double* packedA)
{
	// Loop over the columns of A
	for(int j = 0; j < K; j++)
	{
		double* aij_ptr = &A(0, j);
		
		*packedA = *aij_ptr;
		*(packedA + 1) = *(aij_ptr + 1);
		*(packedA + 2) = *(aij_ptr + 2);
		*(packedA + 3) = *(aij_ptr + 3);
		
		packedA += 4;
	}
}

// ==================================================================================================================
// DOT_PRODUCT_4x4()
// ------------------------------------------------------------------------------------------------------------------
// Compute a 4 x 4 block of C with 4 dot products of rows from A and columns from B.
// Note that A, B, and C are pointers to A(i, 0), B(0, j), and C(i, j) 
// ==================================================================================================================
static void dot_product_4x4(int K, double* A, int lda, double* B, int ldb, double* restrict C, int ldc)
{	
	// Setting up SSE vector registers
	vreg_2d a0k_a1k, a2k_a3k;	// a0k, a1k, a2k, and a3k
	vreg_2d bk0, bk1, bk2, bk3;	// bk0, bk1, bk2, and bk3
	vreg_2d c00_c10, c01_c11, c02_c12, c03_c13; // Rows 0 and 1 of 4x4 C-block
	vreg_2d c20_c30, c21_c31, c22_c32, c23_c33; // Rows 2 and 3 of 4x4 C-block
	
	// Initializing 4x4 C SSE registers
	c00_c10.v = _mm_setzero_pd(); c01_c11.v = _mm_setzero_pd(); // Rows 0 and 1 of 4x4 C-block
	c02_c12.v = _mm_setzero_pd(); c03_c13.v = _mm_setzero_pd(); // Rows 0 and 1 of 4x4 C-block
	c20_c30.v = _mm_setzero_pd(); c21_c31.v = _mm_setzero_pd(); // Rows 2 and 3 of 4x4 C-block
	c22_c32.v = _mm_setzero_pd(); c23_c33.v = _mm_setzero_pd(); // Rows 2 and 3 of 4x4 C-block
	
	// Point to current elements in 4 columns of B
	double* bk0_ptr = &B(0, 0);
	double* bk1_ptr = &B(0, 1);
	double* bk2_ptr = &B(0, 2);
	double* bk3_ptr = &B(0, 3);
	
	// Update 16 values of C simultaneously
	for(int k = 0; k < K; k++)
	{
		// Load up SSE vector registers for A
		a0k_a1k.v = _mm_load_pd(A);
		a2k_a3k.v = _mm_load_pd(A + 2);
		A += 4;
		
		// Load up and duplicate SSE registers for B
		bk0.v = _mm_loaddup_pd( bk0_ptr++ );
		bk1.v = _mm_loaddup_pd( bk1_ptr++ );
		bk2.v = _mm_loaddup_pd( bk2_ptr++ );
		bk3.v = _mm_loaddup_pd( bk3_ptr++ );
		
		// Do first and second rows at once
		c00_c10.v += a0k_a1k.v * bk0.v;
		c01_c11.v += a0k_a1k.v * bk1.v; 
		c02_c12.v += a0k_a1k.v * bk2.v;
		c03_c13.v += a0k_a1k.v * bk3.v;
		
		// Do third and fourth rows at once
		c20_c30.v += a2k_a3k.v * bk0.v;
		c21_c31.v += a2k_a3k.v * bk1.v; 
		c22_c32.v += a2k_a3k.v * bk2.v;
		c23_c33.v += a2k_a3k.v * bk3.v;
	}
	
	// Update first row of C
	C(0, 0) += c00_c10.d[0]; C(0, 1) += c01_c11.d[0]; 		
	C(0, 2) += c02_c12.d[0]; C(0, 3) += c03_c13.d[0];
	
	// Update second row of C
	C(1, 0) += c00_c10.d[1]; C(1, 1) += c01_c11.d[1];
	C(1, 2) += c02_c12.d[1]; C(1, 3) += c03_c13.d[1];	
	
	// Update third row of C
	C(2, 0) += c20_c30.d[0]; C(2, 1) += c21_c31.d[0];
	C(2, 2) += c22_c32.d[0]; C(2, 3) += c23_c33.d[0];
	
	// Update fourth row of C
	C(3, 0) += c20_c30.d[1]; C(3, 1) += c21_c31.d[1];
	C(3, 2) += c22_c32.d[1]; C(3, 3) += c23_c33.d[1];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////