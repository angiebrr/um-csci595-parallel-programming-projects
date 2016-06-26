#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <list>
#include <math.h>
#include "common.h"

// ===================================================================================================================
// mpi.cpp
// -------------------------------------------------------------------------------------------------------------------
// Angela Gross
// CSCI-595: Parallel Programming
// Spring 2015
// -------------------------------------------------------------------------------------------------------------------
// Uses a simple method of binning to efficiently apply forces to an N-body simulation, or the method that divides the
// world into bins uses only the particles in surrounding bins to apply forces to each particle in a given bin.
//
// This basically does everything that OpenMP does, but every processor has its own copy of bins. 
// ===================================================================================================================

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// MACRO AND VARIABLES FOR BINS (0,0 is bottom left corner)
#define B(x, y) bins[(y) * binsN + (x)]
std::vector< std::list<particle_t*> > bins, localBins;
int binsN, binsSize;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// METHOD PROTOTYPES FOR BINNING
void buildBins(int n, particle_t* particles);
void rebuildBins(int n, particle_t* particles);
void calcForce(particle_t* currParticle, double* dmin, double* davg, int* navg);
void applyBinForce(int x, int y, particle_t* currParticle, double* dmin, double* davg, int* navg);
void applyOwnBinForce(int x, int y, particle_t* currParticle, double* dmin, double* davg, int* navg);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// MAIN METHOD
int main(int argc, char** argv)
{    
    int navg, nabsavg = 0;
    double dmin, absmin = 1.0, davg, absavg = 0.0;
    double rdavg,rdmin;
    int rnavg; 
	
	// set up MPI
	int n_proc, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    // process command line parameters
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char* savename = read_string( argc, argv, "-o", NULL );
    char* sumname = read_string( argc, argv, "-s", NULL );
    
    FILE* fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE* fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// initialize particles
	particle_t *particles = (particle_t*) malloc(n * sizeof(particle_t));

	// make MPI particle datatype
	MPI_Datatype PARTICLE;
	MPI_Type_contiguous(6, MPI_DOUBLE, &PARTICLE);
	MPI_Type_commit(&PARTICLE);

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// set up the data partitioning across processors
	int particle_per_proc = (n + n_proc - 1) / n_proc;
	int* partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
	for(int i = 0; i < n_proc + 1; i++)
		partition_offsets[i] = min( i * particle_per_proc, n );

	int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
	for(int i = 0; i < n_proc; i++)
		partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];

	// allocate storage for local partition
	int nlocal = partition_sizes[rank];
	particle_t* local = (particle_t*) malloc(nlocal * sizeof(particle_t));

	// initialize and distribute the particles
	set_size(n);
	if(rank == 0)
		init_particles(n, particles);
	MPI_Scatterv(particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD);
	
	// need to create bins that fit into the width of the world and are the width of the cutoff.
	double worldWidth = get_size();
	double binWidth = cutoff;
	binsN = (int)ceil((worldWidth / binWidth));
	binsSize = binsN * binsN;

	// create and build a bins vector that carries bins of particles (pointers to particles, that is)
	bins.resize(binsSize);

	// simulate a number of time steps
	double simulation_time = read_timer();
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	for(int step = 0; step < NSTEPS; step++)
	{
		navg = 0;
		dmin = 1.0;
		davg = 0.0;

		// collect all global data locally (not good idea to do)
		MPI_Allgatherv(local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD);
		
		// only build the bins on the first step when we have all of the particles
		if(step == 0)
			buildBins(n, particles);
			
		// save current step if necessary (slightly different semantics than in other codes)
		if(find_option( argc, argv, "-no" ) == -1)
			if( fsave && (step%SAVEFREQ) == 0 )
				save(fsave, n, particles);

		// compute local forces
		for(int p = 0; p < nlocal; p++)
			calcForce(&local[p], &dmin, &davg, &navg);
		
		// move particles
		for(int p = 0; p < nlocal; p++)
			move(local[p]);
		
		// rebuild the bins (particles may have moved to other bins)
		rebuildBins(n, particles);

		if( find_option( argc, argv, "-no" ) == -1 )
		{
			MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

			if (rank == 0)
			{
				// computing statistical data
				if (rnavg) 
				{
					absavg +=  rdavg/rnavg;
					nabsavg++;
				}
				if (rdmin < absmin) absmin = rdmin;
			}
		}
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	simulation_time = read_timer() - simulation_time;

	if (rank == 0) 
	{  
		printf( "n = %d, simulation time = %g seconds", n, simulation_time);

		if( find_option( argc, argv, "-no" ) == -1 )
		{
			if (nabsavg) absavg /= nabsavg;
			// 
			//  -the minimum distance absmin between 2 particles during the run of the simulation
			//  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
			//  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
			//
			//  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
			//
			printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
			if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
			if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
		}
		  
		  printf("\n");     
			
		  // Printing summary data
		  if(fsum)
			fprintf(fsum,"%d %d %g\n", n, n_proc, simulation_time);
    }
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
  
    // releasing resources
    if (fsum)
        fclose(fsum);
    free(partition_offsets);
    free(partition_sizes);
    free(local);
    free(particles);
    if(fsave)
        fclose(fsave);
    
    MPI_Finalize();
    
    return 0;
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// BUILDBINS()
// -------------------------------------------------------------------------------------------------------------------
// Populates the bins with particles. Since the particles already contain their exact coordinates, we only need to 
// worry about which bin each particle belongs to. We can find what bin it's in by multiplying by 100 since we found 
// the number of bins by dividing the width of the world by 0.01, or multiplying by 100.
// ===================================================================================================================
void buildBins(int n, particle_t* particles)
{
	for(int p = 0; p < n; p++)
	{
		int x = (int)(particles[p].x  * 100);
		int y = (int)(particles[p].y  * 100);
		B(x,y).push_back(&particles[p]);
	}
}

// ===================================================================================================================
// REBUILDBINS()
// -------------------------------------------------------------------------------------------------------------------
// Iterate over the bins and particles in each bin and check if any particles have moved into different bins. If they
// have, then remove them from the bin and place them in the proper bin.
// ===================================================================================================================
void rebuildBins(int n, particle_t* particles)
{
	// iterate over bins and particles in each bin
	for(int y = 0; y < binsN; y++)
	{
		for(int x = 0; x < binsN; x++)
		{		
			std::list<particle_t*>::iterator iter = B(x, y).begin();
			while(iter != B(x,y).end())
			{
				// get particle and proper bin location
				particle_t* currParticle = (*iter);
				int particleX = (int)(currParticle->x  * 100);
				int particleY = (int)(currParticle->y  * 100);
				
				// remove and add into proper bin
				if(particleX != x || particleY != y)
				{
					B(particleX, particleY).push_back(currParticle);
					iter = B(x, y).erase(iter); // make sure we don't skip any entries
				}
				else
				{
					iter++;
				}
			}
		}
	}
}

// ===================================================================================================================
// CALCFORCE()
// -------------------------------------------------------------------------------------------------------------------
// Applies the force to a particle in the currently examined bin from its own bin and bins immediately surrounding it.
// ===================================================================================================================
void calcForce(particle_t* currParticle, double* dmin, double* davg, int* navg)
{
	int x = (int)(currParticle->x  * 100);
	int y = (int)(currParticle->y  * 100);
	currParticle->ax = currParticle->ay = 0;
	
	// apply force from particles in current bin
	applyOwnBinForce(x, y, currParticle, dmin, davg, navg);
	
	// apply force from particles in bins immediately surrounding the current bin
	if(x < binsN - 1) 
		applyBinForce(x + 1, y, currParticle, dmin, davg, navg); // RIGHT BIN: (if not in right-most column)
	if(y > 0) 
		applyBinForce(x, y - 1, currParticle, dmin, davg, navg); // BOTTOM BIN: (if not in the bottom row)
	if(x > 0)
		applyBinForce(x - 1, y, currParticle, dmin, davg, navg); // LEFT BIN: (if not in left-most column)
	if(y < binsN - 1)
		applyBinForce(x, y + 1, currParticle, dmin, davg, navg); // TOP BIN: (if not in the top row)
	if(x < binsN - 1 && y < binsN - 1)
		applyBinForce(x + 1, y + 1, currParticle, dmin, davg, navg); // TOP-RIGHT BIN: (if not in the top-right corner)
	if(x < binsN - 1 && y > 0)
		applyBinForce(x + 1, y - 1, currParticle, dmin, davg, navg); // BOTTOM-RIGHT BIN: (if not in the bottom-right corner)
	if(x > 0 && y > 0)
		applyBinForce(x - 1, y - 1, currParticle, dmin, davg, navg); // BOTTOM-LEFT BIN: (if not in the bottom-left corner)
	if(x > 0 && y < binsN - 1)
		applyBinForce(x - 1, y + 1, currParticle, dmin, davg, navg); // TOP-LEFT BIN: (if not in the top-left corner)
}

// ===================================================================================================================
// APPLYBINFORCE()
// -------------------------------------------------------------------------------------------------------------------
// Applies the force to all particles within the given bin.
// ===================================================================================================================
void applyBinForce(int x, int y, particle_t* currParticle, double* dmin, double* davg, int* navg)
{	
	for(std::list<particle_t*>::iterator iter = B(x, y).begin(); iter != B(x,y).end(); iter++)
		apply_force(*currParticle, **iter, dmin, davg, navg);
}

// ===================================================================================================================
// APPLYOWNBINFORCE()
// -------------------------------------------------------------------------------------------------------------------
// Applies the force to all particles within the given bin (but checks if the particle is yourself)
// ===================================================================================================================
void applyOwnBinForce(int x, int y, particle_t* currParticle, double* dmin, double* davg, int* navg)
{
	for(std::list<particle_t*>::iterator iter = B(x, y).begin(); iter != B(x,y).end(); iter++)
	{
		if( (*iter) != currParticle )
			apply_force(*currParticle, **iter, dmin, davg, navg);
	}
}

