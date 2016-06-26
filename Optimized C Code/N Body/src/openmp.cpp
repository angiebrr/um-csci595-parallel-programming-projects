#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <list>
#include "common.h"
#include "omp.h"

// ===================================================================================================================
// openmp.cpp
// -------------------------------------------------------------------------------------------------------------------
// Angela Gross
// CSCI-595: Parallel Programming
// Spring 2015
// -------------------------------------------------------------------------------------------------------------------
// Uses a simple method of binning to efficiently apply forces to an N-body simulation, or the method that divides the
// world into bins uses only the particles in surrounding bins to apply forces to each particle in a given bin.
//
// This implementation uses vectors of lists and mostly iterates over particles and not the bins themselves (this 
// avoids using expensive iterators). When it does have to iterate over the bins (i.e. when it is rebuilding all of the
// bins), much care is taken to make sure that we are iterating over contiguous memory.
//
// Unlike serial, all choices were made with efficiency in mind.
// ===================================================================================================================

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// MACRO AND VARIABLES FOR BINS (0,0 is bottom left corner)
#define B(x, y) bins[(y) * binsN + (x)]
std::vector< std::list<particle_t*> > bins;
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
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
    int navg, nabsavg = 0, numthreads; 
    double dmin, absmin = 1.0, davg, absavg = 0.0;
	
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char* savename = read_string( argc, argv, "-o", NULL );
    char* sumname = read_string( argc, argv, "-s", NULL );

    FILE* fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE* fsum = sumname ? fopen ( sumname, "a" ) : NULL;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////	

	// initialize particles
    particle_t* particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size(n);
    init_particles(n, particles);
	
    //  simulate a number of time steps
    double simulation_time = read_timer();
	
	// create bins that fit into the width of the world and are the width of the cutoff.
	double worldWidth = get_size();
	double binWidth = cutoff;
	binsN = (int)ceil((worldWidth / binWidth));
	binsSize = binsN * binsN;
	
	// create and build a bins vector that carries bins of particles (pointers to particles, that is)
	bins.resize(binsSize);
	buildBins(n, particles);
		
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	#pragma omp parallel private(dmin) shared(bins)
	{
		numthreads = omp_get_num_threads();
		
		for(int step = 0; step < 1000; step++)
		{
			navg = 0;
			davg = 0.0;
			dmin = 1.0;
			
			// compute forces by iterating particles
			#pragma omp for reduction (+:navg) reduction(+:davg)		
			for(int p = 0; p < n; p++)
				calcForce(&particles[p], &dmin, &davg, &navg);

			// move particles
			#pragma omp for
			for(int p = 0; p < n; p++) 
				move(particles[p]);
			
			// rebuild the bins (particles may have moved to other bins)
			rebuildBins(n, particles);

			if(find_option(argc, argv, "-no" ) == -1) 
			{
				// compute statistical data
				#pragma omp master
				if (navg) 
				{ 
					absavg += davg/navg;
					nabsavg++;
				}

				#pragma omp critical
				if (dmin < absmin) absmin = dmin; 

				// save if necessary
				#pragma omp master
				if( fsave && (step%SAVEFREQ) == 0 )
					save( fsave, n, particles );
			}
		}
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    simulation_time = read_timer() - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

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
    
    // printing summary data
    if(fsum)
        fprintf(fsum, "%d %d %g\n", n, numthreads, simulation_time);

    // clearing space
    if(fsum)
        fclose(fsum);

	// releasing resources
    free(particles);
	
    if(fsave)
        fclose(fsave);
    
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
	#pragma omp for
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////