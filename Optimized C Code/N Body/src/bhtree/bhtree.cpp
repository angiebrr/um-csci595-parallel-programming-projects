#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "common.h"
#include "bhtree.h"

// ===================================================================================================================
// bhtree.cpp
// -------------------------------------------------------------------------------------------------------------------
// Angela Gross
// CSCI-595: Parallel Programming
// Spring 2015
// -------------------------------------------------------------------------------------------------------------------
// Constructs a Barnes-Hut quadtree, which groups particles that are "sufficiently nearby" and recursively divides the set
// of particles into groups by using a quadtree. It can apply the net forces to N particles in the tree, and does so
// more efficiently than brute-force methods by using the center of mass of "far quadrants" so we don't have to look
// at all N particles when calculating the net force.
//
// The psuedocode provided by this website was used for the insertion and applying forces methods:
// http://www.cs.princeton.edu/courses/archive/fall04/cos126/assignments/barnes-hut.html
// ===================================================================================================================

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double BHTree::theta;

// ROOT CONSTRUCTOR
BHTree::BHTree(double width, double height, double theta)
{
	this->NW = NULL;
	this->NE = NULL;
	this->SW = NULL;
	this->SE = NULL;
	this->particle = NULL;
	this->parent = NULL;
	this->x = 0.0;
	this->y = 0.0;
	this->width = width;
	this->height = height;
	this->nMass = 0.0;
	this->xCOM = 0.0;
	this->yCOM = 0.0;
	this->subWidth = width/2;
	this->subHeight = height/2;
	this->xMid = x + this->subWidth;
	this->yMid = y + this->subHeight;
	this->leaf = true;
	this->theta = theta;
}

// CHILD NODE CONSTRUCTOR
BHTree::BHTree(BHTree* parent, double x, double y, double width, double height)
{
	this->NW = NULL;
	this->NE = NULL;
	this->SW = NULL;
	this->SE = NULL;
	this->particle = NULL;
	this->parent = parent;
	this->x = x;
	this->y = y;
	this->width = width;
	this->height = height;
	this->nMass = 0.0;
	this->xCOM = 0.0;
	this->yCOM = 0.0;
	this->subWidth = width/2;
	this->subHeight = height/2;
	this->xMid = x + this->subWidth;
	this->yMid = y + this->subHeight;
	this->leaf = true;
}
	
// DESTRUCTOR
BHTree::~BHTree() { delete NW; delete NE; delete SW; delete SE; }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// BUILDTREE()
// -------------------------------------------------------------------------------------------------------------------
// Build the Barnes-Hut quadtree with an array of particles.
// ===================================================================================================================
void BHTree::buildTree(int n, particle_t* particles)
{
	// Insert particles
	for(int i = 0; i < n; i++) 
		insert(&particles[i]);
}

// ===================================================================================================================
// RESETTREE()
// -------------------------------------------------------------------------------------------------------------------
// Helps reset the tree without releasing all of the memory that the tree resides in.
// ===================================================================================================================
void BHTree::resetTree()
{
	this->~BHTree();
	this->NW = NULL;
	this->NE = NULL;
	this->SW = NULL;
	this->SE = NULL;
	this->particle = NULL;
	this->parent = NULL;
	this->x = 0.0;
	this->y = 0.0;
	this->width = width;
	this->height = height;
	this->nMass = 0.0;
	this->xCOM = 0.0;
	this->yCOM = 0.0;
	this->subWidth = width/2;
	this->subHeight = height/2;
	this->xMid = x + this->subWidth;
	this->yMid = y + this->subHeight;
	this->leaf = true;
}

// ===================================================================================================================
// INSERT()
// -------------------------------------------------------------------------------------------------------------------
// Helps construct the tree by placing the particles in their proper quadrants according to their x and y positions.
// After inserting, the center of mass is calculated for each quadrant and particle.
// ===================================================================================================================
void BHTree::insert(particle_t* newParticle)
{
	// If it's not a leaf, then recursively insert the particle into the proper quadrant.
	if(!this->leaf)
	{
		// Top left (in NW) is (0, 0)
		bool isWest = newParticle->x < this->xMid;
		bool isNorth = newParticle->y < this->yMid;
		
		if(isWest && isNorth)
			this->NW->insert(newParticle);
		else if(!isWest && isNorth)
			this->NE->insert(newParticle);
		else if(isWest && !isNorth)
			this->SW->insert(newParticle);
		else if(!isWest && !isNorth)
			this->SE->insert(newParticle);
		
		// Denominator for the COM coordinates (the total mass)
		this->nMass = this->NW->nMass + this->NE->nMass + this->SW->nMass + this->SE->nMass;
		
		// Calculation for x,y COM coordinates
		this->xCOM = 	this->NW->nMass*this->NW->xCOM + 
						this->NE->nMass*this->NE->xCOM +
						this->SW->nMass*this->SW->xCOM +
						this->SE->nMass*this->SE->xCOM;
		this->xCOM = this->xCOM / this->nMass; 
		this->yCOM = 	this->NW->nMass*this->NW->yCOM + 
						this->NE->nMass*this->NE->yCOM +
						this->SW->nMass*this->SW->yCOM +
						this->SE->nMass*this->SE->yCOM;
		this->yCOM = this->yCOM / this->nMass; 
	}
	// If it's a leaf, then we have to subdivide if there's already a particle there
	else
	{
		// If this node doesn't have a particle, then assign it one and calculate its mass and COM coordinates.
		if(this->particle == NULL)
		{
			this->particle = newParticle;
			this->nMass = mass;
			this->xCOM = this->particle->x;
			this->yCOM = this->particle->y;
		}
		// Subdivide and move around since a particle already lives here
		else
		{
			// It's now an internal node
			this->leaf = false;
			
			// Subdivide the tree
			this->NW = new BHTree(this, this->x,    this->y,    this->subWidth, this->subHeight);
			this->NE = new BHTree(this, this->xMid, this->y,    this->subWidth, this->subHeight);
			this->SW = new BHTree(this, this->x,    this->yMid, this->subWidth, this->subHeight);
			this->SE = new BHTree(this, this->xMid, this->yMid, this->subWidth, this->subHeight);
			
			// Since there was already a particle here, we have to re-insert the node particle and the given particle
			insert(this->particle);
			insert(newParticle);
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// ===================================================================================================================
// APPLYFORCE()
// -------------------------------------------------------------------------------------------------------------------
// Applies the net force from all quadrant(s) that are within tolerance onto a particle. 
// ===================================================================================================================
void BHTree::applyForce(particle_t* aParticle, double* dmin, double* davg, int* navg)
{
	// If it isn't a leaf, then we check if the distance of the center of mass is within a tolerance.
	if(!this->leaf)
	{	
		// Calculate the distance to the center of mass and load variables for calculating the force
		double dx = this->xCOM - aParticle->x;
		double dy = this->yCOM - aParticle->y;
		double r2 = dx * dx + dy * dy;
		double r = sqrt(r2);
		double fMass = this->nMass;

		// If the distance is within a tolerance, then apply the force by treating the whole quadrant as one mass
		if(this->width / r < BHTree::theta)
		{
			calcForce(aParticle, &dx, &dy, &r2, &r, &fMass, dmin, davg, navg);
		}
		// Otherwise, we just recurse on each sub-quadrant to continue applying the force
		else
		{
			this->NW->applyForce(aParticle, dmin, davg, navg);
			this->NE->applyForce(aParticle, dmin, davg, navg);
			this->SW->applyForce(aParticle, dmin, davg, navg);
			this->SE->applyForce(aParticle, dmin, davg, navg);
		}
	}
	// If it's a leaf node and the particle is non-null and isn't equal to the particle in question, then apply 
	// the force from the single particle.
	else if(this->leaf && this->particle != NULL && this->particle != aParticle)
	{
		// Calculate the distance and load variables for calculating the force
		double dx = this->particle->x - aParticle->x;
		double dy = this->particle->y - aParticle->y;
		double r2 = dx * dx + dy * dy;
		double r = sqrt(r2);
		double fMass = mass;
		
		calcForce(aParticle, &dx, &dy, &r2, &r, &fMass, dmin, davg, navg);
	}
}

// ===================================================================================================================
// CALCFORCE()
// -------------------------------------------------------------------------------------------------------------------
// Uses the force calculations from common.cpp to apply the force 
// ===================================================================================================================
void BHTree::calcForce(particle_t* aParticle, double* dx, double* dy, double* r2, double* r, double* fMass, double* dmin, double* davg, int* navg)
{
	// Don't bother if it's over the cutoff 
	if(*r2 > cutoff*cutoff)
		return;
	
	// Update minimum distance between particles
	if (*r2/(cutoff*cutoff) < *dmin * (*dmin))
		*dmin = (*r)/cutoff;
	
	(*davg) += (*r)/cutoff;
	(*navg)++;
		
	*r2 = fmax(*r2, min_r*min_r);
	*r = fmax(*r, min_r); // to avoid another sqrt 
	
	// Very simple short-range repulsive force
	double coef = (1 - cutoff / *r) / ((*r2) * (*fMass));
	aParticle->ax += coef * (*dx);
	aParticle->ay += coef * (*dy);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////