#ifndef BHTREE_H
#define BHTREE_H

// ===================================================================================================================
// bhtree.h
// -------------------------------------------------------------------------------------------------------------------
// Angela Gross
// CSCI-595: Parallel Programming
// Spring 2015
// -------------------------------------------------------------------------------------------------------------------
// Header file to create a Barnes-Hut quadtree.
// ===================================================================================================================

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class BHTree
{
  private:
	BHTree* NW, * NE, * SW, * SE, * parent;
    particle_t* particle;
    double nMass;
    double xCOM, yCOM;
	double x, y;  
    double xMid, yMid;
    double width, height;
	double subWidth, subHeight;
    bool leaf;
	BHTree(BHTree* parent, double x, double y, double width, double height);
	void insert(particle_t* newParticle);
	void calcForce(particle_t* aParticle, double* dx, double* dy, double* r2, double* r, double* fMass, double* dmin, double* davg, int* navg);
	
  public:
	static double theta;
	BHTree(double width, double height, double theta);
    ~BHTree();
    void buildTree(int n, particle_t* particles);
	void resetTree();
    void applyForce(particle_t* aParticle, double* dmin, double* davg, int* navg);
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // BHTREE_H