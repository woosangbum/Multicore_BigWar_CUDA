#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include "Common/BigWar.h"
#include "Common/DS_timer.h"
#include "Grader.h"
#include "Team.h"
#include <omp.h>
#include <vector>
#include <algorithm>

#define LOOP_I(_loop) for(int i=0; i < _loop; i++)

#define ARMY_DIMENSION 3
#define NUM_RESULTS 100
#define LAST_PAIR (NUM_RESULTS-1)
#define BLOCK_SIZE 256
#define NUM_TEAM 2

typedef unsigned int UINT;
typedef float POS_TYPE;

struct Army2 {
	UINT ID;
	POS_TYPE pos[ARMY_DIMENSION];
};

struct Team2 {
	UINT numArmies = 100;
	Army2 *armies;
};

struct Pair2 {
	UINT A, B;
	POS_TYPE dist;
};

bool kernelCall(Army2* djojoA, Army2* dallianceA, Pair2* _result, unsigned int Njojo, unsigned int Nalliance,
	dim3 _griDim, dim3 _blockDim , Pair2*_resultF);
