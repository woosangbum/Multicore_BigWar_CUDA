#include "kernelCall.h"
__device__ void SWAP(Pair2* arr, int a, int b)
{
    Pair2 temp;
    temp = arr[a];
    arr[a] = arr[b];
    arr[b] = temp;
}
__device__ void SORT(Pair2* arr, int m, int n)
{
    if (m < n)
    {
        int key = m;
        int i = m + 1;
        int j = n;

        while (i <= j)
        {
            while (i <= n && arr[i].dist >= arr[key].dist)
                i++;
            while (j > m && arr[j].dist <= arr[key].dist)
                j--;
            if (i > j)
                SWAP(arr, j, key);
            else
                SWAP(arr, i, j);
        }
        SORT(arr, m, j - 1);
        SORT(arr, j + 1, n);
    }
}

__device__ float dist(Army2* _A, Army2* _B) {
    float d = 0;
    for (int i = 0; i < ARMY_DIMENSION; i++) {
        d += (_A->pos[i] - _B->pos[i]) * (_A->pos[i] - _B->pos[i]);
    }
    return sqrt(d);
}

__global__ void addKernel(float* djojoA, float* dallianceA, float* _result, unsigned int Njojo, unsigned int Nalliance)
{	
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; // col

    struct Team2 jojo;
    struct Team2 alliance;
    Pair2 *results = (Pair2*)_result;
    Pair2 curPair;

    jojo.numArmies = Njojo;
    jojo.armies = (Army2*)djojoA;
    alliance.numArmies = Nalliance;
    alliance.armies = (Army2*)dallianceA;
    //if (ix > jojo.numArmies) return;
    //if (ix > 1000) return;

    for (int iB = 0; iB < alliance.numArmies; iB++) {
    //for (int iB = 0; iB<1000; iB++) {
		curPair.A = jojo.armies[ix].ID;
		curPair.B = alliance.armies[iB].ID;
		curPair.dist = dist(&jojo.armies[curPair.A], &alliance.armies[curPair.B]);
        //printf("%d\n", ix);
        
		if (results[LAST_PAIR].dist > curPair.dist) {
            //lock
			results[LAST_PAIR] = curPair;
			SORT(results, 0, NUM_RESULTS - 1);
            //lock
		}
	}
}


bool kernelCall(float* djojoA, float* dallianceA, float* _result, unsigned int Njojo, unsigned int Nalliance,
    dim3 _griDim, dim3 _blockDim) {

    addKernel << <_griDim, _blockDim >> > (djojoA, dallianceA, _result, Njojo, Nalliance);
	cudaDeviceSynchronize(); // synchronization function

    return true;
}