#include "kernelCall.h"

__device__ float dist(Army2* _A, Army2* _B) {
    float d = 0;
    for (int i = 0; i < ARMY_DIMENSION; i++) {
        d += (_A->pos[i] - _B->pos[i]) * (_A->pos[i] - _B->pos[i]);
    }
    return sqrt(d);
    //return d;
}

__device__ void  SORT(Pair2* b, int size)
{
    int i, j;
    Pair2 tmp;

    for (i = 0; i < size - 1; i++)
    {
        for (j = 0; j < size - i - 1; j++)
        {
            if (b[j].dist > b[j + 1].dist)
            {
                tmp = b[j];
                b[j] = b[j + 1];
                b[j + 1] = tmp;
            }
        }
    }
}


__global__ void globalSort(Pair2* _result, Pair2* _resultF, unsigned int Njojo) {
    if (threadIdx.x == 0) {
        printf("global Merge\n");
        SORT(_result, NUM_RESULTS * ceil((float)Njojo / BLOCK_SIZE));
    }

    __syncthreads();

    _resultF[threadIdx.x] = _result[threadIdx.x];

}

__global__ void addKernel(Army2* djojoA, Army2* dallianceA, Pair2* _result, unsigned int Njojo, unsigned int Nalliance)
{	
	unsigned int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    __shared__ Pair2 maxPair;
    __shared__ Army2 subJojo[BLOCK_SIZE];
    __shared__ Army2 subAlliance[BLOCK_SIZE];
    __shared__ Pair2 subResult[BLOCK_SIZE][NUM_RESULTS];
    __shared__ Pair2* blockResult;


    if (globalIdx > Njojo-1) return;
    else subJojo[localIdx] = djojoA[globalIdx];
    

    if (localIdx == 0) {
        maxPair = { 0, 0, RANGE_MAX };
        blockResult = (Pair2*)subResult;
        LOOP_I(BLOCK_SIZE) {
            for(int j = 0; j < NUM_RESULTS; j++)
                subResult[i][j] = maxPair;
        } 
    }

    __syncthreads();
    
    for (int bID = 0; bID < ceil((float)Nalliance / BLOCK_SIZE); bID++) {
        int offset = bID * BLOCK_SIZE;
        if (offset + localIdx < Nalliance)
            subAlliance[localIdx] = dallianceA[offset + localIdx];

        __syncthreads();


        LOOP_I(BLOCK_SIZE) {
            Pair2 dp;

            if ((offset + localIdx) >= Nalliance) {
                dp = maxPair;
            }
            else {
                dp = { subJojo[localIdx].ID, subAlliance[i].ID,
                dist(&subJojo[localIdx], &subAlliance[i]) };
            }

            if (subResult[localIdx][LAST_PAIR].dist > dp.dist) {
                subResult[localIdx][LAST_PAIR] = dp;
                SORT(subResult[localIdx], NUM_RESULTS);// sort : 각 쓰레드 당 100개 걸러짐
            }

        }
        
    }
    __syncthreads();

    // 각 블록 당 100개.
    if (localIdx == 0) 
        SORT(blockResult, NUM_RESULTS * BLOCK_SIZE);

    __syncthreads();


    // subresult(한 블럭 내의 (각 쓰레드마다 걸린 100개) * (쓰레드수) 모인 배열 -> 최종 100개 걸러내면 한 블럭 당 100개 걸러짐
    // 이후, 전부다 글로벌 메모리에 저장(100개 * 15625 블럭 수)에서 최종 100개 추리기
    LOOP_I(NUM_RESULTS)
        _result[NUM_RESULTS * blockIdx.x + i] = blockResult[i];
    
    if (localIdx == 0)
        printf("processed block : %d\n", blockIdx.x);
}


bool kernelCall(Army2* djojoA, Army2* dallianceA, Pair2* _result, unsigned int Njojo, unsigned int Nalliance,
    dim3 _griDim, dim3 _blockDim, Pair2* _resultF) {

    addKernel << <_griDim, _blockDim >> > (djojoA, dallianceA, _result, Njojo, Nalliance);
	cudaDeviceSynchronize(); // synchronization function
    globalSort << < 1, 100 >> > (_result, _resultF, Njojo);
    cudaDeviceSynchronize(); // synchronization function

    return true;
}