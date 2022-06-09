#include "kernelCall.h"

__device__ float dist(Army2* _A, Army2* _B) {
    float d = 0;
    for (int i = 0; i < ARMY_DIMENSION; i++) {
        d += (_A->pos[i] - _B->pos[i]) * (_A->pos[i] - _B->pos[i]);
    }
    return sqrt(d);
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
    __shared__ Pair2 blockResult[NUM_RESULTS + BLOCK_SIZE];

    if (globalIdx > (Njojo -1)) return;

    if (localIdx == 0) {
        maxPair = { 0, 0, RANGE_MAX };
        LOOP_I(NUM_RESULTS + BLOCK_SIZE) 
             blockResult[i] = { 0, 0, RANGE_MAX };
    }
    __syncthreads();

    // 에외 처리
    if(localIdx < Njojo)
        subJojo[localIdx] = djojoA[globalIdx];

    __syncthreads();


    for (int bID = 0; bID < ceil((float)Nalliance / BLOCK_SIZE); bID++) {
        int offset = bID * BLOCK_SIZE;
        if (offset + localIdx < Nalliance) // 예외처리
            subAlliance[localIdx] = dallianceA[offset + localIdx];
        __syncthreads();


        LOOP_I(BLOCK_SIZE) {
            Pair2 dp;

            if ((offset + localIdx) >= Nalliance) { // 예외처리
                dp = maxPair;
            }
            else {
                dp = { subJojo[localIdx].ID, subAlliance[i].ID,
                dist(&subJojo[localIdx], &subAlliance[i]) };
            }
            
            blockResult[NUM_RESULTS + localIdx] = dp;
            //if (blockIdx.x == 0 && i == 0) {
               // printf("a  : %d / %d %d %lf\n", NUM_RESULTS + localIdx, blockResult[NUM_RESULTS + localIdx].A, blockResult[NUM_RESULTS + localIdx].B, blockResult[NUM_RESULTS + localIdx].dist);
            //}
            //printf("a i : %d / %d %d %lf\n", i, blockResult[NUM_RESULTS + localIdx].A, blockResult[NUM_RESULTS + localIdx].B, blockResult[NUM_RESULTS + localIdx].dist);

            //  자! 병목현상 들어갑니다~
            __syncthreads();

            if (localIdx == 0) {
               //for(int k =0; k< NUM_RESULTS + BLOCK_SIZE; k++)
                    //printf("a i : %d / %d %d %lf\n", i, blockResult[k].A, blockResult[k].B, blockResult[k].dist);
                SORT(blockResult, NUM_RESULTS + BLOCK_SIZE);

                //for (int k = 0; k < NUM_RESULTS + BLOCK_SIZE; k++)
                    //printf("b i : %d /  %d %d %lf\n", i, blockResult[k].A, blockResult[k].B, blockResult[k].dist);
            }
            __syncthreads();


        }

    }
    __syncthreads();

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