#include "kernelCall.h"

__device__ float dist(Army2* _A, Army2* _B) {
    float d = 0;
    for (int i = 0; i < ARMY_DIMENSION; i++) {
        d += (_A->pos[i] - _B->pos[i]) * (_A->pos[i] - _B->pos[i]);
    }
    return sqrt(d);
}

__device__ void QuickSort(Pair2 *arr, int left, int right) {
    int L = left, R = right; Pair2 temp; Pair2 pivot = arr[(left + right) / 2];

    while (L <= R) {
    
        while (arr[L].dist < pivot.dist)
            L++;
        while (arr[R].dist > pivot.dist)
            R--;
        if (L <= R) {
            if (L != R) {
                temp = arr[L];
                arr[L] = arr[R];
                arr[R] = temp;
            }
            L++; R--;
        }
    }
    if (left < R)
        QuickSort(arr, left, R);
    if (L < right)
        QuickSort(arr, L, right);}


__global__ void globalSort(Pair2* _result, Pair2* _resultF, unsigned int Njojo) {
    if (threadIdx.x == 0) {
        printf("global Merge\n"); 
        QuickSort(_result, 0, NUM_RESULTS * ceil((float)Njojo / BLOCK_SIZE) - 1);
    }

    __syncthreads();

    _resultF[threadIdx.x] = _result[threadIdx.x];

}

__global__ void addKernel(Army2* djojoA, Army2* dallianceA, Pair2* _result, unsigned int Njojo, unsigned int Nalliance)
{
    unsigned int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;


    __shared__ Pair2 maxPair;
    __shared__ Army2 maxArmy;
    __shared__ Army2 subJojo[BLOCK_SIZE];
    __shared__ Army2 subAlliance[BLOCK_SIZE];
    __shared__ Pair2 blockResult[NUM_RESULTS + BLOCK_SIZE];

    //if (globalIdx > Njojo-1) return;

    if (localIdx == 0) {
        maxPair = { 0, 0, RANGE_MAX };
        maxArmy.ID = (Nalliance + 1);
        maxArmy.pos[0] = RANGE_MAX;
        maxArmy.pos[1] = RANGE_MAX;
        maxArmy.pos[2] = RANGE_MAX;

        LOOP_I(NUM_RESULTS) {
            blockResult[i] = maxPair;
        }
    }
    __syncthreads();

    subJojo[localIdx] = djojoA[globalIdx]; //A Shared Memory

    __syncthreads();

    for (int bID = 0; bID < ceil((float)Nalliance / BLOCK_SIZE); bID++) {
        int offset = bID * BLOCK_SIZE;
        if ((offset + localIdx) < Nalliance) // B Shared Memory 
            subAlliance[localIdx] = dallianceA[offset + localIdx];
        else
            subAlliance[localIdx] = maxArmy;
        __syncthreads();

        LOOP_I(BLOCK_SIZE) {
            Pair2 dp;

            if ((offset + localIdx) >= Nalliance || globalIdx >= Njojo) { // 예외처리
                dp = maxPair;
            }
            else {
                //dp = { subJojo[localIdx].ID, subAlliance[i].ID, dist(&subJojo[localIdx], &subAlliance[i]) };
                dp = { subJojo[localIdx].ID, subAlliance[(i+localIdx)%BLOCK_SIZE].ID, dist(&subJojo[localIdx], &subAlliance[(i + localIdx) % BLOCK_SIZE]) };
                // No BankConflict
            }

            blockResult[NUM_RESULTS + localIdx] = dp;
            __syncthreads();

            if (localIdx == 0) {
                QuickSort(blockResult, 0,  NUM_RESULTS + BLOCK_SIZE-1);
            }
            __syncthreads();


        }

    }
    __syncthreads();
    if (localIdx == 0) { // 병렬 처리 가능
        LOOP_I(NUM_RESULTS) 
            _result[NUM_RESULTS * blockIdx.x + i] = blockResult[i];
        
        printf("processed block : %d\n", blockIdx.x);
    }
}


bool kernelCall(Army2* djojoA, Army2* dallianceA, Pair2* _result, unsigned int Njojo, unsigned int Nalliance,
    dim3 _griDim, dim3 _blockDim, Pair2* _resultF) {

    addKernel << <_griDim, _blockDim >> > (djojoA, dallianceA, _result, Njojo, Nalliance);
    cudaDeviceSynchronize(); // synchronization function
    globalSort << < 1, 100 >> > (_result, _resultF, Njojo);
    cudaDeviceSynchronize(); // synchronization function

    return true;
}