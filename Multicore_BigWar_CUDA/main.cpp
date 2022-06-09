#include "Common/BigWar.h"
#include "Common/DS_timer.h"
#include "Grader.h"
#include "KernelCall.h"
#include "Team.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;


int main(int argc, char** argv) {
	if (argc < 4) {
		printf("Usage: exe inputA inputB outputFile GT_file(optional for grading) TeamID(optional for grading)\n");
		exit(1);
	}
	bool on = true;
	DS_timer timer(1,1);
	timer.initTimers();
	Pair result[100];

	timer.onTimer(0);
 	// **************************************//
	// Write your code here
	// CAUTION: DO NOT MODITY OTHER PART OF THE main() FUNCTION

	DS_timer timer1(10);
	timer1.initTimers();
	timer1.setTimerName(0, (char*)"Total");
	timer1.setTimerName(1, (char*)"Data load");
	timer1.setTimerName(2, (char*)"Processing");

	FILE* fp1 = NULL;
	UINT numArmies[2] = { 0, };
	Team2* jojo, * alliance;
	Army2* djojoA, * dallianceA;
	Team2 teams [NUM_TEAM];
	Pair2* dresult2, *dresultF;
	Pair2 results2[NUM_RESULTS];

	timer1.onTimer(0);
	for (int i = 0; i < NUM_TEAM; i++) {
		fopen_s(&fp1, argv[i + 1], "rb");
		if (fp1 == NULL) {
			printf("Fail to read the file - %s\n", argv[i + 1]);
			exit(2);
		}
		fread_s(&numArmies[i], sizeof(UINT), sizeof(UINT), 1, fp1);
		printf("%s: %d armys\n", argv[i + 1], numArmies[i]);

		teams[i].numArmies = numArmies[i];
		teams[i].armies = new Army2[numArmies[i]];
		//#pragma omp parallel for
		for (int ID = 0; ID < numArmies[i]; ID++) {
			teams[i].armies[ID].ID = (float)ID;
			if (fread_s(teams[i].armies[ID].pos, sizeof(float) * 3, sizeof(float), 3, fp1) == 0) {

				numArmies[i] = ID;
				break;
			}

		}
		fclose(fp1);
	}
	jojo = &teams[0];
	alliance = &teams[1];

	
	// result2 initialization
	#pragma omp parallel for
	for (int i = 0; i < NUM_RESULTS; i++) {
		results2[i].dist = RANGE_MAX;
		results2[i].A = 0;
		results2[i].B = 0;
	}

	timer1.onTimer(1);

	dim3 blockDim(BLOCK_SIZE);
	dim3 gridDim(ceil((float)(jojo->numArmies) / blockDim.x));

	// device memory allocation
	cudaMalloc(&djojoA, sizeof(Army2) * jojo->numArmies);
	cudaMemset(djojoA, 0, sizeof(Army2) * jojo->numArmies);

	cudaMalloc(&dallianceA, sizeof(Army2) * alliance->numArmies);
	cudaMemset(dallianceA, 0, sizeof(Army2) * alliance->numArmies);

	cudaMalloc(&dresult2, sizeof(Pair2) * NUM_RESULTS * gridDim.x);
	cudaMemset(dresult2, 0, sizeof(Pair2)* NUM_RESULTS * gridDim.x);

	cudaMalloc(&dresultF, sizeof(Pair2)* NUM_RESULTS);
	cudaMemset(dresultF, 0, sizeof(Pair2)* NUM_RESULTS);

	//copy the memory
	cudaMemcpy(djojoA, jojo->armies, sizeof(Army2) * jojo->numArmies, cudaMemcpyHostToDevice);
	cudaMemcpy(dallianceA, alliance->armies, sizeof(Army2) * alliance->numArmies, cudaMemcpyHostToDevice);
	timer1.offTimer(1);

	timer1.onTimer(2);

	//kernel call

	kernelCall(djojoA, dallianceA, dresult2, jojo->numArmies, alliance->numArmies, gridDim, blockDim, dresultF);
	timer1.offTimer(2);


	cudaMemcpy(results2, dresultF, sizeof(Pair2) * NUM_RESULTS, cudaMemcpyDeviceToHost);
	
	//result2 Device -> Host
	#pragma omp parallel for
	LOOP_I(NUM_RESULTS) {
		printf("Results : %d %d %lf\n", results2[i].A, results2[i].B, results2[i].dist);
		result[i].A = results2[i].A;
		result[i].B = results2[i].B;
		result[i].dist = results2[i].dist;
	}


	// Write the result
	FILE* wfp = NULL;
	fopen_s(&wfp, "result_CUDA.txt", "w");
	if (wfp == NULL) {
		printf("Fail to open the output file\n");
		exit(2);
	}
	for (int i = 0; i < NUM_RESULTS; i++) {
		fprintf(wfp, "%d %d %.2f\n", result[i].A, result[i].B, result[i].dist);
	}
	fclose(wfp);
	



	timer1.offTimer(0);
	timer1.printTimer();
	printf("\n\n=============================================================\n");
	printf("=====================Á¤´ä=========================================\n\n");

	//***************************************//
	timer.offTimer(0);
	timer.printTimer(0);

	// Result validation
	if (argc < 5)
		return 0;

	// Grading mode
	if (argc < 6) {
		printf("Not enough argument for grading\n");
		exit(2);
	}

	Grader grader(argv[4]);
	grader.grading(result);

	FILE* fp = NULL;
	fopen_s(&fp, argv[5], "a");
	if (fp == NULL) {
		printf("Fail to open %s\n", argv[5]);
		exit(3);
	}
	fprintf(fp, "%f\t%d\n", timer.getTimer_ms(0), grader.getNumCorrect());
	fclose(fp);

	return 0;
}