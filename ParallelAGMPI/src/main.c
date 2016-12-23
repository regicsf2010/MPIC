/*
 * main.c
 *
 *  Created on: Nov 10, 2016
 *      Author: reginaldo
 */
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <mpi.h>

		/******* AG Parameters *******/
const int NCHROMOSOMES = 500; // must be even ('cause of crossover)
const int NGENERATIONS = 1000;
const int NGENES = 500;
const double CROSSOVERRATE = 0.8;
const double MUTATIONRATE = 0.01;
const double SD = 0.1;
const int RANK = 3;

		/******* Prototype of each function ********/
void initializePopulation(double pop[NCHROMOSOMES][NGENES], double infimum, double maximum);
void calculateFitness(double pop[NCHROMOSOMES][NGENES], double fitnessPop[NCHROMOSOMES], int partition, int size, double genes[]);
void parentSelection(double pop[NCHROMOSOMES][NGENES], double fitnessPop[NCHROMOSOMES], double selected[NCHROMOSOMES][NGENES]);
void crossover(double selected[NCHROMOSOMES][NGENES]);
void mutation(double pop[NCHROMOSOMES][NGENES]);
void survivorSelection(double pop[NCHROMOSOMES][NGENES], double fitnessPop[NCHROMOSOMES],
					   double selected[NCHROMOSOMES][NGENES], double fitnessSelected[NCHROMOSOMES]);


/* Optimization functions prototypes*/
double ackley(double vars[NGENES]);

/* Usage of infimum e maximum */
const double INFIMUM = -40;
const double MAXIMUM = 40;

/* Print the population */
void print(double p[NCHROMOSOMES][NGENES], double fitness[NCHROMOSOMES]) {
	int c, g;
	for (c = 0; c < NCHROMOSOMES; c++) {
		for (g = 0; g < NGENES; g++) {
			printf("%.4f ", p[c][g]);
		}
		printf("| f(x) =  %.4f ", fitness[c]);
		printf("\n");
	}
}

/* Generate a random normal distribution number: prototype function*/
double normal();

void calcFitnessAndSend(int partition, double genes[]);

		/*******  Main application *******/
int main(int argc, char *argv[]) {
	int rank, size;
	MPI_Status status;
	MPI_Init(&argc, &argv);      				 /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);        /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &size);        /* get number of processes */

	const int partition = NCHROMOSOMES/ (size - 1);
	double genes[partition * NGENES];

	int counterIt = -1;

	if(rank == 0) {
		clock_t begin = clock(); // get the system clock
		double pop[NCHROMOSOMES][NGENES], selected[NCHROMOSOMES][NGENES];
		double fitnessPop[NCHROMOSOMES], fitnessSelected[NCHROMOSOMES];

		initializePopulation(pop, INFIMUM, MAXIMUM);
		calculateFitness(pop, fitnessPop, partition, size, genes);

		int i;
		for (i = 0; i < NGENERATIONS; ++i) {
			parentSelection(pop, fitnessPop, selected);
			crossover(selected);
			mutation(selected);
			calculateFitness(selected, fitnessSelected, partition, size, genes);
			survivorSelection(pop, fitnessPop, selected, fitnessSelected); // return survivors in pop matrix
		}

		clock_t end = clock(); // get the system clock
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC; // calculate the elapsed time
		printf("%0.4f", time_spent);

	} else if(rank == 1) {
		while(1){
			if(counterIt == NGENERATIONS) {
				break;
			} else {
				MPI_Recv(genes, partition * NGENES, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, &status);
				calcFitnessAndSend(partition, genes);
				counterIt++;
			}
		}
	} else if(rank == 2) {
		while(1){
			if(counterIt == NGENERATIONS) {
				break;
			} else {
				MPI_Recv(genes, partition * NGENES, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, &status);
				calcFitnessAndSend(partition, genes);
				counterIt++;
			}
		}
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}

void calcFitnessAndSend(int partition, double genes[]) {
	double fit[partition];
	int k, c;
	for(k = 0; k < partition; k++) {
		double vars[NGENES]; int cont = 0;
		for(c = k * NGENES; c < k * NGENES + NGENES; c++) {
			vars[cont]= genes[c];
			cont++;
		}
		fit[k] = ackley(vars);
	}
	MPI_Send(fit, partition, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

/* Initialize the population with random solutions in the feature space */
void initializePopulation(double pop[NCHROMOSOMES][NGENES], double infimum, double maximum) {
	time_t t;
	srand((unsigned) time(&t));
	double val = 0;

	int c, g;
	for (c = 0; c < NCHROMOSOMES; c++) {
		for (g = 0; g < NGENES; g++) {
			val = (double)rand() / (double)RAND_MAX;
			pop[c][g] = (1 - val) * infimum + val * maximum;
		}
	}
}

/* This function store each fitness in each position of fitnessPop vector
 * related to the solution stored in the same pop vector position */
void calculateFitness(double pop[NCHROMOSOMES][NGENES],
					  double fitnessPop[NCHROMOSOMES],
					  int partition, int size, double genes[]) {
	MPI_Status status;
	int c, g, t, r;
	double fit[partition];
	for(t = 1; t < size; t++) {
		int cont = 0;
		for(c = (t-1) * partition; c < (t-1) * partition + partition; c++) {
			for(g = 0; g < NGENES; g++) {
				genes[cont] = pop[c][g];
				cont++;
			}
		}
		MPI_Send(genes, partition * NGENES, MPI_DOUBLE, t, t, MPI_COMM_WORLD);
		MPI_Recv(fit, partition, MPI_DOUBLE, t, 0, MPI_COMM_WORLD, &status);
		int cont1 = 0;
		for(r = (t-1) * partition; r < (t-1) * partition + partition; r++) {
			fitnessPop[r] = fit[cont1];
			cont1++;
		}
	}
}

/* Tournament selection */
void parentSelection(double pop[NCHROMOSOMES][NGENES], double fitnessPop[NCHROMOSOMES], double selected[NCHROMOSOMES][NGENES]) {
	time_t t;
	srand((unsigned) time(&t));

	int idx1 = -1, idx2 = -1;
	int c, j, g;
	for (c = 0; c < NCHROMOSOMES; c++) {
		idx1 = rand() % NCHROMOSOMES;

		for (j = 1; j < RANK; j++) { // j starts at 1
			idx2 = rand() % NCHROMOSOMES;
			if(fitnessPop[idx1] > fitnessPop[idx2])
				idx1 = idx2;
		}

		for (g = 0; g < NGENES; g++) // copy the solution means copy all genes
			selected[c][g] = pop[idx1][g];
	}
}

/* Arithmetic crossover */
void crossover(double selected[NCHROMOSOMES][NGENES]) {
	time_t t;
	srand((unsigned) time(&t));
	double val = 0;

	int c, g;
	for (c = 0; c < NCHROMOSOMES; c += 2) { // two by two
		for (g = 0; g < NGENES; g++) {
			double a1 = selected[c][g];
			double a2 = selected[c + 1][g];
			val = (double)rand() / (double)RAND_MAX;

			if( val < CROSSOVERRATE ) {
				selected[c][g] = a1 + (a2 - a1) * ( (double)rand() / (double)RAND_MAX );
				selected[c + 1][g] = a2 + (a1 - a2) * ( (double)rand() / (double)RAND_MAX );
			}
		}
	}
}

/* Gaussian mutation */
void mutation(double pop[NCHROMOSOMES][NGENES]) {
	time_t t;
	srand((unsigned) time(&t));
	double val = 0;

	int c, g;
	for (c = 0; c < NCHROMOSOMES; c++) {
		for (g = 0; g < NGENES; g++) {
			val = (double)rand() / (double)RAND_MAX;

			if(val < MUTATIONRATE)
				pop[c][g] = pop[c][g] + SD * normal();
		}
	}
}

/* Elitism: (without sorting the two populations) */
void survivorSelection(double pop[NCHROMOSOMES][NGENES], double fitnessPop[NCHROMOSOMES],
		               double selected[NCHROMOSOMES][NGENES], double fitnessSelected[NCHROMOSOMES]) {

	int c, g;
	for (c = 0; c < NCHROMOSOMES; c++) {
		if(fitnessPop[c] > fitnessSelected[c]) {
			for (g = 0; g < NGENES; g++) {
				pop[c][g] = selected[c][g]; // copy each gene
			}
			fitnessPop[c] = fitnessSelected[c]; // copy fitness
		}
	}
}

/* Optimization functions */
double ackley(double vars[NGENES]) {
	double val = 0;
	double s = 0;
	int g;

	for (g = 0; g < NGENES; g++)
		s += pow(vars[g], 2);
	double p1 = -20 * exp(-0.2 * sqrt(1.0 / NGENES * s));

	s = 0;
	for (g = 0; g < NGENES; g++)
		s += cos(2 * M_PI * vars[g]);
	double p2 = -exp(1.0 / NGENES * s) + 20 + M_E;
	return p1 + p2;

	return val;
}

/* Normal distribution: TODO please use a reliable function for normal distribution */
double normal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return normal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}


