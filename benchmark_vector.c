#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main (int argc, char *argv[])
{
	int size;
	double **matrix, *vector, *result;
	clock_t start, end, duration;

    // подготовка
    FILE *fp = fopen(argv[1], "r");
    fscanf(fp, "%d", &size);
    matrix = (double **)malloc(size * sizeof(double *));
    for (int i = 0; i < size; i++)
    {
        matrix[i] = (double *)malloc(size * sizeof(double));
        for (int j = 0; j < size; j++)
        {
            fscanf(fp, "%lf", &matrix[i][j]);
        }
    }
    vector = (double *)malloc(size * sizeof(double));
    result = (double *)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++)
    {
        fscanf(fp, "%lf", &vector[i]);
        result[i] = 0;
    }
    fclose(fp);
    
    // вычисление
	start = clock();

	for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }

	end = clock();
	duration = end - start;
	double duration_sec = (double)duration / (double)CLOCKS_PER_SEC;
	printf("Benchmark\nTime: %f seconds\nResult: ", duration_sec);
	for (int i = 0; i < size; i++)
    {
        printf("%.2f ", result[i]);
    }
	printf("\n\n");
	
    // завершение
    for (int i = 0; i < size; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
    free(vector);
	free(result);
    return 0;
}
