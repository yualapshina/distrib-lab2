#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main (int argc, char *argv[])
{
	double *matrix, *vector, *result, *process_matrix, *process_result;
	int size, n_rows, free_rows, comm_size, my_rank, *buffer_n, *buffer_i;
	clock_t start, end, duration;
	FILE *fp;
	
	// инициализация системы
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	// считывание и сообщение размера вектора
	if (my_rank == 0) 
	{
		start = clock();
		fp = fopen(argv[1], "r");
		fscanf(fp, "%d", &size);
	}
	MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	// выделение памяти под данные процессов
	free_rows = size;
	for (int i = 0; i < my_rank; i++)
	{
		free_rows = free_rows - free_rows / (comm_size - i);
	}
	n_rows = free_rows / (comm_size - my_rank);
	vector = (double *)malloc(size * sizeof(double));
	result = (double *)malloc(size * sizeof(double));
	process_matrix = (double *)malloc(n_rows * size * sizeof(double));
	process_result = (double *)malloc(n_rows * sizeof(double));
	
	// считывание данных
	if (my_rank == 0) 
	{
		matrix = (double *)malloc(size * size * sizeof(double));
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				fscanf(fp, "%lf", &matrix[i * size + j]);
			}
		}
		for (int i = 0; i < size; i++)
		{
			fscanf(fp, "%lf", &vector[i]);
		}
		fclose(fp);
	}
	
	// разделение и сообщение данных
	MPI_Bcast(vector, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	buffer_i = (int *)malloc(comm_size * sizeof(int));
	buffer_n = (int *)malloc(comm_size * sizeof(int));
	free_rows = size;
	n_rows = (size / comm_size);
	buffer_n[0] = n_rows * size;
	buffer_i[0] = 0;
	for (int i = 1; i < comm_size; i++) 
	{
		free_rows -= n_rows;
		n_rows = free_rows / (comm_size - i);
		buffer_n[i] = n_rows * size;
		buffer_i[i] = buffer_i[i - 1] + buffer_n[i - 1];
	}
	MPI_Scatterv(matrix, buffer_n, buffer_i, MPI_DOUBLE, process_matrix, buffer_n[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	n_rows = buffer_n[my_rank] / size;

	// вычисление
	for (int i = 0; i < n_rows; i++) 
	{
		process_result[i] = 0;
		for (int j = 0; j < size; j++)
		{
			process_result[i] += process_matrix[i * size + j] * vector[j];
		}
	}

	// сбор данных
	buffer_i[0] = 0;
	buffer_n[0] = size / comm_size;
	free_rows = size;
	for (int i = 1; i < comm_size; i++) 
	{
		free_rows -= buffer_n[i - 1];
		buffer_n[i] = free_rows / (comm_size - i);
		buffer_i[i] = buffer_i[i - 1] + buffer_n[i - 1];
	}
	MPI_Allgatherv(process_result, buffer_n[my_rank], MPI_DOUBLE, result, buffer_n, buffer_i, MPI_DOUBLE, MPI_COMM_WORLD);
	free(buffer_n);
	free(buffer_i);
	
	// вывод результатов
	if (my_rank == 0) 
	{
		end = clock();
		duration = end - start;
		double duration_sec = (double)duration / (double)CLOCKS_PER_SEC;
		printf("MPI Rows\nTime: %f seconds\nResult: ", duration_sec);
		for (int i = 0; i < size; i++)
		{
			printf("%.2f ", result[i]);
		}
		printf("\n\n");
		free(matrix);
	}
	
	// завершение
	free(vector);
	free(result);
	free(process_matrix);
	free(process_result);
	MPI_Finalize();
	return 0;
}
