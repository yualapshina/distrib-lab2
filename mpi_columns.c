#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main (int argc, char *argv[])
{
	double *matrix, *vector, *result, *process_matrix, *process_vector, *process_result;
	int size, n_rows, free_rows, comm_size, my_rank, *buffer_n, *buffer_i, *buffer_displ;
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
	process_vector = (double *)malloc(n_rows * sizeof(double));
	process_result = (double *)malloc(size * sizeof(double));
	for (int i = 0; i < size; i++)
	{
		process_result[i] = 0;
	}
	
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
	buffer_i = (int *)malloc(comm_size * sizeof(int));
	buffer_n = (int *)malloc(comm_size * sizeof(int));
	buffer_displ = (int *)malloc(comm_size * sizeof(int));
	free_rows = size;
	n_rows = (size / comm_size);
	buffer_n[0] = n_rows;
	buffer_i[0] = 0;
	buffer_displ[0] = 0;
	for (int i = 1; i < comm_size; i++) 
	{
		free_rows -= n_rows;
		n_rows = free_rows / (comm_size - i);
		buffer_n[i] = n_rows;
		buffer_i[i] = buffer_i[i - 1] + buffer_n[i - 1];
		buffer_displ[i] = buffer_i[i];
	}
	for (int i = 0; i < size; i++)
	{
		MPI_Scatterv(matrix, buffer_n, buffer_displ, MPI_DOUBLE, process_matrix + i * buffer_n[my_rank], buffer_n[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		for (int j = 0; j < comm_size; j++)
		{
			buffer_displ[j] += size;
		}
	}
	MPI_Scatterv(vector, buffer_n, buffer_i, MPI_DOUBLE, process_vector, buffer_n[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	n_rows = buffer_n[my_rank];
	
	// вычисление
	for (int i = 0; i < n_rows; i++) 
	{
		for (int j = 0; j < size; j++)
		{
			process_result[j] += process_matrix[j * n_rows + i] * process_vector[i];
		}
	}

	// сбор данных
	MPI_Allreduce(process_result, result, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	free(buffer_n);
	free(buffer_i);
	free(buffer_displ);
	
	// вывод результатов
	if (my_rank == 0) 
	{
		end = clock();
		duration = end - start;
		double duration_sec = (double)duration / (double)CLOCKS_PER_SEC;
		printf("MPI Columns\nTime: %f seconds\nResult: ", duration_sec);
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
	free(process_vector);
	free(process_result);
	MPI_Finalize();
	return 0;
}
