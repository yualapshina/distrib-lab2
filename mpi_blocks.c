#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main (int argc, char *argv[])
{
	double *matrix, *vector, *result, *process_matrix, *process_vector, *process_result, *received;
	int size, n_rows, free_rows, n_chunks, chunk, comm_size, my_rank, *buffer_n, *buffer_i, *buffer_y, *buffer_n_copy, *buffer_i_copy, *buffer_y_copy;
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
	n_rows = free_rows / (comm_size - my_rank);
	vector = (double *)malloc(size * sizeof(double));
	result = (double *)malloc(size * sizeof(double));
	process_matrix = (double *)malloc(size * size * sizeof(double));
	process_vector = (double *)malloc(size * sizeof(double));
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
	buffer_y = (int *)malloc(comm_size * sizeof(int));
	buffer_i_copy = (int *)malloc(comm_size * sizeof(int));
	buffer_n_copy = (int *)malloc(comm_size * sizeof(int));
	buffer_y_copy = (int *)malloc(comm_size * sizeof(int));
	n_chunks = (int)sqrt(comm_size);
	chunk = ceil((double)size / n_chunks);
	buffer_n[0] = chunk;
	buffer_i[0] = 0;
	buffer_y[0] = chunk;
	buffer_n_copy[0] = buffer_n[0];
	buffer_i_copy[0] = buffer_i[0];
	buffer_y_copy[0] = buffer_y[0];
	for (int i = 1; i < comm_size; i++) 
	{
		buffer_n[i] = chunk;
		if (i % n_chunks == n_chunks - 1)
		{
			buffer_n[i] -= (chunk * n_chunks) - size;
		}
		buffer_n_copy[i] = buffer_n[i];
		buffer_i[i] = (i / n_chunks) * size * chunk + (i % n_chunks) * chunk;
		buffer_i_copy[i] = buffer_i[i];
		buffer_y[i] = chunk;
		if (i / n_chunks == n_chunks - 1)
		{
			buffer_y[i] -= (chunk * n_chunks) - size;
		}
		buffer_y_copy[i] = buffer_y[i];
	}
	for (int i = 0; i < size; i++)
	{
		MPI_Scatterv(matrix, buffer_n_copy, buffer_i_copy, MPI_DOUBLE, process_matrix + i * buffer_n[my_rank], buffer_n_copy[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		for (int j = 0; j < comm_size; j++)
		{
			buffer_i[j] = (j / n_chunks) * chunk;
			buffer_i_copy[j] += size;
			buffer_y_copy[j]--;
			if (buffer_y_copy[j] < 1)
			{
				buffer_n_copy[j] = 0;
			}
		}
	}
	MPI_Scatterv(vector, buffer_y, buffer_i, MPI_DOUBLE, process_vector, buffer_y[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	// вычисление
	for (int i = 0; i < buffer_y[my_rank]; i++) 
	{
		for (int j = 0; j < buffer_n[my_rank]; j++)
		{
			process_result[i] += process_matrix[i * buffer_n[my_rank] + j] * process_vector[j];
		}
	}

	// сбор данных
	if (my_rank % n_chunks == 0)
	{
		received = (double *)malloc(buffer_y[my_rank] * sizeof(double));
		for (int i = my_rank + 1; i < my_rank + n_chunks; i++)
		{
			MPI_Recv(received, buffer_y[my_rank], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for (int j = 0; j < buffer_y[my_rank]; j++)
			{
				process_result[j] += received[j];
			}
		}
		free(received);
		
		if (my_rank == 0)
		{
			for (int j = 0; j < buffer_y[my_rank]; j++)
			{
				result[j] = process_result[j];
			}
			int ptr = buffer_y[my_rank];
			received = (double *)malloc(buffer_y[my_rank] * sizeof(double));
			for (int i = n_chunks; i < comm_size; i += n_chunks)
			{
				MPI_Recv(received, buffer_y[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for (int j = 0; j < buffer_y[i]; j++)
				{
					result[ptr + j] = received[j];
				}
			}
			free(received);
		}
		
		else
		{
			MPI_Send(process_result, buffer_y[my_rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
	}
	else
	{
		int dest = my_rank - 1;
		while (dest % n_chunks != 0)
		{
			dest--;
		}
		MPI_Send(process_result, buffer_y[my_rank], MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
	}
	
		
	free(buffer_n);
	free(buffer_i);
	free(buffer_y);
	free(buffer_n_copy);
	free(buffer_i_copy);
	free(buffer_y_copy);
	
	// вывод результатов
	if (my_rank == 0) 
	{
		end = clock();
		duration = end - start;
		double duration_sec = (double)duration / (double)CLOCKS_PER_SEC;
		printf("MPI Blocks\nTime: %f seconds\nResult: ", duration_sec);
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
