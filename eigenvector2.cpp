#include <iostream>
#include <memory>
#include <cstdint>
#include <iomanip>
#include <random>
#include "vectors_and_matrices/array_types.hpp"
#include "mpi.h"


using ptrdiff_t = std::ptrdiff_t;
using size_t = std::size_t;

void fill_random(vec<double> x, double xmin, double xmax, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(xmin, xmax);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}

void fill_random(matrix<double> x, double dispersion, size_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0, dispersion);
    for (ptrdiff_t i = 0; i < x.length(); i++)
    {
        x(i) = dist(rng);
    }
}

void symmetrize(matrix<double> a)
{
    ptrdiff_t n = a.nrows();
    for (ptrdiff_t i = 0; i < n; i++)
    {
        for (ptrdiff_t j = 0; j < i; j++)
        {
            double sym_elt = (a(i, j) + a(j, i)) / 2;
            a(i, j) = sym_elt;
            a(j, i) = sym_elt;
        }
    }
}

double eigenvalue(matrix<double> A, MPI_Comm comm, int* sendcounts, int* displs)
{
    ptrdiff_t n = A.ncols(), m = A.nrows();
    vec<double> v0(n);
    vec<double> v1(n);

    int myrank, world_size;

    MPI_Comm_rank(comm, &myrank);
    // MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // generate a random vector on rank 0 and broadcast it
    if (myrank == 0)
    {
        fill_random(v0, -10.0, 10.0, 24680);
    }

    MPI_Bcast(v0.raw_ptr(), n, MPI_DOUBLE, 0, comm);

    ptrdiff_t iter;
    ptrdiff_t i, j;
    double rq = 1.0, diff = 1.0;
    int niter = 0;
    ptrdiff_t ilocal_start = displs[myrank];
    while (std::abs(diff) > 1e-15)
    {
        // normalize v0
        double norm2 = 0;
        for (i=0; i < n; i++)
        {
            norm2 += v0(i) * v0(i);
        }

        for (i=0; i < n; i++)
        {
            v0(i) /= sqrt(norm2);
        }

        // v1 = A * v0

        for (i=0; i<sendcounts[myrank]; i++)
        {
            v1(ilocal_start+i) = 0;
            for (j=0; j<n; j++)
            {
                v1(ilocal_start+i) += v0(j) * A(i, j);
            }
        }
        // Rayleigh quotient = (x' A x) / (x' x) = dot(v0, v1)
        rq = 0;
        double buf_rq = 0;

        for (j=0; j<sendcounts[myrank]; ++j)
        {
            buf_rq += v1(j+ilocal_start) * v0(j+ilocal_start);
        }
        MPI_Reduce(&buf_rq, &rq, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Bcast(&rq, 1, MPI_DOUBLE, 0, comm);

        diff = 0;
        double buf_diff = 0;
        for (j=0; j<sendcounts[myrank]; ++j)
        {
            double d = v1(j+ilocal_start) / rq - v0(j+ilocal_start);
            if (std::abs(d) > buf_diff)
            {
                buf_diff = std::abs(d);
            }
        }

        MPI_Reduce(&buf_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Bcast(&diff, 1, MPI_DOUBLE, 0, comm);

    //     // Соберем все процессы
        MPI_Allgatherv(v1.raw_ptr() + ilocal_start, sendcounts[myrank], MPI_DOUBLE, v1.raw_ptr(), sendcounts, displs, MPI_DOUBLE, comm);
        
    //     // swap v1 and v0
        vec<double> tmp = v0;
        v0 = v1;
        v1 = tmp;
        niter += 1;

    }

    return rq;
}

// read an integer number from stdin into `n`
void read_integer(int* n, int rank, MPI_Comm comm)
{
    if (rank==0)
    {
        std::cin >> *n;
    }
    // Отправим сообщение всем процессам
    MPI_Bcast(n, 1, MPI_INT, 0, comm);
}

// scatter matrix `source` over processes in communicator `comm` from `root`
void scatter_matrix(matrix<double> source, matrix<double> dest, int root, int* sendcounts, int* displs, int myrank, MPI_Comm comm)
{
    double *src_ptr = source.raw_ptr(), *dest_ptr = dest.raw_ptr();
    // recvcount = sendcounts[myrank] - количество элементов, которые пойдут на i-й процесс
    MPI_Scatterv(src_ptr, sendcounts, displs, MPI_DOUBLE, dest_ptr, sendcounts[myrank], MPI_DOUBLE, root, comm);
}

int main(int argc, char* argv[])
{
    int n;

    int myrank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    read_integer(&n, myrank, MPI_COMM_WORLD);
    
    // Теущий сдвиг
    int current_display = 0;
    // Остаток от размерности матрицы, деленной на количество процессов
    int remain = n % world_size;
    // Целочисленный массив, в i-ячейках которого хранится количество данных, которые пойдут на i-тый процесс
    int *sendcounts = new int[world_size];
    // Целочисленный массив сдвигов относительно начала передаваемых данных
    int *displs = new int[world_size];

    // Рассчитаем массивы sendcounts и displs. Итерируемся по числу процессов
    for (int i = 0; i < world_size; i++) {
        displs[i] = current_display;
        // Количество элементов (чисел), принимаемых i-м процессом 
        sendcounts[i] = int(n / world_size) * n; 
        // Распределим (более-менее) равномерно по процессам остаток от деления
        if (remain > 0) {
           sendcounts[i] += n;  // так как в каждой строке n элементов
           remain--;
        }
        current_display += sendcounts[i];
    }

    // Создадим матрицу (=часть исхохной матрицы)
    matrix<double> A(sendcounts[myrank] / n, n);

    int* sendcounts_ev = sendcounts;
    int* displs_ev = displs;

    // generate matrix on rank 0 (for simplicity)
    if (myrank == 0)
    {
        matrix<double> A_all(n, n);
        fill_random(A_all, 1.0, 9876);
        symmetrize(A_all);
        scatter_matrix(A_all, A, 0, sendcounts, displs, myrank, MPI_COMM_WORLD);
        for (int i = 0; i < world_size; i++) {
            sendcounts_ev[i] /= n;
            displs_ev[i] /= n;
        }
    }
    else
    {
        scatter_matrix(A, A, 0, sendcounts, displs, myrank, MPI_COMM_WORLD);
    }

    MPI_Bcast(sendcounts_ev, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs_ev, world_size, MPI_INT, 0, MPI_COMM_WORLD);

    double t0 = MPI_Wtime();
    
    double q = eigenvalue(A, MPI_COMM_WORLD, sendcounts_ev, displs_ev);

    double t1 = MPI_Wtime();

    if (myrank == 0)
    {
        std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1)
                << "Timing: " << t1 - t0 << " sec\n"
                << "Answer = " << q
                << std::endl;
    }

    MPI_Finalize();

    return 0;
}
