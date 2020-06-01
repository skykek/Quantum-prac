#include <complex>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cassert>

#include <mpi.h>

using namespace std;

typedef std::complex<double> complexd;

complexd *make_quantum_vector(long long vec_size, int rank)
{
    srand(unsigned(time(0)));

    complexd *quantum_vector = new complexd[vec_size];
    double vec_len = 0;

    for (long long i = 0; i < vec_size; i++) {
        unsigned int seed = MPI_Wtime() * (2 << 30) + (rank * 1000 + 5) * i;
        complexd rand_val = complexd(double(rand_r(&seed)) / RAND_MAX, double(rand_r(&seed)) / RAND_MAX);
        vec_len += norm(rand_val);
        quantum_vector[i] = rand_val;
    }

    MPI_Allreduce(&vec_len, &vec_len, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    vec_len = sqrt(vec_len);

    for (long long i = 0; i < vec_size; i++) {
        quantum_vector[i] /= vec_len;
    }
    return quantum_vector;
}

complexd *read_transform_matrix(ifstream &file)
{
    complexd *transform_matrix = new complexd[4];
    for (int i = 0; i < 4; i++) {
        file >> transform_matrix[i];
    }
    return transform_matrix;
}

void make_single_qubit_transform(complexd *quantum_vector, 
        complexd *transform_matrix, long long vec_size, int k)
{   
    long long k_qubit_num = 1ll << (k - 1);

    if (vec_size < k_qubit_num * 2) {
        for (long long i = 0; i < vec_size / 2; i++) {
            complexd tmp1 = quantum_vector[i];
            complexd tmp2 = quantum_vector[i + vec_size / 2];
            quantum_vector[i] = transform_matrix[0] * tmp1 + transform_matrix[1] * tmp2;
            quantum_vector[i + vec_size / 2] = transform_matrix[2] * tmp1 + transform_matrix[3] * tmp2;
        }
    } else {
        for (unsigned long long i = 0; i < vec_size; i += k_qubit_num * 2) {
            for (unsigned long long j = i; j < i + k_qubit_num; j++) {
                complexd tmp1 = quantum_vector[j];
                complexd tmp2 = quantum_vector[j + k_qubit_num];
                quantum_vector[j] = transform_matrix[0] * tmp1 + transform_matrix[1] * tmp2;
                quantum_vector[j + k_qubit_num] = transform_matrix[2] * tmp1 + transform_matrix[3] * tmp2;
            }
        }
    }
}

// complexd *read_quantum_vector(ifstream &file, long long vec_size, long long offset1, long long offset2)
// {
//     complexd *quantum_vector = new complexd[vec_size];

//     file.seekg(offset1 * sizeof(complexd));
//     file.read(reinterpret_cast<char *>(quantum_vector), vec_size / 2 * sizeof(complexd));

//     file.seekg(offset2 * sizeof(complexd));
//     file.read(reinterpret_cast<char *>(quantum_vector + vec_size / 2), vec_size / 2 * sizeof(complexd));

//     return quantum_vector;
// }

complexd *read_quantum_vector(MPI_File fh, long long vec_size, long long offset1, long long offset2)
{
    complexd *quantum_vector = new complexd[vec_size];

    MPI_File_read_at(fh, offset1 * sizeof(complexd), quantum_vector, vec_size / 2, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);
    MPI_File_read_at(fh, offset2 * sizeof(complexd), quantum_vector + vec_size / 2, vec_size / 2, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);

    return quantum_vector;
}

// void write_quantum_vector(complexd *quantum_vector, ofstream &file, long long vec_size, long long offset1, long long offset2)
// {
//     file.seekp(offset1 * sizeof(complexd));
//     file.write(reinterpret_cast<char *>(quantum_vector), vec_size / 2 * sizeof(complexd));

//     file.seekp(offset2 * sizeof(complexd));
//     file.write(reinterpret_cast<char *>(quantum_vector + vec_size / 2), vec_size / 2 * sizeof(complexd));
// }

void write_quantum_vector(complexd *quantum_vector, MPI_File fh, long long vec_size, long long offset1, long long offset2)
{
    MPI_File_write_at(fh, offset1 * sizeof(complexd), quantum_vector, vec_size / 2, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);
    MPI_File_write_at(fh, offset2 * sizeof(complexd), quantum_vector + vec_size / 2, vec_size / 2, MPI_DOUBLE_COMPLEX, MPI_STATUS_IGNORE);
}

int main(int argc, char const *argv[])
{
    MPI_Init(&argc, (char ***)&argv);

    assert(argc == 4 || argc == 5);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n, k;
    n = strtol(argv[1], NULL, 0);
    assert(n > 0);
    k = strtol(argv[2], NULL, 0);
    assert(k > 0 && k <= n);

    string mode(argv[3]);

    ifstream matrix_file;
    matrix_file.open("matrix");
    complexd *transform_matrix = read_transform_matrix(matrix_file);
    matrix_file.close();

    if (size > (1ll << n))
        size = 1ll << n;

    if (rank >= size)
        return MPI_Finalize();

    long long vec_size = (1ll << n) / size;

    int batch_num = 1 << (n - k);
    int batch_size = size / batch_num;

    long long offset1, offset2;
    if (batch_size <= 1) {
        offset1 = (1ll << k) * rank * batch_num / size;
        offset2 = offset1 + vec_size / 2;
    } else {
        offset1 = (1ll << k) * (rank / batch_size) + (rank % batch_size) * vec_size / 2;
        offset2 = offset1 + (1ll << (k - 1));
    }
    
    complexd *quantum_vector;

    if (mode == "file") {
        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, "vector", MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        quantum_vector = read_quantum_vector(fh, vec_size, offset1, offset2); 
    }

    double time_start = MPI_Wtime();

    if (mode != "file") {
        quantum_vector = make_quantum_vector(vec_size, rank);        
    }

    make_single_qubit_transform(quantum_vector, transform_matrix, vec_size, k);
    MPI_Barrier(MPI_COMM_WORLD);

    double time_end = MPI_Wtime();

    if (rank == 0)
        cout << time_end - time_start << endl;

    if (rank == 0) {
        ofstream file;
        file.open("stats", ios::out | ios::app);
        file << time_end - time_start << endl;
        file.close();
    }

    string out_file_name = "out_vector";
    if (argc == 5)
        out_file_name = string(argv[4]); 

    // MPI_File fh;
    // MPI_File_open(MPI_COMM_WORLD, "out_vector_mpi", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    // write_quantum_vector(quantum_vector, fh, vec_size, offset1, offset2);

    return MPI_Finalize();
}