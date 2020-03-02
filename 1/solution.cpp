#include <complex>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

typedef std::complex<double> complexd;

vector<complexd> make_quantum_vector(int n)
{
    srand(unsigned(time(0)));
    long long vector_size = 1ll << n;

    vector<complexd> quantum_vector(vector_size);
    double vec_len = 0;

    #pragma omp parallel for schedule(guided) reduction(+:vec_len)
    for (long long i = 0; i < vector_size; i++) {
        unsigned int seed = omp_get_wtime() + i;
        complexd rand_val = complexd(double(rand_r(&seed)) / RAND_MAX, double(rand_r(&seed)) / RAND_MAX);
        vec_len += norm(rand_val);
        quantum_vector[i] = rand_val;
    }
    vec_len = sqrt(vec_len);

    #pragma omp parallel for schedule(guided)
    for (long long i = 0; i < vector_size; i++) {
        quantum_vector[i] /= vec_len;
    }
    return quantum_vector;
}

vector<complexd> read_transform_matrix(ifstream &file)
{
    vector<complexd> transform_matrix(4);
    for (int i = 0; i < 4; i++) {
        file >> transform_matrix[i];
    }
    return transform_matrix;
}

void make_single_qubit_transform(vector<complexd> &quantum_vector, 
        vector<complexd> &transform_matrix, int k)
{
    long long k_qubit_num = 1ll << (k - 1);

    #pragma omp parallel for schedule(guided)
    for (unsigned long long i = 0; i < quantum_vector.size(); i += k_qubit_num * 2) {
        for (unsigned long long j = i; j < i + k_qubit_num; j++) {
            complexd tmp1 = quantum_vector[j];
            complexd tmp2 = quantum_vector[j + k_qubit_num];
            quantum_vector[j] = transform_matrix[0] * tmp1 + transform_matrix[1] * tmp2;
            quantum_vector[j + k_qubit_num] = transform_matrix[2] * tmp1 + transform_matrix[3] * tmp2;
        }
    }
}

int main(int argc, char const *argv[])
{
    int n, k, thread_num;
    n = strtol(argv[1], NULL, 0);
    k = strtol(argv[2], NULL, 0);
    thread_num = strtol(argv[3], NULL, 0);
    omp_set_num_threads(thread_num);

    if (k <= 0 || k > n)
        return 1;

    ifstream matrix_file;
    matrix_file.open("matrix");
    vector<complexd> transform_matrix = read_transform_matrix(matrix_file);
    matrix_file.close();

    double time_start = omp_get_wtime();
    vector<complexd> quantum_vector = make_quantum_vector(n);

    // for (unsigned long long i = 0; i < quantum_vector.size(); i++)
    //     cout << quantum_vector[i] << ' ';
    // cout << endl;

    make_single_qubit_transform(quantum_vector, transform_matrix, k);
    double time_end = omp_get_wtime();

    // for (unsigned long long i = 0; i < quantum_vector.size(); i++)
    //     cout << quantum_vector[i] << ' ';
    // cout << endl;

    // cout << time_end - time_start << endl;

    ofstream file;
    file.open("stats", ios::out | ios::app);
    file << time_end - time_start << endl;
    file.close();

    return 0;
}