#include "kim_filter.h"

using namespace std;

void Kim_Filter_Deriv4_dPf(double *Pf, int n) {
    // defining the constants
    double gf10 = 0.7311329755609861;
    double gf20 = 0.1681680891936087;

    double gf01 = 0.3412746505356879;
    double gf21 = 0.6591595540319565;

    double gf02 = 0.2351300295562464;
    double gf12 = 0.6689728401317021;
    double gf13 = 0.1959510121583215;
    double gf23 = 0.6591595540319565;
    double gf24 = 0.1681680891936087;

    double alphaf = 0.6651452077642562;
    double betaf = 0.1669709584471488;

    // Create an nxn matrix with all elements initialized to zero

    // Set the diagonal values of the matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // Main diagonal
                Pf[i * n + j] = 1;
            } else if (i == j + 1 || i == j - 1) {
                // Adjacent diagonals
                Pf[i * n + j] = alphaf;
            } else if (i == j + 2 || i == j - 2) {
                // Adjacent super-diagonals
                Pf[i * n + j] = betaf;
            } else {
                Pf[i * n + j] = 0;
            }
        }
    }

    Pf[0 * n + 1] = gf01;
    Pf[0 * n + 2] = gf02;

    Pf[1 * n + 0] = gf10;
    Pf[1 * n + 2] = gf12;
    Pf[1 * n + 3] = gf13;

    Pf[2 * n + 0] = gf20;
    Pf[2 * n + 1] = gf21;
    Pf[2 * n + 3] = gf23;
    Pf[2 * n + 4] = gf24;

    Pf[(n - 3) * n + (n - 5)] = gf24;
    Pf[(n - 3) * n + (n - 4)] = gf23;
    Pf[(n - 3) * n + (n - 2)] = gf21;
    Pf[(n - 3) * n + (n - 1)] = gf20;

    Pf[(n - 2) * n + (n - 4)] = gf13;
    Pf[(n - 2) * n + (n - 3)] = gf12;
    Pf[(n - 2) * n + (n - 1)] = gf10;

    Pf[(n - 1) * n + (n - 3)] = gf02;
    Pf[(n - 1) * n + (n - 2)] = gf01;
}

void Kim_Filter_Deriv4_dQf(double *Qf, int n) {
    // defining some important variables
    double bf20 = -2.81516723801634e-4;
    double bf21 = 1.40758361900817e-3;

    double bf23 = 2.81516723801634e-3;

    double bf24 = -1.40758361900817e-3;
    double bf25 = 2.81516723801634e-4;

    double alphaf1 = 8.558206326059179e-4;
    double alphaf2 = -3.423282530423672e-4;
    double alphaf3 = 5.705470884039454e-5;

    double bf22 = -(bf20 + bf21 + bf23 + bf24 + bf25);
    double alphaf0 = -2 * (alphaf1 + alphaf2 + alphaf3);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // Main diagonal
                Qf[i * n + j] = alphaf0;
            } else if (i == j - 1) {
                // Adjacent diagonals
                Qf[i * n + j] = alphaf1;
            } else if (i == j + 1) {
                // Adjacent diagonals
                Qf[i * n + j] = alphaf1;
            } else if (i == j - 2) {
                // Super diagonals
                Qf[i * n + j] = alphaf2;
            } else if (i == j + 2) {
                // Super diagonals on
                Qf[i * n + j] = alphaf2;
            } else if (i == j - 3) {
                // Super +1 diagonals on
                Qf[i * n + j] = alphaf3;
            } else if (i == j + 3) {
                // Super +1 diagonals on
                Qf[i * n + j] = alphaf3;
            } else {
                Qf[i * n + j] = 0.0;
            }
        }
    }

    Qf[0 * n + 0] = 0;
    Qf[0 * n + 1] = 0;
    Qf[0 * n + 2] = 0;
    Qf[0 * n + 3] = 0;
    Qf[0 * n + 4] = 0;
    Qf[0 * n + 5] = 0;
    Qf[0 * n + 6] = 0;

    Qf[1 * n + 0] = 0;
    Qf[1 * n + 1] = 0;
    Qf[1 * n + 2] = 0;
    Qf[1 * n + 3] = 0;
    Qf[1 * n + 4] = 0;
    Qf[1 * n + 5] = 0;
    Qf[1 * n + 6] = 0;

    Qf[2 * n + 0] = bf20;
    Qf[2 * n + 1] = bf21;
    Qf[2 * n + 2] = bf22;
    Qf[2 * n + 3] = bf23;
    Qf[2 * n + 4] = bf24;
    Qf[2 * n + 5] = bf25;

    Qf[(n - 3) * n + (n - 6)] = bf25;
    Qf[(n - 3) * n + (n - 5)] = bf24;
    Qf[(n - 3) * n + (n - 4)] = bf23;
    Qf[(n - 3) * n + (n - 3)] = bf22;
    Qf[(n - 3) * n + (n - 2)] = bf21;
    Qf[(n - 3) * n + (n - 1)] = bf20;

    Qf[(n - 2) * n + (n - 7)] = 0.0;
    Qf[(n - 2) * n + (n - 6)] = 0.0;
    Qf[(n - 2) * n + (n - 5)] = 0.0;
    Qf[(n - 2) * n + (n - 4)] = 0.0;
    Qf[(n - 2) * n + (n - 3)] = 0.0;
    Qf[(n - 2) * n + (n - 2)] = 0.0;
    Qf[(n - 2) * n + (n - 1)] = 0.0;

    Qf[(n - 1) * n + (n - 7)] = 0.0;
    Qf[(n - 1) * n + (n - 6)] = 0.0;
    Qf[(n - 1) * n + (n - 5)] = 0.0;
    Qf[(n - 1) * n + (n - 4)] = 0.0;
    Qf[(n - 1) * n + (n - 3)] = 0.0;
    Qf[(n - 1) * n + (n - 2)] = 0.0;
    Qf[(n - 1) * n + (n - 1)] = 0.0;
}
bool initKim_Filter_Deriv4(double *RF, const unsigned int n) {
    double *Pf = new double[n * n];
    double *Qf = new double[n * n];
    Kim_Filter_Deriv4_dPf(
        Pf, n);  // define the matrix Pf using the function provided

    // Define the matrix Qf
    Kim_Filter_Deriv4_dQf(Qf, n);

    // Compute the LU decomposition of the matrix Pf
    int *ipiv = new int[n];
    int info;
    int nx = n;  // lapack needs fortran-compatible ints, not const unsigned
    dgetrf_(&nx, &nx, Pf, &nx, ipiv, &info);

    if (info != 0) {
        std::cerr << "LU factorization failed: " << info << std::endl;
        delete[] ipiv;
        return 1;
    }

    // Compute the inverse of the matrix Pf
    double *Pfinv = new double[n * n];
    std::memcpy(Pfinv, Pf, n * n * sizeof(double));
    int lwork = n * n;
    double *work = new double[lwork];
    dgetri_(&nx, Pfinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        std::cerr << "Matrix inversion failed: " << info << std::endl;
        delete[] ipiv;
        delete[] Pfinv;
        delete[] work;
        return 1;
    }

    // Compute the product of the inverted matrix Pinv and matrix Qf
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            RF[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                RF[i * n + j] += Pfinv[i * n + k] * Qf[k * n + j];
            }
        }
    }

    delete[] ipiv;
    delete[] Pfinv;
    delete[] work;
    delete[] Pf;
    delete[] Qf;

    return 0;
}
