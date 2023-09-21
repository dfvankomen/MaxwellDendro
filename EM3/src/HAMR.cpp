#include "HAMR.h"

#include "compact_derivs.h"

using namespace std;

void HAMRDeriv4_dP(double *P, int n) {
    double *tempP = new double[n * n];
    // Define the variables
    double a00 = 1.0;
    double a10 = 0.1023343303;
    double a20 = 0.0347468867;

    double a01 = 9.4133049605;
    double a11 = 1.0;
    double a21 = 0.4064246796;

    double a02 = 10.7741034803;
    double a12 = 1.8854940182;
    double a22 = 1.0;

    double a03 = 0.0;
    double a13 = 0.8582327249;
    double a23 = 0.7683583302;

    double a04 = 0.0;
    double a14 = 0.0;
    double a24 = 0.1623349133;

    double alpha = 0.5747612151;
    double beta1 = 0.0879324249;
    // Set the diagonal values of the array
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index =
                i * n + j;  // Calculate the 1D index for the (i, j) element
            if (i == j) {
                // Main diagonal
                tempP[index] = 1.0;
            } else if (i == j + 1 || i == j - 1) {
                // Adjacent diagonals
                tempP[index] = alpha;
            } else if (i == j + 2 || i == j - 2) {
                // Adjacent super-diagonals
                tempP[index] = beta1;
            } else {
                tempP[index] = 0.0;
            }
        }
    }

    // Set specific values in the array
    tempP[1] = a01;
    tempP[2] = a02;
    tempP[3] = 0.0;

    tempP[n] = a10;
    tempP[n + 2] = a12;
    tempP[n + 3] = a13;

    tempP[2 * n] = a20;
    tempP[2 * n + 1] = a21;
    tempP[2 * n + 3] = a23;
    tempP[2 * n + 4] = a24;

    tempP[(n - 3) * n + (n - 5)] = a24;
    tempP[(n - 3) * n + (n - 4)] = a23;
    tempP[(n - 3) * n + (n - 2)] = a21;
    tempP[(n - 3) * n + (n - 1)] = a20;

    tempP[(n - 2) * n + (n - 4)] = a13;
    tempP[(n - 2) * n + (n - 3)] = a12;
    tempP[(n - 2) * n + (n - 1)] = a10;

    tempP[(n - 1) * n + (n - 3)] = a02;
    tempP[(n - 1) * n + (n - 2)] = a01;

    // compute the transpose because i need to test this
    // TODO: fix the actual population
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            P[INDEX_2D(i, j)] = tempP[INDEX_2D(j, i)];
        }
    }

    delete[] tempP;
}

void HAMRDeriv4_dQ(double *Q, int n) {
    double *tempQ = new double[n * n];
    // Seting the constants
    double a = 1.3069171114;
    double b = 0.9828406281;
    double c = 0.0356295405;

    double p10 = -0.3586079596;
    double p20 = -0.1274311870;

    double p01 = -10.3169611301;
    double p21 = -0.6299599564;

    double p02 = 9.6807767746;
    double p12 = 0.0834751059;

    double p03 = 5.4529053045;
    double p13 = 1.4235697122;
    double p23 = 0.6498856630;

    double p04 = -1.51919598290;
    double p14 = 0.2245783548;
    double p24 = 0.3919470424;

    double p05 = 0.4834876759;
    double p15 = -0.0358453729;
    double p25 = 0.0189402158;

    double p06 = -0.0927590566;
    double p16 = 0.0052970021;
    double p26 = -0.0008894789;

    // from equation 13
    double p00 = -(p01 + p02 + p03 + p04 + p05 + p06);
    double p11 = -(p10 + p12 + p13 + p14 + p15 + p16);
    double p22 = -(p20 + p21 + p23 + p24 + p25 + p26);

    // Set the diagonal values of the array
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int index =
                i * n + j;  // Calculate the 1D index for the (i, j) element
            if (i == j) {
                // Main diagonal
                tempQ[index] = 0.0;
            } else if (i == j - 1) {
                // Adjacent diagonals
                tempQ[index] = a / 2.0;
            } else if (i == j + 1) {
                // Adjacent diagonals
                tempQ[index] = -a / 2.0;
            } else if (i == j - 2) {
                // Super diagonals
                tempQ[index] = b / 4.0;
            } else if (i == j + 2) {
                // Super diagonals
                tempQ[index] = -b / 4.0;
            } else if (i == j - 3) {
                // Super +1 diagonals
                tempQ[index] = c / 6.0;
            } else if (i == j + 3) {
                // Super +1 diagonals
                tempQ[index] = -c / 6.0;
            } else {
                tempQ[index] = 0.0;
            }
        }
    }

    // Set specific values in the array
    tempQ[0] = p00;
    tempQ[1] = p01;
    tempQ[2] = p02;
    tempQ[3] = p03;
    tempQ[4] = p04;
    tempQ[5] = p05;
    tempQ[6] = p06;

    tempQ[n] = p10;
    tempQ[n + 1] = p11;
    tempQ[n + 2] = p12;
    tempQ[n + 3] = p13;
    tempQ[n + 4] = p14;
    tempQ[n + 5] = p15;
    tempQ[n + 6] = p16;

    tempQ[2 * n] = p20;
    tempQ[2 * n + 1] = p21;
    tempQ[2 * n + 2] = p22;
    tempQ[2 * n + 3] = p23;
    tempQ[2 * n + 4] = p24;
    tempQ[2 * n + 5] = p25;
    tempQ[2 * n + 6] = p26;

    tempQ[(n - 3) * n + (n - 7)] = -p26;
    tempQ[(n - 3) * n + (n - 6)] = -p25;
    tempQ[(n - 3) * n + (n - 5)] = -p24;
    tempQ[(n - 3) * n + (n - 4)] = -p23;
    tempQ[(n - 3) * n + (n - 3)] = -p22;
    tempQ[(n - 3) * n + (n - 2)] = -p21;
    tempQ[(n - 3) * n + (n - 1)] = -p20;

    tempQ[(n - 2) * n + (n - 7)] = -p16;
    tempQ[(n - 2) * n + (n - 6)] = -p15;
    tempQ[(n - 2) * n + (n - 5)] = -p14;
    tempQ[(n - 2) * n + (n - 4)] = -p13;
    tempQ[(n - 2) * n + (n - 3)] = -p12;
    tempQ[(n - 2) * n + (n - 2)] = -p11;
    tempQ[(n - 2) * n + (n - 1)] = -p10;

    tempQ[(n - 1) * n + (n - 7)] = -p06;
    tempQ[(n - 1) * n + (n - 6)] = -p05;
    tempQ[(n - 1) * n + (n - 5)] = -p04;
    tempQ[(n - 1) * n + (n - 4)] = -p03;
    tempQ[(n - 1) * n + (n - 3)] = -p02;
    tempQ[(n - 1) * n + (n - 2)] = -p01;
    tempQ[(n - 1) * n + (n - 1)] = -p00;

    // compute the transpose because i need to test this
    // TODO: fix the actual population
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[INDEX_2D(i, j)] = tempQ[INDEX_2D(j, i)];
        }
    }

    delete[] tempQ;
}

bool initHAMRDeriv4(double *R, const unsigned int n) {
    double *P = new double[n * n];
    double *Q = new double[n * n];
    HAMRDeriv4_dP(P, n);  // define the matrix P using the function provided

    // Define the matrix Q
    HAMRDeriv4_dQ(Q, n);

    // Compute the LU decomposition of the matrix P
    int *ipiv = new int[n];
    int info;
    int nx = n;  // lapack needs fortran-compatible ints, not const unsigned
    dgetrf_(&nx, &nx, P, &nx, ipiv, &info);

    if (info != 0) {
        std::cerr << "LU factorization failed: " << info << std::endl;
        delete[] ipiv;
        return 1;
    }

    // Compute the inverse of the matrix P
    double *Pinv = new double[n * n];
    std::memcpy(Pinv, P, n * n * sizeof(double));
    int lwork = n * n;
    double *work = new double[lwork];
    dgetri_(&nx, Pinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        std::cerr << "Matrix inversion failed: " << info << std::endl;
        delete[] ipiv;
        delete[] Pinv;
        delete[] work;
        return 1;
    }

    // Compute the product of the inverted matrix Pinv and matrix Q
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            R[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                R[i * n + j] += Pinv[i * n + k] * Q[k * n + j];
            }
        }
    }

    delete[] ipiv;
    delete[] Pinv;
    delete[] work;
    delete[] P;
    delete[] Q;

    return 0;
}
