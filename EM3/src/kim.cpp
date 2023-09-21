#include "kim.h"

#include "compact_derivs.h"

using namespace std;

// THESE ARE THE OLDER KIM DERIVATIVES

void KimDeriv4_dP(double *P, int n) {
    double *tempP = new double[n * n];
    // defining the constants
    double g10 = 0.11737546726594537;
    double g20 = -0.067477420334188354;
    double g01 = 9.279310823736082;
    double g21 = -0.1945509344676567;
    double g02 = 9.8711877434133051;
    double g12 = 0.92895849448052303;
    double g13 = -0.067839996199150834;
    double g23 = 1.279565347145571;
    double g24 = 0.20842348769505742;

    double alpha = 0.5862704032801503;
    double beta = 0.09549533555017055;

    // Set the diagonal values of the matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // Main diagonal
                tempP[i * n + j] = 1;
                // tempP[INDEX_2D(i, j)] = 1.0;
            } else if (i == j + 1 || i == j - 1) {
                // Adjacent diagonals
                tempP[i * n + j] = alpha;
                // tempP[INDEX_2D(i, j)] = alpha;
            } else if (i == j + 2 || i == j - 2) {
                // Adjacent super-diagonals
                tempP[i * n + j] = beta;
                // tempP[INDEX_2D(i, j)] = beta;
            } else {
                tempP[i * n + j] = 0;
                // tempP[INDEX_2D(i, j)] = 0.0;
            }
        }
    }
    tempP[0 * n + 1] = g01;
    tempP[0 * n + 2] = g02;
    // tempP[INDEX_2D(0, 1)] = g01;
    // tempP[INDEX_2D(0, 2)] = g02;

    tempP[1 * n + 0] = g10;
    tempP[1 * n + 2] = g12;
    tempP[1 * n + 3] = g13;
    // tempP[INDEX_2D(1, 0)] = g10;
    // tempP[INDEX_2D(1, 2)] = g12;
    // tempP[INDEX_2D(1, 3)] = g13;

    tempP[2 * n + 0] = g20;
    tempP[2 * n + 1] = g21;
    tempP[2 * n + 3] = g23;
    tempP[2 * n + 4] = g24;
    // tempP[INDEX_2D(2, 0)] = g20;
    // tempP[INDEX_2D(2, 1)] = g21;
    // tempP[INDEX_2D(2, 3)] = g23;
    // tempP[INDEX_2D(2, 3)] = g24;

    tempP[(n - 3) * n + (n - 5)] = g24;
    tempP[(n - 3) * n + (n - 4)] = g23;
    tempP[(n - 3) * n + (n - 2)] = g21;
    tempP[(n - 3) * n + (n - 1)] = g20;
    // tempP[INDEX_2D(n - 3, n - 5)] = g24;
    // tempP[INDEX_2D(n - 3, n - 4)] = g23;
    // tempP[INDEX_2D(n - 3, n - 2)] = g21;
    // tempP[INDEX_2D(n - 3, n - 1)] = g20;

    tempP[(n - 2) * n + (n - 4)] = g13;
    tempP[(n - 2) * n + (n - 3)] = g12;
    tempP[(n - 2) * n + (n - 1)] = g10;
    // tempP[INDEX_2D(n - 2, n - 4)] = g13;
    // tempP[INDEX_2D(n - 2, n - 3)] = g12;
    // tempP[INDEX_2D(n - 2, n - 1)] = g10;

    tempP[(n - 1) * n + (n - 3)] = g02;
    tempP[(n - 1) * n + (n - 2)] = g01;
    // tempP[INDEX_2D(n - 1, n - 3)] = g02;
    // tempP[INDEX_2D(n - 1, n - 2)] = g01;

    // compute the transpose because i need to test this
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            P[INDEX_2D(i, j)] = tempP[INDEX_2D(j,i)];
        }
    }

    delete[] tempP;
}

void KimDeriv4_dQ(double *Q, int n) {

    double *tempQ = new double[n * n];

    // defining some important variables
    double b10 = -0.4197688256685424;
    double b20 = 0.20875393530974462;

    double b01 = -9.9196421679170452;
    double b21 = -0.36722447739446801;

    double b02 = 10.088151775649886;
    double b12 = 1.1593253854830003;

    double b03 = 4.1769460418803268;
    double b13 = 0.31685797023808876;
    double b23 = 0.98917602108458036;

    double b04 = -0.82222305192207212;
    double b14 = -0.096453054902842381;
    double b24 = 0.63518969715000262;

    double b05 = 0.14757709267988142;
    double b15 = 0.015579947274307879;
    double b25 = 0.0042145635666246068;

    double b06 = -0.014332365879513103;
    double b16 = -0.0014553614585464077;
    double b26 = 0.0010111910030585999;

    double a1 = 0.6431406736919156;
    double a2 = 0.2586011023495066;
    double a3 = 0.007140953479797375;

    double b00 = -(b01 + b02 + b03 + b04 + b05 + b06);
    double b11 = -(b10 + b12 + b13 + b14 + b15 + b16);
    double b22 = -(b20 + b21 + b23 + b24 + b25 + b26);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                // i * n + j
                // Main diagonal
                tempQ[i * n + j] = 0.0;
                // tempQ[INDEX_2D(i, j)] = 0.0;
            } else if (i == j - 1) {
                // Adjacent diagonals
                tempQ[i * n + j] = a1;
                // tempQ[INDEX_2D(i, j)] = a1;
            } else if (i == j + 1) {
                // Adjacent diagonals
                tempQ[i * n + j] = -a1;
                // tempQ[INDEX_2D(i, j)] = -a1;
            } else if (i == j - 2) {
                // Super diagonals
                tempQ[i * n + j] = a2;
                // tempQ[INDEX_2D(i, j)] = a2;
            } else if (i == j + 2) {
                // Super diagonals on
                tempQ[i * n + j] = -a2;
                // tempQ[INDEX_2D(i, j)] = -a2;
            } else if (i == j - 3) {
                // Super +1 diagonals on
                tempQ[i * n + j] = a3;
                // tempQ[INDEX_2D(i, j)] = a3;
            } else if (i == j + 3) {
                // Super +1 diagonals on
                tempQ[i * n + j] = -a3;
                // tempQ[INDEX_2D(i, j)] = -a3;
            } else {
                tempQ[i * n + j] = 0.0;
                // tempQ[INDEX_2D(i, j)] = 0.0;

            }
        }
    }

    tempQ[0 * n + 0] = b00;
    tempQ[0 * n + 1] = b01;
    tempQ[0 * n + 2] = b02;
    tempQ[0 * n + 3] = b03;
    tempQ[0 * n + 4] = b04;
    tempQ[0 * n + 5] = b05;
    tempQ[0 * n + 6] = b06;
    // tempQ[INDEX_2D(0, 0)] = b00;
    // tempQ[INDEX_2D(0, 1)] = b01;
    // tempQ[INDEX_2D(0, 2)] = b02;
    // tempQ[INDEX_2D(0, 3)] = b03;
    // tempQ[INDEX_2D(0, 4)] = b04;
    // tempQ[INDEX_2D(0, 5)] = b05;
    // tempQ[INDEX_2D(0, 6)] = b06;

    tempQ[1 * n + 0] = b10;
    tempQ[1 * n + 1] = b11;
    tempQ[1 * n + 2] = b12;
    tempQ[1 * n + 3] = b13;
    tempQ[1 * n + 4] = b14;
    tempQ[1 * n + 5] = b15;
    tempQ[1 * n + 6] = b16;
    // tempQ[INDEX_2D(1, 0)] = b10;
    // tempQ[INDEX_2D(1, 1)] = b11;
    // tempQ[INDEX_2D(1, 2)] = b12;
    // tempQ[INDEX_2D(1, 3)] = b13;
    // tempQ[INDEX_2D(1, 4)] = b14;
    // tempQ[INDEX_2D(1, 5)] = b15;
    // tempQ[INDEX_2D(1, 6)] = b16;

    tempQ[2 * n + 0] = b20;
    tempQ[2 * n + 1] = b21;
    tempQ[2 * n + 2] = b22;
    tempQ[2 * n + 3] = b23;
    tempQ[2 * n + 4] = b24;
    tempQ[2 * n + 5] = b25;
    tempQ[2 * n + 6] = b26;
    // tempQ[INDEX_2D(2, 0)] = b20;
    // tempQ[INDEX_2D(2, 1)] = b21;
    // tempQ[INDEX_2D(2, 2)] = b22;
    // tempQ[INDEX_2D(2, 3)] = b23;
    // tempQ[INDEX_2D(2, 4)] = b24;
    // tempQ[INDEX_2D(2, 5)] = b25;
    // tempQ[INDEX_2D(2, 6)] = b26;

    tempQ[(n - 3) * n + (n - 7)] = -b26;
    tempQ[(n - 3) * n + (n - 6)] = -b25;
    tempQ[(n - 3) * n + (n - 5)] = -b24;
    tempQ[(n - 3) * n + (n - 4)] = -b23;
    tempQ[(n - 3) * n + (n - 3)] = -b22;
    tempQ[(n - 3) * n + (n - 2)] = -b21;
    tempQ[(n - 3) * n + (n - 1)] = -b20;
    // tempQ[INDEX_2D(n - 3, n - 7)] = -b26;
    // tempQ[INDEX_2D(n - 3, n - 6)] = -b25;
    // tempQ[INDEX_2D(n - 3, n - 5)] = -b24;
    // tempQ[INDEX_2D(n - 3, n - 4)] = -b23;
    // tempQ[INDEX_2D(n - 3, n - 3)] = -b22;
    // tempQ[INDEX_2D(n - 3, n - 2)] = -b21;
    // tempQ[INDEX_2D(n - 3, n - 1)] = -b20;

    tempQ[(n - 2) * n + (n - 7)] = -b16;
    tempQ[(n - 2) * n + (n - 6)] = -b15;
    tempQ[(n - 2) * n + (n - 5)] = -b14;
    tempQ[(n - 2) * n + (n - 4)] = -b13;
    tempQ[(n - 2) * n + (n - 3)] = -b12;
    tempQ[(n - 2) * n + (n - 2)] = -b11;
    tempQ[(n - 2) * n + (n - 1)] = -b10;
    // tempQ[INDEX_2D(n-2, n-7)] = -b16;
    // tempQ[INDEX_2D(n-2, n-6)] = -b15;
    // tempQ[INDEX_2D(n-2, n-5)] = -b14;
    // tempQ[INDEX_2D(n-2, n-4)] = -b13;
    // tempQ[INDEX_2D(n-2, n-3)] = -b12;
    // tempQ[INDEX_2D(n-2, n-2)] = -b11;
    // tempQ[INDEX_2D(n-2, n-1)] = -b10;

    tempQ[(n - 1) * n + (n - 7)] = -b06;
    tempQ[(n - 1) * n + (n - 6)] = -b05;
    tempQ[(n - 1) * n + (n - 5)] = -b04;
    tempQ[(n - 1) * n + (n - 4)] = -b03;
    tempQ[(n - 1) * n + (n - 3)] = -b02;
    tempQ[(n - 1) * n + (n - 2)] = -b01;
    tempQ[(n - 1) * n + (n - 1)] = -b00;
    // tempQ[INDEX_2D(n-1, n-7)] = -b06;
    // tempQ[INDEX_2D(n-1, n-6)] = -b05;
    // tempQ[INDEX_2D(n-1, n-5)] = -b04;
    // tempQ[INDEX_2D(n-1, n-4)] = -b03;
    // tempQ[INDEX_2D(n-1, n-3)] = -b02;
    // tempQ[INDEX_2D(n-1, n-2)] = -b01;
    // tempQ[(INDEX_2D(n-1, n-1)] = -b00;

    // compute the transpose because i need to test this
    // TODO: fix the actual population
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[INDEX_2D(i, j)] = tempQ[INDEX_2D(j,i)];
        }
    }

    delete[] tempQ;

}

bool initKimDeriv4(double *R, const unsigned int n) {
    double *P = new double[n * n];
    double *Q = new double[n * n];
    KimDeriv4_dP(P, n);  // define the matrix P using the function provided

    // Define the matrix Q
    KimDeriv4_dQ(Q, n);

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
