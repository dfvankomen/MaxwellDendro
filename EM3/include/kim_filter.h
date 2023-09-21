#pragma once

#include <cmath>
#include <iostream>
#include <cstring>

bool initKim_Filter_Deriv4(double *RF, const unsigned int n);

extern "C"
{
    // LU decomposition of a general matrix
    void dgetrf_(int *n, int *m, double *Pf, int *lda, int *IPIV, int *INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int *N, double *Qf, int *lda, int *IPIV, double *WORK, int *lwork, int *INFO);
}

