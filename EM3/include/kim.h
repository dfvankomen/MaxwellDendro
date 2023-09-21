#pragma once

#include <cmath>
#include <cstring>
#include <iostream>

void KimDeriv4_dP(double *P, int n);
void KimDeriv4_dQ(double *Q, int n);
bool initKimDeriv4(double *R, const unsigned int n);

extern "C" {
// LU decomposition of a general matrix
void dgetrf_(int *n, int *m, double *P, int *lda, int *IPIV, int *INFO);

// generate inverse of a matrix given its LU decomposition
void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork,
             int *INFO);
}
