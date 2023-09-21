#pragma once

#include <cmath>
#include <cstring>
#include <iostream>

void JTPDeriv6_dP(double *P, int n);
void JTPDeriv6_dQ(double *Q, int n);
bool initJTPDeriv6(double *R, const unsigned int n);

extern "C" {
// LU decomposition of a general matrix
void dgetrf_(int *n, int *m, double *P, int *lda, int *IPIV, int *INFO);

// generate inverse of a matrix given its LU decomposition
void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork,
             int *INFO);
}
