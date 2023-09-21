#pragma once

#include <libxsmm.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "HAMR.h"
#include "JTP6.h"
#include "dendro.h"
#include "kim.h"
#include "kim_filter.h"

#define INDEX_3D(i, j, k) ((i) + nx * ((j) + ny * (k)))

#define INDEX_2D(i, j) ((i) + n * (j))

#define INDEX_N2D(i, j, n) ((i) + (n) * (j))

extern "C" {
// LU decomposition of a general matrix
void dgetrf_(int *n, int *m, double *P, int *lda, int *IPIV, int *INFO);

// generate inverse of a matrix given its LU decomposition
void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork,
             int *INFO);

// multiplies two matrices C = alpha*A*B + beta*C
void dgemm_(char *TA, char *TB, int *M, int *N, int *K, double *ALPHA,
            double *A, int *LDA, double *B, int *LDB, double *BETA, double *C,
            int *LDC);

// generic matrix vector multiplication.
extern "C" void dgemv_(char *trans, int *m, int *n, double *alpha, double *A,
                       int *lda, double *x, int *incx, double *beta, double *y,
                       int *incy);
}

namespace dendro_cfd {

// enum DerType { CFD_P1_O4 = 0, CFD_P1_O6, CFD_Q1_O6_ETA1 };

enum DerType {
    // NO CFD Initialization
    CFD_NONE = -1,

    // the "main" compact finite difference types
    CFD_P1_O4 = 0,
    CFD_P1_O6,
    CFD_Q1_O6_ETA1,
    // isotropic finite difference types
    CFD_KIM_O4,
    CFD_HAMR_O4,
    CFD_JT_O6,
    // additional "helpers" that are mostly for internal/edge building
    CFD_DRCHLT_ORDER_4,
    CFD_DRCHLT_ORDER_6,
    CFD_P1_O4_CLOSE,
    CFD_P1_O6_CLOSE,
    CFD_P1_O4_L4_CLOSE,
    CFD_P1_O6_L6_CLOSE,
    CFD_Q1_O6,
    CFD_Q1_O6_CLOSE,
    CFD_DRCHLT_Q6,
    CFD_DRCHLT_Q6_L6,
    CFD_Q1_O6_ETA1_CLOSE,

};

enum FilterType {
    // NO CFD Initialization
    FILT_NONE = -1,

    // standard filters...
    FILT_KO_DISS = 0,

    // isotropic finite difference types
    FILT_KIM_6,
    FILT_JT_6,
    FILT_JT_8,
   
};

// NOTE: these are going to be used as global parameters if they're not physical
enum BoundaryType {
    BLOCK_CFD_CLOSURE = 0,  // closure gives better results but 6th requires 4
                            // points, and 4th requires 3 points
    BLOCK_CFD_DIRICHLET,
    BLOCK_CFD_LOPSIDE_CLOSURE,
    BLOCK_PHYS_BOUNDARY
};

class CFDMethod {
   public:
    DerType name;
    uint32_t order;
    int32_t Ld;
    int32_t Rd;
    int32_t Lf;
    int32_t Rf;
    double alpha[16];
    double a[16];

    CFDMethod(DerType dertype) {
        switch (dertype) {
            case CFD_P1_O4:
                set_for_CFD_P1_O4();
                break;

            case CFD_P1_O6:
                set_for_CFD_P1_O6();
                break;

            case CFD_Q1_O6_ETA1:
                set_for_CFD_Q1_O6_ETA1();
                break;

            default:
                throw std::invalid_argument(
                    "Invalid CFD method type of " + std::to_string(dertype) +
                    " for initializing the CFDMethod Object");
                break;
        }
    }

    ~CFDMethod() {}

    void set_for_CFD_P1_O4() {
        name = CFD_P1_O4;
        order = 4;
        Ld = 1;
        Rd = 1;
        Lf = 1;
        Rf = 1;
        alpha[0] = 0.25;
        alpha[1] = 1.0;
        alpha[2] = 0.25;

        a[0] = -0.75;
        a[1] = 0.0;
        a[2] = 0.75;
    }

    void set_for_CFD_P1_O6() {
        name = CFD_P1_O6;
        order = 6;
        Ld = 1;
        Rd = 1;
        Lf = 2;
        Rf = 2;

        alpha[0] = 1.0 / 3.0;
        alpha[1] = 1.0;
        alpha[2] = 1.0 / 3.0;

        const double t1 = 1.0 / 36.0;
        a[0] = -t1;
        a[1] = -28.0 * t1;
        a[2] = 0.0;
        a[3] = 28.0 * t1;
        a[4] = t1;
    }

    void set_for_CFD_Q1_O6_ETA1() {
        name = CFD_Q1_O6_ETA1;
        order = 6;
        Ld = 1;
        Rd = 1;
        Lf = 3;
        Rf = 3;

        alpha[0] = 0.37987923;
        alpha[1] = 1.0;
        alpha[2] = 0.37987923;

        a[0] = 0.0023272948;
        a[1] = -0.052602255;
        a[2] = -0.78165660;
        a[3] = 0.0;
        a[4] = 0.78165660;
        a[5] = 0.052602255;
        a[6] = -0.0023272948;
    }
};

void print_square_mat(double *m, const uint32_t n);

DerType getDerTypeForEdges(const DerType derivtype,
                           const BoundaryType boundary);

void buildPandQMatrices(double *P, double *Q, const uint32_t padding,
                        const uint32_t n, const DerType derivtype,
                        const bool is_left_edge = false,
                        const bool is_right_edge = false);

void buildPandQFilterMatrices(double *P, double *Q, const uint32_t padding,
                              const uint32_t n, const FilterType derivtype,
                              const bool is_left_edge = false,
                              const bool is_right_edge = false);

void buildMatrixLeft(double *P, double *Q, int *xib, const DerType dtype,
                     const int nghosts, const int n);

void buildMatrixRight(double *P, double *Q, int *xie, const DerType dtype,
                      const int nghosts, const int n);

void calculateDerivMatrix(double *D, double *P, double *Q, const int n);

void setArrToZero(double *Mat, const int n);

/*
 Computes
     C := alpha*op( A )*op( B ) + beta*C,
*/
void mulMM(double *C, double *A, double *B, int na, int nb);

class CompactFiniteDiff {
   private:
    // STORAGE VARIABLES USED FOR THE DIFFERENT DIMENSIONS
    // Assume that the blocks are all the same size (to start with)
    double *m_R_filter = nullptr;
    double *m_R = nullptr;

    // we also need "left" physical edge, and "right" physical edge
    // and "both" physical edges to be safe
    double *m_R_left = nullptr;
    double *m_R_right = nullptr;
    double *m_R_leftright = nullptr;

    double *m_R_filter_left = nullptr;
    double *m_R_filter_right = nullptr;
    double *m_R_filter_leftright = nullptr;

    // TODO: we're going to want to store the filter and R variables as hash
    // maps
    // // Storage for the R matrix operator (combined P and Q matrices in CFD)
    // std::map<uint32_t, double *> m_R_storage;
    // // Storage for the RF matrix operator (the filter matrix)
    // std::map<uint32_t, double *> m_RF_storage;

    // Temporary storage for operations in progress
    double *m_u1d = nullptr;
    double *m_u2d = nullptr;
    // Additional temporary storage for operations in progress
    double *m_du1d = nullptr;
    double *m_du2d = nullptr;

    // to check for initialization (not used)
    bool m_initialized_matrices = false;

    // storing the derivative and filter types internally
    // could just be the parameter types
    DerType m_deriv_type = CFD_KIM_O4;
    FilterType m_filter_type = FILT_NONE;
    unsigned int m_curr_dim_size = 0;
    unsigned int m_padding_size = 0;

   public:
    CompactFiniteDiff(const unsigned int dim_size,
                      const unsigned int padding_size,
                      const DerType deriv_type = CFD_KIM_O4,
                      const FilterType filter_type = FILT_NONE);
    ~CompactFiniteDiff();

    void change_dim_size(const unsigned int dim_size);

    void initialize_cfd_storage();
    void initialize_cfd_matrix();
    void initialize_cfd_filter();
    void delete_cfd_matrices();

    void set_filter_type(FilterType filter_type) {
        m_filter_type = filter_type;
    }

    void set_deriv_type(const DerType deriv_type) {
        if (deriv_type != CFD_NONE && deriv_type != CFD_P1_O4 &&
            deriv_type != CFD_P1_O6 && deriv_type != CFD_Q1_O6_ETA1 &&
            deriv_type != CFD_KIM_O4 && deriv_type != CFD_HAMR_O4 &&
            deriv_type != CFD_JT_O6) {
            throw std::invalid_argument(
                "Couldn't change deriv type of CFD object, deriv type was not "
                "a valid "
                "'base' "
                "type: deriv_type = " +
                std::to_string(deriv_type));
        }
        m_deriv_type = deriv_type;
    }

    /**
     * Sets the padding size. NOTE however that this does *not* attempt to
     * regenerate the matrices, so be sure to call the initialization
     */
    void set_padding_size(const unsigned int padding_size) {
        m_padding_size = padding_size;
    }

    // the actual derivative computation side of things
    void cfd_x(double *const Dxu, const double *const u, const double dx,
               const unsigned int *sz, unsigned bflag);
    void cfd_y(double *const Dyu, const double *const u, const double dy,
               const unsigned int *sz, unsigned bflag);
    void cfd_z(double *const Dzu, const double *const u, const double dz,
               const unsigned int *sz, unsigned bflag);

    // then the actual filters
    void filter_cfd_x(double *const u, double *const filtx_work,
                      const double dx, const unsigned int *sz, unsigned bflag);
    void filter_cfd_y(double *const u, double *const filty_work,
                      const double dy, const unsigned int *sz, unsigned bflag);
    void filter_cfd_z(double *const u, double *const filtz_work,
                      const double dz, const unsigned int *sz, unsigned bflag);
};

extern CompactFiniteDiff cfd;

/**
 * Initialization of various Compact Methods
 *
 * From this point on various compact finite methods can be calculated and
 * derived
 */

/**
 * Initialization of the P and Q matrices for Kim's 4th Order Compact
 * Derivatives
 *
 * P and Q are assumed to **already by zeroed out**.
 *
 * These derivative coefficients come from Tables I and II of :
 *
 * Jae Wook Kim, "Quasi-disjoint pentadiagonal matrix systems for
 * the parallelization of compact finite-difference schemes and
 * filters," Journal of Computational Physics 241 (2013) 168–194.
 */
void initializeKim4PQ(double *P, double *Q, int n);

/**
 * Initialization of the P and Q matrices for Kim's 6th Order Compact Filter
 *
 * P and Q are assumed to **already by zeroed out**.
 *
 * These filter coefficients come from Tables III and IV of :
 *
 * Jae Wook Kim, "Quasi-disjoint pentadiagonal matrix systems for
 * the parallelization of compact finite-difference schemes and
 * filters," Journal of Computational Physics 241 (2013) 168–194.
 */
void initializeKim6FilterPQ(double *P, double *Q, int n);

}  // namespace dendro_cfd
