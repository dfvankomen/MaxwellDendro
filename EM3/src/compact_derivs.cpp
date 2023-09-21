#include "compact_derivs.h"

#define FASTER_DERIV_CALC_VIA_MATRIX_MULT

namespace dendro_cfd {

// initialize a "global" cfd object
CompactFiniteDiff cfd(0, 0);

CompactFiniteDiff::CompactFiniteDiff(const unsigned int num_dim,
                                     const unsigned int padding_size,
                                     const DerType deriv_type,
                                     const FilterType filter_type) {
    if (deriv_type != CFD_NONE && deriv_type != CFD_P1_O4 &&
        deriv_type != CFD_P1_O6 && deriv_type != CFD_Q1_O6_ETA1 &&
        deriv_type != CFD_KIM_O4 && deriv_type != CFD_HAMR_O4 &&
        deriv_type != CFD_JT_O6) {
        throw std::invalid_argument(
            "Couldn't initialize CFD object, deriv type was not a valid 'base' "
            "type: deriv_type = " +
            std::to_string(deriv_type));
    }

    m_deriv_type = deriv_type;
    m_filter_type = filter_type;
    m_curr_dim_size = num_dim;
    m_padding_size = padding_size;

    initialize_cfd_storage();

    if (num_dim == 0) {
        return;
    }

    if (deriv_type == CFD_NONE) {
        return;
    }

    initialize_cfd_matrix();
    initialize_cfd_filter();
}

CompactFiniteDiff::~CompactFiniteDiff() {
    // make sure we delete the cfd matrix to avoid memory leaks
    delete_cfd_matrices();
}

void CompactFiniteDiff::change_dim_size(const unsigned int dim_size) {
    if (m_curr_dim_size == dim_size) {
        return;
    } else {
        delete_cfd_matrices();

        m_curr_dim_size = dim_size;

        initialize_cfd_storage();

        // if deriv type is none, for some reason, just exit
        if (m_deriv_type == CFD_NONE) {
            return;
        }

        initialize_cfd_matrix();
        initialize_cfd_filter();
    }
}

void CompactFiniteDiff::initialize_cfd_storage() {
    // NOTE: 0 indicates that it's initialized with all elements set to 0
    m_R = new double[m_curr_dim_size * m_curr_dim_size]();
    m_R_left = new double[m_curr_dim_size * m_curr_dim_size]();
    m_R_right = new double[m_curr_dim_size * m_curr_dim_size]();
    m_R_leftright = new double[m_curr_dim_size * m_curr_dim_size]();

    m_R_filter = new double[m_curr_dim_size * m_curr_dim_size]();
    m_R_filter_left = new double[m_curr_dim_size * m_curr_dim_size]();
    m_R_filter_right = new double[m_curr_dim_size * m_curr_dim_size]();
    m_R_filter_leftright = new double[m_curr_dim_size * m_curr_dim_size]();

    // NOTE: the () syntax only works with C++ 11 or greater, may need to
    // use std::fill_n(array, n, 0); to 0 set the data or use std::memset(array,
    // 0, sizeof *array * size)

    m_u1d = new double[m_curr_dim_size];
    m_u2d = new double[m_curr_dim_size * m_curr_dim_size];
    m_du1d = new double[m_curr_dim_size];
    m_du2d = new double[m_curr_dim_size * m_curr_dim_size];
}

void CompactFiniteDiff::initialize_cfd_matrix() {
    // temporary P and Q storage used in calculations
    double *P = new double[m_curr_dim_size * m_curr_dim_size]();
    double *Q = new double[m_curr_dim_size * m_curr_dim_size]();

    // for each cfd matrix that needs to be initialized, we need the "base"
    // matrix, the "left edge" and the "right edge" to be safe.

    // TODO: it might be necessary if the processor knows what boundaries it has
    // but these matrices are small compared to the blocks that they're probably
    // alright plus, these are only calculated once and not over and over again.

    // TODO: need to build up the other three combinations (the last one is
    // likely never going to happen)
    buildPandQMatrices(P, Q, m_padding_size, m_curr_dim_size, m_deriv_type,
                       false, false);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nP MATRIX" << std::endl;
    print_square_mat(P, m_curr_dim_size);

    std::cout << "\nQ MATRIX" << std::endl;
    print_square_mat(Q, m_curr_dim_size);
#endif

    calculateDerivMatrix(m_R, P, Q, m_curr_dim_size);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nDERIV MATRIX" << std::endl;
    print_square_mat(m_R, m_curr_dim_size);
#endif

    // reset P and Q and then do the LEFT version
    setArrToZero(P, m_curr_dim_size * m_curr_dim_size);
    setArrToZero(Q, m_curr_dim_size * m_curr_dim_size);
    buildPandQMatrices(P, Q, m_padding_size, m_curr_dim_size, m_deriv_type,
                       true, false);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nLEFT P MATRIX" << std::endl;
    print_square_mat(P, m_curr_dim_size);

    std::cout << "\nLEFT Q MATRIX" << std::endl;
    print_square_mat(Q, m_curr_dim_size);
#endif

    calculateDerivMatrix(m_R_left, P, Q, m_curr_dim_size);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nLEFT DERIV MATRIX" << std::endl;
    print_square_mat(m_R_left, m_curr_dim_size);
#endif

    // reset P and Q and then do the RIGHT version
    setArrToZero(P, m_curr_dim_size * m_curr_dim_size);
    setArrToZero(Q, m_curr_dim_size * m_curr_dim_size);
    buildPandQMatrices(P, Q, m_padding_size, m_curr_dim_size, m_deriv_type,
                       false, true);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nRIGHT P MATRIX" << std::endl;
    print_square_mat(P, m_curr_dim_size);

    std::cout << "\nRIGHT Q MATRIX" << std::endl;
    print_square_mat(Q, m_curr_dim_size);
#endif

    calculateDerivMatrix(m_R_right, P, Q, m_curr_dim_size);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nRIGHT DERIV MATRIX" << std::endl;
    print_square_mat(m_R_right, m_curr_dim_size);
#endif

    // reset P and Q and then do the LEFTRIGHT version
    setArrToZero(P, m_curr_dim_size * m_curr_dim_size);
    setArrToZero(Q, m_curr_dim_size * m_curr_dim_size);
    buildPandQMatrices(P, Q, m_padding_size, m_curr_dim_size, m_deriv_type,
                       true, true);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nLEFTRIGHT P MATRIX" << std::endl;
    print_square_mat(P, m_curr_dim_size);

    std::cout << "\nLEFTRIGHT Q MATRIX" << std::endl;
    print_square_mat(Q, m_curr_dim_size);
#endif

    calculateDerivMatrix(m_R_leftright, P, Q, m_curr_dim_size);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nLEFTRIGHT DERIV MATRIX" << std::endl;
    print_square_mat(m_R_leftright, m_curr_dim_size);
#endif

    delete[] P;
    delete[] Q;
}

void CompactFiniteDiff::initialize_cfd_filter() {
    // exit early on filter none
    if (m_filter_type == FILT_NONE || m_filter_type == FILT_KO_DISS) {
        return;
    }

    // temporary P and Q storage used in calculations
    double *P = new double[m_curr_dim_size * m_curr_dim_size]();
    double *Q = new double[m_curr_dim_size * m_curr_dim_size]();

    buildPandQFilterMatrices(P, Q, m_padding_size, m_curr_dim_size,
                             m_filter_type, false, false);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nP MATRIX" << std::endl;
    print_square_mat(P, m_curr_dim_size);

    std::cout << "\nQ MATRIX" << std::endl;
    print_square_mat(Q, m_curr_dim_size);
#endif

    calculateDerivMatrix(m_R_filter, P, Q, m_curr_dim_size);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nFILTER MATRIX" << std::endl;
    print_square_mat(m_R, m_curr_dim_size);
#endif

    // reset P and Q and then do the LEFT version
    setArrToZero(P, m_curr_dim_size * m_curr_dim_size);
    setArrToZero(Q, m_curr_dim_size * m_curr_dim_size);
    buildPandQFilterMatrices(P, Q, m_padding_size, m_curr_dim_size,
                             m_filter_type, true, false);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nLEFT P MATRIX" << std::endl;
    print_square_mat(P, m_curr_dim_size);

    std::cout << "\nLEFT Q MATRIX" << std::endl;
    print_square_mat(Q, m_curr_dim_size);
#endif

    calculateDerivMatrix(m_R_left, P, Q, m_curr_dim_size);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nLEFT FILTER MATRIX" << std::endl;
    print_square_mat(m_R_left, m_curr_dim_size);
#endif

    // reset P and Q and then do the RIGHT version
    setArrToZero(P, m_curr_dim_size * m_curr_dim_size);
    setArrToZero(Q, m_curr_dim_size * m_curr_dim_size);
    buildPandQFilterMatrices(P, Q, m_padding_size, m_curr_dim_size,
                             m_filter_type, false, true);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nRIGHT P MATRIX" << std::endl;
    print_square_mat(P, m_curr_dim_size);

    std::cout << "\nRIGHT Q MATRIX" << std::endl;
    print_square_mat(Q, m_curr_dim_size);
#endif

    calculateDerivMatrix(m_R_right, P, Q, m_curr_dim_size);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nRIGHT FILTER MATRIX" << std::endl;
    print_square_mat(m_R_right, m_curr_dim_size);
#endif

    // reset P and Q and then do the LEFTRIGHT version
    setArrToZero(P, m_curr_dim_size * m_curr_dim_size);
    setArrToZero(Q, m_curr_dim_size * m_curr_dim_size);
    buildPandQFilterMatrices(P, Q, m_padding_size, m_curr_dim_size,
                             m_filter_type, true, true);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nLEFTRIGHT P MATRIX" << std::endl;
    print_square_mat(P, m_curr_dim_size);

    std::cout << "\nLEFTRIGHT Q MATRIX" << std::endl;
    print_square_mat(Q, m_curr_dim_size);
#endif

    calculateDerivMatrix(m_R_leftright, P, Q, m_curr_dim_size);

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "\nLEFTRIGHT FILTER MATRIX" << std::endl;
    print_square_mat(m_R_leftright, m_curr_dim_size);
#endif

    delete[] P;
    delete[] Q;
}

void CompactFiniteDiff::delete_cfd_matrices() {
    delete[] m_R_filter;
    delete[] m_R;
    delete[] m_u1d;
    delete[] m_u2d;
    delete[] m_du1d;
    delete[] m_du2d;

    delete[] m_R_left;
    delete[] m_R_right;
    delete[] m_R_leftright;

    delete[] m_R_filter_left;
    delete[] m_R_filter_right;
    delete[] m_R_filter_leftright;
}

void CompactFiniteDiff::cfd_x(double *const Dxu, const double *const u,
                              const double dx, const unsigned int *sz,
                              unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // std::cout << "Nx, ny, nz: " << nx << " " << ny << " " << nz << std::endl;

    char TRANSA = 'N';
    char TRANSB = 'N';

    int M = nx;
    int N = ny;
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    const double alpha = 1.0 / dx;
#else
    double alpha = 1.0 / dx;
#endif
    int K = nx;

    // NOTE: LDA, LDB, and LDC should be nx, ny, and nz
    // TODO: fix for non-square sizes
    int LDA = nx;
    int LDB = ny;
    int LDC = nx;

    double *u_curr_chunk = (double *)u;
    double *du_curr_chunk = (double *)Dxu;

    double beta = 0.0;

    double *R_mat_use = nullptr;

    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_LEFT)) && !(bflag & (1u << OCT_DIR_RIGHT))) {
        R_mat_use = m_R;
    } else if ((bflag & (1u << OCT_DIR_LEFT)) &&
               !(bflag & (1u << OCT_DIR_RIGHT))) {
        R_mat_use = m_R_left;
    } else if (!(bflag & (1u << OCT_DIR_LEFT)) &&
               (bflag & (1u << OCT_DIR_RIGHT))) {
        R_mat_use = m_R_right;
    } else {
        R_mat_use = m_R_leftright;
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, M, N, K, 1.0, 0.0);
    assert(kernel);
#endif

    // const libxsmm_mmfunction<double, double, LIBXSMM_PREFETCH_AUTO>
    // xmm(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, LDA, LDB, LDC, alpha,
    // beta);

    for (unsigned int k = 0; k < nz; k++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        // N = ny;
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        // performs C_mn = alpha * A_mk * B_kn + beta * C_mn

        // for the x_der case, m = k = nx

        kernel(R_mat_use, u_curr_chunk, du_curr_chunk);

#else

        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R_mat_use, &LDA,
               u_curr_chunk, &LDB, &beta, du_curr_chunk, &LDC);

#endif

        u_curr_chunk += nx * ny;
        du_curr_chunk += nx * ny;
    }

    // TODO: investigate why the kernel won't take 1/dx as its alpha
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dxu[ii] *= 1 / dx;
    }
#endif
}

void CompactFiniteDiff::cfd_y(double *const Dyu, const double *const u,
                              const double dy, const unsigned int *sz,
                              unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    char TRANSA = 'N';
    char TRANSB = 'T';
    int M = ny;
    int N = nx;
    int K = ny;

    double alpha = 1.0 / dy;
    double beta = 0.0;

    double *u_curr_chunk = (double *)u;
    double *du_curr_chunk = (double *)Dyu;

    double *R_mat_use = nullptr;
    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_DOWN)) && !(bflag & (1u << OCT_DIR_UP))) {
        R_mat_use = m_R;
    } else if ((bflag & (1u << OCT_DIR_DOWN)) &&
               !(bflag & (1u << OCT_DIR_UP))) {
        R_mat_use = m_R_left;
    } else if (!(bflag & (1u << OCT_DIR_DOWN)) &&
               (bflag & (1u << OCT_DIR_UP))) {
        R_mat_use = m_R_right;
    } else {
        R_mat_use = m_R_leftright;
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_TRANS_B, M, N, K, 1.0, 0.0);
    assert(kernel);
#endif

    for (unsigned int k = 0; k < nz; k++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        kernel(R_mat_use, u_curr_chunk, m_du2d);

#else

        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, R_mat_use, &M,
               u_curr_chunk, &K, &beta, m_du2d, &M);

#endif
        // TODO: see if there's a faster way to copy (i.e. SSE?)
        // the data is transposed so it's much harder to just copy all at once
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                Dyu[INDEX_3D(i, j, k)] = m_du2d[j + i * ny];
            }
        }

        // NOTE: this is probably faster on Intel, but for now we'll do the form
        // above libxsmm_otrans(du_curr_chunk, m_du2d, sizeof(double), ny, nx,
        // nx, ny);
        // TODO: mkl's mkl_domatcopy might be even better!

        // update u_curr_chunk
        u_curr_chunk += nx * ny;
        du_curr_chunk += nx * ny;
    }

    // NOTE: it is currently faster for these derivatives if we calculate them
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dyu[ii] *= 1 / dy;
    }
#endif
}

void CompactFiniteDiff::cfd_z(double *const Dzu, const double *const u,
                              const double dz, const unsigned int *sz,
                              unsigned bflag) {
    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    char TRANSA = 'N';
    char TRANSB = 'N';
    int M = nz;
    int K = nz;
    double alpha = 1.0 / dz;
    double beta = 0.0;

    double *R_mat_use = nullptr;
    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_BACK)) && !(bflag & (1u << OCT_DIR_FRONT))) {
        R_mat_use = m_R;
    } else if ((bflag & (1u << OCT_DIR_BACK)) &&
               !(bflag & (1u << OCT_DIR_FRONT))) {
        R_mat_use = m_R_left;
    } else if (!(bflag & (1u << OCT_DIR_BACK)) &&
               (bflag & (1u << OCT_DIR_FRONT))) {
        R_mat_use = m_R_right;
    } else {
        R_mat_use = m_R_leftright;
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    int N = nx;
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_TRANS_B, M, N, K, 1.0, 0.0);
    assert(kernel);
#else
    int N = 1;
#endif

    for (unsigned int j = 0; j < ny; j++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        for (unsigned int k = 0; k < nz; k++) {
            // copy the slice of X values over
            std::copy_n(&u[INDEX_3D(0, j, k)], nx, &m_u2d[INDEX_N2D(0, k, nx)]);
        }

        // now do the faster math multiplcation
        kernel(R_mat_use, m_u2d, m_du2d);

        // then we just stick it back in, but now in memory it's stored as z0,
        // z1, z2,... then increases in x so we can't just do copy_n
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                Dzu[INDEX_3D(i, j, k)] = m_du2d[k + i * nz];
            }
        }

#else
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                m_u1d[k] = u[INDEX_3D(i, j, k)];
            }
        }

        dgemv_(&TRANSA, &M, &K, &alpha, R_mat_use, &M, m_u1d, &N, &beta, m_du1d,
               &N);

        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                Dzu[INDEX_3D(i, j, k)] = m_du1d[k];
            }
        }

#endif
    }

    // NOTE: it is currently faster for these derivatives if we calculate them
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    for (uint32_t ii = 0; ii < nx * ny * nz; ii++) {
        Dzu[ii] *= 1 / dz;
    }
#endif
}

void CompactFiniteDiff::filter_cfd_x(double *const u, double *const filtx_work,
                                     const double dx, const unsigned int *sz,
                                     unsigned bflag) {
    if (m_filter_type == FILT_NONE || m_filter_type == FILT_KO_DISS) {
        return;
    }

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // copy u to filtx_work
    std::copy_n(u, nx * ny * nz, filtx_work);

    char TRANSA = 'N';
    char TRANSB = 'N';

    int M = nx;
    int N = ny;
    int K = nx;

    // NOTE: LDA, LDB, and LDC should be nx, ny, and nz
    // TODO: fix for non-square sizes
    int LDA = nx;
    int LDB = ny;
    int LDC = nx;

    double *u_curr_chunk = (double *)u;
    double *filtu_curr_chunk = (double *)filtx_work;

    double alpha = 1.0;
    // TODO: beta should actuall be a parameter!
    double beta = 1.0;

    double *RF_mat_use = nullptr;

    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_LEFT)) && !(bflag & (1u << OCT_DIR_RIGHT))) {
        RF_mat_use = m_R_filter;
    } else if ((bflag & (1u << OCT_DIR_LEFT)) &&
               !(bflag & (1u << OCT_DIR_RIGHT))) {
        RF_mat_use = m_R_filter_left;
    } else if (!(bflag & (1u << OCT_DIR_LEFT)) &&
               (bflag & (1u << OCT_DIR_RIGHT))) {
        RF_mat_use = m_R_filter_right;
    } else {
        RF_mat_use = m_R_filter_leftright;
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, M, N, K, 1.0, 1.0);
    assert(kernel);
#endif

    for (unsigned int k = 0; k < nx; k++) {
#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        // performs C_mn = alpha * A_mk * B_kn + beta * C_mn

        // for the x_der case, m = k = nx
        kernel(RF_mat_use, u_curr_chunk, filtu_curr_chunk);

#else
        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, RF_mat_use, &LDA,
               u_curr_chunk, &LDB, &beta, filtu_curr_chunk, &LDC);

#endif
        u_curr_chunk += nx * ny;
        filtu_curr_chunk += nx * ny;
    }

    // we don't want B to overwrite C other wise we end up with errors
    std::copy_n(filtx_work, nx * ny * nz, u);

}

void CompactFiniteDiff::filter_cfd_y(double *const u, double *const filty_work,
                                     const double dy, const unsigned int *sz,
                                     unsigned bflag) {
    if (m_filter_type == FILT_NONE || m_filter_type == FILT_KO_DISS) {
        return;
    }

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    // copy u to filtx_work
    // std::copy_n(u, nx * ny * nz, filty_work);

    char TRANSA = 'N';
    char TRANSB = 'T';
    int M = ny;
    int N = nx;
    int K = ny;
    double alpha = 1.0;

    // TODO: beta needs to be a parameter
    double beta = 1.0;

    double *u_curr_chunk = (double *)u;
    double *filtu_curr_chunk = (double *)filty_work;

    double *RF_mat_use = nullptr;
    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_DOWN)) && !(bflag & (1u << OCT_DIR_UP))) {
        RF_mat_use = m_R_filter;
    } else if ((bflag & (1u << OCT_DIR_DOWN)) &&
               !(bflag & (1u << OCT_DIR_UP))) {
        RF_mat_use = m_R_filter_left;
    } else if (!(bflag & (1u << OCT_DIR_DOWN)) &&
               (bflag & (1u << OCT_DIR_UP))) {
        RF_mat_use = m_R_filter_right;
    } else {
        RF_mat_use = m_R_filter_leftright;
    }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
    typedef libxsmm_mmfunction<double> kernel_type;
    // kernel_type kernel(LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB), M, N, K, alpha,
    // beta);
    // TODO: figure out why an alpha of not 1 is breaking the kernel
    kernel_type kernel(LIBXSMM_GEMM_FLAG_TRANS_B, M, N, K, 1.0, 1.0);
    assert(kernel);
#endif

    for (unsigned int k = 0; k < nz; k++) {
        // transpose into filty_work as a copy
        for (unsigned int j = 0; j < ny; j++) {
            for (unsigned int i = 0; i < nx; i++) {
                filty_work[j + i * ny] = u_curr_chunk[i + j * nx];
            }
        }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one
        // thanks to memory layout, we can just... use this as a matrix
        // so we can just grab the "matrix" of ny x nx for this one

        kernel(RF_mat_use, u_curr_chunk, filty_work);

#else
        dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, RF_mat_use, &M,
               u_curr_chunk, &K, &beta, filty_work, &M);

#endif

        // then transpose right back
        // TODO: see if there's a faster way to copy (i.e. SSE?)
        // the data is transposed so it's much harder to just copy all at once
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int j = 0; j < ny; j++) {
                // u[INDEX_3D(i, j, k)] += filty_work[j + i * ny];
                u_curr_chunk[i + j * nx] = filty_work[j + i * ny];
            }
        }
        u_curr_chunk += nx * ny;
    }
}

void CompactFiniteDiff::filter_cfd_z(double *const u, double *const filtz_work,
                                     const double dz, const unsigned int *sz,
                                     unsigned bflag) {
    if (m_filter_type == FILT_NONE || m_filter_type == FILT_KO_DISS) {
        return;
    }

    const unsigned int nx = sz[0];
    const unsigned int ny = sz[1];
    const unsigned int nz = sz[2];

    char TRANSA = 'N';
    char TRANSB = 'N';
    int M = nz;
    int N = 1;
    int K = nz;
    double alpha = 1.0;
    double beta = 1.0;

    double *RF_mat_use = nullptr;
    // to reduce the number of checks, check for failing bflag first
    if (!(bflag & (1u << OCT_DIR_BACK)) && !(bflag & (1u << OCT_DIR_FRONT))) {
        RF_mat_use = m_R_filter;
    } else if ((bflag & (1u << OCT_DIR_BACK)) &&
               !(bflag & (1u << OCT_DIR_FRONT))) {
        RF_mat_use = m_R_filter_left;
    } else if (!(bflag & (1u << OCT_DIR_BACK)) &&
               (bflag & (1u << OCT_DIR_FRONT))) {
        RF_mat_use = m_R_filter_right;
    } else {
        RF_mat_use = m_R_filter_leftright;
    }

    for (unsigned int j = 0; j < ny; j++) {
        for (unsigned int i = 0; i < nx; i++) {
            for (unsigned int k = 0; k < nz; k++) {
                m_u1d[k] = u[INDEX_3D(i, j, k)];
                filtz_work[k] = u[INDEX_3D(i, j, k)];
            }

#ifdef FASTER_DERIV_CALC_VIA_MATRIX_MULT
            dgemv_(&TRANSA, &M, &K, &alpha, RF_mat_use, &M, m_u1d, &N, &beta,
                   filtz_work, &N);

            for (unsigned int k = 0; k < nz; k++) {
                u[INDEX_3D(i, j, k)] = filtz_work[k];
            }
#else
            dgemm_(&TRANSA, &TRANSB, &M, &N, &K, &alpha, RF_mat_use, &M, m_u1d,
                   &K, &beta, filtz_work, &M);

            for (int k = 0; k < nz; k++) {
                u[INDEX_3D(i, j, k)] = filtz_work[k];
            }

#endif
        }
    }
}

DerType getDerTypeForEdges(const DerType derivtype,
                           const BoundaryType boundary) {
    DerType doptions_CFD_P1_O4[4] = {CFD_P1_O4, CFD_DRCHLT_ORDER_4,
                                     CFD_P1_O4_CLOSE, CFD_P1_O4_L4_CLOSE};
    DerType doptions_CFD_P1_O6[4] = {CFD_P1_O6, CFD_DRCHLT_ORDER_6,
                                     CFD_P1_O6_CLOSE, CFD_P1_O6_L6_CLOSE};
    DerType doptions_CFD_Q1_O6_ETA1[4] = {CFD_Q1_O6_ETA1, CFD_DRCHLT_Q6,
                                          CFD_Q1_O6_ETA1_CLOSE,
                                          CFD_P1_O6_L6_CLOSE};

    // the doptions to use
    DerType *doptions;

    switch (derivtype) {
        case CFD_P1_O4:
            doptions = doptions_CFD_P1_O4;
            break;
        case CFD_P1_O6:
            doptions = doptions_CFD_P1_O6;
            break;
        case CFD_Q1_O6_ETA1:
            doptions = doptions_CFD_Q1_O6_ETA1;
            break;

        default:
            throw std::invalid_argument(
                "Invalid type of CFD derivative called! derivtype=" +
                std::to_string(derivtype));
            break;
    }

    switch (boundary) {
        case BLOCK_CFD_DIRICHLET:
        case BLOCK_PHYS_BOUNDARY:
            return doptions[1];
        case BLOCK_CFD_CLOSURE:
            return doptions[2];
        case BLOCK_CFD_LOPSIDE_CLOSURE:
            return doptions[3];
        default:
            return doptions[1];
    }
}

void buildPandQMatrices(double *P, double *Q, const uint32_t padding,
                        const uint32_t n, const DerType derivtype,
                        const bool is_left_edge, const bool is_right_edge) {
    // NOTE: we're pretending that all of the "mpi" or "block" boundaries
    // are treated equally. We only need to account for physical "left" and
    // "right" edges

    // NOTE: (2) we're also assuming that P and Q are initialized to **zero**.
    // There are no guarantees in this function if they are not.
    // std::cout << derivtype << " is the deriv type" << std::endl;

    uint32_t curr_n = n;
    uint32_t i_start = 0;
    uint32_t i_end = n;
    uint32_t j_start = 0;
    uint32_t j_end = n;

    if (is_left_edge) {
        // initialize the "diagonal" in the padding to 1
        for (uint32_t ii = 0; ii < padding; ii++) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_start += padding;
        j_start += padding;
        curr_n -= padding;
    }

    if (is_right_edge) {
        // initialize bottom "diagonal" in padding to 1 as well
        for (uint32_t ii = n - 1; ii >= n - padding; ii--) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_end -= padding;
        j_end -= padding;
        curr_n -= padding;
    }

    // std::cout << "i : " << i_start << " " << i_end << std::endl;
    // std::cout << "j : " << j_start << " " << j_end << std::endl;

    // NOTE: when at the "edges", we need a temporary array that can be copied
    // over
    double *tempP = nullptr;
    double *tempQ = nullptr;

    if (is_left_edge or is_right_edge) {
        // initialize tempP to be a "smaller" square matrix for use
        tempP = new double[curr_n * curr_n]();
        tempQ = new double[curr_n * curr_n]();
    } else {
        // just use the same pointer value, then no need to adjust later even
        tempP = P;
        tempQ = Q;
    }

    if (derivtype == CFD_P1_O4 || derivtype == CFD_P1_O6 ||
        derivtype == CFD_Q1_O6_ETA1) {
        // NOTE: this is only for the NONISOTROPIC matrices!!!

        // now build up the method object that will be used to calculate the
        // in-between values
        CFDMethod method(derivtype);

        int ibgn = 0;
        int iend = 0;

        DerType leftEdgeDtype;
        DerType rightEdgeDtype;

        if (is_left_edge) {
            leftEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_PHYS_BOUNDARY);
        } else {
            // TODO: update the boundary type based on what we want to build in
            leftEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_CFD_DIRICHLET);
        }

        if (is_right_edge) {
            rightEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_PHYS_BOUNDARY);
        } else {
            // TODO: update the boundary type based on what we want to build in
            rightEdgeDtype = getDerTypeForEdges(
                derivtype, BoundaryType::BLOCK_CFD_DIRICHLET);
        }

        buildMatrixLeft(tempP, tempQ, &ibgn, leftEdgeDtype, padding, curr_n);
        buildMatrixRight(tempP, tempQ, &iend, rightEdgeDtype, padding, curr_n);

        for (int i = ibgn; i <= iend; i++) {
            for (int k = -method.Ld; k <= method.Rd; k++) {
                if (!(i > -1) && !(i < curr_n)) {
                    if (is_left_edge or is_right_edge) {
                        delete[] tempP;
                        delete[] tempQ;
                    }
                    throw std::out_of_range(
                        "I is either less than zero or greater than curr_n! "
                        "i=" +
                        std::to_string(i) +
                        " curr_n=" + std::to_string(curr_n));
                }
                if (!((i + k) > -1) && !((i + k) < curr_n)) {
                    if (is_left_edge or is_right_edge) {
                        delete[] tempP;
                        delete[] tempQ;
                    }
                    throw std::out_of_range(
                        "i + k is either less than 1 or greater than curr_n! "
                        "i=" +
                        std::to_string(i + k) + " k=" + std::to_string(k) +
                        " curr_n=" + std::to_string(curr_n));
                }

                tempP[INDEX_N2D(i, i + k, curr_n)] =
                    method.alpha[k + method.Ld];
            }
            for (int k = -method.Lf; k <= method.Rf; k++) {
                if (!(i > -1) && !(i < curr_n)) {
                    throw std::out_of_range(
                        "(i is either less than zero or greater than curr_n! "
                        "i=" +
                        std::to_string(i) +
                        " curr_n=" + std::to_string(curr_n));
                }
                if (!((i + k) > -1) && !((i + k) < curr_n)) {
                    throw std::out_of_range(
                        "i + k is either less than 1 or greater than curr_n! "
                        "i=" +
                        std::to_string(i + k) + " k=" + std::to_string(k) +
                        " curr_n=" + std::to_string(curr_n));
                }

                tempQ[INDEX_N2D(i, i + k, curr_n)] = method.a[k + method.Lf];
            }
        }
    } else if (derivtype == CFD_KIM_O4) {
        // build Kim4 P and Q

        initializeKim4PQ(tempP, tempQ, curr_n);
    } else if (derivtype == CFD_HAMR_O4) {
        // build HAMR 4 P
        HAMRDeriv4_dP(tempP, curr_n);

        // then build Q
        HAMRDeriv4_dQ(tempQ, curr_n);
    } else if (derivtype == CFD_JT_O6) {
        // build JTP Deriv P
        JTPDeriv6_dP(tempP, curr_n);

        // then build Q
        JTPDeriv6_dQ(tempQ, curr_n);
    } else if (derivtype == CFD_NONE) {
        // just.... do nothing... keep them at zeros
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "dendro_cfd::buildPandQMatrices should never be called with a "
            "CFD_NONE deriv type!");
    } else {
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "The CFD deriv type was not one of the valid options. derivtype=" +
            std::to_string(derivtype));
    }

    // copy the values back in
    // NOTE: the use of j and i assumes ROW-MAJOR order, but it will just copy a
    // square matrix in no matter what, so it's not a big issue
    if (is_left_edge or is_right_edge) {
        // then memcopy the "chunks" to where they go inside the matrix
        uint32_t temp_arr_i = 0;
        // iterate over the rows
        for (uint32_t jj = j_start; jj < j_end; jj++) {
            // ii will only go from empty rows we actually need to fill...
            // j will start at "j_start" and go until "j_end" where we need to
            // fill memory start index of our main array

            uint32_t temp_start = INDEX_N2D(0, temp_arr_i, curr_n);
            // uint32_t temp_end = INDEX_N2D(curr_n - 1, temp_arr_i, curr_n);

            std::copy_n(&tempP[temp_start], curr_n, &P[INDEX_2D(i_start, jj)]);
            std::copy_n(&tempQ[temp_start], curr_n, &Q[INDEX_2D(i_start, jj)]);

            // increment temp_arr "row" value
            temp_arr_i++;
        }
        // clear up our temporary arrays we don't need
        delete[] tempP;
        delete[] tempQ;
    }
    // NOTE: tempP doesn't need to be deleted if it was not initialized,
    // so we don't need to delete it unless we're dealing with left/right edges
}

void buildPandQFilterMatrices(double *P, double *Q, const uint32_t padding,
                              const uint32_t n, const FilterType filtertype,
                              const bool is_left_edge,
                              const bool is_right_edge) {
    // NOTE: we're pretending that all of the "mpi" or "block" boundaries
    // are treated equally. We only need to account for physical "left" and
    // "right" edges

    // NOTE: (2) we're also assuming that P and Q are initialized to **zero**.
    // There are no guarantees in this function if they are not.
    // std::cout << filtertype << " is the filter type" << std::endl;

    uint32_t curr_n = n;
    uint32_t i_start = 0;
    uint32_t i_end = n;
    uint32_t j_start = 0;
    uint32_t j_end = n;

    if (is_left_edge) {
        // initialize the "diagonal" in the padding to 1
        for (uint32_t ii = 0; ii < padding; ii++) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_start += padding;
        j_start += padding;
        curr_n -= padding;
    }

    if (is_right_edge) {
        // initialize bottom "diagonal" in padding to 1 as well
        for (uint32_t ii = n - 1; ii >= n - padding; ii--) {
            P[INDEX_2D(ii, ii)] = 1.0;
            Q[INDEX_2D(ii, ii)] = 1.0;
        }
        i_end -= padding;
        j_end -= padding;
        curr_n -= padding;
    }

    // std::cout << "i : " << i_start << " " << i_end << std::endl;
    // std::cout << "j : " << j_start << " " << j_end << std::endl;

    // NOTE: when at the "edges", we need a temporary array that can be copied
    // over
    double *tempP = nullptr;
    double *tempQ = nullptr;

    if (is_left_edge or is_right_edge) {
        // initialize tempP to be a "smaller" square matrix for use
        tempP = new double[curr_n * curr_n]();
        tempQ = new double[curr_n * curr_n]();
    } else {
        // just use the same pointer value, then no need to adjust later even
        tempP = P;
        tempQ = Q;
    }

    if (filtertype == FILT_KIM_6) {
        // build Kim4 P and Q

        initializeKim6FilterPQ(tempP, tempQ, curr_n);
    } else if (filtertype == FILT_JT_6) {
        // TODO: NOT CURRENTLY IMPLEMENTED
        std::cerr << "WARNING: The JT 6 filter is not yet ready! This will "
                     "lead to unexpected results!"
                  << std::endl;
    } else if (filtertype == FILT_JT_8) {
        // TODO: NOT CURRENTLY IMPLEMENTED
        std::cerr << "WARNING: The JT 8 filter is not yet ready! This will "
                     "lead to unexpected results!"
                  << std::endl;
    } else if (filtertype == FILT_NONE || filtertype == FILT_KO_DISS) {
        // just.... do nothing... keep them at zeros
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "dendro_cfd::buildPandQFilterMatrices should never be called with "
            "a "
            "CFD_NONE deriv type!");
    } else {
        if (is_left_edge or is_right_edge) {
            delete[] tempP;
            delete[] tempQ;
        }
        throw std::invalid_argument(
            "The filter type was not one of the valid options. filtertype=" +
            std::to_string(filtertype));
    }

    // copy the values back in
    // NOTE: the use of j and i assumes ROW-MAJOR order, but it will just copy a
    // square matrix in no matter what, so it's not a big issue
    if (is_left_edge or is_right_edge) {
        // then memcopy the "chunks" to where they go inside the matrix
        uint32_t temp_arr_i = 0;
        // iterate over the rows
        for (uint32_t jj = j_start; jj < j_end; jj++) {
            // ii will only go from empty rows we actually need to fill...
            // j will start at "j_start" and go until "j_end" where we need to
            // fill memory start index of our main array

            uint32_t temp_start = INDEX_N2D(0, temp_arr_i, curr_n);
            // uint32_t temp_end = INDEX_N2D(curr_n - 1, temp_arr_i, curr_n);

            std::copy_n(&tempP[temp_start], curr_n, &P[INDEX_2D(i_start, jj)]);
            std::copy_n(&tempQ[temp_start], curr_n, &Q[INDEX_2D(i_start, jj)]);

            // increment temp_arr "row" value
            temp_arr_i++;
        }
        // clear up our temporary arrays we don't need
        delete[] tempP;
        delete[] tempQ;
    }
    // NOTE: tempP doesn't need to be deleted if it was not initialized,
    // so we don't need to delete it unless we're dealing with left/right edges
}

void calculateDerivMatrix(double *D, double *P, double *Q, const int n) {
    int *ipiv = new int[n];

    int info;
    int nx = n;

    dgetrf_(&nx, &nx, P, &nx, ipiv, &info);

    if (info != 0) {
        delete[] ipiv;
        throw std::runtime_error("LU factorization failed: info=" +
                                 std::to_string(info));
    }

    double *Pinv = new double[n * n];

    // memcpy is faster than the for loops!
    std::memcpy(Pinv, P, n * n * sizeof(double));

    int lwork = n * n;
    double *work = new double[lwork];

    dgetri_(&nx, Pinv, &nx, ipiv, work, &lwork, &info);

    if (info != 0) {
        delete[] ipiv;
        delete[] Pinv;
        delete[] work;
        throw std::runtime_error("Matrix inversion failed: info=" +
                                 std::to_string(info));
    }

#ifdef PRINT_COMPACT_MATRICES
    std::cout << "P INVERSE" << std::endl;
    print_square_mat(Pinv, n);
#endif

    mulMM(D, Pinv, Q, n, n);

    delete[] ipiv;
    delete[] Pinv;
    delete[] work;
}

void mulMM(double *C, double *A, double *B, int na, int nb) {
    /*  M = number of rows of A and C
        N = number of columns of B and C
        K = number of columns of A and rows of B
    */

    char TA[4], TB[4];
    double ALPHA = 1.0;
    double BETA = 0.0;
    sprintf(TA, "N");
    sprintf(TB, "N");
    int M = na;
    int N = nb;
    int K = na;
    int LDA = na;
    int LDB = na;
    int LDC = na;

    dgemm_(TA, TB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

void setArrToZero(double *arr, const int n) {
    for (uint16_t ii = 0; ii < n; ii++) {
        arr[ii] = 0.0;
    }
}

void buildMatrixLeft(double *P, double *Q, int *xib, const DerType dtype,
                     const int nghosts, const int n) {
    int ib = 0;

    switch (dtype) {
        case CFD_DRCHLT_ORDER_4: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 3.0;

            Q[INDEX_2D(0, 0)] = -17.0 / 6.0;
            Q[INDEX_2D(0, 1)] = 3.0 / 2.0;
            Q[INDEX_2D(0, 2)] = 3.0 / 2.0;
            Q[INDEX_2D(0, 3)] = -1.0 / 6.0;
            ib = 1;
        } break;

        case CFD_DRCHLT_ORDER_6: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 5.0;

            P[INDEX_2D(1, 0)] = 2.0 / 11.0;
            P[INDEX_2D(1, 1)] = 1.0;
            P[INDEX_2D(1, 2)] = 2.0 / 11.0;

            Q[INDEX_2D(0, 0)] = -197.0 / 60.0;
            Q[INDEX_2D(0, 1)] = -5.0 / 12.0;
            Q[INDEX_2D(0, 2)] = 5.0;
            Q[INDEX_2D(0, 3)] = -5.0 / 3.0;
            Q[INDEX_2D(0, 4)] = 5.0 / 12.0;
            Q[INDEX_2D(0, 5)] = -1.0 / 20.0;

            Q[INDEX_2D(1, 0)] = -20.0 / 33.0;
            Q[INDEX_2D(1, 1)] = -35.0 / 132.0;
            Q[INDEX_2D(1, 2)] = 34.0 / 33.0;
            Q[INDEX_2D(1, 3)] = -7.0 / 33.0;
            Q[INDEX_2D(1, 4)] = 2.0 / 33.0;
            Q[INDEX_2D(1, 5)] = -1.0 / 132.0;
            ib = 2;
        } break;

        case CFD_DRCHLT_Q6: {
            P[INDEX_2D(0, 0)] = 1.0;
            P[INDEX_2D(0, 1)] = 5.0;

            P[INDEX_2D(1, 0)] = 2.0 / 11.0;
            P[INDEX_2D(1, 1)] = 1.0;
            P[INDEX_2D(1, 2)] = 2.0 / 11.0;

            P[INDEX_2D(2, 1)] = 1.0 / 3.0;
            P[INDEX_2D(2, 2)] = 1.0;
            P[INDEX_2D(2, 3)] = 1.0 / 3.0;

            Q[INDEX_2D(0, 0)] = -197.0 / 60.0;
            Q[INDEX_2D(0, 1)] = -5.0 / 12.0;
            Q[INDEX_2D(0, 2)] = 5.0;
            Q[INDEX_2D(0, 3)] = -5.0 / 3.0;
            Q[INDEX_2D(0, 4)] = 5.0 / 12.0;
            Q[INDEX_2D(0, 5)] = -1.0 / 20.0;

            Q[INDEX_2D(1, 0)] = -20.0 / 33.0;
            Q[INDEX_2D(1, 1)] = -35.0 / 132.0;
            Q[INDEX_2D(1, 2)] = 34.0 / 33.0;
            Q[INDEX_2D(1, 3)] = -7.0 / 33.0;
            Q[INDEX_2D(1, 4)] = 2.0 / 33.0;
            Q[INDEX_2D(1, 5)] = -1.0 / 132.0;

            Q[INDEX_2D(2, 0)] = -1.0 / 36.0;
            Q[INDEX_2D(2, 1)] = -14.0 / 18.0;
            Q[INDEX_2D(3, 2)] = 0.0;
            Q[INDEX_2D(2, 3)] = 14.0 / 18.0;
            Q[INDEX_2D(2, 4)] = 1.0 / 36.0;

            ib = 3;
        } break;

        case CFD_P1_O4_CLOSE: {
            if (nghosts < 3) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "3! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t1 = 1.0 / 72.0;
            Q[INDEX_2D(ib, ib - 3)] = -t1;
            Q[INDEX_2D(ib, ib - 2)] = 10.0 * t1;
            Q[INDEX_2D(ib, ib - 1)] = -53.0 * t1;
            Q[INDEX_2D(ib, ib)] = 0.0;
            Q[INDEX_2D(ib, ib + 1)] = 53.0 * t1;
            Q[INDEX_2D(ib, ib + 2)] = -10.0 * t1;
            Q[INDEX_2D(ib, ib + 3)] = t1;
            ib += 1;
        } break;

        case CFD_P1_O6_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t2 = 1.0 / 300.0;
            Q[INDEX_2D(ib, ib - 4)] = t2;
            Q[INDEX_2D(ib, ib - 3)] = -11.0 * t2;
            Q[INDEX_2D(ib, ib - 2)] = 59.0 * t2;
            Q[INDEX_2D(ib, ib - 1)] = -239.0 * t2;
            Q[INDEX_2D(ib, ib)] = 0.0;
            Q[INDEX_2D(ib, ib + 1)] = 239.0 * t2;
            Q[INDEX_2D(ib, ib + 2)] = -59.0 * t2;
            Q[INDEX_2D(ib, ib + 3)] = 11.0 * t2;
            Q[INDEX_2D(ib, ib + 4)] = -t2;
            ib += 1;
        } break;

        case CFD_P1_O4_L4_CLOSE: {
            if (nghosts < 1) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "1! nghosts = " +
                    std::to_string(nghosts));
            }

            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            const double t3 = 1.0 / 12.0;
            Q[INDEX_2D(ib, ib - 1)] = -3.0 * t3;
            Q[INDEX_2D(ib, ib)] = -10.0 * t3;
            Q[INDEX_2D(ib, ib + 1)] = 18.0 * t3;
            Q[INDEX_2D(ib, ib + 2)] = -6.0 * t3;
            Q[INDEX_2D(ib, ib + 3)] = t3;

            ib += 1;
        } break;

        case CFD_P1_O6_L6_CLOSE: {
            if (nghosts < 2) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "2! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            const double t4 = 1.0 / 60.0;
            P[INDEX_2D(ib, ib)] = 1.0;

            Q[INDEX_2D(ib, ib - 2)] = 2.0 * t4;
            Q[INDEX_2D(ib, ib - 1)] = -24.0 * t4;
            Q[INDEX_2D(ib, ib)] = -35.0 * t4;
            Q[INDEX_2D(ib, ib + 1)] = 80.0 * t4;
            Q[INDEX_2D(ib, ib + 2)] = -30.0 * t4;
            Q[INDEX_2D(ib, ib + 3)] = 8.0 * t4;
            Q[INDEX_2D(ib, ib + 4)] = -1.0 * t4;

            ib += 1;
        } break;

        case CFD_Q1_O6_ETA1_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at "
                    "least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = 0; i < nghosts; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ib = nghosts;
            P[INDEX_2D(ib, ib)] = 1.0;

            Q[INDEX_2D(ib, ib - 4)] = 0.0035978349;
            Q[INDEX_2D(ib, ib - 3)] = -0.038253676;
            Q[INDEX_2D(ib, ib - 2)] = 0.20036969;
            Q[INDEX_2D(ib, ib - 1)] = -0.80036969;
            Q[INDEX_2D(ib, ib)] = 0.0;
            Q[INDEX_2D(ib, ib + 1)] = 0.80036969;
            Q[INDEX_2D(ib, ib + 2)] = -0.20036969;
            Q[INDEX_2D(ib, ib + 3)] = 0.038253676;
            Q[INDEX_2D(ib, ib + 4)] = -0.0035978349;
            ib += 1;
        } break;

            // NOTE: in original initcfd.c file from David Neilsen, this was
            // repeated in the if statement, but in an elif, so it's unreachable
            // anyway since this value is handled in the same way above case
            // CFD_P1_O4_L4_CLOSE: ...

        default:
            throw std::invalid_argument(
                "Unknown derivative type for initializing CFD matrices! "
                "dtype=" +
                std::to_string(dtype));
            break;
    }
    // update xib
    *xib = ib;
}

void buildMatrixRight(double *P, double *Q, int *xie, const DerType dtype,
                      const int nghosts, const int n) {
    int ie = n - 1;

    switch (dtype) {
        case CFD_DRCHLT_ORDER_4: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 3.0;

            Q[INDEX_2D(n - 1, n - 1)] = 17.0 / 6.0;
            Q[INDEX_2D(n - 1, n - 2)] = -3.0 / 2.0;
            Q[INDEX_2D(n - 1, n - 3)] = -3.0 / 2.0;
            Q[INDEX_2D(n - 1, n - 4)] = 1.0 / 6.0;
            ie = n - 2;
        } break;

        case CFD_DRCHLT_ORDER_6: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 5.0;

            P[INDEX_2D(n - 2, n - 1)] = 2.0 / 11.0;
            P[INDEX_2D(n - 2, n - 2)] = 1.0;
            P[INDEX_2D(n - 2, n - 3)] = 2.0 / 11.0;

            Q[INDEX_2D(n - 1, n - 1)] = 197.0 / 60.0;
            Q[INDEX_2D(n - 1, n - 2)] = 5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 3)] = -5.0;
            Q[INDEX_2D(n - 1, n - 4)] = 5.0 / 3.0;
            Q[INDEX_2D(n - 1, n - 5)] = -5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 6)] = 1.0 / 20.0;

            Q[INDEX_2D(n - 2, n - 1)] = 20.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 2)] = 35.0 / 132.0;
            Q[INDEX_2D(n - 2, n - 3)] = -34.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 4)] = 7.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 5)] = -2.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 6)] = 1.0 / 132.0;
            ie = n - 3;
        } break;

        case CFD_DRCHLT_Q6: {
            P[INDEX_2D(n - 1, n - 1)] = 1.0;
            P[INDEX_2D(n - 1, n - 2)] = 5.0;

            P[INDEX_2D(n - 2, n - 1)] = 2.0 / 11.0;
            P[INDEX_2D(n - 2, n - 2)] = 1.0;
            P[INDEX_2D(n - 2, n - 3)] = 2.0 / 11.0;

            P[INDEX_2D(n - 3, n - 2)] = 1.0 / 3.0;
            P[INDEX_2D(n - 3, n - 3)] = 1.0;
            P[INDEX_2D(n - 3, n - 4)] = 1.0 / 3.0;

            Q[INDEX_2D(n - 1, n - 1)] = 197.0 / 60.0;
            Q[INDEX_2D(n - 1, n - 2)] = 5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 3)] = -5.0;
            Q[INDEX_2D(n - 1, n - 4)] = 5.0 / 3.0;
            Q[INDEX_2D(n - 1, n - 5)] = -5.0 / 12.0;
            Q[INDEX_2D(n - 1, n - 6)] = 1.0 / 20.0;

            Q[INDEX_2D(n - 2, n - 1)] = 20.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 2)] = 35.0 / 132.0;
            Q[INDEX_2D(n - 2, n - 3)] = -34.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 4)] = 7.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 5)] = -2.0 / 33.0;
            Q[INDEX_2D(n - 2, n - 6)] = 1.0 / 132.0;

            Q[INDEX_2D(n - 3, n - 1)] = 1.0 / 36.0;
            Q[INDEX_2D(n - 3, n - 2)] = 14.0 / 18.0;
            Q[INDEX_2D(n - 3, n - 3)] = 0.0;
            Q[INDEX_2D(n - 3, n - 4)] = -14.0 / 18.0;
            Q[INDEX_2D(n - 3, n - 5)] = -1.0 / 36.0;

            ie = n - 4;
        } break;

        case CFD_P1_O4_CLOSE: {
            if (nghosts < 3) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "3! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t1 = 1.0 / 72.0;
            Q[INDEX_2D(ie, ie - 3)] = -t1;
            Q[INDEX_2D(ie, ie - 2)] = 10.0 * t1;
            Q[INDEX_2D(ie, ie - 1)] = -53.0 * t1;
            Q[INDEX_2D(ie, ie)] = 0.0;
            Q[INDEX_2D(ie, ie + 1)] = 53.0 * t1;
            Q[INDEX_2D(ie, ie + 2)] = -10.0 * t1;
            Q[INDEX_2D(ie, ie + 3)] = t1;
            ie -= 1;
        } break;

        case CFD_P1_O6_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t2 = 1.0 / 300.0;
            Q[INDEX_2D(ie, ie - 4)] = t2;
            Q[INDEX_2D(ie, ie - 3)] = -11.0 * t2;
            Q[INDEX_2D(ie, ie - 2)] = 59.0 * t2;
            Q[INDEX_2D(ie, ie - 1)] = -239.0 * t2;
            Q[INDEX_2D(ie, ie)] = 0.0;
            Q[INDEX_2D(ie, ie + 1)] = 239.0 * t2;
            Q[INDEX_2D(ie, ie + 2)] = -59.0 * t2;
            Q[INDEX_2D(ie, ie + 3)] = 11.0 * t2;
            Q[INDEX_2D(ie, ie + 4)] = -t2;
            ie -= 1;
        } break;

        case CFD_P1_O4_L4_CLOSE: {
            if (nghosts < 1) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "1! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            const double t3 = 1.0 / 12.0;
            Q[INDEX_2D(ie, ie + 1)] = 3.0 * t3;
            Q[INDEX_2D(ie, ie)] = 10.0 * t3;
            Q[INDEX_2D(ie, ie - 1)] = -18.0 * t3;
            Q[INDEX_2D(ie, ie - 2)] = 6.0 * t3;
            Q[INDEX_2D(ie, ie - 3)] = -t3;

            ie -= 1;
        } break;

        case CFD_P1_O6_L6_CLOSE: {
            if (nghosts < 2) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at least "
                    "2! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            const double t4 = 1.0 / 60.0;
            P[INDEX_2D(ie, ie)] = 1.0;

            Q[INDEX_2D(ie, ie + 2)] = -2.0 * t4;
            Q[INDEX_2D(ie, ie + 1)] = 24.0 * t4;
            Q[INDEX_2D(ie, ie)] = 35.0 * t4;
            Q[INDEX_2D(ie, ie - 1)] = -80.0 * t4;
            Q[INDEX_2D(ie, ie - 2)] = 30.0 * t4;
            Q[INDEX_2D(ie, ie - 3)] = -8.0 * t4;
            Q[INDEX_2D(ie, ie - 4)] = 1.0 * t4;

            ie -= 1;
        } break;

        case CFD_Q1_O6_ETA1_CLOSE: {
            if (nghosts < 4) {
                throw std::invalid_argument(
                    "Not enough dimensionality in ghost points! Need at "
                    "least "
                    "4! nghosts = " +
                    std::to_string(nghosts));
            }
            // set ghost region to identity...don't use these derivs!
            for (int i = n - nghosts; i < n; i++) {
                P[INDEX_2D(i, i)] = 1.0;
                Q[INDEX_2D(i, i)] = 1.0;
            }
            // add closure
            ie = n - nghosts - 1;
            P[INDEX_2D(ie, ie)] = 1.0;

            Q[INDEX_2D(ie, ie - 4)] = 0.0035978349;
            Q[INDEX_2D(ie, ie - 3)] = -0.038253676;
            Q[INDEX_2D(ie, ie - 2)] = 0.20036969;
            Q[INDEX_2D(ie, ie - 1)] = -0.80036969;
            Q[INDEX_2D(ie, ie)] = 0.0;
            Q[INDEX_2D(ie, ie + 1)] = 0.80036969;
            Q[INDEX_2D(ie, ie + 2)] = -0.20036969;
            Q[INDEX_2D(ie, ie + 3)] = 0.038253676;
            Q[INDEX_2D(ie, ie + 4)] = -0.0035978349;
            ie -= 1;
        } break;

        default:
            break;
    }
    // update xib
    *xie = ie;
}

void initializeKim4PQ(double *P, double *Q, int n) {
    const double alpha = 0.5862704032801503;
    const double beta = 9.549533555017055e-2;

    const double a1 = 0.6431406736919156;
    const double a2 = 0.2586011023495066;
    const double a3 = 7.140953479797375e-3;

    const double y00 = 0.0;
    const double y10 = 8.360703307833438e-2;
    const double y20 = 3.250008295108466e-2;
    const double y01 = 5.912678614078549;
    const double y11 = 0.0;
    const double y21 = 0.3998040493524358;
    const double y02 = 3.775623951744012;
    const double y12 = 2.058102869495757;
    const double y22 = 0.0;
    const double y03 = 0.0;
    const double y13 = 0.9704052014790193;
    const double y23 = 0.7719261277615860;
    const double y04 = 0.0;
    const double y14 = 0.0;
    const double y24 = 0.1626635931256900;

    const double b10 = -0.3177447290722621;
    const double b20 = -0.1219006056449124;
    const double b01 = -3.456878182643609;
    const double b21 = -0.6301651351188667;
    const double b02 = 5.839043358834730;
    const double b12 = -2.807631929593225e-2;
    const double b03 = 1.015886726041007;
    const double b13 = 1.593461635747659;
    const double b23 = 0.6521195063966084;
    const double b04 = -0.2246526470654333;
    const double b14 = 0.2533027046976367;
    const double b24 = 0.3938843551210350;
    const double b05 = 8.564940889936562e-2;
    const double b15 = -3.619652460174756e-2;
    const double b25 = 1.904944407973912e-2;
    const double b06 = -1.836710059356763e-2;
    const double b16 = 4.080281419108407e-3;
    const double b26 = -1.027260523947668e-3;

    const double b00 = -(b01 + b02 + b03 + b04 + b05 + b06);
    const double b11 = -(b10 + b12 + b13 + b14 + b15 + b16);
    const double b22 = -(b20 + b21 + b23 + b24 + b25 + b26);

    const int nd = n * n;

    for (int i = 3; i < n - 3; i++) {
        P[INDEX_2D(i, i - 2)] = beta;
        P[INDEX_2D(i, i - 1)] = alpha;
        P[INDEX_2D(i, i)] = 1.0;
        P[INDEX_2D(i, i + 1)] = alpha;
        P[INDEX_2D(i, i + 2)] = beta;
    }

    P[INDEX_2D(0, 0)] = 1.0;
    P[INDEX_2D(0, 1)] = y01;
    P[INDEX_2D(0, 2)] = y02;

    P[INDEX_2D(1, 0)] = y10;
    P[INDEX_2D(1, 1)] = 1.0;
    P[INDEX_2D(1, 2)] = y12;
    P[INDEX_2D(1, 3)] = y13;

    P[INDEX_2D(2, 0)] = y20;
    P[INDEX_2D(2, 1)] = y21;
    P[INDEX_2D(2, 2)] = 1.0;
    P[INDEX_2D(2, 3)] = y23;
    P[INDEX_2D(2, 4)] = y24;

    P[INDEX_2D(n - 3, n - 5)] = y24;
    P[INDEX_2D(n - 3, n - 4)] = y23;
    P[INDEX_2D(n - 3, n - 3)] = 1.0;
    P[INDEX_2D(n - 3, n - 2)] = y21;
    P[INDEX_2D(n - 3, n - 1)] = y20;

    P[INDEX_2D(n - 2, n - 4)] = y13;
    P[INDEX_2D(n - 2, n - 3)] = y12;
    P[INDEX_2D(n - 2, n - 2)] = 1.0;
    P[INDEX_2D(n - 2, n - 1)] = y10;

    P[INDEX_2D(n - 1, n - 3)] = y02;
    P[INDEX_2D(n - 1, n - 2)] = y01;
    P[INDEX_2D(n - 1, n - 1)] = 1.0;

    for (int i = 3; i < n - 3; i++) {
        Q[INDEX_2D(i, i - 3)] = -a3;
        Q[INDEX_2D(i, i - 2)] = -a2;
        Q[INDEX_2D(i, i - 1)] = -a1;
        Q[INDEX_2D(i, i)] = 0.0;
        Q[INDEX_2D(i, i + 1)] = a1;
        Q[INDEX_2D(i, i + 2)] = a2;
        Q[INDEX_2D(i, i + 3)] = a3;
    }

    Q[INDEX_2D(0, 0)] = b00;
    Q[INDEX_2D(0, 1)] = b01;
    Q[INDEX_2D(0, 2)] = b02;
    Q[INDEX_2D(0, 3)] = b03;
    Q[INDEX_2D(0, 4)] = b04;
    Q[INDEX_2D(0, 5)] = b05;
    Q[INDEX_2D(0, 6)] = b06;

    Q[INDEX_2D(1, 0)] = b10;
    Q[INDEX_2D(1, 1)] = b11;
    Q[INDEX_2D(1, 2)] = b12;
    Q[INDEX_2D(1, 3)] = b13;
    Q[INDEX_2D(1, 4)] = b14;
    Q[INDEX_2D(1, 5)] = b15;
    Q[INDEX_2D(1, 6)] = b16;

    Q[INDEX_2D(2, 0)] = b20;
    Q[INDEX_2D(2, 1)] = b21;
    Q[INDEX_2D(2, 2)] = b22;
    Q[INDEX_2D(2, 3)] = b23;
    Q[INDEX_2D(2, 4)] = b24;
    Q[INDEX_2D(2, 5)] = b25;
    Q[INDEX_2D(2, 6)] = b26;

    Q[INDEX_2D(n - 3, n - 1)] = -b20;
    Q[INDEX_2D(n - 3, n - 2)] = -b21;
    Q[INDEX_2D(n - 3, n - 3)] = -b22;
    Q[INDEX_2D(n - 3, n - 4)] = -b23;
    Q[INDEX_2D(n - 3, n - 5)] = -b24;
    Q[INDEX_2D(n - 3, n - 6)] = -b25;
    Q[INDEX_2D(n - 3, n - 7)] = -b26;

    Q[INDEX_2D(n - 2, n - 1)] = -b10;
    Q[INDEX_2D(n - 2, n - 2)] = -b11;
    Q[INDEX_2D(n - 2, n - 3)] = -b12;
    Q[INDEX_2D(n - 2, n - 4)] = -b13;
    Q[INDEX_2D(n - 2, n - 5)] = -b14;
    Q[INDEX_2D(n - 2, n - 6)] = -b15;
    Q[INDEX_2D(n - 2, n - 7)] = -b16;

    Q[INDEX_2D(n - 1, n - 1)] = -b00;
    Q[INDEX_2D(n - 1, n - 2)] = -b01;
    Q[INDEX_2D(n - 1, n - 3)] = -b02;
    Q[INDEX_2D(n - 1, n - 4)] = -b03;
    Q[INDEX_2D(n - 1, n - 5)] = -b04;
    Q[INDEX_2D(n - 1, n - 6)] = -b05;
    Q[INDEX_2D(n - 1, n - 7)] = -b06;
}

void initializeKim6FilterPQ(double *P, double *Q, int n) {
    const double alphaF = 0.6651452077642562;
    const double betaF = 0.1669709584471488;
    const double aF1 = 8.558206326059179e-4;
    const double aF2 = -3.423282530423672e-4;
    const double aF3 = 5.705470884039454e-5;
    const double aF0 = -2.0 * (aF1 + aF2 + aF3);

    const double yF00 = 0.0;
    const double yF10 = 0.7311329755609861;
    const double yF20 = 0.1681680891936087;
    const double yF01 = 0.3412746505356879;
    const double yF11 = 0.0;
    const double yF21 = 0.6591595540319565;
    const double yF02 = 0.2351300295562464;
    const double yF12 = 0.6689728401317021;
    const double yF22 = 0.0;
    const double yF03 = 0.0;
    const double yF13 = 0.1959510121583215;
    const double yF23 = 0.6591595540319565;
    const double yF04 = 0.0;
    const double yF14 = 0.0;
    const double yF24 = 0.1681680891936087;

    const double bF20 = -2.81516723801634e-4;
    const double bF21 = 1.40758361900817e-3;
    const double bF23 = 2.81516723801634e-3;
    const double bF24 = -1.40758361900817e-3;
    const double bF25 = 2.81516723801634e-4;
    const double bF22 = -(bF20 + bF21 + bF23 + bF24 + bF25);

    const int nd = n * n;

    for (int i = 3; i < n - 3; i++) {
        P[INDEX_2D(i, i - 2)] = betaF;
        P[INDEX_2D(i, i - 1)] = alphaF;
        P[INDEX_2D(i, i)] = 1.0;
        P[INDEX_2D(i, i + 1)] = alphaF;
        P[INDEX_2D(i, i + 2)] = betaF;
    }

    P[INDEX_2D(0, 0)] = 1.0;
    P[INDEX_2D(0, 1)] = yF01;
    P[INDEX_2D(0, 2)] = yF02;

    P[INDEX_2D(1, 0)] = yF10;
    P[INDEX_2D(1, 1)] = 1.0;
    P[INDEX_2D(1, 2)] = yF12;
    P[INDEX_2D(1, 3)] = yF13;

    P[INDEX_2D(2, 0)] = yF20;
    P[INDEX_2D(2, 1)] = yF21;
    P[INDEX_2D(2, 2)] = 1.0;
    P[INDEX_2D(2, 3)] = yF23;
    P[INDEX_2D(2, 4)] = yF24;

    P[INDEX_2D(n - 3, n - 5)] = yF24;
    P[INDEX_2D(n - 3, n - 4)] = yF23;
    P[INDEX_2D(n - 3, n - 3)] = 1.0;
    P[INDEX_2D(n - 3, n - 2)] = yF21;
    P[INDEX_2D(n - 3, n - 1)] = yF20;

    P[INDEX_2D(n - 2, n - 4)] = yF13;
    P[INDEX_2D(n - 2, n - 3)] = yF12;
    P[INDEX_2D(n - 2, n - 2)] = 1.0;
    P[INDEX_2D(n - 2, n - 1)] = yF10;

    P[INDEX_2D(n - 1, n - 3)] = yF02;
    P[INDEX_2D(n - 1, n - 2)] = yF01;
    P[INDEX_2D(n - 1, n - 1)] = 1.0;

    for (int i = 0; i < nd; i++) {
        Q[i] = 0.0;
    }
    for (int i = 3; i < n - 3; i++) {
        Q[INDEX_2D(i, i - 3)] = aF3;
        Q[INDEX_2D(i, i - 2)] = aF2;
        Q[INDEX_2D(i, i - 1)] = aF1;
        Q[INDEX_2D(i, i)] = aF0;
        Q[INDEX_2D(i, i + 1)] = aF1;
        Q[INDEX_2D(i, i + 2)] = aF2;
        Q[INDEX_2D(i, i + 3)] = aF3;
    }

    Q[INDEX_2D(0, 0)] = 0.0;
    Q[INDEX_2D(0, 1)] = 0.0;
    Q[INDEX_2D(0, 2)] = 0.0;
    Q[INDEX_2D(0, 3)] = 0.0;

    Q[INDEX_2D(1, 0)] = 0.0;
    Q[INDEX_2D(1, 1)] = 0.0;
    Q[INDEX_2D(1, 2)] = 0.0;
    Q[INDEX_2D(1, 3)] = 0.0;
    Q[INDEX_2D(1, 4)] = 0.0;

    Q[INDEX_2D(2, 0)] = bF20;
    Q[INDEX_2D(2, 1)] = bF21;
    Q[INDEX_2D(2, 2)] = bF22;
    Q[INDEX_2D(2, 3)] = bF23;
    Q[INDEX_2D(2, 4)] = bF24;
    Q[INDEX_2D(2, 5)] = bF25;

    Q[INDEX_2D(n - 3, n - 6)] = bF25;
    Q[INDEX_2D(n - 3, n - 5)] = bF24;
    Q[INDEX_2D(n - 3, n - 4)] = bF23;
    Q[INDEX_2D(n - 3, n - 3)] = bF22;
    Q[INDEX_2D(n - 3, n - 2)] = bF21;
    Q[INDEX_2D(n - 3, n - 1)] = bF20;

    Q[INDEX_2D(n - 2, n - 5)] = 0.0;
    Q[INDEX_2D(n - 2, n - 4)] = 0.0;
    Q[INDEX_2D(n - 2, n - 3)] = 0.0;
    Q[INDEX_2D(n - 2, n - 2)] = 0.0;
    Q[INDEX_2D(n - 2, n - 1)] = 0.0;

    Q[INDEX_2D(n - 1, n - 4)] = 0.0;
    Q[INDEX_2D(n - 1, n - 3)] = 0.0;
    Q[INDEX_2D(n - 1, n - 2)] = 0.0;
    Q[INDEX_2D(n - 1, n - 1)] = 0.0;
}

void print_square_mat(double *m, const uint32_t n) {
    // assumes "col" order in memory
    // J is the row!
    for (uint16_t i = 0; i < n; i++) {
        printf("%3d : ", i);
        // I is the column!
        for (uint16_t j = 0; j < n; j++) {
            printf("%8.3f ", m[INDEX_2D(i, j)]);
        }
        printf("\n");
    }
}

}  // namespace dendro_cfd