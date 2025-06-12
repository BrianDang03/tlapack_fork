/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/base/utils.hpp>
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/hemm.hpp>
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/conjugate.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/lauum_recursive.hpp>
#include <tlapack/lapack/potrf.hpp>
#include <tlapack/lapack/trtri_recursive.hpp>
#include <tlapack/lapack/ung2r.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

using namespace tlapack;

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

//--------------------------------------------------------------------
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          TLAPACK_SCALAR beta_t,
          class T = type_t<matrixC_t>,
          disable_if_allow_optblas_t<pair<matrixA_t, T>,
                                     pair<matrixB_t, T>,
                                     pair<matrixC_t, T>,
                                     pair<alpha_t, T>,
                                     pair<beta_t, T> > = 0>
void hemm_brian(Side side,
                Uplo uplo,
                Op trans,
                const alpha_t& alpha,
                const matrixA_t& A,
                const matrixB_t& B,
                const beta_t& beta,
                matrixC_t& C)
{
    // data traits
    using TA = type_t<matrixA_t>;
    using TB = type_t<matrixB_t>;
    using idx_t = size_type<matrixB_t>;

    // constants
    const idx_t m = nrows(B);
    const idx_t n = ncols(B);

    // check arguments
    // tlapack_check_false(side != Side::Left && side != Side::Right);
    // tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper &&
    //                     uplo != Uplo::General);
    // tlapack_check_false(nrows(A) != ncols(A));
    // tlapack_check_false(nrows(A) != ((side == Side::Left) ? m : n));
    // tlapack_check_false(nrows(C) != m);
    // tlapack_check_false(ncols(C) != n);

    if (side == Side::Left) {
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                // or uplo == Uplo::General
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < m; ++i) {
                        const scalar_type<alpha_t, TB> alphaTimesBij =
                            alpha * B(i, j);
                        scalar_type<TA, TB> sum(0);

                        for (idx_t k = 0; k < i; ++k) {
                            C(k, j) += A(k, i) * alphaTimesBij;
                            sum += conj(A(k, i)) * B(k, j);
                        }
                        C(i, j) = beta * C(i, j) +
                                  real(A(i, i)) * alphaTimesBij + alpha * sum;
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = m - 1; i != idx_t(-1); --i) {
                        const scalar_type<alpha_t, TB> alphaTimesBij =
                            alpha * B(i, j);
                        scalar_type<TA, TB> sum(0);

                        for (idx_t k = i + 1; k < m; ++k) {
                            C(k, j) += A(k, i) * alphaTimesBij;
                            sum += conj(A(k, i)) * B(k, j);
                        }
                        C(i, j) = beta * C(i, j) +
                                  real(A(i, i)) * alphaTimesBij + alpha * sum;
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            // Trans
            if (uplo == Uplo::Upper) {
                // or uplo == Uplo::General
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < j; i++) {
                            sum += A(i, j) * B(k, i);
                        }
                        for (idx_t i = j; i < n; i++) {
                            sum += A(j, i) * B(k, i);
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i <= j; i++) {
                            sum += A(j, i) * B(k, i);
                        }
                        for (idx_t i = j + 1; i < n; i++) {
                            sum += A(i, j) * B(k, i);
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
        }
        else {
            // TransConj
            if (uplo == Uplo::Upper) {
                // or uplo == Uplo::General
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < j; i++) {
                            sum += conj(A(i, j)) * conj(B(k, i));
                        }
                        for (idx_t i = j; i < n; i++) {
                            sum += A(j, i) * conj(B(k, i));
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i <= j; i++) {
                            sum += A(j, i) * conj(B(k, i));
                        }
                        for (idx_t i = j + 1; i < n; i++) {
                            sum += conj(A(i, j)) * conj(B(k, i));
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
        }
    }
    else {  // side == Side::Right

        using scalar_t = scalar_type<alpha_t, TA>;

        if (trans == Op::NoTrans) {
            if (uplo != Uplo::Lower) {
                // uplo == Uplo::Upper or uplo == Uplo::General
                for (idx_t j = 0; j < n; ++j) {
                    {
                        const scalar_t alphaTimesAjj = alpha * real(A(j, j));
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) = beta * C(i, j) + B(i, j) * alphaTimesAjj;
                    }

                    for (idx_t k = 0; k < j; ++k) {
                        const scalar_t alphaTimesAkj = alpha * A(k, j);
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) += B(i, k) * alphaTimesAkj;
                    }

                    for (idx_t k = j + 1; k < n; ++k) {
                        const scalar_t alphaTimesAjk = alpha * conj(A(j, k));
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) += B(i, k) * alphaTimesAjk;
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    {
                        const scalar_t alphaTimesAjj = alpha * real(A(j, j));
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) = beta * C(i, j) + B(i, j) * alphaTimesAjj;
                    }

                    for (idx_t k = 0; k < j; ++k) {
                        const scalar_t alphaTimesAjk = alpha * conj(A(j, k));
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) += B(i, k) * alphaTimesAjk;
                    }

                    for (idx_t k = j + 1; k < n; ++k) {
                        const scalar_t alphaTimesAkj = alpha * A(k, j);
                        for (idx_t i = 0; i < m; ++i)
                            C(i, j) += B(i, k) * alphaTimesAkj;
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            // Trans
            if (uplo == Uplo::Upper) {
                // or uplo == Uplo::General
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < k; i++) {
                            sum += B(i, j) * A(i, k);
                        }
                        for (idx_t i = k; i < m; i++) {
                            sum += B(i, j) * A(k, i);
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < k; i++) {
                            sum += B(i, j) * A(k, i);
                        }
                        for (idx_t i = k; i < m; i++) {
                            sum += B(i, j) * A(i, k);
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
        }
        else {
            // TransConj
            if (uplo == Uplo::Upper) {
                // or uplo == Uplo::General
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < k; i++) {
                            sum += conj(B(i, j)) * A(i, k);
                        }
                        for (idx_t i = k; i < m; i++) {
                            sum += conj(B(i, j)) * conj(A(k, i));
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
            else {
                // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t k = 0; k < m; k++) {
                        T sum(0);
                        for (idx_t i = 0; i < k; i++) {
                            sum += conj(B(i, j)) * conj(A(k, i));
                        }
                        for (idx_t i = k; i < m; i++) {
                            sum += conj(B(i, j)) * A(i, k);
                        }
                        C(j, k) = alpha * sum + beta * C(j, k);
                    }
                }
            }
        }
    }
}

//---------------------------------------------------------
template <TLAPACK_SMATRIX matrixA_t,
          TLAPACK_SMATRIX matrixB_t,
          TLAPACK_SMATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          TLAPACK_SCALAR beta_t>
void mult_hehe(Uplo uplo,
               const alpha_t& alpha,
               matrixA_t& A,
               matrixB_t& B,
               const beta_t& beta,
               matrixC_t& C)
{
    using TB = type_t<matrixB_t>;
    using TA = type_t<matrixA_t>;
    typedef tlapack::real_type<TA> real_t;
    using idx_t = tlapack::size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;

    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    if (m != n) return;

    if (n <= 1) {
        C(0, 0) = alpha * real(A(0, 0)) * real(B(0, 0)) + beta * real(C(0, 0));
        return;
    }

    const idx_t n0 = n / 2;

    if (uplo == Uplo::Upper) {
        const idx_t n0 = n / 2;
        auto A00 = slice(A, range(0, n0), range(0, n0));
        auto A01 = slice(A, range(0, n0), range(n0, n));
        auto A11 = slice(A, range(n0, n), range(n0, n));

        auto B00 = slice(B, range(0, n0), range(0, n0));
        auto B01 = slice(B, range(0, n0), range(n0, n));
        auto B11 = slice(B, range(n0, n), range(n0, n));
        auto C00 = slice(C, range(0, n0), range(0, n0));
        auto C01 = slice(C, range(0, n0), range(n0, n));
        auto C10 = slice(C, range(n0, n), range(0, n0));
        auto C11 = slice(C, range(n0, n), range(n0, n));

        // A00*B00 = C00
        mult_hehe(Uplo::Upper, alpha, A00, B00, beta, C00);

        // A01*B01^H + (A00*B00 + C00) = C00
        gemm(Op::NoTrans, Op::ConjTrans, alpha, A01, B01, real_t(1), C00);

        // A00*B01 + C01 = C01
        hemm_brian(Side::Left, Uplo::Upper, Op::NoTrans, alpha, A00, B01, beta,
                   C01);  // beta

        //(A00*B01 + C01) + A01B11 = C
        hemm_brian(Side::Right, Uplo::Upper, Op::NoTrans, alpha, B11, A01,
                   real_t(1), C01);

        // Creating B01^H and A01^H
        std::vector<TB> B01H_((n - n0) * n0);
        tlapack::LegacyMatrix<TB> B01H(n - n0, n0, &B01H_[0], n - n0);

        std::vector<TA> A01H_((n - n0) * n0);
        tlapack::LegacyMatrix<TA> A01H(n - n0, n0, &A01H_[0], n - n0);

        std::cout << std::endl;
        // conjtranspose(B01, B01H);
        // conjtranspose(A01, A01H);
        for (idx_t i = 0; i < n - n0; ++i)
            for (idx_t j = 0; j < n0; ++j) {
                B01H(i, j) = conj(B01(j, i));
                A01H(i, j) = conj(A01(j, i));
            }

        // A11 * B01H + C10 = C10 // Here Works
        hemm_brian(Side::Left, Uplo::Upper, Op::ConjTrans, alpha, A11, B01,
                   beta,
                   C10);  // beta

        // //A01^H * B00 + (A11*B01^H) // Here Works
        hemm_brian(Side::Right, Uplo::Upper, Op::ConjTrans, alpha, B00, A01,
                   real_t(1), C10);

        // A11*B11
        mult_hehe(Uplo::Upper, alpha, A11, B11, beta, C11);

        // A01^H * B01 + A11*B11
        gemm(Op::ConjTrans, Op::NoTrans, alpha, A01, B01, real_t(1), C11);

        return;
    }

    else {
        auto A00 = slice(A, range(0, n0), range(0, n0));
        auto A10 = slice(A, range(n0, n), range(0, n0));
        auto A11 = slice(A, range(n0, n), range(n0, n));

        auto B00 = slice(B, range(0, n0), range(0, n0));
        auto B10 = slice(B, range(n0, n), range(0, n0));
        auto B11 = slice(B, range(n0, n), range(n0, n));
        auto C00 = slice(C, range(0, n0), range(0, n0));
        auto C01 = slice(C, range(0, n0), range(n0, n));
        auto C10 = slice(C, range(n0, n), range(0, n0));
        auto C11 = slice(C, range(n0, n), range(n0, n));

        std::cout << std::endl;

        // A00*B00 = C00
        mult_hehe(Uplo::Lower, alpha, A00, B00, beta, C00);

        // A01^H*B10 + C00 = C00
        gemm(Op::ConjTrans, Op::NoTrans, alpha, A10, B10, real_t(1), C00);

        // A10*B00 + C10 = C10
        hemm_brian(Side::Right, Uplo::Lower, Op::NoTrans, alpha, B00, A10, beta,
                   C10);

        // A11*B10 + C10 = C10
        hemm_brian(Side::Left, Uplo::Lower, Op::NoTrans, alpha, A11, B10,
                   real_t(1), C10);

        // Creating vectors to put conjtranspose
        std::vector<TB> B01H_(n0 * (n - n0));
        tlapack::LegacyMatrix<TB> B01H(n0, n - n0, &B01H_[0], n0);

        std::vector<TB> A01H_(n0 * (n - n0));
        tlapack::LegacyMatrix<TB> A01H(n0, n - n0, &A01H_[0], n0);

        // conjtranspose(B10, B01H);
        // conjtranspose(A10, A01H);

        for (idx_t i = 0; i < n0; ++i)
            for (idx_t j = 0; j < n - n0; ++j) {
                B01H(i, j) = conj(B10(j, i));
                A01H(i, j) = conj(A10(j, i));
            }

        // A00*B01^H + C01 = C01 // Here Works
        hemm_brian(Side::Left, Uplo::Lower, Op::ConjTrans, alpha, A00, B10,
                   beta, C01);

        // A01^H*B11 + C01 = C01 //Here Works
        hemm_brian(Side::Right, Uplo::Lower, Op::ConjTrans, alpha, B11, A10,
                   real_t(1), C01);

        // A11*B11 = C11
        mult_hehe(Uplo::Lower, alpha, A11, B11, beta, C11);

        // alpha(A10H*B10^H) + 1(C11) = C11
        gemm(Op::NoTrans, Op::ConjTrans, alpha, A10, B10, real_t(1), C11);

        return;
    }
}

//---------------------------------------------------------------------------------
template <typename T>
void run(size_t n, size_t k)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    size_t info;
    typedef tlapack::real_type<T> real_t;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // // Turn it off if m or n are large
    // // bool verbose = false;
    // bool verbose = true;

    // std::vector<T> D_(n * n);
    // tlapack::LegacyMatrix<T> D(n, n, &D_[0], n);

    // std::vector<T> E_(n * n);
    // tlapack::LegacyMatrix<T> E(n, n, &E_[0], n);

    // std::vector<T> F_(n * n);
    // tlapack::LegacyMatrix<T> F(n, n, &F_[0], n);

    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < n; j++) {
    //         D(i, j) = i;
    //         D(j, i) = i;
    //         E(i, j) = i + 2;
    //         E(j, i) = i + 3;
    //         F(i, j) = i + 2;
    //         F(j, i) = i;
    //     }
    // }

    // std::cout << "\nE =";
    // printMatrix(E);
    // std::cout << std::endl;
    // std::cout << "\nF before=:";
    // printMatrix(F);
    // std::cout << std::endl;

    // // Matrices
    // std::vector<T> b_(n * k);
    // tlapack::LegacyMatrix<T> b(n, k, &b_[0], n);

    // std::vector<T> C_(n * n);
    // tlapack::LegacyMatrix<T> C(n, n, &C_[0], n);

    // std::vector<T> y_(n * k);
    // tlapack::LegacyMatrix<T> y(n, k, &y_[0], n);

    // Lower
    // // // Initialize arrays with junk
    // for (size_t j = 0; j < n; ++j) {
    //     for (size_t i = 0; i < j; ++i) {
    //         if constexpr (tlapack::is_complex<T>)
    //             D(i, j) = T(static_cast<float>(0xDEADBEEF),
    //                         static_cast<float>(0xDEADBEEF));
    //         else
    //             D(i, j) = T(static_cast<float>(0xDEADBEEF));
    //     }
    // }

    // Test
    // Trans///////////////////////////////////////////////////////////////////////////////////

    // left A Matrix
    std::vector<T> leftA_(n * k);
    tlapack::LegacyMatrix<T> leftA(n, k, &leftA_[0], n);

    // right A Matrix
    std::vector<T> rightA_(k * n);
    tlapack::LegacyMatrix<T> rightA(k, n, &rightA_[0], k);

    // left C Matrix
    std::vector<T> leftC_(n * k);
    tlapack::LegacyMatrix<T> leftC(n, k, &leftC_[0], n);

    // right C Matrix
    std::vector<T> rightC_(k * n);
    tlapack::LegacyMatrix<T> rightC(k, n, &rightC_[0], k);

    // L
    std::vector<T> L_(n * n);
    tlapack::LegacyMatrix<T> L(n, n, &L_[0], n);

    // U
    std::vector<T> U_(n * n);
    tlapack::LegacyMatrix<T> U(n, n, &U_[0], n);

    // Left B
    std::vector<T> leftB_(k * n);
    tlapack::LegacyMatrix<T> leftB(k, n, &leftB_[0], k);

    // leftB Transpose
    std::vector<T> leftBT_(n * k);
    tlapack::LegacyMatrix<T> leftBT(n, k, &leftBT_[0], n);

    // right B
    std::vector<T> rightB_(n * k);
    tlapack::LegacyMatrix<T> rightB(n, k, &rightB_[0], n);

    // rightB Transpose
    std::vector<T> rightBT_(k * n);
    tlapack::LegacyMatrix<T> rightBT(k, n, &rightBT_[0], k);

    // Fill in L and U
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if constexpr (is_complex<T>) {
                L(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
                L(j, i) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
                U(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
                U(j, i) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));

                if (i == j) {
                    L(i, j) = T(static_cast<float>(rand()) /
                                    static_cast<float>(RAND_MAX),
                                0);
                    U(i, j) = T(static_cast<float>(rand()) /
                                    static_cast<float>(RAND_MAX),
                                0);
                }
            }
            else {
                L(i, j) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));
                L(j, i) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));
                U(i, j) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));
                U(j, i) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));
            }
        }
    }

    // Fill in right A and C
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if constexpr (is_complex<T>) {
                rightA(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            else {
                rightA(i, j) = T(static_cast<float>(rand()) /
                                 static_cast<float>(RAND_MAX));
            }
        }
    }
    lacpy(GENERAL, rightA, rightC);

    // Fill in left A and C
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if constexpr (is_complex<T>) {
                leftA(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            else {
                leftA(i, j) = T(static_cast<float>(rand()) /
                                static_cast<float>(RAND_MAX));
            }
        }
    }
    lacpy(GENERAL, leftA, leftC);

    // Fill lower triangle with deadbeef on the top right
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if constexpr (is_complex<T>) {
                L(i, j) = T(static_cast<float>(0xDEADBEEF),
                            static_cast<float>(0xDEADBEEF));
            }
            else {
                L(i, j) = T(static_cast<float>(0xDEADBEEF));
            }
        }
    }

    // Fill upper triangle with deadbeef on the bottom left
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            if constexpr (is_complex<T>) {
                U(i, j) = T(static_cast<float>(0xDEADBEEF),
                            static_cast<float>(0xDEADBEEF));
            }
            else {
                U(i, j) = T(static_cast<float>(0xDEADBEEF));
            }
        }
    }

    // Fill left B
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if constexpr (is_complex<T>) {
                leftB(i, j) = T(i * j + 1, i * j + 1);
            }
            else {
                leftB(i, j) = i * j + 1;
            }
        }
    }

    // Fill right B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if constexpr (is_complex<T>) {
                rightB(i, j) = T(i * j + 1, i * j + 1);
            }
            else {
                rightB(i, j) = i * j + 1;
            }
        }
    }

    // Fill Transpose left BT
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if constexpr (is_complex<T>) {
                leftBT(i, j) = conj(leftB(j, i));
            }
            else {
                leftBT(i, j) = leftB(j, i);
            }
        }
    }

    // Fill Tranposee right BT
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if constexpr (is_complex<T>) {
                rightBT(i, j) = conj(rightB(j, i));
            }
            else {
                rightBT(i, j) = rightB(j, i);
            }
        }
    }

    // // Print
    // std::cout << "Starting: \n";
    // std::cout << "\nL = ";
    // printMatrix(L);

    // std::cout << "\nU = ";
    // printMatrix(U);

    // std::cout << "\nleftB = ";
    // printMatrix(leftB);

    // std::cout << "\nleftBT = ";
    // printMatrix(leftBT);

    // std::cout << "\nrightB = ";
    // printMatrix(rightB);

    // std::cout << "\nrightBT = ";
    // printMatrix(rightBT);

    real_t al = 5.2;
    real_t be = 9.4;

    // Test Lower Right
    // Trans///////////////////////////////////////////////////////////////////////////////////
    // Fill in right A and C
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if constexpr (is_complex<T>) {
                rightA(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            else {
                rightA(i, j) = T(static_cast<float>(rand()) /
                                 static_cast<float>(RAND_MAX));
            }
        }
    }
    lacpy(GENERAL, rightA, rightC);

    // Fill in left A and C
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if constexpr (is_complex<T>) {
                leftA(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            else {
                leftA(i, j) = T(static_cast<float>(rand()) /
                                static_cast<float>(RAND_MAX));
            }
        }
    }
    lacpy(GENERAL, leftA, leftC);

    std::cout << "\n\nTesting Right Lower";

    hemm(Side::Right, Uplo::Lower, al, L, rightBT, be, rightA);
    real_t rightNormA = lange(FROB_NORM, rightA);

    hemm_brian(Side::Right, Uplo::Lower, Op::ConjTrans, al, L, rightB, be,
               rightC);

    for (idx_t i = 0; i < k; i++) {
        for (idx_t j = 0; j < n; j++) {
            rightA(i, j) -= rightC(i, j);
        }
    }

    real_t rightNormC = lange(FROB_NORM, rightA);
    std::cout << "\n\nThis is rigth norm of A - C = "
              << rightNormC / rightNormA;

    // Test Upper Right
    // Trans///////////////////////////////////////////////////////////////////////////////////
    // Fill in right A and C
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if constexpr (is_complex<T>) {
                rightA(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            else {
                rightA(i, j) = T(static_cast<float>(rand()) /
                                 static_cast<float>(RAND_MAX));
            }
        }
    }
    lacpy(GENERAL, rightA, rightC);

    // Fill in left A and C
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if constexpr (is_complex<T>) {
                leftA(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            else {
                leftA(i, j) = T(static_cast<float>(rand()) /
                                static_cast<float>(RAND_MAX));
            }
        }
    }
    lacpy(GENERAL, leftA, leftC);

    std::cout << "\n\nTesting Right Upper";

    hemm(Side::Right, Uplo::Upper, al, U, rightBT, be, rightA);
    rightNormA = lange(FROB_NORM, rightA);

    hemm_brian(Side::Right, Uplo::Upper, Op::ConjTrans, al, U, rightB, be,
               rightC);

    for (idx_t i = 0; i < k; i++) {
        for (idx_t j = 0; j < n; j++) {
            rightA(i, j) -= rightC(i, j);
        }
    }

    rightNormC = lange(FROB_NORM, rightA);
    std::cout << "\n\nThis is rigth norm of A - C = "
              << rightNormC / rightNormA;

    // Test Lower Left
    // Trans///////////////////////////////////////////////////////////////////////////////////
    // Fill in right A and C
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if constexpr (is_complex<T>) {
                rightA(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            else {
                rightA(i, j) = T(static_cast<float>(rand()) /
                                 static_cast<float>(RAND_MAX));
            }
        }
    }
    lacpy(GENERAL, rightA, rightC);

    // Fill in left A and C
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if constexpr (is_complex<T>) {
                leftA(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            else {
                leftA(i, j) = T(static_cast<float>(rand()) /
                                static_cast<float>(RAND_MAX));
            }
        }
    }
    lacpy(GENERAL, leftA, leftC);

    std::cout << "\n\nTesting Left Lower";

    hemm(Side::Left, Uplo::Lower, al, L, leftBT, be, leftA);
    real_t leftNormA = lange(FROB_NORM, leftA);

    hemm_brian(Side::Left, Uplo::Lower, Op::ConjTrans, al, L, leftB, be, leftC);

    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < k; j++) {
            leftA(i, j) -= leftC(i, j);
        }
    }

    real_t leftNormC = lange(FROB_NORM, leftA);
    std::cout << "\n\nThis is rigth norm of A - C = " << leftNormC / leftNormA;

    // Test Upper Left
    // Trans///////////////////////////////////////////////////////////////////////////////////
    // Fill in right A and C
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if constexpr (is_complex<T>) {
                rightA(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            else {
                rightA(i, j) = T(static_cast<float>(rand()) /
                                 static_cast<float>(RAND_MAX));
            }
        }
    }
    lacpy(GENERAL, rightA, rightC);

    // Fill in left A and C
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if constexpr (is_complex<T>) {
                leftA(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
            else {
                leftA(i, j) = T(static_cast<float>(rand()) /
                                static_cast<float>(RAND_MAX));
            }
        }
    }
    lacpy(GENERAL, leftA, leftC);

    std::cout << "\n\nTesting Left Upper";

    hemm(Side::Left, Uplo::Upper, al, U, leftBT, be, leftA);
    leftNormA = lange(FROB_NORM, leftA);

    hemm_brian(Side::Left, Uplo::Upper, Op::ConjTrans, al, U, leftB, be, leftC);

    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < k; j++) {
            leftA(i, j) -= leftC(i, j);
        }
    }

    leftNormC = lange(FROB_NORM, leftA);
    std::cout << "\n\nThis is rigth norm of A - C = " << leftNormC / leftNormA;

    // for (size_t j = 0; j < n; ++j) {
    //     for (size_t i = j + 1; i < n; ++i) {
    //         if constexpr (tlapack::is_complex<T>)
    //             D(i, j) = T(static_cast<float>(0xDEADBEEF),
    //                         static_cast<float>(0xDEADBEEF));
    //         else
    //             D(i, j) = T(static_cast<float>(0xDEADBEEF));
    //     }
    // }
    // std::cout << "\nD =";
    // printMatrix(D);
    // std::cout << std::endl;

    // // Creating vectors to put conjtranspose
    // std::vector<T> ET_(n * n);
    // tlapack::LegacyMatrix<T> ET(n, n, &ET_[0], n);

    // for (idx_t i = 0; i < n; ++i)
    //     for (idx_t j = 0; j < n; ++j) {
    //         ET(i, j) = E(j, i);
    //     }

    // std::cout << "\nET =";
    // printMatrix(ET);
    // std::cout << std::endl;

    // hemm_brian(Side::Right, Uplo::Lower, Op::Trans, 1, D, E, 0, A);

    // std::cout << "A = ";
    // printMatrix(A);
    // std::cout << std::endl;

    // hemm(Side::Right, Uplo::Lower, 1, D, ET, 0, C);

    // std::cout << "C = ";
    // printMatrix(C);

    // for (idx_t i = 0; i < n; i++) {
    //     for (idx_t j = 0; j < n; j++) {
    //         A(i, j) -= C(i, j);
    //     }
    // }

    // std::cout << std::endl;
    // std::cout << "A = ";
    // printMatrix(A);
    // std::cout << std::endl;

    // Test//////////////////////////////////////////////////////////////////////////////////////

    // // multiplying two upper triangular hermitian matrices
    // // one function with alpha/beta one without so C <- AB and C <-
    // // alpha(AB) + beta(C) make it work for upper and lower
    // mult_hehe(Uplo::Upper, 1.5, D, E, 7.8, F);

    // std::cout << "\nF =";
    // printMatrix(F);
    // std::cout << std::endl;

    // // Generate a random matrix in A
    // for (size_t j = 0; j < n; ++j) {
    //     for (size_t i = 0; i <= j; ++i) {
    //         if constexpr (tlapack::is_complex<T>)
    //             A(i, j) = T(
    //                 static_cast<float>(rand()) /
    //                 static_cast<float>(RAND_MAX), static_cast<float>(rand())
    //                 / static_cast<float>(RAND_MAX));
    //         else
    //             A(i, j) = T(static_cast<float>(rand()) /
    //                         static_cast<float>(RAND_MAX));
    //     }
    //     A(j, j) =
    //         T(n + static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    // }

    // // for (size_t j = 0; j < k; ++j)
    // // for (size_t i = 0; i < n; ++i) {
    // // if constexpr (tlapack::is_complex<T>)
    // // b(i, j) = T(static_cast<float>(rand())/static_cast<float>(RAND_MAX),
    // // static_cast<float>(rand())/static_cast<float>(RAND_MAX)); else b(i, j)
    // =
    // // T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    // // }

    // if (verbose) {
    //     std::cout << std::endl << "A = ";
    //     printMatrix(A);
    //     std::cout << std::endl;
    // }
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int n, k;

    // Default arguments
    n = (argc < 2) ? 200 : atoi(argv[1]);
    k = (argc < 3) ? 300 : atoi(argv[2]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float >( %d, %d )", n, k);
    run<float>(n, k);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", n, k);
    run<double>(n, k);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )", n, k);
    run<long double>(n, k);
    printf("-----------------------\n");

    printf("run complex< float >( %d, %d )", n, k);
    run<std::complex<float> >(n, k);
    printf("-----------------------\n");

    printf("run complex< double >( %d, %d )", n, k);
    run<std::complex<double> >(n, k);
    printf("-----------------------\n");

    return 0;
}
