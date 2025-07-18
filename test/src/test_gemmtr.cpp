/// @file test_gemmtr.cpp
/// @author Luis Carlos Gutierrez, Kyle Cunningham, and Henricus Bouwmeester
/// University of Colorado Denver, USA
/// @brief Test gemmtr
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lantr.hpp>
// Other routines
#include <tlapack/lapack/gemmtr.hpp>

using namespace tlapack;

// Helper to set alpha and beta safely for both real and complex types
template <typename T>
void setScalar(T& alpha, real_type<T> aReal, real_type<T> aImag)
{
    alpha = aReal;
}

template <typename T>
void setScalar(std::complex<T>& alpha, real_type<T> aReal, real_type<T> aImag)
{
    alpha.real(aReal);
    alpha.imag(aImag);
}

TEMPLATE_TEST_CASE("check for gemmtr multiplication",
                   "[gemmtr]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;
    idx_t n, k;

    n = GENERATE(3, 15);
    k = GENERATE(7, 12);
    const Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);
    const Op transA = GENERATE(Op::NoTrans, Op::Trans, Op::ConjTrans);
    const Op transB = GENERATE(Op::NoTrans, Op::Trans, Op::ConjTrans);

    // Generate numbers for alpha and beta
    T alpha, beta;

    srand(3);

    // Random number engine (seed with a random device)
    std::random_device rd;
    std::mt19937 gen(rd());

    // Uniform distribution: 0 or 1
    std::uniform_int_distribution<> dist(0, 1);

    // Generate either -1 or 1
    float valueA = dist(gen) == 0 ? -1.0 : 1.0;
    float valueB = dist(gen) == 0 ? -1.0 : 1.0;

    real_t aReal = real_t(valueA * (float)rand() / (float)RAND_MAX);
    real_t aImag = real_t(valueB * (float)rand() / (float)RAND_MAX);
    real_t bReal = real_t(valueA * (float)rand() / (float)RAND_MAX);
    real_t bImag = real_t(valueB * (float)rand() / (float)RAND_MAX);

    setScalar(alpha, aReal, aImag);
    setScalar(beta, bReal, bImag);

    DYNAMIC_SECTION("n = " << n << " k = " << k << " uplo = " << uplo
                           << " transA = " << transA << " transB = " << transB)
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n + k) * eps;

        idx_t na;
        idx_t ka;
        idx_t nb;
        idx_t kb;

        // Correcting Matrix Dimension when transposed
        if (transA == Op::NoTrans) {
            na = n;
            ka = k;
        }
        else {
            na = k;
            ka = n;
        }

        if (transB == Op::NoTrans) {
            nb = n;
            kb = k;
        }
        else {
            nb = k;
            kb = n;
        }

        // Generating Matrices A, B, C0, C1, C2
        std::vector<T> A_;
        auto A = new_matrix(A_, na, ka);
        std::vector<T> B_;
        auto B = new_matrix(B_, kb, nb);
        std::vector<T> C0_;
        auto C0 = new_matrix(C0_, n, n);
        std::vector<T> C1_;
        auto C1 = new_matrix(C1_, n, n);
        std::vector<T> C2_;
        auto C2 = new_matrix(C2_, n, n);

        // Matrix initilaizations
        mm.random(A);
        mm.random(B);
        mm.random(C0);

        lacpy(GENERAL, C0, C1);
        lacpy(GENERAL, C0, C2);

        {
            // Calculate residuals
            real_t normc = lantr(MAX_NORM, uplo, NON_UNIT_DIAG, C0);
            real_t norma = lange(MAX_NORM, A);
            real_t normb = lange(MAX_NORM, B);

            // Calling gemmtr and gemm
            gemmtr(uplo, transA, transB, alpha, A, B, beta, C1);

            gemm(transA, transB, alpha, A, B, beta, C2);

            // Comparing gemmtr vs gemm
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; j++)  // Check of upper part.
                    for (idx_t i = 0; i <= j; i++)
                        C1(i, j) -= C2(i, j);

                real_t normres =
                    lantr(MAX_NORM, UPPER_TRIANGLE, NON_UNIT_DIAG, C1);
                CHECK(normres <=
                      tol * (abs1(alpha) * norma * normb + abs1(beta) * normc));

                real_t sum(0);
                for (idx_t j = 0; j < n; j++)  // Check strictly lower part
                    for (idx_t i = j + 1; i < n; i++)
                        sum += abs1(C1(i, j) - C0(i, j));

                CHECK(sum == real_t(0));
            }
            else {
                for (idx_t i = 0; i < n; i++)  // Check of lower part
                    for (idx_t j = 0; j <= i; j++)
                        C1(i, j) -= C2(i, j);

                real_t normres =
                    lantr(MAX_NORM, LOWER_TRIANGLE, NON_UNIT_DIAG, C1);
                CHECK(normres <=
                      tol * (abs1(alpha) * norma * normb + abs1(beta) * normc));

                real_t sum(0);
                for (idx_t i = 0; i < n; i++)  // Check strictly upper part
                    for (idx_t j = i + 1; j < n; j++)
                        sum += abs1(C1(i, j) - C0(i, j));

                CHECK(sum == real_t(0));
            }
        }
    }
}
