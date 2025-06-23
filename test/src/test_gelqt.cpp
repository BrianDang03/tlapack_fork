/// @file test_gelqt.cpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @brief Test GELQF and UNGL2 and output a k-by-n orthogonal matrix Q.
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

// Other routines
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/gelqt.hpp>
#include <tlapack/lapack/ungl2.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("LQ factorization of a general m-by-n matrix, blocked",
                   "[lqf]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    const T zero(0);

    idx_t m, n, k, nb;

    m = GENERATE(10, 20, 30);
    n = GENERATE(10, 20, 30);
    k = GENERATE(
        8, 10, 20,
        30);  // k is the number of rows for output Q. Can personalize it.
    nb = GENERATE(2, 3, 7, 12);  // nb is the block height. Can personalize it.

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(m * n) * eps;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> TT_;
    auto TT = new_matrix(TT_, m, nb);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, k, n);

    std::vector<T> tauw(min(m, n));

    mm.random(A);

    lacpy(GENERAL, A, A_copy);

    if (k <= n)  // k must be less than or equal to n, because we cannot get a Q
                 // bigger than n-by-n
    {
        DYNAMIC_SECTION("m = " << m << " n = " << n << " k = " << k
                               << " nb = " << nb)
        {
            gelqt(A, TT);

            // Build tauw vector from matrix TT
            for (idx_t j = 0; j < min(m, n); j += nb) {
                idx_t ib = min(nb, min(m, n) - j);

                for (idx_t i = 0; i < ib; i++)
                    tauw[i + j] = TT(i + j, i);
            }

            // Q is sliced down to the desired size of output Q (k-by-n).
            // It stores the desired number of Householder reflectors that UNGL2
            // will use.
            lacpy(GENERAL, slice(A, range(0, min(m, k)), range(0, n)), Q);

            ungl2(Q, slice(tauw, range(0, min(m, k))));

            // Wq is the identity matrix to check the orthogonality of Q
            std::vector<T> Wq_;
            auto Wq = new_matrix(Wq_, k, k);
            auto orth_Q = check_orthogonality(Q, Wq);
            CHECK(orth_Q <= tol);

            // L is sliced from A after GELQ2
            std::vector<T> L_;
            auto L = new_matrix(L_, min(k, m), k);
            laset(UPPER_TRIANGLE, zero, zero, L);
            lacpy(LOWER_TRIANGLE, slice(A, range(0, min(m, k)), range(0, k)),
                  L);

            // R stores the product of L and Q
            std::vector<T> R_;
            auto R = new_matrix(R_, min(k, m), n);

            // Test A = L * Q
            gemm(NO_TRANS, NO_TRANS, real_t(1.), L, Q, R);
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < min(m, k); ++i)
                    A_copy(i, j) -= R(i, j);

            real_t repres =
                tlapack::lange(tlapack::MAX_NORM,
                               slice(A_copy, range(0, min(m, k)), range(0, n)));
            CHECK(repres <= tol);
        }
    }
}