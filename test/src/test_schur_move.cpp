/// @file test_schur_swap.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test moving of multiple blocks in generalized schur form
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
#include <tlapack/lapack/schur_move.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("move of eigenvalue block gives correct results",
                   "[eigenvalues]",
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

    const T zero(0);
    const T one(1);
    idx_t n = 10;

    idx_t ifst = GENERATE(0, 2, 6, 9);
    idx_t ilst = GENERATE(0, 2, 6, 9);
    idx_t n1 = GENERATE(1, 2);
    idx_t n2 = GENERATE(1, 2);

    if (is_real<T> || (n1 == 1 && n2 == 1)) {
        // ifst and ilst point to the same block, n1 must be equal to n2 for
        // the test to make sense.
        if (ifst == ilst and n1 != n2) n2 = n1;

        const real_t eps = uroundoff<real_t>();
        const real_t tol = real_t(1.0e2 * n) * eps;

        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);
        std::vector<T> Q_;
        auto Q = new_matrix(Q_, n, n);
        std::vector<T> A_copy_;
        auto A_copy = new_matrix(A_copy_, n, n);

        // Generate random matrix in Schur form
        mm.random(A);

        // Zero out the lower triangular part
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = j + 1; i < n; ++i)
                A(i, j) = zero;

        if (n1 == 2) {
            if (ifst < n - 1)
                A(ifst + 1, ifst) = rand_helper<T>(mm.gen);
            else
                A(ifst, ifst - 1) = rand_helper<T>(mm.gen);
        }
        if (n2 == 2) {
            if (ilst < n - 1)
                A(ilst + 1, ilst) = rand_helper<T>(mm.gen);
            else
                A(ilst, ilst - 1) = rand_helper<T>(mm.gen);
        }

        if (is_real<T>) {
            // Put a 2x2 block in the middle
            A(5, 4) = rand_helper<T>(mm.gen);
        }

        lacpy(GENERAL, A, A_copy);
        laset(GENERAL, zero, one, Q);

        DYNAMIC_SECTION("ifst = " << ifst << " n1 = " << n1
                                  << " ilst = " << ilst << " n2 =" << n2)
        {
            schur_move(true, A, Q, ifst, ilst);
            // Calculate residuals

            std::vector<T> res_;
            auto res = new_matrix(res_, n, n);
            std::vector<T> work_;
            auto work = new_matrix(work_, n, n);
            auto orth_res_norm = check_orthogonality(Q, res);
            CHECK(orth_res_norm <= tol);

            auto normA = tlapack::lange(tlapack::FROB_NORM, A);
            auto simil_res_norm =
                check_similarity_transform(A_copy, Q, A, res, work);
            CHECK(simil_res_norm <= tol * normA);
        }
    }
}