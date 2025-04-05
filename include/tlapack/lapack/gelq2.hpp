/// @file gelq2.hpp
/// @author Yuxin Cai, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgelq2.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GELQ2_HH
#define TLAPACK_GELQ2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Worspace query of gelq2()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param tauw Not referenced.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
constexpr WorkInfo gelq2_worksize(const matrix_t& A, const vector_t& tauw)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);

    if (m > 1) {
        auto&& C = rows(A, range{1, m});
        return larf_worksize<T>(RIGHT_SIDE, FORWARD, ROWWISE_STORAGE, row(A, 0),
                                tauw[0], C);
    }
    return WorkInfo(0);
}

/** @copybrief gelq2()
 * Workspace is provided as an argument.
 * @copydetails gelq2()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int gelq2_work(matrix_t& A, vector_t& tauw, work_t& work)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);

    // check arguments
    tlapack_check_false((idx_t)size(tauw) < k);

    for (idx_t j = 0; j < k; ++j) {
        // Define w := A(j,j:n)
        auto w = slice(A, j, range(j, n));

        // Generate elementary reflector H(j) to annihilate A(j,j+1:n)
        larfg(FORWARD, ROWWISE_STORAGE, w, tauw[j]);

        // If either condition is satisfied, Q11 will not be empty
        if (j < k - 1 || k < m) {
            // Apply H(j) to A(j+1:m,j:n) from the right
            auto Q11 = slice(A, range(j + 1, m), range(j, n));
            larf_work(RIGHT_SIDE, FORWARD, ROWWISE_STORAGE, w, tauw[j], Q11,
                      work);
        }
    }

    return 0;
}

/** Computes an LQ factorization of a complex m-by-n matrix A using
 *  an unblocked algorithm.
 *
 * The matrix Q is represented as a product of elementary reflectors.
 * \[
 *          Q = H(k)**H ... H(2)**H H(1)**H,
 * \]
 * where k = min(m,n). Each H(j) has the form
 * \[
 *          H(j) = I - tauw * w * w**H
 * \]
 * where tauw is a complex scalar, and w is a complex vector with
 * \[
 *          w[0] = w[1] = ... = w[j-1] = 0; w[j] = 1,
 * \]
 * with w[j+1]**H through w[n]**H is stored on exit
 * in the jth row of A, and tauw in tauw[j].
 *
 *
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and below the diagonal of the array
 *      contain the m by min(m,n) lower trapezoidal matrix L (L is
 *      lower triangular if m <= n); the elements above the diagonal,
 *      with the array tauw, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 *
 * @param[out] tauw Complex vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int gelq2(matrix_t& A, vector_t& tauw)
{
    using T = type_t<matrix_t>;

    // functor
    Create<matrix_t> new_matrix;

    // Allocates workspace
    WorkInfo workinfo = gelq2_worksize<T>(A, tauw);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return gelq2_work(A, tauw, work);
}
}  // namespace tlapack

#endif  // TLAPACK_GELQ2_HH
