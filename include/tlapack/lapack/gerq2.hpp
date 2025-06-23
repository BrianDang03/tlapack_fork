/// @file gerq2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/blob/master/SRC/zgerq2.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GERQ2_HH
#define TLAPACK_GERQ2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Worspace query of gerq2()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param tau Not referenced.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
constexpr WorkInfo gerq2_worksize(const matrix_t& A, const vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);

    if (m > 1) {
        auto&& C = rows(A, range{1, m});
        return larf_worksize<T>(RIGHT_SIDE, BACKWARD, ROWWISE_STORAGE,
                                row(A, 0), tau[0], C);
    }

    return WorkInfo(0);
}

/** @copybrief gerq2()
 * Workspace is provided as an argument.
 * @copydetails gerq2()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_WORKSPACE work_t>
int gerq2_work(matrix_t& A, vector_t& tau, work_t& work)
{
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t k = min(m, n);

    // check arguments
    tlapack_check_false((idx_t)size(tau) < k);

    // quick return
    if (n <= 0) return 0;

    for (idx_t i2 = 0; i2 < k; ++i2) {
        idx_t i = k - 1 - i2;

        // Define v := A[m-1-i2,0:n-i2]
        auto v = slice(A, m - 1 - i2, range{0, n - i2});

        // Generate the (i+1)-th elementary Householder reflection on v
        larfg(BACKWARD, ROWWISE_STORAGE, v, tau[i]);

        // Apply the reflector to the rest of the matrix
        if (m > i2 + 1) {
            auto C = slice(A, range{0, m - 1 - i2}, range{0, n - i2});
            larf_work(RIGHT_SIDE, BACKWARD, ROWWISE_STORAGE, v, tau[i], C,
                      work);
        }
    }

    return 0;
}

/** Computes an RQ factorization of a matrix A.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1' H_2' ... H_k',
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[n-k+i+1:n] = 0; v[n-k+i-1] = 1,
 * \]
 * with v[1] through v[n-k+i-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the m by n matrix A.
 *      On exit, if m <= n, the upper triangle of the subarray
 *      A(0:m,n-m:n) contains the m by m upper triangular matrix R;
 *      if m >= n, the elements on and above the (m-n)-th subdiagonal
 *      contain the m by n upper trapezoidal matrix R; the remaining
 *      elements, with the array TAU, represent the unitary matrix
 *      Q as a product of elementary reflectors.
 *
 * @param[out] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @ingroup alloc_workspace
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_VECTOR vector_t>
int gerq2(matrix_t& A, vector_t& tau)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;

    // functor
    Create<matrix_t> new_matrix;

    // constants
    const idx_t n = ncols(A);

    // quick return
    if (n <= 0) return 0;

    // Allocates workspace
    WorkInfo workinfo = gerq2_worksize<T>(A, tau);
    std::vector<T> work_;
    auto work = new_matrix(work_, workinfo.m, workinfo.n);

    return gerq2_work(A, tau, work);
}

}  // namespace tlapack

#endif  // TLAPACK_GERQ2_HH
