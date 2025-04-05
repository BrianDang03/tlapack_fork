/// @file ul_mult.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_ul_mult_HH
#define TLAPACK_ul_mult_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/trmm.hpp"

namespace tlapack {

/** ul_mult computes the matrix product of an upper triangular matrix U and a
 * lower triangular unital matrix L Given input matrix A, nonzero part of L is
 * the subdiagonal of A and on the diagonal of L is assumed to be 1, and the
 * nonzero part of U is diagonal and super-diagonal part of A
 *
 * @return  0 if success.
 *
 * @param[in,out] A n-by-n complex matrix.
 *      On entry, subdiagonal of A contains L(lower triangular and unital) and
 * diagonal and superdiagonal part of A contains U(upper triangular). On exit, A
 * is overwritten by L*U
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SMATRIX matrix_t>
int ul_mult(matrix_t& A)
{
    using idx_t = size_type<matrix_t>;
    using T = type_t<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // check arguments
    tlapack_check(nrows(A) == ncols(A));

    // constant
    const idx_t n = ncols(A);

    // if L and U are 1-by-1, then L is 1 and we simply UL=A(0,0)
    if (n == 1) return 0;

    // if n>1
    idx_t n0 = n / 2;

    // break A into four parts
    auto A00 = tlapack::slice(A, range(0, n0), range(0, n0));
    auto A10 = tlapack::slice(A, range(n0, n), range(0, n0));
    auto A01 = tlapack::slice(A, range(0, n0), range(n0, n));
    auto A11 = tlapack::slice(A, range(n0, n), range(n0, n));

    // calculate top left corner
    ul_mult(A00);
    tlapack::gemm(NO_TRANS, NO_TRANS, T(1), A01, A10, T(1), A00);

    // calculate bottom left corner
    tlapack::trmm(LEFT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, T(1), A11,
                  A10);

    // calculate top right
    tlapack::trmm(RIGHT_SIDE, LOWER_TRIANGLE, NO_TRANS, UNIT_DIAG, T(1), A11,
                  A01);

    // calculate bottom right
    ul_mult(A11);

    return 0;

}  // ul_mult

}  // namespace tlapack

#endif  // TLAPACK_ul_mult_HH
