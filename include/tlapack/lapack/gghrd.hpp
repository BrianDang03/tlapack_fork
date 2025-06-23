/// @file gghrd.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zgghrd.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GGHRD_HH
#define TLAPACK_GGHRD_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/rot.hpp"
#include "tlapack/blas/rotg.hpp"

namespace tlapack {

/** Reduces a pair of real square matrices (A, B) to generalized upper
 *  Hessenberg form using unitary transformations, where A is a general matrix
 *  and B is upper triangular.
 *
 * @return  0 if success
 *
 * @param[in] wantq boolean
 * @param[in] wantz boolean
 * @param[in] ilo integer
 * @param[in] ihi integer
 * @param[in,out] A n-by-n matrix.
 * @param[in,out] B n-by-n matrix.
 * @param[in,out] Q n-by-n matrix.
 * @param[in,out] Z n-by-n matrix.
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX A_t,
          TLAPACK_SMATRIX B_t,
          TLAPACK_SMATRIX Q_t,
          TLAPACK_SMATRIX Z_t>
int gghrd(bool wantq,
          bool wantz,
          size_type<A_t> ilo,
          size_type<A_t> ihi,
          A_t& A,
          B_t& B,
          Q_t& Q,
          Z_t& Z)
{
    using T = type_t<A_t>;
    using idx_t = size_type<A_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t n = ncols(A);

    // check arguments
    tlapack_check(ilo >= 0 && ilo < n);
    tlapack_check(ihi > ilo && ihi <= n);
    tlapack_check(n == nrows(A));
    tlapack_check(n == ncols(B));
    tlapack_check(n == nrows(B));
    tlapack_check(n == ncols(Q));
    tlapack_check(n == nrows(Q));
    tlapack_check(n == ncols(Z));
    tlapack_check(n == nrows(Z));

    // quick return
    if (n <= 1) return 0;

    // Zero out lower triangle of B
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = j + 1; i < n; ++i)
            B(i, j) = (T)0;

    for (idx_t j = ilo; j + 2 < ihi; ++j) {
        // Apply sequence of rotations
        for (idx_t i = ihi - 1; i > j + 1; --i) {
            //
            // Rotate rows i-1 and i to eliminate A(i,j)
            //
            real_type<T> c;
            T s;
            rotg(A(i - 1, j), A(i, j), c, s);
            A(i, j) = (T)0;
            {
                auto a1 = slice(A, i - 1, range(j + 1, n));
                auto a2 = slice(A, i, range(j + 1, n));
                rot(a1, a2, c, s);
            }
            {
                auto b1 = slice(B, i - 1, range(i - 1, n));
                auto b2 = slice(B, i, range(i - 1, n));
                rot(b1, b2, c, s);
            }
            if (wantq) {
                auto q1 = slice(Q, range(0, n), i - 1);
                auto q2 = slice(Q, range(0, n), i);
                rot(q1, q2, c, conj(s));
            }
            //
            // The previous step introduced fill-in in B, remove it now
            //
            rotg(B(i, i), B(i, i - 1), c, s);
            B(i, i - 1) = (T)0;
            {
                auto a1 = slice(A, range(0, ihi), i);
                auto a2 = slice(A, range(0, ihi), i - 1);
                rot(a1, a2, c, s);
            }
            {
                auto b1 = slice(B, range(0, i), i);
                auto b2 = slice(B, range(0, i), i - 1);
                rot(b1, b2, c, s);
            }
            if (wantz) {
                auto z1 = slice(Z, range(0, n), i);
                auto z2 = slice(Z, range(0, n), i - 1);
                rot(z1, z2, c, s);
            }
        }
    }

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_GGHRD_HH
