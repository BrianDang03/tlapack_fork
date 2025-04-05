/// @file ungql.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @note Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/zungql.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_UNGQL_HH
#define TLAPACK_UNGQL_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larft.hpp"
#include "tlapack/lapack/ungq.hpp"

namespace tlapack {

/**
 * Options struct for ungql
 */
struct UngqlOpts {
    size_t nb = 32;  ///< Block size
};

/**
 * @brief Generates an m-by-n matrix Q with orthonormal columns,
 *        which is defined as the last n columns of a product of k elementary
 *        reflectors of order m
 * \[
 *     Q  =  H_k ... H_2 H_1
 * \]
 *        The reflectors are stored in the matrix A as returned by geqlf
 *
 * @param[in,out] A m-by-n matrix.
 *      On entry, the (n+k-i)-th column must contains the vector which defines
 *      the elementary reflector $H_i$, for $i=0,1,...,k-1$, as returned by
 *      geqlf. On exit, the m-by-n matrix $Q$.
 *
 * @param[in] tau Real vector of length min(m,n).
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *
 * @return 0 if success
 *
 * @ingroup computational
 */
template <TLAPACK_SMATRIX matrix_t, TLAPACK_SVECTOR vector_t>
int ungql(matrix_t& A, const vector_t& tau, const UngqlOpts& opts = {})
{
    return ungq(BACKWARD, COLUMNWISE_STORAGE, A, tau, UngqOpts{opts.nb});
}

}  // namespace tlapack

#endif  // TLAPACK_UNGQL_HH
