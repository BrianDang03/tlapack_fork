/// @file householder_qr.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_HOUSEHOLDER_QR_HH
#define TLAPACK_HOUSEHOLDER_QR_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/geqr2.hpp"
#include "tlapack/lapack/geqrf.hpp"

namespace tlapack {

/// @brief Variants of the algorithm to compute the QR factorization.
enum class HouseholderQRVariant : char { Level2 = '2', Blocked = 'B' };

/// @brief Options struct for householder_qr()
struct HouseholderQROpts : public GeqrfOpts {
    HouseholderQRVariant variant = HouseholderQRVariant::Blocked;
};

/** Worspace query of householder_qr()
 *
 * @param[in] A m-by-n matrix.
 *
 * @param[in] tau min(n,m) vector.
 *
 * @param[in] opts Options.
 *      - @c opts.variant: Variant of the algorithm to use.
 *
 * @return WorkInfo The amount workspace required.
 *
 * @ingroup workspace_query
 */
template <class T, TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
constexpr WorkInfo householder_qr_worksize(const matrix_t& A,
                                           const vector_t& tau,
                                           const HouseholderQROpts& opts = {})
{
    // Call variant
    if (opts.variant == HouseholderQRVariant::Level2)
        return geqr2_worksize<T>(A, tau);
    else
        return geqrf_worksize<T>(A, tau, opts);
}

/** @copybrief householder_qr()
 * Workspace is provided as an argument.
 * @copydetails householder_qr()
 *
 * @param work Workspace. Use the workspace query to determine the size needed.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_MATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_WORKSPACE workspace_t>
int householder_qr_work(matrix_t& A,
                        vector_t& tau,
                        workspace_t& work,
                        const HouseholderQROpts& opts = {})
{
    // Call variant
    if (opts.variant == HouseholderQRVariant::Level2)
        return geqr2_work(A, tau, work);
    else
        return geqrf_work(A, tau, work, opts);
}

/** Computes a QR factorization of an m-by-n matrix A.
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_1 H_2 ... H_k,
 * \]
 * where k = min(m,n). Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i-1] = 0; v[i] = 1,
 * \]
 * with v[i+1] through v[m-1] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * @return  0 if success
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the elements on and above the diagonal of the array
 *      contain the k-by-n upper trapezoidal matrix R
 *      (R is upper triangular if m >= n); the elements below the diagonal,
 *      with the array tau, represent the unitary matrix Q as a
 *      product of elementary reflectors.
 *
 * @param[out] tau Real vector of length k.
 *      The scalar factors of the elementary reflectors.
 *
 * @param[in] opts Options.
 *
 * @ingroup variant_interface
 */
template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR vector_t>
int householder_qr(matrix_t& A,
                   vector_t& tau,
                   const HouseholderQROpts& opts = {})
{
    // Call variant
    if (opts.variant == HouseholderQRVariant::Level2)
        return geqr2(A, tau);
    else
        return geqrf(A, tau, opts);
}

}  // namespace tlapack

#endif  // TLAPACK_HOUSEHOLDER_QR_HH