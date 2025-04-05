/// @file swap.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_BLAS_SWAP_HH
#define TLAPACK_BLAS_SWAP_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/**
 * Swap vectors, $x <=> y$.
 *
 * @param[in,out] x A n-element vector.
 * @param[in,out] y A n-element vector.
 *
 * @ingroup blas1
 */
template <
    TLAPACK_VECTOR vectorX_t,
    TLAPACK_VECTOR vectorY_t,
    class T = type_t<vectorY_t>,
    disable_if_allow_optblas_t<pair<vectorX_t, T>, pair<vectorY_t, T> > = 0>
void swap(vectorX_t& x, vectorY_t& y)
{
    using idx_t = size_type<vectorY_t>;
    using TX = type_t<vectorX_t>;

    // constants
    const idx_t n = size(x);

    // check arguments
    tlapack_check_false(size(y) != n);

    for (idx_t i = 0; i < n; ++i) {
        const TX aux = x[i];
        x[i] = y[i];
        y[i] = aux;
    }
}

#ifdef TLAPACK_USE_LAPACKPP

template <
    TLAPACK_LEGACY_VECTOR vectorX_t,
    TLAPACK_LEGACY_VECTOR vectorY_t,
    class T = type_t<vectorY_t>,
    enable_if_allow_optblas_t<pair<vectorX_t, T>, pair<vectorY_t, T> > = 0>
void swap(vectorX_t& x, vectorY_t& y)
{
    // Legacy objects
    auto x_ = legacy_vector(x);
    auto y_ = legacy_vector(y);

    // Constants to forward
    const auto& n = x_.n;

    return ::blas::swap(n, x_.ptr, x_.inc, y_.ptr, y_.inc);
}

#endif

/**
 * Swap vectors, $x <=> y$.
 *
 * @see tlapack::swap(vectorX_t& x, vectorY_t& y)
 *
 * @note This overload avoids unexpected behavior as follows.
 *      Without it, the unspecialized call `swap( x, y )` using arrays with
 *      `std::complex` entries would call `std::swap`, while `swap( x, y )`
 *      using arrays with float or double entries would call `tlapack::swap`.
 *      Use @c tlapack::swap(x,y) instead of @c swap(x,y) .
 *
 * @ingroup blas1
 */
template <TLAPACK_VECTOR vector_t>
void swap(vector_t& x, vector_t& y)
{
    return swap<vector_t, vector_t>(x, y);
}

}  // namespace tlapack

#endif  //  #ifndef TLAPACK_BLAS_SWAP_HH
