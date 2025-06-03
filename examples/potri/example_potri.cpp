/// @file example_geqr2.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/potrf.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>
#include <tlapack/blas/hemm.hpp>
#include <tlapack/lapack/lanhe.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) 
    {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t n)
{
    typedef tlapack::real_type<T> real_t;
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if n or n are large
    bool verbose = true;

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);
    std::vector<T> C_;
    auto C = new_matrix(C_, n, n);

    int k = 3;
    std::vector<T> B_;
    auto B = new_matrix(B_, n, k);
    std::vector<T> X_;
    auto X = new_matrix(X_, n, k);
    std::vector<T> Y_;
    auto Y = new_matrix(Y_, n, k);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, k);

    // Initialize arrays A with junk
    for (size_t j = 0; j < n; ++j) 
    {
        for (size_t i = 0; i < n; ++i) 
        {
            if constexpr (tlapack::is_complex<T>)
                A(i, j) = T(static_cast<float>(0xDEADBEEF), static_cast<float>(0xDEADBEEF)); 
            else
                A(i, j) = T(static_cast<float>(0xDEADBEEF));
        }
    }

    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = j; i < n; ++i)
            if constexpr (tlapack::is_complex<T>)
                A(i, j) = T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            else
                A(i, j) = T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    for (size_t j = 0; j < n; ++j)
    {
        if constexpr (tlapack::is_complex<T>)
            A(j, j) += T(static_cast<float>(n), static_cast<float>(0.0));    
        else
            A(j, j) += T(static_cast<float>(n));
    }

    // Frobenius norm of A
    auto normA = tlapack::lanhe(tlapack::FROB_NORM, tlapack::Uplo::Lower, A);

    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(A);
    }

    // Generate a random matrix in B
    for (size_t j = 0; j < k; ++j) 
    {
        for (size_t i = 0; i < n; ++i) 
        {
            if constexpr (tlapack::is_complex<T>)
                B(i, j) = T(static_cast<float>(rand()) / static_cast<float>(RAND_MAX), static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            else
                B(i, j) = T(static_cast<float>(0xDEADBEEF));
        }
    }

    // Print B
    if (verbose) {
        std::cout << std::endl << "B = ";
        printMatrix(B);
    }

    // Create a Copy A into C
    lacpy(tlapack::GENERAL, A, C);

    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        // Save the R matrix
        int info = tlapack::potrf(tlapack::LOWER_TRIANGLE, C);
        if (info < 0)
        {
            printf("Is not SPD\n");
        }
        else
        {
            printf("\nIs SPD");
        }
    }
    // Record end time
    auto endQR = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedQR =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endQR - startQR);

    // Print A
    if (verbose) {
        std::cout << std::endl << "C = ";
        printMatrix(C);
    }

    std::cout << std::endl;
    std::cout << "time = " << elapsedQR.count() * 1.0e-6 << " ms";

    // Test
    //C (Lower) * Y = B
    // Create a Copy B into Y
    lacpy(tlapack::GENERAL, B, Y);
    tlapack::trsm(tlapack::Side::Left, tlapack::Uplo::Lower, tlapack::Op::NoTrans, tlapack::Diag::NonUnit, T(1), C, Y); 

    // Print Y
    if (verbose) {
        std::cout << std::endl << "Y = ";
        printMatrix(Y);
    }

    //C (Trans) * X = Y
    // Create a Copy Y into X
    lacpy(tlapack::GENERAL, Y, X);
    tlapack::trsm(tlapack::Side::Left, tlapack::Uplo::Lower, tlapack::Op::ConjTrans, tlapack::Diag::NonUnit, T(1), C, X); 

    // Print X
    if (verbose) {
        std::cout << std::endl << "X = ";
        printMatrix(X);
    }

    // B - AX
    lacpy(tlapack::GENERAL, B, R);
    tlapack::hemm(tlapack::Side::Left, tlapack::Uplo::Lower, real_t(-1), A, X, real_t(1), R);

    // Print R
    if (verbose) {
        std::cout << std::endl << "R = ";
        printMatrix(R);
    }

    // Norm X
    real_t normX = tlapack::lange(tlapack::FROB_NORM, X);

    // Norm R
    real_t normR =tlapack::lange(tlapack::FROB_NORM, R);

    // Print Normalize
    if (verbose) {
        std::cout << std::endl << "Norm A = " << normA;

        std::cout << std::endl << "Norm X = " << normX;

        std::cout << std::endl << "Norm R = " << normR;

        std::cout << std::endl << "||b - AX|| / (||A|| * ||X||) = " << normR / (normA * normX);
    }
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int n;

    // Default arguments
    n = (argc < 2) ? 7 : atoi(argv[1]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d )", n);
    run<float>(n);
    printf("\n-----------------------\n");

    printf("run< double >( %d )", n);
    run<double>(n);
    printf("\n-----------------------\n");

    printf("run< long double >( %d )", n);
    run<long double>(n);
    printf("\n-----------------------\n");

    printf("run< complex<float> >( %d )", n);
    run<std::complex<long double> >(n);
    printf("\n-----------------------\n");

    return 0;
}
