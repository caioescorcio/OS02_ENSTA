#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {
// Taille du bloc à faire varier pour trouver l'optimum (Question 5) [cite: 43, 46]
const int szBlock = 512; // Essayez 32, 64, 128, 256, 512, 1024...

// Calcul sur un bloc : on utilise l'ordre j, k, i (le plus rapide) 
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
    // Bornes pour ne pas dépasser la taille de la matrice
    int iEnd = std::min(A.nbRows, iRowBlkA + szBlock);
    int jEnd = std::min(B.nbCols, iColBlkB + szBlock);
    int kEnd = std::min(A.nbCols, iColBlkA + szBlock);

    // Ordre j, k, i pour optimiser le cache
    for (int j = iColBlkB; j < jEnd; ++j) {
        for (int k = iColBlkA; k < kEnd; ++k) {
            for (int i = iRowBlkA; i < iEnd; ++i) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}
}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  
  // Boucles externes itérant sur les blocs [cite: 44, 45]
  for (int ib = 0; ib < A.nbRows; ib += szBlock) {
      for (int jb = 0; jb < B.nbCols; jb += szBlock) {
          for (int kb = 0; kb < A.nbCols; kb += szBlock) {
              // On appelle la fonction de calcul sur le sous-bloc courant
              prodSubBlocks(ib, jb, kb, szBlock, A, B, C);
          }
      }
  }
  return C;
}