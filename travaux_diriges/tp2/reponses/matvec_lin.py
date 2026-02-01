# Produit matrice-vecteur v = A.u
import numpy as np
from time import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

dim = 120
Nloc = dim // nbp

# ---------------- Séquentiel (référence) ----------------
if rank == 0:
    deb = time()
    A = np.array([[(i + j) % dim + 1. for i in range(dim)] for j in range(dim)])
    u = np.array([i + 1. for i in range(dim)])
    v = A.dot(u)
    fin = time()
    t0 = fin - deb
    print("Temps séquentiel:", t0)


A = np.array([[(i + j) % dim + 1. for i in range(dim)] for j in range(dim)])
u = np.array([i + 1. for i in range(dim)])

if rank == 0:
    deb = time()


partial_result = np.zeros(Nloc)
for i in range(Nloc):
    partial_result[i] = A[rank * Nloc + i].dot(u)
    
# Matrix A (4x4) and vector u
#
#        u0  u1  u2  u3
#      -----------------
# A0 -> | a00 a01 a02 a03 |   v0 = A0 · u   
# A1 -> | a10 a11 a12 a13 |   v1 = A1 · u
# A2 -> | a20 a21 a22 a23 |   v2 = A2 · u
# A3 -> | a30 a31 a32 a33 |   v3 = A3 · u
#
# ----------------------------------------------
# ROW (line) decomposition (Gather)
# Rank 0: A0, A1 > computes v0, v1
# Rank 1: A2, A3 > computes v2, v3

if rank == 0:
    final_result = np.zeros(dim)
else:
    final_result = None

comm.Gather(partial_result, final_result, root=0)

if rank == 0:
    fin = time()

if rank == 0:
    print("Temps parallèle:", fin - deb)
    print("Speedup = ", t0/(fin - deb))
