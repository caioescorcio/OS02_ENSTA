# Calcul de l'ensemble de Mandelbrot en python
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex,  smooth=False) -> int | float:
        z:    complex
        iter: int

        # On vérifie dans un premier temps si le complexe
        # n'appartient pas à une zone de convergence connue :
        #   1. Appartenance aux disques  C0{(0,0),1/4} et C1{(-1,0),1/4}
        if c.real*c.real+c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1)+c.imag*c.imag < 0.0625:
            return self.max_iterations
        #  2.  Appartenance à la cardioïde {(1/4,0),1/2(1-cos(theta))}
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real-0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5*(1-ct.real/max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Sinon on itère
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations


################################################################

width, height = 1024, 1024

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

rows_total = height
rows_per_rank = [rows_total // nbp + (1 if i < rows_total % nbp else 0) for i in range(nbp)]
y_starts = [sum(rows_per_rank[:i]) for i in range(nbp)]

local_height = rows_per_rank[rank]
y_start = y_starts[rank]
y_end = y_start + local_height

local_conv = np.empty((local_height, width), dtype=np.double)

# On peut changer les paramètres des deux prochaines lignes
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)

scaleX = 3./width
scaleY = 2.25/height
convergence = np.empty((width, height), dtype=np.double)
# Calcul de l'ensemble de mandelbrot :

local_time = 0.0
deb = time()

time_total = 0.0

for i in range(nbp):
    if rank == i:
        for iy, y in enumerate(range(y_start, y_end)):
            for x in range(width):
                c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
                local_conv[iy, x] = mandelbrot_set.convergence(c, smooth=True)
                if rank == 0:
                    convergence = np.empty((height, width), dtype=np.double)
                else:
                    convergence = None
                  
sendbuf = local_conv.flatten()

if rank == 0:
    counts = [rows_per_rank[i]*width for i in range(nbp)]
    displs = [sum(counts[:i]) for i in range(nbp)]
    recvbuf = np.empty(height*width, dtype=np.double)
else:
    recvbuf = None
    counts = None
    displs = None

                  
fin = time()
local_time = fin-deb
comm.Gatherv(sendbuf,[recvbuf, counts, displs, MPI.DOUBLE], root=0)
 
time_total = comm.reduce(local_time, op=MPI.SUM, root=0)


if rank == 0:
    convergence = recvbuf.reshape((height, width))
    print(f"Temps du calcul MOYEN de l'ensemble de Mandelbrot : {time_total/nbp}")
if rank == 0:
    image = Image.fromarray(
        np.uint8(matplotlib.cm.plasma(convergence) * 255)
    )
    image.show()

    # Constitution de l'image résultante :
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
    fin = time()
    print(f"Temps de constitution de l'image : {fin-deb}")
    image.show()
