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

########################################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

width, height = 1024, 1024
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)

scaleX = 3.0 / width
scaleY = 2.25 / height

if rank == 0:
    convergence = np.empty((height, width), dtype=np.double)
    lines = list(range(height))     # lists every line of the mandelbrot
    num_slaves = nbp - 1    # 1 maitre : rang 0
    next_line = 0

    deb = time()
    # send initial tasks
    for i in range(1, nbp): # the initial SLAVE tasks
        if next_line < height:
            comm.send(next_line, dest=i, tag=1) # sends lines for every other rank to do the initial calculation. the 'tag=1' means it is a line to compute 
            next_line += 1

    # receive results and assign new lines
    while next_line < height:
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status) # receives the line with 'tag=2' that means already computed
        src = status.Get_source()
        y, row_data = result
        convergence[y, :] = row_data

        # send next line
        comm.send(next_line, dest=src, tag=1)
        next_line += 1  # send's the new line once the processus is computed in one of the sources

    # collect remaining results
    for i in range(1, nbp):
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status) # receives the last lines
        y, row_data = result
        convergence[y, :] = row_data

    # tell slaves to stop
    for i in range(1, nbp):
        comm.send(-1, dest=i, tag=1)    # stop signal

    # image
    fin = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
    print("time: ", fin-deb)

# for the slaves
else:
    while True:
        y = comm.recv(source=0, tag=1)
        if y == -1:
            break
        row_data = np.array([mandelbrot_set.convergence(complex(-2 + scaleX*x, -1.125 + scaleY*y)) 
                             for x in range(width)])
        comm.send((y, row_data), dest=0, tag=2)
