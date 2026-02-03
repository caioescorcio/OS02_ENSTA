#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nbp;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);

    int token = 0;
    
    // calculate dimension d (d = log2(nbp))
    int d = 0;
    while ((1 << d) < nbp) {
        d++;
    }

    // rank 0 initializes the token
    if (rank == 0) {
        token = 42; // Arbitrary value
        printf("Start: Rank 0 has token %d\n", token);
    }

    // synchronization for timing (Optional)
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // loop through each dimension of the hypercube
    for (int i = 0; i < d; i++) {
        int power_of_two = 1 << i; // 2^i (1, 2, 4, 8...)

        // if my rank is less than 2^i, it means I ALREADY have the token
        // (because I participated in the previous step). I become the SENDER.
        if (rank < power_of_two) {
            int dest = rank + power_of_two;
            // check that the destination exists (safety check)
            if (dest < nbp) {
                MPI_Send(&token, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                printf("Rank %d sends to rank %d (Step %d)\n", rank, dest, i);
            }
        } 
        // otherwise, if I am in the next range (between 2^i and 2^(i+1)),
        // i am the designated RECEIVER for this step.
        else if (rank < (power_of_two * 2)) {
            int source = rank - power_of_two;
            MPI_Recv(&token, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // from now on, this rank possesses the token and can send it in step i+1
        }
        // other processes (rank >= 2^(i+1)) wait for subsequent steps.
    }

    double end = MPI_Wtime();

    // verification
    printf("Rank %d: I received token %d\n", rank, token);

    if (rank == 0) {
        printf("Broadcast completed in %f seconds on dimension %d.\n", end - start, d);
    }

    MPI_Finalize();
    return 0;
}