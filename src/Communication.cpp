
#include <mpi.h>
#include "Communication.hpp"


void Communication::init_parallel(int argc, char **argv){

    MPI_Init(&argc, &argv);

}

void Communication::finalize() {

    MPI_Finalize();
}
