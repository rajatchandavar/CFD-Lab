#include <mpi.h>
#include "Communication.hpp"
#include "Case.hpp"

void Communication::init_parallel(int argc, char **argv){
    MPI_Init(&argc, &argv);
}

double Communication::reduce_min(double rank) {
    double min_value{0};
    MPI_Reduce(&rank, &min_value, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    return min_value;
}

double Communication::reduce_max(double rank) {
    double max_value{0};
    MPI_Reduce(&rank, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_value;
}

double Communication::reduce_sum(double rank) {
    double sum{0};
    MPI_Reduce(&rank, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return sum;
}

void Communication::broadcast(double umax) {
    MPI_Bcast(&umax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void Communication::communicate(Matrix<double> &field){
    auto neighbour = _assign_neighbours(get_rank());

    int data_imax = field.imax()-2;
    int data_jmax = field.jmax()-2;

    MPI_Status status;
    double data_lr_out[data_jmax], data_lr_in[data_jmax], data_tb_out[data_imax], data_tb_in[data_imax];
    if (neighbour['L'] != MPI_PROC_NULL){ // can be removed
        
        for (auto k = 0; k < data_jmax; ++k){
            data_lr_out[k] = field(1, k + 1);
        }

        MPI_Sendrecv(&data_lr_out, data_jmax, MPI_DOUBLE, neighbour['L'], 0,
                     &data_lr_in, data_jmax, MPI_DOUBLE, neighbour['L'], 0, MPI_COMM_WORLD, &status);

        for (auto k = 0; k < data_jmax; ++k){
            field(0, k + 1) = data_lr_in[k];
        }
        
    }

    if (neighbour['R'] != MPI_PROC_NULL){ // can be removed
        
        for (auto k = 0; k < data_jmax; ++k){
            data_lr_out[k] = field(data_imax, k + 1);
        }

        MPI_Sendrecv(&data_lr_out, data_jmax, MPI_DOUBLE, neighbour['R'], 0,
                     &data_lr_in, data_jmax, MPI_DOUBLE, neighbour['R'], 0, MPI_COMM_WORLD, &status);

        for (auto k = 0; k < data_jmax; ++k){
            field(data_imax + 1, k + 1) = data_lr_in[k];
        }
        
    }

    if (neighbour['T'] != MPI_PROC_NULL){ // can be removed
        
        for (auto k = 0; k < data_imax; ++k){
            data_tb_out[k] = field(k + 1, data_jmax);
        }

        MPI_Sendrecv(&data_tb_out, data_imax, MPI_DOUBLE, neighbour['T'], 0,
                     &data_tb_in, data_imax, MPI_DOUBLE, neighbour['T'], 0, MPI_COMM_WORLD, &status);

        for (auto k = 0; k < data_imax; ++k){
            field(k + 1, data_jmax + 1) = data_tb_in[k];
        }
        
    }

    if (neighbour['B'] != MPI_PROC_NULL){ // can be removed
        
        for (auto k = 0; k < data_imax; ++k){
            data_tb_out[k] = field(k + 1, 1);
        }

        MPI_Sendrecv(&data_tb_out, data_imax, MPI_DOUBLE, neighbour['B'], 0,
                     &data_tb_in, data_imax, MPI_DOUBLE, neighbour['B'], 0, MPI_COMM_WORLD, &status);

        for (auto k = 0; k < data_imax; ++k){
            field(k + 1, 0) = data_tb_in[k];
        }
        
    }    
}

int Communication::get_rank(){
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
    return my_rank;
}

int Communication::get_size(){
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &size); 
    return size;
}

void Communication::finalize() {

    MPI_Finalize();

}

std::map<char, int> Communication::_assign_neighbours(int rank){
    std::map<char, int> neighbour;
    /**************************************************************************/
    int iproc = 3, jproc = 2;
    /*************************************************************************/
    int i = rank % iproc;
    int j = (rank - i) / iproc;

    // L - Left, R - Right, T - Top, B - Bottom
    if (i - 1 > 0)
        neighbour['L'] =  i - 1;
    else
        neighbour['L'] = MPI_PROC_NULL;

    if (i + 1 > iproc - 1)
        neighbour['R'] =  i + 1;
    else
        neighbour['R'] = MPI_PROC_NULL;

    if (j - 1 > 0)
        neighbour['T'] =  j - 1;
    else
        neighbour['T'] = MPI_PROC_NULL;

    if (j + 1 > jproc - 1)
        neighbour['B'] =  j + 1;
    else
        neighbour['B'] = MPI_PROC_NULL;

    return neighbour;
}
