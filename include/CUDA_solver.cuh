#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Fields.hpp"
#include "Grid.hpp"
#include "Enums.hpp"
#include "Discretization.hpp"
#include "Boundary.hpp"

#define BLOCK_SIZE 128
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

class CUDA_solver{

    private:
    
    double *T, *T_temp;
    double *U;
    double *V;
    double *P;

    int *geometry_data;

    int *fluid_id;
    int *moving_wall_id; 
    int *fixed_wall_id;
    int *inflow_id;
    int *outflow_id;
    int *adiabatic_id;
    int *hot_id;
    int *cold_id; 

    double *POUT;
    double *UIN;
    double *VIN;

    double wall_temp_a, wall_temp_h, wall_temp_c;

    int domain_size, grid_size, grid_size_x, grid_size_y;
    double *dx, *dy, *dt, *gamma, *alpha;
    int *size_x, *size_y;
    double wall_velocity;

    bool isHeatTransfer;

    dim3 block_size, num_blocks, block_size_2d, num_blocks_2d;

    public:

    CUDA_solver(Fields &, Grid &);

    void pre_process(Fields &, Grid &, Discretization &);
    void post_process(Fields &);
    void calc_T();
    void apply_boundary();
    void calc_fluxes();
    void calc_rs();
    dim3 get_num_blocks(int);
    dim3 get_num_blocks_2d(int, int);
    
    ~CUDA_solver();

};