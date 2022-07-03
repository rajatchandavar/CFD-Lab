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
    
    dtype *gpu_T, *gpu_T_temp;
    dtype *gpu_U;
    dtype *gpu_V;
    dtype *gpu_P;
    dtype *gpu_F;
    dtype *gpu_G;
    dtype *gpu_RS;


    int *gpu_geometry_data;

    int *gpu_fluid_id;
    int *gpu_moving_wall_id; 
    int *gpu_fixed_wall_id;
    int *gpu_inflow_id;
    int *gpu_outflow_id;
    int *gpu_adiabatic_id;
    int *gpu_hot_id;
    int *gpu_cold_id; 

    dtype *gpu_POUT;
    dtype *gpu_UIN;
    dtype *gpu_VIN;

    dtype *gpu_umax, *gpu_vmax;
     char *c1, *c2;
    dtype *gpu_wall_temp_a, *gpu_wall_temp_h, *gpu_wall_temp_c;

    int domain_size, grid_size, grid_size_x, grid_size_y;
    dtype *gpu_dx, *gpu_dy, *gpu_dt, *gpu_gamma, *gpu_alpha, *gpu_beta, *gpu_nu, *gpu_tau;
    dtype *gpu_gx, *gpu_gy;

    int *gpu_size_x, *gpu_size_y;
    dtype *gpu_wall_velocity;

    bool *gpu_isHeatTransfer;

    dim3 block_size, num_blocks, block_size_2d, num_blocks_2d;

    int *geom_check;
    int *d_mutex;

    dtype UIN, VIN, wall_temp_a, wall_temp_h, wall_temp_c;
    public:

    void initialize(Fields &, Grid &, dtype, dtype, dtype, dtype, dtype);
    void pre_process(Fields &, Grid &, Discretization &, dtype);
    void post_process(Fields &);
    void calc_T();
    void apply_boundary();
    void calc_fluxes();
    void calc_rs();
    void calc_velocities();
    void calc_dt();
    dim3 get_num_blocks(int);
    dim3 get_num_blocks_2d(int, int);
    
    ~CUDA_solver();

};