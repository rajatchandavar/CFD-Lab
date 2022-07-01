#include "CUDA_solver.cuh"

#ifndef __CUDACC__
#define __CADACC__
#endif
#define at(var, i, j) var[(j) * (*gpu_size_x) + (i)]

#define check_bound_north(j) ((j + 1) < *gpu_size_y)
#define check_bound_south(j) ((j - 1) >= 0)
#define check_bound_east(i) ((i + 1) < *gpu_size_x)
#define check_bound_west(i) ((i - 1) >=0)

__device__ dtype interpolate(dtype *A, int i, int j, int i_offset, int j_offset, int *gpu_size_x) {
    dtype result = (at(A, i, j) + at(A, i + i_offset, j + j_offset)) / 2;
    return result;
}

__device__ dtype diffusion(dtype *A, int i, int j, dtype gpu_dx, dtype gpu_dy, int *gpu_size_x) {
    dtype result = (at(A, i + 1, j) - 2.0 * at(A, i, j) + at(A, i - 1, j)) / (gpu_dx * gpu_dx) +
                    (at(A, i, j + 1) - 2.0 * at(A, i, j) + at(A, i, j - 1)) / (gpu_dy * gpu_dy);

    return result;
}

//NO NEED TO PASS ALL OF gpu_U, V only surrounding of i,j sufficient
__device__ dtype convection_u(dtype *gpu_U, dtype *gpu_V, dtype gpu_gamma, int i, int j, dtype gpu_dx, dtype gpu_dy, int *gpu_size_x) {

    dtype t1 = interpolate(gpu_U, i, j, 1, 0, gpu_size_x);
    dtype t2 = interpolate(gpu_U, i, j, -1, 0, gpu_size_x);
    dtype du2_dx = 1 / gpu_dx * ((t1 * t1) - (t2 * t2)) + gpu_gamma / gpu_dx *\
            ((fabsf(interpolate(gpu_U, i, j, 1, 0, gpu_size_x)) * (at(gpu_U, i, j) - at(gpu_U, i + 1, j)) / 2) -\
             fabsf(interpolate(gpu_U, i, j, -1, 0, gpu_size_x)) * (at(gpu_U, i - 1, j) - at(gpu_U, i, j)) / 2);
    dtype duv_dy = 1 / gpu_dy * (((interpolate(gpu_V, i, j, 1, 0, gpu_size_x)) * (interpolate(gpu_U, i, j, 0, 1, gpu_size_x))) -\
                         ((interpolate(gpu_V, i, j - 1, 1, 0, gpu_size_x)) * (interpolate(gpu_U, i, j, 0, -1, gpu_size_x)))) +\
                    gpu_gamma / gpu_dy *\
                        ((fabsf(interpolate(gpu_V, i, j, 1, 0, gpu_size_x)) * (at(gpu_U, i, j) - at(gpu_U, i, j + 1)) / 2) -\
                         (fabsf(interpolate(gpu_V, i, j - 1, 1, 0, gpu_size_x)) * (at(gpu_U, i, j - 1) - at(gpu_U, i, j)) / 2));

    dtype result = du2_dx + duv_dy;
    return result;
}

__device__ dtype convection_v(dtype *gpu_U, dtype *gpu_V, dtype gpu_gamma, int i, int j, dtype gpu_dx, dtype gpu_dy, int *gpu_size_x) {

    dtype t1 = interpolate(gpu_V, i, j, 0, 1, gpu_size_x);
    dtype t2 = interpolate(gpu_V, i, j - 1, 0, 1, gpu_size_x);
    dtype dv2_dy = 1 / gpu_dy * ((t1 * t1) - (t2 * t2)) + gpu_gamma / gpu_dy *\
            ((fabsf(interpolate(gpu_V, i, j, 0, 1, gpu_size_x)) * (at(gpu_V, i, j) - at(gpu_V, i, j + 1)) / 2) -\
             fabsf(interpolate(gpu_V, i, j - 1, 0, 1, gpu_size_x)) * (at(gpu_V, i, j - 1) - at(gpu_V, i, j)) / 2);
    dtype duv_dx = 1 / gpu_dx *\
                        (((interpolate(gpu_U, i, j, 0, 1, gpu_size_x)) * (interpolate(gpu_V, i, j, 1, 0, gpu_size_x))) -\
                         ((interpolate(gpu_U, i - 1, j, 0, 1, gpu_size_x)) * (interpolate(gpu_V, i - 1, j, 1, 0, gpu_size_x)))) +\
                    gpu_gamma / gpu_dx *\
                        ((fabsf(interpolate(gpu_U, i, j, 0, 1, gpu_size_x)) * (at(gpu_V, i, j) - at(gpu_V, i + 1, j)) / 2) -\
                         (fabsf(interpolate(gpu_U, i - 1, j, 0, 1, gpu_size_x)) * (at(gpu_V, i - 1, j) - at(gpu_V, i, j)) / 2));

    dtype result = dv2_dy + duv_dx;
    return result;
}

__device__ dtype convection_Tu(dtype *gpu_T, dtype *gpu_U, int i, int j, dtype gpu_dx, dtype gpu_dy, dtype gpu_gamma, int *gpu_size_x) {
    dtype result;
    result = 1 / gpu_dx * (at(gpu_U, i, j) * interpolate(gpu_T, i, j, 1, 0, gpu_size_x) - at(gpu_U, i - 1, j) * interpolate(gpu_T, i - 1, j, 1, 0, gpu_size_x)) +
             gpu_gamma / gpu_dx * (fabsf(at(gpu_U, i, j)) * (at(gpu_T, i, j) - at(gpu_T, i + 1, j)) / 2 - fabsf(at(gpu_U, i - 1, j)) * (at(gpu_T, i - 1, j) - at(gpu_T, i, j)) / 2);
    return result;
}

__device__ dtype convection_Tv(dtype *gpu_T, dtype *gpu_V, int i, int j, dtype gpu_dx, dtype gpu_dy, dtype gpu_gamma, int *gpu_size_x) {
    dtype result;
    result = 1 / gpu_dy * (at(gpu_V, i, j) * interpolate(gpu_T, i, j, 0, 1, gpu_size_x) - at(gpu_V, i, j - 1) * interpolate(gpu_T, i, j - 1, 0, 1, gpu_size_x)) +
             gpu_gamma / gpu_dy * (fabsf(at(gpu_V, i, j)) * (at(gpu_T, i, j) - at(gpu_T, i, j + 1)) / 2 - fabsf(at(gpu_V, i, j - 1)) * (at(gpu_T, i, j - 1) - at(gpu_T, i, j)) / 2);
    return result;
}

dim3 CUDA_solver::get_num_blocks(int size) { return (size + BLOCK_SIZE - 1) / BLOCK_SIZE; }

dim3 CUDA_solver::get_num_blocks_2d(int gpu_size_x, int gpu_size_y) {
    return (dim3((gpu_size_x + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (gpu_size_y + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y));
}

__global__ void FixedWallBoundary(dtype *gpu_U, dtype *gpu_V, dtype *gpu_P, dtype *gpu_T, int *gpu_geometry_data,
int *gpu_fluid_id, int *gpu_moving_wall_id, int *gpu_fixed_wall_id, int *gpu_inflow_id, int *gpu_outflow_id, int *gpu_adiabatic_id, int *gpu_hot_id,
int *gpu_cold_id, dtype *gpu_wall_temp_a, dtype *gpu_wall_temp_h, dtype *gpu_wall_temp_c, bool *gpu_isHeatTransfer, int *gpu_size_x, int
*gpu_size_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *gpu_size_x && j < *gpu_size_y && (at(gpu_geometry_data, i, j) == 3 || at(gpu_geometry_data, i, j) == 5 || at(gpu_geometry_data, i, j) == 6 || at(gpu_geometry_data, i, j) == 7)) {
        // obstacles B_NE (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North and East directions 

        if(check_bound_north(j) && check_bound_east(i) && at(gpu_geometry_data,i,j+1)==0 && at(gpu_geometry_data,i+1,j)==0) {
            at(gpu_U, i, j) = 0.0;
            at(gpu_U, i - 1, j) = -at(gpu_U, i - 1, j + 1);
            at(gpu_V, i, j) = 0.0;
            at(gpu_V, i, j - 1) = -at(gpu_V, i + 1, j - 1);
            at(gpu_P, i, j) = (at(gpu_P, i, j + 1) + at(gpu_P, i + 1, j))/2;

            if(*gpu_isHeatTransfer==1) {
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T, i, j) = (at(gpu_T, i + 1, j) + at(gpu_T, i, j + 1))/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - (at(gpu_T,i, j + 1) + at(gpu_T,i + 1, j) )/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - (at(gpu_T,i, j + 1) + at(gpu_T,i + 1, j) )/2;
            }
        }

        // obstacles B_SE (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South and East directions 

        else if(check_bound_south(j) && check_bound_east(i) && at(gpu_geometry_data,i,j-1)==0 && at(gpu_geometry_data,i+1,j)==0) {

            at(gpu_U, i, j) = 0.0;
            at(gpu_U, i - 1, j) = -at(gpu_U, i - 1, j - 1);
            at(gpu_V, i, j - 1) = 0.0;
            at(gpu_V, i, j) = -at(gpu_V, i + 1, j);
            at(gpu_P,i, j) = (at(gpu_P,i + 1, j) + at(gpu_P,i, j - 1))/2;

            if(*gpu_isHeatTransfer==1) {
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = (at(gpu_T, i + 1, j) + at(gpu_T, i, j - 1))/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T, i, j) = 2*(*gpu_wall_temp_h) - (at(gpu_T,i, j - 1) + at(gpu_T,i + 1, j) )/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - (at(gpu_T,i, j - 1) + at(gpu_T,i + 1, j) )/2;
            }

        }

        // obstacle B_NW (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North and West directions 
            
        else if(check_bound_north(j) && check_bound_west(i) && at(gpu_geometry_data,i,j+1)==0 && at(gpu_geometry_data,i-1,j)==0) {

            at(gpu_U,i - 1, j) = 0.0;
            at(gpu_U,i, j) = -at(gpu_U,i, j + 1);
            at(gpu_V,i, j) = 0.0;
            at(gpu_V,i, j - 1) = -at(gpu_V,i - 1, j - 1);
            at(gpu_P,i,j) = (at(gpu_P,i - 1, j) + at(gpu_P,i, j + 1))/2;

            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = (at(gpu_T,i - 1, j) + at(gpu_T,i, j + 1))/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - (at(gpu_T,i, j + 1) + at(gpu_T,i - 1, j) )/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - (at(gpu_T,i, j + 1) + at(gpu_T,i - 1, j) )/2;
            }

        }

        // obstacle B_SW (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South and West directions 

        else if(check_bound_south(j) && check_bound_west(i) && at(gpu_geometry_data,i,j-1)==0 && at(gpu_geometry_data,i-1,j)==0){
            at(gpu_U,i - 1, j) = 0.0;
            at(gpu_U,i, j) = at(gpu_U,i, j - 1);
            at(gpu_V,i, j - 1) = 0.0;
            at(gpu_V,i, j) = -at(gpu_V,i - 1, j);
            at(gpu_P,i, j) = (at(gpu_P,i - 1, j) + at(gpu_P,i, j - 1))/2;
           
            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                   at(gpu_T,i, j) = (at(gpu_T,i - 1, j) + at(gpu_T,i, j - 1))/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - (at(gpu_T,i, j - 1) + at(gpu_T,i - 1, j) )/2;
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - (at(gpu_T,i, j - 1) + at(gpu_T,i - 1, j) )/2;
            }
            
        }

        // Bottom Wall B_N (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North direction

        else if(check_bound_north(j) && at(gpu_geometry_data,i,j+1)==0){
            at(gpu_U,i, j) = -at(gpu_U,i, j + 1);
            at(gpu_V,i, j) = 0.0;
            at(gpu_P,i, j) = at(gpu_P,i, j + 1);

            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = at(gpu_T,i, j + 1);
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - at(gpu_T,i, j + 1);
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - at(gpu_T,i, j + 1);
            }
        }

        // Top Wall B_S (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South direction

        else if(check_bound_south(j) && at(gpu_geometry_data,i,j-1)==0){

            at(gpu_U,i, j) = -at(gpu_U,i, j - 1);
            at(gpu_V,i, j) = 0.0;
            at(gpu_P,i, j) = at(gpu_P,i, j - 1);

            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = at(gpu_T,i, j - 1);
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2 * (*gpu_wall_temp_h) - at(gpu_T,i, j - 1);
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2 * (*gpu_wall_temp_c) - at(gpu_T,i, j - 1);
            }
        }

        // Left Wall B_E (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the East direction

        else if(check_bound_east(i) && at(gpu_geometry_data,i+1,j)==0){
            at(gpu_U,i, j) = 0.0;
            at(gpu_V,i, j) = -at(gpu_V,i + 1, j);
            at(gpu_P,i, j) = at(gpu_P,i + 1, j);

            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = at(gpu_T,i + 1, j);
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id) 
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - at(gpu_T,i + 1, j);        
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - at(gpu_T,i + 1, j);
            }
        }

        
        /***********************************************************************************************
        * Right Wall B_W (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on the West direction *
        ***********************************************************************************************/

        else if(check_bound_west(i) && at(gpu_geometry_data,i-1,j)==0){
            //Since u grid is staggered, the u velocity of cells to left of ghost layer should be set to 0.
            at(gpu_U,i - 1, j) = 0.0; 
            at(gpu_V,i, j) = -at(gpu_V,i - 1, j);
            at(gpu_P,i, j) = at(gpu_P,i - 1, j);

            if(*gpu_isHeatTransfer==1){
                if(at(gpu_geometry_data,i,j) == *gpu_adiabatic_id)
                    at(gpu_T,i, j) = at(gpu_T,i - 1, j);
                else if (at(gpu_geometry_data,i,j) == *gpu_hot_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_h) - at(gpu_T,i - 1, j);
                else if (at(gpu_geometry_data,i,j) == *gpu_cold_id)
                    at(gpu_T,i, j) = 2*(*gpu_wall_temp_c) - at(gpu_T,i - 1, j);
            }
        }
    }
}

__global__ void MovingWallBoundary(dtype *gpu_U, dtype *gpu_V, dtype *gpu_P, dtype *gpu_wall_velocity, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == 8 && check_bound_south(j)) {       
        at(gpu_U,i, j) = 2*(*gpu_wall_velocity)- at(gpu_U,i, j-1);
        //Since v grid is staggered, the v velocity of cells to below of ghost layer should be set to 0.
        at(gpu_V,i,j - 1) = 0.0;
        at(gpu_P,i,j) = at(gpu_P,i, j-1);
    }
}

__global__ void InFlowBoundary(dtype *gpu_U, dtype *gpu_V, dtype *gpu_P, dtype *gpu_UIN, dtype *gpu_VIN, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == 1 && check_bound_east(i)) {
        at(gpu_U,i,j) = (*gpu_UIN);
        at(gpu_V,i,j) = 2*(*gpu_VIN) - at(gpu_V,i + 1, j);
        at(gpu_P,i,j) = at(gpu_P,i + 1, j);
    }
}

__global__ void OutFlowBoundary(dtype *gpu_U, dtype *gpu_V, dtype *gpu_P, dtype *gpu_POUT, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == 2 && check_bound_west(i)) {
            at(gpu_U,i,j) = at(gpu_U,i - 1,j);
            at(gpu_V,i,j) = at(gpu_V,i - 1,j);
            at(gpu_P,i,j) = 2*(*gpu_POUT) - at(gpu_P,i - 1, j);
    }
}

__global__ void calc_T_kernel(dtype *gpu_T, dtype *gpu_T_temp, dtype *gpu_U, dtype *gpu_V, dtype *gpu_dx, dtype *gpu_dy, dtype *gpu_dt,
                              dtype *gpu_alpha, dtype *gpu_gamma, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    //NEED TO DO THIS ONLY FOR FLUID CELLS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == 0)
        at(gpu_T, i, j) = (*gpu_dt) * (*gpu_alpha * diffusion(gpu_T_temp, i, j, *gpu_dx, *gpu_dy, gpu_size_x) - convection_Tu(gpu_T_temp, gpu_U, i, j, *gpu_dx, *gpu_dy, *gpu_gamma, gpu_size_x) - convection_Tv(gpu_T_temp, gpu_V, i, j, *gpu_dx, *gpu_dy, *gpu_gamma, gpu_size_x)) + at(gpu_T_temp, i, j);
}

__global__ void calc_fluxes_kernel(dtype *gpu_F, dtype *gpu_G, dtype *gpu_U, dtype *gpu_V, dtype *gpu_T, int *gpu_geometry_data, dtype *gpu_gx, dtype *gpu_gy, dtype *gpu_dx, dtype *gpu_dy, int *gpu_size_x, int *gpu_size_y, dtype *gpu_gamma, dtype *gpu_beta,
                             dtype *gpu_nu, dtype *gpu_dt, bool *gpu_isHeatTransfer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data,i,j) == 0){

        at(gpu_F, i, j) = at(gpu_U, i, j) + (*gpu_dt) * ((*gpu_nu) * diffusion(gpu_U, i, j, *gpu_dx, *gpu_dy, gpu_size_x) - convection_u(gpu_U, gpu_V, *gpu_gamma, i, j, *gpu_dx, *gpu_dy, gpu_size_x) + (*gpu_gx));
        at(gpu_G, i, j) = at(gpu_V, i, j) + (*gpu_dt) * ((*gpu_nu) * diffusion(gpu_V, i, j, *gpu_dx, *gpu_dy, gpu_size_x) - convection_v(gpu_U, gpu_V, *gpu_gamma, i, j, *gpu_dx, *gpu_dy, gpu_size_x) + (*gpu_gy));

        if (*gpu_isHeatTransfer) {

                at(gpu_F,i,j) = at(gpu_F,i,j) - (*gpu_beta) * (*gpu_dt) / 2.0 * (at(gpu_T,i,j) + at(gpu_T,i + 1, j)) * (*gpu_gx) - (*gpu_dt) * (*gpu_gx);
                at(gpu_G,i,j) = at(gpu_G,i,j) - (*gpu_beta) * (*gpu_dt) / 2.0 * (at(gpu_T,i,j) + at(gpu_T,i, j + 1)) * (*gpu_gy) - (*gpu_dt) * (*gpu_gy);
        }
    }
}

__global__ void fluxes_bc_kernel(dtype *gpu_F, dtype *gpu_G, dtype *gpu_U, dtype *gpu_V, int *gpu_geometry_data, int *gpu_size_x, int *gpu_size_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < *gpu_size_x && j < *gpu_size_y){

        if (at(gpu_geometry_data, i, j) == 3 || at(gpu_geometry_data, i, j) == 5 || at(gpu_geometry_data, i, j) == 6 || at(gpu_geometry_data, i, j) == 7) {
            // B_NE fixed wall corner cell with fluid cells on the North and East directions

            if (check_bound_north(j) && check_bound_east(i) && at(gpu_geometry_data, i, j + 1) == 0 && at(gpu_geometry_data, i + 1, j) == 0) {
                at(gpu_F, i, j) = at(gpu_U, i, j);
                at(gpu_G, i, j) = at(gpu_V, i, j);
            }

            // B_SE fixed wall corner cell with fluid cells on the South and East directions

            else if (check_bound_south(j) && check_bound_east(i) && at(gpu_geometry_data, i, j - 1) == 0 && at(gpu_geometry_data, i + 1, j) == 0) {
                at(gpu_F, i, j) = at(gpu_U, i, j);
                at(gpu_G, i, j - 1) = at(gpu_V, i, j - 1);
            }

            // B_NW fixed wall corner cell with fluid cells on the North and West directions

            else if (check_bound_north(j) && check_bound_west(i) && at(gpu_geometry_data, i, j + 1) == 0 && at(gpu_geometry_data, i - 1, j) == 0) {
                at(gpu_F, i - 1, j) = at(gpu_U, i - 1, j);
                at(gpu_G, i, j) = at(gpu_V, i, j);
            }

            // B_SW fixed wall corner cell with fluid cells on the South and West directions

            else if (check_bound_south(j) && check_bound_west(i) && at(gpu_geometry_data, i, j - 1) == 0 && at(gpu_geometry_data, i - 1, j) == 0) {
                at(gpu_F, i - 1, j) = at(gpu_U, i - 1, j);
                at(gpu_G, i, j - 1) = at(gpu_V, i, j - 1);
            } 
            else if (check_bound_north(j) && at(gpu_geometry_data, i, j + 1) == 0)
                at(gpu_G, i, j) = at(gpu_V, i, j);

            else if (check_bound_south(j) && at(gpu_geometry_data, i, j - 1) == 0)
                at(gpu_G,i,j - 1) = at(gpu_V,i,j - 1);

            else if (check_bound_west(i) && at(gpu_geometry_data, i - 1, j) == 0)
                at(gpu_F, i - 1, j) = at(gpu_U, i - 1, j);

            else if (check_bound_east(i) && at(gpu_geometry_data, i + 1, j) == 0)
                at(gpu_F, i, j) = at(gpu_U, i, j);

        }

        else if (at(gpu_geometry_data, i, j) == 8) {
            at(gpu_G, i, j - 1) = at(gpu_V, i, j - 1);
        } 
        
        else if (at(gpu_geometry_data, i, j) == 1) {
            at(gpu_F, i, j) = at(gpu_U, i, j);
        }

        else if (at(gpu_geometry_data, i, j) == 2) {

            at(gpu_F, i - 1, j) = at(gpu_U, i - 1, j);
        }
    }
}


__global__ void calc_rs_kernel(dtype *gpu_RS, dtype *gpu_F, dtype *gpu_G,dtype *gpu_dx, dtype *gpu_dy, dtype *gpu_dt, int *gpu_size_x, int *gpu_size_y, int *gpu_geometry_data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *gpu_size_x && j < *gpu_size_y && at(gpu_geometry_data, i, j) == 0)
        at(gpu_RS,i,j) = 1 / (*gpu_dt) * ((at(gpu_F,i, j) - at(gpu_F,i - 1, j)) / (*gpu_dx) + (at(gpu_G,i, j) - at(gpu_G,i, j - 1)) / (*gpu_dy));
}

void CUDA_solver::initialize(Fields &field, Grid &grid, dtype cpu_UIN, dtype cpu_VIN, dtype cpu_wall_temp_a, dtype cpu_wall_temp_h, dtype cpu_wall_temp_c) {

    UIN = cpu_UIN;
    VIN = cpu_VIN;
    wall_temp_a = cpu_wall_temp_a;
    wall_temp_h = cpu_wall_temp_h;
    wall_temp_c = cpu_wall_temp_c;

    block_size = dim3(BLOCK_SIZE);
    block_size_2d = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    domain_size = (grid.domain().domain_size_x + 2) * (grid.domain().domain_size_y + 2);
    cudaMalloc((void **)&gpu_geometry_data, domain_size * sizeof(int));

    cudaMalloc((void **)&geom_check, domain_size * sizeof(int));

    grid_size = grid.imaxb() * grid.jmaxb();
    grid_size_x = grid.imaxb();
    grid_size_y = grid.jmaxb();

    cudaMalloc((void **)&gpu_T, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_T_temp, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_U, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_V, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_P, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_F, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_G, grid_size * sizeof(dtype));
    cudaMalloc((void **)&gpu_RS, grid_size * sizeof(dtype));

    cudaMalloc((void **)&gpu_dx, sizeof(dtype));
    cudaMalloc((void **)&gpu_dy, sizeof(dtype));
    cudaMalloc((void **)&gpu_dt, sizeof(dtype));
    cudaMalloc((void **)&gpu_alpha, sizeof(dtype));
    cudaMalloc((void **)&gpu_gamma, sizeof(dtype));
    cudaMalloc((void **)&gpu_beta, sizeof(dtype));
    cudaMalloc((void **)&gpu_nu, sizeof(dtype));
    cudaMalloc((void **)&gpu_gx, sizeof(dtype));
    cudaMalloc((void **)&gpu_gy, sizeof(dtype));

    cudaMalloc((void **)&gpu_size_x, sizeof(dtype));
    cudaMalloc((void **)&gpu_size_y, sizeof(dtype));

    cudaMalloc((void **)&gpu_fluid_id, sizeof(int));
    cudaMalloc((void **)&gpu_moving_wall_id, sizeof(int));
    cudaMalloc((void **)&gpu_fixed_wall_id, sizeof(int));
    cudaMalloc((void **)&gpu_inflow_id, sizeof(int));
    cudaMalloc((void **)&gpu_outflow_id, sizeof(int));
    cudaMalloc((void **)&gpu_adiabatic_id, sizeof(int));
    cudaMalloc((void **)&gpu_hot_id, sizeof(int));
    cudaMalloc((void **)&gpu_cold_id, sizeof(int));

    cudaMalloc((void **)&gpu_wall_temp_a, sizeof(dtype));
    cudaMalloc((void **)&gpu_wall_temp_h, sizeof(dtype));
    cudaMalloc((void **)&gpu_wall_temp_c, sizeof(dtype));
    cudaMalloc((void **)&gpu_isHeatTransfer, sizeof(bool));

    cudaMalloc((void **)&gpu_wall_velocity, sizeof(dtype));
    cudaMalloc((void **)&gpu_UIN, sizeof(dtype));
    cudaMalloc((void **)&gpu_VIN, sizeof(dtype));
    cudaMalloc((void **)&gpu_POUT, sizeof(dtype));
}

void CUDA_solver::pre_process(Fields &field, Grid &grid, Discretization &discretization, dtype cpu_dt) {

    cudaMemcpy(gpu_geometry_data, grid.get_geometry_data().data(), domain_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(geom_check, gpu_geometry_data, domain_size * sizeof(int), cudaMemcpyDeviceToDevice);
    bool energycheck = field.isHeatTransfer();
    cudaMemcpy(gpu_isHeatTransfer, &energycheck, sizeof(bool), cudaMemcpyHostToDevice);

    if (energycheck)
        cudaMemcpy(gpu_T, field.t_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_U, field.u_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_V, field.v_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_P, field.p_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_F, field.f_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_G, field.g_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_RS, field.rs_matrix().data(), grid_size * sizeof(dtype), cudaMemcpyHostToDevice);

    dtype var = grid.dx();
    cudaMemcpy(gpu_dx, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = grid.dy();
    cudaMemcpy(gpu_dy, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = cpu_dt;
    cudaMemcpy(gpu_dt, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_alpha();
    cudaMemcpy(gpu_alpha, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_beta();
    cudaMemcpy(gpu_beta, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = discretization.get_gamma();
    cudaMemcpy(gpu_gamma, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_nu();
    cudaMemcpy(gpu_nu, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_gx();
    cudaMemcpy(gpu_gx, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    var = field.get_gy();
    cudaMemcpy(gpu_gy, &var, sizeof(dtype), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_size_x, &grid_size_x, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_size_y, &grid_size_y, sizeof(int), cudaMemcpyHostToDevice);

    int var1 = GEOMETRY_PGM::moving_wall_id;
    cudaMemcpy(gpu_moving_wall_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::fixed_wall_id;
    cudaMemcpy(gpu_fixed_wall_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::inflow_id;
    cudaMemcpy(gpu_inflow_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::outflow_id;
    cudaMemcpy(gpu_outflow_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::hot_id;
    cudaMemcpy(gpu_hot_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::cold_id;
    cudaMemcpy(gpu_cold_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);
    var1 = GEOMETRY_PGM::adiabatic_id;
    cudaMemcpy(gpu_adiabatic_id, &(var1), sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(gpu_wall_temp_a, &wall_temp_a, sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_wall_temp_h, &wall_temp_h, sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_wall_temp_c, &wall_temp_c, sizeof(dtype), cudaMemcpyHostToDevice);

    var = LidDrivenCavity::wall_velocity;
    cudaMemcpy(gpu_wall_velocity, &var, sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_UIN, &UIN, sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_VIN, &VIN, sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_POUT, &(GEOMETRY_PGM::POUT), sizeof(dtype), cudaMemcpyHostToDevice);
}

void CUDA_solver::apply_boundary() {

    num_blocks_2d = get_num_blocks_2d(grid_size_x, grid_size_y);

    FixedWallBoundary<<<num_blocks_2d, block_size_2d>>>(gpu_U, gpu_V, gpu_P, gpu_T, gpu_geometry_data, gpu_fluid_id, gpu_moving_wall_id, gpu_fixed_wall_id, gpu_inflow_id, gpu_outflow_id, gpu_adiabatic_id, gpu_hot_id, gpu_cold_id, gpu_wall_temp_a, gpu_wall_temp_h, gpu_wall_temp_c, gpu_isHeatTransfer, gpu_size_x, gpu_size_y);

    MovingWallBoundary<<<num_blocks_2d, block_size_2d>>>(gpu_U, gpu_V, gpu_P, gpu_wall_velocity, gpu_size_x, gpu_size_y, gpu_geometry_data);

    InFlowBoundary<<<num_blocks_2d, block_size_2d>>>(gpu_U, gpu_V, gpu_P, gpu_UIN, gpu_VIN, gpu_size_x, gpu_size_y, gpu_geometry_data);

    OutFlowBoundary<<<num_blocks_2d, block_size_2d>>>(gpu_U, gpu_V, gpu_P, gpu_POUT, gpu_size_x, gpu_size_y, gpu_geometry_data);

}


void CUDA_solver::calc_T() {
    cudaMemcpy(gpu_T_temp, gpu_T, grid_size * sizeof(dtype), cudaMemcpyDeviceToDevice);
    num_blocks_2d = get_num_blocks_2d(grid_size_x, grid_size_y);
    calc_T_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_T, gpu_T_temp, gpu_U, gpu_V, gpu_dx, gpu_dy, gpu_dt, gpu_alpha, gpu_gamma, gpu_size_x, gpu_size_y, gpu_geometry_data);
}

void CUDA_solver::calc_fluxes() {
    num_blocks_2d = get_num_blocks_2d(grid_size_x, grid_size_y);
    calc_fluxes_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_F,gpu_G,gpu_U,gpu_V,gpu_T,gpu_geometry_data,gpu_gx,gpu_gy,gpu_dx,gpu_dy,gpu_size_x, gpu_size_y, gpu_gamma, gpu_beta, gpu_nu, gpu_dt, gpu_isHeatTransfer);
    fluxes_bc_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_F,gpu_G,gpu_U,gpu_V,gpu_geometry_data,gpu_size_x, gpu_size_y);
}

void CUDA_solver::calc_rs() {
    num_blocks_2d = get_num_blocks_2d(grid_size_x, grid_size_y);
    calc_rs_kernel<<<num_blocks_2d, block_size_2d>>>(gpu_RS, gpu_F,gpu_G, gpu_dx,gpu_dy, gpu_dt, gpu_size_x, gpu_size_y, gpu_geometry_data);
}

void CUDA_solver::post_process(Fields &field) {
    
    if (field.isHeatTransfer())
        cudaMemcpy((void *)field.t_matrix().data(), gpu_T, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    
    cudaMemcpy((void *)field.f_matrix().data(), gpu_F, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)field.g_matrix().data(), gpu_G, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)field.rs_matrix().data(), gpu_RS, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)field.u_matrix().data(), gpu_U, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)field.v_matrix().data(), gpu_V, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)field.p_matrix().data(), gpu_P, grid_size * sizeof(dtype), cudaMemcpyDeviceToHost);

}

CUDA_solver::~CUDA_solver() {
    cudaFree(gpu_geometry_data);
    cudaFree(gpu_T);
    cudaFree(gpu_U);
    cudaFree(gpu_V);
    cudaFree(gpu_P);
    cudaFree(gpu_F);
    cudaFree(gpu_G);
    cudaFree(gpu_RS);
    cudaFree(gpu_T_temp);
    cudaFree(gpu_dx);
    cudaFree(gpu_dy);
    cudaFree(gpu_dt);
    cudaFree(gpu_gx);
    cudaFree(gpu_gy);
    cudaFree(gpu_nu);
    cudaFree(gpu_beta);
    cudaFree(gpu_gamma);
    cudaFree(gpu_alpha);
    cudaFree(gpu_size_x);
    cudaFree(gpu_size_y);
    cudaFree(gpu_fluid_id);
    cudaFree(gpu_fixed_wall_id);
    cudaFree(gpu_moving_wall_id);
    cudaFree(gpu_inflow_id);
    cudaFree(gpu_outflow_id);
    cudaFree(gpu_adiabatic_id);
    cudaFree(gpu_hot_id);
    cudaFree(gpu_cold_id);
    cudaFree(gpu_wall_temp_a);
    cudaFree(gpu_wall_temp_c);
    cudaFree(gpu_wall_temp_h);
    cudaFree(gpu_isHeatTransfer);
    cudaFree(gpu_UIN);
    cudaFree(gpu_VIN);
    cudaFree(gpu_POUT);
    cudaFree(gpu_wall_velocity);


    cudaFree(geom_check);
}