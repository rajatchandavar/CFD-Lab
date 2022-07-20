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

    public:

    /**
     * @brief Function to assign memory to variables on the GPU
     *
     * @param[in] field class object containing the field vectors
     * @param[in] grid class object containing details about grid
     * @param[in] Inlet velocity in x direction
     * @param[in] Inlet velocity in y direction
     * @param[in] Adiabatic wall temperature
     * @param[in] Hot wall temperature
     * @param[in] omega for Red-Black scheme
     */
    void initialize(Fields &, Grid &, dtype, dtype, dtype, dtype, dtype, dtype);

    /**
     * @brief Copies data from CPU to GPU
     *
     * @param[in] field class object containing the field vectors
     * @param[in] grid class object containing details about grid
     * @param[in] time step
     * 
     */
    void pre_process(Fields &, Grid &, Discretization &, dtype);

    /**
     * @brief Copies data back from GPU to CPU (needed when outputting the vtk file)
     *
     * @param[in] field class object containing the field vectors
     *
     */
    void post_process(Fields &);

    /**
     * @brief Calculate the temperature
     */
    void calc_T();

    /**
     * @brief Apply the boundary conditions
     */
    void apply_boundary();

    /**
     * @brief Calculating Fluxes
     */
    void calc_fluxes();

    /**
     * @brief Calculating RHS of pressure equaion
     */
    void calc_rs();

    /**
     * @brief Solves the pressure Poisson Equation using Red-Black scheme
     *
     * @param[in] maximum number of iterations
     * @param[in] Tolerance for convergence
     * @param[in] Current time value
     * @param[in] Current time step
     *
     */
    void calc_pressure(int, dtype, dtype, dtype);

    /**
     * @brief Calculates the velocity
     */
    void calc_velocities();

    /**
     * @brief Calculates the dt
     *
     */
    dtype calc_dt();

    /**
     * @brief Calculates the number of 1D blocks needed for the GPU kernel execution
     * 
     * @param[in] dimension of the array
     *
     */
    dim3 get_num_blocks(int);

    /**
     * @brief Calculates the number of 2D blocks needed for the GPU kernel execution
     * 
     * @param[in] x-dimension of the array
     * @param[in] y-dimension of the array
     *
     */
    dim3 get_num_blocks_2d(int, int);
    
    /**
     * @brief Releases the memory assigned on the GPU
     *
     */
    ~CUDA_solver();

    private:
    
    /// Field variables
    dtype *gpu_T, *gpu_T_temp;
    dtype *gpu_U;
    dtype *gpu_V;
    dtype *gpu_P;
    dtype *gpu_F;
    dtype *gpu_G;
    dtype *gpu_RS;

    /// Geometry related data
    int *gpu_geometry_data;
    int *gpu_fluid_id;
    int *gpu_moving_wall_id; 
    int *gpu_fixed_wall_id;
    int *gpu_inflow_id;
    int *gpu_outflow_id;
    int *gpu_adiabatic_id;
    int *gpu_hot_id;
    int *gpu_cold_id; 
    int *gpu_fluid_cells_size;
    int domain_size, grid_size, grid_size_x, grid_size_y;
    int *gpu_size_x, *gpu_size_y;
    int grid_fluid_cells_size;
    
    /// Boundary condition variables
    dtype *gpu_POUT;
    dtype *gpu_UIN;
    dtype *gpu_VIN;
    dtype *gpu_wall_temp_a, *gpu_wall_temp_h, *gpu_wall_temp_c;
    dtype UIN, VIN, wall_temp_a, wall_temp_h, wall_temp_c;
    dtype *gpu_wall_velocity;
    bool *gpu_isHeatTransfer;
    
    /// Pressure SOR and calculate dt related variables
    dtype *gpu_omega, *gpu_coeff, *gpu_rloc, *gpu_val, *gpu_res;
    dtype omg;
    dtype cpu_umax, cpu_vmax, cpu_dx, cpu_dy, cpu_nu, cpu_alpha, cpu_tau;
    dtype *gpu_umax, *gpu_vmax;
    dtype *gpu_dx, *gpu_dy, *gpu_dt, *gpu_gamma, *gpu_alpha, *gpu_beta, *gpu_nu, *gpu_tau;
    dtype *gpu_gx, *gpu_gy;
    int *d_mutex;

    /// Kernel Block size related variables
    dim3 block_size, num_blocks, block_size_2d, num_blocks_2d;

};
