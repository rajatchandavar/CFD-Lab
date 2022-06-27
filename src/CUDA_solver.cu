#include "CUDA_solver.cuh"

#define at(var, i, j) var[ (j) * (*size_x) + (i)]

__device__ double interpolate(double *A, int i, int j, int i_offset, int j_offset, int *size_x) {
    double result =( at(A, i, j) + at(A, i + i_offset, j + j_offset)) / 2;
    return result;
}

__device__ double diffusion(double *A, int i, int j, double dx, double dy, int *size_x) {
    double result = (at(A, i + 1, j) - 2.0 * at(A, i, j) + at(A, i - 1, j)) / (dx * dx) +
                    (at(A, i, j + 1) - 2.0 * at(A, i, j) + at(A, i, j - 1)) / (dy * dy);

    return result;
}

__device__ double convection_Tu(double *T, double *U, int i, int j, double dx, double dy, double gamma, int *size_x)
{
    double result;
    result = 1/dx * ( at(U, i, j) * interpolate(T,i,j,1,0, size_x) - at(U, i - 1,j) * interpolate(T,i-1,j,1,0, size_x)) + 
                        gamma/dx * ( fabsf(at(U, i, j)) * (at(T, i, j) - at(T, i + 1, j)) / 2 - fabsf(at(U, i - 1,j)) * (at(T, i - 1, j) - at(T, i, j)) / 2 );
    return result;
}

__device__ double convection_Tv(double *T, double *V, int i, int j, double dx, double dy, double gamma, int *size_x)
{
    double result;
    result = 1/dy * ( at(V, i, j) * interpolate(T,i,j,0,1, size_x) - at(V, i,j - 1) * interpolate(T,i,j - 1,0,1, size_x)) + 
                        gamma/dy * ( fabsf(at(V, i, j)) * (at(T, i, j) - at(T, i, j + 1)) / 2 - fabsf(at(V, i,j - 1)) * (at(T, i, j - 1) - at(T, i, j)) / 2 );
    return result;
}

dim3 CUDA_solver::get_num_blocks(int size) { return (size + BLOCK_SIZE - 1) / BLOCK_SIZE; }

dim3 CUDA_solver::get_num_blocks_2d(int size_x, int size_y){
    return (dim3((size_x + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (size_y + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y));
}

__global__ void calc_T_kernel(double * T, double *T_temp, double *U, double *V, double *dx, double *dy, double *dt, double *alpha, double *gamma, int *size_x, int *size_y){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < *size_x && j < *size_y)
        at(T,i,j) = (*dt) * (*alpha * diffusion(T_temp,i,j, *dx, *dy, size_x) - convection_Tu(T_temp,U,i,j,*dx, *dy, *gamma, size_x) - convection_Tv(T_temp,V,i,j, *dx, *dy, *gamma, size_x)) + at(T_temp, i,j);
    
}

CUDA_solver::CUDA_solver(Fields &field, Grid &grid){

    block_size = dim3(BLOCK_SIZE);
    block_size_2d = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    domain_size = (grid.domain().domain_size_x + 2) * (grid.domain().domain_size_y + 2);
    cudaMalloc(&geometry_file, domain_size * sizeof(int));

    grid_size = grid.imaxb() * grid.jmaxb();
    grid_size_x = grid.imaxb();
    grid_size_y = grid.jmaxb();
    cudaMalloc((void **)&T, grid_size * sizeof(double));
    cudaMalloc((void **)&T_temp, grid_size * sizeof(double));
    cudaMalloc((void **)&U, grid_size * sizeof(double));
    cudaMalloc((void **)&V, grid_size * sizeof(double));

    cudaMalloc((void **)&dx, sizeof(double));
    cudaMalloc((void **)&dy, sizeof(double));
    cudaMalloc((void **)&dt, sizeof(double));
    cudaMalloc((void **)&alpha, sizeof(double));
    cudaMalloc((void **)&gamma, sizeof(double));
    cudaMalloc((void **)&size_x, sizeof(double));
    cudaMalloc((void **)&size_y, sizeof(double));
}

void CUDA_solver::pre_process(Fields &field, Grid &grid, Discretization &discretization){
    
    cudaMemcpy(geometry_file, grid.get_geometry_data().data(), domain_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(T, field.t_matrix().data(), grid_size * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(U, field.u_matrix().data(), grid_size * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(V, field.v_matrix().data(), grid_size * sizeof(double),cudaMemcpyHostToDevice);


    double var = grid.dx();
    cudaMemcpy(dx, &var, sizeof(double),cudaMemcpyHostToDevice);
    var = grid.dy();
    cudaMemcpy(dy, &var, sizeof(double),cudaMemcpyHostToDevice);
    var = field.dt();
    cudaMemcpy(dt, &var, sizeof(double),cudaMemcpyHostToDevice);
    var = field.get_alpha();
    cudaMemcpy(alpha, &var, sizeof(double),cudaMemcpyHostToDevice);
    var = discretization.get_gamma();
    cudaMemcpy(gamma, &var, sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(size_x, &grid_size_x, sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(size_y, &grid_size_y, sizeof(int),cudaMemcpyHostToDevice);

}

void CUDA_solver::calc_T(){
    cudaMemcpy(T_temp, T, grid_size * sizeof(double), cudaMemcpyDeviceToDevice);
    num_blocks_2d = get_num_blocks_2d(grid_size_x, grid_size_y);
    calc_T_kernel<<<num_blocks_2d, block_size_2d>>>(T, T_temp, U, V, dx, dy, dt, alpha, gamma, size_x, size_y);
}

void CUDA_solver::apply_boundary(){

}

void CUDA_solver::calc_fluxes(){
    
}

void CUDA_solver::calc_rs(){
    
}

void CUDA_solver::post_process(Fields &field){
    cudaMemcpy((void *)field.t_matrix().data(), T, grid_size * sizeof(double), cudaMemcpyDeviceToHost);
}

CUDA_solver::~CUDA_solver(){
    cudaFree(geometry_file);
    cudaFree(T);
    cudaFree(T_temp);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dt);
    cudaFree(gamma);
    cudaFree(alpha);
}
