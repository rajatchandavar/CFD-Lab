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

__device__ double convection_u(double *U, double *V, double gamma, int i, int j, double dx, double dy, int *size_x)
{

    double du2_dx = 1/ dx * (pow(interpolate(U,i,j,1,0,size_x), 2) - pow(interpolate(U,i,j,-1,0, size_x), 2)) +
                        gamma/dx * ((std::abs(interpolate(U,i,j,1,0,size_x)) * (at(U,i, j) - at(U,i + 1, j)) / 2) -
                         std::abs(interpolate(U,i,j,-1,0,size_x))*(at(U,i - 1, j)-at(U,i, j)) / 2) ;
    double duv_dy = 1/ dy * (((interpolate(V,i,j,1,0,size_x)) * (interpolate(U,i,j,0,1,size_x)))  - ( (interpolate(V,i,j-1,1,0,size_x)) * (interpolate(U,i,j,0,-1, size_x)))  ) +
                             gamma / dy * ( (std::abs(interpolate(V,i,j,1,0,size_x))*(at(U,i, j) - at(U,i, j + 1)) / 2) - 
                             (std::abs(interpolate(V,i,j-1,1,0,size_x)) * (at(U,i, j - 1) - at(U,i, j)) / 2 ));    

     double result = du2_dx + duv_dy;
     return result;
}

__device__ double convection_v(double *U, double *V, double gamma, int i, int j, double dx, double dy, int *size_x)
{

    double dv2_dy = 1/ dy * (pow(interpolate(V,i,j,0,1,size_x), 2) - pow(interpolate(V,i,j-1,0,1,size_x), 2)) +
                        gamma/dy * ((std::abs(interpolate(V,i,j,0,1,size_x)) * (at(V,i, j) - at(V,i, j+1)) / 2) -
                        std::abs(interpolate(V, i, j - 1, 0, 1,size_x)) * (at(V,i, j-1)- at(V,i, j)) / 2) ;
    double duv_dx = 1/ dx * (((interpolate(U, i, j, 0, 1, size_x)) * (interpolate(V, i, j, 1, 0,size_x)))  - ( (interpolate(U,i-1,j,0,1,size_x)) * (interpolate(V,i-1,j,1,0,size_x)))) +
                            gamma / dx * ( (std::abs(interpolate(U,i,j,0,1,size_x))*(at(V,i, j) - at(V,i + 1, j)) / 2) - 
                            (std::abs(interpolate(U,i-1,j,0,1,size_x)) * (at(V,i-1, j) - at(V,i, j)) / 2 ));    

    double result = dv2_dy + duv_dx;
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

//Boundaries

__device__ void FixedWallBoundary(double *U, double *V, double *P, double *T, int i, int j) {

        // obstacles B_NE (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North and East directions 

        if(at(geometry_data,i,j+1)==0 && at(geometry_data,i+1,j)==0) {
            at(U, i, j) = 0.0;
            at(U, i - 1, j) = -at(U, i - 1, j + 1);
            at(V, i, j) = 0.0;
            at(V, i, j - 1) = -at(V, i + 1, j - 1);
            at(P, i, j) = ((P, i, j + 1) + (P, i + 1, j))/2;

            if(isHeatTransfer==1) {
                if(at(geometry_data,i,j) == adiabatic_id)
                    at(T, i, j) = (at(T, i + 1, j) + at(T, i, j + 1))/2;
                else if (at(geometry_data,i,j) == hot_id)
                    at(T,i, j) = 2*wall_temp_h - (at(T,i, j + 1) + at(T,i + 1, j) )/2;
                else if (at(geometry_data,i,j) == cold_id)
                    at(T,i, j) = 2*wall_temp_c - (at(T,i, j + 1) + at(T,i + 1, j) )/2;
                }

    }

        // obstacles B_SE (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South and East directions 

            if(at(geometry_data,i,j-1)==0 && at(geometry_data,i+1,j)==0) {

                at(U, i, j) = 0.0;
                at(U, i - 1, j) = -at(U, i - 1, j - 1);
                at(V, i, j - 1) = 0.0;
                at(V, i, j) = -at(V, i + 1, j);
                at(P,i, j) = (at(P,i + 1, j) + at(P,i, j - 1))/2;

                if(isHeatTransfer==1) {
                    if(at(geometry_data,i,j) == adiabatic_id)
                    at(T,i, j) = (at(T, i + 1, j) + at(T, i, j - 1))/2;
                    else if (at(geometry_data,i,j) == hot_id)
                        at(T, i, j) = 2*wall_temp_h - (at(T,i, j - 1) + at(T,i + 1, j) )/2;
                    else if (at(geometry_data,i,j) == cold_id)
                        at(T,i, j) = 2*wall_temp_c - (at(T,i, j - 1) + at(T,i + 1, j) )/2;
                }

            }

        // obstacle B_NW (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North and West directions 
            
            if(at(geometry_data,i,j+1)==0 && at(geometry_data,i-1,j)==0) {

                at(U,i - 1, j) = 0.0;
                at(U,i, j) = -at(U,i, j + 1);
                at(V,i, j) = 0.0;
                at(V,i, j - 1) = -at(V,i - 1, j - 1);
                at(P,i,j) = (at(P,i - 1, j) + at(P,i, j + 1))/2;

                if(isHeatTransfer==1){
                    if(at(geometry_data,i,j) == adiabatic_id)
                        at(T,i, j) = (at(T,i - 1, j) + at(T,i, j + 1))/2;
                    else if (at(geometry_data,i,j) == hot_id)
                        at(T,i, j) = 2*wall_temp_h - (at(T,i, j + 1) + at(T,i - 1, j) )/2;
                    else if (at(geometry_data,i,j) == cold_id)
                        at(T,i, j) = 2*wall_temp_c - (at(T,i, j + 1) + at(T,i - 1, j) )/2;
                }

        }

        // obstacle B_SW (Corner Cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South and West directions 

        else if(at(geometry_data,i,j-1)==0 && at(geometry_data,i-1,j)==0){
            at(U,i - 1, j) = 0.0;
            at(U,i, j) = at(U,i, j - 1);
            at(V,i, j - 1) = 0.0;
            at(V,i, j) = -at(V,i - 1, j);
            at(P,i, j) = (at(P,i - 1, j) + at(P,i, j - 1))/2;
           
            if(isHeatTransfer==1){
                if(at(geometry_data,i,j) == adiabatic_id)
                   at(T,i, j) = (at(T,i - 1, j) + at(T,i, j - 1))/2;
                else if (at(geometry_data,i,j) == hot_id)
                    at(T,i, j) = 2*wall_temp_h - (at(T,i, j - 1) + at(T,i - 1, j) )/2;
                else if (at(geometry_data,i,j) == cold_id)
                    at(T,i, j) = 2*wall_temp_c - (at(T,i, j - 1) + at(T,i - 1, j) )/2;
            }
            
        }

        // Bottom Wall B_N (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the North direction

        else if(at(geometry_data,i,j+1)==0){
            at(U,i, j) = -at(U,i, j + 1);
            at(V,i, j) = 0.0;
            at(P,i, j) = at(P,i, j + 1);

            if(isHeatTransfer==1){
                if(at(geometry_data,i,j) == adiabatic_id)
                    at(T,i, j) = at(T,i, j + 1);
                else if (at(geometry_data,i,j) == hot_id)
                    at(T,i, j) = 2*wall_temp_h - at(T,i, j + 1);
                else if (at(geometry_data,i,j) == cold_id)
                    at(T,i, j) = 2*wall_temp_c - at(T,i, j + 1);
            }
        }

        // Top Wall B_S (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the South direction

        else if(at(geometry_data,i,j-1)==0){

            at(U,i, j) = -at(U,i, j - 1);
            at(V,i, j) = 0.0;
            at(P,i, j) = at(P,i, j - 1);

            if(isHeatTransfer==1){
                if(at(geometry_data,i,j) == adiabatic_id)
                    at(T,i, j) = at(T,i, j - 1);
                else if (at(geometry_data,i,j) == hot_id)
                    at(T,i, j) = 2 * wall_temp_h - at(T,i, j - 1);
                else if (at(geometry_data,i,j) == cold_id)
                    at(T,i, j) = 2 * wall_temp_c - at(T,i, j - 1);
            }
        }

        // Left Wall B_E (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on 
        // the East direction

        else if(at(geometry_data,i+1,j)==0){
            at(U,i, j) = 0.0;
            at(V,i, j) = -at(V,i + 1, j);
            at(P,i, j) = at(P,i + 1, j);

            if(isHeatTransfer==1){
                if(at(geometry_data,i,j) == adiabatic_id)
                    at(T,i, j) = at(T,i + 1, j);
                else if (at(geometry_data,i,j) == hot_id) 
                    at(T,i, j) = 2*wall_temp_h - at(T,i + 1, j);        
                else if (at(geometry_data,i,j) == cold_id)
                    at(T,i, j) = 2*wall_temp_c - at(T,i + 1, j);
            }
        }

        
        /***********************************************************************************************
        * Right Wall B_W (edge cell) - This section applies the appropriate boundary conditions to a fixed wall with fluid cells on the West direction *
        ***********************************************************************************************/

        else if(at(geometry_data,i-1,j))==0){
            //Since u grid is staggered, the u velocity of cells to left of ghost layer should be set to 0.
            at(U,i - 1, j) = 0.0; 
            at(V,i, j) = -at(V,i - 1, j);
            at(P,i, j) = at(P,i - 1, j);

            if(isHeatTransfer==1){
                if(at(geometry_data,i,j) == adiabatic_id)
                    at(T,i, j) = at(T,i - 1, j);
                else if (at(geometry_data,i,j) == hot_id)
                    at(T,i, j) = 2*wall_temp_h - at(T,i - 1, j);
                else if (at(geometry_data,i,j) == cold_id)
                    at(T,i, j) = 2*wall_temp_c - at(T,i - 1, j);
            }
        }
}

__device__ void MovingWallBoundary(double *U, double *V, double *P, double *T, int i, int j) {
        
        at(U,i, j) = 2 * (wall_velocity- at(U,i, j-1));
        //Since v grid is staggered, the v velocity of cells to below of ghost layer should be set to 0.
        at(V,i,j - 1) = 0.0;
        at(P,i,j) = at(P,i, j-1);
}

__device__ void InFlowBoundary(double *U, double *V, double *P, double *T, int i, int j) {
        at(U,i,j) = UIN;
        at(V,i,j) = 2 * VIN - at(V,i + 1, j);
        at(P,i,j) = at(P,i + 1, j);
}

__device__ void OutFlowBoundary(double *U, double *V, double *P, double *T, int i, int j) {

            at(U,i,j) = at(U,i - 1,j);
            at(V,i,j) = at(V,i - 1,j);
            at(P,i,j) = 2 * POUT - at(P,i - 1, j);
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
    cudaMalloc(&geometry_data, domain_size * sizeof(int));

    grid_size = grid.imaxb() * grid.jmaxb();
    grid_size_x = grid.imaxb();
    grid_size_y = grid.jmaxb();

    cudaMalloc((void **)&T, grid_size * sizeof(double));
    cudaMalloc((void **)&T_temp, grid_size * sizeof(double));
    cudaMalloc((void **)&U, grid_size * sizeof(double));
    cudaMalloc((void **)&V, grid_size * sizeof(double));
    cudaMalloc((void **)&P, grid_size * sizeof(double));
    cudaMalloc((void **)&F, grid_size * sizeof(double));
    cudaMalloc((void **)&G, grid_size * sizeof(double));


    cudaMalloc((void **)&dx, sizeof(double));
    cudaMalloc((void **)&dy, sizeof(double));
    cudaMalloc((void **)&dt, sizeof(double));
    cudaMalloc((void **)&alpha, sizeof(double));
    cudaMalloc((void **)&gamma, sizeof(double));
    cudaMalloc((void **)&size_x, sizeof(double));
    cudaMalloc((void **)&size_y, sizeof(double));

    cudaMalloc((void **)&fluid_id, sizeof(int));
    cudaMalloc((void **)&moving_wall_id, sizeof(int));
    cudaMalloc((void **)&fixed_wall_id, sizeof(int));
    cudaMalloc((void **)&inflow_id, sizeof(int));
    cudaMalloc((void **)&outflow_id, sizeof(int));
    cudaMalloc((void **)&adiabatic_id, sizeof(int));
    cudaMalloc((void **)&hot_id, sizeof(int));
    cudaMalloc((void **)&cold_id, sizeof(int));

    cudaMalloc((void **)&wall_temp_a, sizeof(double));
    cudaMalloc((void **)&wall_temp_h, sizeof(double));
    cudaMalloc((void **)&wall_temp_c, sizeof(double));
    cudaMalloc((void **)&isHeatTransfer, sizeof(bool));

    cudaMalloc((void **)&wall_velocity, sizeof(double));
    cudaMalloc((void **)&UIN, sizeof(double));
    cudaMalloc((void **)&VIN, sizeof(double));
    cudaMalloc((void **)&POUT, sizeof(double));

}

void CUDA_solver::pre_process(Fields &field, Grid &grid, Discretization &discretization){
    
    cudaMemcpy(geometry_data, grid.get_geometry_data().data(), domain_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(T, field.t_matrix().data(), grid_size * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(U, field.u_matrix().data(), grid_size * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(V, field.v_matrix().data(), grid_size * sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(P, field.p_matrix().data(), grid_size * sizeof(double),cudaMemcpyHostToDevice);

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

    cudaMemcpy(moving_wall_id, &(GEOMETRY_PGM::moving_wall_id), sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(fixed_wall_id, &(GEOMETRY_PGM::fixed_wall_id), sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(inflow_id, &(GEOMETRY_PGM::inflow_id), sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(outflow_id, &(GEOMETRY_PGM::outflow_id), sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(hot_id, &(GEOMETRY_PGM::hot_id), sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(cold_id, &(GEOMETRY_PGM::cold_id), sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(adiabatic_id, &(GEOMETRY_PGM::adiabatic_id), sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(wall_temp_a, &wall_temp_a, sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(wall_temp_h, &wall_temp_h, sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(wall_temp_c, &wall_temp_c, sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(isHeatTransfer, &field.isHeatTransfer(), sizeof(bool),cudaMemcpyHostToDevice);

    cudaMemcpy(wall_velocity, &(LidDrivenCavity::wall_velocity), sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(UIN, &UIN, sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(VIN, &VIN, sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(POUT, &(GEOMETRY_PGM::POUT), sizeof(double),cudaMemcpyHostToDevice);



}

void CUDA_solver::calc_T(){
    cudaMemcpy(T_temp, T, grid_size * sizeof(double), cudaMemcpyDeviceToDevice);
    num_blocks_2d = get_num_blocks_2d(grid_size_x, grid_size_y);
    calc_T_kernel<<<num_blocks_2d, block_size_2d>>>(T, T_temp, U, V, dx, dy, dt, alpha, gamma, size_x, size_y);
}

__global__ apply_boundary() {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    num_blocks_2d = get_num_blocks_2d(grid_size_x, grid_size_y);

    if (i < *size_x && j < *size_y) {

        if (at(geometry_data,i,j)==3 || at(geometry_data,i,j)==5 || at(geometry_data,i,j)== 6 || at(geometry_data,i,j)== 7)
            FixedWallBoundary(U, V, P, T, i, j);

        else if (at(geometry_data,i,j)==8)
            MovingWallBoundary(U, V, P, T, i, j);

        else if (at(geometry_data,i,j)==1)
            InFlowBoundary(U, V, P, T, i, j);

        else (at(geometry_data,i,j)==2)
            OutFlowBoundary(U, V, P, T, i, j);

    }

}

__device__ void calc_fluxes(double *F, double *G, double *U, double *V, double gx, double gy, int i, int j, double dx, double dy, int *size_x, int *size_y, double gamma, double beta, double nu, double dt, bool isHeatTransfer){



    at(F,i,j) = at(U,i,j) + dt * (nu*diffusion(U,i,j,dx,dy,size_x) - convection_u(U,V,gamma,i,j,dx,dy,size_x) + gx);
    at(G,i,j) = at(V,i,j) + dt * (nu*diffusion(V,i,j,dx,dy,size_x) - convection_v(U,V,gamma,i,j,dx,dy,size_x) + gy);
    
    if(isHeatTransfer){
       
            at(F,i,j) = at(F,i,j) - beta * dt / 2.0 * (at(T,(i,j) + at(T,(i + 1, j)) * gx - dt * gx;
            at(G,i,j) = at(G,i,j) - beta * dt / 2.0 * (at(T,(i,j) + at(T,(i, j + 1)) * gy - dt * gy;
    
    }
    
    if(at(geometry_data,i,j) == 3)
    {
         // B_NE fixed wall corner cell with fluid cells on the North and East directions 

        if(at(geometry_data,i,j+1)==0 && at(geometry_data,i+1,j)==0){
            at(F,i, j) = at(U,i, j);
            at(G,i, j) = at(V,i, j);
        }

        // B_SE fixed wall corner cell with fluid cells on the South and East directions 

       else if(at(geometry_data,i,j-1)==0 && at(geometry_data,i+1,j)==0){
            at(F,i, j) = at(U,i, j);
            at(G,i,j - 1) = at(V,i,j - 1);

        }

        // B_NW fixed wall corner cell with fluid cells on the North and West directions 

        else if(at(geometry_data,i,j+1)==0 && at(geometry_data,i-1,j)==0){
            at(F,i - 1, j) = at(U,i - 1, j);
            at(G,i, j) = at(V,i, j);
        }

        // B_SW fixed wall corner cell with fluid cells on the South and West directions 

        else if(at(geometry_data,i,j-1)==0 && at(geometry_data,i-1,j)==0){
            at(F,i - 1, j) = at(U,i - 1, j);
            at(G,i, j - 1) = at(V,i, j - 1);
        }
        else if(at(geometry_data,i,j+1)==0)
            at(G,i,j) = at(V,i,j);

        else if(at(geometry_data,i,j-1)==0))
            at(G,i,j - 1) = at(V,i,j - 1);

        else if(at(geometry_data,i-1,j)==0)
            at(F,i - 1, j) = at(U,i - 1, j);

        else if(at(geometry_data,i+1,j)==0)
            at(F,i, j) = at(U,i, j);

    }

    else if(at(geometry_data,i,j) == 8)
    {
        at(G,i,j - 1) = at(V,i,j - 1);
    }
    else if (at(geometry_data,i,j) == 1)
    {
        at(F,i,j) = at(U,i,j);
    }

    else if (at(geometry_data,i,j) == 2)
    {

        at(F,i - 1,j) = at(U,i - 1,j);
    }
}

void CUDA_solver::calc_rs(){
    
}

void CUDA_solver::post_process(Fields &field){
    cudaMemcpy((void *)field.t_matrix().data(), T, grid_size * sizeof(double), cudaMemcpyDeviceToHost);
}

CUDA_solver::~CUDA_solver(){
    cudaFree(geometry_data);
    cudaFree(T);
    cudaFree(U);
    cudaFree(V);
    cudaFree(P);
    cudaFree(F);
    cudaFree(G);
    cudaFree(T_temp);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dt);
    cudaFree(gamma);
    cudaFree(alpha);
    cudaFree(size_x);
    cudaFree(size_y);
    cudaFree(fluid_id);
    cudaFree(fixed_wall_id);
    cudaFree(moving_wall_id);
    cudaFree(inflow_id);
    cudaFree(outflow_id);
    cudaFree(adiabatic_id);
    cudaFree(hot_id);
    cudaFree(cold_id);
    cudaFree(wall_temp_a);
    cudaFree(wall_temp_c);
    cudaFree(wall_temp_h);
    cudaFree(isHeatTransfer);
    cudaFree(UIN);
    cudaFree(VIN);
    cudaFree(POUT);
    cudaFree(wall_velocity);
}
