<details><summary>Description</summary>

The project deals with parallelization of the fluG Solver using GPUs. We utilise CUDA API for this task.

</details>

<details><summary>How to build and run the project</summary>

mkdir build \
cd build \
cmake .. \
make \
./fluG ../example_cases/<case_name>/<case_name.dat>

</details>

<details><summary>Files that were edited</summary>

1. CUDA_solver.cpp & CUDA_solver.hpp \
Implemented the CUDA_solver class which contains functions to allocate memory on GPU, send field data to the GPU from CPU, handle bounderies, compute pressureand velocities on GPU, calculate timestep size and send the computed field data from GPU to CPU.
The SOR for pressure solver is implemented with the red-black scheme.

2. Case.cpp \
The calls to pre-process, computation of the field variables using the CUDA_solver class methods , post-process and output the results to vtk.

</details>

<details><summary>Results and Validation</summary>

The result obtained in the parallel cases is similar to those obtained with the serial case, for both the channel flow and fluid trap problems.

![alt text](<./docs/validation.png>) 

</details>

<details><summary>Performance Analysis</summary>

Scaling was performed on the Channel Flow and Fluid Trap case by scaling the no. of grid points in each direction by 3 and 5 times while maintaining the same Reynolds number. \
The charts regarding Performance Analysis can be found in the results directory of this branch.

</details>

<details><summary>Documentation</summary>

The project report can be found in the docs folder.

</details>
