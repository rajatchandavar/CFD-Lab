#include "Boundary.hpp"
#include <cmath>
#include <iostream>

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells) : _cells(cells) {}

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_temperature)
    : _cells(cells), _wall_temperature(wall_temperature) {}

void FixedWallBoundary::apply(Fields &field) {

        // // Bottom wall

        // int imax = field.p_matrix().imax();
        // int jmax = field.p_matrix().jmax();

        // for (int i = 1; i <= imax; i++) {

        //     field.u(i, 0) = -field.u(i, 1);
        //     field.v(i, 0) = 0.0;
        //     field.p(i, 0) = field.p(i, 1);
        //     field.g(i, 0) = field.v(i, 0);
        // }

        // // Left wall

        // for (int j = 1; j <= jmax; j++) {

        //     field.u(0, j) = 0.0;
        //     field.v(0, j) = -field.v(1, j);
        //     field.p(0, j) = field.p(1, j);
        //     field.f(0, j) = field.u(0, j);
        // }


        // // Right wall

        // for (int j = 1; j <= jmax; j++) {

        //     field.u(imax, j) = 0.0;
        //     field.v(imax + 1, j) = -field.v(imax, j);
        //     field.p(imax + 1, j) = field.p(imax, j);
        //     field.f(imax, j) = field.u(imax, j);
        // }

        for (auto currentCell: _cells){
            int i = currentCell->i();
            int j = currentCell->j();
            
            // Bottom Wall
            if(currentCell->is_border(border_position::TOP)){
                field.u(i, j) = -field.u(i, j + 1);
                field.v(i, j) = 0.0;
                field.p(i, j) = field.p(i, j + 1);
                field.g(i, j) = field.v(i, j);
            }

            // Left Wall
            if(currentCell->is_border(border_position::RIGHT)){
                field.u(i, j) = 0.0;
                field.v(i, j) = -field.v(i + 1, j);
                field.p(i, j) = field.p(i + 1, j);
                field.f(i, j) = field.u(i, j);
            }

            // Right Wall
            if(currentCell->is_border(border_position::LEFT)){
                field.u(i, j) = 0.0; //not necessary but done for sanity
                field.u(i - 1, j) = 0.0; //Since u grid is staggered, add appr comments
                field.v(i, j) = -field.v(i - 1, j);
                field.p(i, j) = field.p(i - 1, j);
                field.f(i, j) = field.u(i, j);
            }
        }


}

MovingWallBoundary::MovingWallBoundary(std::vector<Cell *> cells, double wall_velocity) : _cells(cells) {
    _wall_velocity.insert(std::pair(LidDrivenCavity::moving_wall_id, wall_velocity));
}

MovingWallBoundary::MovingWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_velocity,
                                       std::map<int, double> wall_temperature)
    : _cells(cells), _wall_velocity(wall_velocity), _wall_temperature(wall_temperature) {}

void MovingWallBoundary::apply(Fields &field) {
    
        // // Top wall - moving
        // int imax = field.p_matrix().imax() - 2;
        // int jmax = field.p_matrix().jmax() - 2;
        // std::cout << "Moving wall\n" << imax << " " << jmax << '\n';
        // int i = 1;
        // //for (int i = 1; i <= imax; i++) {

        //     //std::cout << field.u(i, jmax + 1) << "\n";
        //     field.u(i, jmax + 1) = 2. - field.u(i, jmax);
        //     //field.v(i, jmax) = 0.0;
        //     //field.p(i, jmax + 1) = field.p(i, jmax);
        //     //field.g(i, jmax) = field.v(i, jmax);
        // //}
        for (auto currentCell: _cells){
            int i = currentCell->i();
            int j = currentCell->j();
            field.u(i, j) = 2 * _wall_velocity[8] - field.u(i, j-1) ;
            std::cout << _wall_velocity[8];
            field.v(i,j) = 0.0; //not necessary but done for sanity
            field.v(i,j - 1) = 0.0;//Since v grid is staggered, add appr comments
            field.p(i,j) = field.p(i, j-1);
            field.g(i,j) = field.v(i,j);
        }
        
}
