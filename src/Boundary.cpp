#include "Boundary.hpp"
#include <cmath>
#include <iostream>

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells) : _cells(cells) {}

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_temperature)
    : _cells(cells), _wall_temperature(wall_temperature) {}

/*****************************************************************************************
 * This function applies boundary conditions to fixed wall as given in equation (15)-(17)
 ****************************************************************************************/
void FixedWallBoundary::apply(Fields &field) {

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
            //not necessary but done for sanity
            field.u(i, j) = 0.0; 
            //Since u grid is staggered, the u velocity of cells to left of ghost layer should be set to 0.
            field.u(i - 1, j) = 0.0; 
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

/***********************************************************************************************
 * This function applies boundary conditions to moving wall as given in equation (15)-(17)
 * The u velocity of moving wall is set such that average at the top boundary is wall velocity.
 **********************************************************************************************/
// Top Wall
void MovingWallBoundary::apply(Fields &field) {
    for (auto currentCell: _cells){
        int i = currentCell->i();
        int j = currentCell->j();
        field.u(i, j) = 2 * _wall_velocity[8] - field.u(i, j-1);
         //not necessary but done for sanity
        field.v(i,j) = 0.0;
        //Since v grid is staggered, the v velocity of cells to below of ghost layer should be set to 0.
        field.v(i,j - 1) = 0.0;
        field.p(i,j) = field.p(i, j-1);
        field.g(i,j) = field.v(i,j);
    }
}
