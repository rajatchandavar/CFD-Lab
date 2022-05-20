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
        }

        // Top Wall
        if(currentCell->is_border(border_position::BOTTOM)){
            field.u(i, j) = -field.u(i, j - 1);
            field.v(i, j) = 0.0;
            field.p(i, j) = field.p(i, j - 1);
        }

        // Left Wall
        if(currentCell->is_border(border_position::RIGHT)){
            field.u(i, j) = 0.0;
            field.v(i, j) = -field.v(i + 1, j);
            field.p(i, j) = field.p(i + 1, j);
        }

        // Right Wall
        if(currentCell->is_border(border_position::LEFT)){
            //Since u grid is staggered, the u velocity of cells to left of ghost layer should be set to 0.
            field.u(i - 1, j) = 0.0; 
            field.v(i, j) = -field.v(i - 1, j);
            field.p(i, j) = field.p(i - 1, j);
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
        //Since v grid is staggered, the v velocity of cells to below of ghost layer should be set to 0.
        field.v(i,j - 1) = 0.0;
        field.p(i,j) = field.p(i, j-1);
    }
}

InFlowBoundary::InFlowBoundary(std::vector<Cell *> cells, double UIN, double VIN) : _cells(cells) {
    _UIN.insert(std::pair(GEOMETRY_PGM::inflow_id, UIN));
    _VIN.insert(std::pair(GEOMETRY_PGM::inflow_id, VIN));
}

InFlowBoundary::InFlowBoundary(std::vector<Cell *> cells, std::map<int, double> UIN, std::map<int, double> VIN,
                                       std::map<int, double> wall_temperature)
    : _cells(cells), _UIN(UIN), _VIN(VIN), _wall_temperature(wall_temperature) {}

void InFlowBoundary::apply(Fields &field) {
    for (auto currentCell: _cells){
        int i = currentCell->i();
        int j = currentCell->j();
            field.u(i,j) = _UIN[1];
            field.v(i,j) = 2 * _VIN[1] - field.v(i + 1, j);
            field.p(i,j) = field.p(i + 1, j);
    }
}

OutFlowBoundary::OutFlowBoundary(std::vector<Cell *> cells, double POUT) : _cells(cells) {
    _POUT.insert(std::pair(GEOMETRY_PGM::outflow_id, POUT));
}

OutFlowBoundary::OutFlowBoundary(std::vector<Cell *> cells, std::map<int, double> POUT,
                                       std::map<int, double> wall_temperature)
    : _cells(cells), _POUT(POUT), _wall_temperature(wall_temperature) {}

void OutFlowBoundary::apply(Fields &field) {
    for (auto currentCell: _cells){
        int i = currentCell->i();
        int j = currentCell->j();

            field.u(i,j) = field.u(i - 1,j);
            field.v(i,j) = field.v(i - 1,j);
            field.p(i,j) = 2 * _POUT[2] - field.p(i - 1, j);
    }
}
