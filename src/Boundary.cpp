#include "Boundary.hpp"
#include <cmath>
#include <iostream>

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells) : _cells(cells) {}

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_temperature)
    : _cells(cells), _wall_temperature(wall_temperature) {}

void FixedWallBoundary::apply(Fields &field) {

        // Bottom wall

        int imax = field.p_matrix().imax();
        int jmax = field.p_matrix().jmax();

        for (int i = 1; i <= imax; i++) {

            field.u(i, 0) = -field.u(i, 1);
            field.v(i, 0) = 0.0;
            field.p(i, 0) = field.p(i, 1);
            field.g(i, 0) = field.v(i, 0);
        }

        // Left wall

        for (int j = 1; j <= jmax; j++) {

            field.u(0, j) = 0.0;
            field.v(0, j) = -field.v(1, j);
            field.p(0, j) = field.p(1, j);
            field.f(0, j) = field.u(0, j);
        }


        // Right wall

        for (int j = 1; j <= jmax; j++) {

            field.u(imax, j) = 0.0;
            field.v(imax + 1, j) = -field.v(imax, j);
            field.p(imax + 1, j) = field.p(imax, j);
            field.f(imax, j) = field.u(imax, j);
        }

}

MovingWallBoundary::MovingWallBoundary(std::vector<Cell *> cells, double wall_velocity) : _cells(cells) {
    _wall_velocity.insert(std::pair(LidDrivenCavity::moving_wall_id, wall_velocity));
}

MovingWallBoundary::MovingWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_velocity,
                                       std::map<int, double> wall_temperature)
    : _cells(cells), _wall_velocity(wall_velocity), _wall_temperature(wall_temperature) {}

void MovingWallBoundary::apply(Fields &field) {
    
        // Top wall - moving
        int imax = field.p_matrix().imax();
        int jmax = field.p_matrix().jmax();

        for (int i = 1; i <= imax; i++) {

            field.u(i, jmax + 1) = 2 - field.u(i, jmax);
            field.v(i, jmax) = 0.0;
            field.p(i, jmax + 1) = field.p(i, jmax);
            field.g(i, jmax) = field.v(i, jmax);
        }
}
