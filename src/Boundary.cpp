#include "Boundary.hpp"
#include <cmath>
#include <iostream>

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells) : _cells(cells) {}

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_temperature)
    : _cells(cells), _wall_temperature(wall_temperature) {}

void FixedWallBoundary::apply(Fields &field) {

        // Bottom wall

        int imax = 10;
        int jmax = 10;

        for (int i = 1; i <= imax; i++) {

            field.u(i, 0) = -field.u(i, 1);
            field.v(i, 0) = 0.0;
        }

        for (int i = 0; i <= imax; i++) {

            field.p(i, 0) = field.p(i, 1);

        }

        // Left wall

        for (int j = 1; j <= jmax; j++) {

            field.u(0, j) = 0.0;
            field.v(0, j) = -field.v(1, j);
        }

        for (int j = 1; j <= jmax; j++) {

            field.p(0, j) = field.p(1, j);
        }

        // Right wall

        for (int j = 1; j <= jmax; j++) {

            field.u(imax, j) = 0.0;
            field.v(imax + 1, j) = -field.v(imax, j);
        }

        for (int j = 1; j <= jmax; j++) {

            field.p(imax + 1, j) = field.p(imax, j);
        }


}

MovingWallBoundary::MovingWallBoundary(std::vector<Cell *> cells, double wall_velocity) : _cells(cells) {
    _wall_velocity.insert(std::pair(LidDrivenCavity::moving_wall_id, wall_velocity));
}

MovingWallBoundary::MovingWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_velocity,
                                       std::map<int, double> wall_temperature)
    : _cells(cells), _wall_velocity(wall_velocity), _wall_temperature(wall_temperature) {}

void MovingWallBoundary::apply(Fields &field) {

    // later work
    int jmaxb = 12;
        int imax = 10;
        int jmax = 10;

        for (int i = 0; i <= imax; i++) {

            field.u(i, jmax) = 2 - field.u(i, jmax - 1);
            field.v(i, jmax) = 0.0;
        }

        for (int i = 0; i <= imax; i++) {

            field.p(i, jmaxb) = field.p(i, jmaxb - 1);
        }

}
