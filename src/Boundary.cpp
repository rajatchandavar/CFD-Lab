#include "Boundary.hpp"
#include <cmath>
#include <iostream>

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells) : _cells(cells) {}

FixedWallBoundary::FixedWallBoundary(std::vector<Cell *> cells, std::map<int, double> wall_temperature)
    : _cells(cells), _wall_temperature(wall_temperature) {}

void FixedWallBoundary::apply(Fields &field) {

        // Bottom wall

        for (int i = 1; i <= imax; i++) {

            _U(i, 0) = -_U(i, 1);
            _V(i, 0) = 0.0;
        }

        for (int i = 0; i <= imax; i++) {

            _P(i, 0) = _P(i, 1);

        }

        // Left wall

        for (int j = 1; j <= jmax; j++) {

            _U(0, j) = 0.0;
            _V(0, j) = -_V(1, j);
        }

        for (int j = 1; j <= jmax; j++) {

            _P(0, j) = _P(1, j);
        }

        // Right wall

        for (int j = 1; j <= jmax; j++) {

            _U(imax, j) = 0.0;
            _V(imax + 1, j) = -_V(imax, j);
        }

        for (int j = 1; j <= jmax; j++) {

            _P(imax + 1, j) = _P(imax, j);
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
    int j = jmaxb();

        for (int i = 0; i <= imax; i++) {

            _U(i, j) = 2 - _U(i, j - 1);
            _V(i, j) = 0.0;
        }

        for (int i = 0; i <= imax; i++) {

            _P(i, jmaxb) = _P(i, jmaxb - 1);
        }

}
