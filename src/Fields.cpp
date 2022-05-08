#include "Fields.hpp"

#include <algorithm>
#include <iostream>

Fields::Fields(double nu, double dt, double tau, int imax, int jmax, double UI, double VI, double PI)
    : _nu(nu), _dt(dt), _tau(tau) {
    _U = Matrix<double>(imax + 2, jmax + 2, UI);
    _V = Matrix<double>(imax + 2, jmax + 2, VI);
    _P = Matrix<double>(imax + 2, jmax + 2, PI);

    _F = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _G = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _RS = Matrix<double>(imax + 2, jmax + 2, 0.0);
}

/********************************************************************************
 * This function calculates fluxes F and G as mentioned in equation (9) and (10)
 *******************************************************************************/
void Fields::calculate_fluxes(Grid &grid) {
    for (auto currentCell : grid.fluid_cells()) {
        int i = currentCell->i();
        int j = currentCell->j();
        _F(i,j) = _U(i,j) + _dt * (_nu*Discretization::diffusion(_U,i,j) - Discretization::convection_u(_U,_V,i,j) + _gx); 
        _G(i,j) = _V(i,j) + _dt * (_nu*Discretization::diffusion(_V,i,j) - Discretization::convection_v(_U,_V,i,j) + _gy);
    }
}

/********************************************************************************
 * This function calculates the RHS of equation (11) i.e. Pressure SOR
 *******************************************************************************/
void Fields::calculate_rs(Grid &grid) {
    for (auto currentCell : grid.fluid_cells()) {
        int i = currentCell->i();
        int j = currentCell->j();
        _RS(i, j) = 1 / _dt * ((_F(i, j) - _F(i - 1, j)) / grid.dx() + 
                               (_G(i, j) - _G(i, j - 1)) / grid.dy()); 
    }
}

/*****************************************************************************************
 * This function updates velocity after Pressure SOR as mentioned in equation (7) and (8)
 ****************************************************************************************/
void Fields::calculate_velocities(Grid &grid) {
    for (auto currentCell : grid.fluid_cells()) {    
        int i = currentCell->i();
        int j = currentCell->j();
        _U(i, j) = _F(i, j) - (_dt/grid.dx())*(_P(i + 1, j) - _P(i, j));
        _V(i, j) = _G(i, j) - (_dt/grid.dy())*(_P(i, j + 1) - _P(i, j));
    }
}

/*****************************************************************************************
 * This function calculate timestep for adaptive time stepping using equation (13)
 ****************************************************************************************/
double Fields::calculate_dt(Grid &grid) {
    double t1 = 1 / (2 * _nu * (1/(grid.dx()*grid.dx()) + 1/(grid.dy()*grid.dy())));
    double u_max = 0, v_max = 0, temp;
    for (int i = 0; i < grid.imaxb(); ++i){
        for(int j=0;j<grid.jmaxb();++j)
        {
            temp = std::abs(_U(i,j));
            if(temp > u_max){
                u_max = temp;
            }
            temp = std::abs(_V(i,j));
            if(temp > v_max){
                v_max = temp;
            }
        }
    double t2 = grid.dx() / u_max;
    double t3 = grid.dy() / v_max;    
    _dt = _tau * std::min({t1, t2, t3});
    return _dt;
}

double &Fields::p(int i, int j) { return _P(i, j); }
double &Fields::u(int i, int j) { return _U(i, j); }
double &Fields::v(int i, int j) { return _V(i, j); }
double &Fields::f(int i, int j) { return _F(i, j); }
double &Fields::g(int i, int j) { return _G(i, j); }
double &Fields::rs(int i, int j) { return _RS(i, j); }

Matrix<double> &Fields::p_matrix() { return _P; }

double Fields::dt() const { return _dt; }
