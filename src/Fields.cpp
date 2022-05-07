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

void Fields::calculate_fluxes(Grid &grid) {
    for (auto currentCell : grid.fluid_cells()) {
        int i = currentCell->i();
        int j = currentCell->j();
        _F(i,j) = _U(i,j) + _dt * (_nu*Discretization::diffusion(_U,i,j) - Discretization::convection_u(_U,_V,i,j) + _gx); 
        _G(i,j) = _V(i,j) + _dt * (_nu*Discretization::diffusion(_V,i,j) - Discretization::convection_v(_U,_V,i,j) + _gy);
    }
}

void Fields::calculate_rs(Grid &grid) {
    for (auto currentCell : grid.fluid_cells()) {
        int i = currentCell->i();
        int j = currentCell->j();
        _RS(i, j) = 1 / _dt * ((_F(i, j) - _F(i - 1, j)) / grid.dx() + 
                               (_G(i, j) - _G(i, j - 1)) / grid.dy()); 
    }
}

void Fields::calculate_velocities(Grid &grid) {
    for (auto currentCell : grid.fluid_cells()) {    
        int i = currentCell->i();
        int j = currentCell->j();
        _U(i, j) = _F(i, j) - (_dt/grid.dx())*(_P(i + 1, j) - _P(i, j));
        _V(i, j) = _G(i, j) - (_dt/grid.dy())*(_P(i, j + 1) - _P(i, j));
    }
}

double Fields::calculate_dt(Grid &grid) {
    double t1 = 1 / (2 * _nu * (1/(grid.dx()*grid.dx()) + 1/(grid.dy()*grid.dy())));
    //std::vector<double> u_max(grid.jmaxb(),0), v_max(grid.jmaxb(),0);
    std::vector<double> u_max(grid.imaxb()*grid.jmaxb(),0), v_max(grid.imaxb()*grid.jmaxb(),0);
    //std::vector<double> u_max{1,2,3},v_max{1,2,3};
    //auto fn = [](auto &a, auto &b) {return abs(a) < abs(b);};
    for (int i = 0; i < grid.imaxb(); ++i){
        for(int j=0;j<grid.jmaxb();++j)
        {
        //std::cout<<"\n"<<*std::max_element(_U.get_row(1).begin(), _U.get_row(1).end(),fn);
        //std::cout << std::abs(*std::max_element(_U.get_row(j).begin(), _U.get_row(j).end()));
        //u_max.push_back(std::abs(*std::max_element(_U.get_row(j).begin(), _U.get_row(j).end(), fn)));
        //v_max.push_back(std::abs(*std::max_element(_V.get_row(j).begin(), _V.get_row(j).end(), fn)));

        u_max.push_back(std::abs(_U(i,j)));
        v_max.push_back(std::abs(_V(i,j)));
        }

    }

    //std::cout << std::abs(*std::max_element(u_max.begin(), u_max.end()));
    
    double t2 = grid.dx() / *std::max_element(u_max.begin(), u_max.end());
    double t3 = grid.dy() / *std::max_element(v_max.begin(), v_max.end());    
    // double t2 = 0.1, t3 = 0.5;
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
