#include "Fields.hpp"

#include <algorithm>
#include <iostream>

Fields::Fields(double nu, double dt, double tau, int imax, int jmax, double UI, double VI, double PI, const Grid &grid, double alpha)
    : _nu(nu), _dt(dt), _tau(tau),_alpha(alpha) {

    _U = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _V = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _P = Matrix<double>(imax + 2, jmax + 2, 0.0);

    _F = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _G = Matrix<double>(imax + 2, jmax + 2, 0.0);
    _RS = Matrix<double>(imax + 2, jmax + 2, 0.0);

    for (auto currentCell : grid.fluid_cells()){
        int i = currentCell->i();
        int j = currentCell->j();
        _U(i,j) = UI;
        _V(i,j) = VI;
        _P(i,j) = PI;
    }
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

    for (auto currentCell: grid.fixed_wall_cells()){

        int i = currentCell->i();
        int j = currentCell->j();

        

        // obstacles B_NE
        if(currentCell->is_border(border_position::TOP) && currentCell->is_border(border_position::RIGHT)){
            _F(i, j) = _U(i, j);
            _G(i, j) = _V(i, j);
        }

        // obstacles B_SE
       else if(currentCell->is_border(border_position::BOTTOM) && currentCell->is_border(border_position::RIGHT)){
            _F(i, j) = _U(i, j);
            _G(i,j - 1) = _V(i,j - 1);

        }

        // obstacles B_NW
        else if(currentCell->is_border(border_position::TOP) && currentCell->is_border(border_position::LEFT)){
            _F(i - 1, j) = _U(i - 1, j);
            _G(i, j) = _V(i, j);
        }

        // obstacles B_SW
        else if(currentCell->is_border(border_position::BOTTOM) && currentCell->is_border(border_position::LEFT)){
            _F(i - 1, j) = _U(i - 1, j);
            _G(i, j - 1) = _V(i, j - 1);
        }
        else if(currentCell -> is_border(border_position::TOP))
            _G(i,j) = _V(i,j);

        else if(currentCell -> is_border(border_position::BOTTOM))
            _G(i,j - 1) = _V(i,j - 1);

        else if(currentCell -> is_border(border_position::LEFT))
            _F(i - 1, j) = _U(i - 1, j);

        else if(currentCell -> is_border(border_position::RIGHT))
            _F(i, j) = _U(i, j);

    }

    for (auto currentCell: grid.moving_wall_cells()){

        int i = currentCell->i();
        int j = currentCell->j();

        _G(i,j - 1) = _V(i,j - 1);
    }

    for (auto currentCell: grid.inflow_cells()){

        int i = currentCell->i();
        int j = currentCell->j();

        _F(i,j) = _U(i,j);
    }

    for (auto currentCell: grid.outflow_cells()){

        int i = currentCell->i();
        int j = currentCell->j();

        _F(i - 1,j) = _U(i - 1,j);
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

    // for (int i = 1; i < grid.imax() + 1; ++i ) {
    //     for (int j = 1; j < grid.jmax() + 1; ++j){
    //         _U(i, j) = _F(i, j) - (_dt/grid.dx()) * (_P(i + 1, j) - _P(i, j));           
    //     }       
    // }

    // for (int i = 1; i < grid.imax() + 1; ++i ) {
    //     for (int j = 1; j < grid.jmax(); ++j){
    //         _V(i, j) = _G(i, j) - (_dt/grid.dy()) * (_P(i, j + 1) - _P(i, j));
    //     }       
    // }

    for (auto currentCell : grid.fluid_cells()){
        int i = currentCell->i();
        int j = currentCell->j();
        if ((currentCell->neighbour(border_position::RIGHT)->type() == cell_type::FLUID) || (currentCell->neighbour(border_position::RIGHT)->type() == cell_type::OUTFLOW)) {
            _U(i, j) = _F(i, j) - (_dt/grid.dx()) * (_P(i + 1, j) - _P(i, j));           
        }
        if ((currentCell->neighbour(border_position::TOP)->type() == cell_type::FLUID) || (currentCell->neighbour(border_position::RIGHT)->type() == cell_type::OUTFLOW)) {
            _V(i, j) = _G(i, j) - (_dt/grid.dy()) * (_P(i, j + 1) - _P(i, j));
        }
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
    }
    double t2 = grid.dx() / u_max;
    double t3 = grid.dy() / v_max;   
    double t4 = 1 / (2 * _alpha * (1/(grid.dx()*grid.dx()) + 1/(grid.dy()*grid.dy())));
    _dt = _tau * std::min({t1, t2, t3, t4});
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
