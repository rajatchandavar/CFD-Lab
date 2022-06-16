#pragma once

#include"Datastructures.hpp"
#include <map>

class Communication{
    public:

    inline static int iproc, jproc;

    static void init_parallel(int argc, char **argv);

    static void communicate(Matrix<double> &);

    static double reduce_min(double );

    static double reduce_max(double );

    static double reduce_sum(double );

    static void broadcast(double );

    static int get_rank();

    static int get_size();
    
    static void finalize();

    private:

    static std::map<char, int> _assign_neighbours(int);

};