#pragma once
#include<Datastructures.hpp>

class Communication{
    public:

    static void init_parallel(int argc, char **argv);

    static void communicate(Matrix<double> &);

    static int get_rank();
    
    static void finalize();

    private:

    static std::map<char, int> _assign_neighbours(int);

};