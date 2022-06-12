#pragma once
#include<Datastructures.hpp>

class Communication{
    public:

    Communication(int, int);

    static void init_parallel(int argc, char **argv);

    static void communicate(Matrix<double> &);

    static int get_rank();
    
    static void finalize();

    static int iproc, jproc;

    private:

    static std::map<char, int> _assign_neighbours(int);

};