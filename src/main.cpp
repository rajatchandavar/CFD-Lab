#include <iostream>
#include <string>
#include <chrono>

#include "Case.hpp"

int main(int argn, char **args) {

    if (argn > 1) {
        std::string file_name{args[1]};

        auto start = std::chrono::system_clock::now();

        Case problem(file_name, argn, args);
        problem.simulate();

        auto end = std::chrono::system_clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();
        std::cout << "Execution Time is: " << elapsed_seconds << "s\n";

    } else {
        std::cout << "Error: No input file is provided to fluidchen." << std::endl;
        std::cout << "Example usage: /path/to/fluidchen /path/to/input_data.dat" << std::endl;
    }
}
