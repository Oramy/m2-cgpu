#include  <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
int main(int argc, char ** argv) {
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;

        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
               // Ne detecte pas CUDA
                return -1;
            } else {
              // Afficher le nombre de device
            }
        }

        // Afficher le nom de la device
        std::cout << "Nom de la device: " << deviceProp.name << std::endl;
        // Donner le numero de version majeur et mineur
        std::cout << "Major version: " << deviceProp.major << std::endl
            << "Minor version: " << deviceProp.minor << std::endl;
        // Donner la taille de la memoire globale
        std::cout << "Total Global mem: " << deviceProp.totalGlobalMem
            << " bytes." << std::endl;
        // Donner la taille de la memoire constante
        std::cout << "Total Const memory: " << deviceProp.totalConstMem
            << " bytes." << std::endl;
        // Donner la taille de la memoire partagee par bloc
        std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock
            << " bytes." << std::endl;
        // Donner le nombre de thread max dans chacune des directions
        for (int i = 0; i < 3; i++) {
            std::cout << "Max threads per block in direction " << i << ": " << deviceProp.maxThreadsDim[i]
                << " threads." << std::endl;
        }

        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock
            << " threads." << std::endl;
        // Donner le taille maximum de la grille pour chaque direction
        for (int i = 0; i < 3; i++) {
            std::cout << "Max Grid size in direction " << i << ": " 
                << deviceProp.maxGridSize[i]
                << " blocks." << std::endl;
        }
        
        // Donner la taille du warp
        std::cout << "Warp size: " << deviceProp.warpSize << " threads." << std::endl;
    }

    return 0;
}
